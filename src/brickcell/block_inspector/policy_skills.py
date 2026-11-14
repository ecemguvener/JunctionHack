"""XPolicyLab Embodiment Bridge, C4/D4 Symmetry Augmentation, and Simplified Pick-Spin-Sort Training.

Implements the streamlined end-to-end manipulation & inspection learning loop:
  Pick block -> Move to low Plate camera (`side`) -> Rotate Joint 5 counter-clockwise
  -> Sort into Good pile (`y = -0.22 m`) or Defective pile (`y = +0.22 m`) conditioned on anomaly flag.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from brickcell.block_inspector.armd_bridge import ArmdFollowerBridge
from brickcell.block_inspector.geometry import DUPLO_CATALOG, BlockSpec
from brickcell.block_inspector.twin import (
    DEFECTIVE_PILE_XY,
    GOOD_PILE_XY,
    SO101InspectionTwin,
)

SO101_XPOLICYLAB_EMBODIMENT = {
    "so101": {
        "arm_dim": [5],
        "ee_dim": [1],
    }
}


def pack_xpolicylab_observation(
    rgb_realsense_1: np.ndarray,
    rgb_realsense_2: np.ndarray,
    rgb_wrist: np.ndarray,
    q_enc_rad: np.ndarray,
    gripper_pct: float,
    task_prompt: str = "pick block present at plate camera spin joint 5 ccw and sort to good pile",
    resize_hw: tuple[int, int] = (224, 224),
) -> dict[str, Any]:
    """Pack multi-camera RGB + 6D state into XPolicyLab msgpack-numpy payload."""
    h, w = resize_hw
    state6 = np.concatenate(
        [np.asarray(q_enc_rad, dtype=np.float32), [float(np.clip(gripper_pct, 0.0, 1.0))]]
    ).astype(np.float32)
    return {
        "images": {
            "cam_high": cv2.resize(rgb_realsense_1, (w, h), interpolation=cv2.INTER_AREA),
            "cam_side": cv2.resize(rgb_realsense_2, (w, h), interpolation=cv2.INTER_AREA),
            "cam_wrist": cv2.resize(rgb_wrist, (w, h), interpolation=cv2.INTER_AREA),
        },
        "state": state6,
        "prompt": task_prompt,
    }


class RemoteXPolicyClient:
    """Msgpack-over-WebSocket client for XPolicyLab / RemoteXPolicyServer (:19000).

    Holds the last safe joint command if a remote chunk exceeds `timeout_s` (250 ms default).
    """

    def __init__(
        self,
        ws_url: str = "ws://127.0.0.1:19000",
        timeout_s: float = 0.25,
        transport_fn: Any = None,
    ) -> None:
        self.ws_url = ws_url
        self.timeout_s = timeout_s
        self.transport_fn = transport_fn
        self.last_used_fallback: bool = False

    def infer_chunk(
        self,
        obs_payload: dict[str, Any],
        fallback_q_rad: np.ndarray,
        fallback_gripper_pct: float,
        horizon: int = 16,
    ) -> np.ndarray:
        """Query remote action chunk `(H, 6)` or return safe stationary hold chunk on timeout."""
        hold_row = np.concatenate(
            [np.asarray(fallback_q_rad, dtype=np.float64), [float(fallback_gripper_pct)]]
        )
        hold_chunk = np.tile(hold_row[None, :], (horizon, 1))
        try:
            import msgpack
            import msgpack_numpy as m

            packed = msgpack.packb(obs_payload, default=m.encode, use_bin_type=True)
            if self.transport_fn is not None:
                raw = self.transport_fn(packed)
            else:
                from websockets.sync.client import connect

                with connect(self.ws_url, open_timeout=self.timeout_s, close_timeout=0.1) as ws:
                    ws.send(packed)
                    raw = ws.recv(timeout=self.timeout_s)
            reply = msgpack.unpackb(raw, object_hook=m.decode, raw=False)
            actions = np.asarray(reply["actions"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != 6:
                self.last_used_fallback = True
                return hold_chunk
            self.last_used_fallback = False
            return actions
        except Exception:
            self.last_used_fallback = True
            return hold_chunk


def enumerate_cuboid_yaw_symmetries(
    xy_yaw: tuple[float, float, float],
    symmetry_group: str = "D4",
    jitter_count: int = 0,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float, float]]:
    """Generate discrete C2/D4 cuboid yaw-orbit poses + optional tabletop SE(2) jitter."""
    bx, by, yaw0 = xy_yaw
    order = 4 if symmetry_group.upper() == "D4" else 2
    step = (2.0 * math.pi) / order
    poses: list[tuple[float, float, float]] = []
    for k in range(order):
        yk = ((yaw0 + k * step + math.pi) % (2.0 * math.pi)) - math.pi
        poses.append((float(bx), float(by), float(yk)))

    if jitter_count > 0:
        gen = rng or np.random.default_rng(42)
        base_poses = list(poses)
        for _ in range(jitter_count):
            px, py, pyaw = base_poses[int(gen.integers(0, len(base_poses)))]
            dx = float(gen.uniform(-0.020, 0.020))
            dy = float(gen.uniform(-0.025, 0.025))
            dyaw = float(gen.uniform(-0.20, 0.20))
            poses.append(
                (
                    float(np.clip(px + dx, 0.18, 0.24)),
                    float(np.clip(py + dy, -0.05, 0.05)),
                    float(((pyaw + dyaw + math.pi) % (2.0 * math.pi)) - math.pi),
                )
            )
    return poses


@dataclass
class SkillTrainingSummary:
    part_name: str
    episodes_recorded: int
    total_frames: int
    chunk_horizon: int
    train_mse: float
    closed_loop_eval_sorted_on_plate: bool
    closed_loop_eval_corner_verified: bool
    dataset_dir: str
    checkpoint_path: str


def _encode_policy_features(
    q_rad: np.ndarray,
    gripper_pct: float,
    phase_tau: float,
    anom_bit: float,
    loc_pos: np.ndarray,
    loc_yaw: float,
) -> np.ndarray:
    """Build 22D state+Fourier-phase feature vector eliminating ReLU spectral blur at grasp/drop."""
    fourier = []
    for freq in (1.0, 2.0, 4.0, 8.0, 16.0):
        ang = freq * math.pi * float(phase_tau)
        fourier.extend([math.sin(ang), math.cos(ang)])
    # Map brick yaw into symmetric C2/D4 canonical [-pi/4, +pi/4] so yaw matches select_best_symmetric_topdown_grasp
    canon_yaw = ((float(loc_yaw) + 0.25 * math.pi) % (0.5 * math.pi)) - 0.25 * math.pi
    return np.concatenate(
        [
            np.asarray(q_rad, dtype=np.float64),
            [float(gripper_pct), float(phase_tau), float(anom_bit)],
            fourier,
            np.asarray(loc_pos, dtype=np.float64),
            [math.sin(4.0 * canon_yaw), math.cos(4.0 * canon_yaw)],
        ]
    )


class ChunkedMLPPolicy:
    """Action-chunking BC policy for Pick -> Plate-Cam CCW Spin -> Conditioned Pile Sort."""

    def __init__(self, chunk_horizon: int = 8, seed: int = 7) -> None:
        self.chunk_horizon = chunk_horizon
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.net = MLPRegressor(
            hidden_layer_sizes=(256, 256),
            activation="relu",
            max_iter=650,
            tol=1e-5,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> float:
        Xs = self.x_scaler.fit_transform(X)
        Ys = self.y_scaler.fit_transform(Y)
        self.net.fit(Xs, Ys)
        pred = self.y_scaler.inverse_transform(self.net.predict(Xs))
        return float(np.mean((pred - Y) ** 2))

    def predict_chunk(self, feat: np.ndarray) -> np.ndarray:
        Xs = self.x_scaler.transform(np.asarray(feat, dtype=np.float64).reshape(1, -1))
        pred = self.y_scaler.inverse_transform(self.net.predict(Xs))[0]
        return pred.reshape(self.chunk_horizon, 6)

    def save_checkpoint(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> Path:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "chunk_horizon": np.array([self.chunk_horizon], dtype=np.int32),
            "n_layers": np.array([len(self.net.coefs_)], dtype=np.int32),
            "x_mean": self.x_scaler.mean_,
            "x_scale": self.x_scaler.scale_,
            "y_mean": self.y_scaler.mean_,
            "y_scale": self.y_scaler.scale_,
        }
        for idx, (w, b) in enumerate(zip(self.net.coefs_, self.net.intercepts_, strict=True)):
            arrays[f"W_{idx}"] = np.asarray(w, dtype=np.float64)
            arrays[f"b_{idx}"] = np.asarray(b, dtype=np.float64)
        if extra_meta:
            for k, v in extra_meta.items():
                arrays[k] = v
        np.savez_compressed(ckpt_path, **arrays)
        return ckpt_path

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> ChunkedMLPPolicy:
        with np.load(Path(path), allow_pickle=False) as npz:
            horizon = int(np.asarray(npz["chunk_horizon"]).reshape(-1)[0])
            n_layers = int(np.asarray(npz["n_layers"]).reshape(-1)[0])
            pol = cls(chunk_horizon=horizon)
            pol.x_scaler.mean_ = np.asarray(npz["x_mean"], dtype=np.float64)
            pol.x_scaler.scale_ = np.asarray(npz["x_scale"], dtype=np.float64)
            pol.x_scaler.var_ = pol.x_scaler.scale_**2
            pol.x_scaler.n_features_in_ = int(pol.x_scaler.mean_.shape[0])
            pol.y_scaler.mean_ = np.asarray(npz["y_mean"], dtype=np.float64)
            pol.y_scaler.scale_ = np.asarray(npz["y_scale"], dtype=np.float64)
            pol.y_scaler.var_ = pol.y_scaler.scale_**2
            pol.y_scaler.n_features_in_ = int(pol.y_scaler.mean_.shape[0])
            pol.net.coefs_ = [np.asarray(npz[f"W_{i}"], dtype=np.float64) for i in range(n_layers)]
            pol.net.intercepts_ = [
                np.asarray(npz[f"b_{i}"], dtype=np.float64) for i in range(n_layers)
            ]
            pol.net.n_layers_ = n_layers + 1
            pol.net.n_outputs_ = int(pol.y_scaler.mean_.shape[0])
            pol.net.out_activation_ = "identity"
            return pol


def infer_xpolicylab_request(
    req: dict[str, Any],
    policy: ChunkedMLPPolicy | None = None,
    default_horizon: int = 16,
) -> np.ndarray:
    """Infer an `(H, 6)` action chunk from an XPolicyLab msgpack-numpy observation payload."""
    state = np.asarray(req["state"], dtype=np.float64).reshape(-1)
    q_rad = state[:5]
    grip = float(state[5]) if len(state) >= 6 else 0.5
    if policy is None:
        return np.tile(np.append(q_rad, grip)[None, :], (default_horizon, 1))
    prompt = str(req.get("prompt", "")).lower()
    anom_bit = 1.0 if ("defective" in prompt or "anomaly" in prompt) else 0.0
    phase_tau = float(req.get("phase_tau", 0.35))
    loc_pos = np.asarray(req.get("loc_pos", [0.23, 0.0, 0.016]), dtype=np.float64)
    loc_yaw = float(req.get("loc_yaw", 0.0))
    feat = _encode_policy_features(q_rad, grip, phase_tau, anom_bit, loc_pos, loc_yaw)
    return policy.predict_chunk(feat)


class LocalXPolicyWebSocketServer:
    """In-process msgpack-numpy WebSocket server mirroring `RemoteXPolicyServer` (:19000)."""

    def __init__(
        self,
        policy: ChunkedMLPPolicy | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        default_horizon: int = 16,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.default_horizon = default_horizon
        self._srv: Any = None
        self._thread: Any = None
        self.requests_served: int = 0

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def handle_packed_request(self, raw: bytes) -> bytes:
        import msgpack
        import msgpack_numpy as m

        req = msgpack.unpackb(raw, object_hook=m.decode, raw=False)
        actions = infer_xpolicylab_request(
            req, policy=self.policy, default_horizon=self.default_horizon
        )
        self.requests_served += 1
        return msgpack.packb({"actions": actions}, default=m.encode, use_bin_type=True)

    def start(self) -> str:
        import threading

        from websockets.sync.server import serve

        def _handle(ws) -> None:
            for raw in ws:
                ws.send(self.handle_packed_request(raw))

        self._srv = serve(_handle, self.host, self.port)
        sock = next(iter(self._srv.socket.getsockname()[:2] for _ in (0,)))
        self.port = int(sock[1])
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self.ws_url

    def stop(self) -> None:
        if self._srv is not None:
            self._srv.shutdown()
            self._srv = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def export_lerobot_v3_metadata(
    out_root: Path,
    part_name: str,
    episode_records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write `meta/info.json`, `meta/episodes.jsonl`, and `kept_episodes_manifest.json` (`lerobot_v3`)."""
    import hashlib

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    total_frames = int(sum(int(ep["length"]) for ep in episode_records))
    info_payload = {
        "codebase_version": "v3.0",
        "robot_type": "so101",
        "fps": 30,
        "synthetic_simulation": True,
        "part_name": part_name,
        "embodiment": SO101_XPOLICYLAB_EMBODIMENT["so101"],
        "total_episodes": len(episode_records),
        "total_frames": total_frames,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
            "observation.images.cam_high": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.cam_side": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.cam_wrist": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    info_path = meta_dir / "info.json"
    info_path.write_text(json.dumps(info_payload, indent=2) + "\n", encoding="utf-8")

    episodes_jsonl = meta_dir / "episodes.jsonl"
    episodes_jsonl.write_text(
        "".join(json.dumps(ep) + "\n" for ep in episode_records),
        encoding="utf-8",
    )

    kept_entries = []
    for ep in episode_records:
        ep_json = out_root / "episodes" / f"episode_{int(ep['episode_index']):04d}.json"
        sha = hashlib.sha256(ep_json.read_bytes()).hexdigest() if ep_json.is_file() else ""
        kept_entries.append({**ep, "episode_sha256": sha})
    manifest_path = out_root / "kept_episodes_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "lerobot_v3",
                "synthetic_simulation": True,
                "kept_episodes": kept_entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return info_path, manifest_path


def collect_train_and_eval_visible_skill(
    part_name: str = "duplo_2x2_3437",
    output_dir: str | Path = "artifacts/block_inspector/skill_lab",
    num_symmetry_episodes: int = 4,
    chunk_horizon: int = 8,
    show_viewer: bool = True,
    realtime_pacing: bool = True,
    hold_viewer_seconds: float = 0.5,
) -> SkillTrainingSummary:
    """Collect pure-contact Pick -> Plate-Camera CCW Spin -> Sort demos, train BC, and eval live."""
    out_root = Path(output_dir)
    episodes_dir = out_root / "episodes"
    aux_depth_dir = out_root / "aux_depth"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    aux_depth_dir.mkdir(parents=True, exist_ok=True)

    spec: BlockSpec = DUPLO_CATALOG[part_name]
    bridge = ArmdFollowerBridge(request_fn=lambda *_: {})
    poses = enumerate_cuboid_yaw_symmetries(
        (0.205, 0.00, 0.15),
        symmetry_group=spec.symmetry_group,
        jitter_count=max(0, num_symmetry_episodes - 4),
    )[:num_symmetry_episodes]

    twin = SO101InspectionTwin(
        block_spec=spec,
        initial_block_xy_yaw=poses[0],
        show_viewer=show_viewer,
        realtime_pacing=realtime_pacing,
    )

    X_rows: list[np.ndarray] = []
    Y_rows: list[np.ndarray] = []
    total_frames = 0
    demo_lengths: list[int] = []
    lerobot_episode_records: list[dict[str, Any]] = []

    try:
        for ep_idx, (bx, by, byaw) in enumerate(poses):
            is_anom_ep = (ep_idx % 2 == 1) and (len(poses) >= 2)
            target_pile = "defective" if is_anom_ep else "good"
            twin.reset_episode_state()
            twin.reset_arm_to_home()
            R_init = np.array(
                [
                    [math.cos(byaw), -math.sin(byaw), 0.0],
                    [math.sin(byaw), math.cos(byaw), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            twin._set_settled_block_pose(
                np.array([bx, by, 0.5 * spec.dims_m[2]], dtype=np.float64), R_init
            )
            twin.status_banner = (
                f"Demo {ep_idx + 1}/{len(poses)}: Pick -> Plate-Cam CCW -> {target_pile.upper()}"
            )
            twin._skill_localize()
            loc_pos = twin.ledger.T_world_block[:3, 3].copy()
            loc_yaw = math.atan2(
                float(twin.ledger.T_world_block[1, 0]),
                float(twin.ledger.T_world_block[0, 0]),
            )
            if is_anom_ep:
                twin.receive_anomaly_message("synthetic_training_anomaly_flag")

            ep_q_raw: list[np.ndarray] = []
            ep_grip_raw: list[float] = []
            ep_actions: list[np.ndarray] = []
            ep_lerobot_actions: list[list[float]] = []

            def _record_hook(tw: SO101InspectionTwin) -> None:
                ep_q_raw.append(tw.current_arm_q())
                ep_grip_raw.append(float(tw.gripper_pct))
                ep_actions.append(np.concatenate([tw.q_cmd, [tw.gripper_pct]]))
                ep_lerobot_actions.append(
                    np.round(bridge.q_to_lerobot_action6(tw.q_cmd, tw.gripper_pct), 3).tolist()
                )

            twin.frame_listeners.append(_record_hook)
            try:
                ok_pick = twin._skill_pick_block()
                ok_spin = twin._skill_present_and_spin_at_plate_camera()
                ok_sort = twin._skill_sort_block_to_pile(target_pile)
                if not (ok_pick.succeeded and ok_spin.succeeded and ok_sort.succeeded):
                    raise RuntimeError(
                        f"Teacher episode {ep_idx} failed: "
                        f"{ok_pick.reason} / {ok_spin.reason} / {ok_sort.reason}"
                    )
            finally:
                twin.frame_listeners.remove(_record_hook)

            rgb_top, d_top = twin.render_rgbd("realsense_1")
            rgb_plate, d_plate = twin.render_rgbd("realsense_2")
            rgb_wrist, _ = twin.render_rgbd("wrist")
            assert d_top is not None and d_plate is not None
            np.savez_compressed(
                aux_depth_dir / f"episode_{ep_idx:04d}.npz",
                overhead_depth_m=d_top.astype(np.float32),
                plate_depth_m=d_plate.astype(np.float32),
                overhead_keyframe_rgb=rgb_top,
                plate_keyframe_rgb=rgb_plate,
                wrist_keyframe_rgb=rgb_wrist,
            )
            (episodes_dir / f"episode_{ep_idx:04d}.json").write_text(
                json.dumps(
                    {
                        "episode_index": ep_idx,
                        "synthetic_teacher": True,
                        "part": spec.name,
                        "block_xy_yaw": [bx, by, byaw],
                        "anomaly_flagged": is_anom_ep,
                        "sorted_pile": target_pile,
                        "num_steps": len(ep_actions),
                        "lerobot_actions_deg_pct": ep_lerobot_actions,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            n_ep = len(ep_actions)
            demo_lengths.append(n_ep)
            total_frames += n_ep
            lerobot_episode_records.append(
                {
                    "episode_index": ep_idx,
                    "tasks": [
                        f"pick {spec.name} spin joint 5 ccw at plate camera and sort to {target_pile} plate"
                    ],
                    "length": n_ep,
                    "anomaly_flagged": is_anom_ep,
                    "sorted_pile": target_pile,
                }
            )
            A = np.asarray(ep_actions, dtype=np.float64)
            anom_bit = 1.0 if is_anom_ep else 0.0
            for t in range(n_ep):
                phase_tau = t / max(1, n_ep - 1)
                feat = _encode_policy_features(
                    ep_q_raw[t], ep_grip_raw[t], phase_tau, anom_bit, loc_pos, loc_yaw
                )
                idxs = np.minimum(np.arange(t, t + chunk_horizon), n_ep - 1)
                X_rows.append(feat)
                Y_rows.append(A[idxs].reshape(-1))

        policy = ChunkedMLPPolicy(chunk_horizon=chunk_horizon)
        train_mse = policy.fit(np.asarray(X_rows), np.asarray(Y_rows))

        # Closed-loop rollout evaluation of trained Pick -> Plate-Cam CCW Spin -> Good Sort policy!
        twin.reset_episode_state()
        twin.reset_arm_to_home()
        bx, by, byaw = poses[0]
        R_eval = np.array(
            [
                [math.cos(byaw), -math.sin(byaw), 0.0],
                [math.sin(byaw), math.cos(byaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        twin._set_settled_block_pose(
            np.array([bx, by, 0.5 * spec.dims_m[2]], dtype=np.float64), R_eval
        )
        twin.status_banner = "Closed-Loop Policy Eval: Pick -> Plate-Cam CCW -> GOOD Pile"
        twin._skill_localize()
        eval_pos = twin.ledger.T_world_block[:3, 3].copy()
        eval_yaw = math.atan2(
            float(twin.ledger.T_world_block[1, 0]),
            float(twin.ledger.T_world_block[0, 0]),
        )

        # Match clean (good-pile) episode horizon length
        clean_lengths = [demo_lengths[i] for i in range(len(demo_lengths)) if i % 2 == 0]
        horizon_steps = int(round(float(np.median(clean_lengths or demo_lengths))))
        exec_stride = max(2, chunk_horizon // 2)
        t_cur = 0
        max_j5_seen = -10.0
        min_j5_seen = +10.0
        while t_cur < horizon_steps:
            phase_tau = min(1.0, t_cur / max(1, horizon_steps - 1))
            feat = _encode_policy_features(
                twin.current_arm_q(), twin.gripper_pct, phase_tau, 0.0, eval_pos, eval_yaw
            )
            chunk = policy.predict_chunk(feat)
            n_run = min(exec_stride, horizon_steps - t_cur)
            for row in chunk[:n_run]:
                q_next = np.clip(
                    row[:5], twin.ik.cal_kin.calibration.lower, twin.ik.cal_kin.calibration.upper
                )
                twin.step_control_30hz(q_next, gripper_pct=float(np.clip(row[5], 0.0, 1.0)))
                j5 = float(twin.current_arm_q()[4])
                if 0.25 <= phase_tau <= 0.80:
                    max_j5_seen = max(max_j5_seen, j5)
                    min_j5_seen = min(min_j5_seen, j5)
                t_cur += 1

        for _ in range(12):
            twin.step_control_30hz(twin.q_cmd, gripper_pct=0.85)

        p_end = twin.ground_truth_block_pose()[:3, 3]
        xy_err = float(np.linalg.norm(p_end[:2] - GOOD_PILE_XY))
        sorted_ok = (
            xy_err <= 0.085 and float(p_end[2]) < 0.065 and (max_j5_seen - min_j5_seen > 2.5)
        )
        if sorted_ok:
            twin.sorted_pile = "good"
            twin.status_banner = "Policy Eval Succeeded: Sorted to GOOD Pile — Viewer Live"
        twin._sync_viewer()
        if show_viewer and twin.is_viewer_running():
            deadline = time.monotonic() + float(hold_viewer_seconds)
            while twin.is_viewer_running() and time.monotonic() < deadline:
                twin._sync_viewer()
                time.sleep(0.05)

        export_lerobot_v3_metadata(out_root, spec.name, lerobot_episode_records)
        ckpt_path = policy.save_checkpoint(
            out_root / f"chunk_bc_{spec.name}.npz",
            extra_meta={
                "train_mse": np.array([train_mse], dtype=np.float64),
                "good_pile_xy": GOOD_PILE_XY,
                "defective_pile_xy": DEFECTIVE_PILE_XY,
            },
        )
        return SkillTrainingSummary(
            part_name=spec.name,
            episodes_recorded=len(poses),
            total_frames=total_frames,
            chunk_horizon=chunk_horizon,
            train_mse=train_mse,
            closed_loop_eval_sorted_on_plate=sorted_ok,
            closed_loop_eval_corner_verified=sorted_ok,
            dataset_dir=str(episodes_dir),
            checkpoint_path=str(ckpt_path),
        )
    finally:
        twin.close()
