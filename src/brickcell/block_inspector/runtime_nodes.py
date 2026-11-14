"""Multi-node BabyROS / Zenoh SHM runtime nodes for simulation and station execution.

Implements the full decoupled runtime graph over `InProcessRuntimeBus` or `BabyRosRuntimeBus`:
  CameraPubNode -> (realsense_1/2/wrist NDDC frames)
    -> PerceptionAndAnomalyNode (Dual-D415 6D pose + OpenVINO CPU Plate-camera anomaly scoring)
      -> `so101/inspection/anomaly` (raw distance >= calibrated threshold; detector never commands arm)
        -> FraxMujocoTwinNode (`so101/skill/execute` & `so101/arm/command`)
          <-> CogManInspectorNode (`RemoteBusSkillExecutor` + `:8095` HUD)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from brickcell.block_inspector.camera_adapters import CamUiReporter, LiveInspectionHudServer
from brickcell.block_inspector.cogman import (
    CogManInspectionAgent,
    GroundedOperator,
    InspectionMode,
    InspectionRunReport,
    SkillOutcome,
    SymbolicState,
)
from brickcell.block_inspector.geometry import (
    DUPLO_CATALOG,
    CameraExtrinsicsRig,
    station_camera_rig,
)
from brickcell.block_inspector.ledger import EightCornerCoverageLedger
from brickcell.block_inspector.perception import localize_clustered_blocks_in_source_plate
from brickcell.block_inspector.runtime_bus import CANONICAL_TOPICS, pack_rgbd_frame_nddc
from brickcell.block_inspector.twin import SOURCE_PILE_XY, SO101InspectionTwin
from brickcell.detector.runtime import OpenVINODetector


class RgbdProvider(Protocol):
    def render_rgbd(self, camera_name: str) -> tuple[np.ndarray, np.ndarray | None]: ...


class CameraPubNode:
    """Publishes Dual-RealSense D415 + Wrist RGB-D NDDC payloads onto canonical BabyROS topics."""

    def __init__(
        self,
        bus: Any,
        source: Any,
        rig: CameraExtrinsicsRig | None = None,
    ) -> None:
        self.bus = bus
        self.source = source
        self.rig = rig or station_camera_rig()
        self.frames_published: int = 0

    def _fetch_cam(self, cam_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        if hasattr(self.source, "render_rgbd"):
            return self.source.render_rgbd(cam_name)
        return self.source.get_rgbd(cam_name)

    def publish_step(
        self, cameras: tuple[str, ...] = ("realsense_1", "realsense_2", "wrist")
    ) -> None:
        ts_ns = time.time_ns()
        topic_map = {
            "realsense_1": "realsense_1_frame",
            "realsense_2": "realsense_2_frame",
            "wrist": "wrist_frame",
        }
        for cam_name in cameras:
            rgb, depth_m = self._fetch_cam(cam_name)
            depth_mm = (
                np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
                if depth_m is not None
                else None
            )
            cam_model = self.rig.by_name(cam_name)
            self.bus.publish(
                CANONICAL_TOPICS[topic_map[cam_name]],
                pack_rgbd_frame_nddc(rgb, ts_ns, depth_mm=depth_mm, intrinsics_k=cam_model.K),
            )
        self.frames_published += 1


class PerceptionAndAnomalyNode:
    """Subscribes to camera + arm state topics, publishes 6D block pose & OpenVINO anomaly alerts.

    Strictly obeys `AGENTS.md`: never issues robot commands or holds a `RobotInterface` handle.
    """

    def __init__(
        self,
        bus: Any,
        detector: OpenVINODetector,
        rig: CameraExtrinsicsRig | None = None,
    ) -> None:
        if detector.threshold is None:
            raise ValueError(
                "PerceptionAndAnomalyNode requires a calibrated OpenVINODetector threshold"
            )
        self.bus = bus
        self.detector = detector
        self.rig = rig or station_camera_rig()
        self._latest_r1: dict[str, np.ndarray] | None = None
        self._latest_r2: dict[str, np.ndarray] | None = None
        self._plate_scan_active: bool = False
        self._held_pos_world: np.ndarray | None = None
        self.anomaly_alerts_published: int = 0
        self.last_raw_distance: float | None = None

        self._unsubs = [
            self.bus.subscribe(CANONICAL_TOPICS["arm_state"], self._on_arm_state),
            self.bus.subscribe(CANONICAL_TOPICS["realsense_1_frame"], self._on_r1),
            self.bus.subscribe(CANONICAL_TOPICS["realsense_2_frame"], self._on_r2),
        ]

    def _on_arm_state(self, msg: dict[str, Any]) -> None:
        scan_arr = msg.get("plate_scan_active")
        self._plate_scan_active = (
            bool(int(np.asarray(scan_arr).reshape(-1)[0])) if scan_arr is not None else False
        )
        hp = msg.get("held_pos_world")
        self._held_pos_world = np.asarray(hp, dtype=np.float64) if hp is not None else None

    def _on_r1(self, msg: dict[str, np.ndarray]) -> None:
        self._latest_r1 = msg

    def _on_r2(self, msg: dict[str, np.ndarray]) -> None:
        self._latest_r2 = msg
        if self._plate_scan_active and self._held_pos_world is not None:
            self._evaluate_plate_frame(msg, self._held_pos_world)

    def _evaluate_plate_frame(self, msg: dict[str, np.ndarray], held_pos_world: np.ndarray) -> None:
        rgb = np.asarray(msg["rgb"], dtype=np.uint8)
        depth_mm = msg.get("depth_mm")
        plate_cam = self.rig.realsense_2
        uv_c, z_c = plate_cam.project_world_points(held_pos_world[None, :])
        u_px, v_px, z_exp = float(uv_c[0, 0]), float(uv_c[0, 1]), float(z_c[0])
        if not (
            24.0 <= u_px <= (plate_cam.width - 24.0)
            and 24.0 <= v_px <= (plate_cam.height - 24.0)
            and 0.12 <= z_exp <= 0.60
        ):
            return
        u0 = max(0, int(round(u_px)) - 48)
        u1 = min(plate_cam.width, int(round(u_px)) + 48)
        v0 = max(0, int(round(v_px)) - 48)
        v1 = min(plate_cam.height, int(round(v_px)) + 48)
        roi_rgb = rgb[v0:v1, u0:u1]
        roi_scored = roi_rgb
        if depth_mm is not None:
            roi_dep_m = np.asarray(depth_mm[v0:v1, u0:u1], dtype=np.float64) / 1000.0
            if not bool(np.any((roi_dep_m > 0.10) & (roi_dep_m < 0.55))):
                return
            fg_m = (roi_dep_m > max(0.08, z_exp - 0.065)) & (roi_dep_m < (z_exp + 0.065))
            bg_iso = np.array([28, 32, 38], dtype=np.uint8)
            roi_scored = np.where(fg_m[:, :, None], roi_rgb, bg_iso)
        if roi_rgb.size == 0 or float(roi_rgb.std()) <= 4.0:
            return

        pred = self.detector.predict(roi_scored)
        self.last_raw_distance = float(pred.score)
        assert self.detector.threshold is not None
        thr_val = float(self.detector.threshold.value)
        if pred.score >= thr_val:
            self.anomaly_alerts_published += 1
            self.bus.publish(
                CANONICAL_TOPICS["inspection_anomaly"],
                {
                    "raw_distance": np.array([pred.score], dtype=np.float64),
                    "threshold": np.array([thr_val], dtype=np.float64),
                    "model_version": self.detector.model_version,
                    "score_type": "raw_patchcore_distance_not_probability",
                },
            )

    def localize_cluster_once(self) -> dict[str, Any]:
        """Fuse latest Dual-D415 NDDC frames and publish `inspection/perception/block_state`."""
        if self._latest_r1 is None or self._latest_r2 is None:
            return {"num_candidates": 0}
        d1 = np.asarray(self._latest_r1["depth_mm"], dtype=np.float32) / 1000.0
        rgb1 = np.asarray(self._latest_r1["rgb"], dtype=np.uint8)
        d2 = np.asarray(self._latest_r2["depth_mm"], dtype=np.float32) / 1000.0
        rgb2 = np.asarray(self._latest_r2["rgb"], dtype=np.uint8)
        cands, _ = localize_clustered_blocks_in_source_plate(
            [(d1, rgb1, self.rig.realsense_1), (d2, rgb2, self.rig.realsense_2)],
            source_plate_xy=SOURCE_PILE_XY,
        )
        payload: dict[str, Any] = {"num_candidates": len(cands)}
        if cands:
            payload["T_world_block"] = cands[0].localized.T_world_block
            payload["part_name"] = cands[0].block_spec.name
        self.bus.publish(CANONICAL_TOPICS["block_state"], payload, transient_local=True)
        return payload

    def close(self) -> None:
        for u in self._unsubs:
            u()
        self._unsubs.clear()


class FraxMujocoTwinNode:
    """Serves `so101/skill/execute` & `so101/arm/command` and listens for `so101/inspection/anomaly`."""

    def __init__(
        self,
        bus: Any,
        twin: SO101InspectionTwin,
        camera_pub: CameraPubNode | None = None,
    ) -> None:
        self.bus = bus
        self.twin = twin
        self.camera_pub = camera_pub
        self._step_counter: int = 0

        self._unsub_anom = self.bus.subscribe(
            CANONICAL_TOPICS["inspection_anomaly"], self._on_anomaly_msg
        )
        self.bus.register_server(CANONICAL_TOPICS["skill_execute"], self._on_skill_rpc)
        self.bus.register_server(CANONICAL_TOPICS["arm_command"], self._on_arm_cmd)
        self.twin.frame_listeners.append(self._on_twin_step)

    def _on_anomaly_msg(self, msg: dict[str, Any]) -> None:
        dist_arr = msg.get("raw_distance")
        thr_arr = msg.get("threshold")
        if dist_arr is not None and thr_arr is not None:
            d_val = float(np.asarray(dist_arr).reshape(-1)[0])
            t_val = float(np.asarray(thr_arr).reshape(-1)[0])
            self.twin.receive_anomaly_message(f"bus_openvino_distance={d_val:.4f}>={t_val:.4f}")
        else:
            self.twin.receive_anomaly_message("bus_anomaly_signal")

    def _on_twin_step(self, tw: SO101InspectionTwin) -> None:
        self._step_counter += 1
        T_held = tw.body3_world_pose() @ tw._T_body3_block_held
        if tw.plate_scan_active and (self._step_counter % 4 == 0):
            self.bus.publish(
                CANONICAL_TOPICS["arm_state"],
                {
                    "q_enc_rad": tw.current_arm_q(),
                    "gripper_pct": np.array([tw.gripper_pct], dtype=np.float64),
                    "plate_scan_active": np.array([1], dtype=np.int32),
                    "held_pos_world": T_held[:3, 3].copy(),
                },
            )
            if self.camera_pub is not None:
                self.camera_pub.publish_step(cameras=("realsense_2",))
        elif self._step_counter % 15 == 0:
            self.bus.publish(
                CANONICAL_TOPICS["arm_state"],
                {
                    "q_enc_rad": tw.current_arm_q(),
                    "gripper_pct": np.array([tw.gripper_pct], dtype=np.float64),
                    "plate_scan_active": np.array([0], dtype=np.int32),
                },
            )

    def _on_arm_cmd(self, req: dict[str, Any]) -> dict[str, Any]:
        if req.get("stop"):
            self.twin.step_control_30hz(
                self.twin.current_arm_q(), gripper_pct=self.twin.gripper_pct
            )
            return {"ok": True, "stopped": True}
        q = np.asarray(req["q_enc_rad"], dtype=np.float64)
        grip = float(req.get("gripper_pct", self.twin.gripper_pct))
        self.twin.step_control_30hz(q, gripper_pct=grip)
        return {"ok": True, "q_enc_rad": self.twin.current_arm_q().tolist()}

    def _on_skill_rpc(self, req: dict[str, Any]) -> dict[str, Any]:
        action = req.get("rpc", "execute")
        block_id = str(req.get("block_id", "block_0"))
        if action == "perceive_state":
            st = self.twin.perceive_symbolic_state(block_id)
            return {
                "block_id": st.block_id,
                "hand_empty": st.hand_empty,
                "block_localized": st.block_localized,
                "block_on_table": st.block_on_table,
                "block_held": st.block_held,
                "block_at_rest": st.block_at_rest,
                "corner_upper": list(st.corner_upper),
                "corner_wrist_reachable": list(st.corner_wrist_reachable),
                "corner_inspected": list(st.corner_inspected),
                "plate_scan_completed": st.plate_scan_completed,
                "anomaly_flagged": st.anomaly_flagged,
                "sorted_pile": st.sorted_pile,
                "source_plate_cleared": st.source_plate_cleared,
                "remaining_source_blocks": st.remaining_source_blocks,
                "sorted_good_count": st.sorted_good_count,
                "sorted_defective_count": st.sorted_defective_count,
            }
        if action == "arm_q":
            return {"q_enc_rad": self.twin.current_arm_q().tolist()}
        if action == "corner_seed_q":
            q_seed = self.twin.corner_seed_q(int(req["corner_index"]))
            return {"q_enc_rad": q_seed.tolist() if q_seed is not None else None}
        if action == "summary":
            return {
                "sorted_counts": dict(self.twin.sorted_counts),
                "batch_sorted_records": [
                    {k: v for k, v in rec.items() if k != "keyframe_rgb"}
                    for rec in self.twin.batch_sorted_records
                ],
                "ledger": self.twin.ledger.summary_dict(),
            }

        op = GroundedOperator(
            name=str(req["operator"]),
            block_id=block_id,
            params=dict(req.get("params") or {}),
        )
        perturb_raw = req.get("cone_perturbation")
        perturb = np.asarray(perturb_raw, dtype=np.float64) if perturb_raw is not None else None
        out = self.twin.execute_operator(op, cone_perturbation=perturb)
        return {
            "succeeded": out.succeeded,
            "reason": out.reason,
            "metadata": out.metadata,
        }

    def close(self) -> None:
        if self._on_twin_step in self.twin.frame_listeners:
            self.twin.frame_listeners.remove(self._on_twin_step)
        self._unsub_anom()


class RemoteBusSkillExecutor:
    """`SkillExecutor` driving `FraxMujocoTwinNode` strictly over BabyROS `so101/skill/execute` RPC."""

    def __init__(self, bus: Any) -> None:
        self.bus = bus
        self.sorted_counts: dict[str, int] = {}
        self.batch_sorted_records: list[dict] = []

    def perceive_symbolic_state(self, block_id: str) -> SymbolicState:
        r = self.bus.call_service(
            CANONICAL_TOPICS["skill_execute"],
            {"rpc": "perceive_state", "block_id": block_id},
        )
        return SymbolicState(
            block_id=r["block_id"],
            hand_empty=bool(r["hand_empty"]),
            block_localized=bool(r["block_localized"]),
            block_on_table=bool(r["block_on_table"]),
            block_held=bool(r["block_held"]),
            block_at_rest=bool(r["block_at_rest"]),
            corner_upper=tuple(bool(x) for x in r["corner_upper"]),
            corner_wrist_reachable=tuple(bool(x) for x in r["corner_wrist_reachable"]),
            corner_inspected=tuple(bool(x) for x in r["corner_inspected"]),
            plate_scan_completed=bool(r["plate_scan_completed"]),
            anomaly_flagged=bool(r["anomaly_flagged"]),
            sorted_pile=r["sorted_pile"],
            source_plate_cleared=bool(r["source_plate_cleared"]),
            remaining_source_blocks=int(r["remaining_source_blocks"]),
            sorted_good_count=int(r["sorted_good_count"]),
            sorted_defective_count=int(r["sorted_defective_count"]),
        )

    def current_arm_q(self) -> np.ndarray:
        r = self.bus.call_service(CANONICAL_TOPICS["skill_execute"], {"rpc": "arm_q"})
        return np.asarray(r["q_enc_rad"], dtype=np.float64)

    def corner_seed_q(self, corner_index: int) -> np.ndarray | None:
        r = self.bus.call_service(
            CANONICAL_TOPICS["skill_execute"],
            {"rpc": "corner_seed_q", "corner_index": int(corner_index)},
        )
        return (
            np.asarray(r["q_enc_rad"], dtype=np.float64) if r.get("q_enc_rad") is not None else None
        )

    def execute_operator(
        self, op: GroundedOperator, cone_perturbation: np.ndarray | None = None
    ) -> SkillOutcome:
        req: dict[str, Any] = {
            "rpc": "execute",
            "operator": op.name,
            "block_id": op.block_id,
            "params": op.params,
        }
        if cone_perturbation is not None:
            req["cone_perturbation"] = cone_perturbation.tolist()
        r = self.bus.call_service(CANONICAL_TOPICS["skill_execute"], req)
        summary = self.bus.call_service(CANONICAL_TOPICS["skill_execute"], {"rpc": "summary"})
        self.sorted_counts = dict(summary.get("sorted_counts", {}))
        self.batch_sorted_records = list(summary.get("batch_sorted_records", []))
        return SkillOutcome(
            succeeded=bool(r["succeeded"]),
            reason=str(r["reason"]),
            metadata=dict(r.get("metadata") or {}),
        )


class CogManInspectorNode:
    """CogMan executive driving `RemoteBusSkillExecutor` over BabyROS + optional `:8095` HUD."""

    def __init__(
        self,
        bus: Any,
        hud_port: int | None = None,
        cam_ui_url: str | None = None,
    ) -> None:
        self.bus = bus
        self.remote_executor = RemoteBusSkillExecutor(bus)
        self.ledger = EightCornerCoverageLedger(
            object_id="block_0",
            block_spec=DUPLO_CATALOG["duplo_2x2_3437"],
            rig=station_camera_rig(),
        )
        self.agent = CogManInspectionAgent(executor=self.remote_executor, ledger=self.ledger)
        self.hud = (
            LiveInspectionHudServer(host="127.0.0.1", port=hud_port)
            if hud_port is not None
            else None
        )
        self.reporter = CamUiReporter(base_url=cam_ui_url) if cam_ui_url else None
        if self.hud is not None:
            self.hud.start()

    def run_episode(
        self,
        instruction: str = "pick and classify all 8 blocks from source plate",
        block_id: str = "block_0",
        mode: InspectionMode | None = None,
        output_dir: str | Path | None = None,
    ) -> InspectionRunReport:
        rep = self.agent.run(
            instruction=instruction,
            block_id=block_id,
            output_dir=output_dir,
            mode=mode,
        )
        self.bus.publish(
            CANONICAL_TOPICS["symbolic_state"],
            {
                "succeeded": rep.succeeded,
                "mode": rep.mode,
                "sorted_pile": rep.sorted_pile,
                "sorted_counts": rep.sorted_counts,
            },
            transient_local=True,
        )
        if self.reporter is not None:
            self.reporter.post_summary(
                {
                    "succeeded": rep.succeeded,
                    "mode": rep.mode,
                    "sorted_counts": rep.sorted_counts,
                    "synthetic_simulation": True,
                }
            )
        return rep

    def close(self) -> None:
        if self.hud is not None:
            self.hud.stop()
