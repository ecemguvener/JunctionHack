"""Physical SO-101 3-Plate Inspection Cell Runtime (Real2Sim + Frax IK + Live Cameras + armd).

Replicates the full verified simulation architecture on the physical Intel bench:
  - `FrameSource` (`LiveBenchHttpFrameSource`): consumes live loopback streams from
    `live_depth.py` (`http://127.0.0.1:8094` Dual-D415 RGB-D + point clouds) and
    `focus_assist.py` (`http://127.0.0.1:8091/frame.jpg` Sonix UVC wrist RGB) without
    fighting V4L2 device locks.
  - `Inspector` (`RealCameraAnomalyInspector`): strictly isolated from robot control
    (zero `RobotInterface` handle per `AGENTS.md`). Evaluates symmetric-flanked CIE-Lab
    dark-mark surface scores + optional `OpenVINODetector` raw PatchCore distances inside
    a bounded-timeout worker (timed-out workers are abandoned and never reused).
  - `RobotInterface` (`PhysicalArmController`): wraps `ArmdFollowerBridge` (`127.0.0.1:8765`)
    with bounded speeds, `SAFE_GRIPPER_MIN_RAW = 1339`, adaptive pitched-wrist Frax IK,
    bench obstacle clearance checks, and immediate `hold` stop on any servo fault.
  - `SkillExecutor` (`PhysicalSO101CellExecutor`): driven by the exact same
    `CogManInspectionAgent` (`mode="clear_source_plate"`), mirroring live state into
    `SO101InspectionTwin` and streaming live decision overlays to `:8096`
    (`brickcell-dashboard`), `:8095` (`LiveInspectionHudServer`), and `:8093` (`camui`).
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from brickcell.block_inspector.armd_bridge import ArmdFollowerBridge, ArmFaultError, DecodedArmState
from brickcell.block_inspector.camera_adapters import (
    CamUiReporter,
    LiveInspectionHudServer,
    TwinDashboardReporter,
)
from brickcell.block_inspector.cogman import (
    CogManInspectionAgent,
    GroundedOperator,
    SkillOutcome,
    SymbolicState,
)
from brickcell.block_inspector.frax_ik import GraspIKSolution, SO101FraxIK
from brickcell.block_inspector.geometry import (
    DUPLO_CATALOG,
    REPO_ROOT,
    T_WORLD_BASE,
    station_camera_rig,
)
from brickcell.block_inspector.ledger import EightCornerCoverageLedger
from brickcell.block_inspector.real2sim import (
    PerceivedBrick,
    Real2SimPipeline,
    Real2SimSceneState,
    project_world_obb_to_image_box,
    real_intel_camera_poses_world,
)
from brickcell.block_inspector.twin import (
    PLATE_DROP_SLOT_OFFSETS_XY,
    SO101InspectionTwin,
    SpawnedBlockInfo,
)

# High upright observation posture clear of Top and Plate camera sightlines & strictly in CCW Joint-5 zone
PHYSICAL_OBSERVATION_HOME_Q = np.array([0.00, -1.15, 0.88, 0.42, 0.55], dtype=np.float64)
PHYSICAL_HOME_GRIPPER_PCT = 0.48

# Physical Plate-camera presentation target (+X forward, -Y right, +Z up) placing held brick
# centered in the real Plate D415 (`243622060450`) frame at z_cam ~ 0.28 m.
# Joint 5 (`wrist_roll`) hits its physical CW stop at ~2282 ticks (+0.36 rad) and rotates
# freely in the Counter-Clockwise (+tick / +rad) direction [2341 .. 4042 ticks] = [+0.45 .. +3.06 rad].
PHYSICAL_PLATE_SCAN_POS = np.array([0.312, -0.025, 0.118], dtype=np.float64)
PHYSICAL_JOINT5_CCW_START_RAD = +0.45
PHYSICAL_JOINT5_CCW_END_RAD = +3.05

# Measured static bench obstacles in calibrated +X-forward robot base frame (m):
# (x, y, z, radius_m, label)
BENCH_OBSTACLE_SPHERES: tuple[tuple[float, float, float, float, str], ...] = (
    (0.060, -0.168, 0.095, 0.048, "bench_can_right"),
    (0.055, +0.255, 0.055, 0.070, "laptop_left_rear"),
    (0.340, +0.268, 0.175, 0.080, "laptop_screen_left"),
    (0.515, -0.240, 0.140, 0.075, "plate_camera_pedestal"),
)

PART_ID_TO_DASHBOARD_ID = {
    "3437": "part-001",
    "3011": "part-002",
    "76371": "part-003",
    "2302": "part-004",
}


@dataclass(frozen=True)
class LiveCameraSnapshot:
    """Validated RGB + optional aligned metric depth snapshot from a live HTTP camera stream."""

    camera_name: str
    rgb: np.ndarray
    depth_m: np.ndarray | None
    timestamp_s: float
    age_s: float
    frame_id: str
    intrinsics: dict[str, Any] = field(default_factory=dict)


class StaleCameraFrameError(RuntimeError):
    """Raised when a live camera stream is offline, stale, or fails integrity checks."""


class LiveBenchHttpFrameSource:
    """`FrameSource` boundary consuming `live_depth.py` (`:8094`) and wrist UVC (`:8091`)."""

    def __init__(
        self,
        live_depth_url: str = "http://127.0.0.1:8094",
        wrist_http_url: str = "http://127.0.0.1:8091/frame.jpg",
        timeout_s: float = 2.0,
        max_frame_age_s: float = 2.5,
    ) -> None:
        self.live_depth_url = live_depth_url.rstrip("/")
        self.wrist_http_url = wrist_http_url
        self.timeout_s = float(timeout_s)
        self.max_frame_age_s = float(max_frame_age_s)

    def fetch_d415_snapshot(self, view: str = "plate") -> LiveCameraSnapshot:
        """Fetch aligned 640x480 RGB + lossless uint16 depth from `/snapshot/{top,plate}`."""
        if view not in ("top", "plate"):
            raise ValueError(f"Unsupported D415 view: {view}")
        url = f"{self.live_depth_url}/snapshot/{view}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
                if resp.status != 200:
                    raise StaleCameraFrameError(f"HTTP {resp.status} from {url}")
                doc = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise StaleCameraFrameError(f"Failed to read D415 {view} snapshot: {exc}") from exc

        if doc.get("error"):
            raise StaleCameraFrameError(f"D415 {view} error: {doc['error']}")
        age_s = float(doc.get("age_s", max(0.0, time.time() - float(doc.get("time", 0.0)))))
        if age_s > self.max_frame_age_s:
            raise StaleCameraFrameError(
                f"Stale D415 {view} snapshot: age_s={age_s:.2f} > {self.max_frame_age_s:.2f}"
            )

        rgb_bytes = base64.b64decode(doc["rgb_jpeg_b64"], validate=True)
        depth_bytes = base64.b64decode(doc["depth_png_b64"], validate=True)
        rgb = np.asarray(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"), dtype=np.uint8)
        depth_u16 = np.asarray(Image.open(io.BytesIO(depth_bytes)))
        if depth_u16.ndim != 2 or depth_u16.dtype != np.uint16:
            raise StaleCameraFrameError(f"Invalid uint16 depth PNG in {view} snapshot")
        scale_m = float(doc.get("depth_scale_m", 0.001))
        depth_m = depth_u16.astype(np.float32) * scale_m
        return LiveCameraSnapshot(
            camera_name=view,
            rgb=rgb,
            depth_m=depth_m,
            timestamp_s=float(doc.get("time", time.time())),
            age_s=age_s,
            frame_id=str(doc.get("frame_id", f"{view}-{time.time_ns()}")),
            intrinsics=dict(doc.get("intrinsics") or {}),
        )

    def fetch_wrist_snapshot(
        self, resize_hw: tuple[int, int] | None = (360, 640)
    ) -> LiveCameraSnapshot:
        """Fetch live Sonix UVC wrist RGB frame from `http://127.0.0.1:8091/frame.jpg`."""
        t_now = time.time()
        try:
            with urllib.request.urlopen(self.wrist_http_url, timeout=self.timeout_s) as resp:
                if resp.status != 200:
                    raise StaleCameraFrameError(f"HTTP {resp.status} from {self.wrist_http_url}")
                raw = resp.read()
        except Exception as exc:
            raise StaleCameraFrameError(f"Failed to read wrist camera: {exc}") from exc

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if resize_hw is not None:
            img = img.resize((resize_hw[1], resize_hw[0]), Image.Resampling.BILINEAR)
        rgb = np.asarray(img, dtype=np.uint8)
        return LiveCameraSnapshot(
            camera_name="wrist",
            rgb=rgb,
            depth_m=None,
            timestamp_s=t_now,
            age_s=0.0,
            frame_id=f"wrist-{int(t_now * 1000)}",
        )

    def get_rgbd(self, camera_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        """Adapter method compatible with `CameraPubNode` (`realsense_1`, `realsense_2`, `wrist`)."""
        if camera_name in ("realsense_1", "overhead", "top"):
            snap = self.fetch_d415_snapshot("top")
            return snap.rgb, snap.depth_m
        if camera_name in ("realsense_2", "side", "plate"):
            snap = self.fetch_d415_snapshot("plate")
            return snap.rgb, snap.depth_m
        if camera_name == "wrist":
            snap = self.fetch_wrist_snapshot()
            return snap.rgb, None
        raise ValueError(f"Unknown camera_name: {camera_name}")

    def perceive_scene(
        self,
        real2sim: Real2SimPipeline,
        q_arm_rad: np.ndarray | None = None,
    ) -> Real2SimSceneState:
        """Perceive fused Dual-D415 (`fused` top + plate) 3D scene via `live_depth.py`."""
        scene = real2sim.perceive_from_live_depth_http(
            base_url=self.live_depth_url,
            q_arm_rad=q_arm_rad,
            timeout_s=self.timeout_s,
            prefer_snapshots=False,
            camera_mode="fused",
        )
        if not scene.bricks:
            scene = real2sim.perceive_from_live_depth_http(
                base_url=self.live_depth_url,
                q_arm_rad=q_arm_rad,
                timeout_s=self.timeout_s,
                prefer_snapshots=True,
                camera_mode="fused",
            )
        return scene


def score_brick_roi_dark_mark(
    roi_rgb: np.ndarray,
    fg_mask: np.ndarray | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Evaluate symmetric-flanked CIE-Lab blackhat raw surface anomaly score on a brick ROI.

    Detects thin dark pen/marker strokes or surface gouges flanked on both transverse sides
    by uniform brick plastic (same lightness and Lab chromaticity), rejecting outer silhouette
    edges, plate rims, circular stud shadows, and underside hollow tubes.
    Never returns or represents a probability: `raw_score` is a Lab contrast distance (>= 0).
    """
    if roi_rgb.size == 0 or min(roi_rgb.shape[:2]) < 20:
        return 0.0, []

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    # Exclude aquamarine/blue KALAS plate plastic and purple PLA robot gripper jaws
    not_plate = ~((hsv[:, :, 0] >= 92) & (hsv[:, :, 0] <= 114) & (hsv[:, :, 1] >= 95))
    not_purple_jaw = ~((hsv[:, :, 0] >= 118) & (hsv[:, :, 0] <= 148) & (hsv[:, :, 1] >= 75))
    valid_fg = (
        not_plate & not_purple_jaw if fg_mask is None else (fg_mask & not_plate & not_purple_jaw)
    )

    interior = cv2.erode(valid_fg.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if int(np.sum(interior)) < 50:
        return 0.0, []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, kernel) * interior

    cand = (blackhat >= 14.0) & interior
    n_comp, lbls, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8))
    best_score = 0.0
    detections: list[dict[str, Any]] = []
    h_img, w_img = L.shape

    for idx in range(1, n_comp):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 9 or area > 280:
            continue
        ys, xs = np.where(lbls == idx)
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(1, len(pts) - 1)
        evals, evecs = np.linalg.eigh(cov)
        l1 = float(max(evals[-1], 1e-6))
        l2 = float(max(evals[0], 0.0))
        linearity = (l1 - l2) / l1
        _, (w_r, h_r), _ = cv2.minAreaRect(pts.astype(np.float32))
        maj = max(float(w_r), float(h_r))
        min_s = max(min(float(w_r), float(h_r)), 1.0)
        if maj < 7.5 or min_s > 6.8 or linearity < 0.70:
            continue

        normal = evecs[:, 0]
        p_plus = np.round(pts + 5.0 * normal[None, :]).astype(int)
        p_minus = np.round(pts - 5.0 * normal[None, :]).astype(int)
        in_bounds = (
            (p_plus[:, 0] >= 0)
            & (p_plus[:, 0] < w_img)
            & (p_plus[:, 1] >= 0)
            & (p_plus[:, 1] < h_img)
            & (p_minus[:, 0] >= 0)
            & (p_minus[:, 0] < w_img)
            & (p_minus[:, 1] >= 0)
            & (p_minus[:, 1] < h_img)
        )
        if int(np.sum(in_bounds)) < 6:
            continue

        pp = p_plus[in_bounds]
        pm = p_minus[in_bounds]
        pv = pts[in_bounds].astype(int)
        both_fg = valid_fg[pp[:, 1], pp[:, 0]] & valid_fg[pm[:, 1], pm[:, 0]]
        if float(np.mean(both_fg)) < 0.72:
            continue

        l_plus = L[pp[:, 1], pp[:, 0]]
        l_minus = L[pm[:, 1], pm[:, 0]]
        l_mid = L[pv[:, 1], pv[:, 0]]
        side_l_diff = float(np.median(np.abs(l_plus - l_minus)))
        ab_diff = float(
            np.median(
                np.hypot(
                    A[pp[:, 1], pp[:, 0]] - A[pm[:, 1], pm[:, 0]],
                    B[pp[:, 1], pp[:, 0]] - B[pm[:, 1], pm[:, 0]],
                )
            )
        )
        flank_contrast = float(np.median(np.minimum(l_plus, l_minus) - l_mid))
        if side_l_diff > 14.5 or ab_diff > 12.5 or flank_contrast < 12.0:
            continue

        raw_score = float(flank_contrast * (0.5 + 0.5 * linearity))
        x0 = int(stats[idx, cv2.CC_STAT_LEFT])
        y0 = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        detections.append(
            {
                "box": [x0, y0, x0 + bw, y0 + bh],
                "score": round(raw_score, 4),
                "label": "dark mark",
                "linearity": round(linearity, 3),
            }
        )
        best_score = max(best_score, raw_score)

    return best_score, detections


@dataclass(frozen=True)
class FrameInspectionResult:
    """Trusted/untrusted result of inspecting a single real camera frame."""

    valid_view: bool
    raw_anomaly_distance: float
    threshold_value: float
    flagged: bool
    detections: list[dict[str, Any]]
    foreground_pixels: int
    reason: str = "ok"


class RealCameraAnomalyInspector:
    """`Inspector` boundary with bounded-timeout worker isolation (zero robot handles!)."""

    def __init__(
        self,
        dark_mark_threshold: float | None = 16.0,
        openvino_detector: Any | None = None,
        worker_timeout_s: float = 2.5,
    ) -> None:
        if dark_mark_threshold is None or not math.isfinite(float(dark_mark_threshold)):
            raise ValueError("RealCameraAnomalyInspector requires a finite dark_mark_threshold")
        if openvino_detector is not None and getattr(openvino_detector, "threshold", None) is None:
            raise ValueError("OpenVINODetector requires a calibrated threshold before deployment")
        self.dark_mark_threshold = float(dark_mark_threshold)
        self.openvino_detector = openvino_detector
        self.worker_timeout_s = float(worker_timeout_s)
        self._executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspector-wkr")
        )
        self.timed_out: bool = False

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _score_sync(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray | None,
        crop_box: list[int],
        expected_z_m: float | None,
        z_band_m: float,
    ) -> FrameInspectionResult:
        u0, v0, u1, v1 = [int(x) for x in crop_box]
        roi_rgb = rgb[v0:v1, u0:u1]
        if roi_rgb.size == 0 or min(roi_rgb.shape[:2]) < 20 or float(roi_rgb.std()) <= 4.0:
            return FrameInspectionResult(
                valid_view=False,
                raw_anomaly_distance=0.0,
                threshold_value=self.dark_mark_threshold,
                flagged=False,
                detections=[],
                foreground_pixels=0,
                reason="empty_or_low_contrast_roi",
            )

        fg_mask: np.ndarray | None = None
        fg_count = int(roi_rgb.shape[0] * roi_rgb.shape[1])
        if depth_m is not None and expected_z_m is not None:
            roi_dep = depth_m[v0:v1, u0:u1]
            fg_mask = (
                np.isfinite(roi_dep)
                & (roi_dep >= max(0.10, expected_z_m - z_band_m))
                & (roi_dep <= expected_z_m + z_band_m)
            )
            fg_count = int(np.sum(fg_mask))
            if fg_count < 45:
                return FrameInspectionResult(
                    valid_view=False,
                    raw_anomaly_distance=0.0,
                    threshold_value=self.dark_mark_threshold,
                    flagged=False,
                    detections=[],
                    foreground_pixels=fg_count,
                    reason=f"insufficient_foreground_depth_pixels({fg_count})",
                )

        dark_score, local_dets = score_brick_roi_dark_mark(roi_rgb, fg_mask=fg_mask)
        full_dets: list[dict[str, Any]] = []
        for d in local_dets:
            bx0, by0, bx1, by1 = d["box"]
            full_dets.append(
                {
                    "box": [u0 + bx0, v0 + by0, u0 + bx1, v0 + by1],
                    "score": d["score"],
                    "label": d["label"],
                }
            )

        flagged = dark_score >= self.dark_mark_threshold
        raw_dist = dark_score
        eff_thr = self.dark_mark_threshold

        if self.openvino_detector is not None:
            bg_iso = np.array([28, 32, 38], dtype=np.uint8)
            roi_iso = (
                np.where(fg_mask[:, :, None], roi_rgb, bg_iso) if fg_mask is not None else roi_rgb
            )
            pred = self.openvino_detector.predict(roi_iso)
            ov_score = float(pred.score)
            ov_thr = float(self.openvino_detector.threshold.value)
            if not math.isfinite(ov_score):
                raise ValueError("Non-finite OpenVINO raw distance")
            if ov_score >= ov_thr:
                flagged = True
                full_dets.append(
                    {
                        "box": [u0, v0, u1, v1],
                        "score": round(ov_score, 4),
                        "label": "patchcore anomaly",
                    }
                )
            if ov_score / max(ov_thr, 1e-6) > raw_dist / max(eff_thr, 1e-6):
                raw_dist = ov_score
                eff_thr = ov_thr

        return FrameInspectionResult(
            valid_view=True,
            raw_anomaly_distance=raw_dist,
            threshold_value=eff_thr,
            flagged=flagged,
            detections=full_dets,
            foreground_pixels=fg_count,
            reason="ok",
        )

    def evaluate_frame(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray | None,
        crop_box: list[int],
        expected_z_m: float | None = None,
        z_band_m: float = 0.065,
    ) -> FrameInspectionResult:
        """Run bounded-timeout inspection; retire worker permanently on timeout (`AGENTS.md`)."""
        if self.timed_out or self._executor is None:
            raise TimeoutError("Inspector worker previously timed out and cannot be reused")
        fut = self._executor.submit(
            self._score_sync, rgb, depth_m, crop_box, expected_z_m, z_band_m
        )
        try:
            return fut.result(timeout=self.worker_timeout_s)
        except concurrent.futures.TimeoutError as exc:
            self.timed_out = True
            # Abandon timed-out worker forever; never reuse it for subsequent cycles!
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
            raise TimeoutError(
                f"Inspector timed out after {self.worker_timeout_s:.2f}s; worker retired"
            ) from exc


def check_bench_obstacle_clearance_m(ik: SO101FraxIK, q_enc_rad: np.ndarray) -> tuple[float, str]:
    """Compute minimum 3D clearance (m) between SO-101 link collision spheres and bench obstacles."""
    frames = {k: (T_WORLD_BASE @ v) for k, v in ik.cal_kin.frames(q_enc_rad).items()}
    link_Test_pts = [
        ("Group 1", np.array([0.0, -0.065, 0.0]), 0.026),
        ("Body1_3", np.array([0.0, -0.065, 0.0]), 0.025),
        ("Group 1_2", np.array([0.0, 0.0, -0.015]), 0.025),
        ("Body3", np.array([0.0, 0.0, -0.045]), 0.032),
        ("Body3", np.array([0.0, 0.0, -0.090]), 0.026),
    ]
    min_clr = float("inf")
    worst_lbl = "none"
    for lname, loc_p, r_link in link_Test_pts:
        T_wl = frames.get(lname)
        if T_wl is None:
            continue
        p_w = (T_wl[:3, :3] @ loc_p) + T_wl[:3, 3]
        for ox, oy, oz, r_obs, lbl in BENCH_OBSTACLE_SPHERES:
            clr = float(np.linalg.norm(p_w - np.array([ox, oy, oz]))) - (r_link + r_obs)
            if clr < min_clr:
                min_clr = clr
                worst_lbl = lbl
    return min_clr, worst_lbl


def solve_adaptive_grasp_pose(
    ik: SO101FraxIK,
    p_target_world: np.ndarray,
    jaw_yaw_world: float,
    q_posture: np.ndarray | None = None,
    pitches_deg: tuple[float, ...] = (90.0, 82.0, 74.0, 65.0, 56.0),
) -> tuple[GraspIKSolution, float]:
    """Solve collision-safe 5-DOF grasp/drop IK within the valid CCW Joint-5 range [+0.45, +3.05] rad."""
    p_tgt = np.asarray(p_target_world, dtype=np.float64)
    seed_posture = (
        np.array([0.0, -0.55, 0.85, 1.05, 1.72], dtype=np.float64)
        if q_posture is None
        else np.asarray(q_posture, dtype=np.float64).copy()
    )
    seed_posture[4] = float(
        np.clip(seed_posture[4], PHYSICAL_JOINT5_CCW_START_RAD, PHYSICAL_JOINT5_CCW_END_RAD)
    )
    best_sol: GraspIKSolution | None = None
    best_pitch = 90.0
    best_score = -1e9

    for pitch in pitches_deg:
        for yaw_cand in (jaw_yaw_world, jaw_yaw_world + math.pi, jaw_yaw_world - math.pi):
            sol = ik.solve_grasp_pose(
                p_tgt,
                jaw_yaw_world=yaw_cand,
                pitch_down_deg=pitch,
                q_posture=seed_posture,
            )
            clr, _ = check_bench_obstacle_clearance_m(ik, sol.q_grasp)
            if clr < 0.012:
                continue
            pos_err = float(np.linalg.norm(sol.p_tcp_world - p_tgt))
            j5 = float(sol.q_grasp[4])
            in_ccw = PHYSICAL_JOINT5_CCW_START_RAD <= j5 <= PHYSICAL_JOINT5_CCW_END_RAD
            if sol.feasible and pos_err <= 0.006 and in_ccw:
                return sol, pitch
            cand_score = (30.0 if in_ccw else -40.0) - 300.0 * pos_err - 0.05 * abs(90.0 - pitch)
            if cand_score > best_score:
                best_score = cand_score
                best_sol = sol
                best_pitch = pitch

    if best_sol is None:
        best_sol = ik.solve_grasp_pose(
            p_tgt, jaw_yaw_world=jaw_yaw_world, pitch_down_deg=65.0, q_posture=seed_posture
        )
        best_pitch = 65.0
    return best_sol, best_pitch


class PhysicalArmController:
    """`RobotInterface` boundary wrapping `ArmdFollowerBridge` with bounded speeds & dry-run mode."""

    def __init__(
        self,
        armd_url: str = "http://127.0.0.1:8765",
        dry_run: bool = False,
        bridge: ArmdFollowerBridge | None = None,
    ) -> None:
        self.armd_url = armd_url.rstrip("/")
        self.dry_run = bool(dry_run)
        self.bridge = bridge or ArmdFollowerBridge(base_url=self.armd_url)
        self._sim_q: np.ndarray = PHYSICAL_OBSERVATION_HOME_Q.copy()
        self._sim_grip: float = PHYSICAL_HOME_GRIPPER_PCT
        state = self.read_state_safe(fresh=True)
        if state is not None:
            self._sim_q = state.q_enc_rad.copy()
            self._sim_grip = float(state.gripper_pct)

    def _try_clear_recoverable_overload(self, raw_state: dict[str, Any]) -> bool:
        """Clear transient overload on axial `wrist_roll` or `gripper` without dropping arm links 1-4."""
        joints = raw_state.get("joints") or {}
        if isinstance(joints, list):
            joints = {j["name"]: j for j in joints if "name" in j}
        cleared_any = False
        for jname in ("wrist_roll", "gripper"):
            j = joints.get(jname) or {}
            errs = list(j.get("errors") or [])
            if j.get("healthy") is False and errs and set(errs).issubset({"overload", "current"}):
                try:
                    self.bridge._call(
                        "POST", f"/arms/follower/joints/{jname}/torque", {"enabled": False}
                    )
                    time.sleep(0.08)
                    self.bridge._call(
                        "POST", f"/arms/follower/joints/{jname}/torque", {"enabled": True}
                    )
                    time.sleep(0.06)
                    cleared_any = True
                except Exception:
                    pass
        return cleared_any

    def _read_with_bus_retry(self, fresh: bool = False) -> DecodedArmState | None:
        for attempt in range(3):
            try:
                raw = self.bridge._call(
                    "GET",
                    f"/arms/follower/state{'?fresh=true' if fresh else ''}",
                    None,
                )
                try:
                    return self.bridge.decode_state_message(raw, stop_on_fault=True)
                except ArmFaultError:
                    if not self.dry_run and self._try_clear_recoverable_overload(raw):
                        raw2 = self.bridge._call("GET", "/arms/follower/state?fresh=true", None)
                        return self.bridge.decode_state_message(raw2, stop_on_fault=True)
                    raise
            except ArmFaultError:
                raise
            except Exception:
                if self.dry_run:
                    return None
                if attempt == 2:
                    raise
                time.sleep(1.15)
        return None

    def read_state_safe(self, fresh: bool = False) -> DecodedArmState | None:
        return self._read_with_bus_retry(fresh=fresh)

    def current_q_and_grip(self, fresh: bool = False) -> tuple[np.ndarray, float]:
        st = self.read_state_safe(fresh=False if not self.dry_run else fresh)
        if st is not None:
            self._sim_q = st.q_enc_rad.copy()
            self._sim_grip = float(st.gripper_pct)
            return st.q_enc_rad.copy(), float(st.gripper_pct)
        return self._sim_q.copy(), float(self._sim_grip)

    def emergency_hold(self) -> None:
        if not self.dry_run:
            try:
                self.bridge.emergency_stop_hold()
            except Exception:
                pass

    def move_to_pose(
        self,
        q_goal: np.ndarray,
        gripper_pct: float,
        speed: int = 185,
        acc: int = 14,
        timeout_s: float = 6.5,
        tolerance_ticks: int = 36,
        include_gripper: bool = True,
        on_step_cb: Any | None = None,
    ) -> DecodedArmState | None:
        """Move physical follower arm smoothly with cosine waypoint interpolation (no staggered torque)."""
        q_clamped = np.clip(
            np.asarray(q_goal, dtype=np.float64),
            self.bridge.calibration.lower,
            self.bridge.calibration.upper,
        )
        # Joint 5 (`wrist_roll`) cannot rotate CW below ~2282 ticks (+0.36 rad); strictly enforce CCW range
        q_clamped[4] = float(
            np.clip(q_clamped[4], PHYSICAL_JOINT5_CCW_START_RAD, PHYSICAL_JOINT5_CCW_END_RAD)
        )
        g_clamped = float(np.clip(gripper_pct, 0.0, 1.0))

        if self.dry_run:
            q_start = self._sim_q.copy()
            for k in range(1, 9):
                tau = k / 8.0
                s = 0.5 * (1.0 - math.cos(math.pi * tau))
                self._sim_q = q_start + s * (q_clamped - q_start)
                if include_gripper:
                    self._sim_grip = g_clamped
                if on_step_cb is not None:
                    on_step_cb(self._sim_q.copy(), self._sim_grip)
            return None

        pre_state = self._read_with_bus_retry(fresh=False)
        if pre_state is None or not pre_state.healthy:
            self.emergency_hold()
            raise ArmFaultError("Follower arm offline or unhealthy prior to motion")

        q_start = pre_state.q_enc_rad.copy()
        dq = q_clamped - q_start
        max_delta = float(np.max(np.abs(dq)))
        eff_speed = max(60, min(int(speed), 280))
        eff_acc = max(10, min(int(acc), 24))

        if max_delta <= 0.06:
            n_segs = 1
            dt_s = 0.0
        else:
            max_ticks = max_delta * (4096.0 / (2.0 * math.pi))
            duration_s = max(0.36, (0.5 * math.pi * max_ticks) / float(eff_speed))
            n_segs = max(2, int(math.ceil(duration_s / 0.065)))
            dt_s = duration_s / n_segs

        t0 = time.monotonic()
        stream_speed = min(460, max(eff_speed, int(round(eff_speed * 1.65))))
        for seg_idx in range(1, n_segs + 1):
            tau = seg_idx / n_segs
            s = 0.5 * (1.0 - math.cos(math.pi * tau))
            q_wp = q_start + s * dq
            is_last = seg_idx == n_segs
            payload = self.bridge.build_positions_payload(
                q_wp,
                g_clamped,
                speed=eff_speed if is_last else stream_speed,
                acc=eff_acc if is_last else 28,
                wait=is_last,
                timeout=float(timeout_s),
                tolerance=int(tolerance_ticks),
                include_gripper=(include_gripper and is_last),
            )
            sent = False
            for attempt in range(3):
                try:
                    self.bridge._call("POST", "/arms/follower/positions", payload)
                    sent = True
                    break
                except Exception as exc:
                    msg = str(exc)
                    if ("wrist_roll" in msg or "gripper" in msg) and (
                        "409" in msg or "joint_fault" in msg or "overload" in msg
                    ):
                        raw_f = self.bridge._call("GET", "/arms/follower/state?fresh=true", None)
                        if self._try_clear_recoverable_overload(raw_f):
                            st_ok = self._read_with_bus_retry(fresh=True)
                            if st_ok is not None:
                                if "wrist_roll" in msg:
                                    q_clamped[4] = st_ok.q_enc_rad[4]
                                    dq = q_clamped - q_start
                                    q_wp[4] = st_ok.q_enc_rad[4]
                                if "gripper" in msg:
                                    g_clamped = float(st_ok.gripper_pct)
                                payload = self.bridge.build_positions_payload(
                                    q_wp,
                                    g_clamped,
                                    speed=eff_speed if is_last else stream_speed,
                                    acc=eff_acc if is_last else 28,
                                    wait=is_last,
                                    timeout=float(timeout_s),
                                    tolerance=int(tolerance_ticks),
                                    include_gripper=(include_gripper and is_last),
                                )
                                continue
                    if isinstance(exc, ArmFaultError):
                        self.emergency_hold()
                        raise
                    if "503" in msg or "bus_error" in msg or "offline" in msg:
                        time.sleep(1.15)
                        continue
                    self.emergency_hold()
                    raise ArmFaultError(f"Motion command failed on armd: {exc}") from exc
            if not sent:
                self.emergency_hold()
                raise ArmFaultError("Follower serial bus dropped repeatedly during motion")

            self._sim_q = q_wp.copy()
            if include_gripper and is_last:
                self._sim_grip = g_clamped
            if not is_last:
                if on_step_cb is not None and seg_idx % 4 == 0:
                    on_step_cb(self._sim_q.copy(), self._sim_grip)
                rem = (t0 + seg_idx * dt_s) - time.monotonic()
                if rem > 0.0:
                    time.sleep(rem)

        post_state = self._read_with_bus_retry(fresh=False)
        if post_state is not None:
            self._sim_q = post_state.q_enc_rad.copy()
            self._sim_grip = float(post_state.gripper_pct)
            if on_step_cb is not None:
                on_step_cb(self._sim_q.copy(), self._sim_grip)
        return post_state

    def close_gripper_until_contact(
        self,
        open_pct: float = PHYSICAL_HOME_GRIPPER_PCT,
        speed: int = 260,
        max_wait_s: float = 3.8,
        on_step_cb: Any | None = None,
    ) -> tuple[np.ndarray, float]:
        """Close gripper jaws at 260 ticks/s until stall on brick (or fully closed), then latch pinch."""
        if self.dry_run:
            self._sim_grip = 0.22
            if on_step_cb is not None:
                on_step_cb(self._sim_q.copy(), self._sim_grip)
            return self._sim_q.copy(), self._sim_grip

        target_ticks = self.bridge.gripper_pct_to_safe_ticks(0.01)
        payload = {
            "joints": {"gripper": {"position": int(target_ticks), "speed": int(speed), "acc": 22}},
            "wait": False,
            "timeout": float(max_wait_s),
            "tolerance": 12,
        }
        self.bridge._call("POST", "/arms/follower/positions", payload)

        t_deadline = time.monotonic() + float(max_wait_s)
        g_prev = float(open_pct)
        settled_polls = 0
        q_now = self._sim_q.copy()
        g_now = g_prev
        time.sleep(0.28)
        while time.monotonic() < t_deadline:
            q_now, g_now = self.current_q_and_grip(fresh=True)
            if on_step_cb is not None:
                on_step_cb(q_now, g_now)
            if g_now <= 0.035:
                break
            if g_now < (open_pct - 0.06) and abs(g_now - g_prev) < 0.008:
                settled_polls += 1
                if settled_polls >= 2:
                    break
            else:
                settled_polls = 0
            g_prev = g_now
            time.sleep(0.16)
        return q_now, g_now


class PhysicalSO101CellExecutor:
    """`SkillExecutor` replicating the 3-plate simulation architecture on the physical cell."""

    def __init__(
        self,
        frame_source: LiveBenchHttpFrameSource,
        arm: PhysicalArmController,
        inspector: RealCameraAnomalyInspector,
        dashboard: TwinDashboardReporter | None = None,
        camui: CamUiReporter | None = None,
        hud: LiveInspectionHudServer | None = None,
        show_twin_viewer: bool = False,
        max_cycles: int = 10,
    ) -> None:
        self.frame_source = frame_source
        self.arm = arm
        self.inspector = inspector
        self.dashboard = dashboard
        self.camui = camui
        self.hud = hud
        self.max_cycles = int(max_cycles)

        self.rig = station_camera_rig()
        self.real_cams = real_intel_camera_poses_world()
        self.ik = SO101FraxIK()
        self.real2sim = Real2SimPipeline(rig=self.rig, enable_temporal_tracking=False)

        slot_specs = list(DUPLO_CATALOG.values())
        prealloc = [
            SpawnedBlockInfo(
                block_id=f"block_{i}",
                spec=slot_specs[i % len(slot_specs)],
                initial_xy_yaw=(0.28, 0.0, 0.0),
            )
            for i in range(10)
        ]
        self.twin = SO101InspectionTwin(
            block_spec=prealloc[0].spec,
            rig=self.rig,
            show_viewer=show_twin_viewer,
            show_camera_windows=False,
            realtime_pacing=False,
            spawned_blocks=prealloc,
        )
        self.ledger: EightCornerCoverageLedger = self.twin.ledger

        self.hand_empty: bool = True
        self.at_rest: bool = False
        self.source_plate_cleared: bool = False
        self.remaining_source_blocks: int = 0
        self.sorted_counts: dict[str, int] = {"good": 0, "defective": 0}
        self.batch_sorted_records: list[dict[str, Any]] = []
        self.plate_scan_active: bool = False
        self.plate_scan_completed: bool = False
        self.plate_scan_valid_frames: int = 0
        self.anomaly_flagged: bool = False
        self.anomaly_messages: list[str] = []
        self.anomaly_untrusted_error: str | None = None
        self.max_plate_anomaly_distance: float | None = None
        self.sorted_pile: str | None = None
        self.status_banner: str = "Physical SO-101 Cell Ready"
        self.synthetic_simulation: bool = False
        self.plate_scan_keyframes: list[np.ndarray] = []

        self._latest_scene: Real2SimSceneState | None = None
        self._active_brick: PerceivedBrick | None = None
        self._active_grasp_sol: GraspIKSolution | None = None
        self._active_open_pct: float = PHYSICAL_HOME_GRIPPER_PCT
        self._active_close_pct: float = 0.14
        self._T_body3_block_held: np.ndarray = np.eye(4, dtype=np.float64)
        self._cycles_completed: int = 0
        self._recent_missed_xys: list[np.ndarray] = []

    def close(self) -> None:
        if self.dashboard is not None:
            self.dashboard.clear_focus()
            self.dashboard.clear_anomalies()
            self.dashboard.stop_keepalive()
        self.inspector.close()
        self.twin.close()

    def _sync_mirror_and_telemetry(
        self,
        q_rad: np.ndarray | None = None,
        grip_pct: float | None = None,
        top_rgb: np.ndarray | None = None,
        plate_rgb: np.ndarray | None = None,
        wrist_rgb: np.ndarray | None = None,
    ) -> None:
        if q_rad is None or grip_pct is None:
            q_cur, g_cur = self.arm.current_q_and_grip(fresh=False)
            q_rad = q_cur if q_rad is None else q_rad
            grip_pct = g_cur if grip_pct is None else grip_pct

        self.twin.set_arm_state_direct(q_rad, grip_pct, sync_viewer=True)
        if self.camui is not None:
            malformed = (
                True if self.anomaly_flagged else (False if self.plate_scan_completed else None)
            )
            self.camui.report(malformed=malformed, detail=self.status_banner)

        if self.hud is not None:
            frames_map: dict[str, np.ndarray] = {}
            if top_rgb is not None:
                frames_map["realsense_1"] = top_rgb
            if plate_rgb is not None:
                frames_map["realsense_2"] = plate_rgb
            if wrist_rgb is not None:
                frames_map["wrist"] = wrist_rgb
            status = self.ledger.summary_dict()
            status.update(
                {
                    "synthetic_simulation": False,
                    "dry_run": self.arm.dry_run,
                    "banner": self.status_banner,
                    "active_part": self._active_brick.spec.name if self._active_brick else "none",
                    "q_deg": np.round(np.degrees(q_rad), 1).tolist(),
                    "gripper_pct": round(float(grip_pct), 3),
                    "hand_empty": self.hand_empty,
                    "remaining_source_blocks": self.remaining_source_blocks,
                    "source_plate_cleared": self.source_plate_cleared,
                    "plate_scan_active": self.plate_scan_active,
                    "plate_scan_completed": self.plate_scan_completed,
                    "plate_scan_valid_frames": self.plate_scan_valid_frames,
                    "score_type": "raw_anomaly_distance_not_probability",
                    "max_plate_anomaly_distance": self.max_plate_anomaly_distance,
                    "anomaly_threshold": self.inspector.dark_mark_threshold,
                    "anomaly_flagged": self.anomaly_flagged,
                    "anomaly_messages": list(self.anomaly_messages),
                    "sorted_pile": self.sorted_pile,
                    "sorted_counts": dict(self.sorted_counts),
                    "batch_sorted_records": list(self.batch_sorted_records),
                }
            )
            self.hud.update_from_frames_and_status(frames_map, status)

    def _project_to_top_box(
        self, T_world_block: np.ndarray, dims_m: tuple[float, float, float]
    ) -> list[int] | None:
        R_top, p_top, K_top = self.real_cams["top"]
        box, _ = project_world_obb_to_image_box(
            T_world_block, dims_m, R_top, p_top, K_top, width=640, height=480, pad_px=8
        )
        return box

    def current_arm_q(self) -> np.ndarray:
        return self.arm.current_q_and_grip(fresh=False)[0]

    def corner_seed_q(self, corner_index: int) -> np.ndarray | None:
        del corner_index
        return None

    def perceive_symbolic_state(self, block_id: str = "block_0") -> SymbolicState:
        return SymbolicState(
            block_id=block_id,
            hand_empty=self.hand_empty,
            block_localized=self.ledger.localized,
            block_on_table=self.hand_empty and (self.sorted_pile is None),
            block_held=not self.hand_empty,
            block_at_rest=self.at_rest,
            corner_upper=(False,) * 8,
            corner_wrist_reachable=(False,) * 8,
            corner_inspected=(False,) * 8,
            plate_scan_completed=self.plate_scan_completed,
            anomaly_flagged=self.anomaly_flagged,
            sorted_pile=self.sorted_pile,
            source_plate_cleared=self.source_plate_cleared,
            remaining_source_blocks=self.remaining_source_blocks,
            sorted_good_count=self.sorted_counts["good"],
            sorted_defective_count=self.sorted_counts["defective"],
        )

    def execute_operator(
        self, op: GroundedOperator, cone_perturbation: np.ndarray | None = None
    ) -> SkillOutcome:
        del cone_perturbation
        self.at_rest = False
        if op.name == "LocalizeNextClusterBlock":
            return self._skill_localize_next_cluster_block()
        if op.name == "PickBlock":
            return self._skill_pick_block()
        if op.name == "PresentAndSpinAtPlateCamera":
            return self._skill_present_and_spin_at_plate_camera()
        if op.name == "SortBlockToPile":
            return self._skill_sort_block_to_pile(str(op.params.get("pile", "good")))
        if op.name == "ParkAtRest":
            return self._skill_park_at_rest()
        return SkillOutcome(False, reason=f"Unsupported physical operator: {op.name}")

    def _select_physical_source_brick(
        self, source_bricks: list[PerceivedBrick], source_xy: np.ndarray
    ) -> tuple[PerceivedBrick, GraspIKSolution, float, float] | None:
        """Choose the safest reachable DUPLO brick in the physical source plate."""
        best_item: tuple[PerceivedBrick, GraspIKSolution, float, float] | None = None
        best_score = -1e9

        for b in source_bricks:
            T_wb = b.T_world_block
            p_b = T_wb[:3, 3]
            R_wb = T_wb[:3, :3]
            hz = float(b.spec.dims_m[2])
            z_grasp = max(float(p_b[2]) + (0.006 if hz > 0.032 else 0.001), 0.018)

            yaw_spans: list[tuple[float, float]] = []
            for idx, span in enumerate(b.spec.dims_m):
                ax = R_wb[:, idx]
                if abs(float(ax[2])) < 0.30 and span <= 0.038:
                    y0 = math.atan2(float(ax[1]), float(ax[0]))
                    yaw_spans.extend([(y0, float(span)), (y0 + math.pi, float(span))])
            if not yaw_spans:
                y0 = math.atan2(float(R_wb[1, 1]), float(R_wb[0, 1]))
                yaw_spans = [(y0, 0.032), (y0 + math.pi, 0.032)]

            for yaw, span in yaw_spans:
                is_narrow = span <= 0.020
                open_pct = 0.36 if is_narrow else PHYSICAL_HOME_GRIPPER_PCT
                close_pct = 0.07 if is_narrow else 0.14
                shift_m = -0.006 if is_narrow else 0.0
                p_tcp = np.array(
                    [
                        float(p_b[0]) + shift_m * math.cos(yaw),
                        float(p_b[1]) + shift_m * math.sin(yaw),
                        z_grasp,
                    ],
                    dtype=np.float64,
                )
                sol, pitch_used = solve_adaptive_grasp_pose(self.ik, p_tcp, jaw_yaw_world=yaw)
                pos_err = float(np.linalg.norm(sol.p_tcp_world - p_tcp))
                if not sol.feasible and pos_err > 0.008:
                    continue

                # Check distance to neighboring bricks along jaw opening direction
                jaw_dir = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
                neighbor_pen = 0.0
                for other in source_bricks:
                    if other.block_id == b.block_id:
                        continue
                    d_xy = other.T_world_block[:2, 3] - p_tcp[:2]
                    if float(np.linalg.norm(d_xy)) < 0.048:
                        u_j = abs(float(np.dot(d_xy, jaw_dir)))
                        if 0.015 <= u_j <= 0.042:
                            neighbor_pen += 8.0

                miss_pen = 0.0
                for m_xy in self._recent_missed_xys[-4:]:
                    if float(np.linalg.norm(p_b[:2] - m_xy)) < 0.026:
                        miss_pen += 18.0

                r_robot = float(math.hypot(p_b[0], p_b[1]))
                r_plate = float(np.linalg.norm(p_b[:2] - source_xy))
                score = (
                    (20.0 if sol.feasible else 0.0)
                    - 400.0 * pos_err
                    - neighbor_pen
                    - miss_pen
                    - 12.0 * r_robot
                    + 4.0 * r_plate
                    - 0.08 * abs(90.0 - pitch_used)
                    + 2.0 * b.confidence
                )
                if score > best_score:
                    best_score = score
                    best_item = (b, sol, open_pct, close_pct)

        return best_item

    def _skill_localize_next_cluster_block(self) -> SkillOutcome:
        """Retract to clear observation pose, perceive live scene via Real2Sim, & pick next brick."""
        self.plate_scan_active = False
        self.plate_scan_completed = False
        self.plate_scan_valid_frames = 0
        self.anomaly_flagged = False
        self.anomaly_messages.clear()
        self.anomaly_untrusted_error = None
        self.max_plate_anomaly_distance = None
        self.sorted_pile = None
        self._active_brick = None
        self._active_grasp_sol = None
        self.ledger.reset_all()

        if self._cycles_completed >= self.max_cycles:
            self.source_plate_cleared = True
            self.remaining_source_blocks = 0
            self.status_banner = (
                f"Cycle limit ({self.max_cycles}) reached — "
                f"Good: {self.sorted_counts['good']} | Defective: {self.sorted_counts['defective']}"
            )
            self._sync_mirror_and_telemetry()
            return SkillOutcome(
                True,
                reason="max_cycles_completed",
                metadata={"sorted_counts": dict(self.sorted_counts)},
            )

        self.status_banner = (
            "LocalizeNextClusterBlock: Retracting arm & scanning live Dual-D415 cloud"
        )
        q_pre_home, _ = self.arm.current_q_and_grip(fresh=False)
        if float(q_pre_home[1]) > -0.60:
            q_lift = q_pre_home.copy()
            q_lift[1] = -0.95
            self.arm.move_to_pose(
                q_lift,
                gripper_pct=PHYSICAL_HOME_GRIPPER_PCT,
                speed=220,
                include_gripper=False,
                on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
            )
        self.arm.move_to_pose(
            PHYSICAL_OBSERVATION_HOME_Q,
            gripper_pct=PHYSICAL_HOME_GRIPPER_PCT,
            speed=240,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        if not self.arm.dry_run:
            time.sleep(0.25)

        q_now, g_now = self.arm.current_q_and_grip(fresh=True)
        top_snap = self.frame_source.fetch_d415_snapshot("top")
        plate_snap = self.frame_source.fetch_d415_snapshot("plate")
        wrist_snap = self.frame_source.fetch_wrist_snapshot()

        scene = self.frame_source.perceive_scene(self.real2sim, q_arm_rad=q_now)

        self._latest_scene = scene
        self.twin.sync_real2sim_scene(scene, settle_steps=0)

        source_bricks = [b for b in scene.bricks if b.support_region == "source"]
        self.remaining_source_blocks = len(source_bricks)
        if not source_bricks:
            self.source_plate_cleared = True
            self.status_banner = (
                f"Source Plate Cleared! Good: {self.sorted_counts['good']} | "
                f"Defective: {self.sorted_counts['defective']}"
            )
            if self.dashboard is not None:
                self.dashboard.clear_focus()
                self.dashboard.clear_anomalies()
            self._sync_mirror_and_telemetry(
                q_now, g_now, top_snap.rgb, plate_snap.rgb, wrist_snap.rgb
            )
            return SkillOutcome(
                True,
                reason="source_plate_empty",
                metadata={"remaining_blocks": 0, "sorted_counts": dict(self.sorted_counts)},
            )

        source_xy = scene.plates["source"].pos_world[:2]
        picked = self._select_physical_source_brick(source_bricks, source_xy)
        if picked is None:
            return SkillOutcome(False, reason="no_reachable_brick_in_source_plate")

        brick, grasp_sol, open_pct, close_pct = picked
        self._active_brick = brick
        self._active_grasp_sol = grasp_sol
        self._active_open_pct = open_pct
        self._active_close_pct = close_pct
        self.ledger.block_spec = brick.spec
        self.ledger.set_initial_pose(brick.T_world_block)

        dash_obj_id = PART_ID_TO_DASHBOARD_ID.get(brick.spec.part_id, "part-001")
        top_box = self._project_to_top_box(brick.T_world_block, brick.spec.dims_m)
        if self.dashboard is not None and top_box is not None:
            self.dashboard.report_focus(
                object_id=dash_obj_id,
                label=f"{brick.spec.name} (target)",
                box=top_box,
            )

        # Evaluate visible resting surfaces across all source bricks so `:8096` dashboard displays
        # both the active focus block AND all visible surface anomalies in real time
        R_pl, p_pl, K_pl = self.real_cams["plate"]
        all_dash_anoms: list[dict[str, Any]] = []
        for cand_b in source_bricks:
            cand_pbox, cand_z = project_world_obb_to_image_box(
                cand_b.T_world_block,
                cand_b.spec.dims_m,
                R_pl,
                p_pl,
                K_pl,
                width=640,
                height=480,
                pad_px=10,
            )
            if cand_pbox is None:
                continue
            try:
                cand_res = self.inspector.evaluate_frame(
                    plate_snap.rgb,
                    plate_snap.depth_m,
                    cand_pbox,
                    expected_z_m=cand_z,
                    z_band_m=0.065,
                )
            except Exception as exc:
                if cand_b.block_id == brick.block_id:
                    self.anomaly_untrusted_error = f"pre_pick_inspector_error:{exc}"
                    return SkillOutcome(False, reason=self.anomaly_untrusted_error)
                continue

            if cand_b.block_id == brick.block_id and cand_res.valid_view:
                self.max_plate_anomaly_distance = cand_res.raw_anomaly_distance
                if cand_res.flagged:
                    self.anomaly_flagged = True
                    self.anomaly_messages.append(
                        f"pre_pick_plate_dark_mark={cand_res.raw_anomaly_distance:.2f}>={cand_res.threshold_value:.2f}"
                    )

            if cand_res.valid_view and cand_res.flagged:
                cand_tbox = self._project_to_top_box(cand_b.T_world_block, cand_b.spec.dims_m)
                if cand_tbox is not None:
                    all_dash_anoms.append(
                        {
                            "box": cand_tbox,
                            "score": cand_res.raw_anomaly_distance,
                            "label": f"dark mark ({cand_res.raw_anomaly_distance:.1f})",
                            "object_id": PART_ID_TO_DASHBOARD_ID.get(
                                cand_b.spec.part_id, "part-001"
                            ),
                        }
                    )

        if self.dashboard is not None:
            self.dashboard.report_anomalies(all_dash_anoms)

        self.status_banner = (
            f"Target Locked: {brick.spec.name} at ({brick.T_world_block[0, 3]:.3f}, "
            f"{brick.T_world_block[1, 3]:.3f}) | {len(source_bricks)} in source"
        )
        self._sync_mirror_and_telemetry(q_now, g_now, top_snap.rgb, plate_snap.rgb, wrist_snap.rgb)
        return SkillOutcome(
            True,
            reason="cluster_block_localized",
            metadata={
                "remaining_blocks": len(source_bricks),
                "inferred_part": brick.spec.name,
                "pos_world_m": [round(float(x), 4) for x in brick.T_world_block[:3, 3]],
            },
        )

    def _skill_pick_block(self) -> SkillOutcome:
        """Execute safe pre-grasp -> top-down/adaptive grasp -> compliant clamp -> vertical lift."""
        if self._active_brick is None or self._active_grasp_sol is None:
            return SkillOutcome(False, reason="no_localized_brick")

        brick = self._active_brick
        grasp_sol = self._active_grasp_sol
        p_tcp = grasp_sol.p_tcp_world
        pre_sol, _ = solve_adaptive_grasp_pose(
            self.ik,
            np.array([p_tcp[0], p_tcp[1], float(p_tcp[2]) + 0.085], dtype=np.float64),
            jaw_yaw_world=grasp_sol.jaw_yaw_world,
            q_posture=grasp_sol.q_grasp,
        )

        self.status_banner = f"PickBlock({brick.spec.part_id}): Approaching above source plate"
        self.arm.move_to_pose(
            pre_sol.q_grasp,
            gripper_pct=self._active_open_pct,
            speed=230,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )

        self.status_banner = f"PickBlock({brick.spec.part_id}): Descending & clamping jaws"
        self.arm.move_to_pose(
            grasp_sol.q_grasp,
            gripper_pct=self._active_open_pct,
            speed=155,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        q_clamped, g_contact = self.arm.close_gripper_until_contact(
            open_pct=self._active_open_pct,
            speed=260,
            max_wait_s=3.8,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        if not self.arm.dry_run and (
            float(g_contact) <= 0.052 or float(g_contact) >= (self._active_open_pct - 0.03)
        ):
            # Jaws closed empty or stopped early — record XY, lift out, and re-localize fresh scene
            self._recent_missed_xys.append(brick.T_world_block[:2, 3].copy())
            self.arm.move_to_pose(
                pre_sol.q_grasp,
                gripper_pct=PHYSICAL_HOME_GRIPPER_PCT,
                speed=240,
                timeout_s=3.2,
                on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
            )
            self.hand_empty = True
            self.ledger.reset_all()
            return SkillOutcome(False, reason=f"grasp_missed_retrying(g={g_contact:.3f})")
        if not self.arm.dry_run:
            # Latch compliant pinch ~80 ticks tighter than contact width for rock-solid carry
            self._active_close_pct = max(0.015, float(g_contact) - 0.055)
            self.arm.move_to_pose(
                grasp_sol.q_grasp,
                gripper_pct=self._active_close_pct,
                speed=180,
                timeout_s=0.65,
            )

        T_body3_grasp = self.ik.fk_world_body3(q_clamped)
        self._T_body3_block_held = np.linalg.inv(T_body3_grasp) @ brick.T_world_block

        self.status_banner = f"PickBlock({brick.spec.part_id}): Lifting clear of source plate"
        self.arm.move_to_pose(
            pre_sol.q_grasp,
            gripper_pct=self._active_close_pct,
            speed=190,
            include_gripper=False,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        self.hand_empty = False
        return SkillOutcome(
            True,
            reason="block_picked",
            metadata={
                "part_name": brick.spec.name,
                "contact_gripper_pct": round(float(g_contact), 3),
                "close_gripper_pct": round(float(self._active_close_pct), 3),
            },
        )

    def _skill_present_and_spin_at_plate_camera(self) -> SkillOutcome:
        """Move held brick to low Plate camera and sweep Joint 5 CCW while scoring live frames."""
        if self.hand_empty or self._active_brick is None:
            return SkillOutcome(False, reason="block_not_held")

        brick = self._active_brick
        dash_obj_id = PART_ID_TO_DASHBOARD_ID.get(brick.spec.part_id, "part-001")
        radial_az = math.atan2(float(PHYSICAL_PLATE_SCAN_POS[1]), float(PHYSICAL_PLATE_SCAN_POS[0]))
        pres_sol = self.ik.solve_grasp_pose(
            PHYSICAL_PLATE_SCAN_POS,
            jaw_yaw_world=radial_az + 0.5 * math.pi,
            pitch_down_deg=56.0,
        )

        q_scan_start = pres_sol.q_grasp.copy()
        q_scan_start[4] = max(
            float(self.ik.cal_kin.calibration.lower[4]) + 0.08, PHYSICAL_JOINT5_CCW_START_RAD
        )
        q_scan_end = pres_sol.q_grasp.copy()
        q_scan_end[4] = min(
            float(self.ik.cal_kin.calibration.upper[4]) - 0.08, PHYSICAL_JOINT5_CCW_END_RAD
        )

        q_cur, _ = self.arm.current_q_and_grip(fresh=False)
        q_carry = 0.5 * (q_cur + q_scan_start)
        q_carry[1] = min(float(q_cur[1]), float(q_scan_start[1]), -0.65)
        self.status_banner = "PresentAndSpinAtPlateCamera: Moving held brick to Plate D415"
        self.arm.move_to_pose(
            q_carry,
            gripper_pct=self._active_close_pct,
            speed=220,
            include_gripper=False,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        self.arm.move_to_pose(
            q_scan_start,
            gripper_pct=self._active_close_pct,
            speed=200,
            include_gripper=False,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )

        self.plate_scan_active = True
        self.plate_scan_valid_frames = 0
        self.plate_scan_keyframes.clear()
        R_pl, p_pl, K_pl = self.real_cams["plate"]
        n_stops = 6
        seen_frame_ids: set[str] = set()
        verified_held_in_depth = False

        for step_idx in range(n_stops + 1):
            tau = step_idx / n_stops
            q_step = q_scan_start.copy()
            q_step[4] = float(q_scan_start[4] + tau * (q_scan_end[4] - q_scan_start[4]))
            swept_deg = math.degrees(q_step[4] - q_scan_start[4])
            self.status_banner = (
                f"Plate Camera CCW Spin +{swept_deg:.0f} deg | "
                f"{'ANOMALY FLAGGED' if self.anomaly_flagged else 'Scanning...'}"
            )
            self.arm.move_to_pose(
                q_step,
                gripper_pct=self._active_close_pct,
                speed=240,
                timeout_s=5.0,
                include_gripper=False,
            )
            if not self.arm.dry_run:
                time.sleep(0.16)

            q_meas, g_meas = self.arm.current_q_and_grip(fresh=True)
            T_held = self.ik.fk_world_body3(q_meas) @ self._T_body3_block_held
            top_box = self._project_to_top_box(T_held, brick.spec.dims_m)
            if self.dashboard is not None and top_box is not None:
                self.dashboard.report_focus(
                    object_id=dash_obj_id,
                    label=f"{brick.spec.name} (CCW +{swept_deg:.0f} deg)",
                    box=top_box,
                )

            try:
                plate_snap = self.frame_source.fetch_d415_snapshot("plate")
                top_snap = self.frame_source.fetch_d415_snapshot("top")
                wrist_snap = self.frame_source.fetch_wrist_snapshot()
            except StaleCameraFrameError as exc:
                self.anomaly_untrusted_error = f"stale_camera_frame:{exc}"
                break

            if not self.arm.dry_run and plate_snap.frame_id in seen_frame_ids:
                continue
            seen_frame_ids.add(plate_snap.frame_id)

            if self.arm.dry_run:
                # In zero-motion dry-run, score the brick's resting Plate ROI across spin steps
                eval_T = brick.T_world_block
            else:
                eval_T = T_held

            plate_box, _ = project_world_obb_to_image_box(
                eval_T,
                brick.spec.dims_m,
                R_pl,
                p_pl,
                K_pl,
                width=640,
                height=480,
                pad_px=22,
            )
            if plate_box is None:
                continue

            try:
                res = self.inspector.evaluate_frame(
                    plate_snap.rgb,
                    None,
                    plate_box,
                    expected_z_m=None,
                    z_band_m=0.075,
                )
            except Exception as exc:
                self.anomaly_untrusted_error = f"inspector_failure:{exc}"
                break

            if res.valid_view:
                self.plate_scan_valid_frames += 1
                if step_idx in (0, 2, 4, 6) and len(self.plate_scan_keyframes) < 4:
                    self.plate_scan_keyframes.append(plate_snap.rgb.copy())
                if self.arm.dry_run or g_meas >= 0.048 or res.foreground_pixels >= 55:
                    verified_held_in_depth = True
                self.max_plate_anomaly_distance = (
                    res.raw_anomaly_distance
                    if self.max_plate_anomaly_distance is None
                    else max(self.max_plate_anomaly_distance, res.raw_anomaly_distance)
                )
                if res.flagged:
                    self.anomaly_flagged = True
                    self.anomaly_messages.append(
                        f"plate_spin_dark_mark={res.raw_anomaly_distance:.2f}>={res.threshold_value:.2f}"
                    )
                    if self.dashboard is not None and top_box is not None:
                        self.dashboard.report_anomalies(
                            [
                                {
                                    "box": top_box,
                                    "score": res.raw_anomaly_distance,
                                    "label": "dark mark",
                                    "object_id": dash_obj_id,
                                }
                            ]
                        )

            self._sync_mirror_and_telemetry(
                q_meas, g_meas, top_snap.rgb, plate_snap.rgb, wrist_snap.rgb
            )

        self.plate_scan_active = False
        if self.anomaly_untrusted_error is not None:
            return SkillOutcome(False, reason=self.anomaly_untrusted_error)
        if not verified_held_in_depth:
            # Grasp slipped — open jaws safely at observation home and refuse untrusted sort
            self.hand_empty = True
            self.arm.move_to_pose(
                PHYSICAL_OBSERVATION_HOME_Q, gripper_pct=PHYSICAL_HOME_GRIPPER_PCT, speed=200
            )
            return SkillOutcome(False, reason="pick_not_verified_in_plate_depth")
        if self.plate_scan_valid_frames < 3:
            return SkillOutcome(
                False, reason=f"insufficient_valid_plate_views({self.plate_scan_valid_frames})"
            )

        self.plate_scan_completed = True
        return SkillOutcome(
            True,
            reason="plate_ccw_scan_completed",
            metadata={
                "valid_plate_views": self.plate_scan_valid_frames,
                "anomaly_flagged": self.anomaly_flagged,
                "max_raw_anomaly_distance": self.max_plate_anomaly_distance,
            },
        )

    def _compute_safe_plate_drop_xy(self, pile: str, slot_idx: int) -> np.ndarray:
        """Compute reachable, collision-safe drop XY inside the localized `good` or `defective` plate."""
        if self._latest_scene is not None and pile in self._latest_scene.plates:
            plate_xy = self._latest_scene.plates[pile].pos_world[:2].copy()
        else:
            plate_xy = np.array([0.35, 0.14] if pile == "good" else [0.22, 0.14], dtype=np.float64)

        # Pull drop anchor up to 36 mm radially inward toward the robot base when plate center is far
        # (stays well inside the 74 mm flat inner bowl of the 19 cm KALAS plate!)
        r_center = float(np.linalg.norm(plate_xy))
        if r_center > 0.340:
            inward = min(0.038, r_center - 0.335)
            plate_xy = plate_xy * ((r_center - inward) / r_center)

        dx, dy = PLATE_DROP_SLOT_OFFSETS_XY[slot_idx % len(PLATE_DROP_SLOT_OFFSETS_XY)]
        target_xy = np.array([plate_xy[0] + 0.55 * dx, plate_xy[1] + 0.55 * dy], dtype=np.float64)
        # Keep strictly clear of left laptop screen (y <= +0.168 m)
        target_xy[1] = min(float(target_xy[1]), 0.168)
        return target_xy

    def _skill_sort_block_to_pile(self, pile: str) -> SkillOutcome:
        """Sort inspected brick into the physical `good` (green) or `defective` (pink) KALAS plate."""
        if self.hand_empty or not self.plate_scan_completed or self.anomaly_untrusted_error:
            return SkillOutcome(False, reason="untrusted_or_unscanned_sort_refused")
        if pile not in {"good", "defective"}:
            return SkillOutcome(False, reason=f"invalid_pile:{pile}")

        slot_idx = int(self.sorted_counts[pile])
        target_xy = self._compute_safe_plate_drop_xy(pile, slot_idx)
        drop_yaw = math.atan2(float(target_xy[1]), float(target_xy[0])) + 0.5 * math.pi

        above_sol, _ = solve_adaptive_grasp_pose(
            self.ik,
            np.array([target_xy[0], target_xy[1], 0.135], dtype=np.float64),
            jaw_yaw_world=drop_yaw,
        )
        release_sol, _ = solve_adaptive_grasp_pose(
            self.ik,
            np.array([target_xy[0], target_xy[1], 0.062], dtype=np.float64),
            jaw_yaw_world=drop_yaw,
            q_posture=above_sol.q_grasp,
        )

        self.status_banner = f"SortBlockToPile({pile.upper()}): Transiting to {pile} KALAS plate"
        # Unwind Joint 5 smoothly at safe elevation before lateral swing
        q_cur, _ = self.arm.current_q_and_grip(fresh=False)
        q_mid = 0.5 * (q_cur + above_sol.q_grasp)
        q_mid[1] = min(float(q_cur[1]), float(above_sol.q_grasp[1]), -0.75)
        q_mid[4] = float(above_sol.q_grasp[4])
        self.arm.move_to_pose(
            q_mid,
            gripper_pct=self._active_close_pct,
            speed=220,
            include_gripper=False,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        self.arm.move_to_pose(
            above_sol.q_grasp,
            gripper_pct=self._active_close_pct,
            speed=210,
            include_gripper=False,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        self.arm.move_to_pose(
            release_sol.q_grasp,
            gripper_pct=self._active_close_pct,
            speed=160,
            include_gripper=False,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        self.arm.move_to_pose(
            release_sol.q_grasp,
            gripper_pct=PHYSICAL_HOME_GRIPPER_PCT,
            speed=260,
            timeout_s=3.2,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        if not self.arm.dry_run:
            time.sleep(0.18)
        self.hand_empty = True

        self.arm.move_to_pose(
            above_sol.q_grasp,
            gripper_pct=PHYSICAL_HOME_GRIPPER_PCT,
            speed=220,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )

        self.sorted_pile = pile
        self.sorted_counts[pile] += 1
        self._cycles_completed += 1
        self.remaining_source_blocks = max(0, self.remaining_source_blocks - 1)

        part_name = self._active_brick.spec.name if self._active_brick else "unknown"
        self.batch_sorted_records.append(
            {
                "order": len(self.batch_sorted_records) + 1,
                "inferred_part_name": part_name,
                "anomaly_flagged": bool(self.anomaly_flagged),
                "max_plate_anomaly_distance": (
                    round(float(self.max_plate_anomaly_distance), 4)
                    if self.max_plate_anomaly_distance is not None
                    else None
                ),
                "sorted_pile": pile,
                "slot_idx": slot_idx,
                "drop_xy_m": [round(float(target_xy[0]), 4), round(float(target_xy[1]), 4)],
                "dry_run": self.arm.dry_run,
            }
        )
        self._sync_mirror_and_telemetry()
        return SkillOutcome(
            True,
            reason=f"sorted_to_{pile}",
            metadata={
                "pile": pile,
                "slot_idx": slot_idx,
                "drop_xy_m": [round(float(target_xy[0]), 4), round(float(target_xy[1]), 4)],
                "sorted_counts": dict(self.sorted_counts),
            },
        )

    def _skill_park_at_rest(self) -> SkillOutcome:
        self.plate_scan_active = False
        self.status_banner = (
            f"ParkAtRest: Completed {self._cycles_completed} cycle(s) "
            f"(Good: {self.sorted_counts['good']} | Defective: {self.sorted_counts['defective']})"
        )
        self.arm.move_to_pose(
            PHYSICAL_OBSERVATION_HOME_Q,
            gripper_pct=PHYSICAL_HOME_GRIPPER_PCT,
            speed=220,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        self.at_rest = True
        self._sync_mirror_and_telemetry()
        return SkillOutcome(True, reason="parked_at_rest")


def run_physical_cell_deployment(
    live_depth_url: str = "http://127.0.0.1:8094",
    wrist_http_url: str = "http://127.0.0.1:8091/frame.jpg",
    armd_url: str = "http://127.0.0.1:8765",
    dashboard_url: str = "http://127.0.0.1:8096",
    camui_url: str = "http://127.0.0.1:8093",
    hud_host: str = "127.0.0.1",
    hud_port: int = 8095,
    dry_run: bool = False,
    max_cycles: int = 8,
    daemon_standby_s: float = 0.0,
    show_twin_viewer: bool = False,
    dark_mark_threshold: float = 16.0,
    openvino_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full physical SO-101 3-plate inspection deployment on `intel`."""
    out_root = (
        Path(output_dir)
        if output_dir is not None
        else (REPO_ROOT / "artifacts" / "block_inspector" / "hardware_deploy")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    ov_detector = None
    if openvino_config_path is not None:
        from brickcell.config import load_config
        from brickcell.detector.runtime import OpenVINODetector

        ov_detector = OpenVINODetector(load_config(Path(openvino_config_path)))

    frame_source = LiveBenchHttpFrameSource(
        live_depth_url=live_depth_url,
        wrist_http_url=wrist_http_url,
    )
    inspector = RealCameraAnomalyInspector(
        dark_mark_threshold=dark_mark_threshold,
        openvino_detector=ov_detector,
    )
    arm = PhysicalArmController(armd_url=armd_url, dry_run=dry_run)
    dashboard = TwinDashboardReporter(base_url=dashboard_url) if dashboard_url else None
    camui = CamUiReporter(base_url=camui_url) if camui_url else None
    hud = LiveInspectionHudServer(host=hud_host, port=hud_port) if hud_port > 0 else None

    if hud is not None:
        hud.start()
    if dashboard is not None:
        dashboard.start_keepalive(period_s=1.8)

    executor = PhysicalSO101CellExecutor(
        frame_source=frame_source,
        arm=arm,
        inspector=inspector,
        dashboard=dashboard,
        camui=camui,
        hud=hud,
        show_twin_viewer=show_twin_viewer,
        max_cycles=max_cycles,
    )
    agent = CogManInspectionAgent(
        executor=executor,
        ledger=executor.ledger,
        max_steps=max(24, max_cycles * 6 + 6),
    )

    try:
        report = agent.run(
            instruction="clear all blocks from the source plate",
            block_id="block_0",
            output_dir=out_root / "reports",
            mode="clear_source_plate",
        )
        last_reason = (
            str(report.executed_trace[-1].get("reason", "completed"))
            if report.executed_trace
            else "no_steps"
        )
        summary = {
            "mode": "physical_cell_dry_run" if dry_run else "physical_cell_live_execution",
            "synthetic_simulation": False,
            "succeeded": bool(report.succeeded),
            "termination_reason": last_reason,
            "elapsed_s": report.elapsed_s,
            "sorted_counts": dict(executor.sorted_counts),
            "batch_sorted_records": list(executor.batch_sorted_records),
            "hud_url": hud.url if hud is not None else None,
            "dashboard_url": dashboard_url,
            "report_json_path": str(out_root / "hardware_deployment_report.json"),
            "cogman_report_json": report.report_json_path,
            "cogman_report_html": report.report_html_path,
        }
        (out_root / "hardware_deployment_report.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

        # Optional standby loop keeping `:8095` HUD and `:8096` dashboard live mirror active
        if daemon_standby_s > 0.0:
            t_end = time.monotonic() + float(daemon_standby_s)
            while time.monotonic() < t_end:
                try:
                    q_now, g_now = arm.current_q_and_grip(fresh=False)
                    sc = frame_source.perceive_scene(executor.real2sim, q_arm_rad=q_now)
                    executor.twin.sync_real2sim_scene(sc, settle_steps=0)
                    if sc.bricks and dashboard is not None:
                        b0 = sc.bricks[0]
                        tbox = executor._project_to_top_box(b0.T_world_block, b0.spec.dims_m)
                        if tbox is not None:
                            dashboard.report_focus(
                                PART_ID_TO_DASHBOARD_ID.get(b0.spec.part_id, "part-001"),
                                f"{b0.spec.name} ({b0.support_region})",
                                tbox,
                            )
                    executor._sync_mirror_and_telemetry(q_now, g_now)
                except Exception:
                    pass
                time.sleep(0.5)

        return summary
    except BaseException:
        arm.emergency_hold()
        raise
    finally:
        executor.close()
        if hud is not None:
            hud.stop()
