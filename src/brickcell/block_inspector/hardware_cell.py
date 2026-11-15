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
from brickcell.inspection.dark_marks import score_brick_roi_dark_mark

# High upright observation posture clear of Top and Plate camera sightlines & strictly in CCW Joint-5 zone
PHYSICAL_OBSERVATION_HOME_Q = np.array([0.00, -1.15, 0.88, 0.42, 0.55], dtype=np.float64)
# Calibrated from physical leader-follower demonstrations (`duplo-grasp-scan-sort-v1`):
#   - Open / release width: 18..24% (~1607..1696 ticks) — compact & clear of neighbor bricks
#   - Clamped contact on brick: 7.2% (16 mm) to 10.3% (32 mm); empty closed floor: 4.3% (1403 ticks)
#   - Commanded hold squeeze: 3.5..4.0% (~1391..1398 ticks), never relaxed during spin or transit
PHYSICAL_HOME_GRIPPER_PCT = 0.22
PHYSICAL_WIDE_PREGRASP_PCT = 0.295
PHYSICAL_NARROW_PREGRASP_PCT = 0.215
PHYSICAL_CARRY_GRIPPER_PCT = 0.035
PHYSICAL_EMPTY_JAW_MAX_PCT = 0.052

# Physical Sonix USB wrist camera (`1920x1080` resized to `640x360`) mount on `Body3`, calibrated
# from 8 non-coplanar 3D-to-2D bench correspondences (left/right jaw tips + bricks + KALAS plates):
SONIX_T_BODY3_CAM = np.array(
    [
        [0.99504, 0.08539, -0.05109, 0.01396],
        [0.06380, -0.94148, -0.33097, 0.06730],
        [-0.07636, 0.32607, -0.94226, -0.00017],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
SONIX_FOCAL_640X360 = 228.164

# Physical Side/Plate-camera presentation target (+X forward, -Y right, +Z up) calibrated from
# leader-follower Episode 0 ([0.413, -0.100, 0.145], pitch 24d) & Episode 1 ([0.391, -0.086, 0.150], pitch 29d),
# placing the held brick dead-center in the Plate D415 (`243622060450`) FOV with a gentle 28-deg pitch.
# Joint 5 (`wrist_roll`) hits its physical CW stop at ~2282 ticks (+0.36 rad) and rotates
# freely in the Counter-Clockwise (+tick / +rad) direction [2341 .. 3989+ ticks] = [+0.45 .. +2.98 rad].
PHYSICAL_PLATE_SCAN_POS = np.array([0.402, -0.090, 0.146], dtype=np.float64)
PHYSICAL_PLATE_SCAN_PITCH_DEG = 28.0
PHYSICAL_JOINT5_CCW_START_RAD = +0.45
PHYSICAL_JOINT5_CCW_END_RAD = +2.98

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
        hdr_time = float("nan")
        hdr_frame_id = ""
        hdr_serial = ""
        try:
            with urllib.request.urlopen(self.wrist_http_url, timeout=self.timeout_s) as resp:
                if resp.status != 200:
                    raise StaleCameraFrameError(f"HTTP {resp.status} from {self.wrist_http_url}")
                raw = resp.read()
                hdrs = getattr(resp, "headers", None)
                if hdrs is not None:
                    try:
                        hdr_time = float(hdrs.get("X-Frame-Time", "nan"))
                    except (TypeError, ValueError):
                        hdr_time = float("nan")
                    hdr_frame_id = str(hdrs.get("X-Frame-Id", "") or "")
                    hdr_serial = str(hdrs.get("X-Camera-Serial", "") or "")
        except Exception as exc:
            raise StaleCameraFrameError(f"Failed to read wrist camera: {exc}") from exc

        cap_ts = hdr_time if math.isfinite(hdr_time) else t_now
        age_s = max(0.0, t_now - cap_ts)
        if math.isfinite(hdr_time) and age_s > self.max_frame_age_s:
            raise StaleCameraFrameError(
                f"Stale wrist camera snapshot: age_s={age_s:.2f} > {self.max_frame_age_s:.2f}"
            )

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        native_wh = img.size
        if resize_hw is not None:
            img = img.resize((resize_hw[1], resize_hw[0]), Image.Resampling.BILINEAR)
        rgb = np.asarray(img, dtype=np.uint8)
        return LiveCameraSnapshot(
            camera_name="wrist",
            rgb=rgb,
            depth_m=None,
            timestamp_s=cap_ts,
            age_s=age_s,
            frame_id=hdr_frame_id or f"wrist-{int(cap_ts * 1000)}",
            intrinsics={
                "serial": hdr_serial or "sonix-wrist",
                "has_capture_headers": bool(hdr_frame_id and math.isfinite(hdr_time)),
                "native_width": int(native_wh[0]),
                "native_height": int(native_wh[1]),
            },
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


def _extract_anomalib_hotspot_boxes(
    anomaly_map: np.ndarray | None,
    u0: int,
    v0: int,
    u1: int,
    v1: int,
    raw_score: float,
    threshold_val: float,
    label: str = "anomalib patchcore",
) -> list[dict[str, Any]]:
    """Map an Anomalib PatchCore `anomaly_map` to full-image pixel bounding boxes for `:8096` HUD."""
    w_crop = max(1, int(u1 - u0))
    h_crop = max(1, int(v1 - v0))
    if anomaly_map is None or np.asarray(anomaly_map).size < 4:
        return [
            {
                "box": [int(u0), int(v0), int(u1), int(v1)],
                "score": round(float(raw_score), 4),
                "label": label,
            }
        ]
    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.ndim > 2:
        amap = np.squeeze(amap)
    if amap.ndim != 2 or not np.all(np.isfinite(amap)):
        return [
            {
                "box": [int(u0), int(v0), int(u1), int(v1)],
                "score": round(float(raw_score), 4),
                "label": label,
            }
        ]
    amap_up = cv2.resize(amap, (w_crop, h_crop), interpolation=cv2.INTER_LINEAR)
    cutoff = max(float(threshold_val), float(np.percentile(amap_up, 85.0)))
    mask = (amap_up >= cutoff).astype(np.uint8)
    n_comp, lbls, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes: list[dict[str, Any]] = []
    for idx in range(1, n_comp):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 9:
            continue
        bx = int(stats[idx, cv2.CC_STAT_LEFT])
        by = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        peak = float(amap_up[lbls == idx].max())
        boxes.append(
            {
                "box": [u0 + bx, v0 + by, u0 + bx + bw, v0 + by + bh],
                "score": round(peak, 4),
                "label": label,
            }
        )
    if not boxes:
        boxes.append(
            {
                "box": [int(u0), int(v0), int(u1), int(v1)],
                "score": round(float(raw_score), 4),
                "label": label,
            }
        )
    return boxes


def load_anomalib_camera_detectors(
    openvino_model_path: str | Path | None = None,
    rgbd_viewer_config_path: str | Path | None = None,
    allow_synthetic: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[int] | None]]:
    """Load plate/wrist RGB models only; top and depth models are excluded.

    Supports:
      1. `configs/rgbd-viewer.json` (`cameras.{top,plate,wrist}.models.{rgb,depth}` + `crop`),
      2. Explicit `--openvino-model` file (`model.xml`) or directory of per-camera `model.xml`s,
      3. Automatic discovery of non-synthetic `artifacts/models/**/model.xml` assets on disk.
    """
    from brickcell.detector.runtime import OpenVINODetector

    serial_to_cam = {
        "350122060653": "top",
        "243622060450": "plate",
        "sonix-wrist": "wrist",
    }
    camera_detectors: dict[str, dict[str, Any]] = {}
    camera_crops: dict[str, list[int] | None] = {}

    def _register_xml(xml_file: Path, cam_hint: str | None = None, mod_hint: str = "rgb") -> None:
        xml_resolved = xml_file.resolve()
        meta_path = xml_resolved.with_suffix(".json")
        if not xml_resolved.exists() or not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not allow_synthetic and bool(meta.get("synthetic", True)):
            return
        contract = meta.get("sensor_contract") or {}
        cam_key = cam_hint or serial_to_cam.get(str(contract.get("serial", "")))
        if cam_key is None:
            low_parts = {p.lower() for p in xml_resolved.parts}
            for cand in ("top", "plate", "side", "wrist"):
                if cand in low_parts or any(cand in p for p in low_parts):
                    cam_key = "plate" if cand == "side" else cand
                    break
        cam_key = cam_key or "default"
        modality = str(contract.get("modality") or mod_hint or "rgb")
        if cam_key not in ("plate", "wrist") or modality != "rgb":
            return
        if (
            meta.get("synthetic") is not False
            or serial_to_cam.get(str(contract.get("serial", ""))) != cam_key
        ):
            raise ValueError("Mark model requires matching real plate/wrist RGB sensor contract")
        det = OpenVINODetector(xml_resolved)
        if det.threshold is None:
            raise ValueError(
                f"Anomalib model {xml_resolved} has no calibrated threshold in {meta_path.name}"
            )
        camera_detectors.setdefault(cam_key, {})[modality] = det
        if isinstance(contract.get("crop"), list) and len(contract["crop"]) == 4:
            camera_crops.setdefault(cam_key, [int(v) for v in contract["crop"]])

    if rgbd_viewer_config_path is not None:
        cfg_file = Path(rgbd_viewer_config_path)
        if cfg_file.exists():
            cfg_doc = json.loads(cfg_file.read_text(encoding="utf-8"))
            for cam_name, settings in (cfg_doc.get("cameras") or {}).items():
                norm_cam = "plate" if cam_name == "side" else str(cam_name)
                if isinstance(settings.get("crop"), list) and len(settings["crop"]) == 4:
                    camera_crops[norm_cam] = [int(v) for v in settings["crop"]]
                for modality, rel_model in (settings.get("models") or {}).items():
                    model_xml = (cfg_file.parent / str(rel_model)).resolve()
                    _register_xml(model_xml, cam_hint=norm_cam, mod_hint=str(modality))

    if openvino_model_path is not None:
        ov_p = Path(openvino_model_path)
        if ov_p.is_file() and ov_p.suffix.lower() == ".xml":
            _register_xml(ov_p)
        elif ov_p.is_file() and ov_p.suffix.lower() == ".json" and ov_p.name != "model.json":
            # Config JSON like rgbd-viewer.json
            extra_det, extra_crops = load_anomalib_camera_detectors(
                rgbd_viewer_config_path=ov_p, allow_synthetic=allow_synthetic
            )
            for ck, mods in extra_det.items():
                camera_detectors.setdefault(ck, {}).update(mods)
            camera_crops.update(extra_crops)
        elif ov_p.is_dir():
            for xml_f in sorted(ov_p.rglob("*.xml")):
                _register_xml(xml_f)

    if not camera_detectors:
        models_dir = REPO_ROOT / "artifacts" / "models"
        if models_dir.exists():
            for xml_f in sorted(models_dir.rglob("model.xml")):
                try:
                    _register_xml(xml_f)
                except Exception:
                    continue

    return camera_detectors, camera_crops


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
    inspector_backend: str = "lab_dark_mark_fallback"


class RealCameraAnomalyInspector:
    """`Inspector` boundary wrapping the same Anomalib architecture for plate/wrist RGB.

    Strictly isolated from robot control (zero robot handles per `AGENTS.md`).
    When Anomalib `OpenVINODetector` model(s) are loaded (`has_anomalib_models == True`),
    Anomalib PatchCore replaces the generic CIE-Lab dark-mark fallback completely!
    """

    def __init__(
        self,
        dark_mark_threshold: float | None = 16.0,
        openvino_detector: Any | None = None,
        camera_detectors: dict[str, dict[str, Any]] | None = None,
        camera_crops: dict[str, list[int] | None] | None = None,
        worker_timeout_s: float = 2.5,
        allow_heuristic_sorting: bool = False,
    ) -> None:
        if (
            dark_mark_threshold is None
            or not math.isfinite(float(dark_mark_threshold))
            or float(dark_mark_threshold) <= 0
        ):
            raise ValueError(
                "RealCameraAnomalyInspector requires a positive finite dark_mark_threshold"
            )
        if type(allow_heuristic_sorting) is not bool:
            raise ValueError("Heuristic sorting selection must be explicit boolean")
        if allow_heuristic_sorting and (openvino_detector is not None or camera_detectors):
            raise ValueError("Do not mix heuristic sorting with Anomalib models")
        self.allow_heuristic_sorting = allow_heuristic_sorting
        if openvino_detector is not None and getattr(openvino_detector, "threshold", None) is None:
            raise ValueError("OpenVINODetector requires a calibrated threshold before deployment")
        self.dark_mark_threshold = float(dark_mark_threshold)
        self.openvino_detector = openvino_detector
        self.camera_detectors: dict[str, dict[str, Any]] = {}
        if camera_detectors:
            for cam_k, mod_map in camera_detectors.items():
                cam_k = "plate" if cam_k == "side" else str(cam_k)
                if cam_k not in ("plate", "wrist"):
                    continue
                for mod_k, det in mod_map.items():
                    if mod_k != "rgb":
                        raise ValueError("Plate/wrist mark inspection accepts RGB models only")
                    if getattr(det, "threshold", None) is None:
                        raise ValueError(
                            f"Anomalib detector for {cam_k}/{mod_k} lacks a calibrated threshold"
                        )
                    self.camera_detectors.setdefault(str(cam_k), {})[str(mod_k)] = det
        if openvino_detector is not None and "plate" not in self.camera_detectors:
            self.camera_detectors["plate"] = {"rgb": openvino_detector}
        self.camera_crops: dict[str, list[int] | None] = dict(camera_crops or {})
        self.worker_timeout_s = float(worker_timeout_s)
        self._executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspector-wkr")
        )
        self.timed_out: bool = False

    @property
    def has_anomalib_models(self) -> bool:
        return bool(self.camera_detectors)

    @property
    def inspector_mode(self) -> str:
        if self.allow_heuristic_sorting:
            return "lab_dark_mark_plate_sort"
        return "anomalib_openvino_cpu" if self.has_anomalib_models else "lab_dark_mark_fallback"

    def configured_cameras(self) -> list[str]:
        return sorted(self.camera_detectors.keys())

    def has_detector_for_camera(self, camera_name: str) -> bool:
        norm = "plate" if camera_name == "side" else camera_name
        if self.allow_heuristic_sorting:
            return norm == "plate"
        return norm in ("plate", "wrist") and norm in self.camera_detectors

    def resolve_crop_for_camera(
        self, camera_name: str, fallback_box: list[int] | None
    ) -> list[int] | None:
        norm = "plate" if camera_name == "side" else camera_name
        cfg_crop = self.camera_crops.get(norm)
        if cfg_crop is not None and len(cfg_crop) == 4:
            return [int(v) for v in cfg_crop]
        return fallback_box

    def _detectors_for_camera(self, camera_name: str) -> dict[str, Any]:
        norm = "plate" if camera_name == "side" else camera_name
        if norm in self.camera_detectors:
            return self.camera_detectors[norm]
        return {}

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
        camera_name: str,
    ) -> FrameInspectionResult:
        from brickcell.inspection.block_surface import SurfaceViewError
        from brickcell.inspection.rgbd import encode_depth
        from brickcell.inspection.types import Decision

        h_img, w_img = rgb.shape[:2]
        if (
            len(crop_box) != 4
            or any(type(v) is not int for v in crop_box)
            or not (
                0 <= crop_box[0] < crop_box[2] <= w_img and 0 <= crop_box[1] < crop_box[3] <= h_img
            )
        ):
            return FrameInspectionResult(
                False,
                0.0,
                self.dark_mark_threshold,
                False,
                [],
                0,
                reason="invalid_or_clipped_crop",
                inspector_backend=self.inspector_mode,
            )
        u0 = max(0, min(w_img - 1, int(crop_box[0])))
        v0 = max(0, min(h_img - 1, int(crop_box[1])))
        u1 = max(u0 + 1, min(w_img, int(crop_box[2])))
        v1 = max(v0 + 1, min(h_img, int(crop_box[3])))
        roi_rgb = rgb[v0:v1, u0:u1]
        if roi_rgb.size == 0 or min(roi_rgb.shape[:2]) < 16 or float(roi_rgb.std()) <= 4.0:
            return FrameInspectionResult(
                valid_view=False,
                raw_anomaly_distance=0.0,
                threshold_value=self.dark_mark_threshold,
                flagged=False,
                detections=[],
                foreground_pixels=0,
                reason="empty_or_low_contrast_roi",
                inspector_backend=self.inspector_mode,
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
                    inspector_backend=self.inspector_mode,
                )

        cam_dets = self._detectors_for_camera(camera_name)
        if cam_dets:
            # Primary path: wrap trained Anomalib PatchCore OpenVINODetector(s) (`rgb` + optional `depth`),
            # replacing the generic Lab heuristic completely!
            flagged = False
            all_valid = True
            best_ratio = -1.0
            best_raw = 0.0
            best_thr = 1.0
            full_dets: list[dict[str, Any]] = []
            reasons: list[str] = []

            rgb_det = cam_dets.get("rgb")
            if rgb_det is not None:
                try:
                    pred_rgb = rgb_det.predict(roi_rgb)
                except SurfaceViewError as exc:
                    return FrameInspectionResult(
                        valid_view=False,
                        raw_anomaly_distance=0.0,
                        threshold_value=float(rgb_det.threshold.value),
                        flagged=False,
                        detections=[],
                        foreground_pixels=fg_count,
                        reason=f"surface_reinspect:{exc}",
                        inspector_backend=self.inspector_mode,
                    )
                sc_rgb = float(pred_rgb.score)
                thr_rgb = float(rgb_det.threshold.value)
                if not math.isfinite(sc_rgb):
                    raise ValueError(f"Non-finite Anomalib RGB raw distance on {camera_name}")
                dec_rgb = rgb_det.threshold.decide(sc_rgb)
                if dec_rgb == Decision.ERROR:
                    raise ValueError(f"Anomalib RGB Decision.ERROR on {camera_name}")
                if dec_rgb == Decision.REINSPECT:
                    all_valid = False
                    reasons.append(f"{camera_name}_rgb_reinspect")
                elif dec_rgb == Decision.DEFECTIVE:
                    flagged = True
                    full_dets.extend(
                        _extract_anomalib_hotspot_boxes(
                            pred_rgb.anomaly_map,
                            u0,
                            v0,
                            u1,
                            v1,
                            sc_rgb,
                            thr_rgb,
                            label=f"anomalib {camera_name} rgb",
                        )
                    )
                ratio = sc_rgb / max(thr_rgb, 1e-6)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_raw = sc_rgb
                    best_thr = thr_rgb

            dep_det = cam_dets.get("depth")
            if dep_det is not None and depth_m is not None:
                roi_dep_m = depth_m[v0:v1, u0:u1]
                roi_dep_u16 = np.where(
                    np.isfinite(roi_dep_m) & (roi_dep_m > 0.0),
                    np.clip(np.round(roi_dep_m / 0.001), 0, 65535),
                    0,
                ).astype(np.uint16)
                enc_dep, valid_dep = encode_depth(roi_dep_u16, 0.001)
                if float(valid_dep.mean()) < 0.45:
                    all_valid = False
                    reasons.append(f"{camera_name}_insufficient_encoded_depth")
                else:
                    pred_dep = dep_det.predict(enc_dep)
                    sc_dep = float(pred_dep.score)
                    thr_dep = float(dep_det.threshold.value)
                    if not math.isfinite(sc_dep):
                        raise ValueError(f"Non-finite Anomalib depth raw distance on {camera_name}")
                    dec_dep = dep_det.threshold.decide(sc_dep)
                    if dec_dep == Decision.ERROR:
                        raise ValueError(f"Anomalib depth Decision.ERROR on {camera_name}")
                    if dec_dep == Decision.REINSPECT:
                        all_valid = False
                        reasons.append(f"{camera_name}_depth_reinspect")
                    elif dec_dep == Decision.DEFECTIVE:
                        flagged = True
                        full_dets.extend(
                            _extract_anomalib_hotspot_boxes(
                                pred_dep.anomaly_map,
                                u0,
                                v0,
                                u1,
                                v1,
                                sc_dep,
                                thr_dep,
                                label=f"anomalib {camera_name} depth",
                            )
                        )
                    ratio = sc_dep / max(thr_dep, 1e-6)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_raw = sc_dep
                        best_thr = thr_dep

            return FrameInspectionResult(
                valid_view=flagged or all_valid,
                raw_anomaly_distance=best_raw,
                threshold_value=best_thr,
                flagged=flagged,
                detections=full_dets,
                foreground_pixels=fg_count,
                reason="ok" if (flagged or all_valid) else ";".join(reasons) or "reinspect",
                inspector_backend="anomalib_openvino_cpu",
            )

        if self.has_anomalib_models:
            # Another camera has a real Anomalib model, so never mix in generic Lab scores here
            return FrameInspectionResult(
                valid_view=False,
                raw_anomaly_distance=0.0,
                threshold_value=self.dark_mark_threshold,
                flagged=False,
                detections=[],
                foreground_pixels=fg_count,
                reason=f"no_anomalib_model_for_{camera_name}",
                inspector_backend="anomalib_openvino_cpu",
            )

        dark_score, local_dets = score_brick_roi_dark_mark(roi_rgb, fg_mask=fg_mask)
        full_dets = []
        for d in local_dets:
            bx0, by0, bx1, by1 = d["box"]
            full_dets.append(
                {
                    "box": [u0 + bx0, v0 + by0, u0 + bx1, v0 + by1],
                    "score": d["score"],
                    "label": d["label"],
                }
            )

        return FrameInspectionResult(
            valid_view=self.allow_heuristic_sorting,
            raw_anomaly_distance=dark_score,
            threshold_value=self.dark_mark_threshold,
            flagged=self.allow_heuristic_sorting and dark_score >= self.dark_mark_threshold,
            detections=full_dets,
            foreground_pixels=fg_count,
            reason=(
                "explicit_lab_contrast_rule"
                if self.allow_heuristic_sorting
                else "uncalibrated_dark_mark_reference_not_trusted"
            ),
            inspector_backend=self.inspector_mode,
        )

    def evaluate_frame(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray | None,
        crop_box: list[int],
        expected_z_m: float | None = None,
        z_band_m: float = 0.065,
        camera_name: str = "plate",
    ) -> FrameInspectionResult:
        """Run bounded-timeout inspection; retire worker permanently on timeout (`AGENTS.md`)."""
        if self.timed_out or self._executor is None:
            raise TimeoutError("Inspector worker previously timed out and cannot be reused")
        camera_name = "plate" if camera_name == "side" else camera_name
        if camera_name not in ("plate", "wrist") or (
            self.allow_heuristic_sorting and camera_name != "plate"
        ):
            return FrameInspectionResult(
                False,
                0.0,
                self.dark_mark_threshold,
                False,
                [],
                0,
                reason="camera_excluded_from_mark_inspection",
                inspector_backend=self.inspector_mode,
            )
        # RGB mark detection must not use depth gating or depth features.
        depth_m, expected_z_m = None, None
        fut = self._executor.submit(
            self._score_sync, rgb, depth_m, crop_box, expected_z_m, z_band_m, camera_name
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


def _ray_intersect_z_plane(
    u: float,
    v: float,
    R_wc: np.ndarray,
    p_wc: np.ndarray,
    K: np.ndarray,
    z_world: float,
) -> np.ndarray | None:
    """Intersect a 2D pixel `(u, v)` ray from camera `(R_wc, p_wc, K)` with horizontal plane `z = z_world`."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    d_cam = np.array([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=np.float64)
    d_world = R_wc @ d_cam
    if abs(float(d_world[2])) < 1e-6:
        return None
    lam = (float(z_world) - float(p_wc[2])) / float(d_world[2])
    if lam <= 0.02 or lam > 2.50:
        return None
    return p_wc + lam * d_world


def _infer_duplo_spec_from_2d_extents(short_m: float, long_m: float) -> Any:
    """Infer closest `DUPLO_CATALOG` spec from 2D planar rectangle extents (zero depth required)."""
    s = max(0.010, min(float(short_m), float(long_m)))
    L = max(float(short_m), float(long_m))
    if s <= 0.0235 and L <= 0.042:
        return DUPLO_CATALOG["duplo_1x2x2_76371"]
    if L >= 0.055:
        return DUPLO_CATALOG["duplo_2x4_3011"]
    if L >= 0.041:
        return DUPLO_CATALOG["duplo_2x3_curved_top_2302"]
    return DUPLO_CATALOG["duplo_2x2_3437"]


def localize_plates_and_bricks_top_rgb(
    top_rgb: np.ndarray,
    R_top: np.ndarray,
    p_top: np.ndarray,
    K_top: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[PerceivedBrick]]:
    """Depth-free 2D RGB localization of all 3 KALAS plates and all DUPLO bricks on the source plate.

    Uses crisp 2D RGB contours + flat tabletop ray-plane homography (`z = 0.016 m`) relative to the
    detected aquamarine KALAS source plate, eliminating D415 stereo depth shadows and Z noise.
    """
    from scipy.spatial.transform import Rotation

    h_img, w_img = top_rgb.shape[:2]
    # Calibrated physical plate priors from user's leader-follower demonstrations
    plate_xy_map: dict[str, np.ndarray] = {
        "source": np.array([0.255, 0.008, 0.0], dtype=np.float64),
        "good": np.array([0.395, 0.116, 0.0], dtype=np.float64),
        "defective": np.array([0.229, 0.136, 0.0], dtype=np.float64),
    }
    if top_rgb.size == 0 or float(top_rgb.std()) < 6.0:
        return plate_xy_map, []

    hsv = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 1. Detect the 3 colored KALAS plates in 2D RGB (`source` aquamarine, `good` green, `defective` pink)
    plate_hsv_masks = {
        "source": (hsv[:, :, 0] >= 88)
        & (hsv[:, :, 0] <= 115)
        & (hsv[:, :, 1] >= 80)
        & (hsv[:, :, 2] >= 50),
        "good": (hsv[:, :, 0] >= 36)
        & (hsv[:, :, 0] <= 82)
        & (hsv[:, :, 1] >= 65)
        & (hsv[:, :, 2] >= 55),
        "defective": (
            ((hsv[:, :, 0] >= 145) | (hsv[:, :, 0] <= 10))
            & (hsv[:, :, 1] >= 55)
            & (hsv[:, :, 2] >= 80)
        ),
    }
    src_hull_mask: np.ndarray | None = None
    src_plate_uv: tuple[float, float] | None = None

    for role, pmask in plate_hsv_masks.items():
        clean = cv2.morphologyEx(pmask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_c = None
        best_area = 0.0
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < 1400.0:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 1e-6:
                continue
            cu = float(M["m10"] / M["m00"])
            cv_y = float(M["m01"] / M["m00"])
            pw = _ray_intersect_z_plane(cu, cv_y, R_top, p_top, K_top, z_world=0.008)
            if pw is None or not (0.06 <= pw[0] <= 0.46 and -0.28 <= pw[1] <= 0.28):
                continue
            if area > best_area:
                best_area = area
                best_c = (c, cu, cv_y, pw)
        if best_c is not None:
            c_poly, cu, cv_y, pw = best_c
            # Track live movable plate XY directly from 2D RGB (priors only used if off-camera!)
            plate_xy_map[role] = np.array([float(pw[0]), float(pw[1]), 0.0], dtype=np.float64)
            if role == "source":
                hull = cv2.convexHull(c_poly)
                fill = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.drawContours(fill, [hull], -1, 1, thickness=-1)
                # Shrink inward by 8 px to exclude sloped plate outer rim
                src_hull_mask = cv2.erode(fill, np.ones((15, 15), np.uint8)) > 0
                src_plate_uv = (cu, cv_y)

    if src_hull_mask is None or int(np.sum(src_hull_mask)) < 800:
        return plate_xy_map, []

    # 2. Segment ALL DUPLO bricks inside the aquamarine plate interior in crisp 2D RGB (zero depth!)
    aqua_pixels = src_hull_mask & plate_hsv_masks["source"]
    if int(np.sum(aqua_pixels)) >= 200:
        med_lab = np.median(lab[aqua_pixels], axis=0)
    else:
        med_lab = np.array([145.0, 98.0, 108.0], dtype=np.float32)

    # Chromatic + specular-immune foreground: exclude aquamarine plate pixels (across both lit and
    # shaded bowl regions), ultra-bright white ceiling glare streaks, and any purple arm link
    # connected to the exterior of the plate (while preserving purple/lavender DUPLO bricks inside!).
    is_aqua_broad = (
        (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 114) & (hsv[:, :, 1] >= 68) & (hsv[:, :, 2] >= 42)
    )
    is_specular_glare = (hsv[:, :, 2] >= 242) & (hsv[:, :, 1] <= 38)
    purple_raw = (hsv[:, :, 0] >= 115) & (hsv[:, :, 0] <= 146) & (hsv[:, :, 1] >= 75)
    n_p, p_lbl = cv2.connectedComponents(purple_raw.astype(np.uint8))
    arm_purple = np.zeros_like(purple_raw)
    outside_plate = ~src_hull_mask
    for pl in range(1, n_p):
        comp = p_lbl == pl
        if np.any(comp & outside_plate):
            arm_purple |= comp

    delta_ab = np.linalg.norm(lab[:, :, 1:] - med_lab[None, None, 1:], axis=2)
    delta_e = np.linalg.norm(lab - med_lab[None, None, :], axis=2)
    fg_raw = (
        src_hull_mask
        & (~arm_purple)
        & (~is_aqua_broad)
        & (~is_specular_glare)
        & ((delta_ab >= 11.5) | (delta_e >= 18.0))
    )
    fg_u8 = cv2.morphologyEx(fg_raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Split touching bricks cleanly via distance-transform watershed when multiple peaks exist
    dist_map = cv2.distanceTransform(fg_u8, cv2.DIST_L2, 5)
    sure_fg = (dist_map >= 6.5).astype(np.uint8)
    n_mk, markers = cv2.connectedComponents(sure_fg)
    if n_mk > 2:
        sure_bg = cv2.dilate(fg_u8, np.ones((5, 5), np.uint8), iterations=1)
        unknown = (sure_bg > 0) & (sure_fg == 0)
        ws_markers = markers + 1
        ws_markers[unknown] = 0
        ws_in = cv2.cvtColor(fg_u8 * 255, cv2.COLOR_GRAY2BGR)
        cv2.watershed(ws_in, ws_markers)
        comp_labels = list(range(2, n_mk + 1))
        label_img = ws_markers
    else:
        n_cc, label_img = cv2.connectedComponents(fg_u8)
        comp_labels = list(range(1, n_cc))

    # Anchor differential ray-plane homography to the calibrated source plate center
    p_src_anchor = plate_xy_map["source"]
    p_src_raw = (
        _ray_intersect_z_plane(src_plate_uv[0], src_plate_uv[1], R_top, p_top, K_top, z_world=0.016)
        if src_plate_uv is not None
        else None
    )
    xy_shift = (
        (p_src_anchor[:2] - p_src_raw[:2])
        if p_src_raw is not None
        else np.zeros(2, dtype=np.float64)
    )

    bricks: list[PerceivedBrick] = []
    for lbl in comp_labels:
        mask_b = (label_img == lbl) & (fg_u8 > 0)
        px_count = int(np.sum(mask_b))
        if px_count < 85 or px_count > 6500:
            continue
        ys, xs = np.where(mask_b)
        pts_uv = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        rect = cv2.minAreaRect(pts_uv)
        (cu, cv_y), (rw, rh), _ = rect
        if min(rw, rh) < 7.0:
            continue

        p_center = _ray_intersect_z_plane(cu, cv_y, R_top, p_top, K_top, z_world=0.016)
        if p_center is None:
            continue
        p_xy = p_center[:2] + xy_shift
        if float(np.linalg.norm(p_xy - p_src_anchor[:2])) > 0.086:
            continue

        box_uv = cv2.boxPoints(rect)
        corners_w: list[np.ndarray] = []
        for bu, bv in box_uv:
            cw = _ray_intersect_z_plane(float(bu), float(bv), R_top, p_top, K_top, z_world=0.016)
            if cw is not None:
                corners_w.append(cw[:2] + xy_shift)
        if len(corners_w) != 4:
            continue

        e01 = corners_w[1] - corners_w[0]
        e12 = corners_w[2] - corners_w[1]
        len01 = float(np.linalg.norm(e01))
        len12 = float(np.linalg.norm(e12))
        if len01 >= len12:
            long_vec, long_m, short_m = e01, len01, len12
        else:
            long_vec, long_m, short_m = e12, len12, len01

        spec = _infer_duplo_spec_from_2d_extents(short_m, long_m)
        # In DUPLO_CATALOG, dims_m[0] <= dims_m[1] (X_block = short axis, Y_block = long axis)
        yaw_long = math.atan2(float(long_vec[1]), float(long_vec[0]))
        yaw_block = yaw_long - 0.5 * math.pi

        hz = float(spec.dims_m[2])
        z_center = 0.006 + 0.5 * hz
        T_wb = np.eye(4, dtype=np.float64)
        T_wb[:3, :3] = Rotation.from_euler("z", yaw_block).as_matrix()
        T_wb[:3, 3] = [float(p_xy[0]), float(p_xy[1]), z_center]
        T_mesh = T_wb.copy()
        T_mesh[2, 3] = 0.006

        mean_rgb_tuple = tuple(float(v) for v in np.mean(top_rgb[mask_b], axis=0))
        bricks.append(
            PerceivedBrick(
                block_id=f"rgb_block_{len(bricks)}",
                spec=spec,
                T_world_block=T_wb,
                T_world_mesh_origin=T_mesh,
                stable_pose_name="rgb_top_planar",
                xy_yaw=(float(p_xy[0]), float(p_xy[1]), float(yaw_block)),
                support_region="source",
                icp_fitness=0.98,
                icp_rmse_m=0.001,
                confidence=0.96,
                has_visual_defect=False,
                num_points=px_count,
                mean_rgb=mean_rgb_tuple,
                cluster_pts_world=np.tile(T_wb[:3, 3], (16, 1)),
            )
        )

    return plate_xy_map, bricks


@dataclass(frozen=True)
class JawCorridorObservation:
    """Live inner-surface pixel geometry of the two physical purple gripper jaws in Wrist RGB."""

    u_left_pad: float
    u_right_pad: float
    u_pinch: float
    gap_px: float
    v_tip: float
    jaw_axis_angle_rad: float
    jaws_dilated_mask: np.ndarray


def detect_wrist_jaw_corridor(wrist_rgb: np.ndarray) -> JawCorridorObservation | None:
    """Track the inner pinch surfaces of both purple SO-101 gripper jaws in a wrist RGB frame.

    Selects only royal-purple connected components anchored to the bottom border of the image
    (`y0 + bh >= H - 18`) so lavender/purple DUPLO bricks inside a plate are never mistaken for jaws.
    """
    if wrist_rgb.size == 0:
        return None
    h_w, w_w = wrist_rgb.shape[:2]
    hsv = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2HSV)
    purple_mask = (
        (hsv[:, :, 0] >= 113) & (hsv[:, :, 0] <= 146) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 28)
    )
    k_close = max(3, int(round(7.0 * w_w / 640.0)) | 1)
    k_open = max(3, int(round(5.0 * w_w / 640.0)) | 1)
    purple_u8 = cv2.morphologyEx(
        purple_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8)
    )
    purple_u8 = cv2.morphologyEx(purple_u8, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    n_cc, cc_lbl, stats, centroids = cv2.connectedComponentsWithStats(purple_u8)

    min_jaw_area = int(450 * (w_w * h_w) / (640.0 * 360.0))
    bottom_margin = max(12, int(round(20.0 * h_w / 360.0)))
    left_lbl: int | None = None
    right_lbl: int | None = None
    left_area = 0
    right_area = 0

    for lbl in range(1, n_cc):
        y0 = int(stats[lbl, cv2.CC_STAT_TOP])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if (y0 + bh) < (h_w - bottom_margin) or area < min_jaw_area:
            continue
        cx = float(centroids[lbl][0])
        if cx < 0.54 * w_w and area > left_area:
            left_area = area
            left_lbl = lbl
        elif cx >= 0.46 * w_w and area > right_area:
            right_area = area
            right_lbl = lbl

    if left_lbl is None or right_lbl is None or left_lbl == right_lbl:
        return None

    left_jaw = cc_lbl == left_lbl
    right_jaw = cc_lbl == right_lbl
    v_l_tip = int(np.min(np.where(left_jaw)[0]))
    v_r_tip = int(np.min(np.where(right_jaw)[0]))
    v_start = max(v_l_tip, v_r_tip) + max(2, int(round(3.0 * h_w / 360.0)))
    v_end = min(h_w - 6, max(v_l_tip, v_r_tip) + max(18, int(round(36.0 * h_w / 360.0))))

    inner_pairs: list[tuple[float, float]] = []
    for v in range(v_start, v_end):
        xl = np.where(left_jaw[v])[0]
        xr = np.where(right_jaw[v])[0]
        if len(xl) > 0 and len(xr) > 0:
            ul = float(np.max(xl))
            ur = float(np.min(xr))
            if ur > ul + 8.0:
                inner_pairs.append((ul, ur))

    if len(inner_pairs) < 5:
        return None

    arr = np.asarray(inner_pairs, dtype=np.float64)
    top_slice = arr[: min(14, len(arr))]
    u_l_pad = float(np.median(top_slice[:, 0]))
    u_r_pad = float(np.median(top_slice[:, 1]))
    u_pinch = 0.5 * (u_l_pad + u_r_pad)
    gap_px = float(u_r_pad - u_l_pad)
    v_tip = 0.5 * float(v_l_tip + v_r_tip)

    row_l = min(h_w - 1, v_l_tip + 4)
    row_r = min(h_w - 1, v_r_tip + 4)
    xl_t = np.where(left_jaw[row_l])[0]
    xr_t = np.where(right_jaw[row_r])[0]
    if len(xl_t) > 0 and len(xr_t) > 0:
        jaw_angle = math.atan2(
            float(v_r_tip - v_l_tip),
            max(8.0, float(np.min(xr_t)) - float(np.max(xl_t))),
        )
    else:
        jaw_angle = 0.0

    k_dil = max(5, int(round(9.0 * w_w / 640.0)) | 1)
    jaws_dil = (
        cv2.dilate((left_jaw | right_jaw).astype(np.uint8), np.ones((k_dil, k_dil), np.uint8)) > 0
    )
    return JawCorridorObservation(
        u_left_pad=u_l_pad,
        u_right_pad=u_r_pad,
        u_pinch=u_pinch,
        gap_px=gap_px,
        v_tip=v_tip,
        jaw_axis_angle_rad=jaw_angle,
        jaws_dilated_mask=jaws_dil,
    )


def _sonix_cam_pose_and_intrinsics(
    T_wb3: np.ndarray, w_w: int, h_w: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(R_wc, p_wc, K_scaled)` for the calibrated physical Sonix USB wrist camera."""
    T_wc = T_wb3 @ SONIX_T_BODY3_CAM
    sx = float(w_w) / 640.0
    sy = float(h_w) / 360.0
    K = np.array(
        [
            [SONIX_FOCAL_640X360 * sx, 0.0, 0.5 * float(w_w)],
            [0.0, SONIX_FOCAL_640X360 * sy, 0.5 * float(h_w)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return T_wc[:3, :3], T_wc[:3, 3], K


def servo_grasp_with_jaw_corridor(
    wrist_rgb: np.ndarray,
    q_pose: np.ndarray,
    p_tcp_nominal: np.ndarray,
    jaw_yaw_nominal: float,
    ik: SO101FraxIK,
    wrist_cam: Any | None = None,
    max_step_m: float = 0.032,
    gain: float = 0.90,
) -> tuple[np.ndarray, float, list[int] | None, bool, dict[str, Any]]:
    """Closed-loop Inner-Jaw Corridor Visual Servoing (`IBVS + calibrated Sonix ray-plane`).

    Tracks the inner purple surfaces `[u_left_pad, u_right_pad]` of the physical gripper jaws AND
    segments the target DUPLO brick cleanly (immune to glossy KALAS plate reflections), driving the
    jaw corridor directly around the brick's graspable short width (`32 mm` or `16 mm`).
    Returns `(p_tcp_refined, jaw_yaw_refined, wrist_crop_box, straddle_ready, telemetry)`.
    """
    h_w, w_w = wrist_rgb.shape[:2]
    if wrist_rgb.size == 0 or float(wrist_rgb.std()) < 5.0:
        return p_tcp_nominal.copy(), float(jaw_yaw_nominal), None, False, {"status": "blank"}

    jaw_obs = detect_wrist_jaw_corridor(wrist_rgb)
    T_wb3 = ik.fk_world_body3(q_pose)
    T_tcp_cur = ik.fk_world_tcp(q_pose)

    if jaw_obs is not None or wrist_cam is None:
        R_wc, p_wc, K_scaled = _sonix_cam_pose_and_intrinsics(T_wb3, w_w, h_w)
    else:
        # Synthetic unit-test fallback when simulated wrist frames lack physical purple jaws
        T_wc = wrist_cam.world_transform(T_wb3)
        R_wc, p_wc = T_wc[:3, :3], T_wc[:3, 3]
        sx = float(w_w) / float(wrist_cam.width)
        sy = float(h_w) / float(wrist_cam.height)
        K_scaled = wrist_cam.K.copy()
        K_scaled[0, :] *= sx
        K_scaled[1, :] *= sy

    # Project point on brick mid-plane (`z = 0.018 m`) directly beneath the live TCP:
    p_under_tcp = np.array(
        [float(T_tcp_cur[0, 3]), float(T_tcp_cur[1, 3]), 0.018], dtype=np.float64
    )
    pc_u = R_wc.T @ (p_under_tcp - p_wc)
    if float(pc_u[2]) > 0.02:
        u_proj = float(K_scaled[0, 0] * (pc_u[0] / pc_u[2]) + K_scaled[0, 2])
        v_proj = float(K_scaled[1, 1] * (pc_u[1] / pc_u[2]) + K_scaled[1, 2])
    else:
        u_proj, v_proj = 0.50 * w_w, 0.60 * h_w

    # Anchor horizontal pinch target directly to the LIVE detected inner surfaces of the two jaws!
    if jaw_obs is not None:
        u_tgt = float(jaw_obs.u_pinch)
        v_tgt = float(np.clip(v_proj, 0.22 * h_w, min(0.88 * h_w, jaw_obs.v_tip + 6.0)))
        jaws_excl = jaw_obs.jaws_dilated_mask
    else:
        u_tgt = float(np.clip(u_proj, 0.20 * w_w, 0.80 * w_w))
        v_tgt = float(np.clip(v_proj, 0.20 * h_w, 0.84 * h_w))
        hsv_tmp = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2HSV)
        jaws_excl = (hsv_tmp[:, :, 0] >= 118) & (hsv_tmp[:, :, 0] <= 148) & (hsv_tmp[:, :, 1] >= 75)

    # Specular-glare-immune chromatic brick segmentation inside the visible aquamarine plate
    hsv = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    is_aqua = (
        (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 114) & (hsv[:, :, 1] >= 68) & (hsv[:, :, 2] >= 40)
    )
    is_glare = (hsv[:, :, 2] >= 240) & (hsv[:, :, 1] <= 40)

    aqua_u8 = cv2.morphologyEx(is_aqua.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts_p, _ = cv2.findContours(aqua_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_p and max(cv2.contourArea(c) for c in cnts_p) >= 600.0:
        plate_cnt = max(cnts_p, key=cv2.contourArea)
        hull = cv2.convexHull(plate_cnt)
        plate_fill = np.zeros((h_w, w_w), dtype=np.uint8)
        cv2.drawContours(plate_fill, [hull], -1, 1, -1)
        erode_px = max(7, int(round(13.0 * w_w / 640.0)) | 1)
        search_roi = (cv2.erode(plate_fill, np.ones((erode_px, erode_px), np.uint8)) > 0) & (
            ~jaws_excl
        )
    else:
        search_roi = np.zeros((h_w, w_w), dtype=bool)
        search_roi[int(0.08 * h_w) : int(0.90 * h_w), int(0.14 * w_w) : int(0.86 * w_w)] = True
        search_roi &= ~jaws_excl

    aqua_ref = search_roi & is_aqua
    if int(np.sum(aqua_ref)) < 180:
        cbox = [int(0.25 * w_w), int(0.18 * h_w), int(0.75 * w_w), int(0.82 * h_w)]
        return p_tcp_nominal.copy(), float(jaw_yaw_nominal), cbox, False, {"status": "no_aqua_bg"}

    med_aqua = np.median(lab[aqua_ref], axis=0)
    delta_ab = np.linalg.norm(lab[:, :, 1:] - med_aqua[None, None, 1:], axis=2)
    delta_e = np.linalg.norm(lab - med_aqua[None, None, :], axis=2)
    fg = search_roi & (~is_aqua) & (~is_glare) & ((delta_ab >= 11.5) | (delta_e >= 18.0))
    fg_u8 = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # Split touching bricks via distance-transform watershed if needed
    dist_map = cv2.distanceTransform(fg_u8, cv2.DIST_L2, 5)
    sure_fg = (dist_map >= 8.0).astype(np.uint8)
    n_mk, markers = cv2.connectedComponents(sure_fg)
    cand_contours: list[np.ndarray] = []
    if n_mk > 2:
        sure_bg = cv2.dilate(fg_u8, np.ones((5, 5), np.uint8), iterations=1)
        unknown = (sure_bg > 0) & (sure_fg == 0)
        ws_m = markers + 1
        ws_m[unknown] = 0
        cv2.watershed(cv2.cvtColor(fg_u8 * 255, cv2.COLOR_GRAY2BGR), ws_m)
        for lbl in range(2, n_mk + 1):
            sub_u8 = ((ws_m == lbl) & (fg_u8 > 0)).astype(np.uint8)
            sc, _ = cv2.findContours(sub_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cand_contours.extend(sc)
    else:
        sc, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand_contours.extend(sc)

    min_brick_area = 220.0 * (w_w * h_w) / (640.0 * 360.0)
    max_brick_area = 0.30 * w_w * h_w
    valid_bricks: list[tuple[float, np.ndarray, tuple[float, float]]] = []
    for c in cand_contours:
        area = float(cv2.contourArea(c))
        if area < min_brick_area or area > max_brick_area:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            continue
        cu = float(M["m10"] / M["m00"])
        cv_y = float(M["m01"] / M["m00"])
        d_px = math.hypot(cu - u_tgt, cv_y - v_tgt)
        valid_bricks.append((d_px, c, (cu, cv_y)))

    if not valid_bricks:
        cbox = [
            max(0, int(u_tgt - 0.22 * w_w)),
            max(0, int(v_tgt - 0.24 * h_w)),
            min(w_w, int(u_tgt + 0.22 * w_w)),
            min(h_w, int(v_tgt + 0.24 * h_w)),
        ]
        return p_tcp_nominal.copy(), float(jaw_yaw_nominal), cbox, False, {"status": "no_brick"}

    valid_bricks.sort(key=lambda t: t[0])
    _, best_cnt, _ = valid_bricks[0]
    other_centers_uv = [vb[2] for vb in valid_bricks[1:]]

    x_b, y_b, bw, bh = cv2.boundingRect(best_cnt)
    pad = 14
    crop_box = [
        max(0, int(x_b - pad)),
        max(0, int(y_b - pad)),
        min(w_w, int(x_b + bw + pad)),
        min(h_w, int(y_b + bh + pad)),
    ]

    rect = cv2.minAreaRect(best_cnt)
    (bu, bv), (rw, rh), _ = rect
    p_brick_w = _ray_intersect_z_plane(bu, bv, R_wc, p_wc, K_scaled, z_world=0.018)
    p_jaw_w = _ray_intersect_z_plane(u_tgt, v_tgt, R_wc, p_wc, K_scaled, z_world=0.018)
    p_refined = p_tcp_nominal.copy()
    yaw_refined = float(jaw_yaw_nominal)
    d_xy_norm = 0.0

    if p_brick_w is not None and p_jaw_w is not None:
        d_xy = p_brick_w[:2] - p_jaw_w[:2]
        d_xy_norm = float(np.linalg.norm(d_xy))
        if d_xy_norm > max_step_m:
            d_xy = d_xy * (max_step_m / d_xy_norm)
        p_refined[0] = float(p_tcp_nominal[0] + gain * d_xy[0])
        p_refined[1] = float(p_tcp_nominal[1] + gain * d_xy[1])

    # Align gripper jaw closing axis across the brick's graspable faces in world XY
    box_pts = cv2.boxPoints(rect)
    cw_pts = [
        _ray_intersect_z_plane(float(pu), float(pv), R_wc, p_wc, K_scaled, z_world=0.018)
        for pu, pv in box_pts
    ]
    yaw_err_rad = 0.0
    if all(p is not None for p in cw_pts):
        cw_arr = [p[:2] for p in cw_pts if p is not None]
        e01 = cw_arr[1] - cw_arr[0]
        e12 = cw_arr[2] - cw_arr[1]
        l01 = float(np.linalg.norm(e01))
        l12 = float(np.linalg.norm(e12))
        short_vec = e12 if l01 >= l12 else e01
        long_vec = e01 if l01 >= l12 else e12
        aspect = max(l01, l12) / max(min(l01, l12), 1e-4)

        base_short_yaw = math.atan2(float(short_vec[1]), float(short_vec[0]))
        if aspect >= 1.18:
            # Elongated brick (2x4, 2x3, 1x2x2): MUST grasp across short transverse width!
            cand_axes = [base_short_yaw]
        else:
            # Square 2x2 cube: EITHER pair of opposite walls is 32 mm — prefer the axis farthest
            # from neighboring bricks and closest to current jaw yaw!
            cand_axes = [base_short_yaw, math.atan2(float(long_vec[1]), float(long_vec[0]))]

        other_w = [
            _ray_intersect_z_plane(ou, ov, R_wc, p_wc, K_scaled, z_world=0.018)
            for ou, ov in other_centers_uv
        ]
        best_yaw_cost = 1e9
        for ax_yaw in cand_axes:
            j_dir = np.array([math.cos(ax_yaw), math.sin(ax_yaw)], dtype=np.float64)
            crowd_pen = 0.0
            if p_brick_w is not None:
                for ow in other_w:
                    if ow is None:
                        continue
                    diff_w = ow[:2] - p_brick_w[:2]
                    if float(np.linalg.norm(diff_w)) < 0.052:
                        crowd_pen += 0.60 * abs(float(np.dot(diff_w, j_dir))) / 0.052
            for k in (0, 1, -1):
                cand_y = ax_yaw + k * math.pi
                ang_diff = abs((cand_y - jaw_yaw_nominal + math.pi) % (2 * math.pi) - math.pi)
                cost_y = ang_diff + crowd_pen
                if cost_y < best_yaw_cost:
                    best_yaw_cost = cost_y
                    yaw_refined = float(cand_y)
                    yaw_err_rad = ang_diff

    # Direct Inner-Jaw Straddle Check in pixel space!
    du_px = abs(float(bu) - u_tgt)
    dv_px = abs(float(bv) - v_tgt)
    short_px = min(float(rw), float(rh))
    if jaw_obs is not None:
        fits_gap = (float(bu) - 0.5 * short_px) >= (jaw_obs.u_left_pad + 3.5) and (
            float(bu) + 0.5 * short_px
        ) <= (jaw_obs.u_right_pad - 3.5)
    else:
        fits_gap = du_px <= 6.0

    straddle_ok = bool(
        fits_gap
        and du_px <= max(5.0, 0.075 * (jaw_obs.gap_px if jaw_obs else 70.0))
        and dv_px <= 6.5
        and yaw_err_rad <= math.radians(7.0)
        and d_xy_norm <= 0.0035
    )
    info = {
        "status": "ok",
        "du_px": round(du_px, 1),
        "dv_px": round(dv_px, 1),
        "d_xy_mm": round(d_xy_norm * 1000.0, 1),
        "yaw_err_deg": round(math.degrees(yaw_err_rad), 1),
        "jaw_gap_px": round(jaw_obs.gap_px, 1) if jaw_obs is not None else None,
        "u_pinch": round(u_tgt, 1),
        "fits_gap": fits_gap,
        "short_px": round(short_px, 1),
        "long_px": round(max(float(rw), float(rh)), 1),
    }
    return p_refined, yaw_refined, crop_box, straddle_ok, info


def refine_grasp_with_wrist_rgb(
    wrist_rgb: np.ndarray,
    q_pre: np.ndarray,
    p_tcp_nominal: np.ndarray,
    jaw_yaw_nominal: float,
    ik: SO101FraxIK,
    wrist_cam: Any,
) -> tuple[np.ndarray, float, list[int] | None]:
    """Eye-in-hand 2D RGB visual servoing tracking the inner gripper jaw surfaces around the block."""
    p_ref, yaw_ref, cbox, _, _ = servo_grasp_with_jaw_corridor(
        wrist_rgb,
        q_pose=q_pre,
        p_tcp_nominal=p_tcp_nominal,
        jaw_yaw_nominal=jaw_yaw_nominal,
        ik=ik,
        wrist_cam=wrist_cam,
    )
    return p_ref, yaw_ref, cbox


def refine_plate_drop_with_wrist_rgb(
    wrist_rgb: np.ndarray,
    pile: str,
    q_above: np.ndarray,
    target_xy_nominal: np.ndarray,
    slot_offset_xy: tuple[float, float],
    ik: SO101FraxIK,
    wrist_cam: Any,
) -> np.ndarray:
    """Eye-in-hand 2D RGB drop refinement over the movable `good` (green) or `defective` (pink) plate.

    Uses the calibrated physical Sonix camera mount (`SONIX_T_BODY3_CAM`) while masking out the
    purple gripper jaws to center the drop slot cleanly inside the live observed green/pink bowl!
    """
    if wrist_rgb.size == 0 or float(wrist_rgb.std()) < 6.0:
        return target_xy_nominal.copy()
    h_w, w_w = wrist_rgb.shape[:2]
    jaw_obs = detect_wrist_jaw_corridor(wrist_rgb)
    hsv = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2HSV)
    if pile == "good":
        pmask = (
            (hsv[:, :, 0] >= 35)
            & (hsv[:, :, 0] <= 84)
            & (hsv[:, :, 1] >= 60)
            & (hsv[:, :, 2] >= 45)
        )
    else:
        pmask = (
            ((hsv[:, :, 0] >= 145) | (hsv[:, :, 0] <= 10))
            & (hsv[:, :, 1] >= 45)
            & (hsv[:, :, 2] >= 75)
        )
    if jaw_obs is not None:
        pmask &= ~jaw_obs.jaws_dilated_mask

    clean_p = cv2.morphologyEx(pmask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts_pl, _ = cv2.findContours(clean_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_pl:
        return target_xy_nominal.copy()
    best_pl = max(cnts_pl, key=cv2.contourArea)
    if float(cv2.contourArea(best_pl)) < 600.0:
        return target_xy_nominal.copy()
    M = cv2.moments(best_pl)
    if M["m00"] <= 1e-6:
        return target_xy_nominal.copy()
    cu = float(M["m10"] / M["m00"])
    cv_y = float(M["m01"] / M["m00"])

    T_wb3 = ik.fk_world_body3(q_above)
    if jaw_obs is not None or wrist_cam is None:
        R_wc, p_wc, K_scaled = _sonix_cam_pose_and_intrinsics(T_wb3, w_w, h_w)
    else:
        T_wc = wrist_cam.world_transform(T_wb3)
        R_wc, p_wc = T_wc[:3, :3], T_wc[:3, 3]
        sx = float(w_w) / float(wrist_cam.width)
        sy = float(h_w) / float(wrist_cam.height)
        K_scaled = wrist_cam.K.copy()
        K_scaled[0, :] *= sx
        K_scaled[1, :] *= sy

    p_plate_vis = _ray_intersect_z_plane(cu, cv_y, R_wc, p_wc, K_scaled, z_world=0.008)
    p_under = np.array(
        [float(target_xy_nominal[0]), float(target_xy_nominal[1]), 0.008], dtype=np.float64
    )
    pc_u = R_wc.T @ (p_under - p_wc)
    if p_plate_vis is None or float(pc_u[2]) <= 0.02:
        return target_xy_nominal.copy()

    u_u = float(K_scaled[0, 0] * (pc_u[0] / pc_u[2]) + K_scaled[0, 2])
    v_u = float(K_scaled[1, 1] * (pc_u[1] / pc_u[2]) + K_scaled[1, 2])
    p_ref = _ray_intersect_z_plane(u_u, v_u, R_wc, p_wc, K_scaled, z_world=0.008)
    if p_ref is None:
        return target_xy_nominal.copy()

    # Differential shift from current nominal XY to live observed plate centroid + slot offset
    desired_from_vis = p_plate_vis[:2] + 0.45 * np.asarray(slot_offset_xy, dtype=np.float64)
    d_xy = desired_from_vis - p_ref[:2]
    norm_d = float(np.linalg.norm(d_xy))
    if norm_d > 0.035:
        d_xy = d_xy * (0.035 / norm_d)
    refined_xy = np.array(
        [
            float(target_xy_nominal[0] + 0.80 * d_xy[0]),
            float(target_xy_nominal[1] + 0.80 * d_xy[1]),
        ],
        dtype=np.float64,
    )
    refined_xy[1] = min(float(refined_xy[1]), 0.185)
    return refined_xy


def check_bench_obstacle_clearance_m(
    ik: SO101FraxIK,
    q_enc_rad: np.ndarray,
) -> tuple[float, str]:
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
    pitches_deg: tuple[float, ...] = (88.0, 82.0, 74.0, 65.0, 56.0),
) -> tuple[GraspIKSolution, float]:
    """Solve collision-safe 5-DOF grasp/drop IK within the valid CCW Joint-5 range [+0.45, +2.98] rad."""
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
    best_pitch = float(pitches_deg[0])
    best_score = -1e9

    for pitch in pitches_deg:
        for yaw_cand in (jaw_yaw_world, jaw_yaw_world + math.pi, jaw_yaw_world - math.pi):
            sol = ik.solve_grasp_pose(
                p_tgt,
                jaw_yaw_world=yaw_cand,
                pitch_down_deg=pitch,
                q_posture=seed_posture,
            )
            q_cand = sol.q_grasp.copy()
            j5_raw = float(q_cand[4])
            # Fold modulo 2*pi into CCW window if possible, else allow small <=18 deg snap at seam
            for shift_2pi in (0.0, 2.0 * math.pi, -2.0 * math.pi):
                j5_try = j5_raw + shift_2pi
                if PHYSICAL_JOINT5_CCW_START_RAD <= j5_try <= PHYSICAL_JOINT5_CCW_END_RAD:
                    q_cand[4] = j5_try
                    break
            j5 = float(q_cand[4])
            in_ccw = PHYSICAL_JOINT5_CCW_START_RAD <= j5 <= PHYSICAL_JOINT5_CCW_END_RAD
            j5_snap_pen = 0.0
            if not in_ccw:
                j5_clamped = float(
                    np.clip(j5, PHYSICAL_JOINT5_CCW_START_RAD, PHYSICAL_JOINT5_CCW_END_RAD)
                )
                seam_err = abs(j5 - j5_clamped)
                if seam_err <= math.radians(18.5):
                    q_cand[4] = j5_clamped
                    in_ccw = True
                    j5_snap_pen = 12.0 * seam_err

            clr, _ = check_bench_obstacle_clearance_m(ik, q_cand)
            if clr < 0.012:
                continue
            T_tcp = ik.fk_world_tcp(q_cand)
            pos_err = float(np.linalg.norm(T_tcp[:3, 3] - p_tgt))
            cand_sol = GraspIKSolution(
                feasible=bool((sol.feasible or j5_snap_pen > 0.0) and pos_err <= 0.008 and in_ccw),
                q_grasp=q_cand,
                p_tcp_world=T_tcp[:3, 3].copy(),
                jaw_yaw_world=math.atan2(float(T_tcp[1, 0]), float(T_tcp[0, 0])),
                cost=sol.cost,
                manipulability_5d=sol.manipulability_5d,
            )
            if cand_sol.feasible and pos_err <= 0.006 and j5_snap_pen == 0.0:
                return cand_sol, pitch
            cand_score = (
                (30.0 if in_ccw else -40.0)
                - 300.0 * pos_err
                - j5_snap_pen
                - 0.04 * abs(pitches_deg[0] - pitch)
            )
            if cand_score > best_score:
                best_score = cand_score
                best_sol = cand_sol
                best_pitch = pitch

    if best_sol is None:
        best_sol = ik.solve_grasp_pose(
            p_tgt, jaw_yaw_world=jaw_yaw_world, pitch_down_deg=65.0, q_posture=seed_posture
        )
        best_sol.q_grasp[4] = float(
            np.clip(best_sol.q_grasp[4], PHYSICAL_JOINT5_CCW_START_RAD, PHYSICAL_JOINT5_CCW_END_RAD)
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
        self.holding_object: bool = False
        self._sim_q: np.ndarray = PHYSICAL_OBSERVATION_HOME_Q.copy()
        self._sim_grip: float = PHYSICAL_HOME_GRIPPER_PCT
        state = self.read_state_safe(fresh=True)
        if state is not None:
            self._sim_q = state.q_enc_rad.copy()
            self._sim_grip = float(state.gripper_pct)

    def clear_gripper_overload_for_release(self) -> None:
        """Clear any latched gripper clamping overload ONLY when intentionally opening jaws."""
        self.holding_object = False
        if self.dry_run:
            return
        try:
            raw = self.bridge._call("GET", "/arms/follower/state?fresh=true", None)
            joints = raw.get("joints") or {}
            if isinstance(joints, list):
                joints = {j["name"]: j for j in joints if "name" in j}
            gj = joints.get("gripper") or {}
            if gj.get("healthy") is False:
                self.bridge._call(
                    "POST", "/arms/follower/joints/gripper/torque", {"enabled": False}
                )
                time.sleep(0.06)
                self.bridge._call("POST", "/arms/follower/joints/gripper/torque", {"enabled": True})
                time.sleep(0.05)
        except Exception:
            pass

    def _try_clear_recoverable_overload(self, raw_state: dict[str, Any]) -> bool:
        """Clear transient overload on axial `wrist_roll` (and `gripper` ONLY when not holding a brick)."""
        joints = raw_state.get("joints") or {}
        if isinstance(joints, list):
            joints = {j["name"]: j for j in joints if "name" in j}
        cleared_any = False
        allowed_joints = ("wrist_roll",) if self.holding_object else ("wrist_roll", "gripper")
        for jname in allowed_joints:
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
                    return self.bridge.decode_state_message(
                        raw,
                        stop_on_fault=True,
                        allow_gripper_overload=self.holding_object,
                    )
                except ArmFaultError:
                    if not self.dry_run and self._try_clear_recoverable_overload(raw):
                        raw2 = self.bridge._call("GET", "/arms/follower/state?fresh=true", None)
                        return self.bridge.decode_state_message(
                            raw2,
                            stop_on_fault=True,
                            allow_gripper_overload=self.holding_object,
                        )
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
            self._sim_grip = 0.092
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
        n_twin_slots = max(24, int(self.max_cycles) + 4)
        prealloc = [
            SpawnedBlockInfo(
                block_id=f"block_{i}",
                spec=slot_specs[i % len(slot_specs)],
                initial_xy_yaw=(0.28, 0.0, 0.0),
            )
            for i in range(n_twin_slots)
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
        self._rgb_plate_positions: dict[str, np.ndarray] = {}
        self._active_brick: PerceivedBrick | None = None
        self._active_grasp_sol: GraspIKSolution | None = None
        self._active_open_pct: float = PHYSICAL_WIDE_PREGRASP_PCT
        self._active_close_pct: float = PHYSICAL_CARRY_GRIPPER_PCT
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
                    "inspector_mode": self.inspector.inspector_mode,
                    "anomalib_cameras": self.inspector.configured_cameras(),
                    "score_type": (
                        "raw_patchcore_distance_not_probability"
                        if self.inspector.has_anomalib_models
                        else "raw_anomaly_distance_not_probability"
                    ),
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
        """Choose the safest reachable DUPLO brick in the source plate across arbitrary shapes/yaws."""
        best_item: tuple[PerceivedBrick, GraspIKSolution, float, float] | None = None
        best_score = -1e9
        if len(self._recent_missed_xys) >= max(3, len(source_bricks) * 2):
            # Reset missed-XY history once remaining bricks were tried so every brick clears to 100%
            self._recent_missed_xys.clear()

        for b in source_bricks:
            T_wb = b.T_world_block
            p_b = T_wb[:3, 3]
            R_wb = T_wb[:3, :3]
            # Deeper flat-wall grasp height (`21 mm` standard, `26 mm` tall 1x2x2) for full pad grip
            hz = float(b.spec.dims_m[2])
            z_grasp = 0.026 if hz > 0.032 else 0.021

            miss_count = sum(
                1
                for m_xy in self._recent_missed_xys[-6:]
                if float(np.linalg.norm(p_b[:2] - m_xy)) < 0.026
            )

            yaw_spans: list[tuple[float, float, int]] = []
            for idx, span in enumerate(b.spec.dims_m):
                ax = R_wb[:, idx]
                if abs(float(ax[2])) < 0.30 and span <= 0.038:
                    y0 = math.atan2(float(ax[1]), float(ax[0]))
                    yaw_spans.extend([(y0, float(span), idx), (y0 + math.pi, float(span), idx)])
            if not yaw_spans:
                y0 = math.atan2(float(R_wb[1, 1]), float(R_wb[0, 1]))
                yaw_spans = [(y0, 0.032, 0), (y0 + math.pi, 0.032, 0)]

            along_offsets = (0.0, -0.014, +0.014) if max(b.spec.dims_m[:2]) >= 0.056 else (0.0,)
            radial_vec = p_b[:2] - source_xy
            r_plate = float(np.linalg.norm(radial_vec))
            radial_unit = radial_vec / max(r_plate, 1e-6)
            # Subtle inward nudge if a brick is resting right against the sloped outer KALAS lip
            rim_nudge = -0.003 * radial_unit if r_plate > 0.058 else np.zeros(2, dtype=np.float64)

            for yaw, span, axis_idx in yaw_spans:
                is_narrow = span <= 0.021
                base_open = (
                    PHYSICAL_NARROW_PREGRASP_PCT if is_narrow else PHYSICAL_WIDE_PREGRASP_PCT
                )
                open_pct = min(0.32, base_open + (0.02 if miss_count >= 1 else 0.0))
                close_pct = PHYSICAL_CARRY_GRIPPER_PCT
                shift_m = -0.004 if is_narrow else 0.0
                jaw_dir = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
                perp_dir = np.array([-jaw_dir[1], jaw_dir[0]], dtype=np.float64)

                # When near the curved outer plate rim, penalize jaw orientations that jam into the rim!
                rim_jaw_pen = (
                    8.0 * abs(float(np.dot(jaw_dir, radial_unit))) if r_plate > 0.048 else 0.0
                )
                # On retry 1 of a square 2x2 brick, try the orthogonal pair of walls (`axis_idx == 1`)!
                retry_axis_bonus = 5.0 if (miss_count % 2 == 1 and axis_idx == 1) else 0.0

                for s_along in along_offsets:
                    p_tcp = np.array(
                        [
                            float(p_b[0])
                            + rim_nudge[0]
                            + shift_m * jaw_dir[0]
                            + s_along * perp_dir[0],
                            float(p_b[1])
                            + rim_nudge[1]
                            + shift_m * jaw_dir[1]
                            + s_along * perp_dir[1],
                            z_grasp,
                        ],
                        dtype=np.float64,
                    )
                    sol, pitch_used = solve_adaptive_grasp_pose(
                        self.ik,
                        p_tcp,
                        jaw_yaw_world=yaw,
                        pitches_deg=(88.0, 84.0, 78.0, 70.0, 60.0),
                    )
                    pos_err = float(np.linalg.norm(sol.p_tcp_world - p_tcp))
                    if not sol.feasible and pos_err > 0.009:
                        continue

                    # Check distance to neighboring bricks along jaw opening direction AND overall crowding
                    neighbor_pen = 0.0
                    for other in source_bricks:
                        if other.block_id == b.block_id:
                            continue
                        d_xy = other.T_world_block[:2, 3] - p_tcp[:2]
                        d_norm = float(np.linalg.norm(d_xy))
                        if d_norm < 0.055:
                            neighbor_pen += 3.0 * (0.055 - d_norm) / 0.055
                            u_j = abs(float(np.dot(d_xy, jaw_dir)))
                            if 0.012 <= u_j <= 0.044:
                                neighbor_pen += 12.0

                    # Defer to easier bricks only after >= 2 consecutive misses at this spot
                    miss_pen = 22.0 * max(0, miss_count - 1) + (4.0 if miss_count == 1 else 0.0)
                    r_robot = float(math.hypot(p_b[0], p_b[1]))
                    rim_crowd_pen = 28.0 * max(0.0, r_plate - 0.046)
                    score = (
                        (20.0 if sol.feasible else 0.0)
                        - 400.0 * pos_err
                        - neighbor_pen
                        - rim_jaw_pen
                        - rim_crowd_pen
                        - miss_pen
                        + retry_axis_bonus
                        - 12.0 * abs(s_along)
                        - 8.0 * r_robot
                        - 0.08 * abs(88.0 - pitch_used)
                        + 2.0 * b.confidence
                    )
                    if score > best_score:
                        best_score = score
                        best_item = (b, sol, open_pct, close_pct)

        return best_item

    def _evaluate_camera_crop_and_record(
        self,
        snap: LiveCameraSnapshot,
        crop_box: list[int] | None,
        stage_tag: str,
        expected_z_m: float | None = None,
        z_band_m: float = 0.070,
        require_configured_detector: bool = False,
    ) -> FrameInspectionResult | None:
        """Run `RealCameraAnomalyInspector` on `snap` (`top`, `plate`, or `wrist`) and fuse flags."""
        if require_configured_detector and not self.inspector.has_detector_for_camera(
            snap.camera_name
        ):
            return None
        eff_box = self.inspector.resolve_crop_for_camera(snap.camera_name, crop_box)
        if eff_box is None:
            return None
        res = self.inspector.evaluate_frame(
            snap.rgb,
            snap.depth_m,
            eff_box,
            expected_z_m=expected_z_m,
            z_band_m=z_band_m,
            camera_name=snap.camera_name,
        )
        if res.valid_view:
            self.max_plate_anomaly_distance = (
                res.raw_anomaly_distance
                if self.max_plate_anomaly_distance is None
                else max(self.max_plate_anomaly_distance, res.raw_anomaly_distance)
            )
            if res.flagged:
                self.anomaly_flagged = True
                self.anomaly_messages.append(
                    f"{stage_tag}_{snap.camera_name}_{res.inspector_backend}="
                    f"{res.raw_anomaly_distance:.3f}>={res.threshold_value:.3f}"
                )
        return res

    def _skill_localize_next_cluster_block(self) -> SkillOutcome:
        """Retract to observation home, localize plates & bricks via 2D Top-RGB, & sync Real2Sim."""
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
        if self.hand_empty:
            self.arm.clear_gripper_overload_for_release()

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

        self.status_banner = f"LocalizeNextClusterBlock: Retracting arm & scanning 2D RGB ({self.inspector.inspector_mode})"
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

        # Run D415 depth perception strictly in parallel for the Real2Sim MuJoCo 3D Twin mirror!
        scene: Real2SimSceneState | None = None
        try:
            scene = self.frame_source.perceive_scene(self.real2sim, q_arm_rad=q_now)
            self._latest_scene = scene
            self.twin.sync_real2sim_scene(scene, settle_steps=0)
        except Exception:
            scene = self._latest_scene

        # Primary pick/place localization: crisp 2D Top-RGB (zero stereo depth drift!)
        R_top, p_top, K_top = self.real_cams["top"]
        rgb_plates, rgb_bricks = localize_plates_and_bricks_top_rgb(
            top_snap.rgb, R_top, p_top, K_top
        )
        self._rgb_plate_positions = rgb_plates

        if rgb_bricks:
            source_bricks = rgb_bricks
            source_xy = rgb_plates["source"][:2]
        elif scene is not None:
            # Fallback for synthetic uniform-color mock tests where 2D plate rings are not drawn
            source_bricks = [b for b in scene.bricks if b.support_region == "source"]
            source_xy = scene.plates["source"].pos_world[:2]
            for role_k, pl_obj in scene.plates.items():
                if role_k not in self._rgb_plate_positions or float(top_snap.rgb.std()) < 8.0:
                    self._rgb_plate_positions[role_k] = pl_obj.pos_world.copy()
        else:
            source_bricks = []
            source_xy = rgb_plates["source"][:2]

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

        # Multi-camera pre-pick inspection across BOTH `top` D415 AND `plate` (`side`) D415 feeds
        R_pl, p_pl, K_pl = self.real_cams["plate"]
        all_dash_anoms: list[dict[str, Any]] = []
        for cand_b in source_bricks:
            cand_tbox = self._project_to_top_box(cand_b.T_world_block, cand_b.spec.dims_m)
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
            is_target = cand_b.block_id == brick.block_id

            # 1) Overhead `top` camera evaluation
            if cand_tbox is not None and self.inspector.has_detector_for_camera("top"):
                try:
                    top_res = self.inspector.evaluate_frame(
                        top_snap.rgb,
                        None,
                        self.inspector.resolve_crop_for_camera("top", cand_tbox) or cand_tbox,
                        expected_z_m=None,
                        camera_name="top",
                    )
                    if is_target and top_res.valid_view:
                        self.max_plate_anomaly_distance = (
                            top_res.raw_anomaly_distance
                            if self.max_plate_anomaly_distance is None
                            else max(self.max_plate_anomaly_distance, top_res.raw_anomaly_distance)
                        )
                        if top_res.flagged:
                            self.anomaly_flagged = True
                            self.anomaly_messages.append(
                                f"pre_pick_top_{top_res.inspector_backend}="
                                f"{top_res.raw_anomaly_distance:.3f}>={top_res.threshold_value:.3f}"
                            )
                    if top_res.valid_view and top_res.flagged:
                        all_dash_anoms.append(
                            {
                                "box": cand_tbox,
                                "score": top_res.raw_anomaly_distance,
                                "label": f"anomaly ({top_res.raw_anomaly_distance:.2f})",
                                "object_id": PART_ID_TO_DASHBOARD_ID.get(
                                    cand_b.spec.part_id, "part-001"
                                ),
                            }
                        )
                except Exception as exc:
                    if is_target:
                        self.anomaly_untrusted_error = f"pre_pick_top_inspector_error:{exc}"
                        return SkillOutcome(False, reason=self.anomaly_untrusted_error)

            # 2) Low `plate` (`side`) camera evaluation
            if cand_pbox is not None:
                try:
                    cand_res = self.inspector.evaluate_frame(
                        plate_snap.rgb,
                        plate_snap.depth_m,
                        self.inspector.resolve_crop_for_camera("plate", cand_pbox) or cand_pbox,
                        expected_z_m=cand_z,
                        z_band_m=0.065,
                        camera_name="plate",
                    )
                except Exception as exc:
                    if is_target:
                        self.anomaly_untrusted_error = f"pre_pick_plate_inspector_error:{exc}"
                        return SkillOutcome(False, reason=self.anomaly_untrusted_error)
                    continue

                if is_target and cand_res.valid_view:
                    self.max_plate_anomaly_distance = (
                        cand_res.raw_anomaly_distance
                        if self.max_plate_anomaly_distance is None
                        else max(self.max_plate_anomaly_distance, cand_res.raw_anomaly_distance)
                    )
                    if cand_res.flagged:
                        self.anomaly_flagged = True
                        self.anomaly_messages.append(
                            f"pre_pick_plate_{cand_res.inspector_backend}="
                            f"{cand_res.raw_anomaly_distance:.3f}>={cand_res.threshold_value:.3f}"
                        )

                if cand_res.valid_view and cand_res.flagged and cand_tbox is not None:
                    all_dash_anoms.append(
                        {
                            "box": cand_tbox,
                            "score": cand_res.raw_anomaly_distance,
                            "label": f"anomaly ({cand_res.raw_anomaly_distance:.2f})",
                            "object_id": PART_ID_TO_DASHBOARD_ID.get(
                                cand_b.spec.part_id, "part-001"
                            ),
                        }
                    )

        if self.dashboard is not None:
            self.dashboard.report_anomalies(all_dash_anoms)

        self.status_banner = (
            f"Target Locked (2D RGB): {brick.spec.name} at ({brick.T_world_block[0, 3]:.3f}, "
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
                "inspector_mode": self.inspector.inspector_mode,
            },
        )

    def _skill_pick_block(self) -> SkillOutcome:
        """2-Stage Closed-Loop Inner-Jaw Corridor Visual Servoing -> Vertical Plunge -> Confirm Pick."""
        if self._active_brick is None or self._active_grasp_sol is None:
            return SkillOutcome(False, reason="no_localized_brick")

        brick = self._active_brick
        grasp_sol = self._active_grasp_sol
        p_tcp = grasp_sol.p_tcp_world.copy()
        jaw_yaw = float(grasp_sol.jaw_yaw_world)
        steep_pitches = (88.0, 84.0, 78.0, 70.0, 60.0)

        pre_sol, _ = solve_adaptive_grasp_pose(
            self.ik,
            np.array([p_tcp[0], p_tcp[1], float(p_tcp[2]) + 0.075], dtype=np.float64),
            jaw_yaw_world=jaw_yaw,
            q_posture=grasp_sol.q_grasp,
            pitches_deg=steep_pitches,
        )

        self.status_banner = (
            f"PickBlock({brick.spec.part_id}): High pre-grasp (open={self._active_open_pct:.2f})"
        )
        self.arm.move_to_pose(
            pre_sol.q_grasp,
            gripper_pct=self._active_open_pct,
            speed=230,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        if not self.arm.dry_run:
            time.sleep(0.18)

        # Pass 1 (High Pre-Grasp ~9.6 cm): Align jaw yaw + coarse XY relative to inner jaw surfaces
        try:
            q_pre_meas, g_pre_meas = self.arm.current_q_and_grip(fresh=True)
            wrist_pre_snap = self.frame_source.fetch_wrist_snapshot()
            p_refined, yaw_refined, wrist_crop, _, servo_info = servo_grasp_with_jaw_corridor(
                wrist_pre_snap.rgb,
                q_pose=q_pre_meas,
                p_tcp_nominal=p_tcp,
                jaw_yaw_nominal=jaw_yaw,
                ik=self.ik,
                wrist_cam=self.rig.wrist,
                max_step_m=0.036,
                gain=0.92,
            )
            try:
                self._evaluate_camera_crop_and_record(
                    wrist_pre_snap,
                    wrist_crop,
                    stage_tag="pre_grasp",
                )
            except Exception as exc:
                self.anomaly_untrusted_error = f"wrist_pre_grasp_inspector_error:{exc}"
                return SkillOutcome(False, reason=self.anomaly_untrusted_error)

            if not self.arm.dry_run and servo_info.get("status") == "ok":
                p_tcp = p_refined
                jaw_yaw = yaw_refined
                grasp_sol, _ = solve_adaptive_grasp_pose(
                    self.ik,
                    p_tcp,
                    jaw_yaw_world=jaw_yaw,
                    q_posture=grasp_sol.q_grasp,
                    pitches_deg=steep_pitches,
                )
                jaw_yaw = float(grasp_sol.jaw_yaw_world)
                pre_sol, _ = solve_adaptive_grasp_pose(
                    self.ik,
                    np.array([p_tcp[0], p_tcp[1], float(p_tcp[2]) + 0.075], dtype=np.float64),
                    jaw_yaw_world=jaw_yaw,
                    q_posture=pre_sol.q_grasp,
                    pitches_deg=steep_pitches,
                )
                self._active_grasp_sol = grasp_sol
            self._sync_mirror_and_telemetry(q_pre_meas, g_pre_meas, wrist_rgb=wrist_pre_snap.rgb)
        except StaleCameraFrameError:
            pass

        # Pass 2 (Low-Hover Straddle Loop at z = z_grasp + 27 mm ≈ 48 mm, just above brick studs!):
        # Track the inner purple gripper surfaces directly around the brick and trim until centered!
        if not self.arm.dry_run:
            z_low = float(p_tcp[2]) + 0.027
            low_sol, _ = solve_adaptive_grasp_pose(
                self.ik,
                np.array([p_tcp[0], p_tcp[1], z_low], dtype=np.float64),
                jaw_yaw_world=jaw_yaw,
                q_posture=grasp_sol.q_grasp,
                pitches_deg=steep_pitches,
            )
            self.status_banner = (
                f"PickBlock({brick.spec.part_id}): Low-hover inner-jaw straddle align"
            )
            self.arm.move_to_pose(
                low_sol.q_grasp,
                gripper_pct=self._active_open_pct,
                speed=195,
                include_gripper=False,
                on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
            )
            time.sleep(0.15)

            for trim_iter in range(2):
                try:
                    q_low_meas, _ = self.arm.current_q_and_grip(fresh=True)
                    wrist_low_snap = self.frame_source.fetch_wrist_snapshot()
                    p_trim, yaw_trim, _, straddle_ok, low_info = servo_grasp_with_jaw_corridor(
                        wrist_low_snap.rgb,
                        q_pose=q_low_meas,
                        p_tcp_nominal=p_tcp,
                        jaw_yaw_nominal=jaw_yaw,
                        ik=self.ik,
                        wrist_cam=self.rig.wrist,
                        max_step_m=0.020,
                        gain=0.85,
                    )
                    if low_info.get("status") != "ok" or straddle_ok:
                        self._sync_mirror_and_telemetry(
                            q_low_meas, self._active_open_pct, wrist_rgb=wrist_low_snap.rgb
                        )
                        break

                    d_trim = float(np.linalg.norm(p_trim[:2] - p_tcp[:2]))
                    dyaw_trim = abs((yaw_trim - jaw_yaw + math.pi) % (2 * math.pi) - math.pi)
                    if d_trim < 0.0022 and dyaw_trim < math.radians(4.5):
                        break

                    p_tcp = p_trim
                    jaw_yaw = yaw_trim
                    grasp_sol, _ = solve_adaptive_grasp_pose(
                        self.ik,
                        p_tcp,
                        jaw_yaw_world=jaw_yaw,
                        q_posture=grasp_sol.q_grasp,
                        pitches_deg=steep_pitches,
                    )
                    jaw_yaw = float(grasp_sol.jaw_yaw_world)
                    low_sol, _ = solve_adaptive_grasp_pose(
                        self.ik,
                        np.array([p_tcp[0], p_tcp[1], z_low], dtype=np.float64),
                        jaw_yaw_world=jaw_yaw,
                        q_posture=low_sol.q_grasp,
                        pitches_deg=steep_pitches,
                    )
                    pre_sol, _ = solve_adaptive_grasp_pose(
                        self.ik,
                        np.array([p_tcp[0], p_tcp[1], float(p_tcp[2]) + 0.075], dtype=np.float64),
                        jaw_yaw_world=jaw_yaw,
                        q_posture=pre_sol.q_grasp,
                        pitches_deg=steep_pitches,
                    )
                    self._active_grasp_sol = grasp_sol
                    self.status_banner = (
                        f"PickBlock({brick.spec.part_id}): Jaw-corridor trim #{trim_iter + 1} "
                        f"(dxy={d_trim * 1000:.1f}mm, gap={low_info.get('jaw_gap_px')}px)"
                    )
                    self.arm.move_to_pose(
                        low_sol.q_grasp,
                        gripper_pct=self._active_open_pct,
                        speed=175,
                        include_gripper=False,
                        on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(
                            q, g, wrist_rgb=wrist_low_snap.rgb
                        ),
                    )
                    time.sleep(0.12)
                except StaleCameraFrameError:
                    break

        self.status_banner = f"PickBlock({brick.spec.part_id}): Vertical plunge & clamping jaws"
        self.arm.move_to_pose(
            grasp_sol.q_grasp,
            gripper_pct=self._active_open_pct,
            speed=145,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        # Mark `holding_object = True` BEFORE clamping so gripper holding torque is never relaxed!
        self.arm.holding_object = True
        q_clamped, g_contact = self.arm.close_gripper_until_contact(
            open_pct=self._active_open_pct,
            speed=260,
            max_wait_s=3.8,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        if not self.arm.dry_run and (
            float(g_contact) <= PHYSICAL_EMPTY_JAW_MAX_PCT
            or float(g_contact) >= (self._active_open_pct - 0.022)
        ):
            # Jaws closed empty or stopped early — release cleanly, record XY, and retry pick!
            self.arm.clear_gripper_overload_for_release()
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

        # Command demonstrated strong carry pinch (~0.035 = ~1391 ticks) — never relaxed while held!
        self._active_close_pct = (
            PHYSICAL_CARRY_GRIPPER_PCT
            if self.arm.dry_run
            else max(0.024, min(0.040, float(g_contact) - 0.058))
        )
        if not self.arm.dry_run:
            self.arm.move_to_pose(
                grasp_sol.q_grasp,
                gripper_pct=self._active_close_pct,
                speed=185,
                timeout_s=0.70,
            )

        T_body3_grasp = self.ik.fk_world_body3(q_clamped)
        self._T_body3_block_held = np.linalg.inv(T_body3_grasp) @ brick.T_world_block

        self.status_banner = f"PickBlock({brick.spec.part_id}): Lifting & confirming physical grip"
        self.arm.move_to_pose(
            pre_sol.q_grasp,
            gripper_pct=self._active_close_pct,
            speed=185,
            include_gripper=False,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        if not self.arm.dry_run:
            time.sleep(0.22)

        # Post-lift physical pick confirmation: verify jaws did NOT close to empty stop during lift!
        q_lifted, g_lifted = self.arm.current_q_and_grip(fresh=True)
        if not self.arm.dry_run and (
            float(g_lifted) <= PHYSICAL_EMPTY_JAW_MAX_PCT
            or float(g_lifted) < (float(g_contact) - 0.042)
        ):
            self.arm.clear_gripper_overload_for_release()
            self._recent_missed_xys.append(brick.T_world_block[:2, 3].copy())
            self.arm.move_to_pose(
                pre_sol.q_grasp,
                gripper_pct=PHYSICAL_HOME_GRIPPER_PCT,
                speed=230,
                timeout_s=2.8,
                on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
            )
            self.hand_empty = True
            self.ledger.reset_all()
            return SkillOutcome(
                False,
                reason=f"pick_not_confirmed_after_lift_retrying(g_lift={g_lifted:.3f},g_contact={g_contact:.3f})",
            )

        # Also score close-up wrist view of the held block right after lift!
        try:
            wrist_lift_snap = self.frame_source.fetch_wrist_snapshot()
            h_wl, w_wl = wrist_lift_snap.rgb.shape[:2]
            held_w_box = [
                int(0.24 * w_wl),
                int(0.28 * h_wl),
                int(0.76 * w_wl),
                int(0.92 * h_wl),
            ]
            self._evaluate_camera_crop_and_record(
                wrist_lift_snap, held_w_box, stage_tag="post_lift"
            )
            self._sync_mirror_and_telemetry(q_lifted, g_lifted, wrist_rgb=wrist_lift_snap.rgb)
        except StaleCameraFrameError:
            pass
        except Exception as exc:
            self.anomaly_untrusted_error = f"wrist_post_lift_inspector_error:{exc}"
            return SkillOutcome(False, reason=self.anomaly_untrusted_error)

        self.hand_empty = False
        return SkillOutcome(
            True,
            reason="block_picked_and_confirmed",
            metadata={
                "part_name": brick.spec.name,
                "contact_gripper_pct": round(float(g_contact), 3),
                "lifted_gripper_pct": round(float(g_lifted), 3),
                "close_gripper_pct": round(float(self._active_close_pct), 3),
            },
        )

    def _skill_present_and_spin_at_plate_camera(self) -> SkillOutcome:
        """Present held brick and require the selected policy's scheduled RGB views."""
        from brickcell.inspection.mark_scan import MarkScanEvidence
        from brickcell.inspection.types import Decision

        self.plate_scan_completed = False
        if self.hand_empty or self._active_brick is None:
            return SkillOutcome(False, reason="block_not_held")
        heuristic = getattr(self.inspector, "allow_heuristic_sorting", False)
        missing_models = [
            camera
            for camera in ("plate", "wrist")
            if not self.inspector.has_detector_for_camera(camera)
        ]
        if missing_models and not heuristic:
            return SkillOutcome(False, reason=f"missing_mark_models({','.join(missing_models)})")
        scan_evidence = (
            MarkScanEvidence(required=frozenset(("plate", i) for i in range(7)))
            if heuristic
            else MarkScanEvidence()
        )

        brick = self._active_brick
        dash_obj_id = PART_ID_TO_DASHBOARD_ID.get(brick.spec.part_id, "part-001")
        radial_az = math.atan2(float(PHYSICAL_PLATE_SCAN_POS[1]), float(PHYSICAL_PLATE_SCAN_POS[0]))
        q_demo_seed = np.array([0.45, 0.65, -0.65, 0.62, 1.72], dtype=np.float64)
        pres_sol = self.ik.solve_grasp_pose(
            PHYSICAL_PLATE_SCAN_POS,
            jaw_yaw_world=radial_az + 0.5 * math.pi,
            pitch_down_deg=PHYSICAL_PLATE_SCAN_PITCH_DEG,
            q_posture=q_demo_seed,
        )

        q_scan_start = pres_sol.q_grasp.copy()
        q_scan_start[4] = max(
            float(self.ik.cal_kin.calibration.lower[4]) + 0.08, PHYSICAL_JOINT5_CCW_START_RAD
        )
        q_scan_end = pres_sol.q_grasp.copy()
        q_scan_end[4] = min(
            float(self.ik.cal_kin.calibration.upper[4]) - 0.08, PHYSICAL_JOINT5_CCW_END_RAD
        )

        self.arm.holding_object = True
        q_cur, _ = self.arm.current_q_and_grip(fresh=False)
        q_carry = 0.5 * (q_cur + q_scan_start)
        q_carry[1] = min(float(q_cur[1]), float(q_scan_start[1]), -0.45)
        self.status_banner = "PresentAndSpinAtPlateCamera: Moving held brick to Side/Plate D415"
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

        # Pre-spin grip verification: if block slipped during transit, re-localize & retry pick!
        _, g_arrival = self.arm.current_q_and_grip(fresh=True)
        if not self.arm.dry_run and float(g_arrival) <= PHYSICAL_EMPTY_JAW_MAX_PCT:
            self.arm.clear_gripper_overload_for_release()
            self.hand_empty = True
            self.ledger.reset_all()
            self.arm.move_to_pose(
                PHYSICAL_OBSERVATION_HOME_Q,
                gripper_pct=PHYSICAL_HOME_GRIPPER_PCT,
                speed=210,
                on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
            )
            return SkillOutcome(False, reason=f"pick_dropped_retrying(g_arr={g_arrival:.3f})")

        self.plate_scan_active = True
        self.plate_scan_valid_frames = 0
        self.plate_scan_keyframes.clear()
        R_pl, p_pl, K_pl = self.real_cams["plate"]
        n_stops = 6
        seen_frame_ids: set[str] = set()
        verified_held = False

        for step_idx in range(n_stops + 1):
            tau = step_idx / n_stops
            q_step = q_scan_start.copy()
            q_step[4] = float(q_scan_start[4] + tau * (q_scan_end[4] - q_scan_start[4]))
            swept_deg = math.degrees(q_step[4] - q_scan_start[4])
            self.status_banner = (
                f"Side Camera CCW Spin +{swept_deg:.0f} deg | "
                f"{'ANOMALY FLAGGED' if self.anomaly_flagged else 'Inspecting plate and wrist...'}"
            )
            # `include_gripper=False` ensures gripper goal NEVER relaxes during the spin!
            self.arm.move_to_pose(
                q_step,
                gripper_pct=self._active_close_pct,
                speed=230,
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

            eval_T = brick.T_world_block if self.arm.dry_run else T_held
            plate_box, _ = project_world_obb_to_image_box(
                eval_T,
                brick.spec.dims_m,
                R_pl,
                p_pl,
                K_pl,
                width=640,
                height=480,
                pad_px=26,
            )
            if plate_box is None:
                continue

            try:
                res_plate = self._evaluate_camera_crop_and_record(
                    plate_snap,
                    plate_box,
                    stage_tag=f"spin_{step_idx}",
                    expected_z_m=None,
                )
                scan_evidence.record(
                    "plate",
                    step_idx,
                    plate_snap.frame_id,
                    valid=res_plate is not None and res_plate.valid_view,
                    flagged=res_plate is not None and res_plate.flagged,
                )
                if not heuristic and step_idx in (0, 3, 6):
                    hw_w = wrist_snap.rgb.shape[:2]
                    w_box = [
                        int(0.24 * hw_w[1]),
                        int(0.28 * hw_w[0]),
                        int(0.76 * hw_w[1]),
                        int(0.92 * hw_w[0]),
                    ]
                    res_wrist = self._evaluate_camera_crop_and_record(
                        wrist_snap,
                        w_box,
                        stage_tag=f"spin_{step_idx}",
                        require_configured_detector=True,
                    )
                    scan_evidence.record(
                        "wrist",
                        step_idx,
                        wrist_snap.frame_id,
                        valid=res_wrist is not None and res_wrist.valid_view,
                        flagged=res_wrist is not None and res_wrist.flagged,
                    )
            except Exception as exc:
                self.anomaly_untrusted_error = f"inspector_failure:{exc}"
                break

            if res_plate is not None and res_plate.valid_view:
                self.plate_scan_valid_frames += 1
                if step_idx in (0, 2, 4, 6) and len(self.plate_scan_keyframes) < 4:
                    self.plate_scan_keyframes.append(plate_snap.rgb.copy())
                if self.arm.dry_run or g_meas > PHYSICAL_EMPTY_JAW_MAX_PCT:
                    verified_held = True
                if self.anomaly_flagged and self.dashboard is not None and top_box is not None:
                    self.dashboard.report_anomalies(
                        [
                            {
                                "box": top_box,
                                "score": float(self.max_plate_anomaly_distance or 0.0),
                                "label": (
                                    "anomalib defect"
                                    if self.inspector.has_anomalib_models
                                    else "dark mark"
                                ),
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
        if not verified_held:
            self.arm.clear_gripper_overload_for_release()
            self.hand_empty = True
            self.ledger.reset_all()
            self.arm.move_to_pose(
                PHYSICAL_OBSERVATION_HOME_Q, gripper_pct=PHYSICAL_HOME_GRIPPER_PCT, speed=200
            )
            return SkillOutcome(False, reason="pick_dropped_retrying(not_held_during_spin)")
        if self.plate_scan_valid_frames < 3:
            return SkillOutcome(
                False, reason=f"insufficient_valid_plate_views({self.plate_scan_valid_frames})"
            )

        if scan_evidence.decision() == Decision.REINSPECT or (heuristic and scan_evidence.missing):
            return SkillOutcome(
                False, reason=f"incomplete_plate_wrist_scan({scan_evidence.missing})"
            )

        self.plate_scan_completed = True
        return SkillOutcome(
            True,
            reason="plate_ccw_scan_completed",
            metadata={
                "valid_plate_views": self.plate_scan_valid_frames,
                "anomaly_flagged": self.anomaly_flagged,
                "max_raw_anomaly_distance": self.max_plate_anomaly_distance,
                "inspector_mode": self.inspector.inspector_mode,
                "mark_scan_decision": str(scan_evidence.decision()),
                "missing_mark_views": scan_evidence.missing,
            },
        )

    def _compute_safe_plate_drop_xy(
        self, pile: str, slot_idx: int
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Compute reachable, collision-safe drop XY inside the live movable `good` or `defective` plate."""
        if pile in self._rgb_plate_positions:
            plate_xy = self._rgb_plate_positions[pile][:2].copy()
        elif self._latest_scene is not None and pile in self._latest_scene.plates:
            plate_xy = self._latest_scene.plates[pile].pos_world[:2].copy()
        else:
            plate_xy = (
                np.array([0.395, 0.116], dtype=np.float64)
                if pile == "good"
                else np.array([0.229, 0.136], dtype=np.float64)
            )

        # Keep inside SO-101 pitched-wrist kinematic envelope (r <= 0.415 m demonstrated in Episode 0!)
        r_center = float(np.linalg.norm(plate_xy))
        if r_center > 0.412:
            inward = min(0.045, r_center - 0.405)
            plate_xy = plate_xy * ((r_center - inward) / r_center)

        dx, dy = PLATE_DROP_SLOT_OFFSETS_XY[slot_idx % len(PLATE_DROP_SLOT_OFFSETS_XY)]
        slot_off = (0.45 * float(dx), 0.45 * float(dy))
        target_xy = np.array(
            [plate_xy[0] + slot_off[0], plate_xy[1] + slot_off[1]], dtype=np.float64
        )
        # Keep strictly clear of left laptop screen (y <= +0.168 m)
        target_xy[1] = min(float(target_xy[1]), 0.168)
        return target_xy, slot_off

    def _skill_sort_block_to_pile(self, pile: str) -> SkillOutcome:
        """Sort inspected brick into the live movable `good` (green) or `defective` (pink) KALAS plate."""
        if self.hand_empty or not self.plate_scan_completed or self.anomaly_untrusted_error:
            return SkillOutcome(False, reason="untrusted_or_unscanned_sort_refused")
        if pile not in {"good", "defective"}:
            return SkillOutcome(False, reason=f"invalid_pile:{pile}")

        # Re-read fresh 2D Top-RGB plate positions right before sorting in case a plate was moved!
        try:
            top_live = self.frame_source.fetch_d415_snapshot("top")
            R_top, p_top, K_top = self.real_cams["top"]
            fresh_plates, _ = localize_plates_and_bricks_top_rgb(top_live.rgb, R_top, p_top, K_top)
            if float(top_live.rgb.std()) >= 8.0:
                self._rgb_plate_positions.update(fresh_plates)
        except Exception:
            pass

        slot_idx = int(self.sorted_counts[pile])
        target_xy, slot_off = self._compute_safe_plate_drop_xy(pile, slot_idx)
        drop_yaw = math.atan2(float(target_xy[1]), float(target_xy[0])) + 0.5 * math.pi
        drop_pitches = (72.0, 62.0, 52.0, 42.0, 34.0)

        above_sol, _ = solve_adaptive_grasp_pose(
            self.ik,
            np.array([target_xy[0], target_xy[1], 0.135], dtype=np.float64),
            jaw_yaw_world=drop_yaw,
            pitches_deg=drop_pitches,
        )

        self.status_banner = f"SortBlockToPile({pile.upper()}): Transiting to {pile} KALAS disk"
        # Unwind Joint 5 smoothly at safe elevation before lateral swing while keeping firm grip
        q_cur, _ = self.arm.current_q_and_grip(fresh=False)
        q_mid = 0.5 * (q_cur + above_sol.q_grasp)
        q_mid[1] = min(float(q_cur[1]), float(above_sol.q_grasp[1]), -0.65)
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

        # Eye-in-hand Wrist RGB refinement over the green/pink plate bowl before descent!
        if not self.arm.dry_run:
            try:
                q_abv, _ = self.arm.current_q_and_grip(fresh=True)
                wrist_drop_snap = self.frame_source.fetch_wrist_snapshot()
                refined_xy = refine_plate_drop_with_wrist_rgb(
                    wrist_drop_snap.rgb,
                    pile=pile,
                    q_above=q_abv,
                    target_xy_nominal=target_xy,
                    slot_offset_xy=slot_off,
                    ik=self.ik,
                    wrist_cam=self.rig.wrist,
                )
                if float(np.linalg.norm(refined_xy - target_xy)) >= 0.005:
                    target_xy = refined_xy
                    drop_yaw = math.atan2(float(target_xy[1]), float(target_xy[0])) + 0.5 * math.pi
            except Exception:
                pass

        release_sol, _ = solve_adaptive_grasp_pose(
            self.ik,
            np.array([target_xy[0], target_xy[1], 0.060], dtype=np.float64),
            jaw_yaw_world=drop_yaw,
            q_posture=above_sol.q_grasp,
            pitches_deg=drop_pitches,
        )
        self.arm.move_to_pose(
            release_sol.q_grasp,
            gripper_pct=self._active_close_pct,
            speed=160,
            include_gripper=False,
            on_step_cb=lambda q, g: self._sync_mirror_and_telemetry(q, g),
        )
        # Clear any clamping overload right before intentional release over target KALAS disk!
        self.arm.clear_gripper_overload_for_release()
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
                "inspector_mode": self.inspector.inspector_mode,
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
    max_cycles: int = 24,
    daemon_standby_s: float = 0.0,
    show_twin_viewer: bool = False,
    dark_mark_threshold: float = 16.0,
    openvino_config_path: str | Path | None = None,
    rgbd_viewer_config_path: str | Path | None = None,
    require_anomalib: bool = False,
    output_dir: str | Path | None = None,
    inspection_mode: str = "lab-dark-mark",
) -> dict[str, Any]:
    """Run the full physical SO-101 3-plate inspection deployment on `intel`."""
    out_root = (
        Path(output_dir)
        if output_dir is not None
        else (REPO_ROOT / "artifacts" / "block_inspector" / "hardware_deploy")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if inspection_mode not in ("patchcore", "lab-dark-mark"):
        raise ValueError("Unknown inspection mode")
    heuristic = inspection_mode == "lab-dark-mark"
    if heuristic and (require_anomalib or openvino_config_path is not None):
        raise ValueError("lab-dark-mark cannot be combined with Anomalib model selection")
    if heuristic:
        cam_detectors, cam_crops = {}, {}
    else:
        cam_detectors, cam_crops = load_anomalib_camera_detectors(
            openvino_model_path=openvino_config_path,
            rgbd_viewer_config_path=rgbd_viewer_config_path,
        )
    missing_models = [
        camera for camera in ("plate", "wrist") if "rgb" not in cam_detectors.get(camera, {})
    ]
    if missing_models and not heuristic:
        raise ValueError(
            f"missing_mark_models({','.join(missing_models)}): deployment refused before "
            "camera/arm initialization or pickup. Both plate and wrist require real, "
            "calibrated RGB OpenVINO models. Configure cameras.<camera>.models.rgb in "
            "configs/mark-viewer.json (paths relative to the config), or pass "
            "--openvino-model. Each export needs model.xml, model.bin and model.json "
            "with a calibrated threshold and matching sensor contract. "
            "Uncalibrated pilots and lab_dark_mark_fallback cannot authorize sorting."
        )

    inspector = RealCameraAnomalyInspector(
        dark_mark_threshold=dark_mark_threshold,
        camera_detectors=cam_detectors,
        camera_crops=cam_crops,
        allow_heuristic_sorting=heuristic,
    )
    frame_source = LiveBenchHttpFrameSource(
        live_depth_url=live_depth_url,
        wrist_http_url=wrist_http_url,
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
        max_steps=max(60, max_cycles * 10),
        max_replans=max(24, max_cycles * 3),
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
            "inspector_mode": inspector.inspector_mode,
            "anomalib_cameras": inspector.configured_cameras(),
            "inspection_policy": inspection_mode,
            "threshold_kind": "operator_selected_lab_contrast"
            if heuristic
            else "validation_calibrated",
            "lab_contrast_threshold": dark_mark_threshold if heuristic else None,
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
