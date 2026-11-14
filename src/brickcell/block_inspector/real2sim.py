"""Real-Time Lightweight Real2Sim Point-Cloud Pipeline for the SO-101 Inspection Cell.

Ingests sensor-fused 3D colored point clouds from the dual Intel RealSense D415 rig
(or teammates' `live_depth.py` / `realsense_depth_capture.py` / `.ply` point clouds),
filters out non-target distractors (human arms/hands, trash, wires/cables, robot links,
and table/plate support surfaces), localizes all LEGO DUPLO bricks (`assets/duplo/*.stl`)
across their 6D stable poses (`assets/duplo/stable_poses.json`) and the 3 IKEA KALAS plates,
replicates the clean scene in MuJoCo 4.0.1 Desktop, and closes the loop for policy training
and deployment.
"""

from __future__ import annotations

import base64
import io
import json
import math
import socket
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN, KMeans

from brickcell.block_inspector.armd_bridge import ArmdFollowerBridge
from brickcell.block_inspector.cogman import CogManInspectionAgent, InspectionMode
from brickcell.block_inspector.geometry import (
    DUPLO_CATALOG,
    REPO_ROOT,
    T_WORLD_BASE,
    BlockSpec,
    CameraExtrinsicsRig,
    CameraModel,
    so3_geodesic_angle_rad,
    station_camera_rig,
    station_plates_config,
)
from brickcell.block_inspector.perception import (
    SOURCE_PLATE_FLOOR_Z_M,
    SOURCE_PLATE_OUTER_RADIUS_M,
    _split_touching_cluster_if_needed,
    refine_cuboid_icp,
)
from brickcell.block_inspector.twin import (
    SO101InspectionTwin,
    SpawnedBlockInfo,
)

DistractorCategory = Literal["human_arm", "wire", "trash", "robot_arm", "support_surface"]

# Full 3-plate tabletop cell workspace bounds (+X forward, +Y left, +Z up)
REAL2SIM_WORKSPACE_MIN = np.array([-0.02, -0.36, -0.015], dtype=np.float64)
REAL2SIM_WORKSPACE_MAX = np.array([0.46, +0.36, +0.360], dtype=np.float64)

# Calibrated physical Intel RealSense D415 tabletop registration (Top `350122060653` & Plate `243622060450`)
# Aligns the wooden tabletop plane to z_world = 0 and the physical SO-101 base to [0, 0, 0] (+X forward).
REAL_TOP_TO_WORLD_R = np.array(
    [
        [-0.94596661, 0.27834966, -0.16634496],
        [0.32404133, 0.79325277, -0.51550679],
        [-0.01199911, -0.54144374, -0.84064957],
    ],
    dtype=np.float64,
)
REAL_TOP_ORIGIN_CAM = np.array([0.3199, -0.0735, 0.9318], dtype=np.float64)
REAL_TOP_FROM_PLATE_R = np.array(
    [
        [-0.66655404, -0.35842738, 0.65363256],
        [0.66451680, 0.11166054, 0.73888385],
        [-0.33782116, 0.92685584, 0.16375323],
    ],
    dtype=np.float64,
)
REAL_TOP_FROM_PLATE_T = np.array([-0.22566542, -0.24678325, 0.76562479], dtype=np.float64)

# Exact link OBB centers & half-extents (m) from artifacts/dynamics/published_priors_verified.json
# Inflated to cover servo hubs, fastener overhangs, and the 3D-printed wrist camera mount.
_SO101_LINK_OBBS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "Body5_1": ((-0.0004, -0.0067, 0.0059), (0.085, 0.078, 0.070)),
    "Body1_2": ((-0.0209, -0.0002, -0.0085), (0.065, 0.062, 0.088)),
    "Group 1": ((-0.0593, -0.0114, 0.0207), (0.105, 0.065, 0.072)),
    "Body1_3": ((-0.0683, 0.0015, 0.0184), (0.112, 0.062, 0.072)),
    "Group 1_2": ((-0.0281, 0.0028, -0.0114), (0.078, 0.062, 0.072)),
    "Body3": ((-0.0024, -0.0025, -0.0450), (0.088, 0.085, 0.105)),
    "Body5_2": ((0.0394, -0.0142, -0.0184), (0.075, 0.055, 0.062)),
}


@dataclass(frozen=True)
class StablePoseEntry:
    """One resting orientation from `assets/duplo/stable_poses.json`."""

    name: str
    rpy_deg: tuple[float, float, float]
    R_local: np.ndarray


@dataclass(frozen=True)
class PerceivedPlate:
    """Localized 3D state of one IKEA KALAS plate (`source`, `good`, or `defective`)."""

    role: str
    pos_world: np.ndarray
    rgba: tuple[float, float, float, float]
    inlier_count: int
    from_prior: bool = False


@dataclass(frozen=True)
class PerceivedBrick:
    """6D localized LEGO DUPLO brick ready for MuJoCo replication and policy execution."""

    block_id: str
    spec: BlockSpec
    T_world_block: np.ndarray  # (4, 4) body-centered frame matching SO101InspectionTwin
    T_world_mesh_origin: np.ndarray  # (4, 4) STL base-origin frame matching assets/duplo/*.stl
    stable_pose_name: str
    xy_yaw: tuple[float, float, float]
    support_region: str  # "source", "good", "defective", or "table"
    icp_fitness: float
    icp_rmse_m: float
    confidence: float
    has_visual_defect: bool
    num_points: int
    mean_rgb: tuple[float, float, float]
    cluster_pts_world: np.ndarray

    def color_name(self) -> str:
        """Classify brick into one of the 6 physical DUPLO colors (`lavender`, `brown`, `yellow`, `green`, `white`, `red`)."""
        r, g, b = (float(x) for x in self.mean_rgb)
        rgb_u8 = np.clip(np.round(self.mean_rgb), 0, 255).astype(np.uint8).reshape(1, 1, 3)
        h, s, v = (int(x) for x in cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)[0, 0])
        if s <= 58 and v >= 150:
            return "white"
        # Distinguish Red vs Brown by hue + green/red ratio
        g_over_r = g / max(r, 1.0)
        if (h <= 6 or h >= 162) and g_over_r < 0.36:
            return "red"
        if 6 <= h <= 17 or (h <= 17 and g_over_r >= 0.36):
            return "brown"
        if 18 <= h <= 36:
            return "yellow"
        if 37 <= h <= 84:
            return "green"
        if 85 <= h <= 161:
            return "lavender"
        return "red"

    def display_rgba(self) -> tuple[float, float, float, float]:
        """Return vivid LEGO DUPLO RGBA for the 6 physical brick colors (`lavender`, `brown`, `yellow`, `green`, `white`, `red`)."""
        cname = self.color_name()
        palette = {
            "white": (0.94, 0.96, 0.96, 1.0),  # LEGO White
            "red": (0.88, 0.14, 0.14, 1.0),  # LEGO Bright Red
            "brown": (0.52, 0.27, 0.12, 1.0),  # LEGO Reddish Brown
            "yellow": (0.95, 0.78, 0.12, 1.0),  # LEGO Bright Yellow
            "green": (0.14, 0.74, 0.28, 1.0),  # LEGO Bright Green
            "lavender": (0.68, 0.56, 0.88, 1.0),  # LEGO Medium Lavender
        }
        return palette.get(cname, tuple(float(x) for x in self.spec.rgba))  # type: ignore[return-value]


@dataclass(frozen=True)
class RejectedCluster:
    """Non-brick clutter cluster ignored by the Real2Sim filter (human arm, wire, trash)."""

    category: DistractorCategory
    reason: str
    centroid_world: np.ndarray
    bbox_dims_m: tuple[float, float, float]
    num_points: int
    sample_pts_world: np.ndarray


@dataclass
class Real2SimSceneState:
    """Complete single-frame or tracked output of the Real2Sim perception pipeline."""

    timestamp_s: float
    frame_index: int
    bricks: list[PerceivedBrick]
    plates: dict[str, PerceivedPlate]
    rejected_clusters: list[RejectedCluster]
    raw_point_count: int
    downsampled_point_count: int
    robot_masked_point_count: int
    foreground_point_count: int
    latency_ms: float
    provenance: str = "sensor_fused_point_cloud"

    def to_spawned_blocks(self, source_only: bool = False) -> list[SpawnedBlockInfo]:
        """Convert perceived DUPLO bricks into `SpawnedBlockInfo` records for `SO101InspectionTwin`."""
        selected = [
            b for b in self.bricks if (not source_only or b.support_region in {"source", "table"})
        ]
        if not selected and self.bricks:
            selected = list(self.bricks)
        out: list[SpawnedBlockInfo] = []
        for idx, b in enumerate(selected):
            out.append(
                SpawnedBlockInfo(
                    block_id=f"block_{idx}",
                    spec=b.spec,
                    initial_xy_yaw=b.xy_yaw,
                    has_visual_defect=b.has_visual_defect,
                    initial_T_world_block=b.T_world_block.copy(),
                )
            )
        return out

    def plate_positions_dict(self) -> dict[str, np.ndarray]:
        """Return `{role: pos_world}` for `source`, `good`, and `defective` KALAS plates."""
        return {role: p.pos_world.copy() for role, p in self.plates.items()}

    def distractor_counts_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {"human_arm": 0, "wire": 0, "trash": 0, "robot_arm": 0}
        for rc in self.rejected_clusters:
            counts[rc.category] = counts.get(rc.category, 0) + 1
        return counts

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "frame_index": self.frame_index,
            "latency_ms": round(self.latency_ms, 3),
            "raw_point_count": self.raw_point_count,
            "downsampled_point_count": self.downsampled_point_count,
            "robot_masked_point_count": self.robot_masked_point_count,
            "foreground_point_count": self.foreground_point_count,
            "num_perceived_bricks": len(self.bricks),
            "bricks": [
                {
                    "block_id": b.block_id,
                    "part_name": b.spec.name,
                    "part_id": b.spec.part_id,
                    "color_name": b.color_name(),
                    "stable_pose": b.stable_pose_name,
                    "support_region": b.support_region,
                    "pos_world_m": [round(float(x), 4) for x in b.T_world_block[:3, 3]],
                    "xy_yaw": [round(float(x), 4) for x in b.xy_yaw],
                    "icp_fitness": round(b.icp_fitness, 3),
                    "icp_rmse_mm": round(b.icp_rmse_m * 1000.0, 2),
                    "confidence": round(b.confidence, 3),
                    "has_visual_defect": b.has_visual_defect,
                    "num_points": b.num_points,
                }
                for b in self.bricks
            ],
            "plates": {
                role: {
                    "pos_world_m": [round(float(x), 4) for x in p.pos_world],
                    "inlier_count": p.inlier_count,
                    "from_prior": p.from_prior,
                }
                for role, p in self.plates.items()
            },
            "distractor_counts": self.distractor_counts_by_category(),
            "rejected_clusters": [
                {
                    "category": rc.category,
                    "reason": rc.reason,
                    "centroid_world_m": [round(float(x), 4) for x in rc.centroid_world],
                    "bbox_dims_m": [round(float(x), 4) for x in rc.bbox_dims_m],
                    "num_points": rc.num_points,
                }
                for rc in self.rejected_clusters
            ],
        }


def load_duplo_stable_poses(
    path: Path | None = None,
) -> dict[str, list[StablePoseEntry]]:
    """Load stable resting poses for all 4 LEGO DUPLO bricks from `assets/duplo/stable_poses.json`."""
    json_path = path or (REPO_ROOT / "assets" / "duplo" / "stable_poses.json")
    part_id_to_name = {spec.part_id: spec.name for spec in DUPLO_CATALOG.values()}
    out: dict[str, list[StablePoseEntry]] = {}

    if json_path.exists():
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        for pid, pinfo in doc.get("parts", {}).items():
            entries: list[StablePoseEntry] = []
            for sp in pinfo.get("stable_poses", []):
                rpy = tuple(float(v) for v in sp["rpy_deg"])
                R_loc = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
                entries.append(StablePoseEntry(name=str(sp["name"]), rpy_deg=rpy, R_local=R_loc))
            if entries:
                out[pid] = entries
                if pid in part_id_to_name:
                    out[part_id_to_name[pid]] = entries

    # Ensure every catalog brick has at least upright & upside_down fallback poses
    default_entries = [
        StablePoseEntry("upright", (0.0, 0.0, 0.0), np.eye(3, dtype=np.float64)),
        StablePoseEntry(
            "upside_down",
            (180.0, 0.0, 0.0),
            Rotation.from_euler("xyz", (180.0, 0.0, 0.0), degrees=True).as_matrix(),
        ),
    ]
    for spec in DUPLO_CATALOG.values():
        if spec.name not in out:
            out[spec.name] = default_entries
        if spec.part_id not in out:
            out[spec.part_id] = out[spec.name]
    return out


def voxel_downsample_xyzrgb(
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    voxel_size_m: float = 0.0025,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast vectorized 3D voxel-grid downsampling (<1 ms for 50k points)."""
    pts = np.asarray(points_world, dtype=np.float64)
    cols = np.asarray(colors_rgb, dtype=np.uint8)
    if len(pts) == 0 or voxel_size_m <= 0.0:
        return pts, cols

    mins = pts.min(axis=0)
    coords = np.floor((pts - mins) / voxel_size_m).astype(np.int64)
    # Pack 3D voxel indices into a single 64-bit integer key (20 bits per axis)
    keys = (
        ((coords[:, 0] & 0xFFFFF) << 40)
        | ((coords[:, 1] & 0xFFFFF) << 20)
        | (coords[:, 2] & 0xFFFFF)
    )
    _, first_idx = np.unique(keys, return_index=True)
    first_idx.sort()
    return pts[first_idx], cols[first_idx]


def _is_real_intel_tabletop_packet(packet: dict[str, Any], pts_cam: np.ndarray) -> bool:
    """Detect whether a depth packet comes from the physical Intel D415 640x480 tabletop rig."""
    intr = packet.get("intrinsics") or {}
    if float(intr.get("fx", 0.0)) > 500.0:
        return True
    serial = str(packet.get("serial", ""))
    if len(pts_cam) > 0 and serial != "243622060450" and float(np.median(pts_cam[:, 2])) > 0.60:
        return True
    return False


def real_intel_camera_poses_world() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return `(R_world_cv, p_world_m, K_3x3)` for the calibrated physical `top` and `plate` D415s."""
    R_w_top = REAL_TOP_TO_WORLD_R.copy()
    p_w_top = (-REAL_TOP_ORIGIN_CAM) @ REAL_TOP_TO_WORLD_R.T
    R_w_plate = REAL_TOP_TO_WORLD_R @ REAL_TOP_FROM_PLATE_R
    p_w_plate = (REAL_TOP_FROM_PLATE_T - REAL_TOP_ORIGIN_CAM) @ REAL_TOP_TO_WORLD_R.T
    K_top = np.array(
        [[603.94, 0.0, 313.15], [0.0, 603.02, 235.87], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    K_plate = np.array(
        [[607.79, 0.0, 320.07], [0.0, 606.97, 247.16], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return {
        "top": (R_w_top, p_w_top, K_top),
        "plate": (R_w_plate, p_w_plate, K_plate),
    }


def project_world_obb_to_image_box(
    T_world_block: np.ndarray,
    dims_m: tuple[float, float, float],
    R_world_cv: np.ndarray,
    p_world_cam: np.ndarray,
    K: np.ndarray,
    width: int = 640,
    height: int = 480,
    pad_px: int = 6,
) -> tuple[list[int] | None, float]:
    """Project a 3D brick OBB into `[u0, v0, u1, v1]` pixel bounds and expected `z_cam` depth (m)."""
    hx, hy, hz = 0.5 * float(dims_m[0]), 0.5 * float(dims_m[1]), 0.5 * float(dims_m[2])
    corners_loc = np.array(
        [
            [sx * hx, sy * hy, sz * hz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    corners_w = (corners_loc @ T_world_block[:3, :3].T) + T_world_block[:3, 3]
    pts_cam = (corners_w - p_world_cam[None, :]) @ R_world_cv
    z_center = float(np.dot(T_world_block[:3, 3] - p_world_cam, R_world_cv[:, 2]))
    valid = pts_cam[:, 2] > 0.06
    if not np.any(valid):
        return None, z_center
    pv = pts_cam[valid]
    u = K[0, 0] * (pv[:, 0] / pv[:, 2]) + K[0, 2]
    v = K[1, 1] * (pv[:, 1] / pv[:, 2]) + K[1, 2]
    u0 = max(0, int(math.floor(float(np.min(u)))) - pad_px)
    v0 = max(0, int(math.floor(float(np.min(v)))) - pad_px)
    u1 = min(width, int(math.ceil(float(np.max(u)))) + pad_px)
    v1 = min(height, int(math.ceil(float(np.max(v)))) + pad_px)
    if u1 - u0 < 6 or v1 - v0 < 6:
        return None, z_center
    return [u0, v0, u1, v1], z_center


def _transform_real_intel_cam_to_world(
    pts_cam: np.ndarray,
    cols_rgb: np.ndarray,
    is_plate_cam: bool = False,
) -> np.ndarray:
    """Map physical Intel D415 camera-frame points (`top` or `plate`) into SO-101 world frame."""
    pts_top = (
        (pts_cam @ REAL_TOP_FROM_PLATE_R.T) + REAL_TOP_FROM_PLATE_T if is_plate_cam else pts_cam
    )
    pts_w = (pts_top - REAL_TOP_ORIGIN_CAM) @ REAL_TOP_TO_WORLD_R.T
    if len(pts_w) >= 60 and len(cols_rgb) == len(pts_w):
        hsv = cv2.cvtColor(cols_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        wood_mask = (
            (pts_w[:, 0] > 0.08)
            & (pts_w[:, 0] < 0.44)
            & (pts_w[:, 1] > -0.22)
            & (pts_w[:, 1] < 0.22)
            & (np.abs(pts_w[:, 2]) < 0.025)
            & (hsv[:, 0] >= 10)
            & (hsv[:, 0] <= 24)
            & (hsv[:, 1] >= 75)
            & (hsv[:, 1] <= 185)
        )
        if int(np.sum(wood_mask)) >= 25:
            pts_w = pts_w.copy()
            pts_w[:, 2] -= float(np.median(pts_w[wood_mask, 2]))
    return pts_w


def ingest_teammate_depth_packet(
    packet: dict[str, Any],
    camera: CameraModel,
    T_world_cam_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a `live_depth.py` / `realsense_depth_capture.py` point-cloud packet into world frame.

    Packet format from `scripts/live_depth.py` / `scripts/realsense_depth_capture.py`:
      `{"serial": "...", "intrinsics": {...}, "points": [[x_cam, y_cam, z_cam, r, g, b], ...]}`
    """
    raw_pts = packet.get("points", [])
    if not raw_pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    arr = np.asarray(raw_pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    pts_cam = arr[:, :3]
    if arr.shape[1] >= 6:
        cols = np.clip(np.round(arr[:, 3:6]), 0, 255).astype(np.uint8)
    else:
        cols = np.full((len(pts_cam), 3), 180, dtype=np.uint8)

    valid = np.all(np.isfinite(pts_cam), axis=1) & (pts_cam[:, 2] >= 0.05) & (pts_cam[:, 2] <= 1.50)
    pts_cam = pts_cam[valid]
    cols = cols[valid]
    if len(pts_cam) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    if T_world_cam_override is None and _is_real_intel_tabletop_packet(packet, pts_cam):
        serial = str(packet.get("serial", ""))
        is_plate = (serial == "243622060450") or (
            camera.name in {"realsense_2", "plate"} and serial != "350122060653"
        )
        pts_world = _transform_real_intel_cam_to_world(pts_cam, cols, is_plate_cam=is_plate)
    else:
        T_wc = (
            T_world_cam_override if T_world_cam_override is not None else camera.world_transform()
        )
        pts_world = (pts_cam @ T_wc[:3, :3].T) + T_wc[:3, 3]

    in_ws = np.all(
        (pts_world >= REAL2SIM_WORKSPACE_MIN) & (pts_world <= REAL2SIM_WORKSPACE_MAX), axis=1
    )
    return pts_world[in_ws], cols[in_ws]


def fuse_teammate_top_and_plate_packets(
    top_packet: dict[str, Any] | None,
    plate_packet: dict[str, Any] | None,
    rig: CameraExtrinsicsRig | None = None,
    provisional_transform: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse `top` (`350122060653`) and `plate` (`243622060450`) depth packets into world frame.

    Supports optional `provisional-transform.json` (`top_from_plate_R`, `top_from_plate_t_m`)
    produced by `scripts/board_alignment_pilot.py` / `scripts/merge_brick_preview.py`.
    """
    cam_rig = rig or station_camera_rig()

    def _pkt_is_real(pkt: dict[str, Any] | None) -> bool:
        if pkt is None:
            return False
        if float((pkt.get("intrinsics") or {}).get("fx", 0.0)) > 500.0:
            return True
        raw = pkt.get("points", [])
        if not raw:
            return False
        arr = np.asarray(raw[: min(len(raw), 500)], dtype=np.float64)
        return (
            arr.ndim == 2 and arr.shape[1] >= 3 and _is_real_intel_tabletop_packet(pkt, arr[:, :3])
        )

    is_real_top = _pkt_is_real(top_packet)
    is_real_plate = _pkt_is_real(plate_packet)

    if (is_real_top or is_real_plate) and provisional_transform is None:
        pts_list: list[np.ndarray] = []
        col_list: list[np.ndarray] = []
        if top_packet is not None:
            p_t, c_t = ingest_teammate_depth_packet(top_packet, cam_rig.realsense_1)
            if len(p_t):
                pts_list.append(p_t)
                col_list.append(c_t)
        if plate_packet is not None:
            p_p, c_p = ingest_teammate_depth_packet(plate_packet, cam_rig.realsense_2)
            if len(p_p):
                # When top camera is present on the real table, keep elevated/in-air points from plate
                # camera so grazing-angle bowl-rim shadows do not smear resting bricks inside the bowl.
                if pts_list:
                    elev = p_p[:, 2] > 0.045
                    p_p, c_p = p_p[elev], c_p[elev]
                if len(p_p):
                    pts_list.append(p_p)
                    col_list.append(c_p)
        if not pts_list:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
        return np.vstack(pts_list), np.vstack(col_list)

    T_world_top = cam_rig.realsense_1.world_transform()
    if provisional_transform and "top_from_plate_R" in provisional_transform:
        T_top_plate = np.eye(4, dtype=np.float64)
        T_top_plate[:3, :3] = np.asarray(
            provisional_transform["top_from_plate_R"], dtype=np.float64
        ).reshape(3, 3)
        T_top_plate[:3, 3] = np.asarray(
            provisional_transform["top_from_plate_t_m"], dtype=np.float64
        ).reshape(3)
        T_world_plate = T_world_top @ T_top_plate
    else:
        T_world_plate = cam_rig.realsense_2.world_transform()

    pts_list = []
    col_list = []
    if top_packet is not None:
        p_t, c_t = ingest_teammate_depth_packet(
            top_packet, cam_rig.realsense_1, T_world_cam_override=T_world_top
        )
        if len(p_t):
            pts_list.append(p_t)
            col_list.append(c_t)
    if plate_packet is not None:
        p_p, c_p = ingest_teammate_depth_packet(
            plate_packet, cam_rig.realsense_2, T_world_cam_override=T_world_plate
        )
        if len(p_p):
            pts_list.append(p_p)
            col_list.append(c_p)

    if not pts_list:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(pts_list), np.vstack(col_list)


def unproject_rgbd_workspace(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    camera: CameraModel,
    stride: int = 2,
    z_min_m: float = 0.06,
    z_max_m: float = 1.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject aligned RGB-D frame into the full 3-plate cell workspace."""
    d = np.asarray(depth_m, dtype=np.float64)[::stride, ::stride]
    c = np.asarray(rgb, dtype=np.uint8)[::stride, ::stride]
    rows, cols = d.shape
    u_grid, v_grid = np.meshgrid(
        np.arange(0, camera.width, stride)[:cols],
        np.arange(0, camera.height, stride)[:rows],
    )
    valid = np.isfinite(d) & (d >= z_min_m) & (d <= z_max_m)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    z = d[valid]
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    colors = c[valid]

    fx, fy = camera.K[0, 0], camera.K[1, 1]
    cx, cy = camera.K[0, 2], camera.K[1, 2]
    pts_cam = np.column_stack([(u - cx) * z / fx, (v - cy) * z / fy, z])

    T_wc = camera.world_transform()
    pts_world = (pts_cam @ T_wc[:3, :3].T) + T_wc[:3, 3]
    in_ws = np.all(
        (pts_world >= REAL2SIM_WORKSPACE_MIN) & (pts_world <= REAL2SIM_WORKSPACE_MAX), axis=1
    )
    return pts_world[in_ws], colors[in_ws]


def decode_live_depth_snapshot(
    packet: dict[str, Any],
    rig: CameraExtrinsicsRig | None = None,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a `/snapshot/{top,plate}` JSON packet from `scripts/live_depth.py` into world frame.

    Matches `scripts/patchcore_depth_viewer.py` (`rgb_jpeg_b64`, `depth_png_b64`, `depth_scale_m`).
    """
    cam_rig = rig or station_camera_rig()
    serial = str(packet.get("serial", ""))
    cam = (
        cam_rig.realsense_2
        if serial == str(cam_rig.realsense_2.serial_or_dev)
        else cam_rig.realsense_1
    )
    rgb = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(packet["rgb_jpeg_b64"], validate=True))).convert(
            "RGB"
        ),
        dtype=np.uint8,
    )
    depth_u16 = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(packet["depth_png_b64"], validate=True)))
    )
    if depth_u16.ndim != 2 or depth_u16.dtype != np.uint16:
        raise ValueError("Expected lossless uint16 depth PNG in live_depth snapshot")
    scale_m = float(packet.get("depth_scale_m", 0.001))
    depth_m = depth_u16.astype(np.float64) * scale_m
    intr = packet.get("intrinsics") or {}
    if float(intr.get("fx", 0.0)) > 500.0:
        d = depth_m[::stride, ::stride]
        c = rgb[::stride, ::stride]
        v_grid, u_grid = np.indices(d.shape)
        u_px = (u_grid * stride).astype(np.float64)
        v_px = (v_grid * stride).astype(np.float64)
        valid = np.isfinite(d) & (d >= 0.15) & (d <= 1.35)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
        z = d[valid]
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        ppx = float(intr["ppx"])
        ppy = float(intr["ppy"])
        pts_cam = np.column_stack([(u_px[valid] - ppx) * z / fx, (v_px[valid] - ppy) * z / fy, z])
        cols = c[valid]
        is_plate = serial == "243622060450"
        pts_world = _transform_real_intel_cam_to_world(pts_cam, cols, is_plate_cam=is_plate)
        in_ws = np.all(
            (pts_world >= REAL2SIM_WORKSPACE_MIN) & (pts_world <= REAL2SIM_WORKSPACE_MAX), axis=1
        )
        return pts_world[in_ws], cols[in_ws]
    return unproject_rgbd_workspace(depth_m, rgb, cam, stride=stride)


def _load_ply_xyzrgb(path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load either ASCII 1.0 or binary_little_endian 1.0 PLY (`realsense_depth_capture.py`).

    Returns `(pts, cols, is_binary_sdk)`.
    """
    with path.open("rb") as stream:
        header: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Truncated PLY header in {path}")
            header.append(line.decode("ascii", errors="ignore").strip())
            if header[-1] == "end_header":
                break
        is_binary = any("binary_little_endian" in h for h in header)
        if is_binary:
            count = int(next(x.split()[-1] for x in header if x.startswith("element vertex")))
            dtype = np.dtype(
                [(n, "<f4") for n in "xyz"] + [(n, "u1") for n in ("red", "green", "blue")]
            )
            v = np.fromfile(stream, dtype=dtype, count=count)
            if len(v) == 0:
                return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8), True
            pts = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
            cols = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8)
            return pts, cols, True

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_end = 0
    for idx, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = idx + 1
            break
    pts_list: list[list[float]] = []
    cols_list: list[list[int]] = []
    for line in lines[header_end:]:
        parts = line.strip().split()
        if len(parts) >= 6:
            pts_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
            cols_list.append([int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))])
        elif len(parts) >= 3:
            pts_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
            cols_list.append([180, 180, 180])
    if not pts_list:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8), False
    return (
        np.asarray(pts_list, dtype=np.float64),
        np.clip(cols_list, 0, 255).astype(np.uint8),
        False,
    )


def load_point_cloud_file(
    path: str | Path,
    rig: CameraExtrinsicsRig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a colored 3D point cloud from `.ply` (ASCII or SDK binary), `.npz`, or `.json`."""
    p = Path(path)
    suffix = p.suffix.lower()
    cam_rig = rig or station_camera_rig()
    if suffix == ".ply":
        pts, cols, is_binary_sdk = _load_ply_xyzrgb(p)
        if len(pts) == 0:
            return pts, cols
        fname = p.name.lower()
        is_plate = ("plate" in fname) or (str(cam_rig.realsense_2.serial_or_dev) in fname)
        cam = cam_rig.realsense_2 if is_plate else cam_rig.realsense_1
        # RealSense SDK binary PLY exports (`compare_depth_scans.py` / `realsense_depth_capture.py`)
        # use SDK viewer axes (x, -y, -z) with negative Z
        if is_binary_sdk or float(np.median(pts[:, 2])) < -0.05:
            pts_cam = np.column_stack([pts[:, 0], -pts[:, 1], -pts[:, 2]])
            valid = (
                np.all(np.isfinite(pts_cam), axis=1)
                & (pts_cam[:, 2] >= 0.05)
                & (pts_cam[:, 2] <= 1.50)
            )
            pts_cam, cols = pts_cam[valid], cols[valid]
            T_wc = cam.world_transform()
            return (pts_cam @ T_wc[:3, :3].T) + T_wc[:3, 3], cols
        # Merged review PLYs from `board_alignment_pilot.py` / `merge_brick_preview.py` in camera frame
        if float(np.median(pts[:, 2])) > 0.38 and (
            "merged" in fname or "top" in fname or "plate" in fname
        ):
            T_wc = cam.world_transform()
            return (pts @ T_wc[:3, :3].T) + T_wc[:3, 3], cols
        return pts, cols

    if suffix == ".npz":
        data = np.load(p)
        if "points_world" in data and "colors_rgb" in data:
            return (
                np.asarray(data["points_world"], dtype=np.float64),
                np.asarray(data["colors_rgb"], dtype=np.uint8),
            )
        if "overhead_depth_m" in data and "plate_depth_m" in data:
            p1, c1 = unproject_rgbd_workspace(
                data["overhead_depth_m"], data["overhead_keyframe_rgb"], cam_rig.realsense_1
            )
            p2, c2 = unproject_rgbd_workspace(
                data["plate_depth_m"], data["plate_keyframe_rgb"], cam_rig.realsense_2
            )
            return np.vstack([p1, p2]), np.vstack([c1, c2])
        raise ValueError(f"Unsupported .npz point cloud schema in {p}")

    if suffix == ".json":
        doc = json.loads(p.read_text(encoding="utf-8"))
        if "points_world" in doc:
            return (
                np.asarray(doc["points_world"], dtype=np.float64),
                np.asarray(doc.get("colors_rgb", []), dtype=np.uint8),
            )
        if "rgb_jpeg_b64" in doc and "depth_png_b64" in doc:
            return decode_live_depth_snapshot(doc, rig=cam_rig)
        serial = str(doc.get("serial", ""))
        cam = (
            cam_rig.realsense_2
            if serial == str(cam_rig.realsense_2.serial_or_dev)
            else cam_rig.realsense_1
        )
        return ingest_teammate_depth_packet(doc, cam)

    raise ValueError(f"Unsupported point cloud file extension: {p}")


def save_point_cloud_ply(
    path: str | Path,
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
) -> Path:
    """Write an `(N, 3)` world point cloud and `(N, 3)` uint8 RGB array to an ASCII `.ply` file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points_world, dtype=np.float64)
    cols = np.clip(np.asarray(colors_rgb), 0, 255).astype(np.uint8)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(pts)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for (x, y, z), (r, g, b) in zip(pts, cols, strict=False):
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def filter_robot_arm_points(
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    q_arm_rad: np.ndarray | None = None,
    cal_kin: Any | None = None,
    rig: CameraExtrinsicsRig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strip 3D points belonging to the SO-101 robot arm links, gripper, and static camera housings.

    Returns `(kept_points, kept_colors, removed_robot_points)`.
    """
    if len(points_world) == 0 or q_arm_rad is None:
        return points_world, colors_rgb, np.zeros((0, 3), dtype=np.float64)

    if cal_kin is None:
        from so101_kinematics import CalibratedSO101Kinematics

        cal_kin = CalibratedSO101Kinematics("follower")

    frames_base = cal_kin.frames(np.asarray(q_arm_rad, dtype=np.float64))
    world_frames = {k: (T_WORLD_BASE @ T_bl) for k, T_bl in frames_base.items()}

    # Compute fast world AABB of the 7 main robot links + static side camera housing
    link_entries: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    aabb_mins: list[np.ndarray] = []
    aabb_maxs: list[np.ndarray] = []
    for link_name, (c_loc, h_loc) in _SO101_LINK_OBBS.items():
        T_wl = world_frames.get(link_name)
        if T_wl is None:
            continue
        R_wl = T_wl[:3, :3]
        p_wl = T_wl[:3, 3]
        c_arr = np.asarray(c_loc, dtype=np.float64)
        h_arr = np.asarray(h_loc, dtype=np.float64)
        c_world = (R_wl @ c_arr) + p_wl
        w_half = np.abs(R_wl) @ h_arr
        aabb_mins.append(c_world - w_half)
        aabb_maxs.append(c_world + w_half)
        link_entries.append((R_wl, p_wl, c_arr, h_arr))

    cam_rig = rig or station_camera_rig()
    for cam in (cam_rig.realsense_1, cam_rig.realsense_2):
        T_wc = cam.world_transform()
        R_wc = T_wc[:3, :3]
        p_wc = T_wc[:3, 3]
        c_arr = np.array([0.0, 0.0, -0.014], dtype=np.float64)
        h_arr = np.array([0.060, 0.025, 0.028], dtype=np.float64)
        c_world = (R_wc @ c_arr) + p_wc
        w_half = np.abs(R_wc) @ h_arr
        aabb_mins.append(c_world - w_half)
        aabb_maxs.append(c_world + w_half)
        link_entries.append((R_wc, p_wc, c_arr, h_arr))

    if "Body3" in world_frames:
        T_ww = world_frames["Body3"] @ cam_rig.wrist.T_parent_cvcam
        R_ww = T_ww[:3, :3]
        p_ww = T_ww[:3, 3]
        c_arr = np.zeros(3, dtype=np.float64)
        h_arr = np.array([0.035, 0.035, 0.035], dtype=np.float64)
        w_half = np.abs(R_ww) @ h_arr
        aabb_mins.append(p_ww - w_half)
        aabb_maxs.append(p_ww + w_half)
        link_entries.append((R_ww, p_ww, c_arr, h_arr))

    if not link_entries:
        return points_world, colors_rgb, np.zeros((0, 3), dtype=np.float64)

    g_min = np.min(aabb_mins, axis=0)
    g_max = np.max(aabb_maxs, axis=0)
    in_aabb = np.all((points_world >= g_min) & (points_world <= g_max), axis=1)
    on_robot = np.zeros(len(points_world), dtype=bool)
    if np.any(in_aabb):
        pts_sub = points_world[in_aabb]
        sub_mask = np.zeros(len(pts_sub), dtype=bool)
        for R_wl, p_wl, c_arr, h_arr in link_entries:
            p_local = (pts_sub - p_wl) @ R_wl
            sub_mask |= np.all(np.abs(p_local - c_arr) <= h_arr, axis=1)
        on_robot[in_aabb] = sub_mask

    # Also mask purple 3D-printed PLA robot arm / wrist mount points in the arm envelope
    # (restricted to near-base x < 0.165 m or elevated z > 0.052 m so Lavender bricks in bowls/table are preserved)
    if len(colors_rgb) == len(points_world):
        hsv_r = cv2.cvtColor(colors_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        is_purple_pla = (
            ((points_world[:, 0] < 0.165) | (points_world[:, 2] > 0.052))
            & (points_world[:, 0] < 0.25)
            & (points_world[:, 1] < 0.08)
            & (points_world[:, 2] > 0.010)
            & (hsv_r[:, 0] >= 114)
            & (hsv_r[:, 0] <= 150)
            & (hsv_r[:, 1] >= 45)
        )
        on_robot |= is_purple_pla
    return points_world[~on_robot], colors_rgb[~on_robot], points_world[on_robot]


def localize_and_subtract_support_surfaces(
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    table_z_threshold_m: float = 0.0038,
    max_plate_shift_m: float = 0.048,
) -> tuple[np.ndarray, np.ndarray, dict[str, PerceivedPlate]]:
    """Localize the 3 IKEA KALAS plates (`source`, `good`, `defective`) and subtract support surfaces.

    Refines each plate's XY center from rim/floor point chromaticity & elevation if sufficient
    plate points are visible (including global tabletop discovery when plates are moved in reality),
    otherwise anchors to the calibrated station config (`configs/sim-scene.yaml`).
    """
    plates_cfg = station_plates_config()
    # OpenCV HSV hue bands (0..180) for the 3 pastel KALAS plates:
    # - `source` aquamarine/blue -> Hue ~ 80..110
    # - `good` pastel green -> Hue ~ 48..86
    # - `defective` pastel pink -> Hue ~ 148..180 or 0..10 (with pastel saturation <= 155)
    plate_hue_predicates = {
        "source": lambda h, s: (h >= 86.0) & (h <= 112.0) & (s >= 35.0),
        "good": lambda h, s: (h >= 48.0) & (h < 86.0) & (s >= 28.0),
        "defective": lambda h, s: ((h >= 150.0) | (h <= 6.0)) & (s >= 22.0) & (s <= 125.0),
    }

    if len(points_world) > 0:
        hsv = cv2.cvtColor(colors_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hue = hsv[:, 0].astype(np.float64)
        sat = hsv[:, 1].astype(np.float64)
        val = hsv[:, 2].astype(np.float64)
    else:
        hue = np.zeros(0, dtype=np.float64)
        sat = np.zeros(0, dtype=np.float64)
        val = np.zeros(0, dtype=np.float64)

    localized_plates: dict[str, PerceivedPlate] = {}
    plate_floor_z_map: dict[str, float] = {}
    for role, pspec in plates_cfg.items():
        nom_pos = np.asarray(pspec.pos_m, dtype=np.float64)
        nom_xy = nom_pos[:2].copy()
        if len(points_world) == 0:
            localized_plates[role] = PerceivedPlate(
                role=role,
                pos_world=nom_pos.copy(),
                rgba=pspec.rgba,
                inlier_count=0,
                from_prior=True,
            )
            plate_floor_z_map[role] = SOURCE_PLATE_FLOOR_Z_M
            continue

        r_nom = np.linalg.norm(points_world[:, :2] - nom_xy[None, :], axis=1)
        z = points_world[:, 2]
        hue_ok = plate_hue_predicates[role](hue, sat) & (val >= 50.0)
        if role == "defective":
            hue_ok &= val >= 150.0
        in_plate_zone = (
            (r_nom <= SOURCE_PLATE_OUTER_RADIUS_M + max_plate_shift_m)
            & (z >= 0.0015)
            & (z <= 0.0275)
            & hue_ok
        )
        inlier_pts = points_world[in_plate_zone]
        rim_pts = inlier_pts[(inlier_pts[:, 2] >= 0.0085) & (inlier_pts[:, 2] <= 0.0255)]
        pos_w = nom_pos.copy()
        from_prior = True

        if len(rim_pts) >= 40:
            # Known-radius (R_rim = 0.0785 m) Gauss-Newton circle-center fit invariant to one-sided camera density
            est_xy = nom_xy.copy()
            r_target = 0.0785
            xy_pts = rim_pts[:, :2]
            angs = np.arctan2(xy_pts[:, 1] - nom_xy[1], xy_pts[:, 0] - nom_xy[0])
            octants = np.floor((angs + math.pi) / (0.25 * math.pi)).astype(int) % 8
            if len(np.unique(octants)) >= 5:
                for _ in range(8):
                    vec = xy_pts - est_xy[None, :]
                    dists = np.maximum(np.linalg.norm(vec, axis=1), 1e-6)
                    res = dists - r_target
                    u_dir = vec / dists[:, None]
                    w = np.where(np.abs(res) <= 0.008, 1.0, 0.008 / np.maximum(np.abs(res), 1e-6))
                    J = -u_dir
                    H = J.T @ (J * w[:, None]) + 1e-3 * np.eye(2)
                    g = J.T @ (w * res)
                    step = -np.linalg.solve(H, g)
                    est_xy = est_xy + np.clip(step, -0.015, 0.015)
                shift = float(np.linalg.norm(est_xy - nom_xy))
                final_dists = np.linalg.norm(xy_pts - est_xy[None, :], axis=1)
                med_rad_err = float(np.median(np.abs(final_dists - r_target)))
                angs_est = np.arctan2(xy_pts[:, 1] - est_xy[1], xy_pts[:, 0] - est_xy[0])
                octs_est = np.floor((angs_est + math.pi) / (0.25 * math.pi)).astype(int) % 8
                if (
                    shift <= max_plate_shift_m
                    and med_rad_err <= 0.014
                    and len(np.unique(octs_est)) >= 5
                ):
                    pos_w = np.array([float(est_xy[0]), float(est_xy[1]), 0.0], dtype=np.float64)
                    from_prior = False

        # Global tabletop plate discovery when a plate is moved outside its nominal prior zone in reality
        if from_prior:
            cand_mask = (
                hue_ok
                & (z >= -0.008)
                & (z <= 0.028)
                & (
                    points_world[:, 0] >= 0.105
                    if role == "defective"
                    else points_world[:, 0] >= 0.175
                )
                & (points_world[:, 0] <= 0.475)
                & (np.abs(points_world[:, 1]) <= 0.245)
                & ~((points_world[:, 0] < 0.220) & (points_world[:, 1] > 0.195))
            )
            if role == "source":
                cand_mask &= sat >= 100.0
            elif role == "good":
                cand_mask &= sat >= 80.0
            cand_pts = points_world[cand_mask]
            if len(cand_pts) >= 28:
                db = DBSCAN(eps=0.032, min_samples=10).fit(cand_pts[:, :2])
                best_lbl = -1
                best_n = 0
                for lbl in set(db.labels_):
                    if lbl < 0:
                        continue
                    c_sub = cand_pts[db.labels_ == lbl]
                    span_xy = np.percentile(c_sub[:, :2], 95, axis=0) - np.percentile(
                        c_sub[:, :2], 5, axis=0
                    )
                    if (
                        len(c_sub) > best_n
                        and 0.065 <= float(max(span_xy)) <= 0.230
                        and float(min(span_xy)) >= 0.050
                    ):
                        best_n = len(c_sub)
                        best_lbl = int(lbl)
                if best_lbl >= 0 and best_n >= 28:
                    cluster_pts = cand_pts[db.labels_ == best_lbl]
                    lo = np.percentile(cluster_pts[:, :2], 3, axis=0)
                    hi = np.percentile(cluster_pts[:, :2], 97, axis=0)
                    center_xy = 0.5 * (lo + hi)
                    pos_w = np.array(
                        [float(center_xy[0]), float(center_xy[1]), 0.0], dtype=np.float64
                    )
                    inlier_pts = cluster_pts
                    from_prior = False

        r_inner = np.linalg.norm(points_world[:, :2] - pos_w[:2][None, :], axis=1)
        inner_z = z[(r_inner <= 0.072) & hue_ok & (z >= -0.005) & (z <= 0.020)]
        if len(inner_z) >= 15 and not from_prior:
            plate_floor_z_map[role] = float(
                np.clip(np.percentile(inner_z, 25), SOURCE_PLATE_FLOOR_Z_M, 0.0105)
            )
        else:
            plate_floor_z_map[role] = SOURCE_PLATE_FLOOR_Z_M

        localized_plates[role] = PerceivedPlate(
            role=role,
            pos_world=pos_w,
            rgba=pspec.rgba,
            inlier_count=int(len(inlier_pts)),
            from_prior=from_prior,
        )

    # Enforce physical non-overlap between localized bowls on the tabletop (min center-to-center = 164 mm)
    min_bowl_dist_m = 0.164
    roles_order = ("source", "good", "defective")
    for _ in range(4):
        for i_r in range(len(roles_order)):
            for j_r in range(i_r + 1, len(roles_order)):
                pa = localized_plates[roles_order[i_r]]
                pb = localized_plates[roles_order[j_r]]
                diff = pb.pos_world[:2] - pa.pos_world[:2]
                d = float(np.linalg.norm(diff))
                if 1e-4 < d < min_bowl_dist_m:
                    push = 0.5 * (min_bowl_dist_m - d) * (diff / d)
                    pa.pos_world[:2] -= push
                    pb.pos_world[:2] += push

    if len(points_world) == 0:
        return points_world, colors_rgb, localized_plates

    z = points_world[:, 2]
    keep = z > table_z_threshold_m

    # Subtract natural wooden tabletop grain (warm pine wood at z <= 11.5 mm outside plates)
    is_wood_table = (
        (z <= 0.0115)
        & (hue >= 10.0)
        & (hue <= 24.0)
        & (sat >= 65.0)
        & (sat <= 188.0)
        & (val >= 70.0)
        & (val <= 208.0)
    )
    for plate in localized_plates.values():
        r_p = np.linalg.norm(points_world[:, :2] - plate.pos_world[:2][None, :], axis=1)
        is_wood_table &= r_p > SOURCE_PLATE_OUTER_RADIUS_M
    keep &= ~is_wood_table

    for role, plate in localized_plates.items():
        pxy = plate.pos_world[:2]
        p_floor_z = plate_floor_z_map.get(role, SOURCE_PLATE_FLOOR_Z_M)
        r_p = np.linalg.norm(points_world[:, :2] - pxy[None, :], axis=1)
        in_flat = r_p <= 0.074
        in_rim = (r_p > 0.074) & (r_p <= SOURCE_PLATE_OUTER_RADIUS_M + 0.004)
        hue_ok = plate_hue_predicates[role](hue, sat)

        is_plate_floor = in_flat & (z <= p_floor_z + 0.0032)
        is_plate_rim = in_rim & (z <= 0.0268) & (hue_ok | (z <= p_floor_z + 0.010))
        # Preserve elevated cyan/blue DUPLO bricks sitting inside the aquamarine/blue source plate
        # (whose top/side surfaces sit > 8.0 mm above p_floor_z inside the flat inner bowl).
        is_plate_plastic = (
            (in_flat & (z <= p_floor_z + 0.0080)) | (in_rim & (z <= 0.0272))
        ) & hue_ok
        keep = keep & (~is_plate_floor) & (~is_plate_rim) & (~is_plate_plastic)

    return points_world[keep], colors_rgb[keep], localized_plates


def _support_region_and_floor_z(
    cx: float,
    cy: float,
    plates: dict[str, PerceivedPlate],
) -> tuple[str, float]:
    """Determine which KALAS plate (`source`, `good`, `defective`) or `table` supports `(cx, cy)`."""
    for role, plate in plates.items():
        d = float(math.hypot(cx - float(plate.pos_world[0]), cy - float(plate.pos_world[1])))
        if d <= SOURCE_PLATE_OUTER_RADIUS_M:
            return role, SOURCE_PLATE_FLOOR_Z_M
    return "table", 0.0


def _cluster_3d_linearity(pts: np.ndarray) -> float:
    """Compute 3D PCA linearity `(lambda_1 - lambda_2) / lambda_1` in `[0, 1]`."""
    if len(pts) < 6:
        return 0.0
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1, len(pts) - 1)
    eigvals = np.linalg.eigvalsh(cov)
    l1 = float(max(eigvals[-1], 1e-12))
    l2 = float(max(eigvals[-2], 0.0))
    return float(np.clip((l1 - l2) / l1, 0.0, 1.0))


def _match_duplo_color_and_defect(
    cluster_cols: np.ndarray,
    allow_real_palette: bool = False,
) -> tuple[BlockSpec, float, float, float, bool]:
    """Match cluster RGB to the 4 DUPLO catalog colors while detecting surface defect blemishes.

    When `allow_real_palette=True` (for clusters inside localized KALAS plates or near the gripper),
    also recognizes physical white and orange/brown DUPLO bricks present in the real Intel cell.
    Returns `(best_spec, min_chrom_dist, median_sat, median_val, has_visual_defect)`.
    """
    hsv = cv2.cvtColor(cluster_cols.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat = hsv[:, 1].astype(np.float64)
    val = hsv[:, 2].astype(np.float64)
    hue = hsv[:, 0].astype(np.float64)
    med_sat = float(np.median(sat))
    med_val = float(np.median(val))

    # Exclude bright magenta/white defect decal pixels when computing dominant brick body color
    is_magenta_blemish = (hue >= 140.0) & (hue <= 172.0) & (sat >= 110.0)
    is_white_core = (sat < 35.0) & (val > 215.0)
    body_mask = ~(is_magenta_blemish | is_white_core)
    body_cols = cluster_cols[body_mask] if int(np.sum(body_mask)) >= 12 else cluster_cols

    cols_norm = body_cols.astype(np.float64) / 255.0
    med_rgb = np.median(cols_norm, axis=0)
    obs_c = med_rgb / max(float(np.sum(med_rgb)), 1e-6)

    med_hue = (
        float(np.median(hsv[body_mask, 0]))
        if int(np.sum(body_mask)) >= 12
        else float(np.median(hue))
    )

    best_spec: BlockSpec | None = None
    best_dist = 1e9
    for spec in DUPLO_CATALOG.values():
        spec_rgb = np.asarray(spec.rgba[:3], dtype=np.float64)
        exp_c = spec_rgb / max(float(np.sum(spec_rgb)), 1e-6)
        chrom_d = float(np.linalg.norm(obs_c - exp_c))
        # Compute expected OpenCV hue (0..180) directly from `spec.rgba[:3]`
        spec_u8 = np.clip(np.round(spec_rgb * 255.0), 0, 255).astype(np.uint8).reshape(1, 1, 3)
        exp_hue = float(cv2.cvtColor(spec_u8, cv2.COLOR_RGB2HSV)[0, 0, 0])
        dh = abs(med_hue - exp_hue)
        dh = min(dh, 180.0 - dh)
        # Add hue penalty if hue lies outside the brick's color family (e.g. skin tone / orange board on table)
        hue_pen = 0.18 if dh > (14.5 if allow_real_palette else 9.5) else 0.0

        total_d = chrom_d + hue_pen
        if total_d < best_dist:
            best_dist = total_d
            best_spec = spec

    assert best_spec is not None
    if allow_real_palette:
        g_over_r = float(med_rgb[1]) / max(float(med_rgb[0]), 1e-6)
        # 1. White DUPLO brick
        if med_val >= 150.0 and med_sat <= 60.0:
            best_dist = min(best_dist, 0.08)
        # 2. Red DUPLO brick
        elif (med_hue <= 6.5 or med_hue >= 162.0) and med_sat >= 75.0 and g_over_r < 0.36:
            best_dist = min(best_dist, 0.075)
            best_spec = DUPLO_CATALOG["duplo_2x2_3437"]
        # 3. Brown DUPLO brick (warm reddish-brown / dark orange-brown)
        elif (6.0 <= med_hue <= 18.0 and med_sat >= 75.0 and med_val >= 55.0) or (
            med_hue <= 18.0 and g_over_r >= 0.36 and med_sat >= 75.0 and med_val >= 55.0
        ):
            best_dist = min(best_dist, 0.085)
            best_spec = DUPLO_CATALOG["duplo_1x2x2_76371"]
        # 4. Yellow DUPLO brick
        elif 18.0 < med_hue <= 37.0 and med_sat >= 75.0 and med_val >= 95.0:
            best_dist = min(best_dist, 0.075)
            best_spec = DUPLO_CATALOG["duplo_1x2x2_76371"]
        # 5. Green DUPLO brick
        elif 37.0 < med_hue <= 85.0 and med_sat >= 60.0 and med_val >= 60.0:
            best_dist = min(best_dist, 0.08)
            best_spec = DUPLO_CATALOG["duplo_2x3_curved_top_2302"]
        # 6. Lavender DUPLO brick (includes purple-lilac and camera white-balance-shifted periwinkle in bowl)
        elif 88.0 <= med_hue <= 161.0 and med_sat >= 28.0 and med_val >= 70.0:
            best_dist = min(best_dist, 0.08)
            best_spec = DUPLO_CATALOG["duplo_2x2_3437"]

    has_defect = bool(int(np.sum(is_magenta_blemish)) >= 4)
    return best_spec, best_dist, med_sat, med_val, has_defect


def _fit_6d_stable_pose_to_brick_cluster(
    cluster_pts: np.ndarray,
    cluster_cols: np.ndarray,
    support_z_m: float,
    stable_poses_map: dict[str, list[StablePoseEntry]],
    lock_table_resting: bool = True,
) -> tuple[BlockSpec, str, np.ndarray, float, float, bool]:
    """Jointly infer `BlockSpec`, `stable_pose_name`, and 6D `T_world_block` via surface ICP."""
    pts_xy = cluster_pts[:, :2].astype(np.float32)
    (cx, cy), (w_rect, h_rect), angle_deg = cv2.minAreaRect(pts_xy)
    yaw_base = math.radians(angle_deg)
    if w_rect < h_rect:
        w_rect, h_rect = h_rect, w_rect
        yaw_base += 0.5 * math.pi

    obs_major = max(float(w_rect), float(h_rect))
    obs_minor = min(float(w_rect), float(h_rect))
    z_top = float(np.percentile(cluster_pts[:, 2], 96))
    obs_h = max(z_top - support_z_m, 0.012)

    color_spec, _, _, _, has_defect = _match_duplo_color_and_defect(
        cluster_cols, allow_real_palette=True
    )

    # Score candidate (spec, stable_pose) pairs by 3D bounding-box compatibility + color match
    cols_norm = cluster_cols.astype(np.float64) / 255.0
    med_rgb = np.median(cols_norm, axis=0)
    obs_c = med_rgb / max(float(np.sum(med_rgb)), 1e-6)

    candidate_hyps: list[tuple[float, BlockSpec, StablePoseEntry]] = []
    for spec in DUPLO_CATALOG.values():
        spec_rgb = np.asarray(spec.rgba[:3], dtype=np.float64)
        exp_c = spec_rgb / max(float(np.sum(spec_rgb)), 1e-6)
        chrom_err = float(np.linalg.norm(obs_c - exp_c))
        if spec.name == color_spec.name:
            chrom_err *= 0.65
        # In the real cell, 2x4 bricks (`3011`) and 2x2 bricks (`3437`) occur in multiple LEGO colors
        # (e.g. Red 2x4, White 2x4, Cyan 2x2). Let measured 3D geometry dominate when footprint is unambiguous.
        if obs_major >= 0.051 or obs_major <= 0.038:
            chrom_err = min(chrom_err, 0.16)

        for sp_entry in stable_poses_map.get(spec.name, []):
            rot_dims = np.abs(sp_entry.R_local) @ np.asarray(spec.dims_m, dtype=np.float64)
            exp_maj = float(max(rot_dims[0], rot_dims[1]))
            exp_min = float(min(rot_dims[0], rot_dims[1]))
            exp_h = float(rot_dims[2])
            geom_err = (
                2.4 * abs(obs_h - exp_h)
                + 2.2 * abs(obs_major - exp_maj)
                + 1.5 * abs(obs_minor - exp_min)
            )
            # Slight prior preference for upright/upside-down unless height clearly indicates side pose
            pose_prior = (
                0.0
                if sp_entry.name == "upright"
                else (0.012 if sp_entry.name == "upside_down" else 0.025)
            )
            cost = chrom_err + 12.0 * geom_err + pose_prior
            candidate_hyps.append((cost, spec, sp_entry))

    candidate_hyps.sort(key=lambda item: item[0])
    # Only evaluate second hypothesis if its cost is nearly tied with the best hypothesis
    hyps_to_run = [candidate_hyps[0]]
    if len(candidate_hyps) > 1 and (candidate_hyps[1][0] - candidate_hyps[0][0]) < 0.045:
        hyps_to_run.append(candidate_hyps[1])

    pts_icp = cluster_pts
    if len(pts_icp) > 220:
        idx_sub = np.linspace(0, len(pts_icp) - 1, 220, dtype=int)
        pts_icp = pts_icp[idx_sub]

    best_spec = hyps_to_run[0][1]
    best_pose_name = hyps_to_run[0][2].name
    best_T = np.eye(4, dtype=np.float64)
    best_fit = -1.0
    best_rmse = 1e9
    best_score = -1e9

    for prior_cost, spec, sp_entry in hyps_to_run:
        rot_dims = np.abs(sp_entry.R_local) @ np.asarray(spec.dims_m, dtype=np.float64)
        lx_r, wy_r, hz_r = float(rot_dims[0]), float(rot_dims[1]), float(rot_dims[2])
        yaw0 = yaw_base
        if (lx_r < wy_r and w_rect >= h_rect) or (lx_r > wy_r and w_rect < h_rect):
            yaw0 += 0.5 * math.pi
        z_center = (
            support_z_m + 0.5 * hz_r if lock_table_resting else float(np.median(cluster_pts[:, 2]))
        )

        # Because a cuboid's surface is invariant under 180-deg yaw (and 90-deg yaw when lx_r == wy_r),
        # a single ICP seed at yaw0 finds the exact optimum without redundant quarter-turn solves!
        R_world = Rotation.from_euler("z", yaw0).as_matrix() @ sp_entry.R_local
        T0 = np.eye(4, dtype=np.float64)
        T0[:3, :3] = R_world
        T0[:3, 3] = [float(cx), float(cy), z_center]
        T_opt, fit, rmse = refine_cuboid_icp(
            pts_icp, T0, spec.dims_m, max_iters=10, lock_table_resting=lock_table_resting
        )
        score = fit - 20.0 * rmse - 0.35 * prior_cost
        if score > best_score:
            best_score = score
            best_spec = spec
            best_pose_name = sp_entry.name
            best_T = T_opt
            best_fit = fit
            best_rmse = rmse

    # Disambiguate C1 asymmetric slope direction for `duplo_2x3_curved_top_2302` (+Y is tall 2-stud end)
    if best_spec.symmetry_group == "C1" and best_pose_name == "upright":
        p_local = (cluster_pts - best_T[:3, 3]) @ best_T[:3, :3]
        pos_y = p_local[p_local[:, 1] > 0.005, 2]
        neg_y = p_local[p_local[:, 1] < -0.005, 2]
        if len(pos_y) >= 5 and len(neg_y) >= 5:
            if float(np.percentile(neg_y, 85)) > float(np.percentile(pos_y, 85)) + 0.0012:
                best_T[:3, :3] = Rotation.from_euler("z", math.pi).as_matrix() @ best_T[:3, :3]

    return best_spec, best_pose_name, best_T, best_fit, best_rmse, has_defect


def _segment_plate_interior_bricks_by_color(
    pts: np.ndarray, cols: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Segment touching multi-color and same-color DUPLO bricks inside a localized plate.

    Supports all 6 physical brick colors: `white`, `red`, `brown`, `yellow`, `green`, and `lavender`.
    """
    if len(pts) < 7:
        return []
    hsv = cv2.cvtColor(cols.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = hsv[:, 0].astype(np.float64)
    s = hsv[:, 1].astype(np.float64)
    v = hsv[:, 2].astype(np.float64)
    r_ch = cols[:, 0].astype(np.float64)
    g_ch = cols[:, 1].astype(np.float64)
    g_over_r = g_ch / np.maximum(r_ch, 1.0)

    fam = np.full(len(pts), -1, dtype=np.int32)
    # 0: white DUPLO brick
    fam[(s < 58.0) & (v >= 150.0)] = 0
    # 1: red DUPLO brick (low G/R ratio separates bright red from reddish-brown)
    fam[(s >= 75.0) & (v >= 65.0) & (((h <= 6.5) & (g_over_r < 0.36)) | (h >= 162.0))] = 1
    # 2: brown DUPLO brick (warm reddish-brown / dark orange-brown)
    fam[
        (s >= 75.0) & (v >= 55.0) & (((h > 6.5) & (h < 18.5)) | ((h <= 6.5) & (g_over_r >= 0.36)))
    ] = 2
    # 3: yellow DUPLO brick
    fam[(s >= 75.0) & (v >= 95.0) & (h >= 18.5) & (h <= 38.0)] = 3
    # 4: green DUPLO brick
    fam[(s >= 60.0) & (v >= 60.0) & (h > 38.0) & (h <= 86.0)] = 4
    # 5: high-saturation periwinkle / camera-shifted lavender inside blue bowl
    fam[(s >= 145.0) & (v >= 65.0) & (h >= 90.0) & (h <= 116.0)] = 5
    # 6: pastel lavender / lilac / violet DUPLO brick
    fam[(s >= 28.0) & (s < 145.0) & (v >= 75.0) & (h >= 98.0) & (h <= 161.0)] = 6

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fid in range(7):
        mask = fam == fid
        if int(np.sum(mask)) < 7:
            continue
        f_pts = pts[mask]
        f_cols = cols[mask]
        eps = 0.0115 if fid == 5 else 0.0145
        labels = DBSCAN(eps=eps, min_samples=4).fit_predict(f_pts)
        for lbl in sorted(set(labels)):
            if lbl < 0:
                continue
            sub_pts = f_pts[labels == lbl]
            sub_cols = f_cols[labels == lbl]
            if len(sub_pts) < 7:
                continue
            _, (w_r, h_r), _ = cv2.minAreaRect(sub_pts[:, :2].astype(np.float32))
            maj = max(float(w_r), float(h_r))
            min_s = min(float(w_r), float(h_r))
            # Split touching same-color bricks only if footprint strictly exceeds a single 2x4 brick (64x32 mm + margin)
            if (maj > 0.076 or min_s > 0.047) and len(sub_pts) >= 16:
                k_split = 2 if maj <= 0.095 else 3
                km = KMeans(n_clusters=k_split, n_init=5, random_state=0).fit(sub_pts[:, :2])
                for k_id in range(k_split):
                    k_mask = km.labels_ == k_id
                    if int(np.sum(k_mask)) >= 7:
                        out.append((sub_pts[k_mask], sub_cols[k_mask]))
            else:
                out.append((sub_pts, sub_cols))
    return out


def _split_touching_duplo_cluster(
    c_pts: np.ndarray, c_cols: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split touching multi-brick clusters by catalog color or oversize footprint without splitting a single 2x4."""
    pts_xy = c_pts[:, :2].astype(np.float32)
    _, (w_r, h_r), _ = cv2.minAreaRect(pts_xy)
    max_span = max(float(w_r), float(h_r))
    min_span = min(float(w_r), float(h_r))

    cat_chroms = []
    for spec in DUPLO_CATALOG.values():
        rgb = np.asarray(spec.rgba[:3], dtype=np.float64)
        cat_chroms.append(rgb / max(float(np.sum(rgb)), 1e-6))
    cat_mat = np.vstack(cat_chroms)

    cols_f = c_cols.astype(np.float64) / 255.0
    sums = np.maximum(np.sum(cols_f, axis=1, keepdims=True), 1e-6)
    chrom = cols_f / sums
    dists = np.linalg.norm(chrom[:, None, :] - cat_mat[None, :, :], axis=2)
    nearest_cat = np.argmin(dists, axis=1)
    min_dist = np.min(dists, axis=1)

    valid_groups = [
        k
        for k in range(len(cat_chroms))
        if int(np.sum((nearest_cat == k) & (min_dist < 0.19))) >= 45
    ]
    if len(valid_groups) >= 2:
        return _split_touching_cluster_if_needed(c_pts, c_cols)

    # Single-color cluster: only split if strictly larger than a single 2x4 brick (64x32 mm + margin)
    if max_span <= 0.076 and min_span <= 0.046:
        return [(c_pts, c_cols)]
    return _split_touching_cluster_if_needed(c_pts, c_cols)


def perceive_bricks_and_reject_clutter(
    fg_pts: np.ndarray,
    fg_cols: np.ndarray,
    plates: dict[str, PerceivedPlate],
    stable_poses_map: dict[str, list[StablePoseEntry]] | None = None,
    ee_pos_world: np.ndarray | None = None,
) -> tuple[list[PerceivedBrick], list[RejectedCluster]]:
    """5-Gate distractor rejection + 6D DUPLO brick localization over foreground point cloud.

    Explicitly filters out and logs:
      1. `human_arm` (elevated/hovering or skin-toned/large boundary-intruding arm/hand clusters)
      2. `wire` (thin elongated cables/wires with high 3D PCA linearity or aspect ratio)
      3. `trash` (achromatic/non-catalog color clutter, oversize cups/boxes, undersize scraps, poor ICP fit)
    While localizing DUPLO bricks resting inside plates, on the table, or lifted in-air / in-gripper.
    """
    if len(fg_pts) < 10:
        return [], []

    poses_map = stable_poses_map or load_duplo_stable_poses()

    # Separate points inside any localized plate (r <= 0.084 m, 0.004 <= z_rel <= 0.060 m)
    # so dense touching multi-color piles inside the source plate are segmented by color family first.
    in_plate_mask = np.zeros(len(fg_pts), dtype=bool)
    for p in plates.values():
        r_p = np.linalg.norm(fg_pts[:, :2] - p.pos_world[:2], axis=1)
        z_rel_p = fg_pts[:, 2] - (float(p.pos_world[2]) + SOURCE_PLATE_FLOOR_Z_M)
        in_plate_mask |= (r_p <= 0.084) & (z_rel_p >= 0.004) & (z_rel_p <= 0.060)

    candidate_clusters: list[tuple[np.ndarray, np.ndarray]] = []
    if int(np.sum(in_plate_mask)) >= 7:
        plate_pts = fg_pts[in_plate_mask]
        plate_cols = fg_cols[in_plate_mask]
        if len(plate_pts) < 600:
            # Sparse/stride-4 real camera cloud: segment directly by LEGO color family + 3D DBSCAN
            candidate_clusters.extend(
                _segment_plate_interior_bricks_by_color(plate_pts, plate_cols)
            )
        else:
            # Dense synthetic or stride-1/2 cloud: check spatial clusters first, then split touching bricks
            p_labels = DBSCAN(eps=0.0075, min_samples=6).fit_predict(plate_pts)
            for lbl in sorted(set(p_labels)):
                if lbl < 0:
                    continue
                sub_pts = plate_pts[p_labels == lbl]
                sub_cols = plate_cols[p_labels == lbl]
                if len(sub_pts) < 10:
                    continue
                _, (w_r, h_r), _ = cv2.minAreaRect(sub_pts[:, :2].astype(np.float32))
                maj = max(float(w_r), float(h_r))
                min_s = min(float(w_r), float(h_r))
                if maj > 0.068 or min_s > 0.042:
                    seg = _segment_plate_interior_bricks_by_color(sub_pts, sub_cols)
                    if seg:
                        candidate_clusters.extend(seg)
                    else:
                        candidate_clusters.extend(_split_touching_duplo_cluster(sub_pts, sub_cols))
                else:
                    seg = [
                        (sp, sc)
                        for sp, sc in _segment_plate_interior_bricks_by_color(sub_pts, sub_cols)
                        if len(sp) >= 18
                    ]
                    if len(seg) >= 2:
                        candidate_clusters.extend(seg)
                    else:
                        candidate_clusters.append((sub_pts, sub_cols))

    out_pts = fg_pts[~in_plate_mask]
    out_cols = fg_cols[~in_plate_mask]
    if len(out_pts) >= 12:
        labels = DBSCAN(eps=0.0078, min_samples=7).fit_predict(out_pts)
        raw_clusters = [
            (out_pts[labels == lbl], out_cols[labels == lbl])
            for lbl in sorted(set(labels))
            if lbl >= 0 and int(np.sum(labels == lbl)) >= 14
        ]
        for r_pts, r_cols in raw_clusters:
            c_xyz = r_pts.mean(axis=0)
            _, floor_z = _support_region_and_floor_z(float(c_xyz[0]), float(c_xyz[1]), plates)
            z_bot = float(np.percentile(r_pts[:, 2], 5)) - floor_z
            z_top = float(np.percentile(r_pts[:, 2], 96)) - floor_z
            _, chrom_d, med_sat, _, _ = _match_duplo_color_and_defect(r_cols)
            if z_bot <= 0.018 and 0.013 <= z_top <= 0.049 and med_sat >= 55.0 and chrom_d <= 0.22:
                candidate_clusters.extend(_split_touching_duplo_cluster(r_pts, r_cols))
            else:
                candidate_clusters.append((r_pts, r_cols))

    bricks: list[PerceivedBrick] = []
    rejected: list[RejectedCluster] = []

    for c_pts, c_cols in candidate_clusters:
        if len(c_pts) < 7:
            continue

        centroid = c_pts.mean(axis=0)
        region, floor_z = _support_region_and_floor_z(
            float(centroid[0]), float(centroid[1]), plates
        )
        in_plate = region in ("source", "good", "defective")
        min_pts_req = 7 if in_plate else 16
        if len(c_pts) < min_pts_req:
            continue

        pts_xy = c_pts[:, :2].astype(np.float32)
        _, (w_r, h_r), _ = cv2.minAreaRect(pts_xy)
        major_span = max(float(w_r), float(h_r))
        minor_span = min(float(w_r), float(h_r))
        z_bottom_rel = max(0.0, float(np.percentile(c_pts[:, 2], 5)) - floor_z)
        z_top_rel = max(0.002, float(np.percentile(c_pts[:, 2], 96)) - floor_z)
        height_span = max(0.002, z_top_rel - z_bottom_rel)
        bbox_dims = (major_span, minor_span, height_span)

        sample_idx = np.linspace(0, len(c_pts) - 1, min(28, len(c_pts)), dtype=int)
        sample_pts = c_pts[sample_idx].copy()

        linearity = _cluster_3d_linearity(c_pts)
        aspect_2d = major_span / max(minor_span, 1e-4)
        _, chrom_dist, med_sat, med_val, _ = _match_duplo_color_and_defect(
            c_cols, allow_real_palette=in_plate
        )

        # Reject peripheral tabletop clutter outside the active plate/arm manipulation zone
        dist_to_nearest_plate = min(
            float(np.linalg.norm(centroid[:2] - p.pos_world[:2])) for p in plates.values()
        )
        near_ee = (
            ee_pos_world is not None and float(np.linalg.norm(centroid - ee_pos_world)) <= 0.075
        )
        if not in_plate and not near_ee:
            # Stickered laptop at right (x < 0.18, y > 0.10 or x < 0.245, y > 0.155) or peripheral objects far from plates
            if (
                (float(centroid[0]) < 0.180 and float(centroid[1]) > 0.100)
                or (float(centroid[0]) < 0.245 and float(centroid[1]) > 0.155)
                or (dist_to_nearest_plate > 0.195 and float(centroid[0]) > 0.38)
            ):
                rejected.append(
                    RejectedCluster(
                        category="trash",
                        reason=f"peripheral_clutter(x={centroid[0]:.2f},y={centroid[1]:.2f})",
                        centroid_world=centroid,
                        bbox_dims_m=bbox_dims,
                        num_points=len(c_pts),
                        sample_pts_world=sample_pts,
                    )
                )
                continue

        # Check if cluster is a compact, LEGO-colored DUPLO brick being lifted in mid-air / gripper
        is_plausible_lifted_brick = (
            (0.022 < z_bottom_rel <= 0.220)
            and (0.016 <= major_span <= 0.068)
            and (0.012 <= minor_span <= 0.042)
            and (0.010 <= height_span <= 0.042)
            and (aspect_2d <= 2.8)
            and (linearity <= 0.75)
            and (
                near_ee
                or (dist_to_nearest_plate <= 0.14 and med_sat >= 85.0 and chrom_dist <= 0.125)
            )
        )

        # Gate 1: Support-surface grounding & elevation gate (Human arm / hovering hand)
        is_elevated = (z_bottom_rel > 0.044 or z_top_rel > 0.054) and not is_plausible_lifted_brick
        is_large_intrusion = major_span > 0.082 and (z_top_rel > 0.032 or z_bottom_rel > 0.012)
        if is_elevated or is_large_intrusion:
            rejected.append(
                RejectedCluster(
                    category="human_arm",
                    reason=(
                        f"elevated_or_intruding_arm(z_bot={z_bottom_rel * 1000:.1f}mm,"
                        f"z_top={z_top_rel * 1000:.1f}mm,span={major_span * 1000:.1f}mm)"
                    ),
                    centroid_world=centroid,
                    bbox_dims_m=bbox_dims,
                    num_points=len(c_pts),
                    sample_pts_world=sample_pts,
                )
            )
            continue

        # Gate 2: 3D PCA linearity & 2D aspect ratio gate (Wires / USB cables)
        min_span_thresh = 0.0085 if in_plate else 0.0115
        aspect_thresh = 3.85 if in_plate else 3.35
        lin_thresh = 0.94 if (in_plate and minor_span >= 0.016) else (0.88 if in_plate else 0.82)
        if minor_span < min_span_thresh or aspect_2d > aspect_thresh or linearity > lin_thresh:
            rejected.append(
                RejectedCluster(
                    category="wire",
                    reason=(
                        f"elongated_wire_geometry(minor={minor_span * 1000:.1f}mm,"
                        f"aspect={aspect_2d:.2f},linearity={linearity:.2f})"
                    ),
                    centroid_world=centroid,
                    bbox_dims_m=bbox_dims,
                    num_points=len(c_pts),
                    sample_pts_world=sample_pts,
                )
            )
            continue

        # Gate 3: Physical DUPLO bounding-box extent gate (Oversize/undersize trash)
        z_top_check = height_span if is_plausible_lifted_brick else z_top_rel
        min_z_top = 0.0095 if in_plate else 0.012
        if (
            major_span > 0.075
            or minor_span > 0.046
            or major_span < 0.012
            or z_top_check < min_z_top
            or z_top_check > 0.052
        ):
            rejected.append(
                RejectedCluster(
                    category="trash",
                    reason=(
                        f"non_duplo_extents(major={major_span * 1000:.1f}mm,"
                        f"minor={minor_span * 1000:.1f}mm,h={z_top_check * 1000:.1f}mm)"
                    ),
                    centroid_world=centroid,
                    bbox_dims_m=bbox_dims,
                    num_points=len(c_pts),
                    sample_pts_world=sample_pts,
                )
            )
            continue

        # Gate 4: HSV saturation & catalog chromaticity gate (Achromatic trash, cardboard, orange scraps, skin)
        is_white_plate_brick = in_plate and med_val >= 158.0 and med_sat <= 60.0
        if not is_white_plate_brick and (med_sat < 55.0 or med_val < 40.0 or chrom_dist > 0.145):
            # Distinguish skin-tone hand resting on table vs general trash
            hsv_c = cv2.cvtColor(c_cols.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            med_h = float(np.median(hsv_c[:, 0]))
            is_skin_like = (
                (4.0 <= med_h <= 20.0) and (35.0 <= med_sat <= 155.0) and major_span > 0.048
            )
            cat: DistractorCategory = "human_arm" if is_skin_like else "trash"
            rejected.append(
                RejectedCluster(
                    category=cat,
                    reason=(
                        f"non_catalog_color(sat={med_sat:.0f},val={med_val:.0f},"
                        f"chrom_dist={chrom_dist:.3f})"
                    ),
                    centroid_world=centroid,
                    bbox_dims_m=bbox_dims,
                    num_points=len(c_pts),
                    sample_pts_world=sample_pts,
                )
            )
            continue

        # Gate 5: 6D Asset Stable-Pose & Surface ICP verification gate
        eff_floor_z = float(np.percentile(c_pts[:, 2], 6)) if is_plausible_lifted_brick else floor_z
        spec, pose_name, T_wb, fit, rmse, has_def = _fit_6d_stable_pose_to_brick_cluster(
            c_pts,
            c_cols,
            support_z_m=eff_floor_z,
            stable_poses_map=poses_map,
            lock_table_resting=not is_plausible_lifted_brick,
        )
        min_fit = 0.38 if in_plate else 0.48
        max_rmse = 0.0115 if in_plate else 0.0090
        if fit < min_fit or rmse > max_rmse:
            rejected.append(
                RejectedCluster(
                    category="trash",
                    reason=f"failed_surface_icp(fitness={fit:.2f},rmse={rmse * 1000:.1f}mm)",
                    centroid_world=centroid,
                    bbox_dims_m=bbox_dims,
                    num_points=len(c_pts),
                    sample_pts_world=sample_pts,
                )
            )
            continue

        if is_plausible_lifted_brick:
            region = "in_gripper" if near_ee else "in_air"
            T_wb[2, 3] = float(np.median(c_pts[:, 2]))

        # Compute STL base-origin transform T_world_mesh_origin (shifted by -0.5 * hz along local Z)
        T_mesh = T_wb.copy()
        T_mesh[:3, 3] = T_wb[:3, 3] + T_wb[:3, :3] @ np.array(
            [0.0, 0.0, -0.5 * spec.dims_m[2]], dtype=np.float64
        )
        yaw = math.atan2(float(T_wb[1, 0]), float(T_wb[0, 0]))
        mean_rgb_tuple = tuple(float(x) for x in np.mean(c_cols.astype(np.float64), axis=0))
        conf = float(np.clip(0.65 * fit + 0.35 * max(0.0, 1.0 - rmse / 0.009), 0.0, 1.0))

        bricks.append(
            PerceivedBrick(
                block_id=f"block_{len(bricks)}",
                spec=spec,
                T_world_block=T_wb,
                T_world_mesh_origin=T_mesh,
                stable_pose_name=pose_name,
                xy_yaw=(float(T_wb[0, 3]), float(T_wb[1, 3]), float(yaw)),
                support_region=region,
                icp_fitness=fit,
                icp_rmse_m=rmse,
                confidence=conf,
                has_visual_defect=has_def,
                num_points=len(c_pts),
                mean_rgb=mean_rgb_tuple,  # type: ignore[arg-type]
                cluster_pts_world=c_pts,
            )
        )

    # Sort perceived bricks deterministically (source plate first, then by distance to plate center)
    src_xy = plates["source"].pos_world[:2]
    bricks.sort(
        key=lambda b: (
            0
            if b.support_region == "source"
            else (1 if b.support_region in ("in_gripper", "in_air") else 2),
            float(np.linalg.norm(b.T_world_block[:2, 3] - src_xy)),
        )
    )
    renumbered: list[PerceivedBrick] = []
    for idx, b in enumerate(bricks):
        renumbered.append(
            PerceivedBrick(
                block_id=f"block_{idx}",
                spec=b.spec,
                T_world_block=b.T_world_block,
                T_world_mesh_origin=b.T_world_mesh_origin,
                stable_pose_name=b.stable_pose_name,
                xy_yaw=b.xy_yaw,
                support_region=b.support_region,
                icp_fitness=b.icp_fitness,
                icp_rmse_m=b.icp_rmse_m,
                confidence=b.confidence,
                has_visual_defect=b.has_visual_defect,
                num_points=b.num_points,
                mean_rgb=b.mean_rgb,
                cluster_pts_world=b.cluster_pts_world,
            )
        )
    return renumbered, rejected


@dataclass
class _TrackedBrickSlot:
    brick: PerceivedBrick
    missed_frames: int = 0
    latched_in_gripper: bool = False
    T_ee_to_brick: np.ndarray | None = None


class Real2SimSceneTracker:
    """Real-time frame-to-frame SE(3) brick tracker with EMA filtering, occlusion hold, and gripper latching."""

    def __init__(
        self,
        ema_alpha: float = 0.58,
        max_assoc_dist_m: float = 0.042,
        max_missed_frames: int = 5,
    ) -> None:
        self.ema_alpha = float(np.clip(ema_alpha, 0.05, 1.0))
        self.max_assoc_dist_m = max_assoc_dist_m
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[str, _TrackedBrickSlot] = {}
        self._next_id: int = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    def update_latched_brick_from_arm(
        self,
        T_world_ee: np.ndarray,
        gripper_pct: float,
    ) -> list[PerceivedBrick]:
        """Update any brick currently grasped by the real SO-101 arm at the arm's 25-30 Hz poll rate."""
        if not self._tracks:
            return []
        ee_pos = T_world_ee[:3, 3]
        g_norm = float(gripper_pct) / 100.0 if float(gripper_pct) > 1.05 else float(gripper_pct)
        # Latch closest brick if gripper is partially closed (0.06 <= g_norm <= 0.78) and near a brick
        is_grasping = 0.06 <= g_norm <= 0.78
        any_latched = any(s.latched_in_gripper for s in self._tracks.values())

        if is_grasping and not any_latched:
            best_tid: str | None = None
            best_d = 0.052
            for tid, slot in self._tracks.items():
                d = float(np.linalg.norm(slot.brick.T_world_block[:3, 3] - ee_pos))
                if d < best_d:
                    best_d = d
                    best_tid = tid
            if best_tid is not None:
                slot = self._tracks[best_tid]
                slot.latched_in_gripper = True
                slot.T_ee_to_brick = np.linalg.inv(T_world_ee) @ slot.brick.T_world_block

        for tid, slot in self._tracks.items():
            if not slot.latched_in_gripper:
                continue
            if not is_grasping and g_norm > 0.82:
                # Gripper opened -> release brick
                slot.latched_in_gripper = False
                slot.T_ee_to_brick = None
                continue
            T_rel = slot.T_ee_to_brick if slot.T_ee_to_brick is not None else np.eye(4)
            T_new = T_world_ee @ T_rel
            T_new[2, 3] = max(0.012, float(T_new[2, 3]))
            slot.brick = self._rebind_id(
                slot.brick,
                tid,
                T_new,
                support_region="in_gripper",
            )
            slot.missed_frames = 0

        active = [
            slot.brick
            for slot in sorted(
                self._tracks.values(), key=lambda s: int(s.brick.block_id.split("_")[-1])
            )
        ]
        return [self._rebind_id(b, f"block_{i}", b.T_world_block) for i, b in enumerate(active)]

    def update(self, detections: list[PerceivedBrick]) -> list[PerceivedBrick]:
        if not self._tracks:
            out: list[PerceivedBrick] = []
            for d in detections:
                bid = f"block_{self._next_id}"
                self._next_id += 1
                nb = self._rebind_id(d, bid, d.T_world_block)
                self._tracks[bid] = _TrackedBrickSlot(brick=nb, missed_frames=0)
                out.append(nb)
            return out

        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = list(range(len(detections)))
        matches: list[tuple[str, int]] = []

        # Greedy nearest-neighbor matching gated by part name + color similarity and 3D distance
        pairs: list[tuple[float, str, int]] = []
        for tid in unmatched_tracks:
            t_brick = self._tracks[tid].brick
            t_rgb = np.asarray(t_brick.mean_rgb, dtype=np.float64)
            for didx in unmatched_dets:
                d_brick = detections[didx]
                if t_brick.spec.name != d_brick.spec.name:
                    continue
                dist = float(
                    np.linalg.norm(t_brick.T_world_block[:3, 3] - d_brick.T_world_block[:3, 3])
                )
                if dist <= self.max_assoc_dist_m:
                    d_rgb = np.asarray(d_brick.mean_rgb, dtype=np.float64)
                    col_pen = 0.00015 * float(np.linalg.norm(t_rgb - d_rgb))
                    pairs.append((dist + col_pen, tid, didx))
        pairs.sort(key=lambda x: x[0])

        for _, tid, didx in pairs:
            if tid in unmatched_tracks and didx in unmatched_dets:
                matches.append((tid, didx))
                unmatched_tracks.remove(tid)
                unmatched_dets.remove(didx)

        alpha = self.ema_alpha
        for tid, didx in matches:
            slot = self._tracks[tid]
            prev = slot.brick
            det = detections[didx]
            T_smooth = det.T_world_block.copy()
            T_smooth[:3, 3] = (1.0 - alpha) * prev.T_world_block[:3, 3] + alpha * det.T_world_block[
                :3, 3
            ]
            # Blend rotation only if geodesic angle is small (avoids averaging across discrete symmetry jumps)
            if so3_geodesic_angle_rad(prev.T_world_block[:3, :3], det.T_world_block[:3, :3]) < 0.35:
                R_rel = prev.T_world_block[:3, :3].T @ det.T_world_block[:3, :3]
                rvec = Rotation.from_matrix(R_rel).as_rotvec()
                T_smooth[:3, :3] = (
                    prev.T_world_block[:3, :3] @ Rotation.from_rotvec(alpha * rvec).as_matrix()
                )
            updated = self._rebind_id(
                det,
                tid,
                T_smooth,
                has_defect=(prev.has_visual_defect or det.has_visual_defect),
            )
            self._tracks[tid] = _TrackedBrickSlot(
                brick=updated,
                missed_frames=0,
                latched_in_gripper=slot.latched_in_gripper,
                T_ee_to_brick=slot.T_ee_to_brick,
            )

        for tid in list(unmatched_tracks):
            slot = self._tracks[tid]
            if slot.latched_in_gripper:
                # Keep brick alive while carried by the gripper even if occluded by gripper fingers
                slot.missed_frames = 0
                continue
            slot.missed_frames += 1
            if slot.missed_frames > self.max_missed_frames:
                del self._tracks[tid]

        for didx in unmatched_dets:
            det = detections[didx]
            bid = f"block_{self._next_id}"
            self._next_id += 1
            nb = self._rebind_id(det, bid, det.T_world_block)
            self._tracks[bid] = _TrackedBrickSlot(brick=nb, missed_frames=0)

        active = [
            slot.brick
            for slot in sorted(
                self._tracks.values(), key=lambda s: int(s.brick.block_id.split("_")[-1])
            )
        ]
        # Ensure contiguous block_0..block_{N-1} naming for MuJoCo twin compatibility
        return [self._rebind_id(b, f"block_{i}", b.T_world_block) for i, b in enumerate(active)]

    @staticmethod
    def _rebind_id(
        b: PerceivedBrick,
        block_id: str,
        T_wb: np.ndarray,
        has_defect: bool | None = None,
        support_region: str | None = None,
    ) -> PerceivedBrick:
        T_mesh = T_wb.copy()
        T_mesh[:3, 3] = T_wb[:3, 3] + T_wb[:3, :3] @ np.array(
            [0.0, 0.0, -0.5 * b.spec.dims_m[2]], dtype=np.float64
        )
        yaw = math.atan2(float(T_wb[1, 0]), float(T_wb[0, 0]))
        return PerceivedBrick(
            block_id=block_id,
            spec=b.spec,
            T_world_block=T_wb,
            T_world_mesh_origin=T_mesh,
            stable_pose_name=b.stable_pose_name,
            xy_yaw=(float(T_wb[0, 3]), float(T_wb[1, 3]), float(yaw)),
            support_region=b.support_region if support_region is None else support_region,
            icp_fitness=b.icp_fitness,
            icp_rmse_m=b.icp_rmse_m,
            confidence=b.confidence,
            has_visual_defect=b.has_visual_defect if has_defect is None else has_defect,
            num_points=b.num_points,
            mean_rgb=b.mean_rgb,
            cluster_pts_world=b.cluster_pts_world,
        )


def synthesize_cluttered_scene_point_cloud(
    spawned_blocks: list[SpawnedBlockInfo] | None = None,
    num_blocks: int = 4,
    seed: int = 42,
    include_human_arm: bool = True,
    include_wires: bool = True,
    include_trash: bool = True,
    include_robot_arm_points: bool = True,
    q_arm_rad: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[SpawnedBlockInfo], dict[str, Any]]:
    """Synthesize a realistic sensor-fused 3D colored point cloud with clutter & distractors.

    Renders the genuine Dual-D415 RGB-D point cloud of the 3-plate DUPLO scene in MuJoCo 4.0.1
    and injects 3D point clouds of:
      - A human forearm + hand reaching into the workspace (`human_arm`)
      - Two USB/power cables snaking across the tabletop (`wire`)
      - Trash objects: a grey/white cup, a brown cardboard scrap, and an orange calibration fragment (`trash`)
      - Points on the SO-101 robot arm links (`robot_arm`)
    """
    rng = np.random.default_rng(seed)
    ref_twin = SO101InspectionTwin(
        num_blocks=num_blocks,
        cluster_seed=seed,
        spawned_blocks=spawned_blocks,
        show_viewer=False,
        realtime_pacing=False,
    )
    try:
        gt_blocks: list[SpawnedBlockInfo] = []
        for sb in ref_twin.spawned_blocks:
            bid = ref_twin.block_body_ids[sb.block_id]
            T_gt = np.eye(4, dtype=np.float64)
            T_gt[:3, :3] = ref_twin.data.xmat[bid].reshape(3, 3).copy()
            T_gt[:3, 3] = ref_twin.data.xpos[bid].copy()
            yaw_gt = math.atan2(float(T_gt[1, 0]), float(T_gt[0, 0]))
            gt_blocks.append(
                SpawnedBlockInfo(
                    block_id=sb.block_id,
                    spec=sb.spec,
                    initial_xy_yaw=(float(T_gt[0, 3]), float(T_gt[1, 3]), float(yaw_gt)),
                    has_visual_defect=sb.has_visual_defect,
                    initial_T_world_block=T_gt,
                )
            )

        rgb1, d1 = ref_twin.render_rgbd("realsense_1")
        rgb2, d2 = ref_twin.render_rgbd("realsense_2")
        assert d1 is not None and d2 is not None
        p1, c1 = unproject_rgbd_workspace(d1, rgb1, ref_twin.rig.realsense_1, stride=2)
        p2, c2 = unproject_rgbd_workspace(d2, rgb2, ref_twin.rig.realsense_2, stride=2)
        q_current = ref_twin.current_arm_q() if q_arm_rad is None else np.asarray(q_arm_rad)
    finally:
        ref_twin.close()

    pts_list = [p1, p2]
    col_list = [c1, c2]
    injected_counts: dict[str, int] = {
        "human_arm_points": 0,
        "wire_points": 0,
        "trash_points": 0,
        "robot_arm_points": 0,
    }

    if include_human_arm:
        # Forearm + hand upper surface reaching in from (+0.39, +0.18, +0.14) toward (+0.295, +0.115, +0.068)
        t_grid, theta_grid = np.meshgrid(
            np.linspace(0.0, 1.0, 28),
            np.linspace(0.15 * math.pi, 0.85 * math.pi, 18),
        )
        t_param = t_grid.ravel()
        theta = theta_grid.ravel()
        n_arm = len(t_param)
        rad = np.where(t_param < 0.35, 0.018, 0.024) + rng.normal(0.0, 0.0006, size=n_arm)
        p_start = np.array([0.295, 0.115, 0.068], dtype=np.float64)
        p_end = np.array([0.390, 0.180, 0.140], dtype=np.float64)
        axis = p_end - p_start
        axis /= np.linalg.norm(axis)
        ortho1 = np.array([-axis[1], axis[0], 0.0], dtype=np.float64)
        ortho1 /= np.linalg.norm(ortho1)
        ortho2 = np.cross(axis, ortho1)
        if ortho2[2] < 0.0:
            ortho2 = -ortho2
        spine = p_start[None, :] + t_param[:, None] * (p_end - p_start)[None, :]
        arm_pts = (
            spine
            + (rad * np.cos(theta))[:, None] * ortho1[None, :]
            + (rad * np.sin(theta))[:, None] * ortho2[None, :]
        )
        # Realistic warm skin tone RGB [215, 162, 132] with illumination variance
        arm_cols = np.clip(
            np.array([215.0, 162.0, 132.0])[None, :] + rng.normal(0.0, 8.0, size=(n_arm, 3)),
            0,
            255,
        ).astype(np.uint8)
        pts_list.append(arm_pts)
        col_list.append(arm_cols)
        injected_counts["human_arm_points"] += n_arm

    if include_wires:
        # Cable #1: black USB cable snaking along x in [0.14, 0.25], y ~ -0.118
        n_w1 = 220
        s1 = np.linspace(0.14, 0.25, n_w1)
        w1_pts = np.column_stack(
            [
                s1 + rng.normal(0.0, 0.001, size=n_w1),
                -0.118 + 0.008 * np.sin(36.0 * s1) + rng.normal(0.0, 0.0015, size=n_w1),
                0.009 + 0.002 * np.cos(20.0 * s1) + rng.uniform(-0.002, 0.002, size=n_w1),
            ]
        )
        w1_cols = np.clip(28.0 + rng.normal(0.0, 6.0, size=(n_w1, 3)), 0, 255).astype(np.uint8)

        # Cable #2: dark-grey braided cord near x ~ 0.365, y in [-0.12, +0.03] (well clear of human arm)
        n_w2 = 200
        s2 = np.linspace(-0.12, 0.03, n_w2)
        w2_pts = np.column_stack(
            [
                0.365 + 0.008 * np.sin(32.0 * s2) + rng.normal(0.0, 0.0015, size=n_w2),
                s2 + rng.normal(0.0, 0.001, size=n_w2),
                0.010 + rng.uniform(-0.0025, 0.0025, size=n_w2),
            ]
        )
        w2_cols = np.clip(62.0 + rng.normal(0.0, 8.0, size=(n_w2, 3)), 0, 255).astype(np.uint8)

        pts_list.extend([w1_pts, w2_pts])
        col_list.extend([w1_cols, w2_cols])
        injected_counts["wire_points"] += n_w1 + n_w2

    if include_trash:
        # Trash #1: crumpled white/grey paper cup at (0.335, -0.210) (oversize + low saturation)
        n_t1 = 220
        ang1 = rng.uniform(0.0, 2.0 * math.pi, size=n_t1)
        r1 = rng.uniform(0.0, 0.042, size=n_t1)
        t1_pts = np.column_stack(
            [
                0.335 + r1 * np.cos(ang1),
                -0.210 + r1 * np.sin(ang1),
                rng.uniform(0.006, 0.022, size=n_t1),
            ]
        )
        t1_cols = np.clip(210.0 + rng.normal(0.0, 10.0, size=(n_t1, 3)), 0, 255).astype(np.uint8)

        # Trash #2: dull brown cardboard scrap at (0.145, +0.115) (non-catalog low-saturation brown)
        n_t2 = 180
        t2_pts = np.column_stack(
            [
                0.145 + rng.uniform(-0.016, 0.016, size=n_t2),
                0.115 + rng.uniform(-0.014, 0.014, size=n_t2),
                rng.uniform(0.006, 0.018, size=n_t2),
            ]
        )
        t2_cols = np.clip(
            np.array([138.0, 112.0, 88.0])[None, :] + rng.normal(0.0, 7.0, size=(n_t2, 3)),
            0,
            255,
        ).astype(np.uint8)

        # Trash #3: orange calibration board fragment at (0.285, -0.098) (high saturation, non-catalog hue ~ 12)
        n_t3 = 180
        t3_pts = np.column_stack(
            [
                0.285 + rng.uniform(-0.015, 0.015, size=n_t3),
                -0.098 + rng.uniform(-0.014, 0.014, size=n_t3),
                rng.uniform(0.006, 0.018, size=n_t3),
            ]
        )
        t3_cols = np.clip(
            np.array([235.0, 118.0, 28.0])[None, :] + rng.normal(0.0, 6.0, size=(n_t3, 3)),
            0,
            255,
        ).astype(np.uint8)

        pts_list.extend([t1_pts, t2_pts, t3_pts])
        col_list.extend([t1_cols, t2_cols, t3_cols])
        injected_counts["trash_points"] += n_t1 + n_t2 + n_t3

    if include_robot_arm_points:
        from so101_kinematics import CalibratedSO101Kinematics

        cal_kin = CalibratedSO101Kinematics("follower")
        frames = cal_kin.frames(q_current)
        rob_pts_list: list[np.ndarray] = []
        for lname in ("Group 1", "Body1_3", "Body3"):
            T_wl = T_WORLD_BASE @ frames[lname]
            local_offsets = rng.normal(0.0, 0.010, size=(60, 3))
            p_w = (local_offsets @ T_wl[:3, :3].T) + T_wl[:3, 3]
            rob_pts_list.append(p_w)
        rob_pts = np.vstack(rob_pts_list)
        rob_cols = np.full((len(rob_pts), 3), [228, 222, 206], dtype=np.uint8)
        pts_list.append(rob_pts)
        col_list.append(rob_cols)
        injected_counts["robot_arm_points"] += len(rob_pts)

    all_pts = np.vstack(pts_list)
    all_cols = np.vstack(col_list)
    return (
        all_pts,
        all_cols,
        gt_blocks,
        {
            "q_arm_rad": q_current.tolist(),
            "injected_counts": injected_counts,
        },
    )


@dataclass
class Real2SimClosedLoopReport:
    """Summary of an end-to-end Real2Sim -> MuJoCo Desktop -> Closed-Loop Policy/Planner run."""

    provenance: str
    initial_scene: dict[str, Any]
    replicated_brick_count: int
    mean_replication_pos_err_mm: float
    distractor_counts: dict[str, int]
    controller_type: Literal["cogman_planner", "chunked_mlp_policy"]
    execution_succeeded: bool
    sorted_counts: dict[str, int]
    batch_sorted_records: list[dict[str, Any]]
    policy_train_mse: float | None = None
    elapsed_s: float = 0.0
    report_json_path: str | None = None


class Real2SimPipeline:
    """Real-time, lightweight Real2Sim pipeline for the SO-101 3-Plate DUPLO Inspection Cell.

    1. Ingests sensor-fused 3D colored point clouds (Dual-D415 RGB-D, `live_depth.py` JSON packets,
       `.ply` clouds, or live HTTP streams).
    2. Masks out the SO-101 robot arm via calibrated FK, localizes & subtracts the table and 3 KALAS
       plates (`source`, `good`, `defective`), and rejects human arms, wires, and trash via a 5-gate
       geometric + chromatic + 6D surface ICP filter.
    3. Replicates the clean asset-backed scene in MuJoCo 4.0.1 Desktop (`SO101InspectionTwin`) and
       closes the loop for policy training and deployment.
    """

    def __init__(
        self,
        rig: CameraExtrinsicsRig | None = None,
        voxel_size_m: float = 0.0030,
        enable_temporal_tracking: bool = False,
    ) -> None:
        self.rig = rig or station_camera_rig()
        self.voxel_size_m = voxel_size_m
        self.stable_poses_map = load_duplo_stable_poses()
        self.tracker = Real2SimSceneTracker() if enable_temporal_tracking else None
        self._frame_counter = 0
        from so101_kinematics import CalibratedSO101Kinematics

        self._cal_kin = CalibratedSO101Kinematics("follower")

    def perceive_point_cloud(
        self,
        points_world: np.ndarray,
        colors_rgb: np.ndarray,
        q_arm_rad: np.ndarray | None = None,
        provenance: str = "sensor_fused_point_cloud",
    ) -> Real2SimSceneState:
        """Run the full real-time Real2Sim perception & distractor-filtering pipeline on a world cloud."""
        t0 = time.perf_counter()
        self._frame_counter += 1
        raw_count = int(len(points_world))

        # 1. Crop to 3-plate cell workspace & voxel-downsample
        if raw_count > 0:
            in_ws = np.all(
                (points_world >= REAL2SIM_WORKSPACE_MIN) & (points_world <= REAL2SIM_WORKSPACE_MAX),
                axis=1,
            )
            pts_ws = points_world[in_ws]
            cols_ws = colors_rgb[in_ws]
        else:
            pts_ws = points_world
            cols_ws = colors_rgb

        pts_ds, cols_ds = voxel_downsample_xyzrgb(pts_ws, cols_ws, voxel_size_m=self.voxel_size_m)
        ds_count = int(len(pts_ds))

        # 2. Robot self-masking via calibrated SO-101 forward kinematics
        pts_no_rob, cols_no_rob, rob_pts = filter_robot_arm_points(
            pts_ds, cols_ds, q_arm_rad=q_arm_rad, cal_kin=self._cal_kin, rig=self.rig
        )
        rob_masked_count = int(len(rob_pts))

        # 3. Localize & subtract table and 3 IKEA KALAS plates
        fg_pts, fg_cols, plates = localize_and_subtract_support_surfaces(pts_no_rob, cols_no_rob)
        fg_count = int(len(fg_pts))

        ee_pos_world: np.ndarray | None = None
        if q_arm_rad is not None and len(q_arm_rad) >= 5:
            try:
                q5 = np.asarray(q_arm_rad, dtype=np.float64)[:5]
                T_wb = T_WORLD_BASE @ self._cal_kin.frames(q5)["Body3"]
                ee_pos_world = T_wb[:3, 3].copy()
            except Exception:
                ee_pos_world = None

        # 4. 5-gate distractor rejection + 6D brick asset pose estimation
        bricks, rejected = perceive_bricks_and_reject_clutter(
            fg_pts,
            fg_cols,
            plates=plates,
            stable_poses_map=self.stable_poses_map,
            ee_pos_world=ee_pos_world,
        )
        if rob_masked_count > 0:
            sample_idx = np.linspace(0, len(rob_pts) - 1, min(24, len(rob_pts)), dtype=int)
            rejected.append(
                RejectedCluster(
                    category="robot_arm",
                    reason=f"masked_by_so101_fk({rob_masked_count}_pts)",
                    centroid_world=rob_pts.mean(axis=0),
                    bbox_dims_m=tuple(
                        float(x) for x in (rob_pts.max(axis=0) - rob_pts.min(axis=0))
                    ),  # type: ignore[arg-type]
                    num_points=rob_masked_count,
                    sample_pts_world=rob_pts[sample_idx].copy(),
                )
            )

        if self.tracker is not None:
            bricks = self.tracker.update(bricks)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Real2SimSceneState(
            timestamp_s=time.time(),
            frame_index=self._frame_counter,
            bricks=bricks,
            plates=plates,
            rejected_clusters=rejected,
            raw_point_count=raw_count,
            downsampled_point_count=ds_count,
            robot_masked_point_count=rob_masked_count,
            foreground_point_count=fg_count,
            latency_ms=latency_ms,
            provenance=provenance,
        )

    def perceive_rgbd_frames(
        self,
        frames: list[tuple[np.ndarray, np.ndarray, CameraModel]],
        q_arm_rad: np.ndarray | None = None,
        stride: int = 1,
        provenance: str = "dual_d415_rgbd_fusion",
    ) -> Real2SimSceneState:
        """Unproject and fuse multi-camera RGB-D frames, then perceive the Real2Sim scene."""
        pts_list: list[np.ndarray] = []
        col_list: list[np.ndarray] = []
        for depth_m, rgb, cam in frames:
            p, c = unproject_rgbd_workspace(depth_m, rgb, cam, stride=stride)
            if len(p):
                pts_list.append(p)
                col_list.append(c)
        if not pts_list:
            return self.perceive_point_cloud(
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.uint8),
                q_arm_rad=q_arm_rad,
                provenance=provenance,
            )
        return self.perceive_point_cloud(
            np.vstack(pts_list),
            np.vstack(col_list),
            q_arm_rad=q_arm_rad,
            provenance=provenance,
        )

    def perceive_teammate_packets(
        self,
        top_packet: dict[str, Any] | None = None,
        plate_packet: dict[str, Any] | None = None,
        provisional_transform: dict[str, Any] | None = None,
        q_arm_rad: np.ndarray | None = None,
    ) -> Real2SimSceneState:
        """Ingest teammates' `live_depth.py` / `realsense_depth_capture.py` packets and perceive scene."""
        pts_w, cols_w = fuse_teammate_top_and_plate_packets(
            top_packet,
            plate_packet,
            rig=self.rig,
            provisional_transform=provisional_transform,
        )
        return self.perceive_point_cloud(
            pts_w,
            cols_w,
            q_arm_rad=q_arm_rad,
            provenance="teammate_live_depth_packets",
        )

    def perceive_from_live_depth_http(
        self,
        base_url: str = "http://127.0.0.1:8094",
        q_arm_rad: np.ndarray | None = None,
        timeout_s: float = 1.5,
        prefer_snapshots: bool = False,
        camera_mode: Literal["top", "plate", "fused"] = "top",
    ) -> Real2SimSceneState:
        """Fetch live point clouds or aligned snapshots from teammates' `scripts/live_depth.py` (`:8094`).

        Defaults to `camera_mode="top"` (single overhead RealSense D415 `350122060653`) for a
        clean, holistic overhead view accurately aligned to the SO-101 arm base without cross-camera
        stereo double-surfaces.
        """
        base = base_url.rstrip("/")

        def _fetch_json(url: str) -> dict[str, Any] | None:
            try:
                with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception:
                return None

        if prefer_snapshots:
            s_top = _fetch_json(f"{base}/snapshot/top") if camera_mode in ("top", "fused") else None
            s_plate = (
                _fetch_json(f"{base}/snapshot/plate") if camera_mode in ("plate", "fused") else None
            )
            pts_list: list[np.ndarray] = []
            col_list: list[np.ndarray] = []
            for snap in (s_top, s_plate):
                if snap and "rgb_jpeg_b64" in snap and "depth_png_b64" in snap:
                    p_s, c_s = decode_live_depth_snapshot(snap, rig=self.rig)
                    if len(p_s):
                        pts_list.append(p_s)
                        col_list.append(c_s)
            if pts_list:
                pts_all = np.vstack(pts_list)
                cols_all = np.vstack(col_list)
                if len(pts_all) >= 200:
                    hsv_all = cv2.cvtColor(cols_all.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(
                        -1, 3
                    )
                    is_wood = (
                        (hsv_all[:, 0] >= 11)
                        & (hsv_all[:, 0] <= 26)
                        & (hsv_all[:, 1] >= 80)
                        & (hsv_all[:, 2] >= 75)
                        & (pts_all[:, 0] >= 0.08)
                        & (pts_all[:, 0] <= 0.46)
                        & (np.abs(pts_all[:, 1]) <= 0.19)
                        & (np.abs(pts_all[:, 2]) <= 0.015)
                    )
                    if int(np.sum(is_wood)) >= 80:
                        pts_all[:, 2] -= float(np.median(pts_all[is_wood, 2]))
                return self.perceive_point_cloud(
                    pts_all,
                    cols_all,
                    q_arm_rad=q_arm_rad,
                    provenance=f"live_depth_http_snapshots:{camera_mode}",
                )

        top_pkt = _fetch_json(f"{base}/points/top") if camera_mode in ("top", "fused") else None
        plate_pkt = (
            _fetch_json(f"{base}/points/plate") if camera_mode in ("plate", "fused") else None
        )
        if (top_pkt and "points" in top_pkt) or (plate_pkt and "points" in plate_pkt):
            return self.perceive_teammate_packets(
                top_packet=top_pkt,
                plate_packet=plate_pkt,
                q_arm_rad=q_arm_rad,
            )

        merged_pkt = _fetch_json(f"{base}/points/merged")
        if merged_pkt and "points" in merged_pkt:
            p_m, c_m = ingest_teammate_depth_packet(merged_pkt, self.rig.realsense_1)
            return self.perceive_point_cloud(
                p_m, c_m, q_arm_rad=q_arm_rad, provenance="live_depth_http_merged"
            )

        return self.perceive_teammate_packets(
            top_packet=top_pkt,
            plate_packet=plate_pkt,
            q_arm_rad=q_arm_rad,
        )

    def perceive_capture_directory(
        self,
        directory: str | Path,
        q_arm_rad: np.ndarray | None = None,
    ) -> Real2SimSceneState:
        """Ingest a capture folder from `realsense_depth_capture.py`, `board_alignment_pilot.py`, or `compare_depth_scans.py`."""
        folder = Path(directory)
        top_json = folder / "top-points.json"
        plate_json = folder / "plate-points.json"
        prov_tf_path = folder / "provisional-transform.json"
        if top_json.exists() or plate_json.exists():
            top_pkt = (
                json.loads(top_json.read_text(encoding="utf-8")) if top_json.exists() else None
            )
            plate_pkt = (
                json.loads(plate_json.read_text(encoding="utf-8")) if plate_json.exists() else None
            )
            prov_tf = (
                json.loads(prov_tf_path.read_text(encoding="utf-8"))
                if prov_tf_path.exists()
                else None
            )
            return self.perceive_teammate_packets(
                top_packet=top_pkt,
                plate_packet=plate_pkt,
                provisional_transform=prov_tf,
                q_arm_rad=q_arm_rad,
            )

        # Check for aligned PNG pairs `{serial|top|plate}-color.png` + `{serial|top|plate}-depth16.png`
        frames: list[tuple[np.ndarray, np.ndarray, CameraModel]] = []
        for prefix, cam in [
            (str(self.rig.realsense_1.serial_or_dev), self.rig.realsense_1),
            (str(self.rig.realsense_2.serial_or_dev), self.rig.realsense_2),
            ("top", self.rig.realsense_1),
            ("plate", self.rig.realsense_2),
        ]:
            c_path = folder / f"{prefix}-color.png"
            d_path = folder / f"{prefix}-depth16.png"
            meta_path = folder / f"{prefix}-metadata.json"
            if c_path.exists() and d_path.exists():
                rgb = np.asarray(Image.open(c_path).convert("RGB"), dtype=np.uint8)
                depth_u16 = np.asarray(Image.open(d_path))
                scale_m = 0.001
                if meta_path.exists():
                    meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
                    scale_m = float(meta_doc.get("depth_scale_m", 0.001))
                depth_m = depth_u16.astype(np.float64) * scale_m
                frames.append((depth_m, rgb, cam))
        if frames:
            return self.perceive_rgbd_frames(
                frames, q_arm_rad=q_arm_rad, provenance=f"capture_dir_rgbd:{folder.name}"
            )

        # Check for SDK `.ply` files (`top.ply`, `plate.ply`, `350122060653.ply`, `243622060450.ply`)
        pts_list: list[np.ndarray] = []
        col_list: list[np.ndarray] = []
        for ply_name in (
            f"{self.rig.realsense_1.serial_or_dev}.ply",
            f"{self.rig.realsense_2.serial_or_dev}.ply",
            "top.ply",
            "plate.ply",
            "provisional-merged-scene.ply",
            "brick-001-rough-merged.ply",
        ):
            ply_path = folder / ply_name
            if ply_path.exists():
                p_i, c_i = load_point_cloud_file(ply_path, rig=self.rig)
                if len(p_i):
                    pts_list.append(p_i)
                    col_list.append(c_i)
        if pts_list:
            return self.perceive_point_cloud(
                np.vstack(pts_list),
                np.vstack(col_list),
                q_arm_rad=q_arm_rad,
                provenance=f"capture_dir_ply:{folder.name}",
            )

        raise FileNotFoundError(f"No recognized depth/point-cloud files found in {folder}")

    def replicate_in_twin(
        self,
        scene_state: Real2SimSceneState,
        show_viewer: bool = True,
        realtime_pacing: bool = True,
        scene_dir: str | Path | None = None,
        anomaly_scorer=None,
        anomaly_threshold: float | None = None,
        anomaly_message_fn=None,
    ) -> SO101InspectionTwin:
        """Instantiate `SO101InspectionTwin` in MuJoCo 4.0.1 Desktop replicating the perceived scene.

        Only the localized LEGO DUPLO brick assets and 3 IKEA KALAS plates are instantiated in the
        MuJoCo physics world; all human arms, wires, and trash clusters are ignored.
        """
        spawned = scene_state.to_spawned_blocks(source_only=False)
        if not spawned:
            spawned = [
                SpawnedBlockInfo(
                    block_id="block_0",
                    spec=DUPLO_CATALOG["duplo_2x2_3437"],
                    initial_xy_yaw=(0.23, 0.00, 0.0),
                    has_visual_defect=False,
                )
            ]

        twin = SO101InspectionTwin(
            block_spec=spawned[0].spec,
            rig=self.rig,
            scene_dir=scene_dir,
            initial_block_xy_yaw=spawned[0].initial_xy_yaw,
            show_viewer=show_viewer,
            realtime_pacing=realtime_pacing,
            anomaly_scorer=anomaly_scorer,
            anomaly_threshold=anomaly_threshold,
            anomaly_message_fn=anomaly_message_fn,
            spawned_blocks=spawned,
            plate_positions=scene_state.plate_positions_dict(),
        )
        twin.sync_real2sim_scene(scene_state, settle_steps=18)
        return twin

    def close_the_loop(
        self,
        scene_state: Real2SimSceneState,
        controller: Literal["cogman_planner", "chunked_mlp_policy"] = "cogman_planner",
        show_viewer: bool = True,
        realtime_pacing: bool = False,
        output_dir: str | Path | None = None,
        ground_truth_blocks: list[SpawnedBlockInfo] | None = None,
        forward_to_armd: bool = False,
        armd_bridge: ArmdFollowerBridge | None = None,
        hold_viewer_seconds: float = 0.0,
    ) -> Real2SimClosedLoopReport:
        """Replicate perceived scene in MuJoCo Desktop and close the loop with policy/planner execution."""
        t0 = time.monotonic()
        out_root = (
            Path(output_dir)
            if output_dir is not None
            else (REPO_ROOT / "artifacts" / "block_inspector" / "real2sim")
        )
        out_root.mkdir(parents=True, exist_ok=True)

        # Compute replication accuracy against ground truth if available
        pos_errs_mm: list[float] = []
        if ground_truth_blocks and scene_state.bricks:
            for b in scene_state.bricks:
                same_part = [
                    gt
                    for gt in ground_truth_blocks
                    if gt.spec.name == b.spec.name and gt.initial_T_world_block is not None
                ]
                if same_part:
                    d_mm = min(
                        1000.0
                        * float(
                            np.linalg.norm(
                                b.T_world_block[:3, 3] - gt.initial_T_world_block[:3, 3]  # type: ignore[index]
                            )
                        )
                        for gt in same_part
                    )
                    pos_errs_mm.append(d_mm)
        mean_err_mm = float(np.mean(pos_errs_mm)) if pos_errs_mm else 0.0

        # Anomaly callback inspects the real rendered Plate-camera ROI during the CCW spin for surface defects
        def _surface_defect_detector(
            _tw: SO101InspectionTwin, _step_k: int, _q4_rad: float, rgb_plate: np.ndarray
        ) -> str | None:
            if _tw.anomaly_flagged:
                return None
            T_held = _tw.body3_world_pose() @ _tw._T_body3_block_held
            uv, z_c = _tw.rig.realsense_2.project_world_points(T_held[:3, 3].reshape(1, 3))
            if z_c[0] <= 0.05:
                return None
            u_px, v_px = int(round(float(uv[0, 0]))), int(round(float(uv[0, 1])))
            u0 = max(0, u_px - 36)
            u1 = min(rgb_plate.shape[1], u_px + 36)
            v0 = max(0, v_px - 36)
            v1 = min(rgb_plate.shape[0], v_px + 36)
            roi = rgb_plate[v0:v1, u0:u1]
            if roi.size == 0:
                return None
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            is_mag = (hsv[:, :, 0] >= 140) & (hsv[:, :, 0] <= 172) & (hsv[:, :, 1] >= 115)
            mag_count = int(np.sum(is_mag))
            if mag_count >= 12:
                return f"surface_blemish_detected_in_plate_cam(pixels={mag_count})"
            return None

        twin = self.replicate_in_twin(
            scene_state,
            show_viewer=show_viewer,
            realtime_pacing=realtime_pacing,
            anomaly_message_fn=_surface_defect_detector,
        )
        bridge = armd_bridge or (
            ArmdFollowerBridge(request_fn=lambda *_: {}) if forward_to_armd else None
        )

        try:
            if bridge is not None:
                twin.frame_listeners.append(
                    lambda tw: bridge.send_joint_command_safe(tw.q_cmd, tw.gripper_pct)
                )

            train_mse: float | None = None
            if controller == "cogman_planner":
                mode: InspectionMode = (
                    "clear_source_plate" if len(twin.spawned_blocks) > 1 else "plate_spin_sort"
                )
                instruction = (
                    "clear all blocks from the source plate"
                    if mode == "clear_source_plate"
                    else "investigate this block"
                )
                agent = CogManInspectionAgent(executor=twin, ledger=twin.ledger)
                run_rep = agent.run(
                    instruction=instruction,
                    block_id="block_0",
                    output_dir=out_root / "reports",
                    mode=mode,
                )
                exec_ok = bool(run_rep.succeeded)
            else:
                from brickcell.block_inspector.policy_skills import (
                    ChunkedMLPPolicy,
                    _encode_policy_features,
                )

                # Train an action-chunked BC policy directly from the Real2Sim replicated state and roll it out!
                b0 = twin.spawned_blocks[0]
                loc_pos = (
                    b0.initial_T_world_block[:3, 3].copy()
                    if b0.initial_T_world_block is not None
                    else np.array(
                        [b0.initial_xy_yaw[0], b0.initial_xy_yaw[1], 0.5 * b0.spec.dims_m[2]],
                        dtype=np.float64,
                    )
                )
                loc_yaw = float(b0.initial_xy_yaw[2])
                target_pile = "defective" if b0.has_visual_defect else "good"
                anom_bit = 1.0 if b0.has_visual_defect else 0.0

                twin.ledger.set_initial_pose(twin.ground_truth_block_pose())
                if b0.has_visual_defect:
                    twin.receive_anomaly_message("real2sim_perceived_surface_defect")

                ep_q: list[np.ndarray] = []
                ep_g: list[float] = []
                ep_a: list[np.ndarray] = []

                def _rec(tw: SO101InspectionTwin) -> None:
                    ep_q.append(tw.current_arm_q())
                    ep_g.append(float(tw.gripper_pct))
                    ep_a.append(np.concatenate([tw.q_cmd, [tw.gripper_pct]]))

                twin.frame_listeners.append(_rec)
                try:
                    ok1 = twin._skill_pick_block()
                    ok2 = twin._skill_present_and_spin_at_plate_camera()
                    ok3 = twin._skill_sort_block_to_pile(target_pile)
                    if not (ok1.succeeded and ok2.succeeded and ok3.succeeded):
                        raise RuntimeError("Real2Sim teacher rollout failed")
                finally:
                    twin.frame_listeners.remove(_rec)

                chunk_h = 8
                A = np.asarray(ep_a, dtype=np.float64)
                n_ep = len(A)
                X_rows: list[np.ndarray] = []
                Y_rows: list[np.ndarray] = []
                for t in range(n_ep):
                    tau = t / max(1, n_ep - 1)
                    X_rows.append(
                        _encode_policy_features(ep_q[t], ep_g[t], tau, anom_bit, loc_pos, loc_yaw)
                    )
                    idxs = np.minimum(np.arange(t, t + chunk_h), n_ep - 1)
                    Y_rows.append(A[idxs].reshape(-1))

                policy = ChunkedMLPPolicy(chunk_horizon=chunk_h)
                train_mse = policy.fit(np.asarray(X_rows), np.asarray(Y_rows))

                # Reset twin to the Real2Sim perceived initial scene and deploy the learned policy closed-loop!
                twin.reset_episode_state()
                twin.reset_arm_to_home()
                twin.sync_real2sim_scene(scene_state, settle_steps=15)
                twin.status_banner = (
                    f"Real2Sim Closed-Loop Policy Rollout -> {target_pile.upper()} Plate"
                )
                t_cur = 0
                stride = 4
                while t_cur < n_ep:
                    tau = min(1.0, t_cur / max(1, n_ep - 1))
                    feat = _encode_policy_features(
                        twin.current_arm_q(), twin.gripper_pct, tau, anom_bit, loc_pos, loc_yaw
                    )
                    chunk = policy.predict_chunk(feat)
                    for row in chunk[: min(stride, n_ep - t_cur)]:
                        q_next = np.clip(
                            row[:5],
                            twin.ik.cal_kin.calibration.lower,
                            twin.ik.cal_kin.calibration.upper,
                        )
                        twin.step_control_30hz(q_next, gripper_pct=float(np.clip(row[5], 0.0, 1.0)))
                        t_cur += 1
                for _ in range(10):
                    twin.step_control_30hz(twin.q_cmd, gripper_pct=0.75)

                p_final = twin.ground_truth_block_pose()[:3, 3]
                dest_xy = scene_state.plates[target_pile].pos_world[:2]
                xy_err = float(np.linalg.norm(p_final[:2] - dest_xy))
                exec_ok = xy_err <= 0.088 and float(p_final[2]) < 0.075
                if exec_ok:
                    twin.sorted_pile = target_pile
                    twin.sorted_counts[target_pile] = 1
                    twin.remaining_source_blocks = max(0, len(twin.spawned_blocks) - 1)
                twin._sync_viewer()

            if show_viewer and twin.is_viewer_running() and hold_viewer_seconds > 0.0:
                deadline = time.monotonic() + float(hold_viewer_seconds)
                while twin.is_viewer_running() and time.monotonic() < deadline:
                    twin._sync_viewer()
                    time.sleep(0.05)

            clean_records = []
            for rec in twin.batch_sorted_records:
                r_copy = dict(rec)
                r_copy.pop("keyframe_rgb", None)
                clean_records.append(r_copy)

            report_path = out_root / "real2sim_closed_loop_report.json"
            report = Real2SimClosedLoopReport(
                provenance=scene_state.provenance,
                initial_scene=scene_state.to_summary_dict(),
                replicated_brick_count=len(scene_state.bricks),
                mean_replication_pos_err_mm=round(mean_err_mm, 2),
                distractor_counts=scene_state.distractor_counts_by_category(),
                controller_type=controller,
                execution_succeeded=exec_ok,
                sorted_counts=dict(twin.sorted_counts),
                batch_sorted_records=clean_records,
                policy_train_mse=round(train_mse, 7) if train_mse is not None else None,
                elapsed_s=round(time.monotonic() - t0, 2),
                report_json_path=str(report_path),
            )
            report_path.write_text(json.dumps(report.__dict__, indent=2) + "\n", encoding="utf-8")
            return report
        finally:
            twin.close()

    def run_live_real2sim_mirror(
        self,
        live_depth_url: str = "http://127.0.0.1:8094",
        armd_url: str = "http://127.0.0.1:8765",
        ssh_host: str = "intel",
        auto_ssh_tunnel: bool = True,
        prefer_snapshots: bool = False,
        camera_mode: Literal["top", "plate", "fused"] = "top",
        show_viewer: bool = True,
        duration_s: float = 0.0,
        max_brick_slots: int = 10,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run the real-time live Real2Sim mirror in MuJoCo 4.0.1 Desktop connected to the physical cell.

        - Automatically opens/verifies the SSH port-forward tunnel to `intel` (`:8094` and `:8765`).
        - Continuously polls the real SO-101 robot arm state (`GET /arms/follower/state`) at ~25-30 Hz
          and mirrors `q_enc_rad` and `gripper_pct` directly into MuJoCo (`twin.set_arm_state_direct`).
        - Continuously polls the RealSense D415 depth point cloud (`camera_mode="top"` by default,
          using `/points/top` or `/snapshot/top`) in a background perception thread, localizing all 3
          IKEA KALAS plates and all DUPLO bricks (resting in plates, on table, or lifted in-air / in-gripper)
          while rejecting human arms, laptops, wires, and trash.
        - Updates plate positions (`model.body_pos`), brick 6D poses (`data.qpos`), brick STL meshes
          (`model.geom_dataid`), and perceived LEGO RGB colors (`model.geom_rgba`) live in MuJoCo Desktop.
        """
        out_root = (
            Path(output_dir)
            if output_dir is not None
            else (REPO_ROOT / "artifacts" / "block_inspector" / "real2sim")
        )
        out_root.mkdir(parents=True, exist_ok=True)

        tunnel_proc: subprocess.Popen[bytes] | None = None
        if auto_ssh_tunnel:
            tunnel_proc = ensure_intel_ssh_tunnel(
                ssh_host=ssh_host,
                live_depth_url=live_depth_url,
                armd_url=armd_url,
            )

        if self.tracker is None:
            self.tracker = Real2SimSceneTracker(
                ema_alpha=0.62, max_assoc_dist_m=0.045, max_missed_frames=6
            )

        # Fetch initial arm state and initial depth point-cloud scene
        q_init, grip_init, arm_meta = fetch_live_armd_follower_state(armd_url, timeout_s=1.2)
        initial_scene = self.perceive_from_live_depth_http(
            base_url=live_depth_url,
            q_arm_rad=q_init,
            timeout_s=2.5,
            prefer_snapshots=prefer_snapshots,
            camera_mode=camera_mode,
        )

        # Pre-allocate `max_brick_slots` dynamic brick bodies in MuJoCo so bricks can appear, move,
        # or be picked up in real time without recompiling the MJCF model.
        slot_specs = list(DUPLO_CATALOG.values())
        preallocated_blocks: list[SpawnedBlockInfo] = []
        for idx in range(max(max_brick_slots, len(initial_scene.bricks), 1)):
            if idx < len(initial_scene.bricks):
                b = initial_scene.bricks[idx]
                preallocated_blocks.append(
                    SpawnedBlockInfo(
                        block_id=f"block_{idx}",
                        spec=b.spec,
                        initial_xy_yaw=b.xy_yaw,
                        has_visual_defect=b.has_visual_defect,
                        initial_T_world_block=b.T_world_block.copy(),
                    )
                )
            else:
                spec_i = slot_specs[idx % len(slot_specs)]
                preallocated_blocks.append(
                    SpawnedBlockInfo(
                        block_id=f"block_{idx}",
                        spec=spec_i,
                        initial_xy_yaw=(0.0, 0.0, 0.0),
                        has_visual_defect=False,
                        initial_T_world_block=np.array(
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, -0.50 - 0.05 * idx],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            dtype=np.float64,
                        ),
                    )
                )

        twin = SO101InspectionTwin(
            block_spec=preallocated_blocks[0].spec,
            rig=self.rig,
            initial_block_xy_yaw=preallocated_blocks[0].initial_xy_yaw,
            show_viewer=show_viewer,
            show_camera_windows=False,
            realtime_pacing=False,
            spawned_blocks=preallocated_blocks,
            plate_positions=initial_scene.plate_positions_dict(),
        )

        latest_lock = threading.Lock()
        latest_q: np.ndarray | None = q_init
        latest_grip: float = grip_init
        latest_scene: Real2SimSceneState = initial_scene
        scene_dirty: bool = True
        stop_event = threading.Event()
        depth_frames_processed = 1
        arm_updates_processed = 1 if q_init is not None else 0

        def _depth_worker() -> None:
            nonlocal latest_scene, scene_dirty, depth_frames_processed
            while not stop_event.is_set():
                with latest_lock:
                    q_cur = latest_q.copy() if latest_q is not None else None
                try:
                    sc = self.perceive_from_live_depth_http(
                        base_url=live_depth_url,
                        q_arm_rad=q_cur,
                        timeout_s=1.8,
                        prefer_snapshots=prefer_snapshots,
                        camera_mode=camera_mode,
                    )
                    if sc.raw_point_count > 0:
                        with latest_lock:
                            latest_scene = sc
                            scene_dirty = True
                            depth_frames_processed += 1
                except Exception:
                    pass
                time.sleep(0.02)

        worker = threading.Thread(target=_depth_worker, name="real2sim-live-depth", daemon=True)
        worker.start()

        t_start = time.monotonic()
        last_log_t = 0.0
        try:
            if q_init is not None:
                twin.set_arm_state_direct(q_init, grip_init, sync_viewer=False)
            twin.sync_real2sim_scene(initial_scene, settle_steps=0)

            while True:
                elapsed = time.monotonic() - t_start
                if duration_s > 0.0 and elapsed >= duration_s:
                    break
                if show_viewer and not twin.is_viewer_running():
                    break

                # 1. Poll physical SO-101 robot arm state at ~25-30 Hz
                q_now, grip_now, _ = fetch_live_armd_follower_state(armd_url, timeout_s=0.18)
                if q_now is not None:
                    with latest_lock:
                        latest_q = q_now
                        latest_grip = grip_now
                        arm_updates_processed += 1
                    twin.set_arm_state_direct(q_now, grip_now, sync_viewer=False)
                    if self.tracker is not None:
                        try:
                            T_ee = T_WORLD_BASE @ self._cal_kin.frames(q_now[:5])["Body3"]
                            latched_bricks = self.tracker.update_latched_brick_from_arm(
                                T_ee, grip_now
                            )
                            if any(b.support_region == "in_gripper" for b in latched_bricks):
                                with latest_lock:
                                    latest_scene.bricks = latched_bricks
                                    scene_dirty = True
                        except Exception:
                            pass

                # 2. Apply latest perceived 3D scene (plates + DUPLO bricks) to MuJoCo
                sc_to_apply: Real2SimSceneState | None = None
                with latest_lock:
                    if scene_dirty:
                        sc_to_apply = latest_scene
                        scene_dirty = False
                    sc_cur = latest_scene
                    q_disp = latest_q
                    g_disp = latest_grip

                if sc_to_apply is not None:
                    twin.sync_real2sim_scene(sc_to_apply, settle_steps=0)
                else:
                    twin._sync_viewer()

                q0_deg = math.degrees(float(q_disp[0])) if q_disp is not None else 0.0
                g_pct_disp = float(g_disp) * 100.0 if float(g_disp) <= 1.05 else float(g_disp)
                twin.status_banner = (
                    f"LIVE REAL2SIM MIRROR | depth #{depth_frames_processed} ({sc_cur.latency_ms:.0f}ms) | "
                    f"{len(sc_cur.bricks)} bricks | 3 plates | arm q0={q0_deg:+.1f}deg grip={g_pct_disp:.1f}%"
                )

                if elapsed - last_log_t >= 2.0:
                    last_log_t = elapsed
                    src_p = sc_cur.plates["source"].pos_world
                    print(
                        f"[Real2Sim Live] t={elapsed:5.1f}s | depth_frames={depth_frames_processed} "
                        f"| arm_polls={arm_updates_processed} | bricks={len(sc_cur.bricks)} "
                        f"| source_plate=({src_p[0]:.3f},{src_p[1]:.3f}) "
                        f"| grip={g_pct_disp:.1f}%",
                        flush=True,
                    )

                time.sleep(0.033)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            worker.join(timeout=1.5)
            twin.close()
            if tunnel_proc is not None and tunnel_proc.poll() is None:
                try:
                    tunnel_proc.terminate()
                except Exception:
                    pass

        with latest_lock:
            final_scene = latest_scene
            final_q = latest_q
            final_grip = latest_grip

        summary = {
            "mode": "live_real2sim_mirror",
            "camera_mode": camera_mode,
            "live_depth_url": live_depth_url,
            "armd_url": armd_url,
            "elapsed_s": round(time.monotonic() - t_start, 2),
            "depth_frames_processed": depth_frames_processed,
            "arm_updates_processed": arm_updates_processed,
            "latest_arm_q_rad": final_q.tolist() if final_q is not None else None,
            "latest_gripper_pct": round(float(final_grip), 2),
            "armd_connected": bool(arm_meta.get("ok", False) or final_q is not None),
            "latest_scene": final_scene.to_summary_dict(),
        }
        report_path = out_root / "real2sim_live_mirror_report.json"
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary["report_json_path"] = str(report_path)
        return summary


_ARMD_STATE_DECODER: ArmdFollowerBridge | None = None


def decode_armd_follower_state_dict(
    doc: dict[str, Any],
) -> tuple[np.ndarray | None, float]:
    """Decode `GET /arms/follower/state` JSON into `(q_enc_rad, gripper_pct_0_to_1)`."""
    global _ARMD_STATE_DECODER
    if _ARMD_STATE_DECODER is None:
        _ARMD_STATE_DECODER = ArmdFollowerBridge(request_fn=lambda *_: {})
    dec_state = _ARMD_STATE_DECODER.decode_state_message(doc, stop_on_fault=False)
    if dec_state is not None:
        return dec_state.q_enc_rad.copy(), float(dec_state.gripper_pct)
    dec = doc.get("dec") or {}
    q_rad = dec.get("q_enc_rad")
    if isinstance(q_rad, list) and len(q_rad) >= 5:
        g_raw = float(dec.get("gripper_pct", 0.0))
        g_norm = g_raw / 100.0 if g_raw > 1.05 else g_raw
        return np.asarray(q_rad[:5], dtype=np.float64), float(np.clip(g_norm, 0.0, 1.0))
    return None, 0.0


def fetch_live_armd_follower_state(
    armd_url: str = "http://127.0.0.1:8765",
    timeout_s: float = 0.25,
) -> tuple[np.ndarray | None, float, dict[str, Any]]:
    """Fetch live physical SO-101 follower arm joint angles (rad) and gripper (0..1) from `armd`."""
    url = f"{armd_url.rstrip('/')}/arms/follower/state"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
        q_rad, grip = decode_armd_follower_state_dict(doc)
        if q_rad is not None:
            return q_rad, grip, doc
    except Exception:
        pass
    return None, 0.0, {}


def _is_local_port_open(host: str, port: int, timeout_s: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def ensure_intel_ssh_tunnel(
    ssh_host: str = "intel",
    live_depth_url: str = "http://127.0.0.1:8094",
    armd_url: str = "http://127.0.0.1:8765",
) -> subprocess.Popen[bytes] | None:
    """Ensure localhost port-forwarding to `intel` for `live_depth.py` (`:8094`) and `armd` (`:8765`)."""
    depth_port = int(live_depth_url.rstrip("/").split(":")[-1].split("/")[0])
    armd_port = int(armd_url.rstrip("/").split(":")[-1].split("/")[0])
    if _is_local_port_open("127.0.0.1", depth_port) and _is_local_port_open("127.0.0.1", armd_port):
        return None

    try:
        proc = subprocess.Popen(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=6",
                "-o",
                "ExitOnForwardFailure=yes",
                "-N",
                "-L",
                f"{depth_port}:127.0.0.1:{depth_port}",
                "-L",
                f"{armd_port}:127.0.0.1:{armd_port}",
                ssh_host,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _is_local_port_open("127.0.0.1", depth_port) or _is_local_port_open(
                "127.0.0.1", armd_port
            ):
                return proc
            if proc.poll() is not None:
                break
            time.sleep(0.15)
        return proc
    except Exception:
        return None
