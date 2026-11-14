"""Canonical frames, 3-camera extrinsics/intrinsics rig, and LEGO DUPLO 8-corner geometry."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DUPLO_ASSETS_DIR = REPO_ROOT / "assets" / "duplo"
DEFAULT_SCENE_YAML = REPO_ROOT / "configs" / "sim-scene.yaml"
PRIORS_JSON_PATH = (
    REPO_ROOT / "configs" / "published_priors_verified.json"
    if (REPO_ROOT / "configs" / "published_priors_verified.json").is_file()
    else REPO_ROOT / "artifacts" / "dynamics" / "published_priors_verified.json"
)
PIXEL_PITCH_M = 3e-6


def _load_verified_cad_registrations() -> tuple[float, np.ndarray]:
    """Load verified base height and Body3<-camera_mount registration from published_priors_verified.json."""
    base_z_m = 0.02932886598977181
    T_b3_cm = np.eye(4, dtype=np.float64)
    if PRIORS_JSON_PATH.is_file():
        raw = json.loads(PRIORS_JSON_PATH.read_text(encoding="utf-8"))
        regs = raw.get("registrations", {})
        if "Body5_1" in regs:
            base_z_m = abs(float(regs["Body5_1"]["t_source_to_local_m"][2]))
        if "Body3" in regs:
            T_b3_cm[:3, :3] = np.asarray(regs["Body3"]["R_source_to_local"], dtype=np.float64)
            T_b3_cm[:3, 3] = np.asarray(regs["Body3"]["t_source_to_local_m"], dtype=np.float64)
    return base_z_m, T_b3_cm


BASE_ORIGIN_Z_M, T_BODY3_CAMERA_MOUNT = _load_verified_cad_registrations()

# Canonical base-to-world SE(3): R_z(+90 deg) + verified 29.33 mm base-plate height above z=0 table.
# Maps native SO-101 Body5_1 (-Y forward, +X left, +Z up) -> REP-103 world (+X forward, +Y left, +Z up).
T_WORLD_BASE = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, BASE_ORIGIN_Z_M],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

T_BASE_WORLD = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -BASE_ORIGIN_Z_M],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# OpenCV optical frame (+X right, +Y down, +Z forward) <-> MuJoCo camera frame (+X right, +Y up, -Z forward)
R_CV_TO_MJ = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

# Canonical 8 material corner signs s^(i) in {-1, +1}^3:
# c_0..c_3 = canonical -Z_B bottom corners; c_4..c_7 = canonical +Z_B top corners.
CORNER_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [-1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)

CORNER_NAMES = tuple(f"corner_{i}" for i in range(8))

# Measured open-mouth grasp TCP in Body3 local frame:
# Fixed jaw inner face is at x = -0.008 m, open moving jaw at x = +0.048 m => symmetric clearance at x = +0.015 m, z = -0.088 m.
NOMINAL_TCP_OFFSET_BODY3 = np.array([0.015, 0.000, -0.088], dtype=np.float64)


def nominal_canted_wrist_cam_transform(cant_deg: float = 25.0) -> np.ndarray:
    """Nominal 25-deg downward-canted top-wrist bracket T_Body3_cam (OpenCV +Z optical forward).

    In Body3 at horizontal reference:
      -Z_Body3 points forward toward gripper jaws,
      +Y_Body3 points upward,
      +X_Body3 points rightward.
    Returns a proper SE(3) matrix (det(R) == +1).
    """
    g = math.radians(cant_deg)
    cg, sg = math.cos(g), math.sin(g)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.000],
            [0.0, -cg, -sg, +0.045],
            [0.0, +sg, -cg, -0.020],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def look_at_cv_transform(
    eye_world: np.ndarray, target_world: np.ndarray, up_world: np.ndarray | None = None
) -> np.ndarray:
    """Construct SE(3) T_world_cvcam (+Z optical forward toward target, +X right, +Y down)."""
    eye = np.asarray(eye_world, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    up = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if up_world is None
        else np.asarray(up_world, dtype=np.float64)
    )
    z_axis = target - eye
    norm_z = float(np.linalg.norm(z_axis))
    if norm_z < 1e-9:
        raise ValueError("Camera eye and target cannot coincide")
    z_axis /= norm_z
    if abs(float(np.dot(z_axis, up))) > 0.995:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    out = np.eye(4, dtype=np.float64)
    out[:3, 0] = x_axis
    out[:3, 1] = y_axis
    out[:3, 2] = z_axis
    out[:3, 3] = eye
    return out


def cv_cam_to_mujoco_cam(T_parent_cvcam: np.ndarray) -> np.ndarray:
    """Convert OpenCV camera pose (+Z forward, +Y down) to MuJoCo camera pose (-Z forward, +Y up)."""
    T_mj = np.array(T_parent_cvcam, dtype=np.float64, copy=True)
    T_mj[:3, :3] = T_mj[:3, :3] @ R_CV_TO_MJ
    return T_mj


def mujoco_cam_to_cv_cam(T_parent_mjcam: np.ndarray) -> np.ndarray:
    """Convert MuJoCo camera pose (-Z forward, +Y up) to OpenCV camera pose (+Z forward, +Y down)."""
    T_cv = np.array(T_parent_mjcam, dtype=np.float64, copy=True)
    T_cv[:3, :3] = T_cv[:3, :3] @ R_CV_TO_MJ
    return T_cv


def intrinsics_from_fovy(width: int, height: int, fovy_deg: float) -> np.ndarray:
    """Compute pinhole intrinsics matrix K for an ideal MuJoCo / calibrated RGB-D camera."""
    fy = 0.5 * height / math.tan(0.5 * math.radians(fovy_deg))
    fx = fy
    cx = 0.5 * width
    cy = 0.5 * height
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CameraModel:
    """Intrinsic and extrinsic calibration for one static or wrist-mounted camera."""

    name: str
    alias: str  # Teammate bench alias: "overhead", "side", or "wrist"
    serial_or_dev: str
    width: int
    height: int
    fovy_deg: float
    K: np.ndarray  # (3, 3) float64
    T_parent_cvcam: np.ndarray  # (4, 4) float64: T_world_cam for static, T_body3_cam for wrist
    is_wrist: bool = False
    has_depth: bool = True
    explicit_intrinsics: tuple[float, float, float, float] | None = None

    def mujoco_sensor_attrs(self) -> dict[str, str] | None:
        """MuJoCo 4.0.1 <camera> sensor attributes for exact off-center pinhole intrinsics."""
        if self.explicit_intrinsics is None:
            return None
        fx, fy, cx, cy = self.explicit_intrinsics
        return {
            "resolution": f"{self.width} {self.height}",
            "sensorsize": f"{self.width * PIXEL_PITCH_M:.8g} {self.height * PIXEL_PITCH_M:.8g}",
            "focalpixel": f"{fx:.8g} {fy:.8g}",
            "principalpixel": (
                f"{(self.width - 1) / 2.0 - cx:.8g} {(self.height - 1) / 2.0 - cy:.8g}"
            ),
        }

    def project_world_points(
        self, points_world: np.ndarray, T_world_body3: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project (N, 3) world points into pixel coordinates (u, v) and optical depth z_cam."""
        T_world_cam = self.world_transform(T_world_body3)
        R_wc = T_world_cam[:3, :3]
        p_wc = T_world_cam[:3, 3]
        pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
        pts_cam = (pts - p_wc) @ R_wc
        z = pts_cam[:, 2]
        z_safe = np.where(np.abs(z) > 1e-9, z, 1e-9)
        u = self.K[0, 0] * (pts_cam[:, 0] / z_safe) + self.K[0, 2]
        v = self.K[1, 1] * (pts_cam[:, 1] / z_safe) + self.K[1, 2]
        return np.column_stack([u, v]), z

    def world_transform(self, T_world_body3: np.ndarray | None = None) -> np.ndarray:
        if not self.is_wrist:
            return self.T_parent_cvcam
        if T_world_body3 is None:
            raise ValueError("T_world_body3 is required to resolve wrist camera world pose")
        return np.asarray(T_world_body3, dtype=np.float64) @ self.T_parent_cvcam


@dataclass(frozen=True)
class CameraExtrinsicsRig:
    """Calibrated 3-camera inspection cell rig (2x static RealSense D415 + 1x wrist UVC)."""

    realsense_1: CameraModel  # Serial 318123021715, bench alias "overhead"
    realsense_2: CameraModel  # Serial 344223025431, bench alias "side"
    wrist: CameraModel  # Sonix UVC /dev/video44, bench alias "wrist"

    def cameras(self) -> tuple[CameraModel, CameraModel, CameraModel]:
        return (self.realsense_1, self.realsense_2, self.wrist)

    def by_name(self, name: str) -> CameraModel:
        for cam in self.cameras():
            if name in (cam.name, cam.alias):
                return cam
        raise KeyError(f"Unknown camera name or alias: {name!r}")

    def to_dict(self) -> dict:
        return {
            "schema": 1,
            "frame_convention": "world_plus_x_forward_cv_plus_z_optical",
            "cameras": {
                cam.name: {
                    "alias": cam.alias,
                    "serial_or_dev": cam.serial_or_dev,
                    "width": cam.width,
                    "height": cam.height,
                    "fovy_deg": cam.fovy_deg,
                    "K": cam.K.tolist(),
                    "T_parent_cvcam": cam.T_parent_cvcam.tolist(),
                    "is_wrist": cam.is_wrist,
                    "has_depth": cam.has_depth,
                }
                for cam in self.cameras()
            },
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> CameraExtrinsicsRig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cams = raw["cameras"]
        built = {}
        for name, row in cams.items():
            built[name] = CameraModel(
                name=name,
                alias=row["alias"],
                serial_or_dev=row["serial_or_dev"],
                width=int(row["width"]),
                height=int(row["height"]),
                fovy_deg=float(row["fovy_deg"]),
                K=np.asarray(row["K"], dtype=np.float64),
                T_parent_cvcam=np.asarray(row["T_parent_cvcam"], dtype=np.float64),
                is_wrist=bool(row["is_wrist"]),
                has_depth=bool(row["has_depth"]),
            )
        return cls(
            realsense_1=built["realsense_1"],
            realsense_2=built["realsense_2"],
            wrist=built["wrist"],
        )


def nominal_camera_rig(
    width: int = 640, height: int = 480, wrist_cant_deg: float = 25.0
) -> CameraExtrinsicsRig:
    """Default 3-camera inspection rig matching ARCHITECTURE.md §2.2, §3.4 & multicam aliases.

    Complementary ~105-deg azimuth separation between RealSense #1 (front-right overhead)
    and RealSense #2 (front-left side) observes 4-5 faces of any tabletop cuboid simultaneously
    within < 0.40 m metric depth range. Replace via CameraExtrinsicsRig.load_json(...) once
    physical ChArUco extrinsics are captured.
    """
    target = np.array([0.23, 0.00, 0.015], dtype=np.float64)
    T_rs1 = look_at_cv_transform(np.array([0.44, -0.18, 0.24]), target)
    T_rs2 = look_at_cv_transform(np.array([0.38, +0.22, 0.21]), target)
    T_wrist = nominal_canted_wrist_cam_transform(wrist_cant_deg)

    K_d415 = intrinsics_from_fovy(width, height, fovy_deg=42.5)
    K_wrist = intrinsics_from_fovy(width, height, fovy_deg=52.0)

    return CameraExtrinsicsRig(
        realsense_1=CameraModel(
            name="realsense_1",
            alias="overhead",
            serial_or_dev="318123021715",
            width=width,
            height=height,
            fovy_deg=42.5,
            K=K_d415,
            T_parent_cvcam=T_rs1,
            is_wrist=False,
            has_depth=True,
        ),
        realsense_2=CameraModel(
            name="realsense_2",
            alias="side",
            serial_or_dev="344223025431",
            width=width,
            height=height,
            fovy_deg=42.5,
            K=K_d415,
            T_parent_cvcam=T_rs2,
            is_wrist=False,
            has_depth=True,
        ),
        wrist=CameraModel(
            name="wrist",
            alias="wrist",
            serial_or_dev="/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
            width=width,
            height=height,
            fovy_deg=52.0,
            K=K_wrist,
            T_parent_cvcam=T_wrist,
            is_wrist=True,
            has_depth=False,
        ),
    )


def _ortho_rot(mat_3x3) -> np.ndarray:
    m = np.asarray(mat_3x3, dtype=np.float64).reshape(3, 3)
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _parse_scene_yaml_camera(
    entry: dict,
    canonical_name: str,
    alias: str,
    serial_or_dev: str,
    is_wrist: bool,
    has_depth: bool,
) -> CameraModel:
    width = int(entry.get("width", 640))
    height = int(entry.get("height", 480))
    T_parent_cv = np.eye(4, dtype=np.float64)
    T_parent_cv[:3, 3] = np.asarray(entry["pos"], dtype=np.float64)
    if "rotation_cv" in entry:
        T_parent_cv[:3, :3] = _ortho_rot(entry["rotation_cv"])
    elif "lookat" in entry:
        up = np.asarray(entry.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)
        T_parent_cv = look_at_cv_transform(T_parent_cv[:3, 3], np.asarray(entry["lookat"]), up)
    else:
        raise ValueError(f"Unsupported camera orientation in sim-scene.yaml for {alias!r}")

    if is_wrist and entry.get("attach") == "camera_mount":
        # Map Menagerie camera_mount frame onto calibrated SO-101 Body3 frame via verified STL registration
        T_parent_cv = T_BODY3_CAMERA_MOUNT @ T_parent_cv

    if "fx" in entry and "fy" in entry:
        fx = float(entry["fx"])
        fy = float(entry["fy"])
        cx = float(entry.get("cx", 0.5 * (width - 1)))
        cy = float(entry.get("cy", 0.5 * (height - 1)))
        fovy_deg = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        explicit = (fx, fy, cx, cy)
    else:
        fovy_deg = float(entry.get("fovy", 42.5))
        K = intrinsics_from_fovy(width, height, fovy_deg=fovy_deg)
        explicit = None

    return CameraModel(
        name=canonical_name,
        alias=alias,
        serial_or_dev=serial_or_dev,
        width=width,
        height=height,
        fovy_deg=fovy_deg,
        K=K,
        T_parent_cvcam=T_parent_cv,
        is_wrist=is_wrist,
        has_depth=has_depth,
        explicit_intrinsics=explicit,
    )


@dataclass(frozen=True)
class PlateSpec:
    """IKEA KALAS sorting/source plate pose and RGBA color in +X-forward world frame."""

    name: str  # "source", "good", or "defective"
    pos_m: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


# 8 non-overlapping placement slot XY offsets (meters) inside an Ø150 mm KALAS plate floor:
# Slot 0 at plate center, Slots 1..7 around an r = 42 mm ring so up to 8 bricks settle cleanly.
PLATE_DROP_SLOT_OFFSETS_XY: tuple[tuple[float, float], ...] = (
    (0.000, 0.000),
    *(
        (
            round(0.042 * math.cos(2.0 * math.pi * k / 7.0), 5),
            round(0.042 * math.sin(2.0 * math.pi * k / 7.0), 5),
        )
        for k in range(7)
    ),
)

DEFAULT_PLATES: dict[str, PlateSpec] = {
    "source": PlateSpec(
        name="source",
        pos_m=(0.23, 0.00, 0.0),
        rgba=(0.28, 0.82, 0.80, 1.0),  # Aquamarine blue input cluster plate
    ),
    "good": PlateSpec(
        name="good",
        pos_m=(0.10, -0.22, 0.0),
        rgba=(0.62, 0.86, 0.76, 1.0),
    ),
    "defective": PlateSpec(
        name="defective",
        pos_m=(0.10, +0.22, 0.0),
        rgba=(0.90, 0.72, 0.78, 1.0),
    ),
}


def station_plates_config(scene_yaml_path: str | Path | None = None) -> dict[str, PlateSpec]:
    """Load the 3 IKEA KALAS plates (`source`, `good`, `defective`) from configs/sim-scene.yaml."""
    env_override = os.environ.get("BRICKCELL_SCENE_CONFIG")
    path = Path(scene_yaml_path or env_override or DEFAULT_SCENE_YAML)
    out = dict(DEFAULT_PLATES)
    if not path.is_file():
        return out
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    plates_cfg = raw.get("plates", {})
    for pname, default_spec in DEFAULT_PLATES.items():
        entry = plates_cfg.get(pname)
        if not isinstance(entry, dict):
            continue
        col = entry.get("color", default_spec.rgba[:3])
        if isinstance(col, str) and col.startswith("#") and len(col) == 7:
            rgb = (
                int(col[1:3], 16) / 255.0,
                int(col[3:5], 16) / 255.0,
                int(col[5:7], 16) / 255.0,
            )
        else:
            rgb = (float(col[0]), float(col[1]), float(col[2]))
        pos_raw = entry.get("pos", default_spec.pos_m)
        pos = (float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
        out[pname] = PlateSpec(name=pname, pos_m=pos, rgba=(*rgb, 1.0))
    return out


def station_camera_rig(scene_yaml_path: str | Path | None = None) -> CameraExtrinsicsRig:
    """Load the measured physical station camera rig from configs/sim-scene.yaml (`sim/station-twin`)."""
    env_override = os.environ.get("BRICKCELL_SCENE_CONFIG")
    path = Path(scene_yaml_path or env_override or DEFAULT_SCENE_YAML)
    if not path.is_file():
        return nominal_camera_rig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cams = raw.get("cameras", {})
    if not {"overhead", "side", "wrist"} <= set(cams):
        return nominal_camera_rig()
    return CameraExtrinsicsRig(
        realsense_1=_parse_scene_yaml_camera(
            cams["overhead"],
            canonical_name="realsense_1",
            alias="overhead",
            serial_or_dev="350122060653",
            is_wrist=False,
            has_depth=True,
        ),
        realsense_2=_parse_scene_yaml_camera(
            cams["side"],
            canonical_name="realsense_2",
            alias="side",
            serial_or_dev="243622060450",
            is_wrist=False,
            has_depth=True,
        ),
        wrist=_parse_scene_yaml_camera(
            cams["wrist"],
            canonical_name="wrist",
            alias="wrist",
            serial_or_dev="/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
            is_wrist=True,
            has_depth=False,
        ),
    )


@dataclass(frozen=True)
class BlockSpec:
    """Physical and geometric specification of a rectangular / LEGO DUPLO inspection block.

    Body frame B is centered at the cuboid's geometric midpoint (z_B in [-H/2, +H/2]),
    while LDraw STL/OBJ meshes have their origin at bottom-center (offset (0, 0, -H/2) in B).
    """

    part_id: str
    name: str
    dims_m: tuple[float, float, float]  # (L_x, W_y, H_z) bounding box for 8 corners
    body_height_m: float  # solid wall height excluding top studs
    mass_kg: float
    rgba: tuple[float, float, float, float]
    symmetry_group: str  # "D4", "D2", or "C1"
    mesh_stem: str | None = None

    @property
    def stl_path(self) -> Path | None:
        if not self.mesh_stem:
            return None
        return DUPLO_ASSETS_DIR / f"{self.mesh_stem}.stl"

    @property
    def obj_path(self) -> Path | None:
        if not self.mesh_stem:
            return None
        return DUPLO_ASSETS_DIR / f"{self.mesh_stem}.obj"

    def corners_local(self) -> np.ndarray:
        """Return (8, 3) canonical corner coordinates in body-centered frame B."""
        half = 0.5 * np.asarray(self.dims_m, dtype=np.float64)
        return CORNER_SIGNS * half[None, :]

    def in_plane_yaw_symmetries(self) -> tuple[float, ...]:
        """Discrete in-plane tabletop yaw symmetries (radians) preserving cuboid shape."""
        if self.symmetry_group == "D4":
            return (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
        if self.symmetry_group == "D2":
            return (0.0, math.pi)
        return (0.0,)

    def obstacle_spheres_local(self) -> tuple[np.ndarray, np.ndarray]:
        """Cover the cuboid with 4 protective collision spheres in body-centered frame B."""
        lx, wy, hz = self.dims_m
        radius = float(max(0.018, 0.55 * min(lx, wy, hz) + 0.006))
        if lx >= wy:
            offsets = np.linspace(-0.35 * lx, 0.35 * lx, 4)
            centers = np.column_stack([offsets, np.zeros(4), np.zeros(4)])
        else:
            offsets = np.linspace(-0.35 * wy, 0.35 * wy, 4)
            centers = np.column_stack([np.zeros(4), offsets, np.zeros(4)])
        return centers.astype(np.float64), np.full(4, radius, dtype=np.float64)


# Official LDraw CC-BY-4.0 LEGO DUPLO bricks extracted into assets/duplo/
DUPLO_CATALOG: dict[str, BlockSpec] = {
    "duplo_2x2_3437": BlockSpec(
        part_id="3437",
        name="duplo_2x2_3437",
        dims_m=(0.0320, 0.0320, 0.0236),
        body_height_m=0.0192,
        mass_kg=0.0064,
        rgba=(0.80, 0.08, 0.08, 1.0),
        symmetry_group="D4",
        mesh_stem="duplo_2x2_3437",
    ),
    "duplo_2x4_3011": BlockSpec(
        part_id="3011",
        name="duplo_2x4_3011",
        dims_m=(0.0640, 0.0320, 0.0236),
        body_height_m=0.0192,
        mass_kg=0.0112,
        rgba=(0.05, 0.33, 0.80, 1.0),
        symmetry_group="D2",
        mesh_stem="duplo_2x4_3011",
    ),
    "duplo_1x2x2_76371": BlockSpec(
        part_id="76371",
        name="duplo_1x2x2_76371",
        dims_m=(0.0320, 0.0160, 0.0428),
        body_height_m=0.0384,
        mass_kg=0.0074,
        rgba=(0.97, 0.76, 0.05, 1.0),
        symmetry_group="D2",
        mesh_stem="duplo_1x2x2_76371",
    ),
    "duplo_2x3_curved_top_2302": BlockSpec(
        part_id="2302",
        name="duplo_2x3_curved_top_2302",
        dims_m=(0.0320, 0.0480, 0.0236),
        body_height_m=0.0192,
        mass_kg=0.0076,
        rgba=(0.10, 0.60, 0.28, 1.0),
        symmetry_group="C1",
        mesh_stem="duplo_2x3_curved_top_2302",
    ),
}


def corner_bisectors_local() -> np.ndarray:
    """Outward unit vertex bisectors b_hat_i = s^(i) / sqrt(3) in body frame B, shape (8, 3)."""
    return CORNER_SIGNS / math.sqrt(3.0)


def corner_facet_normals_local() -> np.ndarray:
    """Three orthogonal outward facet normals per corner in body frame B, shape (8, 3, 3)."""
    normals = np.zeros((8, 3, 3), dtype=np.float64)
    for i, (sx, sy, sz) in enumerate(CORNER_SIGNS):
        normals[i, 0] = (sx, 0.0, 0.0)
        normals[i, 1] = (0.0, sy, 0.0)
        normals[i, 2] = (0.0, 0.0, sz)
    return normals


def corner_positions_world(
    T_world_block: np.ndarray, dims_m: tuple[float, float, float] | np.ndarray
) -> np.ndarray:
    """Compute (8, 3) material corner positions in +X-forward world coordinates."""
    T = np.asarray(T_world_block, dtype=np.float64)
    half = 0.5 * np.asarray(dims_m, dtype=np.float64)
    local = CORNER_SIGNS * half[None, :]
    return (local @ T[:3, :3].T) + T[:3, 3]


def corner_facet_normals_world(T_world_block: np.ndarray) -> np.ndarray:
    """Compute (8, 3, 3) outward facet normals in +X-forward world coordinates.

    For each upper-hemisphere corner, reorders the 3 facet normals so row 2 is the most
    upward-facing facet (+Z_world dot n_f maximal), matching build_corner_cone_ref's [side1, side2, top] convention.
    """
    R = np.asarray(T_world_block, dtype=np.float64)[:3, :3]
    raw = np.einsum("ij,kfj->kfi", R, corner_facet_normals_local())
    ordered = np.empty_like(raw)
    for k in range(8):
        z_dots = raw[k, :, 2]
        top_idx = int(np.argmax(z_dots))
        side_idxs = [idx for idx in range(3) if idx != top_idx]
        # Deterministic side ordering via cross-product handedness around +Z
        n0, n1 = raw[k, side_idxs[0]], raw[k, side_idxs[1]]
        if np.cross(n0, n1)[2] < 0:
            n0, n1 = n1, n0
        ordered[k, 0] = n0
        ordered[k, 1] = n1
        ordered[k, 2] = raw[k, top_idx]
    return ordered


def corner_upper_hemisphere_mask(T_world_block: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """Evaluate CornerUpperHemisphere(b, c_i): z_world · (R_world_B b_hat_i) > 0.25.

    Evaluates to +0.5774 > 0.25 on exact 4/8 exposed top corners across all 6 resting cuboid faces
    (0 deg, +/-90 deg roll/pitch, and 180 deg flip).
    """
    R = np.asarray(T_world_block, dtype=np.float64)[:3, :3]
    bisectors_world = corner_bisectors_local() @ R.T
    return bisectors_world[:, 2] > threshold


def so3_geodesic_angle_rad(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle ||Log(R_a^T R_b)|| in [0, pi] radians on SO(3)."""
    R_rel = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    cos_theta = float(np.clip(0.5 * (np.trace(R_rel) - 1.0), -1.0, 1.0))
    return math.acos(cos_theta)
