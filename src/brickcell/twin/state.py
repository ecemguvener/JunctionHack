"""Shared contract of the station twin. numpy only: no MuJoCo, no SciPy, no RealSense import.

Frames and units
  World frame = robot base frame of configs/sim-scene.yaml: origin at the SO-101 base on the
  table, +Z up, +X from the base to the bricks. Lengths in metres. Quaternions (w, x, y, z).

  Brick pose = pose of the MESH frame in the world: the frame of assets/duplo/<mesh>.stl
  (origin at the bottom centre of the part, +Z up through the studs). The stable resting
  orientations in assets/duplo/stable_poses.json use the same frame.

  The MuJoCo body of a brick is NOT the mesh frame: its origin is the grasp point. Use
  ``body_pose_from_mesh_pose`` / ``mesh_pose_from_body_pose`` to convert.

  Plate pose = position of the plate body origin (centre of the plate, on the table). Plates
  turn about Z only; a KALAS plate is round, so yaw is not observable and is not stored.
  Plates: "good" (green, sort target), "defective" (pink, sort target), "spare" (blue, not a
  sort target).

  Brick copies: the sim has INSTANCES_PER_PART bodies of each part. The first copy has the
  part id ("part-002"); the others add "-2", "-3", ... ("part-002-2"). The pick-and-sort demo
  uses only the first copies. ``part_of(object_id)`` gives the Part of any copy.

  Arm = follower joint angles in the MuJoCo joint convention (radians), names as in the model:
  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll; gripper as the MuJoCo joint
  value (radians).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
PLATE_NAMES = ("good", "defective", "spare")
SORT_PLATES = ("good", "defective")  # the pick-and-sort targets
INSTANCES_PER_PART = 4


@dataclass(frozen=True)
class Part:
    """Geometry that connects the mesh frame to the MuJoCo body of one brick (see world.BRICKS)."""

    object_id: str  # MuJoCo body name, e.g. "part-001"
    part: str  # LEGO DUPLO part number
    mesh: str  # file stem in assets/duplo
    mesh_yaw: float  # rad; the mesh geom is turned by this yaw about the body Z axis
    grip_z: float  # m; body origin = mesh origin + grip_z along the mesh Z axis


# Must match brickcell.simulation.world.BRICKS (tests/test_twin_state.py checks it).
PARTS: dict[str, Part] = {
    "part-001": Part("part-001", "3437", "duplo_2x2_3437", 0.0, 0.0105),
    "part-002": Part("part-002", "3011", "duplo_2x4_3011", 0.0, 0.0105),
    "part-003": Part("part-003", "76371", "duplo_1x2x2_76371", math.pi / 2, 0.017),
    "part-004": Part("part-004", "2302", "duplo_2x3_curved_top_2302", -math.pi / 2, 0.0105),
}
PART_BY_NUMBER = {p.part: p for p in PARTS.values()}


def instance_id(part_id: str, index: int) -> str:
    """Body name of copy ``index`` (1-based) of a part: 1 -> "part-002", 2 -> "part-002-2"."""
    if part_id not in PARTS or not 1 <= index <= INSTANCES_PER_PART:
        raise ValueError(f"No brick copy {index} of {part_id!r}")
    return part_id if index == 1 else f"{part_id}-{index}"


# All brick bodies of the sim, first copies first: part-001..part-004, part-001-2, ...
BRICK_IDS: tuple[str, ...] = tuple(
    instance_id(part_id, k) for k in range(1, INSTANCES_PER_PART + 1) for part_id in PARTS
)
_PART_OF_ID = {
    instance_id(part_id, k): part_id for part_id in PARTS for k in range(1, INSTANCES_PER_PART + 1)
}


def part_of(object_id: str) -> Part:
    """The Part of a brick body (any copy)."""
    try:
        return PARTS[_PART_OF_ID[object_id]]
    except KeyError:
        raise ValueError(f"Unknown brick {object_id!r}") from None


# D415 serials (RealSense SDK) -> camera names in configs/sim-scene.yaml.
CAMERA_SERIALS = {"overhead": "350122060653", "side": "243622060450"}


# --------------------------------------------------------------------------- rotations


def quat_to_matrix(quat) -> np.ndarray:
    w, x, y, z = np.asarray(quat, float) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(rotation) -> np.ndarray:
    """Unit quaternion (w, x, y, z) with w >= 0."""
    m = np.asarray(rotation, float)
    k = np.array(
        [
            [m[0, 0] - m[1, 1] - m[2, 2], 0, 0, 0],
            [m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], 0, 0],
            [m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1], 0],
            [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]],
        ]
    )
    values, vectors = np.linalg.eigh(k / 3.0, UPLO="L")
    x, y, z, w = vectors[:, int(np.argmax(values))]
    quat = np.array([w, x, y, z])
    return quat if w >= 0 else -quat


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_angle_deg(a, b) -> float:
    """Angle of the rotation between two rotation matrices, degrees."""
    r = np.asarray(a, float).T @ np.asarray(b, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def body_pose_from_mesh_pose(object_id: str, pos, quat) -> tuple[np.ndarray, np.ndarray]:
    """MuJoCo body (pos, quat) of a brick from its mesh-frame pose in the world."""
    part = part_of(object_id)
    r_mesh = quat_to_matrix(quat)
    r_body = r_mesh @ rot_z(-part.mesh_yaw)
    p_body = np.asarray(pos, float) + r_body @ np.array([0.0, 0.0, part.grip_z])
    return p_body, matrix_to_quat(r_body)


def mesh_pose_from_body_pose(object_id: str, pos, quat) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``body_pose_from_mesh_pose``."""
    part = part_of(object_id)
    r_body = quat_to_matrix(quat)
    p_mesh = np.asarray(pos, float) - r_body @ np.array([0.0, 0.0, part.grip_z])
    return p_mesh, matrix_to_quat(r_body @ rot_z(part.mesh_yaw))


# --------------------------------------------------------------------------- sensor data


@dataclass
class RGBDFrame:
    """One camera frame. Depth is aligned to the colour image (same size, same intrinsics)."""

    camera: str  # name in configs/sim-scene.yaml ("overhead", "side")
    rgb: np.ndarray  # (H, W, 3) uint8, RGB order
    depth_m: np.ndarray  # (H, W) float32 metres along the optical axis; 0 = no data
    fx: float
    fy: float
    cx: float
    cy: float
    timestamp: float = 0.0  # unix seconds
    serial: str = ""
    synthetic: bool = False  # True for MuJoCo renders


def load_fixture(directory: str | Path, reduce: str = "median") -> list[RGBDFrame]:
    """Load a fixture written by the station capture (meta.json + <camera>.npz).

    Each npz has rgb (N, H, W, 3) uint8, depth (N, H, W) uint16 raw units, timestamps_ms (N,).
    ``reduce="median"`` gives one frame per camera: per-pixel median of the valid depth values
    (0 where fewer than half of the frames are valid) and the last colour image.
    """
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    frames = []
    for camera, info in meta.items():
        if not isinstance(info, dict) or not (directory / f"{camera}.npz").exists():
            continue
        data = np.load(directory / f"{camera}.npz")
        depth = data["depth"].astype(np.float32) * float(info["depth_scale_m"])
        if reduce == "median":
            valid = depth > 0
            stack = np.where(valid, depth, np.nan)
            with np.errstate(all="ignore"):
                med = np.nanmedian(stack, axis=0)
            med[valid.sum(0) < max(1, depth.shape[0] // 2)] = 0
            depth_m = np.nan_to_num(med, nan=0.0).astype(np.float32)
        else:
            depth_m = depth[-1]
        frames.append(
            RGBDFrame(
                camera=camera,
                rgb=data["rgb"][-1],
                depth_m=depth_m,
                fx=float(info["fx"]),
                fy=float(info["fy"]),
                cx=float(info["cx"]),
                cy=float(info["cy"]),
                timestamp=float(data["timestamps_ms"][-1]) / 1000.0,
                serial=str(info.get("serial", "")),
            )
        )
    return frames


# --------------------------------------------------------------------------- station state


@dataclass
class BrickState:
    object_id: str  # sim body name: "part-00N" or a copy "part-00N-K" (see BRICK_IDS)
    part: str  # DUPLO part number
    pos: list[float]  # mesh-frame origin in the world (m)
    quat: list[float]  # mesh-frame orientation in the world (w, x, y, z)
    color: list[float]  # observed mean RGB, 0..1
    stable_pose: str | None = None  # id in stable_poses.json, None if not snapped
    support: str = "table"  # "table", "plate_good", "plate_defective", "other"
    confidence: float = 0.0  # 0..1
    residual_mm: float | None = None  # median model-to-cloud distance of the fit
    cameras: list[str] = field(default_factory=list)  # cameras that saw it
    color_name: str | None = None  # nearest colour in configs/brick-palette.yaml


@dataclass
class PlateState:
    name: str  # "good", "defective" or "spare"
    pos: list[float]  # plate body origin in the world (m); z = 0 on the table
    confidence: float = 0.0


@dataclass
class ArmState:
    joints: dict[str, float]  # MuJoCo joint name -> rad (ARM_JOINTS)
    gripper: float | None = None  # MuJoCo gripper joint value (rad)
    timestamp: float = 0.0
    source: str = ""  # e.g. "servo-readonly", "armd", "fixture"


@dataclass
class StationState:
    timestamp: float  # unix seconds of the newest sensor frame
    bricks: list[BrickState] = field(default_factory=list)
    plates: list[PlateState] = field(default_factory=list)
    arm: ArmState | None = None
    cameras: dict[str, dict] = field(default_factory=dict)  # name -> {serial, timestamp, ...}
    diagnostics: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict) -> StationState:
        if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported station state schema {raw.get('schema_version')}")
        try:
            bricks = [BrickState(**b) for b in raw.get("bricks", [])]
            plates = [PlateState(**p) for p in raw.get("plates", [])]
            arm = ArmState(**raw["arm"]) if raw.get("arm") else None
        except TypeError as exc:
            raise ValueError(f"Bad station state: {exc}") from None
        for b in bricks:
            if b.object_id not in _PART_OF_ID:
                raise ValueError(f"Unknown brick {b.object_id!r}")
            if b.part != part_of(b.object_id).part:
                raise ValueError(f"{b.object_id} is part {part_of(b.object_id).part}, not {b.part}")
            if len(b.pos) != 3 or len(b.quat) != 4:
                raise ValueError(f"Bad pose for {b.object_id}")
            if not all(math.isfinite(v) for v in (*b.pos, *b.quat)):
                raise ValueError(f"Non-finite pose for {b.object_id}")
        for p in plates:
            if p.name not in PLATE_NAMES or len(p.pos) != 3:
                raise ValueError(f"Bad plate {p.name!r}")
        if arm is not None and set(arm.joints) - set(ARM_JOINTS):
            raise ValueError(f"Unknown arm joints {sorted(set(arm.joints) - set(ARM_JOINTS))}")
        return cls(
            timestamp=float(raw["timestamp"]),
            bricks=bricks,
            plates=plates,
            arm=arm,
            cameras=dict(raw.get("cameras", {})),
            diagnostics=dict(raw.get("diagnostics", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> StationState:
        return cls.from_dict(json.loads(text))
