"""Scene configuration for the MuJoCo simulation: camera poses, intrinsics and colours.

This module is pure Python (numpy and PyYAML only). It must not import mujoco, because the core
environment loads and tests it. ``world.py`` turns the result into the MuJoCo model.

Frames and units
- Lengths are in metres. ``fovy`` is in degrees. Pixel values use the OpenCV convention: the
  centre of the top-left pixel is (0, 0), u goes right and v goes down.
- The world frame is the robot base frame: origin at the SO-101 base on the table, +Z up, +X from
  the base to the bricks. A camera with ``attach`` uses the frame of that body.
- MuJoCo camera frame: +X right, +Y up, the camera looks along -Z.
- OpenCV camera frame: +X right, +Y down, the camera looks along +Z.
- Conversion (camera-to-parent rotations): ``R_mujoco = R_opencv @ diag(1, -1, -1)``.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "sim-scene.yaml"
ENV_VAR = "BRICKCELL_SCENE_CONFIG"
CV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])
DEFAULT_SIZE = (256, 256)  # width, height of a camera that has no configured resolution
INSPECTION_SIZE = (256, 256)  # the detector crop in world.inspection_crop needs this size
MAX_SIZE = 4096
PIXEL_PITCH = 3e-6  # m; only the ratio focal length / sensor size changes the image

# Built-in defaults. configs/sim-scene.yaml repeats them; a test keeps the two equal.
DEFAULT_BRICK_COLORS: dict[str, tuple[float, float, float]] = {
    "part-001": (0.80, 0.08, 0.08),
    "part-002": (0.05, 0.33, 0.80),
    "part-003": (0.97, 0.76, 0.05),
    "part-004": (0.10, 0.60, 0.28),
}
DEFAULT_PLATE_COLORS: dict[str, tuple[float, float, float]] = {
    "good": (0.62, 0.86, 0.76),
    "defective": (0.90, 0.72, 0.78),
    # Blue spare plate (not a sort target): real KALAS colour in the overhead D415 image
    # (fixture-001, median of the plate floor, RGB 15 173 207) x 1.08 sim lighting gain.
    "spare": (0.06, 0.73, 0.88),
}
# Floor: the table surface at z = 0 that bricks and plates stand on. Light brown, as the real
# wooden table: the rendered colour matches the median wood colour RGB (161, 118, 61) of the real
# D415 images (fixture-001, 2026-11-14) under the sim lights.
DEFAULT_FLOOR_COLOR: tuple[float, float, float] = (0.68, 0.50, 0.26)
DEFAULT_CAMERAS: dict[str, dict] = {
    "inspection": {
        "pos": [0.55, 0, 0.24],
        "xyaxes": [0, 1, 0, -0.287348, 0, 0.957826],
        "fovy": 22,
        "width": 256,
        "height": 256,
    },
    "overview": {
        "pos": [0.3, 0, 1.1],
        "xyaxes": [1, 0, 0, 0, 1, 0],
        "fovy": 50,
        "width": 256,
        "height": 256,
    },
}

ORIENTATION_KEYS = ("lookat", "xyaxes", "quat", "rotation_cv")
INTRINSIC_KEYS = ("fx", "fy", "cx", "cy")
CAMERA_KEYS = {"pos", *ORIENTATION_KEYS, "up", "fovy", *INTRINSIC_KEYS, "width", "height"} | {
    "attach",
    "enabled",
}
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*")


class SceneConfigError(ValueError):
    """The scene config has a value that cannot be used."""


# ------------------------------------------------------------------------------- colours


def parse_color(value: object, where: str = "color") -> tuple[float, float, float]:
    """Return (r, g, b) in 0-1 from ``"#rrggbb"`` or from three numbers in 0-1."""
    if isinstance(value, str):
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", value.strip()):
            raise SceneConfigError(f'{where}: "{value}" is not a colour. Use "#rrggbb".')
        text = value.strip()
        return tuple(int(text[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]
    try:
        rgb = _numbers(value, 3, where)
    except SceneConfigError:
        raise SceneConfigError(
            f'{where}: use "#rrggbb" or three numbers [r, g, b] from 0 to 1.'
        ) from None
    if min(rgb) < 0 or max(rgb) > 1:
        raise SceneConfigError(
            f'{where}: the numbers {list(rgb)} must be from 0 to 1. For 0-255 values use "#rrggbb".'
        )
    return rgb  # type: ignore[return-value]


def color_hex(rgb) -> str:
    return "#" + "".join(f"{int(round(255 * min(max(float(v), 0.0), 1.0))):02x}" for v in rgb[:3])


def parse_brick_color_arg(text: str) -> tuple[str, tuple[float, float, float]]:
    """Parse a command-line value such as ``part-001=#ff0000`` or ``part-001=0.8,0.1,0.1``."""
    part, sep, value = text.partition("=")
    if not sep:
        raise SceneConfigError(f'"{text}": use PART=COLOUR, for example part-001=#ff0000.')
    part, value = part.strip(), value.strip()
    _known(part, DEFAULT_BRICK_COLORS, "brick colour", "part id")
    if value.startswith("#"):
        return part, parse_color(value, f"brick colour {part}")
    try:
        numbers = [float(v) for v in value.split(",")]
    except ValueError:
        numbers = None
    return part, parse_color(numbers, f"brick colour {part}")


# ------------------------------------------------------------------------------ rotations


def rotation_from_lookat(pos, lookat, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera-to-parent rotation of a camera at ``pos`` that looks at ``lookat``.

    ``up`` is the direction that points to the top of the image, as near as possible.
    """
    forward = np.asarray(lookat, float) - np.asarray(pos, float)
    distance = np.linalg.norm(forward)
    if distance < 1e-9:
        raise SceneConfigError("lookat must be a different point from pos.")
    forward /= distance
    right = np.cross(forward, np.asarray(up, float))
    if np.linalg.norm(right) < 1e-6:
        raise SceneConfigError(
            "up is parallel to the view direction. Give a different up, e.g. up: [1, 0, 0]."
        )
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(right, forward), -forward))


def rotation_from_xyaxes(xyaxes) -> np.ndarray:
    """MuJoCo ``xyaxes``: the camera X axis, then the Y axis (made orthogonal to X)."""
    x, y = np.asarray(xyaxes[:3], float), np.asarray(xyaxes[3:], float)
    if np.linalg.norm(x) < 1e-9:
        raise SceneConfigError("xyaxes: the X axis is zero.")
    x = x / np.linalg.norm(x)
    y = y - x * (x @ y)
    if np.linalg.norm(y) < 1e-9:
        raise SceneConfigError("xyaxes: the X and Y axes are parallel.")
    y /= np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


def rotation_from_quat(quat) -> np.ndarray:
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(quat, float) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_rotation(rotation) -> np.ndarray:
    """Unit quaternion (w, x, y, z) with w >= 0 of a rotation matrix."""
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


def mujoco_from_opencv(rotation_cv) -> np.ndarray:
    """Camera-to-parent rotation in the MuJoCo camera frame from the OpenCV camera frame."""
    return np.asarray(rotation_cv, float) @ CV_TO_MUJOCO


def opencv_from_mujoco(rotation) -> np.ndarray:
    """Camera-to-parent rotation in the OpenCV camera frame from the MuJoCo camera frame."""
    return np.asarray(rotation, float) @ CV_TO_MUJOCO


def pose_from_extrinsics(rotation_world_to_camera, translation) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV extrinsics (x_cam = R x_world + t, e.g. from solvePnP) for the config.

    Returns ``(pos, rotation_cv)``: the camera position in the world and the camera-to-world
    rotation in the OpenCV camera frame.
    """
    r = np.asarray(rotation_world_to_camera, float)
    return -r.T @ np.asarray(translation, float), r.T


def project(points, pos, rotation_cv, fx, fy, cx, cy) -> np.ndarray:
    """Pinhole projection of parent-frame points to OpenCV pixels (u, v). No lens distortion."""
    local = (np.atleast_2d(np.asarray(points, float)) - np.asarray(pos, float)) @ np.asarray(
        rotation_cv, float
    )
    return np.column_stack(
        (fx * local[:, 0] / local[:, 2] + cx, fy * local[:, 1] / local[:, 2] + cy)
    )


def fovy_from_focal(fy: float, height: float) -> float:
    """Vertical field of view in degrees of a pinhole camera."""
    return math.degrees(2 * math.atan(height / (2 * fy)))


def focal_from_fovy(fovy: float, height: float) -> float:
    """Focal length in pixels (fx = fy) of a MuJoCo camera with a vertical field of view."""
    return height / (2 * math.tan(math.radians(fovy) / 2))


# -------------------------------------------------------------------------------- cameras


@dataclass(frozen=True)
class Camera:
    """One camera, resolved from the config. Use ``parse_camera`` to make it."""

    name: str
    pos: tuple[float, float, float]  # in the world, or in the ``attach`` body frame
    # What the scene XML gets: ("xyaxes", 6 numbers) or ("quat", w x y z), MuJoCo camera frame.
    orientation: tuple[str, tuple[float, ...]]
    fovy: float  # degrees, vertical; computed from fy and height if intrinsics are given
    width: int
    height: int
    intrinsics: tuple[float, float, float, float] | None = None  # fx, fy, cx, cy (OpenCV pixels)
    attach: str | None = None  # body name; None = fixed in the world
    enabled: bool = True

    def rotation(self) -> np.ndarray:
        """Camera-to-parent rotation, MuJoCo camera frame (+X right, +Y up, looks along -Z)."""
        kind, values = self.orientation
        return rotation_from_xyaxes(values) if kind == "xyaxes" else rotation_from_quat(values)

    def rotation_cv(self) -> np.ndarray:
        """Camera-to-parent rotation, OpenCV camera frame (+X right, +Y down, looks along +Z)."""
        return opencv_from_mujoco(self.rotation())

    def quat(self) -> np.ndarray:
        """MuJoCo camera quaternion (w, x, y, z) in the parent frame."""
        return quat_from_rotation(self.rotation())

    def pinhole(self) -> tuple[float, float, float, float]:
        """fx, fy, cx, cy of the rendered image in OpenCV pixels."""
        if self.intrinsics is not None:
            return self.intrinsics
        focal = focal_from_fovy(self.fovy, self.height)
        return focal, focal, (self.width - 1) / 2, (self.height - 1) / 2

    def sensor(self) -> dict[str, tuple[float, float]] | None:
        """MuJoCo sensor attributes for a camera with intrinsics; None for a plain fovy camera.

        MuJoCo measures the principal point from the image centre, and a positive value moves
        the optical axis to the left (x) or up (y) in the image. The image centre is
        ((W-1)/2, (H-1)/2) in OpenCV pixels. A render check confirmed the signs (MuJoCo 4.0.1).
        """
        if self.intrinsics is None:
            return None
        fx, fy, cx, cy = self.intrinsics
        return {
            "resolution": (self.width, self.height),
            "sensorsize": (self.width * PIXEL_PITCH, self.height * PIXEL_PITCH),
            "focalpixel": (fx, fy),
            "principalpixel": ((self.width - 1) / 2 - cx, (self.height - 1) / 2 - cy),
        }

    def xml(self) -> str:
        """The ``<camera>`` element for the scene XML."""
        kind, values = self.orientation
        attrs = f'name="{self.name}" pos="{_text(self.pos)}" {kind}="{_text(values)}"'
        sensor = self.sensor()
        if sensor is None:
            return f'<camera {attrs} fovy="{self.fovy!r}"/>'
        extra = " ".join(f'{key}="{_text(value)}"' for key, value in sensor.items())
        return f"<camera {attrs} {extra}/>"

    def as_dict(self) -> dict:
        fx, fy, cx, cy = self.pinhole()
        return {
            "name": self.name,
            "enabled": self.enabled,
            "attach": self.attach,
            "pos": list(self.pos),
            "quat": self.quat().tolist(),
            "rotation_cv": self.rotation_cv().tolist(),
            "fovy": self.fovy,
            "width": self.width,
            "height": self.height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }


def parse_camera(name: str, entry: object) -> Camera:
    where = f"camera '{name}'"
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise SceneConfigError(f"{where}: use only letters, digits, '_' and '-' in the name.")
    if not isinstance(entry, Mapping):
        raise SceneConfigError(f"{where}: must be a mapping with pos, an orientation and fovy.")
    unknown = sorted(set(entry) - CAMERA_KEYS)
    if unknown:
        raise SceneConfigError(f"{where}: unknown keys {unknown}. Allowed: {sorted(CAMERA_KEYS)}.")
    if "pos" not in entry:
        raise SceneConfigError(f"{where}: pos [x, y, z] is necessary.")
    pos = _numbers(entry["pos"], 3, f"{where}: pos")

    given = [key for key in ORIENTATION_KEYS if key in entry]
    if len(given) != 1:
        problem = f"{given} are given" if given else "none is given"
        raise SceneConfigError(
            f"{where}: give exactly one of {list(ORIENTATION_KEYS)}, but {problem}."
        )
    kind = given[0]
    if "up" in entry and kind != "lookat":
        raise SceneConfigError(f"{where}: up is only for lookat.")
    try:
        if kind == "lookat":
            up = _numbers(entry.get("up", (0, 0, 1)), 3, "up")
            rotation = rotation_from_lookat(pos, _numbers(entry["lookat"], 3, "lookat"), up)
            orientation = ("quat", tuple(quat_from_rotation(rotation).tolist()))
        elif kind == "xyaxes":
            values = _numbers(entry["xyaxes"], 6, "xyaxes")
            rotation_from_xyaxes(values)  # reject zero or parallel axes
            orientation = ("xyaxes", values)
        elif kind == "quat":
            values = np.array(_numbers(entry["quat"], 4, "quat"))
            if np.linalg.norm(values) < 1e-9:
                raise SceneConfigError("quat: must not be zero.")
            orientation = ("quat", tuple((values / np.linalg.norm(values)).tolist()))
        else:
            rotation = mujoco_from_opencv(_rotation_matrix(entry["rotation_cv"]))
            orientation = ("quat", tuple(quat_from_rotation(rotation).tolist()))
    except SceneConfigError as exc:
        raise SceneConfigError(f"{where}: {exc}") from None

    intrinsic = [key for key in INTRINSIC_KEYS if key in entry]
    width = _size(entry.get("width"), f"{where}: width")
    height = _size(entry.get("height"), f"{where}: height")
    if intrinsic and "fovy" in entry:
        raise SceneConfigError(f"{where}: give fovy or the intrinsics {intrinsic}, not both.")
    if intrinsic:
        if "fx" not in entry or "fy" not in entry:
            raise SceneConfigError(f"{where}: intrinsics need fx and fy (cx and cy are optional).")
        if width is None or height is None:
            raise SceneConfigError(
                f"{where}: intrinsics need width and height (the image size of the calibration)."
            )
        fx, fy = (_positive(entry[key], f"{where}: {key}") for key in ("fx", "fy"))
        cx = _numbers([entry.get("cx", (width - 1) / 2)], 1, f"{where}: cx")[0]
        cy = _numbers([entry.get("cy", (height - 1) / 2)], 1, f"{where}: cy")[0]
        intrinsics, fovy = (fx, fy, cx, cy), fovy_from_focal(fy, height)
    elif "fovy" in entry:
        intrinsics, fovy = None, _positive(entry["fovy"], f"{where}: fovy")
        if fovy >= 180:
            raise SceneConfigError(f"{where}: fovy must be less than 180 degrees.")
    else:
        raise SceneConfigError(f"{where}: give fovy (degrees) or the intrinsics fx, fy, cx, cy.")

    attach = entry.get("attach")
    if attach is not None and (not isinstance(attach, str) or not NAME.fullmatch(attach)):
        raise SceneConfigError(f"{where}: attach must be a body name, e.g. camera_mount.")
    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        raise SceneConfigError(f"{where}: enabled must be true or false.")
    return Camera(
        name=name,
        pos=pos,  # type: ignore[arg-type]
        orientation=orientation,
        fovy=fovy,
        width=width or DEFAULT_SIZE[0],
        height=height or DEFAULT_SIZE[1],
        intrinsics=intrinsics,
        attach=attach,
        enabled=enabled,
    )


# ---------------------------------------------------------------------------------- scene


@dataclass(frozen=True)
class SceneConfig:
    cameras: dict[str, Camera]  # every entry, also the disabled ones
    brick_colors: dict[str, tuple[float, float, float]]
    plate_colors: dict[str, tuple[float, float, float]]
    source: str = "built-in defaults"
    floor_color: tuple[float, float, float] = DEFAULT_FLOOR_COLOR

    def world_cameras(self) -> list[Camera]:
        """Enabled cameras that are fixed in the world."""
        return [c for c in self.cameras.values() if c.enabled and c.attach is None]

    def attached_cameras(self) -> list[Camera]:
        """Enabled cameras that are mounted on a body."""
        return [c for c in self.cameras.values() if c.enabled and c.attach is not None]

    def camera_size(self, name: str) -> tuple[int, int]:
        """Render size (width, height) of a camera; DEFAULT_SIZE if it is not in the config."""
        camera = self.cameras.get(name)
        return (camera.width, camera.height) if camera and camera.enabled else DEFAULT_SIZE

    def offscreen_size(self) -> tuple[int, int]:
        """Offscreen buffer (width, height) that is large enough for every enabled camera."""
        enabled = [c for c in self.cameras.values() if c.enabled]
        return (
            max([640, *(c.width for c in enabled)]),
            max([480, *(c.height for c in enabled)]),
        )

    def with_brick_colors(self, colors: Mapping[str, object]) -> SceneConfig:
        """A copy with other brick colours ({part id: colour})."""
        merged = dict(self.brick_colors)
        for part, value in colors.items():
            _known(part, DEFAULT_BRICK_COLORS, "brick colour", "part id")
            merged[part] = parse_color(value, f"brick colour {part}")
        return replace(self, brick_colors=merged)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "cameras": [camera.as_dict() for camera in self.cameras.values()],
            "bricks": _colors_dict(self.brick_colors),
            "plates": _colors_dict(self.plate_colors),
            "floor": _colors_dict({"floor": self.floor_color})["floor"],
        }


def parse_scene_config(raw: object, source: str = "dict") -> SceneConfig:
    """Validate a config mapping. Entries that are not there keep the built-in defaults."""
    raw = {} if raw is None else raw
    if not isinstance(raw, Mapping):
        raise SceneConfigError(f"Scene config ({source}): the top level must be a mapping.")
    unknown = sorted(set(raw) - {"cameras", "bricks", "plates", "floor"})
    if unknown:
        raise SceneConfigError(
            f"Scene config ({source}): unknown keys {unknown}. "
            "Allowed: cameras, bricks, plates, floor."
        )
    try:
        entries = {**DEFAULT_CAMERAS, **_mapping(raw.get("cameras"), "cameras")}
        cameras = {name: parse_camera(name, entry) for name, entry in entries.items()}
        inspection = cameras["inspection"]
        if not inspection.enabled or (inspection.width, inspection.height) != INSPECTION_SIZE:
            raise SceneConfigError(
                "camera 'inspection': must be enabled and 256x256. The detector crop needs it."
            )
        return SceneConfig(
            cameras=cameras,
            brick_colors=_colors(raw.get("bricks"), DEFAULT_BRICK_COLORS, "bricks", "part id"),
            plate_colors=_colors(raw.get("plates"), DEFAULT_PLATE_COLORS, "plates", "plate"),
            source=source,
            floor_color=_floor_color(raw.get("floor")),
        )
    except SceneConfigError as exc:
        raise SceneConfigError(f"Scene config ({source}): {exc}") from None


def load_scene_config(source: object = None) -> SceneConfig:
    """Load the scene config.

    ``source`` is a SceneConfig, a mapping, a YAML path, or None. None uses the path in the
    BRICKCELL_SCENE_CONFIG environment variable, then configs/sim-scene.yaml, then the built-in
    defaults if that file is not there.
    """
    if isinstance(source, SceneConfig):
        return source
    if isinstance(source, Mapping):
        return parse_scene_config(source)
    if source is None:
        source = os.environ.get(ENV_VAR) or None
        if source is None:
            if not DEFAULT_PATH.is_file():
                return parse_scene_config({}, "built-in defaults")
            source = DEFAULT_PATH
    path = Path(source)  # type: ignore[arg-type]
    if not path.is_file():
        raise SceneConfigError(f"Scene config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise SceneConfigError(f"Scene config ({path}): YAML error: {exc}") from None
    return parse_scene_config(raw, str(path))


# -------------------------------------------------------------------------------- helpers


def _text(values) -> str:
    return " ".join(repr(v) for v in values)


def _known(key: object, allowed: Mapping, what: str, kind: str) -> None:
    if key not in allowed:
        raise SceneConfigError(f"{what}: unknown {kind} '{key}'. Known: {list(allowed)}.")


def _mapping(value: object, where: str) -> Mapping:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SceneConfigError(f"{where}: must be a mapping.")
    return value


def _numbers(value: object, count: int, where: str) -> tuple[float, ...]:
    ok = isinstance(value, (list, tuple)) and len(value) == count
    if ok:
        ok = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in value  # type: ignore[union-attr]
        )
    if not ok:
        raise SceneConfigError(f"{where}: must be a list of {count} numbers, not {value!r}.")
    return tuple(float(v) for v in value)  # type: ignore[union-attr]


def _positive(value: object, where: str) -> float:
    number = _numbers([value], 1, where)[0]
    if number <= 0:
        raise SceneConfigError(f"{where}: must be more than 0.")
    return number


def _size(value: object, where: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_SIZE:
        raise SceneConfigError(f"{where}: must be a whole number of pixels from 1 to {MAX_SIZE}.")
    return value


def _rotation_matrix(value: object) -> np.ndarray:
    """A 3x3 rotation from 3 rows of 3 numbers (or 9 numbers); small rounding errors are fixed."""
    rows = value
    if isinstance(value, (list, tuple)) and len(value) == 3:
        if all(isinstance(row, (list, tuple)) for row in value):
            rows = [v for row in value for v in row]
    matrix = np.array(_numbers(rows, 9, "rotation_cv")).reshape(3, 3)
    if np.abs(matrix.T @ matrix - np.eye(3)).max() > 1e-3:
        raise SceneConfigError(
            "rotation_cv: is not a rotation matrix (the columns must be unit length and "
            "at right angles). Give the 3x3 camera-to-world rotation."
        )
    if np.linalg.det(matrix) < 0:
        raise SceneConfigError("rotation_cv: has determinant -1 (a mirror), not a rotation.")
    u, _, vt = np.linalg.svd(matrix)
    return u @ vt


def _colors(value: object, defaults: Mapping, where: str, kind: str) -> dict:
    colors = dict(defaults)
    for key, entry in _mapping(value, where).items():
        _known(key, defaults, where, kind)
        if not isinstance(entry, Mapping) or set(entry) != {"color"}:
            raise SceneConfigError(
                f"{where}: '{key}' must be a mapping with one key, e.g. color: \"#ff0000\"."
            )
        colors[key] = parse_color(entry["color"], f"{where}: '{key}' color")
    return colors


def _floor_color(value: object) -> tuple[float, float, float]:
    if value is None:
        return DEFAULT_FLOOR_COLOR
    if not isinstance(value, Mapping) or set(value) != {"color"}:
        raise SceneConfigError('floor: must be a mapping with one key, e.g. color: "#050505".')
    return parse_color(value["color"], "floor color")


def _colors_dict(colors: Mapping) -> dict:
    return {key: {"color": color_hex(rgb), "rgb": list(rgb)} for key, rgb in colors.items()}
