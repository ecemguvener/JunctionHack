"""RGB-D acquisition for the station: two RealSense D415 cameras, or a saved fixture.

Imports: numpy, the standard library and (only in live mode, lazily) pyrealsense2.

One snapshot = N frames of each camera. ``RawCapture.to_frame`` makes one RGBDFrame: the
per-pixel median of the valid depth values (0 where fewer than half of the frames are valid)
and the last colour image. No hole-filling filter is used: it invents depth.

Fixture format (the same as artifacts/twin-fixtures/fixture-001):
  meta.json      {camera: {serial, depth_scale_m, width, height, fx, fy, cx, cy,
                  distortion_model, coeffs, depth_stream, frames, firmware}, captured_utc: ...}
  <camera>.npz   rgb (N, H, W, 3) uint8, depth (N, H, W) uint16 raw units, timestamps_ms (N,)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from brickcell.twin.state import CAMERA_SERIALS, RGBDFrame

log = logging.getLogger("brickcell.twin.realsense")

COLOR_SIZE = (640, 480)
DEPTH_SIZE = (1280, 720)
DEPTH_STREAM_TEXT = "1280x720 z16 aligned to colour 640x480"


class CameraError(RuntimeError):
    """A camera cannot be used."""


class CameraBusyError(CameraError):
    """Another process holds the camera."""


class CameraNotFoundError(CameraError):
    """No camera with this serial is connected."""


def median_depth(depth_raw: np.ndarray, depth_scale_m: float) -> np.ndarray:
    """Per-pixel median of the valid depth values in metres; 0 where < half are valid.

    The same rule as ``state.load_fixture(reduce="median")``.
    """
    raw = np.asarray(depth_raw)
    if raw.ndim == 2:
        return raw.astype(np.float32) * np.float32(depth_scale_m)
    # Sort along the frames: the invalid zeros come first, the valid values after them.
    n = raw.shape[0]
    ordered = np.sort(raw, axis=0)
    count = (raw > 0).sum(0)
    first = n - count
    lo = np.take_along_axis(ordered, (first + (count - 1) // 2).clip(0, n - 1)[None], 0)[0]
    hi = np.take_along_axis(ordered, (first + count // 2).clip(0, n - 1)[None], 0)[0]
    med = (
        lo.astype(np.float32) * np.float32(depth_scale_m)
        + hi.astype(np.float32) * np.float32(depth_scale_m)
    ) / np.float32(2)
    med[count < max(1, n // 2)] = 0
    return med.astype(np.float32)


@dataclass
class RawCapture:
    """All raw frames of one camera from one snapshot."""

    camera: str
    serial: str
    rgb: np.ndarray  # (N, H, W, 3) uint8
    depth: np.ndarray  # (N, H, W) uint16
    timestamps_ms: np.ndarray  # (N,) unix milliseconds
    depth_scale_m: float
    intrinsics: dict  # width, height, fx, fy, cx, cy, distortion_model, coeffs
    firmware: str = ""
    extra: dict = field(default_factory=dict)

    def to_frame(self) -> RGBDFrame:
        k = self.intrinsics
        return RGBDFrame(
            camera=self.camera,
            rgb=self.rgb[-1],
            depth_m=median_depth(self.depth, self.depth_scale_m),
            fx=float(k["fx"]),
            fy=float(k["fy"]),
            cx=float(k["cx"]),
            cy=float(k["cy"]),
            timestamp=float(self.timestamps_ms[-1]) / 1000.0,
            serial=self.serial,
        )

    def meta(self) -> dict:
        k = self.intrinsics
        return {
            "serial": self.serial,
            "depth_scale_m": float(self.depth_scale_m),
            "width": int(k["width"]),
            "height": int(k["height"]),
            "fx": float(k["fx"]),
            "fy": float(k["fy"]),
            "cx": float(k["cx"]),
            "cy": float(k["cy"]),
            "distortion_model": str(k.get("distortion_model", "")),
            "coeffs": [float(c) for c in k.get("coeffs", [])],
            "depth_stream": DEPTH_STREAM_TEXT,
            "frames": int(len(self.rgb)),
            "firmware": self.firmware,
        }


def save_fixture(directory: str | Path, captures: dict[str, RawCapture], extra: dict | None = None):
    """Write captures as a fixture. The directory must not exist or must be empty."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"{directory} is not empty")
    directory.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    for name, cap in captures.items():
        np.savez_compressed(
            directory / f"{name}.npz",
            rgb=cap.rgb,
            depth=cap.depth,
            timestamps_ms=np.asarray(cap.timestamps_ms, float),
        )
        meta[name] = cap.meta()
    meta["captured_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for key, value in (extra or {}).items():
        meta[key] = value  # load_fixture skips entries that have no <name>.npz
    (directory / "meta.json").write_text(json.dumps(meta, indent=1))
    return directory


# --------------------------------------------------------------------------- sources


class FixtureSource:
    """Replay a saved fixture. Each snapshot gives the same stored frames."""

    mode = "fixture"

    def __init__(self, directory: str | Path, frames: int | None = None):
        self.directory = Path(directory)
        meta_path = self.directory / "meta.json"
        if not meta_path.is_file():
            raise CameraError(f"Fixture {self.directory}: no meta.json")
        self.meta = json.loads(meta_path.read_text())
        self.captures: dict[str, RawCapture] = {}
        for name, info in self.meta.items():
            path = self.directory / f"{name}.npz"
            if not isinstance(info, dict) or not path.exists():
                continue
            data = np.load(path)
            rgb, depth, stamps = data["rgb"], data["depth"], data["timestamps_ms"]
            if frames:
                rgb, depth, stamps = rgb[-frames:], depth[-frames:], stamps[-frames:]
            self.captures[name] = RawCapture(
                camera=name,
                serial=str(info.get("serial", "")),
                rgb=rgb,
                depth=depth,
                timestamps_ms=np.asarray(stamps, float),
                depth_scale_m=float(info["depth_scale_m"]),
                intrinsics={
                    k: info.get(k)
                    for k in (
                        "width",
                        "height",
                        "fx",
                        "fy",
                        "cx",
                        "cy",
                        "distortion_model",
                        "coeffs",
                    )
                },
                firmware=str(info.get("firmware", "")),
            )
        if not self.captures:
            raise CameraError(f"Fixture {self.directory}: no camera data")
        self.cameras = list(self.captures)

    def open(self) -> dict:
        return {n: c.intrinsics for n, c in self.captures.items()}

    def grab(self, frames: int | None = None) -> dict[str, RawCapture]:
        return dict(self.captures)

    def close(self) -> None:
        pass


class RealSenseCamera:
    """One D415, opened by SDK serial. Colour 640x480 rgb8 + depth 1280x720 z16 at 30 fps."""

    def __init__(
        self, name: str, serial: str, fps: int = 30, warmup: int = 45, timeout_ms: int = 5000
    ):
        self.name, self.serial, self.fps = name, serial, fps
        self.warmup, self.timeout_ms = warmup, timeout_ms
        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001
        self.intrinsics: dict = {}
        self.firmware = ""

    def open(self) -> dict:
        import pyrealsense2 as rs

        present = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
        if self.serial not in present:
            raise CameraNotFoundError(
                f"Camera {self.name} (serial {self.serial}) is not connected. "
                f"Connected RealSense serials: {present or 'none'}."
            )
        pipeline, config = rs.pipeline(), rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, *COLOR_SIZE, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, *DEPTH_SIZE, rs.format.z16, self.fps)
        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            text = str(exc)
            if "busy" in text.lower() or "resource" in text.lower():
                raise CameraBusyError(
                    f"Camera {self.name} (serial {self.serial}) is busy: another process uses "
                    f"it (for example the live-depth viewer /tmp/brickcell-live-depth.py, port "
                    f"8094). Stop that process first. SDK error: {text}"
                ) from None
            raise CameraError(f"Camera {self.name} (serial {self.serial}): {text}") from None
        device = profile.get_device()
        self.depth_scale = float(device.first_depth_sensor().get_depth_scale())
        self.firmware = device.get_info(rs.camera_info.firmware_version)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        k = stream.get_intrinsics()
        self.intrinsics = {
            "width": k.width,
            "height": k.height,
            "fx": k.fx,
            "fy": k.fy,
            "cx": k.ppx,
            "cy": k.ppy,
            "distortion_model": str(k.model),
            "coeffs": list(k.coeffs),
        }
        self.align = rs.align(rs.stream.color)
        self.pipeline = pipeline
        for _ in range(self.warmup):  # auto-exposure settles
            pipeline.wait_for_frames(self.timeout_ms)
        return self.intrinsics

    def grab(self, frames: int) -> RawCapture:
        if self.pipeline is None:
            self.open()
        # Drop the frames that wait in the queue: they can be old.
        for _ in range(2):
            self.pipeline.wait_for_frames(self.timeout_ms)
        rgbs, depths, stamps = [], [], []
        for _ in range(frames):
            fs = self.align.process(self.pipeline.wait_for_frames(self.timeout_ms))
            colour, depth = fs.get_color_frame(), fs.get_depth_frame()
            if not colour or not depth:
                continue
            rgbs.append(np.asanyarray(colour.get_data()).copy())
            depths.append(np.asanyarray(depth.get_data()).copy())
            stamps.append(fs.get_timestamp())
        if not rgbs:
            raise CameraError(f"Camera {self.name}: no complete frame")
        return RawCapture(
            camera=self.name,
            serial=self.serial,
            rgb=np.stack(rgbs),
            depth=np.stack(depths),
            timestamps_ms=np.asarray(stamps, float),
            depth_scale_m=self.depth_scale,
            intrinsics=dict(self.intrinsics),
            firmware=self.firmware,
        )

    def close(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None


class RealSenseSource:
    """Both D415s. The pipelines stay open between snapshots; the two cameras grab in parallel."""

    mode = "live"

    def __init__(self, serials: dict[str, str] | None = None, warmup: int = 45, fps: int = 30):
        serials = serials or CAMERA_SERIALS
        self.devices = {
            n: RealSenseCamera(n, s, fps=fps, warmup=warmup) for n, s in serials.items()
        }
        self.cameras = list(self.devices)
        self.lock = threading.Lock()

    def open(self) -> dict:
        intrinsics = {}
        try:
            for name, device in self.devices.items():
                intrinsics[name] = device.open()
                log.info(
                    "camera %s serial %s firmware %s intrinsics %s",
                    name,
                    device.serial,
                    device.firmware,
                    intrinsics[name],
                )
        except Exception:
            self.close()
            raise
        return intrinsics

    def grab(self, frames: int) -> dict[str, RawCapture]:
        with self.lock, ThreadPoolExecutor(len(self.devices)) as pool:
            futures = {n: pool.submit(d.grab, frames) for n, d in self.devices.items()}
            return {n: f.result() for n, f in futures.items()}

    def close(self) -> None:
        for device in self.devices.values():
            try:
                device.close()
            except Exception as exc:  # noqa: BLE001 - close the other camera too
                log.warning("close %s: %s", device.name, exc)
