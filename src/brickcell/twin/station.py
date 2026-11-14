"""Station service: owns the two D415 cameras, reads the arm, runs perception, serves JSON.

Run on the robot laptop (Intel):
    python -m brickcell.twin.station                       # live cameras, no arm
    python -m brickcell.twin.station --fixture DIR         # replay a saved fixture
    python -m brickcell.twin.station --arm-source file --arm-file joints.json

Imports: numpy, OpenCV, PyYAML, the standard library; pyrealsense2 only in live mode. No SciPy,
no MuJoCo. Perception (brickcell.twin.perception) is imported lazily. If it is missing or
fails, the service still serves a StationState with the error in ``diagnostics``.

HTTP API (bind 127.0.0.1, default port 8095; config: ``station:`` in configs/twin.yaml)
  GET  /health                  service status (JSON)
  GET  /state[?fresh=1]         latest StationState JSON; fresh=1 takes a new snapshot now
  GET  /frame/<camera>.jpg      last colour frame
  GET  /depth/<camera>.png      median depth, colourised (black = no data)
  GET  /overlay/<camera>.jpg    state projected into the colour frame (brick axes, plates)
  POST /live {"enabled": bool, "hz": float <= 2}   take snapshots in the background
  POST /record {"dir": PATH?, "name": NAME?, "fresh": bool?}
                                save the raw frames of the latest snapshot as a fixture
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import yaml

from brickcell.simulation.scene_config import SceneConfig, load_scene_config
from brickcell.twin.realsense import (
    CameraError,
    FixtureSource,
    RawCapture,
    RealSenseSource,
    save_fixture,
)
from brickcell.twin.state import CAMERA_SERIALS, RGBDFrame, StationState, quat_to_matrix

log = logging.getLogger("brickcell.twin.station")

ROOT = Path(__file__).resolve().parents[3]
TWIN_CONFIG = ROOT / "configs" / "twin.yaml"
MAX_LIVE_HZ = 2.0

DEFAULTS: dict = {
    "host": "127.0.0.1",  # keep localhost only; use an SSH tunnel from other machines
    "port": 8095,
    "frames": 10,  # frames per snapshot (per camera); depth = per-pixel median
    "warmup": 45,  # frames to drop after a camera opens (auto exposure)
    "fps": 30,
    "serials": dict(CAMERA_SERIALS),  # camera name -> RealSense serial
    "live_hz": 0.5,  # default rate of POST /live (max 2)
    "live_on_start": False,
    "record_dir": "~/twin-fixtures",  # POST /record writes <record_dir>/fixture-NNN
    "intrinsics_tolerance_px": 0.5,  # warn if SDK intrinsics differ from sim-scene.yaml
    "depth_range_m": [0.15, 1.5],  # colour scale of /depth/<camera>.png
    "jpeg_quality": 90,
    "overlay_axis_m": 0.03,  # length of the brick axes in /overlay
    "arm_source": "none",  # none, file, armd, servo (live reads need user approval)
    "arm_file": None,
    "armd_url": "http://127.0.0.1:8765",
    "servo_port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00",
    "gripper_closed_rad": 0.0,  # MuJoCo gripper angle at 0 % (closed); an assumption
    "gripper_open_rad": 1.7453292,  # MuJoCo gripper angle at 100 % (open); an assumption
}


def load_station_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """The ``station:`` section of configs/twin.yaml on top of DEFAULTS."""
    path = Path(path) if path else TWIN_CONFIG
    section: dict = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text()) or {}
        section = raw.get("station") or {}
    elif path != TWIN_CONFIG:
        raise FileNotFoundError(f"Config file not found: {path}")
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"{path}: unknown station keys {unknown}. Allowed: {sorted(DEFAULTS)}")
    config = {
        **DEFAULTS,
        **section,
        **{k: v for k, v in (overrides or {}).items() if v is not None},
    }
    config["live_hz"] = min(float(config["live_hz"]), MAX_LIVE_HZ)
    return config


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def dumps(data) -> bytes:
    return json.dumps(data, default=_json_default).encode()


# --------------------------------------------------------------------------- perception


class Perception:
    """Lazy wrapper of brickcell.twin.perception. The import is tried again until it works."""

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = config_path
        self.module = None
        self.config = None
        self.tracker = None  # one PerceptionTracker per station process (memory between frames)
        self.status = "not loaded"

    def _load(self) -> None:
        if self.module is not None:
            return
        import importlib

        module = importlib.import_module("brickcell.twin.perception")
        loader = getattr(module, "load_perception_config", None)
        self.config = loader(self.config_path) if loader else None
        self.module = module

    def locate(self, frames: list[RGBDFrame], scene: SceneConfig) -> tuple[StationState, str]:
        """(state, error text). The state is empty if perception is missing or fails."""
        try:
            self._load()
            tracker_class = getattr(self.module, "PerceptionTracker", None)
            if tracker_class is None:
                state = self.module.locate(frames, scene, self.config)
            else:
                if self.tracker is None:
                    self.tracker = tracker_class(scene, self.config)
                state = self.tracker.update(frames, scene)
            if not isinstance(state, StationState):
                raise TypeError(f"locate() returned {type(state).__name__}, not StationState")
            self.status = "ok"
            return state, ""
        except ImportError as exc:
            self.status = "missing"
            return StationState(timestamp=0.0), f"perception module missing: {exc}"
        except Exception as exc:  # noqa: BLE001 - serve the frames even if perception fails
            self.status = "error"
            text = f"{type(exc).__name__}: {exc}"
            log.warning("perception failed: %s\n%s", text, traceback.format_exc())
            return StationState(timestamp=0.0), text


# --------------------------------------------------------------------------- station


def intrinsics_check(scene: SceneConfig, name: str, intrinsics: dict, tolerance: float) -> dict:
    """Compare the SDK intrinsics with configs/sim-scene.yaml. Log a warning if they differ."""
    camera = scene.cameras.get(name)
    if camera is None or camera.intrinsics is None:
        log.warning("camera %s: no intrinsics in the scene config (%s)", name, scene.source)
        return {"ok": False, "reason": "not in the scene config"}
    sdk = [float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy")]
    diff = {k: round(a - b, 3) for k, a, b in zip(("fx", "fy", "cx", "cy"), sdk, camera.pinhole())}
    size_ok = (int(intrinsics["width"]), int(intrinsics["height"])) == (camera.width, camera.height)
    worst = max(abs(v) for v in diff.values())
    ok = worst <= tolerance and size_ok
    if not ok:
        log.warning(
            "camera %s: SDK intrinsics differ from %s by %.2f px (size ok: %s): %s",
            name,
            scene.source,
            worst,
            size_ok,
            diff,
        )
    else:
        log.info("camera %s: SDK intrinsics match %s (max diff %.3f px)", name, scene.source, worst)
    return {"ok": ok, "max_diff_px": round(worst, 3), "diff_px": diff, "size_ok": size_ok}


class Station:
    def __init__(
        self,
        source,
        scene: SceneConfig,
        config: dict,
        arm_reader=None,
        perception: Perception | None = None,
    ):
        self.source = source
        self.scene = scene
        self.config = config
        self.arm_reader = arm_reader
        self.perception = perception or Perception()
        self.snapshot_lock = threading.Lock()  # one snapshot at a time
        self.data_lock = threading.Lock()  # latest results
        self.state: StationState | None = None
        self.captures: dict[str, RawCapture] = {}
        self.frames: dict[str, RGBDFrame] = {}
        self.arm_details: dict = {}
        self.intrinsics: dict[str, dict] = {}
        self.intrinsics_checks: dict[str, dict] = {}
        self.snapshots = 0
        self.last_error = ""
        self.last_snapshot_wall = 0.0
        self.live_enabled = False
        self.live_hz = float(config["live_hz"])
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- life cycle

    def open(self) -> None:
        self.intrinsics = self.source.open()
        tol = float(self.config["intrinsics_tolerance_px"])
        for name, k in self.intrinsics.items():
            self.intrinsics_checks[name] = intrinsics_check(self.scene, name, k, tol)
        self._thread = threading.Thread(target=self._live_loop, name="live", daemon=True)
        self._thread.start()
        if self.config.get("live_on_start"):
            self.set_live(True, self.live_hz)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.source.close()
        if self.arm_reader:
            self.arm_reader.close()

    # -- snapshot

    def snapshot(self) -> StationState:
        with self.snapshot_lock:
            try:
                state = self._snapshot()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise
            self.last_error = ""
            return state

    def _snapshot(self) -> StationState:
        t0 = time.perf_counter()
        captures = self.source.grab(int(self.config["frames"]))
        t1 = time.perf_counter()
        frames = {name: cap.to_frame() for name, cap in captures.items()}
        t2 = time.perf_counter()
        state, perception_error = self.perception.locate(list(frames.values()), self.scene)
        t3 = time.perf_counter()
        arm, arm_details, arm_error = None, {}, ""
        if self.arm_reader is not None:
            try:
                arm, arm_details = self.arm_reader.read()
            except Exception as exc:  # noqa: BLE001 - the arm is optional
                arm_error = f"{type(exc).__name__}: {exc}"
        t4 = time.perf_counter()
        newest = max(f.timestamp for f in frames.values())
        state.timestamp = state.timestamp or newest
        state.arm = arm if arm is not None else state.arm
        state.cameras = {name: self._camera_info(captures[name], frames[name]) for name in frames}
        station_diag = {
            "mode": self.source.mode,
            "snapshot": self.snapshots + 1,
            "frames_per_camera": {n: int(len(c.rgb)) for n, c in captures.items()},
            "timing_s": {
                "grab": round(t1 - t0, 4),
                "median": round(t2 - t1, 4),
                "perception": round(t3 - t2, 4),
                "arm": round(t4 - t3, 4),
                "total": round(t4 - t0, 4),
            },
            "perception": self.perception.status,
            "arm_source": getattr(self.arm_reader, "source", "none"),
        }
        if perception_error:
            station_diag["perception_error"] = perception_error
            state.diagnostics.setdefault("error", perception_error)
        if arm_error:
            station_diag["arm_error"] = arm_error
        if arm_details:
            station_diag["arm"] = arm_details
        state.diagnostics["station"] = station_diag
        with self.data_lock:
            self.state, self.captures, self.frames = state, captures, frames
            self.arm_details = arm_details
            self.snapshots += 1
            self.last_snapshot_wall = time.time()
        log.info(
            "snapshot %d: grab %.2f s, perception %.2f s (%s), total %.2f s",
            self.snapshots,
            t1 - t0,
            t3 - t2,
            self.perception.status,
            t4 - t0,
        )
        return state

    def _camera_info(self, cap: RawCapture, frame: RGBDFrame) -> dict:
        k = cap.intrinsics
        return {
            "serial": cap.serial,
            "timestamp": frame.timestamp,
            "width": int(k["width"]),
            "height": int(k["height"]),
            "fx": float(k["fx"]),
            "fy": float(k["fy"]),
            "cx": float(k["cx"]),
            "cy": float(k["cy"]),
            "depth_scale_m": cap.depth_scale_m,
            "frames": int(len(cap.rgb)),
            "depth_valid_fraction": round(float((frame.depth_m > 0).mean()), 4),
            "intrinsics_check": self.intrinsics_checks.get(cap.camera, {}),
        }

    def latest(self, fresh: bool = False) -> StationState:
        if fresh or self.state is None:
            return self.snapshot()
        with self.data_lock:
            return self.state

    def latest_frame(self, camera: str) -> RGBDFrame:
        if not self.frames:
            self.snapshot()
        with self.data_lock:
            if camera not in self.frames:
                raise KeyError(camera)
            return self.frames[camera]

    # -- live mode

    def set_live(self, enabled: bool, hz: float | None = None) -> dict:
        if hz is not None:
            hz = float(hz)
            if not 0 < hz <= MAX_LIVE_HZ:
                raise ValueError(f"hz must be > 0 and <= {MAX_LIVE_HZ}")
            self.live_hz = hz
        self.live_enabled = bool(enabled)
        self._wake.set()
        return {"enabled": self.live_enabled, "hz": self.live_hz}

    def _live_loop(self) -> None:
        while not self._stop.is_set():
            if not self.live_enabled:
                self._wake.wait()
                self._wake.clear()
                continue
            start = time.monotonic()
            try:
                self.snapshot()
            except Exception as exc:  # noqa: BLE001 - keep the loop; /health shows the error
                log.warning("live snapshot failed: %s", exc)
            delay = max(0.0, 1.0 / self.live_hz - (time.monotonic() - start))
            self._wake.wait(delay)
            self._wake.clear()

    # -- record

    def record(
        self, directory: str | Path | None = None, name: str | None = None, fresh: bool = False
    ) -> dict:
        if fresh or not self.captures:
            self.snapshot()
        with self.data_lock:
            captures, state = dict(self.captures), self.state
        if directory is None:
            base = Path(os.path.expanduser(str(self.config["record_dir"])))
            if name:
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
                    raise ValueError(f"bad fixture name {name!r}")
                directory = base / name
            else:
                directory = next_fixture_dir(base)
        extra = {}
        if state is not None and state.arm is not None:
            extra["arm"] = {**state.arm.__dict__, "details": self.arm_details}
        path = save_fixture(directory, captures, extra)
        log.info("recorded fixture %s", path)
        return {
            "path": str(path),
            "cameras": sorted(captures),
            "frames": {n: int(len(c.rgb)) for n, c in captures.items()},
        }

    # -- health

    def health(self) -> dict:
        age = time.time() - self.last_snapshot_wall if self.last_snapshot_wall else None
        return {
            "ok": not self.last_error,
            "mode": self.source.mode,
            "cameras": list(self.source.cameras),
            "intrinsics_check": self.intrinsics_checks,
            "snapshots": self.snapshots,
            "last_snapshot_age_s": round(age, 3) if age is not None else None,
            "last_error": self.last_error,
            "live": {"enabled": self.live_enabled, "hz": self.live_hz},
            "perception": self.perception.status,
            "arm_source": getattr(self.arm_reader, "source", "none"),
            "scene_config": self.scene.source,
            "pid": os.getpid(),
        }


def next_fixture_dir(base: Path) -> Path:
    numbers = [
        int(m.group(1))
        for p in base.glob("fixture-*")
        if (m := re.fullmatch(r"fixture-(\d+)", p.name))
    ]
    return base / f"fixture-{max(numbers, default=0) + 1:03d}"


# --------------------------------------------------------------------------- images


def colourise_depth(depth_m: np.ndarray, near: float, far: float) -> np.ndarray:
    """BGR image: TURBO colour scale from near (red) to far (blue); black = no data."""
    valid = depth_m > 0
    scaled = np.clip((far - depth_m) / max(far - near, 1e-6), 0, 1)
    image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def project_points(points, camera_pos, rotation_cv, fx, fy, cx, cy):
    """World points -> (pixels (N, 2), in_front (N,)). OpenCV camera frame, no distortion."""
    local = (np.atleast_2d(np.asarray(points, float)) - np.asarray(camera_pos, float)) @ rotation_cv
    z = local[:, 2]
    safe = np.where(z > 1e-6, z, 1.0)
    pixels = np.column_stack((fx * local[:, 0] / safe + cx, fy * local[:, 1] / safe + cy))
    return pixels, z > 1e-6


def draw_overlay(
    frame: RGBDFrame, state: StationState | None, scene: SceneConfig, axis_m: float = 0.03
) -> np.ndarray:
    """BGR image: the state projected into the colour frame with the scene camera pose."""
    image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    camera = scene.cameras.get(frame.camera)
    lines = [f"{frame.camera} {time.strftime('%H:%M:%S', time.gmtime(frame.timestamp))} UTC"]
    if camera is None or camera.attach is not None:
        lines.append("no world pose for this camera in the scene config")
    else:
        pose = (np.asarray(camera.pos, float), camera.rotation_cv())
        k = (frame.fx, frame.fy, frame.cx, frame.cy)

        def px(points):
            return project_points(points, *pose, *k)

        def segment(a, b, colour, width=2):
            (pa, pb), front = px([a, b])
            if front.all():
                cv2.line(image, _pt(pa), _pt(pb), colour, width, cv2.LINE_AA)

        origin = np.zeros(3)
        for axis, colour in zip(np.eye(3) * 0.05, ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            segment(origin, axis, colour, 2)
        if state is not None:
            for plate in state.plates:
                (p,), front = px([plate.pos])
                if front[0]:
                    cv2.circle(image, _pt(p), 8, (255, 255, 255), 2, cv2.LINE_AA)
                    _label(image, f"plate {plate.name} {plate.confidence:.2f}", _pt(p, 10, -10))
            for brick in state.bricks:
                pos = np.asarray(brick.pos, float)
                rot = quat_to_matrix(brick.quat)
                for i, colour in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                    segment(pos, pos + rot[:, i] * axis_m, colour, 2)
                (p,), front = px([pos])
                if front[0]:
                    bgr = tuple(int(255 * c) for c in reversed(brick.color[:3]))
                    cv2.circle(image, _pt(p), 5, bgr, -1, cv2.LINE_AA)
                    cv2.circle(image, _pt(p), 6, (255, 255, 255), 1, cv2.LINE_AA)
                    _label(image, f"{brick.object_id} {brick.confidence:.2f}", _pt(p, 8, 14))
    if state is not None:
        lines.append(f"bricks {len(state.bricks)}  plates {len(state.plates)}")
        err = state.diagnostics.get("station", {}).get("perception_error")
        if err:
            lines.append(f"perception: {err[:80]}")
        if state.arm is not None:
            lines.append("arm " + " ".join(f"{v:+.2f}" for v in state.arm.joints.values()))
    for i, text in enumerate(lines):
        _label(image, text, (8, 20 + 18 * i))
    return image


def _pt(p, dx=0, dy=0) -> tuple[int, int]:
    x, y = float(p[0]) + dx, float(p[1]) + dy
    x = min(max(x, -1e5), 1e5) if math.isfinite(x) else 0
    y = min(max(y, -1e5), 1e5) if math.isfinite(y) else 0
    return int(round(x)), int(round(y))


def _label(image, text, org) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- HTTP


INDEX = """<!doctype html><title>Station twin</title>
<h3>Station twin</h3><p><a href=/health>/health</a> · <a href=/state>/state</a> ·
<a href="/state?fresh=1">/state?fresh=1</a></p>{rows}"""


def make_handler(station: Station):
    class Handler(BaseHTTPRequestHandler):
        server_version = "brickcell-station/1"

        def log_message(self, fmt, *args):
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200) -> None:
            self._send(status, dumps(data), "application/json")

        def _error(self, status: int, text: str) -> None:
            self._json({"error": text}, status)

        def _image(self, image: np.ndarray, ext: str) -> None:
            params = [cv2.IMWRITE_JPEG_QUALITY, int(station.config["jpeg_quality"])]
            ok, data = cv2.imencode(ext, image, params if ext == ".jpg" else [])
            if not ok:
                return self._error(500, "image encode failed")
            self._send(200, data.tobytes(), "image/jpeg" if ext == ".jpg" else "image/png")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("the body must be a JSON object")
            return data

        def do_GET(self):  # noqa: N802 - http.server API
            url = urlparse(self.path)
            query = parse_qs(url.query)
            path = url.path.rstrip("/") or "/"
            try:
                if path == "/":
                    rows = "".join(
                        f"<p>{c}: <img src=/frame/{c}.jpg width=320> "
                        f"<img src=/depth/{c}.png width=320> "
                        f"<img src=/overlay/{c}.jpg width=320></p>"
                        for c in station.source.cameras
                    )
                    return self._send(200, INDEX.format(rows=rows).encode(), "text/html")
                if path == "/health":
                    return self._json(station.health())
                if path == "/state":
                    fresh = query.get("fresh", ["0"])[0].lower() in ("1", "true", "yes")
                    return self._json(station.latest(fresh).to_dict())
                match = re.fullmatch(r"/(frame|depth|overlay)/([A-Za-z0-9_-]+)\.(jpg|png)", path)
                if match:
                    kind, camera, ext = match.groups()
                    frame = station.latest_frame(camera)
                    if kind == "frame":
                        image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                    elif kind == "depth":
                        near, far = station.config["depth_range_m"]
                        image = colourise_depth(frame.depth_m, float(near), float(far))
                    else:
                        image = draw_overlay(
                            frame,
                            station.state,
                            station.scene,
                            float(station.config["overlay_axis_m"]),
                        )
                    return self._image(image, "." + ext)
                return self._error(404, f"no route {path}")
            except KeyError as exc:
                return self._error(404, f"unknown camera {exc}")
            except CameraError as exc:
                return self._error(503, str(exc))
            except Exception as exc:  # noqa: BLE001 - report, keep the server
                log.exception("GET %s", self.path)
                return self._error(500, f"{type(exc).__name__}: {exc}")

        def do_POST(self):  # noqa: N802 - http.server API
            path = urlparse(self.path).path.rstrip("/")
            try:
                body = self._body()
                if path == "/live":
                    if "enabled" not in body:
                        raise ValueError('give {"enabled": true|false, "hz": <= 2}')
                    return self._json(station.set_live(bool(body["enabled"]), body.get("hz")))
                if path == "/record":
                    return self._json(
                        station.record(
                            body.get("dir"), body.get("name"), bool(body.get("fresh", False))
                        )
                    )
                return self._error(404, f"no route {path}")
            except (ValueError, FileExistsError) as exc:
                return self._error(400, str(exc))
            except CameraError as exc:
                return self._error(503, str(exc))
            except Exception as exc:  # noqa: BLE001
                log.exception("POST %s", self.path)
                return self._error(500, f"{type(exc).__name__}: {exc}")

    return Handler


def serve(station: Station, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(station))
    server.daemon_threads = True
    return server


# --------------------------------------------------------------------------- CLI


def build_station(args) -> Station:
    config = load_station_config(
        args.config,
        {
            "port": args.port,
            "frames": args.frames,
            "record_dir": args.record_dir,
            "arm_source": "none" if args.no_arm else args.arm_source,
            "arm_file": args.arm_file,
        },
    )
    scene = load_scene_config(args.scene_config)
    if args.fixture:
        source = FixtureSource(args.fixture, frames=None)
    else:
        serials = {n: str(s) for n, s in config["serials"].items()}
        source = RealSenseSource(serials, warmup=int(config["warmup"]), fps=int(config["fps"]))
    arm_reader = None
    if config["arm_source"] not in (None, "", "none"):
        from brickcell.twin.arm import make_arm_reader

        arm_reader = make_arm_reader(config["arm_source"], config["arm_file"], config)
    return Station(source, scene, config, arm_reader, Perception(args.config))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixture", type=Path, help="replay this fixture (no hardware)")
    parser.add_argument("--port", type=int, help="HTTP port (default 8095)")
    parser.add_argument("--config", type=Path, help="twin config (default configs/twin.yaml)")
    parser.add_argument("--scene-config", type=Path, help="scene config (default sim-scene.yaml)")
    parser.add_argument("--no-arm", action="store_true", help="same as --arm-source none")
    parser.add_argument("--arm-source", choices=("none", "file", "armd", "servo"))
    parser.add_argument("--arm-file", type=Path, help="JSON file for --arm-source file")
    parser.add_argument("--frames", type=int, help="frames per snapshot (default 10)")
    parser.add_argument("--record-dir", type=Path, help="folder for POST /record fixtures")
    parser.add_argument(
        "--no-initial-snapshot", action="store_true", help="do not take a snapshot at start"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        station = build_station(args)
        station.open()
    except CameraError as exc:
        log.error("%s", exc)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2
    try:
        server = serve(station, str(station.config["host"]), int(station.config["port"]))
    except OSError as exc:
        log.error("cannot listen on %s:%s: %s", station.config["host"], station.config["port"], exc)
        station.close()
        return 2
    host, port = server.server_address[:2]
    log.info(
        "station service on http://%s:%s (mode %s, arm %s)",
        host,
        port,
        station.source.mode,
        station.config["arm_source"],
    )

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    if not args.no_initial_snapshot:
        try:
            station.snapshot()
        except Exception as exc:  # noqa: BLE001 - /health shows it
            log.error("initial snapshot failed: %s", exc)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        station.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
