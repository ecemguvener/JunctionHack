"""Station twin dashboard: live Top camera (real or MuJoCo sim), fused D415 point cloud, overlays.

The dashboard runs on the Intel laptop next to its sources. All sources are loopback HTTP:

  real Top RGB    live depth service  GET /rgb/top                 JPEG
  point clouds    live depth service  GET /points/top, /points/plate  camera-frame XYZRGB JSON
  sim Top render  MuJoCo sim server   GET /camera/overhead.png     PNG
  camera poses    MuJoCo sim server   GET /scene                   world pose + intrinsics
  sim state       MuJoCo sim server   GET /health

The focus block and the anomaly boxes are STUBS. Another service can POST them to
/api/focus and /api/anomalies. Until it does, the UI shows "not connected".

The dashboard only reads. It sends no robot or simulator commands. It never opens a camera.
Dependencies: Python 3.12, numpy, OpenCV (headless).
"""

from __future__ import annotations

import argparse
import collections
import functools
import gzip
import json
import math
import threading
import time
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

STATIC = Path(__file__).resolve().parent / "static"
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css"}
# World box (robot base frame, metres) that the point cloud keeps: table, plates, arm.
WORKSPACE = ((-0.15, 0.65), (-0.55, 0.45), (-0.02, 0.35))
MAX_CLOUD_POINTS = 40000
MAX_BODY = 64 * 1024
MAX_ITEMS = 50
# Table levelling: each camera's table plane must lie within this band and tilt of z = 0.
TABLE_BAND_M = 0.08
MAX_TILT_DEG = 8.0
LEVEL_PERIOD_S = 10.0
SENSOR_IDS = {"top": 0, "plate": 1}


def fetch(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (loopback URLs from CLI)
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return response.read()


def fetch_json(url: str, timeout: float) -> dict:
    return json.loads(fetch(url, timeout))


# --------------------------------------------------------------------------- image feeds


@dataclass(frozen=True)
class Frame:
    seq: int
    data: bytes
    time: float  # dashboard receive time (unix seconds)


def check_jpeg(data: bytes) -> bytes:
    if not (data.startswith(b"\xff\xd8") and len(data) > 100):
        raise ValueError("Not a JPEG image")
    return data


def png_to_jpeg(data: bytes, quality: int = 80) -> bytes:
    """Decode any image (PNG or JPEG) and encode it as a JPEG of the given quality."""
    import cv2

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Not a decodable image")
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return jpeg.tobytes()


class Feed:
    """Polls one image URL and keeps the newest frame as JPEG."""

    def __init__(self, name: str, url: str, fps: float, convert=check_jpeg, timeout: float = 3):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.name, self.url, self.period, self.convert, self.timeout = (
            name,
            url,
            1 / fps,
            convert,
            timeout,
        )
        self.cond = threading.Condition()
        self.frame: Frame | None = None
        self.error: str | None = None
        self.times: collections.deque[float] = collections.deque(maxlen=30)

    def poll_once(self) -> None:
        try:
            data = self.convert(fetch(self.url, self.timeout))
        except Exception as exc:  # noqa: BLE001 - report every source failure in the status
            with self.cond:
                self.error = f"{type(exc).__name__}: {exc}"
            return
        now = time.time()
        with self.cond:
            self.frame = Frame((self.frame.seq + 1) if self.frame else 1, data, now)
            self.error = None
            self.times.append(now)
            self.cond.notify_all()

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            start = time.monotonic()
            self.poll_once()
            stop.wait(max(0.0, self.period - (time.monotonic() - start)))

    def wait(self, after: int, timeout: float) -> Frame | None:
        with self.cond:
            self.cond.wait_for(lambda: self.frame is not None and self.frame.seq > after, timeout)
            return self.frame if self.frame is not None and self.frame.seq > after else None

    def status(self, now: float, stale_s: float) -> dict:
        with self.cond:
            frame, times, error = self.frame, list(self.times), self.error
        age = None if frame is None else now - frame.time
        fps = (
            (len(times) - 1) / (times[-1] - times[0])
            if len(times) > 2 and times[-1] > times[0]
            else None
        )
        return dict(
            ok=age is not None and age <= stale_s,
            age_s=None if age is None else round(age, 3),
            fps=None if fps is None else round(fps, 1),
            seq=0 if frame is None else frame.seq,
            error=error,
        )


# --------------------------------------------------------------------------- scene (camera poses)


def parse_camera(entry: dict) -> dict:
    rotation = np.asarray(entry["world_rotation_cv"], float)
    position = np.asarray(entry["world_pos"], float)
    numbers = [entry[k] for k in ("fx", "fy", "cx", "cy", "width", "height")]
    if rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError(f"Camera {entry.get('name')}: bad pose shape")
    if not (
        np.isfinite(rotation).all()
        and np.isfinite(position).all()
        and all(map(math.isfinite, numbers))
    ):
        raise ValueError(f"Camera {entry.get('name')}: nonfinite pose or intrinsics")
    if abs(np.linalg.det(rotation) - 1) > 1e-3:
        raise ValueError(f"Camera {entry.get('name')}: rotation is not proper")
    return dict(
        pos=position.tolist(),
        rotation_cv=rotation.tolist(),
        fx=float(entry["fx"]),
        fy=float(entry["fy"]),
        cx=float(entry["cx"]),
        cy=float(entry["cy"]),
        width=int(entry["width"]),
        height=int(entry["height"]),
    )


class Scene:
    """Camera poses and intrinsics from the sim server (one source of truth), plus sim health."""

    def __init__(self, sim_url: str, timeout: float = 3):
        self.sim_url, self.timeout = sim_url.rstrip("/"), timeout
        self.lock = threading.Lock()
        self.cameras: dict[str, dict] = {}
        self.loaded_at: float | None = None
        self.health: dict | None = None
        self.health_at: float | None = None
        self.error: str | None = None

    def poll_once(self) -> None:
        try:
            scene = fetch_json(self.sim_url + "/scene", self.timeout)
            cameras = {
                c["name"]: parse_camera(c) for c in scene["cameras"] if c.get("enabled", True)
            }
            health = fetch_json(self.sim_url + "/health", self.timeout)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
            return
        now = time.time()
        with self.lock:
            self.cameras, self.loaded_at, self.error = cameras, now, None
            self.health = {
                k: health.get(k) for k in ("mode", "busy", "stopped", "run_id", "objects")
            }
            self.health_at = now

    def run(self, stop: threading.Event, period: float = 5) -> None:
        while not stop.is_set():
            self.poll_once()
            stop.wait(period)

    def camera(self, name: str) -> dict | None:
        with self.lock:
            return self.cameras.get(name)

    def status(self, now: float) -> dict:
        with self.lock:
            age = None if self.health_at is None else now - self.health_at
            return dict(
                ok=bool(self.cameras) and age is not None and age <= 15,
                age_s=None if age is None else round(age, 3),
                cameras=sorted(self.cameras),
                sim=self.health,
                error=self.error,
            )


# --------------------------------------------------------------------------- point cloud


def to_world(points: np.ndarray, camera: dict) -> np.ndarray:
    """Camera-frame (OpenCV) XYZ rows -> world XYZ rows. world = R @ p + pos."""
    rotation = np.asarray(camera["rotation_cv"], float)
    return points @ rotation.T + np.asarray(camera["pos"], float)


def crop(xyz: np.ndarray, box=WORKSPACE) -> np.ndarray:
    keep = np.ones(len(xyz), bool)
    for axis, (low, high) in enumerate(box):
        keep &= (xyz[:, axis] >= low) & (xyz[:, axis] <= high)
    return keep


CLOUD_FORMAT = (
    "8-byte little-endian records: int16 x, y, z (mm, robot base frame); "
    "uint16 colour = sensor << 15 | RGB555 (sensor 0 top, 1 plate)"
)


def pack(xyz: np.ndarray, rgb: np.ndarray, sensor: np.ndarray) -> bytes:
    """See CLOUD_FORMAT."""
    if not (len(xyz) == len(rgb) == len(sensor)):
        raise ValueError("Point arrays differ in length")
    if len(sensor) and not np.isin(sensor, (0, 1)).all():
        raise ValueError("sensor ids must be 0 or 1")
    out = np.zeros((len(xyz), 4), "<u2")
    out[:, :3] = np.clip(np.round(xyz * 1000), -32768, 32767).astype("<i2").view("<u2")
    c = np.clip(rgb, 0, 255).astype(np.uint32) >> 3
    out[:, 3] = (np.asarray(sensor, np.uint32) << 15) | (c[:, 0] << 10) | (c[:, 1] << 5) | c[:, 2]
    return out.tobytes()


def unpack(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (xyz mm int16 (N,3), rgb555 (N,), sensor (N,))."""
    rec = np.frombuffer(data, "<u2").reshape(-1, 4)
    return rec[:, :3].view("<i2"), rec[:, 3] & 0x7FFF, rec[:, 3] >> 15


@dataclass(frozen=True)
class Level:
    """Rigid correction that puts one camera's table plane on z = 0: w' = R (w - c) + (cx, cy, 0)."""

    rotation: np.ndarray
    centre: np.ndarray
    tilt_deg: float
    offset_mm: float
    inliers: int

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        out = (xyz - self.centre) @ self.rotation.T
        out[:, :2] += self.centre[:2]
        return out

    def as_dict(self) -> dict:
        return dict(
            tilt_deg=round(self.tilt_deg, 2),
            offset_mm=round(self.offset_mm, 1),
            inliers=self.inliers,
        )


def rotation_to_z(normal: np.ndarray) -> np.ndarray:
    """Proper rotation that turns the unit vector `normal` (z > 0) into +z."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    s, c = np.linalg.norm(axis), float(normal @ z)
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def fit_table(xyz: np.ndarray, rng, band=TABLE_BAND_M, tol=0.005, iters=300) -> Level | None:
    """RANSAC plane on points near z = 0. None if no table plane is reliable enough."""
    near = xyz[np.abs(xyz[:, 2]) < band]
    if len(near) < 300:
        return None
    sample = near[rng.choice(len(near), min(len(near), 4000), replace=False)]
    best, best_count = None, 0
    for _ in range(iters):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n[2] < 0:
            n = -n
        if n[2] < math.cos(math.radians(MAX_TILT_DEG)):
            continue
        count = int((np.abs((sample - a) @ n) < tol).sum())
        if count > best_count:
            best, best_count = (n, a), count
    if best is None or best_count < 0.3 * len(sample):
        return None
    n, a = best
    inliers = sample[np.abs((sample - a) @ n) < tol]
    centre = inliers.mean(0)  # least-squares refinement on the inliers
    n = np.linalg.svd(inliers - centre, full_matrices=False)[2][-1]
    n = n if n[2] > 0 else -n
    tilt = math.degrees(math.acos(min(1.0, float(n[2]))))
    if tilt > MAX_TILT_DEG or abs(centre[2]) > band:
        return None
    return Level(rotation_to_z(n), centre, tilt, float(centre[2] * 1000), len(inliers))


@dataclass(frozen=True)
class CloudSource:
    sensor: str  # live depth name: "top" or "plate"
    scene_camera: str  # sim scene name: "overhead" or "side"
    url: str


@dataclass(frozen=True)
class Cloud:
    seq: int
    data: bytes
    gz: bytes  # the same data, gzip level 1
    count: int
    time: float
    cameras: dict


class CloudWorker:
    """Fetches each D415 cloud, moves it into the robot base frame, crops and packs it."""

    def __init__(
        self, sources: list[CloudSource], scene: Scene, hz: float, stale_s: float, timeout=3
    ):
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.sources, self.scene, self.period, self.stale_s, self.timeout = (
            sources,
            scene,
            1 / hz,
            stale_s,
            timeout,
        )
        self.cond = threading.Condition()
        self.cloud: Cloud | None = None
        self.per_camera: dict[str, dict] = {
            s.sensor: dict(ok=False, error="not fetched") for s in sources
        }
        self.rng = np.random.default_rng(0)
        self.levels: dict[str, tuple[float, Level | None]] = {}

    def level(self, sensor: str, xyz: np.ndarray) -> Level | None:
        """Table plane fit per camera, refreshed every LEVEL_PERIOD_S. Keeps the last good fit."""
        at, previous = self.levels.get(sensor, (-math.inf, None))
        if time.monotonic() - at < LEVEL_PERIOD_S:
            return previous
        fit = fit_table(xyz, self.rng)
        self.levels[sensor] = (time.monotonic(), fit or previous)
        return fit or previous

    def fetch_camera(self, source: CloudSource) -> tuple[np.ndarray, np.ndarray, dict]:
        camera = self.scene.camera(source.scene_camera)
        if camera is None:
            raise RuntimeError(f"No pose for sim camera {source.scene_camera!r} yet")
        payload = fetch_json(source.url, self.timeout)
        age = float(payload.get("age_s", math.inf))
        if not math.isfinite(age) or age > self.stale_s:
            raise RuntimeError(f"Stale depth frame ({age:.1f} s)")
        points = np.asarray(payload["points"], float).reshape(-1, 6)
        points = points[np.isfinite(points).all(axis=1)]
        xyz = to_world(points[:, :3], camera)
        level = self.level(source.sensor, xyz)
        if level is not None:
            xyz = level.apply(xyz)
        keep = crop(xyz)
        info = dict(
            ok=True,
            table=level.as_dict() if level else None,
            points=int(keep.sum()),
            raw_points=len(points),
            age_s=round(age, 3),
            serial=payload.get("serial"),
            error=None,
        )
        return xyz[keep], points[keep, 3:6], info

    def poll_once(self) -> None:
        parts, per_camera = [], {}
        for source in self.sources:
            try:
                xyz, rgb, info = self.fetch_camera(source)
            except Exception as exc:  # noqa: BLE001
                per_camera[source.sensor] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
                continue
            per_camera[source.sensor] = info
            parts.append((xyz, rgb, np.full(len(xyz), SENSOR_IDS.get(source.sensor, 1))))
        with self.cond:
            self.per_camera = per_camera
        if not parts:
            return
        xyz, rgb, sensor = (np.concatenate(a) for a in zip(*parts))
        if len(xyz) > MAX_CLOUD_POINTS:
            pick = self.rng.choice(len(xyz), MAX_CLOUD_POINTS, replace=False)
            xyz, rgb, sensor = xyz[pick], rgb[pick], sensor[pick]
        data = pack(xyz, rgb, sensor)
        gz = gzip.compress(data, compresslevel=1, mtime=0)
        with self.cond:
            seq = (self.cloud.seq + 1) if self.cloud else 1
            self.cloud = Cloud(seq, data, gz, len(xyz), time.time(), per_camera)
            self.cond.notify_all()

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            start = time.monotonic()
            self.poll_once()
            stop.wait(max(0.0, self.period - (time.monotonic() - start)))

    def wait(self, after: int, timeout: float) -> Cloud | None:
        with self.cond:
            self.cond.wait_for(lambda: self.cloud is not None and self.cloud.seq > after, timeout)
            return self.cloud if self.cloud is not None and self.cloud.seq > after else None

    def status(self, now: float) -> dict:
        with self.cond:
            cloud, per_camera = self.cloud, dict(self.per_camera)
        age = None if cloud is None else now - cloud.time
        return dict(
            ok=age is not None
            and age <= self.stale_s
            and any(c.get("ok") for c in per_camera.values()),
            age_s=None if age is None else round(age, 3),
            points=0 if cloud is None else cloud.count,
            seq=0 if cloud is None else cloud.seq,
            cameras=per_camera,
        )


# --------------------------------------------------------------------------- overlay stubs


def _box(value, width: int, height: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("box must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (float(v) for v in value)
    if not all(map(math.isfinite, (x0, y0, x1, y1))):
        raise ValueError("box must be finite")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"box must be inside the {width}x{height} Top image")
    return [x0, y0, x1, y1]


def _text(value, limit: int = 60) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"text fields must be strings of at most {limit} characters")
    return value


class Overlay:
    """Boxes pushed by another service (stub interface). Empty and 'not connected' until then."""

    def __init__(self, kind: str, stale_s: float, width: int = 640, height: int = 480):
        self.kind, self.stale_s, self.width, self.height = kind, stale_s, width, height
        self.lock = threading.Lock()
        self.payload: dict | None = None
        self.at: float | None = None
        self.capture_times: collections.OrderedDict[tuple, float] = collections.OrderedDict()

    def update(self, body: dict, now: float | None = None) -> dict:
        if not isinstance(body, dict):
            raise ValueError("JSON object required")
        source = _text(body.get("source"))
        if not source:
            raise ValueError("source is required")
        if self.kind == "anomalies":
            items = body.get("items")
            if not isinstance(items, list) or len(items) > MAX_ITEMS:
                raise ValueError(f"items must be a list of at most {MAX_ITEMS}")
            clean = []
            for item in items:
                score = item.get("score")
                if score is not None and not (
                    isinstance(score, (int, float)) and math.isfinite(score)
                ):
                    raise ValueError("score must be a finite number")
                clean.append(
                    dict(
                        box=_box(item.get("box"), self.width, self.height),
                        score=score,
                        label=_text(item.get("label")),
                        object_id=_text(item.get("object_id")),
                    )
                )
            payload = dict(source=source, items=clean)
        else:
            object_id = _text(body.get("object_id"))
            payload = dict(
                source=source,
                object_id=object_id,
                label=_text(body.get("label")),
                box=_box(body.get("box"), self.width, self.height) if object_id else None,
            )
        received = time.time() if now is None else now
        evidence = body.get("evidence")
        if evidence is not None:
            if not isinstance(evidence, dict):
                raise ValueError("evidence must be an object")
            captured = evidence.get("captured_at")
            if type(captured) not in (int, float) or not math.isfinite(captured):
                raise ValueError("captured_at must be finite acquisition time")
            if captured > received + 0.5:
                raise ValueError("captured_at is in the future; check source clock")
            evidence = {
                "captured_at": captured,
                **{
                    k: _text(evidence.get(k), 160)
                    for k in (
                        "frame_id",
                        "camera_id",
                        "mode",
                        "box_space",
                        "method",
                        "model_version",
                        "threshold_version",
                    )
                },
            }
            if not evidence["frame_id"] or not evidence["camera_id"]:
                raise ValueError("evidence requires source frame_id and camera_id")
            if evidence["mode"] not in ("real", "sim"):
                raise ValueError("evidence mode must be real or sim")
            if evidence["box_space"] not in ("top_projected_object", "source_image_pixels"):
                raise ValueError("Unsupported evidence box_space")
        payload["evidence"] = evidence
        with self.lock:
            if evidence:
                key = (source, evidence["mode"], evidence["camera_id"], evidence["frame_id"])
                if key in self.capture_times and self.capture_times[key] != evidence["captured_at"]:
                    raise ValueError("Cannot change acquisition time for a previously seen frame")
                self.capture_times[key] = evidence["captured_at"]
                self.capture_times.move_to_end(key)
                while len(self.capture_times) > 1024:
                    self.capture_times.popitem(last=False)
            self.payload, self.at = payload, received
        return payload

    def status(self, now: float) -> dict:
        with self.lock:
            payload, at = self.payload, self.at
        if payload is None:
            return dict(
                connected=False,
                drawable=False,
                stub=True,
                evidence_state="missing",
                reason="No publisher result received",
            )
        age = now - at
        if age > self.stale_s:
            return dict(
                connected=False,
                drawable=False,
                stub=True,
                evidence_state="publisher_stale",
                reason=f"Last update {age:.0f} s ago",
                source=payload["source"],
            )
        evidence = payload.get("evidence")
        result = dict(connected=True, stub=False, age_s=round(age, 3), drawable=False, **payload)
        if evidence is None:
            return dict(
                **result,
                evidence_state="unverified",
                reason="Publisher online; capture identity/time missing. Boxes hidden.",
            )
        capture_age = now - evidence["captured_at"]
        result["capture_age_s"] = round(capture_age, 3)
        if not -0.5 <= capture_age <= self.stale_s:
            return dict(
                **result,
                evidence_state="capture_stale",
                reason="Inspection capture expired; heartbeat cannot refresh it.",
            )
        if evidence["box_space"] != "top_projected_object":
            return dict(
                **result,
                evidence_state="source_frame_required",
                reason="Source-image regions need the exact scored frame; live overlay hidden.",
            )
        result["drawable"] = True
        return dict(
            **result,
            evidence_state="projected_object",
            reason="Recent projected object estimate, not defect-pixel localization.",
        )


# --------------------------------------------------------------------------- HTTP


class Dashboard:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.real = Feed(
            "real",
            args.real_url,
            args.real_fps,
            convert=functools.partial(png_to_jpeg, quality=args.jpeg_quality),
        )
        self.sim = Feed(
            "sim",
            f"{args.sim_url.rstrip('/')}/camera/{args.sim_camera}.png",
            args.sim_fps,
            convert=functools.partial(png_to_jpeg, quality=args.jpeg_quality),
        )
        self.scene = Scene(args.sim_url)
        self.cloud = CloudWorker(args.cloud, self.scene, args.cloud_hz, args.stale_after)
        self.anomalies = Overlay("anomalies", args.overlay_stale)
        self.focus = Overlay("focus", args.overlay_stale)
        self.started = time.time()

    def start(self) -> list[threading.Thread]:
        threads = [threading.Thread(target=self.scene.run, args=(self.stop,), daemon=True)]
        threads += [
            threading.Thread(target=w.run, args=(self.stop,), daemon=True)
            for w in (self.real, self.sim, self.cloud)
        ]
        for thread in threads:
            thread.start()
        return threads

    def config(self) -> dict | None:
        top = self.scene.camera(self.args.sim_camera)
        if top is None:
            return None
        plate = self.scene.camera(
            next((s.scene_camera for s in self.args.cloud if s.sensor == "plate"), "side")
        )
        return dict(
            image=dict(width=top["width"], height=top["height"]),
            cameras={"top": top, **({"plate": plate} if plate else {})},
            pivot=list(self.args.pivot),
            workspace=WORKSPACE,
            sim_camera=self.args.sim_camera,
            cloud_format=CLOUD_FORMAT,
        )

    def status(self) -> dict:
        now = time.time()
        stale = self.args.stale_after
        sources = dict(
            real=self.real.status(now, stale),
            sim=self.sim.status(now, max(stale, 3 / self.args.sim_fps)),
            scene=self.scene.status(now),
            cloud=self.cloud.status(now),
        )
        return dict(
            time=now,
            uptime_s=round(now - self.started, 1),
            ok=all(s["ok"] for s in sources.values()),
            sources=sources,
            focus=self.focus.status(now),
            anomalies=self.anomalies.status(now),
        )


def make_handler(app: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def send(self, status: int, kind: str, body: bytes, headers: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, data) -> None:
            self.send(status, "application/json", json.dumps(data, allow_nan=False).encode())

        def do_GET(self):
            url = urlparse(self.path)
            route, query = url.path, parse_qs(url.query)
            if route == "/":
                return self.static("index.html")
            if route.startswith("/static/"):
                return self.static(route[len("/static/") :])
            if route == "/api/status":
                return self.send_json(200, app.status())
            if route == "/api/config":
                config = app.config()
                return (
                    self.send_json(200, config)
                    if config
                    else self.send_json(503, {"error": "Waiting for sim scene"})
                )
            if route == "/api/cloud":
                try:
                    after = int(query.get("after", ["0"])[0] or 0)
                except ValueError:
                    return self.send_json(400, {"error": "after must be an integer"})
                return self.cloud(after)
            if route in ("/frame/real.jpg", "/frame/sim.jpg"):
                # ?after=SEQ waits up to 2 s for a newer frame (204 if none): the client pulls
                # the next frame only after the last one arrived, so the rate fits the link.
                feed = app.real if "real" in route else app.sim
                try:
                    after = int(query.get("after", ["0"])[0] or 0)
                except ValueError:
                    return self.send_json(400, {"error": "after must be an integer"})
                frame = feed.wait(after, 2.0 if after else 0)
                if frame is None and after:
                    return self.no_content()
                if frame is None:
                    return self.send_json(
                        503, {"error": f"No {feed.name} frame yet", "detail": feed.error}
                    )
                headers = {"X-Frame-Seq": str(frame.seq), "X-Frame-Time": f"{frame.time:.3f}"}
                return self.send(200, "image/jpeg", frame.data, headers)
            if route in ("/stream/real.mjpg", "/stream/sim.mjpg"):
                return self.mjpeg(app.real if "real" in route else app.sim)
            self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            route = urlparse(self.path).path
            target = {"/api/anomalies": app.anomalies, "/api/focus": app.focus}.get(route)
            if target is None:
                return self.send_json(404, {"error": "Not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self.send_json(400, {"error": f"Body must be 1..{MAX_BODY} bytes of JSON"})
            try:
                payload = target.update(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError, AttributeError) as exc:
                return self.send_json(400, {"error": str(exc)})
            self.send_json(200, {"ok": True, "stored": payload})

        def static(self, name: str) -> None:
            path = (STATIC / name).resolve()
            if path.parent != STATIC or path.suffix not in STATIC_TYPES or not path.is_file():
                return self.send_json(404, {"error": "Not found"})
            self.send(200, STATIC_TYPES[path.suffix], path.read_bytes())

        def no_content(self) -> None:
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def cloud(self, after: int) -> None:
            cloud = app.cloud.wait(after, timeout=2.0)
            if cloud is None:
                return self.no_content()
            headers = {
                "X-Cloud-Seq": str(cloud.seq),
                "X-Cloud-Count": str(cloud.count),
                "X-Cloud-Time": f"{cloud.time:.3f}",
                "Vary": "Accept-Encoding",
            }
            body = cloud.data
            if "gzip" in (self.headers.get("Accept-Encoding") or ""):
                body, headers["Content-Encoding"] = cloud.gz, "gzip"
            self.send(200, "application/octet-stream", body, headers)

        def mjpeg(self, feed: Feed) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            seq = 0
            try:
                while not app.stop.is_set():
                    frame = feed.wait(seq, timeout=5)
                    if frame is None:
                        continue
                    seq = frame.seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame.data)}\r\n\r\n".encode())
                    self.wfile.write(frame.data + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

    return Handler


def parse_cloud(spec: str) -> CloudSource:
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "use SENSOR=SCENE_CAMERA=URL, e.g. top=overhead=http://..."
        )
    return CloudSource(*parts)


def parse_pivot(text: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in text.split(","))
    if len(values) != 3 or not all(map(math.isfinite, values)):
        raise argparse.ArgumentTypeError("use X,Y,Z in metres")
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8096)
    p.add_argument(
        "--real-url", default="http://127.0.0.1:8094/rgb/top", help="real Top camera JPEG"
    )
    p.add_argument("--sim-url", default="http://127.0.0.1:18081", help="MuJoCo sim server API")
    p.add_argument(
        "--sim-camera", default="overhead", help="sim camera that matches the real Top camera"
    )
    p.add_argument(
        "--cloud",
        type=parse_cloud,
        action="append",
        metavar="SENSOR=SCENE_CAMERA=URL",
        help="point cloud source; repeat. Default: top=overhead and plate=side on :8094",
    )
    p.add_argument("--real-fps", type=float, default=12)
    p.add_argument(
        "--jpeg-quality", type=int, default=75, help="JPEG quality of the re-encoded frames"
    )
    p.add_argument("--sim-fps", type=float, default=4)
    p.add_argument("--cloud-hz", type=float, default=2)
    p.add_argument(
        "--stale-after", type=float, default=3, help="seconds before a source shows as stale"
    )
    p.add_argument(
        "--overlay-stale", type=float, default=5, help="seconds before pushed overlays expire"
    )
    p.add_argument(
        "--pivot",
        type=parse_pivot,
        default=(0.25, -0.13, 0.0),
        help="orbit centre X,Y,Z (m, robot base frame): the plates",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.cloud:
        args.cloud = [
            CloudSource("top", "overhead", "http://127.0.0.1:8094/points/top"),
            CloudSource("plate", "side", "http://127.0.0.1:8094/points/plate"),
        ]
    app = Dashboard(args)
    app.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    print(f"Dashboard on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
