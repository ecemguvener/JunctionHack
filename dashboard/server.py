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


def png_to_jpeg(data: bytes, quality: int = 80, resize_wh: tuple[int, int] | None = None) -> bytes:
    """Decode any image (PNG or JPEG) and encode it as a JPEG of the given quality."""
    import cv2

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Not a decodable image")
    if resize_wh is not None and (image.shape[1], image.shape[0]) != resize_wh:
        image = cv2.resize(image, resize_wh, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return jpeg.tobytes()


def analyze_camera_features(cam_slot: str, image_bgr: np.ndarray) -> dict:
    """Extract 2D jaw corridor (green/cyan/yellow) and DUPLO brick minAreaRect + grasp axes."""
    import cv2

    h_w, w_w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 1. Detect purple SO-101 gripper jaw corridor (primarily visible in wrist view)
    corridor: dict = {"ok": False}
    jaws_mask = np.zeros((h_w, w_w), dtype=np.uint8)
    if cam_slot == "wrist":
        jaw_raw = ((H >= 113) & (H <= 146) & (S >= 70) & (V >= 28)).astype(np.uint8) * 255
        jaw_clean = cv2.morphologyEx(jaw_raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        jaw_clean = cv2.morphologyEx(jaw_clean, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(jaw_clean)
        left_cands: list[tuple[int, int]] = []
        right_cands: list[tuple[int, int]] = []
        for idx in range(1, num_labels):
            x0, y0, bw, bh, area = stats[idx]
            cx = float(centroids[idx][0])
            if int(area) < 350 or (int(y0) + int(bh)) < (h_w - 28):
                continue
            if cx < 0.54 * w_w:
                left_cands.append((int(area), idx))
            if cx >= 0.46 * w_w:
                right_cands.append((int(area), idx))
        if left_cands and right_cands:
            left_idx = max(left_cands)[1]
            right_idx = max(right_cands)[1]
            if left_idx != right_idx:
                left_jaw = (labels == left_idx).astype(np.uint8) * 255
                right_jaw = (labels == right_idx).astype(np.uint8) * 255
                jaws_mask = left_jaw | right_jaw
                l_box_y = int(stats[left_idx][1])
                r_box_y = int(stats[right_idx][1])
                v_start = max(int(0.42 * h_w), min(l_box_y, r_box_y))
                v_end = int(0.95 * h_w)
                edge_pairs: list[tuple[int, int, int]] = []
                for v in range(v_start, v_end):
                    xs_l = np.where(left_jaw[v] > 0)[0]
                    xs_r = np.where(right_jaw[v] > 0)[0]
                    if len(xs_l) > 0 and len(xs_r) > 0:
                        ul = int(xs_l.max())
                        ur = int(xs_r.min())
                        if ur > ul + 15:
                            edge_pairs.append((ul, ur, v))
                if len(edge_pairs) >= 12:
                    pairs = np.array(edge_pairs, dtype=np.float64)
                    top_n = min(len(pairs), 55)
                    u_left_pad = float(np.median(pairs[:top_n, 0]))
                    u_right_pad = float(np.median(pairs[:top_n, 1]))
                    u_pinch = 0.5 * (u_left_pad + u_right_pad)
                    corridor = {
                        "ok": True,
                        "u_left_pad": round(u_left_pad, 1),
                        "u_right_pad": round(u_right_pad, 1),
                        "u_pinch": round(u_pinch, 1),
                        "gap_px": round(u_right_pad - u_left_pad, 1),
                        "v_top": int(pairs[0, 2]),
                        "v_bottom": int(pairs[-1, 2]),
                        "left_edge_uv": [[int(p[0]), int(p[2])] for p in edge_pairs[::2]],
                        "right_edge_uv": [[int(p[1]), int(p[2])] for p in edge_pairs[::2]],
                    }

    # 2. Segment DUPLO bricks inside the aquamarine source plate (minAreaRect + short grasp axis)
    jaws_dilated = (
        cv2.dilate(jaws_mask, np.ones((9, 9), np.uint8), iterations=1)
        if np.any(jaws_mask)
        else jaws_mask
    )
    is_aqua = (H >= 85) & (H <= 114) & (S >= 68) & (V >= 40) & (jaws_dilated == 0)
    aqua_u8 = (is_aqua.astype(np.uint8)) * 255
    aqua_u8 = cv2.morphologyEx(aqua_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(aqua_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bricks_out: list[dict] = []
    min_plate_area = 1200 if cam_slot == "wrist" else 450
    if cnts:
        plate_cnt = max(cnts, key=cv2.contourArea)
        if float(cv2.contourArea(plate_cnt)) >= min_plate_area:
            hull = cv2.convexHull(plate_cnt)
            hull_mask = np.zeros((h_w, w_w), dtype=np.uint8)
            cv2.drawContours(hull_mask, [hull], -1, 255, -1)
            erode_k = 11 if cam_slot == "wrist" else 7
            plate_interior = cv2.erode(
                hull_mask, np.ones((erode_k, erode_k), np.uint8), iterations=1
            )
            search_roi = cv2.subtract(plate_interior, jaws_dilated)
            aqua_pixels = lab[is_aqua]
            if len(aqua_pixels) >= 100:
                L0, A0, B0 = np.median(aqua_pixels, axis=0)
            else:
                L0, A0, B0 = 150.0, 105.0, 95.0
            delta_ab = np.hypot(lab[:, :, 1] - A0, lab[:, :, 2] - B0)
            delta_e = np.sqrt((0.45 * (lab[:, :, 0] - L0)) ** 2 + delta_ab**2)
            is_glare = (S < 32) & (V > 212) & (delta_ab < 10.0)
            fg = (
                (search_roi > 0)
                & (~is_aqua)
                & (~is_glare)
                & ((delta_ab >= 11.5) | (delta_e >= 18.0))
            )
            fg_u8 = (fg.astype(np.uint8)) * 255
            fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            fcnts, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fg_filled = np.zeros_like(fg_u8)
            min_brick_area = 260.0 if cam_slot == "wrist" else 90.0
            for c in fcnts:
                if cv2.contourArea(c) >= min_brick_area:
                    cv2.drawContours(fg_filled, [c], -1, 255, -1)
            dist_thresh = 8.0 if cam_slot == "wrist" else 4.5
            dist_map = cv2.distanceTransform(fg_filled, cv2.DIST_L2, 5)
            fg_split = fg_filled.copy()
            if float(dist_map.max()) > dist_thresh * 1.5:
                sure_fg = (dist_map >= dist_thresh).astype(np.uint8) * 255
                sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                n_seeds, markers = cv2.connectedComponents(sure_fg)
                if n_seeds > 1:
                    markers = markers + 1
                    unknown = cv2.subtract(fg_filled, sure_fg)
                    markers[unknown == 255] = 0
                    ws = cv2.watershed(cv2.cvtColor(fg_filled, cv2.COLOR_GRAY2BGR), markers)
                    fg_split[ws == -1] = 0
            bcnts, _ = cv2.findContours(fg_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in bcnts:
                area = float(cv2.contourArea(c))
                if area < min_brick_area:
                    continue
                rect = cv2.minAreaRect(c)
                rw, rh = float(rect[1][0]), float(rect[1][1])
                if min(rw, rh) < (12.0 if cam_slot == "wrist" else 7.0):
                    continue
                box_pts = cv2.boxPoints(rect)
                e01 = box_pts[1] - box_pts[0]
                e12 = box_pts[2] - box_pts[1]
                l01 = float(np.linalg.norm(e01))
                l12 = float(np.linalg.norm(e12))
                short_vec = e12 if l01 >= l12 else e01
                short_norm = max(float(np.linalg.norm(short_vec)), 1e-6)
                ux = float(short_vec[0] / short_norm)
                uy = float(short_vec[1] / short_norm)
                bu, bv = float(rect[0][0]), float(rect[0][1])
                half_span = 0.5 * min(l01, l12) + (12.0 if cam_slot == "wrist" else 7.0)
                bricks_out.append(
                    {
                        "center_uv": [round(bu, 1), round(bv, 1)],
                        "box_uv": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in box_pts],
                        "axis_uv": [
                            [round(bu - half_span * ux, 1), round(bv - half_span * uy, 1)],
                            [round(bu + half_span * ux, 1), round(bv + half_span * uy, 1)],
                        ],
                        "area_px": round(area, 1),
                    }
                )
    bricks_out.sort(key=lambda b: b["center_uv"][1], reverse=True)
    return {
        "width": w_w,
        "height": h_w,
        "corridor": corridor,
        "bricks": bricks_out,
    }


class Feed:
    """Polls one image URL and keeps the newest frame as JPEG (plus optional 2D feature analysis)."""

    def __init__(
        self,
        name: str,
        url: str,
        fps: float,
        convert=check_jpeg,
        timeout: float = 3,
        cam_slot: str | None = None,
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.name, self.url, self.period, self.convert, self.timeout = (
            name,
            url,
            1 / fps,
            convert,
            timeout,
        )
        self.cam_slot = cam_slot
        self.cond = threading.Condition()
        self.frame: Frame | None = None
        self.features: dict | None = None
        self.error: str | None = None
        self.times: collections.deque[float] = collections.deque(maxlen=30)

    def poll_once(self) -> None:
        try:
            data = self.convert(fetch(self.url, self.timeout))
            features = None
            if self.cam_slot is not None:
                import cv2

                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    features = analyze_camera_features(self.cam_slot, img)
        except Exception as exc:  # noqa: BLE001 - report every source failure in the status
            with self.cond:
                self.error = f"{type(exc).__name__}: {exc}"
            return
        now = time.time()
        with self.cond:
            self.frame = Frame((self.frame.seq + 1) if self.frame else 1, data, now)
            if features is not None:
                self.features = features
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

    def feature_status(self, now: float, stale_s: float) -> dict:
        with self.cond:
            frame, features = self.frame, self.features
        age = None if frame is None else now - frame.time
        if frame is None or features is None or age is None or age > stale_s:
            return dict(ok=False, corridor={"ok": False}, bricks=[])
        return dict(ok=True, age_s=round(age, 3), **features)


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
        self.real2sim: dict | None = None
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
            if isinstance(scene.get("real2sim"), dict):
                self.real2sim = scene["real2sim"]
            self.health = {
                k: health.get(k) for k in ("mode", "busy", "stopped", "run_id", "objects")
            }
            self.health_at = now

    def run(self, stop: threading.Event, period: float = 1.5) -> None:
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
        self.real2sim_state: dict | None = None
        self.real2sim_at: float | None = None

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
        now_t = time.time()
        with self.scene.lock:
            r2s = self.scene.real2sim
        with self.cond:
            seq = (self.cloud.seq + 1) if self.cloud else 1
            self.cloud = Cloud(seq, data, gz, len(xyz), now_t, per_camera)
            if isinstance(r2s, dict):
                self.real2sim_state = r2s
                self.real2sim_at = now_t
            self.cond.notify_all()

    def real2sim_status(self, now: float) -> dict:
        with self.cond:
            state, at = self.real2sim_state, self.real2sim_at
        if state is None:
            with self.scene.lock:
                state, at = self.scene.real2sim, self.scene.health_at
        if state is None or at is None:
            return dict(ok=False, bricks=[], plates={})
        age = now - at
        return dict(
            ok=age <= max(self.stale_s * 3, 10.0),
            age_s=round(age, 3),
            bricks=state.get("bricks", []),
            plates=state.get("plates", {}),
            camera_top=state.get("camera_top"),
            camera_plate=state.get("camera_plate"),
            camera_wrist=state.get("camera_wrist"),
        )

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
        real_base = (
            args.real_url[: -len("/rgb/top")]
            if args.real_url.endswith("/rgb/top")
            else args.real_url.rstrip("/")
        )
        real_plate_url = getattr(args, "real_plate_url", None) or f"{real_base}/rgb/plate"
        real_wrist_url = getattr(args, "real_wrist_url", None) or (
            "http://127.0.0.1:8091/frame.jpg"
            if args.real_url == "http://127.0.0.1:8094/rgb/top"
            else f"{real_base}/rgb/wrist"
        )
        sim_base = args.sim_url.rstrip("/")
        self.real = Feed(
            "real",
            args.real_url,
            args.real_fps,
            convert=functools.partial(png_to_jpeg, quality=args.jpeg_quality),
            cam_slot="top",
        )
        self.real_plate = Feed(
            "real_plate",
            real_plate_url,
            max(2.0, args.real_fps * 0.5),
            convert=functools.partial(png_to_jpeg, quality=args.jpeg_quality),
            cam_slot="plate",
        )
        self.real_wrist = Feed(
            "real_wrist",
            real_wrist_url,
            max(4.0, args.real_fps * 0.75),
            convert=functools.partial(png_to_jpeg, quality=args.jpeg_quality, resize_wh=(640, 360)),
            cam_slot="wrist",
        )
        self.sim = Feed(
            "sim",
            f"{sim_base}/camera/{args.sim_camera}.png",
            args.sim_fps,
            convert=functools.partial(png_to_jpeg, quality=args.jpeg_quality),
            cam_slot="top",
        )
        self.sim_plate = Feed(
            "sim_plate",
            f"{sim_base}/camera/side.png",
            args.sim_fps,
            convert=functools.partial(png_to_jpeg, quality=args.jpeg_quality),
            cam_slot="plate",
        )
        self.sim_wrist = Feed(
            "sim_wrist",
            f"{sim_base}/camera/wrist.png",
            args.sim_fps,
            convert=functools.partial(png_to_jpeg, quality=args.jpeg_quality, resize_wh=(640, 360)),
            cam_slot="wrist",
        )
        self.feeds = {
            ("real", "top"): self.real,
            ("real", "plate"): self.real_plate,
            ("real", "wrist"): self.real_wrist,
            ("sim", "top"): self.sim,
            ("sim", "plate"): self.sim_plate,
            ("sim", "wrist"): self.sim_wrist,
        }
        self.scene = Scene(args.sim_url)
        self.cloud = CloudWorker(args.cloud, self.scene, args.cloud_hz, args.stale_after)
        self.anomalies = Overlay("anomalies", args.overlay_stale)
        self.focus = Overlay("focus", args.overlay_stale)
        self.started = time.time()

    def start(self) -> list[threading.Thread]:
        threads = [threading.Thread(target=self.scene.run, args=(self.stop,), daemon=True)]
        threads += [
            threading.Thread(target=w.run, args=(self.stop,), daemon=True)
            for w in (
                self.real,
                self.real_plate,
                self.real_wrist,
                self.sim,
                self.sim_plate,
                self.sim_wrist,
                self.cloud,
            )
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
        wrist = self.scene.camera("wrist")
        return dict(
            image=dict(width=top["width"], height=top["height"]),
            cameras={
                "top": top,
                **({"plate": plate} if plate else {}),
                **({"wrist": wrist} if wrist else {}),
            },
            pivot=list(self.args.pivot),
            workspace=WORKSPACE,
            sim_camera=self.args.sim_camera,
            cloud_format=CLOUD_FORMAT,
        )

    def status(self) -> dict:
        now = time.time()
        stale = self.args.stale_after
        sim_stale = max(stale, 3 / self.args.sim_fps)
        sources = dict(
            real=self.real.status(now, stale),
            sim=self.sim.status(now, sim_stale),
            scene=self.scene.status(now),
            cloud=self.cloud.status(now),
        )
        camera_feeds = dict(
            real=dict(
                top=sources["real"],
                plate=self.real_plate.status(now, stale),
                wrist=self.real_wrist.status(now, stale),
            ),
            sim=dict(
                top=sources["sim"],
                plate=self.sim_plate.status(now, sim_stale),
                wrist=self.sim_wrist.status(now, sim_stale),
            ),
        )
        camera_features = dict(
            real=dict(
                top=self.real.feature_status(now, stale),
                plate=self.real_plate.feature_status(now, stale),
                wrist=self.real_wrist.feature_status(now, stale),
            ),
            sim=dict(
                top=self.sim.feature_status(now, sim_stale),
                plate=self.sim_plate.feature_status(now, sim_stale),
                wrist=self.sim_wrist.feature_status(now, sim_stale),
            ),
        )
        return dict(
            time=now,
            uptime_s=round(now - self.started, 1),
            ok=all(s["ok"] for s in sources.values()),
            sources=sources,
            camera_feeds=camera_feeds,
            camera_features=camera_features,
            focus=self.focus.status(now),
            anomalies=self.anomalies.status(now),
            real2sim=self.cloud.real2sim_status(now),
        )


def _get_studio_engine(app: Dashboard):
    engine = getattr(app, "_studio_engine", None)
    if engine is None:
        import sys

        for candidate in (
            Path(__file__).resolve().parent.parent / "src",
            Path("/home/intel/brickcell-deploy/src"),
        ):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        from brickcell.studio.engine import SO101StudioEngine

        engine = SO101StudioEngine()
        app._studio_engine = engine
    return engine


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
            if route.startswith(("/api/so101/", "/api/studio/", "/frame/studio/")):
                try:
                    from brickcell.studio.server import handle_studio_get

                    res = handle_studio_get(_get_studio_engine(app), route, url.query)
                except Exception:  # noqa: BLE001
                    if route == "/api/so101/ui-extension.js":
                        return self.send(200, "text/javascript", b"")
                    res = None
                if res is not None:
                    code, ctype, data, extra_headers = res
                    return self.send(code, ctype, data, extra_headers)
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
            frame_routes = {
                "/frame/real.jpg": app.real,
                "/frame/sim.jpg": app.sim,
                "/frame/real/top.jpg": app.real,
                "/frame/real/plate.jpg": app.real_plate,
                "/frame/real/wrist.jpg": app.real_wrist,
                "/frame/sim/top.jpg": app.sim,
                "/frame/sim/plate.jpg": app.sim_plate,
                "/frame/sim/wrist.jpg": app.sim_wrist,
            }
            if route in frame_routes:
                # ?after=SEQ waits up to 2 s for a newer frame (204 if none): the client pulls
                # the next frame only after the last one arrived, so the rate fits the link.
                feed = frame_routes[route]
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
            stream_routes = {
                "/stream/real.mjpg": app.real,
                "/stream/sim.mjpg": app.sim,
                "/stream/real/top.mjpg": app.real,
                "/stream/real/plate.mjpg": app.real_plate,
                "/stream/real/wrist.mjpg": app.real_wrist,
                "/stream/sim/top.mjpg": app.sim,
                "/stream/sim/plate.mjpg": app.sim_plate,
                "/stream/sim/wrist.mjpg": app.sim_wrist,
            }
            if route in stream_routes:
                return self.mjpeg(stream_routes[route])
            self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            route = urlparse(self.path).path
            if route.startswith(("/api/so101/", "/api/studio/")):
                from brickcell.studio.server import handle_studio_post

                length = int(self.headers.get("Content-Length") or 0)
                if length < 0 or length > MAX_BODY:
                    return self.send_json(
                        400, {"error": f"Body must be 0..{MAX_BODY} bytes of JSON"}
                    )
                try:
                    raw_body = json.loads(self.rfile.read(length) if length > 0 else b"{}")
                except (ValueError, TypeError) as exc:
                    return self.send_json(400, {"error": str(exc)})
                res = handle_studio_post(_get_studio_engine(app), route, raw_body)
                if res is not None:
                    code, payload = res
                    return self.send_json(code, payload)
                return self.send_json(404, {"error": "Not found"})
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
