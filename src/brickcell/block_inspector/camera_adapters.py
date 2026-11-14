"""Pluggable Camera Adapters (Twin, Teammate Multicam/CamUI, and Live Browser HUD).

Supports seamless drop-in switching between:
1. Local MuJoCo 4.0.1 Digital Twin (`TwinCameraSource`)
2. Teammate's physical camera server (`MulticamHttpSource` @ 127.0.0.1:8092 + `CamUiReporter` @ 127.0.0.1:8093)
3. Live local HTTP 3-camera + 8-corner telemetry HUD (`LiveInspectionHudServer` @ 127.0.0.1:8095)
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol

import numpy as np
from PIL import Image


class CameraWorldSource(Protocol):
    """Unified multi-camera frame provider for perception and corner inspection."""

    def get_rgb(self, camera_name: str) -> np.ndarray: ...
    def get_rgbd(self, camera_name: str) -> tuple[np.ndarray, np.ndarray | None]: ...


class TwinCameraSource:
    """Reads rendered 640x480 RGB and metric depth directly from SO101InspectionTwin."""

    def __init__(self, twin) -> None:
        self.twin = twin

    def get_rgb(self, camera_name: str) -> np.ndarray:
        rgb, _ = self.twin.render_rgbd(camera_name)
        return rgb

    def get_rgbd(self, camera_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        return self.twin.render_rgbd(camera_name)


# Alias mapping matching teammate's multicam_server.py (origin/jack-camera-ui)
MULTICAM_ENDPOINT_MAP = {
    "realsense_1": "overhead",
    "overhead": "overhead",
    "realsense_2": "side",
    "side": "side",
    "wrist": "wrist",
}


class MulticamHttpSource:
    """Adapter for teammate's multicam_server.py (http://127.0.0.1:8092).

    Maps canonical camera names (`realsense_1`, `realsense_2`, `wrist`) to
    `/image/{overhead,side,wrist}` and optional `/depth/{overhead,side}` raw mm/float endpoints.
    Never connects unless explicitly instantiated and called.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8092", timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def get_rgb(self, camera_name: str) -> np.ndarray:
        slot = MULTICAM_ENDPOINT_MAP.get(camera_name, camera_name)
        url = f"{self.base_url}/image/{slot}"
        with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
            raw = resp.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def get_rgbd(self, camera_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        rgb = self.get_rgb(camera_name)
        slot = MULTICAM_ENDPOINT_MAP.get(camera_name, camera_name)
        if slot == "wrist":
            return rgb, None
        depth_url = f"{self.base_url}/depth/{slot}"
        try:
            with urllib.request.urlopen(depth_url, timeout=self.timeout_s) as resp:
                buf = resp.read()
            arr = np.load(io.BytesIO(buf), allow_pickle=False)
            if arr.dtype == np.uint16:
                depth_m = arr.astype(np.float32) * 0.001
            else:
                depth_m = arr.astype(np.float32)
            return rgb, depth_m
        except (urllib.error.URLError, OSError, ValueError):
            return rgb, None


class CamUiReporter:
    """Posts inspection status to teammate's camui.py (`POST /api/detection` on :8093)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8093", timeout_s: float = 1.5) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def report(self, malformed: bool | None, detail: str) -> bool:
        payload = json.dumps({"malformed": malformed, "detail": detail}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/detection",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError):
            return False


class LiveInspectionHudServer:
    """Threaded HTTP dashboard serving live 3-camera JPEGs and 8-corner status JSON."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8095) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._jpegs: dict[str, bytes] = {}
        self._status: dict = {"status": "idle", "inspected_count": 0, "corners": []}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        actual_port = self._httpd.server_port if self._httpd is not None else self.port
        return f"http://{self.host}:{actual_port}"

    def update_from_twin(self, twin, step_counter: int = 0) -> None:
        """Refresh cached camera frames and 8-corner status (throttled to ~6 Hz during motion)."""
        if step_counter % 5 != 0:
            return
        jpegs: dict[str, bytes] = {}
        for cam_name in ("realsense_1", "realsense_2", "wrist"):
            rgb, _ = twin.render_rgbd(cam_name)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=82)
            jpegs[cam_name] = buf.getvalue()
        summary = twin.ledger.summary_dict()
        summary.update(
            {
                "synthetic_simulation": True,
                "banner": twin.status_banner,
                "active_part": twin.block_spec.name,
                "active_target_corner": twin.active_target_corner,
                "q_deg": np.round(np.degrees(twin.current_arm_q()), 1).tolist(),
                "gripper_pct": round(float(twin.gripper_pct), 3),
                "hand_empty": bool(twin.hand_empty),
                "remaining_source_blocks": int(twin.remaining_source_blocks),
                "source_plate_cleared": bool(twin.source_plate_cleared),
                "plate_scan_active": bool(twin.plate_scan_active),
                "plate_scan_completed": bool(twin.plate_scan_completed),
                "plate_scan_valid_frames": int(twin.plate_scan_valid_frames),
                "score_type": "raw_patchcore_distance_not_probability",
                "max_plate_anomaly_distance": twin.max_plate_anomaly_distance,
                "anomaly_threshold": twin.anomaly_threshold,
                "anomaly_flagged": bool(twin.anomaly_flagged),
                "anomaly_messages": list(twin.anomaly_messages),
                "sorted_pile": twin.sorted_pile,
                "sorted_counts": dict(twin.sorted_counts),
                "batch_sorted_records": list(twin.batch_sorted_records),
            }
        )
        with self._lock:
            self._jpegs.update(jpegs)
            self._status = summary

    def start(self) -> str:
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # noqa: ANN001
                return

            def do_GET(self):  # noqa: N802
                if self.path in ("/", "/index.html"):
                    body = _HUD_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.startswith("/api/status"):
                    with parent._lock:
                        body = json.dumps(parent._status).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.startswith("/frame/"):
                    cam = self.path.split("/frame/", 1)[1].split("?", 1)[0]
                    cam = {"overhead": "realsense_1", "side": "realsense_2"}.get(cam, cam)
                    with parent._lock:
                        jpg = parent._jpegs.get(cam)
                    if jpg is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(jpg)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

        self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


_HUD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>SO-101 Live 3-Camera + 8-Corner HUD</title>
<style>
body { font-family: system-ui, sans-serif; background: #0b1120; color: #e2e8f0; margin: 18px; }
.banner { background: #1e293b; padding: 12px 16px; border-radius: 10px; border-left: 4px solid #38bdf8; margin-bottom: 14px; }
.cams { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.cam-box { background: #1e293b; padding: 10px; border-radius: 10px; text-align: center; }
.cam-box img { width: 100%; border-radius: 6px; background: #020617; }
.corners { display: grid; grid-template-columns: repeat(8, 1fr); gap: 8px; margin-top: 14px; }
.c-pill { padding: 10px; border-radius: 8px; background: #1e293b; text-align: center; border: 2px solid #334155; font-size: 13px; }
.c-pill.ok { border-color: #22c55e; background: #052e16; }
.c-pill.active { border-color: #38bdf8; background: #0c4a6e; }
</style></head><body>
<div class="banner"><b id="banner">Connecting to MuJoCo 4.0.1 Digital Twin...</b>
<span id="meta" style="float:right;color:#94a3b8"></span></div>
<div class="cams">
  <div class="cam-box"><div><b>RealSense D415 #1 (Overhead)</b></div><img id="c1" src="/frame/realsense_1"/></div>
  <div class="cam-box"><div><b>RealSense D415 #2 (Side)</b></div><img id="c2" src="/frame/realsense_2"/></div>
  <div class="cam-box"><div><b>Sonix UVC Wrist (25° Canted)</b></div><img id="cw" src="/frame/wrist"/></div>
</div>
<div class="corners" id="corners"></div>
<script>
async function tick() {
  const t = Date.now();
  document.getElementById('c1').src = '/frame/realsense_1?t=' + t;
  document.getElementById('c2').src = '/frame/realsense_2?t=' + t;
  document.getElementById('cw').src = '/frame/wrist?t=' + t;
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    document.getElementById('banner').textContent = s.banner || 'Running';
    document.getElementById('meta').textContent = `Verified: ${s.inspected_count || 0}/8`;
    const grid = document.getElementById('corners');
    if (s.corners) {
      grid.innerHTML = s.corners.map(c => {
        const cls = c.inspected ? 'ok' : (s.active_target_corner === c.corner_index ? 'active' : '');
        return `<div class="c-pill ${cls}"><b>c_${c.corner_index}</b><br/>${c.inspected ? 'VERIFIED' : c.last_reason}<br/>dwell ${c.dwell_count}</div>`;
      }).join('');
    }
  } catch (e) {}
}
setInterval(tick, 220);
tick();
</script></body></html>
"""
