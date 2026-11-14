"""Three-camera inspection bench. No robot control; loopback only.

Run with PYTHONPATH=src. OpenCV/numpy suffice for preview and dataset capture.
Optional real OpenVINO models require the core environment and --models JSON.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import threading
import time
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from brickcell.inspection.multiview import classify_burst, fuse_views

CAMERAS = {
    "overhead": ("D415 overhead · 318123021715", "318123021715"),
    "side": ("D415 side · 344223025431", "344223025431"),
    "wrist": ("Sonix wrist · upright", "http://127.0.0.1:8091/frame.jpg"),
}


def cropped(image, crop):
    if not crop:
        return image
    x0, y0, x1, y1 = crop
    h, w = image.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and x1 - x0 >= 16 and y1 - y0 >= 16):
        raise ValueError("Crop must be within the camera image and at least 16x16")
    return image[y0:y1, x0:x1]


class Camera:
    def __init__(self, name, source):
        self.name, self.source = name, source
        self.lock = threading.Lock()
        self.frames = collections.deque(maxlen=15)
        self.error = "Waiting for camera"
        self.crop = None
        self.cap = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while not self.stopped.is_set():
            try:
                if self.source.startswith("http"):
                    with urlopen(self.source, timeout=3) as response:
                        image = cv2.imdecode(
                            np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR
                        )
                else:
                    if self.cap is None:
                        paths = list(Path("/dev/v4l/by-id").glob(f"*{self.source}-video-index0"))
                        if not paths:
                            raise RuntimeError("Camera stable path not found")
                        self.cap = cv2.VideoCapture(str(paths[0]), cv2.CAP_V4L2)
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ok, image = self.cap.read()
                    if not ok:
                        raise RuntimeError("Camera returned no frame")
                if image is None or not image.size:
                    raise RuntimeError("Empty camera image")
                with self.lock:
                    self.frames.append((time.time(), image))
                    self.error = ""
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.stopped.wait(1)
            self.stopped.wait(0.12)
        if self.cap is not None:
            self.cap.release()

    def snapshot(self):
        with self.lock:
            return list(self.frames), self.error, self.crop

    def status(self):
        frames, error, crop = self.snapshot()
        scores, thumbs = [], []
        for _, image in frames:
            roi = cropped(image, crop)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1)
            scores.append(
                1000
                * float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())
                / max(float(gray.var()), 1)
            )
            thumbs.append(cv2.resize(gray, (64, 64)))
        median = statistics.median(scores) if scores else None
        mad = statistics.median(abs(s - median) for s in scores) if scores else None
        age = time.time() - frames[-1][0] if frames else None
        motion = float(np.mean(np.abs(thumbs[-1] - thumbs[-2]))) if len(thumbs) > 1 else None
        return {
            "name": self.name,
            "title": CAMERAS[self.name][0],
            "error": error,
            "age_s": age,
            "frames": len(frames),
            "crop": crop,
            "resolution": list(frames[-1][1].shape[1::-1]) if frames else None,
            "focus_median": median,
            "focus_jitter_percent": 100 * mad / max(median, 1e-6) if median is not None else None,
            "motion": motion,
            "ready": not error and age is not None and age < 2 and len(frames) >= 10,
        }


PAGE = """<!doctype html><meta charset="utf-8"><title>Three-camera inspection</title>
<style>body{background:#11151b;color:#eee;font:16px system-ui;margin:24px}h1{font-size:26px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.card{background:#202731;padding:14px;border-radius:12px}
img{width:100%;min-height:180px;object-fit:contain}input,select,button{font:inherit;padding:8px;margin:4px;background:#303c4b;color:white;border:1px solid #61738a;border-radius:5px}
input.crop{width:90%}p{color:#bdc8d6}#result{padding:18px;background:#202731;margin:16px 0;white-space:pre-wrap}
@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>Three cameras → one brick inspection</h1>
<p>Focus is smoothed over 15 frames; jitter is shown separately. Focus is not a defect score.
Camera views are sampled in software, not hardware-synchronized. Hold the same brick still in all required views.</p>
<div class="grid" id="cameras"></div>
<p>Crop coordinates are X0,Y0,X1,Y1 in the original image. Select the brick face, excluding gripper/background.
Use the identical crop for collection and each camera's trained model.</p>
<input id="object" placeholder="Physical brick ID" value="brick-001">
<input id="session" placeholder="Capture session" value="session-a">
<select id="label"><option value="good">Human label: good</option><option value="defective">Human label: defective</option></select>
<select id="scope"><option value="all">Capture all three views</option><option value="side">Collect side only</option><option value="overhead">Collect overhead only</option><option value="wrist">Collect wrist only</option></select>
<label><input type="checkbox" id="confirmed">Same brick is visible and stationary in the selected crops</label><br>
<button onclick="capture()">Capture all three views / inspect</button>
<div id="result">No inspection yet. Real-image models and view calibration are required for a defect decision.</div>
<script>
const names=['overhead','side','wrist'];
for(const name of names){let card=document.createElement('section');card.className='card';
card.innerHTML=`<h2 id="title-${name}">${name}</h2><img id="img-${name}"><p id="status-${name}">Connecting…</p><input class="crop" id="crop-${name}" placeholder="Crop e.g. 200,120,440,360"><button onclick="setCrop('${name}')">Apply crop</button>`;document.querySelector('#cameras').append(card)}
async function setCrop(name){try{let raw=document.getElementById('crop-'+name).value.trim();let crop=raw?raw.split(',').map(Number):null;
let r=await fetch('/crop',{method:'POST',body:JSON.stringify({camera:name,crop})});let j=await r.json();if(!r.ok)throw Error(j.error)}catch(e){alert(e.message)}}
async function update(){try{let s=await(await fetch('/status')).json();for(let c of s.cameras){document.getElementById('title-'+c.name).textContent=c.title;
document.getElementById('status-'+c.name).textContent=c.error||`${c.resolution} · median focus ${c.focus_median?.toFixed(1)} · jitter ±${c.focus_jitter_percent?.toFixed(1)}% · age ${c.age_s?.toFixed(1)}s`;
document.getElementById('img-'+c.name).src='/image/'+c.name+'?t='+Date.now()}}
catch(e){}setTimeout(update,750)}update();
async function capture(){let r=document.getElementById('result');r.textContent='Capturing burst…';try{let scope=document.getElementById('scope').value;let res=await fetch('/capture',{method:'POST',body:JSON.stringify({object_id:document.getElementById('object').value,session:document.getElementById('session').value,label:document.getElementById('label').value,cameras:scope==='all'?names:[scope],confirmed:document.getElementById('confirmed').checked})});let j=await res.json();r.textContent=JSON.stringify(j,null,2)}catch(e){r.textContent=e.message}}
</script>"""


class Bench:
    def __init__(self, output, model_config=None):
        self.output = output
        self.capture_lock = threading.Lock()
        self.models = {}
        if model_config:
            from brickcell.detector.runtime import OpenVINODetector

            for camera, settings in json.loads(model_config.read_text()).items():
                if camera not in CAMERAS:
                    raise ValueError("Unknown model camera")
                detector = OpenVINODetector(Path(settings["model"]))
                if detector.metadata.get("synthetic", True) or detector.threshold is None:
                    raise ValueError("Each camera needs a calibrated real-image model")
                self.models[camera] = detector, settings["crop"]
        self.cameras = {name: Camera(name, source) for name, (_, source) in CAMERAS.items()}
        for name, (_, crop) in self.models.items():
            self.cameras[name].crop = crop

    def capture(self, payload):
        if not payload.get("confirmed"):
            raise ValueError("Confirm that the same stationary brick is visible in every crop")
        selected = payload.get("cameras", list(CAMERAS))
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(name, str) or name not in CAMERAS for name in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("Select unique known cameras")
        for key in ("object_id", "session"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", payload.get(key, "")):
                raise ValueError("Use a valid physical object ID and session ID")
        if payload.get("label") not in ("good", "defective"):
            raise ValueError("A human label is required for dataset capture")
        # A fresh burst starts AFTER the operator's confirmation, never previous-object history.
        initial_crops = {name: self.cameras[name].snapshot()[2] for name in selected}
        if any(not crop for crop in initial_crops.values()):
            raise ValueError("Set a brick-face crop for each selected camera before capture")
        started = time.time()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            packets = {name: self.cameras[name].snapshot() for name in selected}
            if all(sum(t >= started for t, _ in frames) >= 10 for frames, _, _ in packets.values()):
                break
            time.sleep(0.1)
        batch = uuid.uuid4().hex
        views = []
        saved = {}
        times = {}
        for name, (frames, error, crop) in packets.items():
            frames = [(t, image) for t, image in frames if t >= started]
            fresh = frames and time.time() - frames[-1][0] < 2
            if error or not fresh or len(frames) < 10 or not crop:
                views.append(classify_burst(name, payload["object_id"], [], None))
                saved[name] = {"error": error or "Need crop and 10 fresh frames"}
                continue
            if crop != initial_crops[name]:
                raise ValueError("Camera crop changed during capture; retry")
            times[name] = frames[-1][0]
            roi = cropped(frames[-1][1], crop)
            path = (
                self.output
                / "raw"
                / payload["label"]
                / payload["session"]
                / payload["object_id"]
                / name
            )
            path.mkdir(parents=True, exist_ok=True)
            file = path / f"{batch}.png"
            if not cv2.imwrite(str(file), roi):
                raise RuntimeError("Could not save camera image")
            saved[name] = {"image": str(file), "crop": crop, "captured_at": frames[-1][0]}
            thumbs = [
                cv2.resize(cv2.cvtColor(cropped(im, crop), cv2.COLOR_BGR2GRAY), (64, 64)).astype(
                    np.float32
                )
                for _, im in frames[-10:]
            ]
            motion = max(float(np.mean(np.abs(a - b))) for a, b in zip(thumbs, thumbs[1:]))
            saved[name]["burst_max_motion_graylevels"] = motion
            scores = []
            detector, model_crop = self.models.get(name, (None, None))
            # Conservative motion heuristic; validate/tune on real calibration bursts.
            valid = detector is not None and model_crop == crop and motion <= 5
            saved[name]["model_version"] = detector.model_version if detector else None
            saved[name]["threshold_version"] = detector.threshold.version if detector else None
            if valid:
                from brickcell.inspection.images import QualityConfig, check_view

                for _, image in frames[-10:]:
                    rgb = cv2.cvtColor(cropped(image, crop), cv2.COLOR_BGR2RGB)
                    if not check_view(rgb, QualityConfig()).valid:
                        valid = False
                        break
                    scores.append(detector.predict(rgb).score)
            threshold = detector.threshold if detector else None
            views.append(
                classify_burst(
                    name,
                    payload["object_id"],
                    scores,
                    threshold.value if threshold else None,
                    threshold.uncertainty_margin if threshold else 0,
                    calibrated=valid,
                )
            )
        result = fuse_views(payload["object_id"], tuple(CAMERAS), views)
        if len(times) == 3 and max(times.values()) - min(times.values()) > 0.5:
            result.update(decision="REINSPECT", reason="Camera timestamps differ by more than 0.5s")
        result.update(
            batch_id=batch,
            views=[asdict(v) for v in views],
            captures=saved,
            human_label=payload["label"],
            synthetic=False,
            captured_cameras=selected,
            note="Software-aligned stationary-object burst; not hardware synchronization",
        )
        report = self.output / "reports" / f"{batch}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, allow_nan=False))
        return result


def handler(bench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, kind, body):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                route = self.path.split("?")[0]
                if route == "/":
                    self.send(200, "text/html; charset=utf-8", PAGE.encode())
                elif route == "/status":
                    self.send(
                        200,
                        "application/json",
                        json.dumps(
                            {
                                "cameras": [c.status() for c in bench.cameras.values()],
                                "calibrated_models": list(bench.models),
                            },
                            allow_nan=False,
                        ).encode(),
                    )
                elif route.startswith("/image/"):
                    cam = bench.cameras[route.rsplit("/", 1)[1]]
                    frames, error, crop = cam.snapshot()
                    if not frames or time.time() - frames[-1][0] > 2 or error:
                        raise ValueError(error or "No fresh frame")
                    image = frames[-1][1].copy()
                    if crop:
                        x0, y0, x1, y1 = crop
                        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 3)
                    h, w = image.shape[:2]
                    if w > 960:
                        image = cv2.resize(image, (960, round(h * 960 / w)))
                    ok, jpeg = cv2.imencode(".jpg", image)
                    if not ok:
                        raise ValueError("JPEG encode failed")
                    self.send(200, "image/jpeg", jpeg.tobytes())
                else:
                    self.send(404, "text/plain", b"Not found")
            except Exception as exc:
                self.send(503, "application/json", json.dumps({"error": str(exc)}).encode())

        def do_POST(self):
            acquired = False
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 8192:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/crop":
                    if bench.capture_lock.locked():
                        raise ValueError("Wait for the current capture before changing crops")
                    camera = bench.cameras[payload["camera"]]
                    crop = payload["crop"]
                    frames, _, _ = camera.snapshot()
                    if not frames:
                        raise ValueError("Camera must be connected before setting crop")
                    if crop is not None and (
                        len(crop) != 4
                        or any(not isinstance(v, int) or not math.isfinite(v) for v in crop)
                    ):
                        raise ValueError("Crop needs four integer pixel coordinates")
                    cropped(frames[-1][1], crop)
                    with camera.lock:
                        camera.crop = crop
                    result = {"camera": camera.name, "crop": crop}
                elif self.path == "/capture":
                    acquired = bench.capture_lock.acquire(blocking=False)
                    if not acquired:
                        raise ValueError("Another capture is running")
                    result = bench.capture(payload)
                else:
                    raise ValueError("Unknown operation")
                self.send(200, "application/json", json.dumps(result, allow_nan=False).encode())
            except Exception as exc:
                self.send(
                    400,
                    "application/json",
                    json.dumps({"error": str(exc), "decision": "ERROR"}).encode(),
                )
            finally:
                if acquired:
                    bench.capture_lock.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--output", type=Path, default=Path("data/multicam"))
    parser.add_argument("--models", type=Path)
    args = parser.parse_args()
    bench = Bench(args.output, args.models)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(bench))
    print(f"Three-camera bench ready on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        for camera in bench.cameras.values():
            camera.stopped.set()
        for camera in bench.cameras.values():
            camera.thread.join(timeout=4)


if __name__ == "__main__":
    main()
