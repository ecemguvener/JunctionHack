"""Run the CPU PatchCore RGB/depth review UI beside the Intel SDK camera service."""

import argparse
import base64
import html
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image

from brickcell.detector.runtime import OpenVINODetector
from brickcell.inspection.rgbd import RGBDInspector, RGBInspector, encode_depth

PAGE = """<!doctype html><meta charset=utf-8><title>PatchCore + depth review</title>
<style>body{background:#0b1422;color:#e6f0ff;font:17px system-ui;margin:30px}article{display:inline-block;vertical-align:top;background:#17263b;border:1px solid #3d5677;border-radius:12px;padding:20px;width:44%;margin:1%}img{width:48%}pre{white-space:pre-wrap;font:14px monospace}iframe{width:100%;height:700px;border:0}input,button{padding:8px;font:inherit}@media(max-width:800px){article{width:90%}}</style>
<h1>PatchCore RGB + depth viewer</h1><p>D415 RGB and aligned depth come from the same frameset. The arm camera supplies RGB only. Each modality needs its own real trained model and calibrated threshold. Scores are distances, not probabilities.</p>
<p><strong>Review only:</strong> camera views do not verify hidden surfaces or object identity. No robot commands are issued.</p><div id=views></div>
<h2>Live 3D views</h2><iframe title="Live D415 depth clouds" src="SOURCE_URL"></iframe>
<script>for(const name of CAMERA_NAMES){let card=document.createElement('article');card.innerHTML=`<h2>${name === 'wrist' ? 'Arm / wrist (RGB only)' : name}</h2><p class=status>Connecting…</p><label>Brick crop (x0,y0,x1,y1) <input placeholder="e.g. 220,190,280,260"></label><button>Apply</button><p>${RGB_CAMERAS.includes(name) ? 'RGB input' : 'RGB input · depth input'}</p><img class=rgb><img class=depth><p>${RGB_CAMERAS.includes(name) ? 'RGB anomaly overlay' : 'RGB anomaly overlay · depth anomaly overlay'} (when model returns a map)</p><img class=rgbmap><img class=depthmap><pre></pre>`;views.append(card);card.querySelector('button').onclick=async()=>{try{let crop=card.querySelector('input').value.split(',').map(Number);let r=await fetch('/crop/'+name,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({crop})});let d=await r.json();if(!r.ok)throw Error(d.error);card.querySelector('.status').textContent='Crop updated'}catch(e){card.querySelector('.status').textContent=e.message}};
async function tick(){try{let r=await fetch('/api/'+name);let d=await r.json();if(!r.ok)throw Error(d.error);card.querySelector('.status').textContent=d.result.decision+' · '+d.result.reason;card.querySelector('pre').textContent=JSON.stringify(d.result,null,2);for(let [key,selector] of [['rgb','.rgb'],['depth','.depth'],['rgb_map','.rgbmap'],['depth_map','.depthmap']]){let img=card.querySelector(selector);if(d.images[key]){img.src='data:image/png;base64,'+d.images[key];img.hidden=false}else{img.removeAttribute('src');img.hidden=true}}}catch(e){card.querySelector('.status').textContent=e.message;card.querySelector('pre').textContent='No current result';for(let img of card.querySelectorAll('img')){img.removeAttribute('src');img.hidden=true}}setTimeout(tick,1500)}tick()}
</script>"""


def png(image):
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def decode_snapshot(packet):
    if packet.get("synthetic") is not False or packet.get("aligned_to") != "color":
        raise ValueError("Require real depth aligned to color")
    rgb = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(packet["rgb_jpeg_b64"], validate=True))).convert(
            "RGB"
        )
    )
    depth = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(packet["depth_png_b64"], validate=True)))
    )
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("Expected lossless uint16 depth PNG")
    return rgb, depth


def read_rgb_frame(source):
    """Read the helper's upright raw JPEG; only capture metadata permits inference.

    Receive time cannot establish capture freshness. Legacy helpers still provide
    a preview, but must add the three headers below to enable trusted inspection.
    """
    with urlopen(source, timeout=4) as response:
        rgb = np.asarray(Image.open(io.BytesIO(response.read())).convert("RGB"))
        packet = {
            "time": float(response.headers.get("X-Frame-Time", "nan")),
            "frame_id": response.headers.get("X-Frame-Id", ""),
            "serial": response.headers.get("X-Camera-Serial", ""),
        }
    if not packet["frame_id"] or not np.isfinite(packet["time"]):
        packet.update(time=0, frame_id="")
    return rgb, packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="http://127.0.0.1:18094")
    parser.add_argument("--config", type=Path, default=Path("configs/rgbd-viewer.json"))
    parser.add_argument("--port", type=int, default=18097)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    inspectors = {}
    lock = threading.Lock()
    for camera, settings in config["cameras"].items():
        detectors = {}
        for modality, model in settings.get("models", {}).items():
            model_path = (args.config.parent / model).resolve()
            metadata = json.loads(model_path.with_suffix(".json").read_text())
            allowed = ("rgb",) if settings.get("type") == "rgb" else ("rgb", "depth")
            if modality not in allowed or metadata.get("synthetic", True):
                raise ValueError("Refusing synthetic/unknown model before native compilation")
            detectors[modality] = OpenVINODetector(model_path)
        inspector_type = RGBInspector if settings.get("type") == "rgb" else RGBDInspector
        inspectors[camera] = inspector_type(
            settings["serial"],
            settings.get("crop"),
            detectors,
            settings.get("min_depth_fraction", 0.6),
        )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, kind="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(
                    200,
                    PAGE.replace("SOURCE_URL", html.escape(args.source, quote=True))
                    .replace("CAMERA_NAMES", json.dumps(list(inspectors)))
                    .replace(
                        "RGB_CAMERAS",
                        json.dumps(
                            [
                                name
                                for name, inspector in inspectors.items()
                                if isinstance(inspector, RGBInspector)
                            ]
                        ),
                    )
                    .encode(),
                    "text/html; charset=utf-8",
                )
                return
            camera = self.path.removeprefix("/api/")
            if not self.path.startswith("/api/") or camera not in inspectors:
                self.send_error(404)
                return
            try:
                if not lock.acquire(blocking=False):
                    self.send(503, b'{"error":"Inference busy; retrying"}')
                    return
                try:
                    settings = config["cameras"][camera]
                    if isinstance(inspectors[camera], RGBInspector):
                        rgb, packet = read_rgb_frame(settings["source"])
                        result, previews = inspectors[camera].inspect(
                            rgb, packet["time"], packet["serial"], packet["frame_id"]
                        )
                        if not packet["frame_id"]:
                            result.update(
                                decision="REINSPECT",
                                reason="Preview only: capture metadata missing from arm camera source",
                            )
                    else:
                        with urlopen(args.source + "/snapshot/" + camera, timeout=4) as response:
                            packet = json.load(response)
                        rgb, depth = decode_snapshot(packet)
                        result, previews = inspectors[camera].inspect(
                            rgb,
                            depth,
                            packet["depth_scale_m"],
                            packet["time"],
                            packet["serial"],
                            packet["frame_id"],
                        )
                    if packet["frame_id"] and time.time() - packet["time"] > 3:
                        result.update(
                            decision="REINSPECT",
                            reason="Result exceeded freshness deadline",
                            channels={},
                            depth_used_for_anomaly_score=False,
                        )
                        previews = {}
                    images = {}
                    for modality in inspectors[camera].modalities:
                        image = previews.get(modality)
                        if image is None:
                            image = (
                                rgb
                                if modality == "rgb"
                                else encode_depth(depth, packet["depth_scale_m"])[0]
                            )
                        images[modality] = png(image)
                        amap = previews.get(modality + "_map")
                        if amap is not None:
                            scaled = (amap - amap.min()) / max(float(np.ptp(amap)), 1e-8)
                            heat = cv2.applyColorMap(
                                (scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
                            )[:, :, ::-1]
                            heat = cv2.resize(heat, (image.shape[1], image.shape[0]))
                            images[modality + "_map"] = png(
                                (0.6 * image + 0.4 * heat).astype(np.uint8)
                            )
                    result["map_display"] = (
                        "Per-frame normalized heatmaps; colors are not calibrated thresholds"
                    )
                    body = json.dumps(dict(result=result, images=images), allow_nan=False).encode()
                finally:
                    lock.release()
                self.send(200, body)
            except Exception as exc:
                self.send(503, json.dumps(dict(error=str(exc), decision="ERROR")).encode())

        def do_POST(self):
            camera = self.path.removeprefix("/crop/")
            if not self.path.startswith("/crop/") or camera not in inspectors:
                self.send_error(404)
                return
            try:
                if self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Expected application/json")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 1024:
                    raise ValueError("Invalid request size")
                crop = json.loads(self.rfile.read(size))["crop"]
                if (
                    len(crop) != 4
                    or any(type(v) is not int for v in crop)
                    or not (0 <= crop[0] < crop[2] and 0 <= crop[1] < crop[3])
                    or min(crop[2] - crop[0], crop[3] - crop[1]) < 16
                ):
                    raise ValueError(
                        "Require nonnegative integer crop of at least 16x16; frame bounds checked at inference"
                    )
                with lock:
                    old = inspectors[camera]
                    inspectors[camera] = type(old)(
                        old.serial, crop, old.detectors, old.min_depth_fraction, old.quality
                    )
                self.send(
                    200,
                    json.dumps(
                        dict(
                            camera=camera,
                            crop=crop,
                            persistence="Runtime only; update config to keep after restart",
                        )
                    ).encode(),
                )
            except Exception as exc:
                self.send(400, json.dumps(dict(error=str(exc))).encode())

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PatchCore/depth review: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
