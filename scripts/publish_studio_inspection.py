"""Publish real same-frame PatchCore previews to Physical AI Studio camera views.

Reference pilots are always UNCALIBRATED. This process has no robot interface.
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch
from anomalib.models import Patchcore
from PIL import Image

from brickcell.inspection.block_surface import SurfaceViewError, surface_input, surface_score
from brickcell.util import sha256


def jpeg(rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((1280, 1280))
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=85)
    return base64.b64encode(stream.getvalue()).decode()


def inspect_frame(model, metadata, rgb, packet, crop):
    """Return display evidence; a failed view never receives a heatmap or verdict."""
    if packet.get("synthetic") is not False:
        raise ValueError("Only explicitly real camera frames may be published")
    if packet.get("serial") != metadata["serial"]:
        raise ValueError("Camera serial does not match reference sensor")
    if not packet.get("frame_id") or not 0 <= time.time() - packet["time"] <= 3:
        raise ValueError("Missing acquisition metadata or stale frame")
    body = dict(
        camera=metadata["camera"],
        serial=packet["serial"],
        frame_id=packet["frame_id"],
        captured_at=packet["time"],
        synthetic=False,
        model_version=metadata["checkpoint_sha256"],
        threshold_version=None,
        decision="REINSPECT",
        reason="Invalid crop",
        raw_distance=None,
        image_jpeg=jpeg(rgb),
        overlay_jpeg=None,
    )
    x0, y0, x1, y1 = crop
    h, w = rgb.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        return body
    query = rgb[y0:y1, x0:x1]
    try:
        tensor, mask = surface_input(
            query, tuple(metadata["image_size"]), metadata.get("surface_profile")
        )
        with torch.inference_mode():
            prediction = model(torch.from_numpy(tensor))
        distance, amap = surface_score(
            float(prediction.pred_score.item()),
            prediction.anomaly_map.squeeze().cpu().numpy(),
            mask,
        )
        # Same frame, same crop; colors show relative response, not a defect boundary.
        heat = cv2.applyColorMap(
            np.uint8(np.clip(amap / max(distance, 1e-8), 0, 1) * 255), cv2.COLORMAP_TURBO
        )
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        heat = cv2.resize(heat, (x1 - x0, y1 - y0))
        overlay = rgb.copy()
        overlay[y0:y1, x0:x1] = np.uint8(query * 0.55 + heat * 0.45)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 0), 1)
        body.update(
            decision="UNCALIBRATED",
            reason="PatchCore preview; heat colors are relative, no pass/fail threshold",
            raw_distance=distance,
            overlay_jpeg=jpeg(overlay),
        )
    except (SurfaceViewError, ValueError) as exc:
        body["reason"] = str(exc)[:500]
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--studio-url", action="append", required=True)
    parser.add_argument("--source", default="http://127.0.0.1:18094/snapshot/plate")
    parser.add_argument("--serial", required=True, help="Expected real camera serial")
    parser.add_argument("--crop", type=int, nargs=4, required=True)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 0.1:
        parser.error("interval must be at least 0.1 seconds")
    metadata = json.loads((args.pilot / "pilot.json").read_text())
    if metadata.get("camera") != "plate" or metadata.get("synthetic") is not False:
        raise ValueError("This snapshot publisher requires a real plate RGB reference pilot")
    metadata["serial"] = args.serial
    checkpoint = args.pilot / "reference.ckpt"
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Reference checkpoint hash mismatch")
    torch.set_num_threads(4)
    model = (
        Patchcore.load_from_checkpoint(str(checkpoint), map_location="cpu", pre_trained=False)
        .cpu()
        .eval()
    )
    while True:
        try:
            with urlopen(args.source, timeout=4) as response:
                packet = json.load(response)
            rgb = np.asarray(
                Image.open(
                    io.BytesIO(base64.b64decode(packet["rgb_jpeg_b64"], validate=True))
                ).convert("RGB")
            )
            body = inspect_frame(model, metadata, rgb, packet, args.crop)
            for studio in args.studio_url:
                request = Request(
                    studio.rstrip("/") + "/api/so101/inspection",
                    data=json.dumps(body, allow_nan=False).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=3) as response:
                    json.load(response)
            print(f"{body['frame_id']}: {body['decision']} · {body['reason']}", flush=True)
        except Exception as exc:
            if args.once:
                raise
            print(f"Inspection unavailable: {exc}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
