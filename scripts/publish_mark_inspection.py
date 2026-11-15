"""Publish fresh real-camera dark-mark overlays to the existing Studio camera wall.

Visual heuristic only. No trained-model, calibrated-defect or robot-motion claims.
"""

import argparse
import base64
import io
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image

from brickcell.inspection.dark_marks import score_brick_roi_dark_mark
from brickcell.util import sha256


def jpeg(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="JPEG", quality=85)
    return base64.b64encode(stream.getvalue()).decode()


def locate_red_brick(rgb):
    """Locate one dominant red region; reject absent/ambiguous targets."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (
        ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 40)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [
        row
        for row in stats[1:count]
        if row[4] >= 200
        and min(row[2], row[3]) >= 20
        and row[4] < rgb.shape[0] * rgb.shape[1] * 0.2
    ]
    candidates.sort(key=lambda row: row[4], reverse=True)
    if not candidates or (len(candidates) > 1 and candidates[1][4] > candidates[0][4] * 0.35):
        return None
    x, y, w, h, _ = map(int, candidates[0])
    pad = max(8, round(max(w, h) * 0.12))
    return [
        max(0, x - pad),
        max(0, y - pad),
        min(rgb.shape[1], x + w + pad),
        min(rgb.shape[0], y + h + pad),
    ]


def inspect_frame(rgb, packet, crop, serial, contrast=16.0, detector=None):
    if (
        packet.get("synthetic") is not False
        or packet.get("serial") != serial
        or not packet.get("frame_id")
        or not math.isfinite(packet.get("time", float("nan")))
        or not 0 <= time.time() - packet["time"] <= 3
    ):
        raise ValueError("Require a fresh real frame with matching camera identity")
    if detector is not None:
        if detector.threshold is None:
            raise ValueError("Calibrated OpenVINODetector requires a frozen threshold")
        if crop is None:
            crop = [0, 0, rgb.shape[1], rgb.shape[0]]
        x0, y0, x1, y1 = crop
        h, w = rgb.shape[:2]
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or min(x1 - x0, y1 - y0) < 20:
            raise ValueError("Select an in-frame brick crop of at least 20x20")
        roi = rgb[y0:y1, x0:x1]
        from brickcell.inspection.block_surface import SurfaceViewError

        thr_val = float(detector.threshold.value)
        try:
            prediction = detector.predict(roi)
        except SurfaceViewError as exc:
            return dict(
                camera="plate",
                serial=serial,
                frame_id=packet["frame_id"],
                captured_at=packet["time"],
                synthetic=False,
                model_version=detector.model_version,
                threshold_version=detector.threshold.version,
                threshold=thr_val,
                decision="REINSPECT",
                reason=f"Calibrated PatchCore waiting for red brick surface ({exc})",
                raw_distance=None,
                image_jpeg=jpeg(rgb),
                overlay_jpeg=None,
                automatic_sorting_allowed=False,
            )
        score = float(prediction.score)
        decision = str(detector.threshold.decide(score))
        found = decision == "DEFECTIVE"
        overlay = rgb.copy()
        if prediction.anomaly_map is not None:
            amap = np.asarray(prediction.anomaly_map, dtype=np.float32)
            heat = cv2.applyColorMap(
                np.uint8(np.clip(amap / max(score, thr_val, 1e-6), 0, 1) * 255),
                cv2.COLORMAP_TURBO,
            )
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            heat = cv2.resize(heat, (x1 - x0, y1 - y0))
            overlay[y0:y1, x0:x1] = np.uint8(roi * 0.65 + heat * 0.35)
        color = (255, 70, 155) if found else (40, 220, 170)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        cv2.rectangle(overlay, (0, 0), (w, 46), (15, 22, 33), -1)
        label = (
            f"DEFECTIVE ({score:.3f} >= {thr_val:.3f}) - PINK PLATE"
            if found
            else f"{decision} ({score:.3f} < {thr_val:.3f}) - GREEN PLATE"
        )
        cv2.putText(overlay, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"PatchCore OpenVINO CPU | model {detector.model_version}",
            (10, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        reason = (
            f"Calibrated PatchCore DEFECTIVE ({score:.3f} >= {thr_val:.3f}); pink plate."
            if found
            else f"Calibrated PatchCore {decision} ({score:.3f} vs {thr_val:.3f}); green plate."
        )
        return dict(
            camera="plate",
            serial=serial,
            frame_id=packet["frame_id"],
            captured_at=packet["time"],
            synthetic=False,
            model_version=detector.model_version,
            threshold_version=detector.threshold.version,
            threshold=thr_val,
            decision=decision if decision in ("GOOD", "DEFECTIVE") else "UNCALIBRATED",
            reason=reason,
            raw_distance=score,
            image_jpeg=jpeg(rgb),
            overlay_jpeg=jpeg(overlay),
            automatic_sorting_allowed=True,
        )
    if not math.isfinite(contrast) or contrast <= 0:
        raise ValueError("Contrast must be finite and positive")
    scorer = Path(__file__).resolve().parents[1] / "src/brickcell/inspection/dark_marks.py"
    body = dict(
        camera="plate",
        serial=serial,
        frame_id=packet["frame_id"],
        captured_at=packet["time"],
        synthetic=False,
        model_version=f"dark-mark-heuristic-{sha256(scorer)[:12]}-contrast-{contrast:g}",
        threshold_version=None,
        decision="REINSPECT",
        reason="No unambiguous red brick located; waiting for a valid view.",
        raw_distance=None,
        image_jpeg=jpeg(rgb),
        overlay_jpeg=None,
        automatic_sorting_allowed=False,
    )
    if crop is None:
        return body
    x0, y0, x1, y1 = crop
    h, w = rgb.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or min(x1 - x0, y1 - y0) < 20:
        raise ValueError("Select an in-frame brick crop of at least 20x20")
    score, boxes = score_brick_roi_dark_mark(rgb[y0:y1, x0:x1])
    found = [box for box in boxes if box["score"] >= contrast]
    label = "MARK DETECTED - PINK PLATE" if found else "NO MARK FOUND IN SELECTED VIEW"
    reason = (
        "Dark mark detected; proposed destination: pink plate. Surface scan; motion disabled."
        if found
        else "No dark mark found in this view. Surface scan; full-brick status unverified."
    )
    overlay = rgb.copy()
    color = (255, 70, 155) if found else (40, 220, 170)
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 2)
    for box in found:
        bx0, by0, bx1, by1 = box["box"]
        cv2.rectangle(overlay, (x0 + bx0 - 2, y0 + by0 - 2), (x0 + bx1 + 2, y0 + by1 + 2), color, 2)
    cv2.rectangle(overlay, (0, 0), (w, 46), (15, 22, 33), -1)
    cv2.putText(overlay, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.putText(
        overlay,
        "PatchCore Inspection | surface scan",
        (10, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    body.update(
        decision="UNCALIBRATED", reason=reason, raw_distance=score, overlay_jpeg=jpeg(overlay)
    )
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="http://127.0.0.1:18094/snapshot/plate")
    parser.add_argument("--studio-url", default="http://127.0.0.1:17860")
    parser.add_argument("--serial", default="243622060450")
    parser.add_argument(
        "--crop", type=int, nargs=4, help="Fixed brick crop; default tracks one red brick"
    )
    parser.add_argument("--contrast", type=float, default=16.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument(
        "--openvino-model",
        type=Path,
        help="Optional calibrated OpenVINO model.xml for PatchCore plate inspection",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 0.1:
        parser.error("interval must be finite and at least 0.1 seconds")
    if not math.isfinite(args.contrast) or args.contrast <= 0:
        parser.error("contrast must be finite and positive")
    detector = None
    if args.openvino_model is not None:
        from brickcell.detector.runtime import OpenVINODetector

        detector = OpenVINODetector(args.openvino_model)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    last_frame = None
    while True:
        try:
            with urlopen(args.source, timeout=4) as response:
                packet = json.load(response)
            if packet.get("frame_id") == last_frame:
                raise ValueError("Repeated camera frame")
            rgb = np.asarray(
                Image.open(
                    io.BytesIO(base64.b64decode(packet["rgb_jpeg_b64"], validate=True))
                ).convert("RGB")
            )
            if args.crop:
                crop = args.crop
            elif detector is not None:
                contract_crop = (detector.metadata.get("sensor_contract") or {}).get("crop")
                crop = contract_crop if contract_crop else [0, 0, rgb.shape[1], rgb.shape[0]]
            else:
                crop = locate_red_brick(rgb)
            body = inspect_frame(rgb, packet, crop, args.serial, args.contrast, detector=detector)
            if time.time() - body["captured_at"] > 3:
                raise ValueError("Inspection exceeded frame freshness deadline")
            request = Request(
                args.studio_url.rstrip("/") + "/api/so101/inspection",
                data=json.dumps(body, allow_nan=False).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=3) as response:
                json.load(response)
            last_frame = body["frame_id"]
            event = {k: v for k, v in body.items() if k not in ("image_jpeg", "overlay_jpeg")}
            event.update(crop=crop, contrast=args.contrast, received_at=time.time())
            with args.events.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(f"{last_frame}: {body['reason']}", flush=True)
        except Exception as exc:
            if args.once:
                raise
            print(f"Inspection unavailable: {exc}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
