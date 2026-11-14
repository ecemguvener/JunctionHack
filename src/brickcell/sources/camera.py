"""Explicit, on-demand RGB camera capture. This module never commands a robot."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from brickcell.inspection.images import QualityConfig, check_view
from brickcell.sources.directory import Frame
from brickcell.util import json_default, save_json


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("Use letters, digits, hyphens or underscores for IDs")
    return value


def crop_rgb(bgr: np.ndarray, crop: list[int] | None) -> np.ndarray:
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Camera must return a nonempty uint8 BGR color frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if crop:
        x0, y0, x1, y1 = crop
        h, w = rgb.shape[:2]
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"Crop {crop} is outside the {w}x{h} camera frame")
        rgb = rgb[y0:y1, x0:x1]
    if min(rgb.shape[:2]) < 16:
        raise ValueError("Captured image/crop must be at least 16x16")
    return np.ascontiguousarray(rgb)


class OpenCVCamera:
    """Open only the explicitly selected camera; do not scan other cameras."""

    def __init__(self, source: str, warmup: int = 10):
        if warmup < 0:
            raise ValueError("warmup must be nonnegative")
        self.source = source
        self.warmup = warmup
        self.device = None

    def __enter__(self):
        source = int(self.source) if self.source.isdecimal() else self.source
        self.device = cv2.VideoCapture(source)
        try:
            if not self.device.isOpened():
                raise RuntimeError(
                    f"Cannot open camera {self.source}; check device and permissions"
                )
            for _ in range(self.warmup):
                self.read()
        except Exception:
            self.device.release()
            raise
        return self

    def read(self) -> np.ndarray:
        ok, bgr = self.device.read()
        if not ok or bgr is None or not bgr.size:
            raise RuntimeError("Camera returned no frame; no inspection decision was made")
        return bgr

    def __exit__(self, *_):
        self.device.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Camera index or Linux stable video path")
    parser.add_argument("--label", required=True, choices=["good", "defective"])
    parser.add_argument("--session", required=True, type=identifier)
    parser.add_argument("--object-id", required=True, type=identifier)
    parser.add_argument("--view", default="front", type=identifier)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--quality-config", type=Path, default=Path("configs/inspection.yaml"))
    parser.add_argument("--model", type=Path, help="Optional real-camera-trained OpenVINO model")
    args = parser.parse_args(argv)
    try:
        import yaml

        quality = QualityConfig(**yaml.safe_load(args.quality_config.read_text()))
        # Load before accessing camera; never apply a synthetic calibration to real images.
        detector = None
        if args.model:
            from brickcell.detector.runtime import OpenVINODetector

            detector = OpenVINODetector(args.model)
            if detector.metadata.get("synthetic", True):
                raise ValueError("Use a real-camera-trained model, not the synthetic demo model")
        with OpenCVCamera(args.source, args.warmup) as camera:
            rgb = crop_rgb(camera.read(), args.crop)
        timestamp = datetime.now(timezone.utc).isoformat()
        frame_id = uuid.uuid4().hex
        path = args.output / args.label / args.session / args.object_id / args.view
        path.mkdir(parents=True, exist_ok=True)
        image = path / f"{frame_id}.png"
        Image.fromarray(rgb).save(image)
        report = {
            "image": str(image.resolve()),
            "source": args.source,
            "timestamp": timestamp,
            "object_id": args.object_id,
            "session_id": args.session,
            "view_id": args.view,
            "human_label": args.label,
            "crop": args.crop,
            "synthetic": False,
            "quality": asdict(check_view(rgb, quality)),
            "inspection": None,
        }
        if detector:
            from brickcell.inspection.service import InspectionService

            result = InspectionService(
                detector, quality, Path("artifacts/camera") / frame_id
            ).inspect(Frame(args.object_id, frame_id, timestamp, args.view, image))
            report["inspection"] = asdict(result)
        save_json(image.with_suffix(".json"), report)
        print(json.dumps(report, default=json_default, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "decision": "ERROR"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
