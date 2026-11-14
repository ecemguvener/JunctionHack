"""Capture aligned RGB/depth frames and point clouds from Intel RealSense D415s.

This is a hardware-side utility for the Intel laptop. It intentionally remains
separate from the core RGB inspection environment: ``pyrealsense2`` is optional
and is only required on the machine attached to the D415 cameras.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def capture(serial: str, output: Path, frames: int, width: int, height: int) -> dict[str, object]:
    try:
        import numpy as np
        import pyrealsense2 as rs
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - hardware-only dependency
        raise SystemExit("Install pyrealsense2, numpy and Pillow on the Intel laptop") from exc

    output.mkdir(parents=True, exist_ok=True)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    pointcloud = rs.pointcloud()
    try:
        for _ in range(20):
            pipeline.wait_for_frames()
        frameset = None
        for _ in range(frames):
            frameset = align.process(pipeline.wait_for_frames())
        assert frameset is not None
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError(f"{serial}: missing aligned color or depth frame")
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        color_path = output / f"{serial}-color.png"
        depth_path = output / f"{serial}-depth16.png"
        preview_path = output / f"{serial}-depth-preview.png"
        ply_path = output / f"{serial}.ply"
        Image.fromarray(color[:, :, ::-1]).save(color_path)
        Image.fromarray(depth).save(depth_path)
        depth_units = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_m = depth.astype(np.float32) * depth_units
        valid = depth_m[depth_m > 0]
        if valid.size:
            clipped = np.clip(depth_m / max(float(np.percentile(valid, 98)), 1e-6), 0, 1)
            Image.fromarray((255 * (1 - clipped)).astype(np.uint8)).save(preview_path)
        pointcloud.map_to(color_frame)
        points = pointcloud.calculate(depth_frame)
        points.export_to_ply(str(ply_path), color_frame)
        metadata = {
            "serial": serial,
            "width": int(color.shape[1]),
            "height": int(color.shape[0]),
            "depth_scale_m": float(depth_units),
            "valid_depth_pixels": int(valid.size),
            "depth_min_m": float(valid.min()) if valid.size else None,
            "depth_max_m": float(valid.max()) if valid.size else None,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "aligned_to": "color",
            "files": [color_path.name, depth_path.name, preview_path.name, ply_path.name],
            "synthetic": False,
        }
        (output / f"{serial}-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
    finally:
        pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    results = [
        capture(serial, args.output, args.frames, args.width, args.height) for serial in args.serial
    ]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
