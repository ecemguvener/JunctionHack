"""Dedicated BabyROS node running the calibrated PatchCore OpenVINO model on the wrist camera.

Publishes wrist RGB frames onto `so101/camera/wrist/frame` (unless `--subscribe-only` is set for
external `camera_pub`), runs OpenVINO CPU anomaly inference inside `WristAnomalyNode`, publishes
structured inspection events on `so101/perception/wrist_anomaly`, appends optional JSONL logs, and
updates `camui.py` (`POST /api/detection` with `camera="wrist"`). Never issues robot commands.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from brickcell.block_inspector.camera_adapters import CamUiReporter
from brickcell.block_inspector.hardware_cell import LiveBenchHttpFrameSource
from brickcell.block_inspector.runtime_bus import (
    CANONICAL_TOPICS,
    BabyRosRuntimeBus,
    pack_rgbd_frame_nddc,
)
from brickcell.block_inspector.runtime_nodes import WristAnomalyNode
from brickcell.detector.runtime import OpenVINODetector

DEFAULT_MODEL = Path(
    "/home/intel/brickcell-deploy/artifacts/models/plate-calibrated-707-v1/model.xml"
)


def parse_crop(text: str | None) -> list[int] | None:
    if not text:
        return None
    parts = [int(v.strip()) for v in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Crop must have four integers: x0,y0,x1,y1")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--openvino-model",
        type=Path,
        default=Path(os.environ.get("WRIST_OPENVINO_MODEL", str(DEFAULT_MODEL))),
        help="Path to calibrated OpenVINO model.xml",
    )
    parser.add_argument(
        "--wrist-url",
        default=os.environ.get("WRIST_HTTP_URL", "http://127.0.0.1:8091/frame.jpg"),
        help="Live wrist camera JPEG URL",
    )
    parser.add_argument(
        "--camui-url",
        default=os.environ.get("CAMUI_URL", "http://127.0.0.1:8093"),
        help="Camera wall base URL (POST /api/detection)",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help="Optional JSONL event log path",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Optional ROI crop x0,y0,x1,y1 (default: full wrist frame)",
    )
    parser.add_argument("--serial", default=os.environ.get("WRIST_SERIAL", "sonix-wrist"))
    parser.add_argument(
        "--hz",
        type=float,
        default=float(os.environ.get("WRIST_ANOMALY_HZ", "4.0")),
    )
    parser.add_argument(
        "--subscribe-only",
        action="store_true",
        help="Consume external `so101/camera/wrist/frame` BabyROS messages instead of polling HTTP",
    )
    parser.add_argument("--once", action="store_true", help="Score one wrist frame and exit")
    args = parser.parse_args()

    shm_bytes = int(os.environ.get("BABYROS_SHM_BYTES", "134217728"))
    detector = OpenVINODetector(args.openvino_model)
    camui = CamUiReporter(base_url=args.camui_url) if args.camui_url else None
    bus = BabyRosRuntimeBus(shm_pool_bytes=shm_bytes)
    node = WristAnomalyNode(
        bus=bus,
        detector=detector,
        camui=camui,
        crop_box=parse_crop(args.crop),
        serial=args.serial,
        events_path=args.events,
    )
    frame_source = (
        None
        if args.subscribe_only
        else LiveBenchHttpFrameSource(
            live_depth_url="http://127.0.0.1:8094",
            wrist_http_url=args.wrist_url,
        )
    )
    period_s = 1.0 / max(0.5, float(args.hz))
    try:
        while True:
            t0 = time.monotonic()
            if frame_source is not None:
                try:
                    snap = frame_source.fetch_wrist_snapshot()
                    payload = pack_rgbd_frame_nddc(snap.rgb, time.time_ns())
                    payload["frame_id"] = snap.frame_id
                    bus.publish(CANONICAL_TOPICS["wrist_frame"], payload)
                except Exception as exc:
                    print(f"[wrist_anomaly_node] frame fetch error: {exc}", flush=True)
            if args.once:
                time.sleep(0.15)
                break
            rem = period_s - (time.monotonic() - t0)
            if rem > 0.0:
                time.sleep(rem)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        bus.close()


if __name__ == "__main__":
    main()
