"""Save repeated same-frameset RGB/uint16-depth snapshots for one physical object."""

import argparse
import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

from brickcell.inspection.rgbd import decode_snapshot, encode_depth
from brickcell.sources.camera import identifier
from brickcell.util import save_json, versions


def fetch(source, camera):
    with urlopen(source.rstrip("/") + "/snapshot/" + camera, timeout=5) as response:
        packet = json.load(response)
    if not -0.5 <= time.time() - packet["time"] <= 3 or not packet.get("frame_id"):
        raise ValueError("Snapshot is stale or lacks identity")
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="http://127.0.0.1:18094")
    parser.add_argument("--object-id", type=identifier, required=True)
    parser.add_argument("--session-id", type=identifier, required=True)
    parser.add_argument("--label", choices=["good", "defective", "review"], required=True)
    parser.add_argument("--pose", type=identifier, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--interval", type=float, default=0.3)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.count <= 100 or not 0.1 <= args.interval <= 10:
        parser.error("Use a new output directory, count 2–100 and interval .1–10 seconds")
    args.output.mkdir(parents=True)
    records = []
    seen = {camera: set() for camera in ["top", "plate"]}
    serials = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index in range(args.count):
            futures = {camera: pool.submit(fetch, args.source, camera) for camera in seen}
            pair = {camera: future.result() for camera, future in futures.items()}
            for camera, packet in pair.items():
                _rgb, depth = decode_snapshot(packet)
                if packet["frame_id"] in seen[camera]:
                    raise ValueError("Repeated frame ID; capture cannot claim fresh observations")
                if camera in serials and serials[camera] != packet["serial"]:
                    raise ValueError("Camera identity changed during sequence")
                serials[camera] = packet["serial"]
                seen[camera].add(packet["frame_id"])
                folder = args.output / camera
                folder.mkdir(exist_ok=True)
                stem = f"{index:03d}"
                paths = {
                    name: folder / (stem + suffix)
                    for name, suffix in [
                        ("snapshot", ".json"),
                        ("rgb", "-rgb.jpg"),
                        ("depth", "-depth16.png"),
                        ("depth_preview", "-depth-preview.png"),
                    ]
                }
                save_json(paths["snapshot"], packet)
                paths["rgb"].write_bytes(base64.b64decode(packet["rgb_jpeg_b64"], validate=True))
                paths["depth"].write_bytes(base64.b64decode(packet["depth_png_b64"], validate=True))
                preview, valid = encode_depth(depth, packet["depth_scale_m"])
                Image.fromarray(preview).save(paths["depth_preview"])
                records.append(
                    dict(
                        camera=camera,
                        frame_id=packet["frame_id"],
                        serial=packet["serial"],
                        captured_at=packet["time"],
                        depth_scale_m=packet["depth_scale_m"],
                        files={
                            name: str(path.relative_to(args.output)) for name, path in paths.items()
                        },
                        sha256={
                            name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in paths.items()
                        },
                        valid_depth_fraction=float(valid.mean()),
                    )
                )
            print(f"Saved pair {index + 1}/{args.count}", flush=True)
            if index + 1 < args.count:
                time.sleep(args.interval)
    report = dict(
        object_id=args.object_id,
        session_id=args.session_id,
        human_label=args.label,
        pose=args.pose,
        synthetic=False,
        inspection_result=None,
        review_status="pending_manual_image_review",
        eligible_for_training=False,
        dataset_partition=None,
        snapshots=records,
        versions=versions(),
        notes=[
            "RGB/depth matched within each camera frameset",
            "Cameras not hardware synchronized",
            "Repeated frames are one physical object/session group, not independent samples",
        ],
    )
    save_json(args.output / "capture-record.json", report)
    print(
        json.dumps(
            dict(output=str(args.output), per_camera_count=args.count, object_id=args.object_id),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
