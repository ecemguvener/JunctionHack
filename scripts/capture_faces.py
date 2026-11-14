"""Timed clean-face capture through the loopback camera bench."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--label", choices=("good", "defective"), default="good")
    parser.add_argument("--faces", nargs="+", required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--camera", default="side", choices=("side", "overhead", "wrist"))
    parser.add_argument("--output", type=Path, default=Path("data/face-capture-runs"))
    args = parser.parse_args()
    if args.interval < 3 or len(set(args.faces)) != len(args.faces):
        parser.error("interval must be >=3 seconds and face labels must be unique")
    records = []
    for index, face in enumerate(args.faces):
        print(f"\nFace {index + 1}/{len(args.faces)}: {face}", flush=True)
        if index == 0:
            input("Place the first face, then press Enter to capture... ")
        else:
            print("Rotate the same brick now.", flush=True)
            for remaining in range(int(args.interval), 0, -1):
                print(f"  {remaining:2d}s", end="\r", flush=True)
                time.sleep(1)
        payload = {
            "cameras": [args.camera],
            "confirmed": True,
            "object_id": args.object_id,
            "session": args.session,
            "label": args.label,
        }
        try:
            request = Request(
                args.url.rstrip("/") + "/capture",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=20) as response:
                result = json.load(response)
        except Exception as exc:
            print(f"Capture failed for {face}: {exc}", file=sys.stderr)
            return 1
        result.update(
            face=face,
            object_id=args.object_id,
            session_id=args.session,
            camera=args.camera,
            human_label=args.label,
            captured_at=datetime.now(timezone.utc).isoformat(),
            sequence_index=index,
        )
        records.append(result)
        print(f"Captured {face}: {result.get('batch_id')}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"{args.object_id}-{args.session}.json"
    output.write_text(json.dumps({"synthetic": False, "captures": records}, indent=2))
    print(f"Saved sequence record: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
