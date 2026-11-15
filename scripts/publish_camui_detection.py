"""Bridge fresh live detector events to camui; display only, no robot commands."""

import argparse
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen


def detection_payload(event, now=None):
    now = time.time() if now is None else now
    neutral = dict(
        malformed=None,
        detail="No fresh supported detector result",
        camera="side",
        captured_at=now,
        source="live_dark_mark_heuristic_not_patchcore",
    )
    stamp = event.get("captured_at")
    if (
        event.get("synthetic") is not False
        or event.get("camera") != "plate"
        or event.get("serial") != "243622060450"
        or not isinstance(event.get("frame_id"), str)
        or not event["frame_id"]
        or type(stamp) not in (int, float)
        or not math.isfinite(stamp)
        or not 0 <= now - stamp <= 3
    ):
        return neutral
    score, threshold = event.get("raw_distance"), event.get("contrast")
    if (
        event.get("decision") != "UNCALIBRATED"
        or not str(event.get("model_version", "")).startswith("dark-mark-heuristic-")
        or type(score) not in (int, float)
        or not math.isfinite(score)
        or score < 0
        or type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        return neutral
    found = score >= threshold
    return dict(
        malformed=found,
        detail=("DARK MARK FLAGGED" if found else "No dark mark flagged in this view")
        + " | unvalidated dark-mark rule; separate from colour sorting",
        camera="side",
        captured_at=stamp,
        frame_id=event["frame_id"],
        source=neutral["source"],
    )


def latest_event(path):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        lines = stream.read().splitlines()
    if not lines:
        raise ValueError("No detector events")
    return json.loads(lines[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--camui-url", default="http://127.0.0.1:8093")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            event = latest_event(args.events)
        except (OSError, ValueError):
            event = {}
        body = detection_payload(event)
        try:
            request = Request(
                args.camui_url.rstrip("/") + "/api/detection",
                data=json.dumps(body, allow_nan=False).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                response.read()
        except OSError as exc:
            print(f"Camera warning unavailable: {exc}", flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(0.5)


if __name__ == "__main__":
    main()
