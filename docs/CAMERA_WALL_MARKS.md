# Camera-wall mark inspection

Live mark overlays appear in Physical AI Studio's existing **SO-101 Architecture
→ Live Twin & Control → Live Camera Feeds** plate-camera tile. They use the real
SDK RGB snapshot, not generated images. The publisher does not control the arm.

The image-only CIE-Lab symmetric-flank heuristic highlights narrow dark strokes.
A stroke above the configured contrast setting produces pink boxes and
`MARK DETECTED - PINK PLATE`. The overlay's workflow title is **PatchCore Inspection**.
This is display branding; the actual detector remains the CIE-Lab rule and its
backend source/version metadata retain that identity. The status reports a
surface scan and a proposed destination; it is not a calibrated PatchCore decision or proof
that an object was physically sorted. A clear-looking view does not establish
that the whole brick is good. Missing, ambiguous or stale views cannot authorize
sorting.

The default tracker follows one dominant red region and rejects absent or
ambiguous targets. It is a color heuristic, not verified object identity. Use an
explicit crop for other brick colors or a stationary inspection area. Background
red objects, occlusion, specular highlights and markings wider than the stroke
filter remain limitations. The existing scorer also excludes blue plate and
purple gripper colors; it is not validated on blue/purple bricks.

On the laptop with the existing SDK/Studio SSH forwards:

```sh
uv sync --locked
uv run python scripts/publish_mark_inspection.py \
  --events artifacts/camera-wall-marks/events.jsonl
```

For an explicit brick crop, add `--crop x0 y0 x1 y1`. Avoid old fixed crops when
the brick moves. `--contrast 16` is the existing Lab heuristic setting, not a
validation-calibrated defect threshold. `--once` publishes one fresh frame.

The current independent Intel publisher runs from:

`/home/intel/brickcell-deploy/artifacts/camera-wall-runtime-v1`

Its `publisher.log`, `publisher.pid` and `events.jsonl` record operation. It uses
the existing minimal OpenVINO environment, needs only NumPy/OpenCV/Pillow for
the image heuristic, and reads the existing camera service without opening a
camera device. Equivalent launch from an updated Intel checkout:

```sh
PYTHONPATH=src .venv/bin/python scripts/publish_mark_inspection.py \
  --source http://127.0.0.1:8094/snapshot/plate \
  --studio-url http://127.0.0.1:7860 \
  --events artifacts/camera-wall-marks/events.jsonl
```

Use one publisher per camera; older reference publishers can overwrite its
frames. Source and overlay share the same immutable acquisition ID and time.
The existing Studio API rejects replayed/out-of-order/stale evidence and clears
overlays when they expire. JSONL events preserve the actual crop, contrast,
scorer source hash, frame identity, score and result. All mark results remain
UNCALIBRATED and `automatic_sorting_allowed: false`; no physical sort gate was
removed or bypassed.
