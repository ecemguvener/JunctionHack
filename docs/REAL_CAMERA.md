# First real-camera inspection

Goal: capture a real brick image, train/calibrate on real images, and inspect a
new capture. No robot movement is needed. The simulated-camera model is not a
real-camera model. The capture command refuses synthetic models.

## 1. Choose and check the camera

Run from the repository root in the core Python 3.12 environment after
`uv sync --locked`. If `uv` is unavailable but `.venv` is installed, replace
`uv run python` with `.venv/bin/python`.

For a Mac webcam, use `--source 0` below (or the known index of your USB camera).
macOS may ask for camera permission for your terminal. Capture is on demand;
the command does not scan devices or access cameras until explicitly run.

For the D415 on the Intel station, read `hardware.md` and `ssh.md`. List
`/dev/v4l/by-id/` and select its RGB/color device, not a depth/infrared node or
the built-in IPU7 devices. Pass the full stable path as `--source`. Verify the
saved image is color and is from the intended camera. Capture must run on the
computer physically connected to the camera; SSH does not forward a webcam.
The OpenCV capture command uses the video device, not the RealSense depth SDK.
Depth is not needed for this appearance baseline. OpenCV device support and
permissions still need verification on the actual station.

## 2. Capture one image and review it

Place one brick on a plain background, centered with space around its edges.
Keep the camera, lighting, distance and brick face consistent.

```bash
uv run python -m brickcell.sources.camera --source 0 \
  --label good --session train-a --object-id brick-001 --view front
```

The command prints the saved PNG path and quality checks, and saves a JSON
sidecar with source, timestamp, object/session IDs and crop. Open the PNG to
check focus, color and framing. No model runs unless `--model` is supplied.
Quality heuristics check blur, contrast and foreground position; they do not
prove a brick is present. Failed checks remain recorded for manual review.

Optional `--crop X0 Y0 X1 Y1` crops in camera pixels before saving or inference.
Choose it now and use the same crop for every training/evaluation/live capture.
Do not reuse the simulator crop; it belongs to another camera.

`--label` is your annotation for organizing photos, not a prediction. Repeat
the command for different bricks; it generates unique filenames. Keep a
physical brick's ID even if you later mark it. Do not treat many adjacent
frames of one brick as independent objects.

## 3. Collect independent groups, then train

Reserve distinct physical bricks and capture sessions for fitting, calibration,
and final testing. The splitter needs at least three independent all-good
object/session groups and two groups containing defects; these are structural
minimums, not sufficient accuracy evidence. Use more examples as described in
`DATA_COLLECTION.md`. Avoid using one session ID across every object: that
correctly makes them one indivisible leakage group.

Capture clean objects with `--label good`; marked objects with
`--label defective`. Remove unusable captures from the training collection
after manual review. Keep quality-rejection examples separately.

```bash
uv run python -m brickcell import-data --directory data/raw --output data/real-import.json
uv run python -m brickcell split-data --manifest data/real-import.json --output data/real-manifest.json
uv run python -m brickcell validate-data --manifest data/real-manifest.json
uv run python -m brickcell train --config configs/patchcore-real.yaml
uv run python -m brickcell export --config configs/patchcore-real.yaml
uv run python -m brickcell parity --config configs/patchcore-real.yaml
uv run python -m brickcell evaluate --config configs/patchcore-real.yaml
```

Tune model/threshold/quality settings on validation only, then freeze them
before evaluating test. New experiments need a new output directory; do not
overwrite a calibrated model or reshuffle an existing test set.

## 4. Inspect a new real capture

```bash
uv run python -m brickcell.sources.camera --source 0 \
  --label defective --session live-check-a --object-id brick-101 --view front \
  --output data/live-check \
  --model artifacts/models/patchcore-real-v1/model.xml
```

Use the same crop and quality configuration as collection. Output includes the
score, decision, model identity and heatmap path. The supplied human label is
not passed to the detector. `GOOD`, `DEFECTIVE`, `REINSPECT` and `ERROR` are
possible. Do not force an invalid view into a good/defective decision. Keep
live-check images outside the frozen training/validation/test collection.

First compare clean and marked captures, then challenge it with moved bricks,
shadows and smaller marks. Record misses and false rejects. Only after real
image decisions are useful should the camera source be connected to a robot
controller. This command never commands an arm.
