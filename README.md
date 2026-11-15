# Brick anomaly data: positive (clean) captures

Real captures from the plate D415 camera (serial 243622060450) on the Intel cell, 2026-11-15.
The SO-101 follower holds one DUPLO brick and turns it with `wrist_roll`. The app that made
these is `scripts/wrist_rotation_capture.py` on the `anomaly` branch.

**positive = a clean brick with no anomaly.** All 570 frames are positive. There are no defect frames.

| object_id | Brick | Grips | Frames |
|---|---|---|---|
| `white-2x4_3011` | white 2x4 (3011) | grip-001 … 006 | 180 |
| `red-2x2_3437` | red 2x2 (3437) | grip-007 … 010 | 120 |
| `yellow-1x2x2_76371` | yellow 1x2x2 (76371) | grip-011 … 016 | 180 |
| `lavender-2x2_3437` | lavender 2x2 (3437) | grip-017 … 019 | 90 |

## Layout

```
positive/index.jsonl                      one line per frame (labels, wrist angle, paths)
positive/<object_id>/<grip_id>/run.json   settings, grip result, per-frame joint and camera data
positive/<object_id>/<grip_id>/frame_NNN.jpg   640x480 RGB
positive/<object_id>/<grip_id>/depth_NNN.png   aligned 16-bit depth (see depth_scale_m in run.json)
```

Each grip is one hand placement of the brick, then a sweep of 30 frames, `wrist_roll` 2365 → 4040 ticks (about 147°).

## Limits

- The brick labels were set after the capture, from the images. The app recorded every run as red 2x2 (the form default). `capture_object_id` in each `run.json` keeps the original id.
- There is one physical brick of each type, from one session. For train/val/test, split by `grip_id`. You cannot test on a brick that the model did not see.
- In some frames a hand or other bricks show in the background (for example, the last frames of grip-001 and grip-017).
