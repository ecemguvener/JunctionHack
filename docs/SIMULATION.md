# Two-bowl camera-driven SO-101 simulation

The working loop is **rendered camera → Anomalib 2.7.0 PatchCore → OpenVINO CPU → inspection supervisor → simulated SO-101**. It runs locally on the Apple M4 Max. The browser is a viewer; separate local Python processes run physics and inference. No hardware, remote servers or credentials are used.

## Run with the existing local assets

Terminal 1, from the repository root:

```bash
PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" \
  artifacts/mujoco-venv/bin/python -m brickcell.simulation.server
```

Open **http://127.0.0.1:19091/**. Click **Run / replay full inspection loop**, or use Terminal 2:

```bash
uv run python -m brickcell.simulation.run --reset
```

The arm picks four LEGO DUPLO bricks, presents each at the inspection camera, captures a fresh frame, executes OpenVINO inference, moves to the selected plate, opens its jaws, and checks simulated plate occupancy and settling. Green plate = GOOD; pink plate = DEFECTIVE. (The API and file names still say "bowl".) The panel shows the last inspection image, object ID, raw score, model version, inference latency, action and counts. This is an inspection snapshot, not continuous video. Drag the scene to orbit; scroll to zoom. A run takes roughly 70 seconds at visible real-time speed.

New ports: **19091 viewer / 18081 API**. The earlier conveyor at 19090 is a different application. Stop this server with Ctrl+C. Both services bind only to loopback.

## Reproduce from a fresh clone

Keep simulator dependencies separate from Anomalib:

```bash
uv sync --locked
uv venv artifacts/mujoco-venv --python 3.12
uv pip sync --python artifacts/mujoco-venv/bin/python configs/mujoco-requirements.txt

PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" \
  artifacts/mujoco-venv/bin/python -m brickcell.simulation.generate --output data/two-bowl
uv run python -m brickcell validate-data --manifest data/two-bowl/manifest.json
uv run python -m brickcell train --config configs/patchcore-sim.yaml
uv run python -m brickcell export --config configs/patchcore-sim.yaml
uv run python -m brickcell parity --config configs/patchcore-sim.yaml
uv run python -m brickcell evaluate --config configs/patchcore-sim.yaml
```

First use downloads the official Menagerie robot and official pretrained ResNet-18 weights; later runs use caches. Generation refuses to overwrite its output directory; training refuses to overwrite checkpoints. The separate requirements snapshot pins plugin 0.5.0, Physical AI 0.4.0, MuJoCo 4.0.1, Menagerie 2026.9.4, mjviser 0.0.17 and viser 1.1.1. Plugin internals (`compose_scene`, `place_home`, `ArmIK`) are version-sensitive; retest after upgrades.

Use `artifacts/models/patchcore-sim/` for this camera. The earlier `artifacts/models/patchcore/` was trained on drawings and is not suitable here. No MuJoCo dependency was added to the core environment; its bridge uses standard-library HTTP.

## Objects, plates and physics

The four objects are LEGO DUPLO parts **3437** (2×2), **3011** (2×4), **76371** (1×2×2) and **2302** (2×3 curved top). The meshes come from the LDraw library (CC BY 4.0); `assets/duplo/README.md` gives the source and the converter. The jaws close across the 32 mm side of every part; the other side faces the inspection camera.

The sorting targets are IKEA KALAS plates (804.613.80, Ø186 × 23.5 mm). The plate shape is generated in `world.py` from a measured cross-section, so no IKEA file is in the repository. Collision uses a floor disc and a ring of sloped wall boxes. The arm puts up to four bricks on one plate in fixed slots (two rows, diagonal first) and releases each brick 6 mm above the plate floor.

All physical values are in one block, `PHYSICS` in `src/brickcell/simulation/world.py`: gravity, timestep and solver, friction per contact pair, contact softness and brick masses. **They are estimates, not measurements.** Friction values are the middle of dry ranges in engineering tables (ABS/ABS 0.35, ABS/polypropylene plate 0.30, ABS/table 0.40, ABS/PLA jaw 0.40). Masses come from part volume and ABS density. To calibrate, weigh each brick on a kitchen scale, and do a tilt test for each pair: tilt the surface slowly until the brick slips; μ = tan(slip angle). Enter the values in `PHYSICS`.

**The detector must be retrained.** The models in `artifacts/models/patchcore-sim/` were trained on the earlier orange 32 mm cubes. Generate new data and repeat the training steps above before you use the live loop.

## Cameras and visible defects

There are **three named virtual cameras**: `inspection`, `overview`, and `wrist`. Only **inspection** feeds the detector. The freely orbiting browser view is a spectator view, not an inference input. Wrist/overview cameras exist in the model but are not streamed into a policy or detector here.

The inspection camera renders 256×256 RGB. Both training and live capture take the fixed crop `[48,90,208,220]`, producing 160×130 RGB. Existing detector preprocessing resizes to 128×128, scales to [0,1], and applies embedded normalization once. No simulator label, object name, pose, or defect metadata enters the detector.

The synthetic defect is a dark 14 × 10 mm rectangular mark on the camera-facing side. The jaws close across the other axis to keep that face visible. **This does not inspect all six faces**, hidden cracks, undersides, reflections or arbitrary defects. Reinspection offsets the same view slightly; it does not re-grasp or guarantee coverage.

The dataset has 24 good training images, 12 calibration images (6 good / 6 defective), and 12 held-out rendered test images (6 / 6). Each image follows an actual simulated contact grasp. Camera position and brick color vary slightly using a seeded generator. Sessions and IDs are partitioned; cloned geometry is shared across synthetic fixtures. These tests do not measure real-world or new-object generalization. Only `val` calibrates the threshold.

The camera quality config checks contrast, blur and foreground fraction. Since jaws appear in the crop, the uniform-background ROI heuristic uses the entire crop. It cannot establish presence by itself. The simulator separately verifies lift and inspection pose through physics state; hardware must replace those checks with observations/sensors.

## What is real software and what is simulated?

| Component | Implementation |
|---|---|
| Defect inference | Actual Anomalib-trained PatchCore exported to OpenVINO CPU |
| Sort decision | Existing calibrated threshold and supervisor |
| Camera image | MuJoCo RGB render with fixed training/live crop |
| Arm motion | Scripted waypoints and IK using known pickup poses |
| Grasp/carry/release | MuJoCo friction/contact; no weld or carried-object teleport |
| Success check | Simulated object height, plate position and low velocity |
| Reset | Explicitly restores four fixture bricks between runs |
| Learned VLA/ACT and vision-based picking | Not implemented by this simulation |

Fixture labels control appearance only and are absent from camera/action responses. The server refuses plate movement without a valid published result for the current object. The supervisor checks object/frame/view/timestamp identity, bounds retries and limits action/inference waits. These are local software interlocks, not an authenticated or safety-certified hardware controller.

This proves a replacement for `MockRobot` can use the existing detector and supervisor. It does not prove real trajectories, grip force, joint-unit compatibility, or learned-policy reliability.

## Test and inspect reports

```bash
uv run pytest

# Start the simulator first, then execute actual model and live physics tests:
BRICKCELL_MODEL_CONFIG=configs/patchcore-sim.yaml \
BRICKCELL_QUALITY_CONFIG=configs/inspection-sim.yaml \
BRICKCELL_SIMULATION_URL=http://127.0.0.1:18081 \
  uv run pytest -m 'model or simulation'

curl --fail http://127.0.0.1:18081/health
```

Offline tests cover loopback restrictions, RGB/path handling, action tokens, stale frames, invalid/uncertain decisions and destination authorization. A real-model regression retains an identical opaque object ID and changes only the camera pixels; the commanded plate changes. The opt-in live integration executes four full contact-physics cycles.

CLI runs save `events.jsonl`, camera PNGs, anomaly overlays and `summary.json` under unique directories in `artifacts/two-bowl/runs/`; `latest.json` holds the latest report. Browser replay output is `artifacts/two-bowl/viewer-replay.log`. Integration tests write under `artifacts/two-bowl/tests/`. Model provenance, calibration, export, parity and final test metrics are under `artifacts/models/patchcore-sim/`.

Next: collect real camera images and teleoperation demonstrations at the station, calibrate the arm, then replace scripted motion with a hardware driver/policy behind the same boundary. Simulate failures and handoffs; do not substitute synthetic accuracy for real-data validation. See `INTEL_DEPLOYMENT.md`.

## Inspect the 3D models

Use this viewer to compare the DUPLO bricks and the KALAS plate in the simulation with the real objects. It does not load the robot.

```bash
MUJOCO_GL=egl PYTHONPATH=src \
  artifacts/mujoco-venv/bin/python -m brickcell.simulation.model_viewer --port 19092
```

Open **http://127.0.0.1:19092/**. The server binds only to loopback. From another computer, use an SSH tunnel (`ssh -L 19092:127.0.0.1:19092 <host>`).

- The models are at true scale. A grid cell is 1 cm. The bright grid lines are 10 cm apart.
- Click a model to select it. Drag the gizmo arrows, planes, or rings to move and turn it. **Reset layout** puts all models back.
- **Selected model** shows the position, the rotation, and the top height above the table. The buttons put the model on the table, make it upright, turn it 90°, or tip it 90°.
- **Collision shapes** shows what the physics touches: the convex hull of each brick and the floor disc and wall boxes of the plate.
- **Measure** mode: click two points on a model or on the table. The panel shows the distance in mm. With snap on, a click near a mesh corner uses the corner.
- **Physics drop test** drops the four bricks into the plate with MuJoCo and plays the result back. The panel shows where each brick stopped.
- **Sizes** shows the bounding box of each mesh (measured at start-up), the part number, and the mass in the sim.

The values come from `world.py`. If you change a brick or the plate there, restart the viewer.
