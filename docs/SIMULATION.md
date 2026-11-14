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

## Scene config: cameras and colours

`configs/sim-scene.yaml` sets the camera poses, the camera intrinsics, the brick colours and the plate colours. You do not edit code. The file has a comment for each field. The simulation loads it at start-up; restart the program after a change. To use a different file, give `--scene-config PATH` (`server`, `model_viewer`, `generate`) or set `BRICKCELL_SCENE_CONFIG=PATH` (all programs). The parser is `src/brickcell/simulation/scene_config.py`; it has no MuJoCo import, so `tests/test_scene_config.py` runs in the core environment.

**Frames and units.** Lengths are in metres and `fovy` is in degrees. The world frame is the robot base frame: the origin is at the SO-101 base on the table, +Z is up, and +X points from the base to the bricks. Pixels use the OpenCV convention: the centre of the top-left pixel is (0, 0), u goes right, v goes down.

**Cameras.** Each entry under `cameras` has these fields:

| Field | Meaning |
|---|---|
| `pos` | `[x, y, z]` of the camera in the world frame (or in the `attach` body frame). |
| `lookat` (+ `up`) | Point that the camera looks at. `up` is the direction of the image top (default `[0, 0, 1]`). |
| `xyaxes` | MuJoCo convention: camera X axis (image right), then Y axis (image up). |
| `quat` | `[w, x, y, z]`, MuJoCo camera frame: +X right, +Y up, the camera looks along −Z. |
| `rotation_cv` | 3×3 camera-to-world rotation, OpenCV camera frame: +X right, +Y down, the camera looks along +Z. |
| `fovy` | Vertical field of view. The principal point is the image centre. |
| `fx`, `fy`, `cx`, `cy` | Pinhole intrinsics in pixels for a `width` × `height` image. Use them as an alternative to `fovy`. |
| `width`, `height` | Render size in pixels (default 256 × 256). |
| `attach` | Name of a model body. The camera moves with that body; `pos` and the orientation are in its frame. |
| `enabled` | `false` keeps the camera out of the model. |

Give exactly one of `lookat`, `xyaxes`, `quat`, `rotation_cv`, and give `fovy` or the intrinsics. The loader stops with a clear message if a value is wrong.

- OpenCV to MuJoCo: `R_mujoco = R_cv · diag(1, −1, −1)`. If you have OpenCV extrinsics (`x_cam = R·x_world + t`), use `rotation_cv = Rᵀ` and `pos = −Rᵀ·t` (`scene_config.pose_from_extrinsics`).
- Intrinsics go to the MuJoCo camera attributes `resolution`, `sensorsize`, `focalpixel` and `principalpixel`, so `fx ≠ fy` and an off-centre principal point are correct in the render: `principalpixel = ((W−1)/2 − cx, (H−1)/2 − cy)`, `fovy = 2·atan(H / (2·fy))`. A render check with probe spheres gave a mean error of 0.1 px (maximum 0.24 px). There is no lens distortion: undistort the real images, or accept the difference.
- `inspection` and `overview` are the built-in cameras. `inspection` must stay enabled and 256 × 256, because the detector crop depends on it. If you move it, generate new data and train the detector again.
- `overhead` (real camera "Top") and `side` (real camera "Plate") have measured poses, refined by `python -m brickcell.twin.calibrate` (see `TWIN.md`).
- `wrist` (real camera "Gripper") comes from the SO-101 robot profile: body `camera_mount`, `pos [0, −0.055, −0.045]`, `quat [0, 0, −0.281157, 0.959662]`, `fovy 75`. To replace it, add a `wrist` entry with `attach: camera_mount`; the file has an example in a comment. For a camera with `attach`, the simulation compiles the scene a second time from the plugin's `MjSpec` (`world.attach_cameras`).

`World.camera(name)` renders each camera at its configured size. `World.cameras()` gives the pose and the pinhole intrinsics of each camera.

**Colours.** `bricks` has a `color` for each part id (`part-001` … `part-004`), and `plates` has one for `good` and `defective`. A colour is `"#rrggbb"` or `[r, g, b]` with numbers from 0 to 1. `generate --palette off` uses the configured brick colour as the centre of its colour jitter. The default `--palette all` gives each brick a colour from `configs/brick-palette.yaml` (18 DUPLO colours); see `MODAL_TRAINING.md`. For one run, `--brick-color part-001=#ff0000` on the server changes a colour without a file change (you can give it more than once). **The detector learns the brick colours:** after a permanent colour change, generate new data and train again.

**Server API** (loopback only, default port 18081):

| Request | Result |
|---|---|
| `GET /scene` | The cameras (pose, intrinsics, state) and the brick and plate colours. |
| `GET /camera/<name>.png` | The full image of a simulation camera at its configured size. For viewing only: it has no frame identity, so the supervisor cannot use it for a sort decision. `/capture` is not changed. |
| `POST /brick_color` with `{"object_id": "part-001", "color": "#ff0000"}` | Changes the colour in the running model and in the browser viewer. The change is for this run only. |

```bash
curl http://127.0.0.1:18081/scene
curl -o overview.png http://127.0.0.1:18081/camera/overview.png
curl -X POST -d '{"object_id": "part-001", "color": "#ff0000"}' http://127.0.0.1:18081/brick_color
```

**Browser viewer.** The panel has a **Sim camera** folder (select a camera to see its live image, about 5 images per second, and compare it with the real stream; the world-fixed cameras show as frustums in the 3D view) and a **Brick colours** folder (one colour picker for each brick). The model viewer (`model_viewer`) has a colour picker for the selected model and a **Station layout** button: it shows the configured cameras as frustums with the arm base, the plates and the bricks at their simulation positions. Start it with `--station` to open in that layout.

## Cameras and visible defects

There are **three named virtual cameras** by default: `inspection`, `overview`, and `wrist`; the scene config can add more. Only **inspection** feeds the detector. The freely orbiting browser view is a spectator view, not an inference input. Wrist/overview cameras exist in the model but are not streamed into a policy or detector here.

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

- **Colour** (in **Selected model**) changes the colour of the selected model in this viewer. The panel shows the line to copy into `configs/sim-scene.yaml`.
- **Station layout** shows the cameras from the scene config as frustums, with the arm base, the plate positions and the brick start positions. **Reset layout** goes back to the model check.

The values come from `world.py` and `configs/sim-scene.yaml`. If you change a brick, the plate, a camera or a colour there, restart the viewer.
