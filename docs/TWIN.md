# Station twin: real cameras → MuJoCo

The station twin makes the MuJoCo simulation copy the real station. Two D415 RGB-D cameras
("Top" = sim camera `overhead`, "Plate" = sim camera `side`) find each DUPLO brick and each
KALAS plate. The arm joint angles come from the follower servos (read only). The sim puts its
bricks, plates and arm at the same places, on request or live.

```
Intel laptop                                    sim computer (or the same laptop)
D415 x2 ─▶ station (port 8095) ─ GET /state ─▶ sim server /twin/sync, /twin/live ─▶ MuJoCo
           realsense → perception → StationState JSON           sim_sync.apply_state
```

## Parts

| Module | Job |
|---|---|
| `src/brickcell/twin/state.py` | Shared contract: `StationState`, `BrickState`, `PlateState`, `ArmState`, frames and units, brick copies. numpy only. |
| `src/brickcell/twin/realsense.py`, `station.py` | Owns both D415s. A snapshot = median depth of 10 frames (no hole filling). Runs perception. HTTP service. |
| `src/brickcell/twin/perception.py`, `geometry.py` | RGB-D → brick part, stable pose, 6-DoF pose, colour name, support; plate centres. numpy + OpenCV. |
| `src/brickcell/twin/arm.py` | Servo ticks → MuJoCo joint angles (CAD → MuJoCo map checked by forward kinematics: 1.0 mm mean TCP difference). |
| `src/brickcell/twin/sim_sync.py` | StationState → MuJoCo World: plates, bricks, arm, no-overlap guard, relax, settle, report. |
| `src/brickcell/twin/calibrate.py` | Refines the D415 poses in `configs/sim-scene.yaml` from the table plane and the board corners. |
| `src/brickcell/twin/synthetic.py` | MuJoCo RGB-D renders with known poses, for perception tests. |
| `src/brickcell/palette.py`, `configs/brick-palette.yaml` | 18 DUPLO colours (12 main + 6). Perception names colours; the sim and the training data use them. |

Config: `configs/twin.yaml` (sections `station`, `perception`, `sim`, `calibration`). Camera
poses and intrinsics are only in `configs/sim-scene.yaml`.

## Frames

World = robot base frame (origin at the SO-101 base on the table, +Z up). A brick pose is the
pose of its mesh frame (`assets/duplo/<mesh>.stl`, origin at the bottom centre). The MuJoCo body
origin of a brick is its grasp point; `body_pose_from_mesh_pose` converts.

## Run it

1. Copy the code to the Intel laptop: `scripts/deploy_twin.sh`.
2. On Intel, stop any other process that holds the D415s (for example the live-depth viewer on
   port 8094). Then start the station (the deploy script prints the full command):

   ```bash
   cd /home/intel/brickcell-twin && PYTHONPATH=src /home/intel/.local/bin/uv run --offline \
     --no-project --python 3.12 --with pyrealsense2 --with numpy --with opencv-python-headless \
     --with pyyaml --with scipy python -m brickcell.twin.station --arm-source none
   ```

   Without hardware, play back a recorded fixture: `--fixture artifacts/twin-fixtures/fixture-002`.
3. Start the sim server with the station URL (use an SSH tunnel if the sim runs on another
   computer: `ssh -N -L 8095:127.0.0.1:8095 intel`):

   ```bash
   MUJOCO_GL=egl PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" \
     artifacts/mujoco-venv/bin/python -m brickcell.simulation.server --station-url http://127.0.0.1:8095
   ```

4. In the viewer (http://127.0.0.1:19091/), open **Station twin** and click **Sync from cameras**,
   or set **Live**. The same with the API:

   ```bash
   curl -X POST -d '{"fresh": true}' http://127.0.0.1:18081/twin/sync
   curl -X POST -d '{"enabled": true, "hz": 1}' http://127.0.0.1:18081/twin/live
   curl http://127.0.0.1:18081/twin/status
   ```

All services bind to loopback only.

## Station API (port 8095)

| Request | Result |
|---|---|
| `GET /state[?fresh=1]` | Latest StationState; `fresh=1` takes a new snapshot. |
| `GET /frame/<camera>.jpg`, `/depth/<camera>.png`, `/overlay/<camera>.jpg` | Colour, depth, and the result drawn on the colour image. |
| `POST /live {"enabled": true, "hz": 1}` | Snapshots in a loop (at most 2 Hz). |
| `POST /record {"name": "fixture-003"}` | Saves the raw frames as a fixture (format of `artifacts/twin-fixtures/fixture-001`). |
| `GET /health` | Cameras, intrinsics check, timings, last error. |

## What the sim does with a state

1. Moves the plates (`good`, `defective`, `spare`). An absent spare plate goes to its park spot.
2. Puts each measured brick on its sim body. Each part has 4 bodies (`part-002`, `part-002-2`, …);
   unused bodies park out of view of every camera.
3. Sets the arm joints (if the state has them) and holds them.
4. **No-overlap guard:** moves overlapping bricks apart in x, y with 0.5 mm clearance, before any
   physics, so contact and friction between bricks cannot push them.
5. Relax (zero velocity) and settle (normal physics). The report gives per brick: overlap,
   separation, drift, gap, colour and flags.

Brick colours come from the palette (`sim.color_mode: palette`). The pick-and-sort demo uses only
`part-001` … `part-004`, the good and the defective plate; its camera images at reset do not change.

## Calibration

```bash
.venv/bin/python -m brickcell.twin.calibrate --fixture artifacts/twin-fixtures/fixture-001 \
  --test-fixture artifacts/twin-fixtures/fixture-002 [--board-image overhead=...] [--write]
```

It levels each camera on the table plane and fits the side camera to the overhead camera from
the board corners. Result on 2026-11-14: the cameras agree to 1.3 mm (median) on the board
corners; before, they were 55 mm apart. The pose of the cameras relative to the robot base
(x, y, yaw) is from the earlier SO-101 model fit and is not measured again. To measure it, touch
the board labels A–D with the closed gripper tip and record a fixture and the joint angles at
each label.

## Offline check

```bash
MUJOCO_GL=egl PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" \
  artifacts/mujoco-venv/bin/python -m brickcell.twin.apply --state state.json --out out/ \
  --real-fixture artifacts/twin-fixtures/fixture-002
```

It writes the sim camera images and sim/real blends.

## Limits

- Perception accuracy on synthetic renders: 100% of found bricks within 5 mm / 5° (with D415
  noise). On real data, a brick that touches another brick can be missed or fitted to the wrong
  part. The overhead camera is about 0.9 m from the table; small bricks look too low in its depth.
- The side camera gives depth for about half of its image.
- Live arm reads are written but not tested on hardware. The gripper angle map is an assumption.
- Perception: 0.3–0.45 s for one `locate` call; the station keeps one `PerceptionTracker`, so a
  warm live update takes 0.13–0.18 s (both D415 snapshots then add about 0.4 s of capture).
