# First station data collection session

Objective: leave the station with a small, reviewed set of inspection images and synchronized demonstrations. Check the first recordings before spending the session collecting more. The workbench is a reviewer, not a robot-control application.

## Assign owners before collecting

| Owner | Deliverable |
|---|---|
| Robot operator | Confirm station calibration, safe reachable presentations, camera placement and which process owns the arm; record supervised leader/follower episodes |
| Inspection owner | Measure brick families, assign physical IDs, define acceptable versus defective marker/contamination appearances, label images and faces |
| Data/software owner | Pin the actual recorder version, record camera IDs/formats/timestamps and action conventions, validate/import the first batch |
| Evaluation owner | Reserve independent objects/sessions/episodes; keep final test data out of threshold/model selection; record pass/fail reasons |

One person can cover several roles. Tell the robotics teammate that this branch leaves `arm-api/`, calibration and hardware-control scripts untouched. Coordinate recording ownership: armd and LeRobot cannot own the same serial port simultaneously. Use the team's existing hardware procedures; no connection/movement commands are supplied by the workbench.

## Before the first recording

1. Identify the actual cameras and their roles. Record stable device IDs, RGB/BGR convention, resolution, FPS, exposure, lighting, mounting and timestamp clock. Avoid automatic exposure changes during an inspection sequence where feasible; record any changes.
2. Measure one square/long brick family in millimetres; the earlier 20–40 mm estimate is provisional. Start with separated objects of one family before expanding shapes/colors.
3. Assign stable physical object IDs. Define front/right/back/left relative to the brick and studded top; photograph a face-reference guide. Bottom is excluded. Do not mark a normal inspected surface merely to add an ID.
4. Agree marker-mark/contamination examples, including small marks, edges and low contrast. Define whether cosmetic seams and existing scratches count. Keep good examples across the same normal color/lighting/pose variation.
5. Assign disjoint physical objects and sessions to train, calibration (`val`) and final test. Preserve the same object ID if it is later marked. Keep camera/setup metadata alongside the dataset.
6. Confirm actual teleoperation/recording versions with the operator. Repo station documentation mentions LeRobot 0.7 while the isolated ACT experiment uses 0.6. Select an explicit supported recording/training path; do not mix environments or assume raw files are interchangeable.

## Inspection pilot

Capture one complete five-face sequence on a good object and one on a deliberately marked object. These are pilot collection checks, not enough data for model performance or the partition requirements.

For every face: hold still, verify focus, ensure the full intended surface is visible and the gripper is not hiding a possible mark, capture a fresh image, and enter one CSV row using the workbench template. Record `object_id`, family, color, session, face, face label, defect type and `synthetic=false`. Include camera ID and raw capture timestamp in the retained capture log; the current image CSV does not substitute for that sensor log.

If fingers cover a side, stop claiming that face is inspected. Plan a reachable re-grasp or fixture with the operator. Repeating a camera frame or relabeling one image as another face is not coverage. Capture quality and geometry need human review even when all five labels exist.

Import the CSV and open inspection review using `WORKBENCH.md`. Before continuing, check:

- The images load and colors match the object; IDs/labels/faces match the physical capture log.
- All five intended surfaces appear, sharp and unobscured; bottom is not accidentally named top.
- Duplicate frames are rejected and train/val/test object/session groups remain independent.
- Missing classes/partitions are explained as incomplete collection, not fixed by moving held-out frames into training.

After this pilot, expand independent normal objects/sessions plus labeled good/defective calibration and test groups. Train PatchCore only on normal images. Calibrate thresholds on val, freeze them, then evaluate final test objects. Report object-level five-face outcomes separately from frame-level metrics.

## Demonstration pilot

Record 3–5 short supervised episodes first: reach, grasp, lift, present/hold at inspection, then place/release where appropriate for the agreed policy task. Vary initial positions modestly within the operator's verified workspace. Record failure/abort episodes with honest labels, separately from successful-teacher training. Do not record a collision as a success because the last joint command completed.

Retain the native recorder dataset, camera timestamps, frame references, observed joint state, commanded action, gripper convention, episode/task boundaries, object/session IDs, calibration/version references and operator success notes. State the action timestamp convention explicitly. The current NPZ reviewer expects overview/wrist frames at 128×128 RGB, six radian state/action coordinates, and uniform timestamps; real streams will need a reviewed adapter that preserves provenance. Do not silently fabricate missing frames or convert gripper percentage to a joint angle without a measured mapping.

Review the first episode at start/middle/end and near grasp/release:

- Both camera frames correspond to the same moment and to the plotted joint state/action.
- Joint names/order, direction, units and timing match the recorder's documented semantics.
- Gaps, repeated frames, timestamp jumps, out-of-limit values and failed grasps remain visible.
- Measured physical limits correspond to that arm/calibration/coordinate system. The simulation limit profile must not be reused.
- No complete episode, physical object or capture session is split between training and held-out evaluation.

**Stop and correct the recording pipeline if these checks fail.** The existing synthetic teacher recordings already demonstrate why: scripted completion coexists with recorded wrist-flex limit violations. More training cannot repair a mis-specified action dataset.

Once the pilot is accepted, expand reviewed demonstrations, preserving fresh evaluation episodes/positions. Run a small CPU/GPU training pilot and bounded closed-loop evaluation before a larger cloud job. Choose a provider, pinned GPU environment, dataset/checkpoint storage and explicit runtime/spend cap with the team. No cloud access is needed for this workbench.

## What is not complete yet

Real-camera adapters, native station-dataset conversion, five-face robot presentations/re-grasps, sensor-backed grasp/release verification, robust learned movement, hardware integration and Intel CPU/GPU/NPU measurements remain separate milestones. A readiness report is a data-quality gate, not permission or evidence for autonomous hardware operation.
