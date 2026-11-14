# Team handoff and next milestone

## What is ready to share

Share the code as a feature branch / draft PR in `ecemguvener/JunctionHack`. A hosted website is unnecessary: the viewer, inference and simulator run locally. The project has Python packaging, a core lockfile, a separate policy lockfile, pinned simulator requirements, tests, reproducible commands and validation reports. It is a software prototype, not a production hardware release.

The branch should contain source, configs, tests, documentation and locks. Generated datasets, checkpoints, reports, caches and environments stay Git-ignored. The package's source archive uses an explicit allowlist and excludes the pre-existing unrelated `ssh.md`. The repository's existing history is not rewritten by this work. If team model/data sharing is needed later, use approved artifact storage with checksums and dataset access controls; do not commit the local caches or credentials.

The optional wheel installs the Python package. Use the repository checkout for configs, documentation and policy scripts. A fresh teammate should first follow README installation, run `make test`, then reproduce the simulator/model assets using `docs/SIMULATION.md`. A screenshot or video demonstrates appearance; the event logs and tests provide the verification evidence.

## Current readiness

| Workstream | Status |
|---|---|
| Dataset validation, Anomalib training/export/CPU inference | Working and tested with synthetic data |
| Camera-to-detector-to-two-bowl simulated loop | Working; scripted contact-physics motion |
| Five-face result aggregation | Implemented and tested; not yet connected to robot presentations |
| Five-face visibility/re-grasp sequence | Not implemented or verified |
| Synthetic demonstration recording | Working, synchronized, hashed, episode-separated |
| Official ACT training/checkpoint reload | Small CPU smoke test passed |
| Learned pick/present controller | Experimental: smoke policy 0/2 rollout successes |
| Real object inspection data and human demonstrations | Not yet collected |
| Real cameras, SO-101 driver and Intel deployment | Not yet exercised |
| Cloud GPU jobs / hosted deployment | Not needed for sharing; none launched |

## Why two datasets?

Inspection learns **appearance**: images of normal objects, plus labeled defects for threshold calibration and evaluation. Policy learning learns **actions over time**: synchronized images, robot joint state, commanded actions and episode/task boundaries. An isolated defect photo contains no demonstration of how to move; a successful pick trajectory does not establish whether a mark is an acceptable appearance.

These are two logical datasets, not necessarily two separate recording sessions. We can extract held inspection frames from teleoperation episodes, then add object/family/color/face/defect labels. Preserve shared physical-object/session grouping so derived frames do not leak between partitions. Keep RGB conventions, camera IDs, timestamps and joint units explicit.

## Human/team tasks now

| Owner | Concrete task |
|---|---|
| Objects/inspection teammate | Inventory square versus long/narrow bricks; measure each family's dimensions; assign physical IDs; define marker/contamination defect examples. Confirm that the studded face is top and bottom is excluded. |
| Robotics teammate/operator | Get the station's camera and leader/follower setup instructions; plan reachable presentations and how to expose finger-covered sides. Once hardware arrives, calibrate and record synchronized demonstrations under supervision. |
| ML/software teammate | Review camera framing and recorded episodes; expand data and training; evaluate learned rollouts; keep action-limit failures visible. Fit real-image anomaly models after collection. |
| Integration/evaluation teammate | Reproduce the branch on another machine, run tests, define independent test objects/positions and track full-cycle success, misses, false rejects, reinspection and latency. |

One person can cover several roles. Before camera/robot access, ordinary photos help inventory and label design; they should not be treated as a ready station-camera training set. Exact dimensions remain needed: “2–4 cm” is sufficient for a provisional fixture, not final grasp calibration.

## Recommended order

1. Share the working prototype with its experimental limitations and get another teammate to reproduce it.
2. Keep the two-bowl demonstration as the regression baseline. Use simulation for data recording, policy handoffs, coverage logic and failure handling.
3. Establish the five-face physical presentation plan and capture contract. A wrist turn alone cannot expose a surface hidden by the fingers; re-grasp/fixture handling may be needed.
4. Improve the ACT experiment using reviewed demonstrations and bounded evaluation. Do not promote the 50-step smoke checkpoint.
5. As soon as the station is available, prioritize real camera data and real leader/follower recordings over adding visual detail to simulation.
6. Transfer the code to the Intel laptop, run doctor/tests on CPU, implement real camera and robot adapters, and validate one physical pick–present–inspect–sort cycle.
7. Expand shape families, five-face coverage and GPU/NPU optimization only after the baseline is reliable.

Our next practical hardware milestone is one reliably inspected and sorted real brick. Full five-face acceptance requires all five face observations, calibrated models and verified visibility; the current single-view demo cannot claim that requirement.

## Cloud GPU decision

No GPU needs provisioning now. The small ACT smoke training ran on the M4. Before a paid/cloud run, prepare a reviewed dataset, reproducible GPU-specific environment, selected ACT/SmolVLA recipe, evaluation seeds, checkpoint storage, and a short runtime/spend-capped pilot. Use the organizers' supported instance and training recipe when available. The existing CPU policy lock is not a CUDA environment. Credentials belong in the provider's approved secrets flow, never in the repository or chat attachments.
