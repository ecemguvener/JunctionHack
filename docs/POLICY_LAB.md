# Policy preparation before physical hardware

This is an experimental path alongside the verified two-bowl Anomalib demo. It now records synchronized synthetic demonstrations, trains the **official Physical AI Studio ACT implementation**, reloads a checkpoint, and evaluates it in teacher-free MuJoCo rollouts. It does not replace the working scripted sorting controller.

## Executed result on the M4

- Six successful scripted-teacher pick-and-present episodes, **2,250 frames** at 50 Hz: four train episodes, one validation, one test. Each episode has a different seed/pickup position. Images are 128×128 RGB from overview and wrist; six observed joint angles and six commanded joint targets are in radians. State/image/action are captured before the corresponding physics step.
- Official `physicalai-train==0.3.0`, runtime 0.3.0, LeRobot 0.6.0, torch 2.10.0 and torchvision 0.25.0 installed natively in `policy/.venv`. Versions are fixed by `policy/uv.lock`. This is separate from both the Anomalib and MuJoCo environments.
- Small ACT configuration: ResNet-18 pretrained backbone, 128-dimensional transformer, two encoder layers, one decoder layer, chunk size 16, first four actions executed before replanning. This is ACT, not a language-conditioned VLA.
- **50 CPU optimization steps succeeded in 12.52 seconds**. Loss changed from 7.724 to 0.617. A seeded validation batch had action MAE 0.0710 radians. These are training diagnostics, not manipulation accuracy. Reloaded checkpoint predictions matched exactly.
- Strict rollout evaluation on seeds 5200/5201: **0/2 successes**, both rejected before movement for out-of-limit predictions. Explicitly enabled, logged simulator-limit saturation allowed 600 steps per rollout, but still **0/2 successes**, with 600 saturated action values each. No scripted IK ran during those policy steps. Do not deploy this smoke checkpoint to hardware or claim that it has learned reliable picking.
- Native imports emitted duplicate AVFoundation class warnings from PyAV/OpenCV. Training and the bounded rollouts completed despite them. No simulator or training dependency was installed into the core Anomalib environment.

The experiment establishes the complete learning/evaluation plumbing and shows why low training loss alone is insufficient. More reviewed demonstrations and training are needed before substituting learned movement. The source API requires tuple-shaped feature metadata; the wrapper was tested against the installed release.

## Reproduce

From the repository root, after installing the core and separate MuJoCo environment as described in `SIMULATION.md`:

```bash
uv sync --project policy --locked --python 3.12

PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" \
  artifacts/mujoco-venv/bin/python -m brickcell.simulation.record_policy \
  --output data/policy-smoke --episodes 6 --seed 4200

PYTHONPATH=src TORCH_HOME="$PWD/.cache/torch" \
  policy/.venv/bin/python policy/train_act.py

PYTHONPATH=src policy/.venv/bin/python policy/evaluate_act.py

# Optional diagnostic only; saturation is explicitly counted in the report:
PYTHONPATH=src policy/.venv/bin/python policy/evaluate_act.py \
  --clip-to-sim-limits --output artifacts/policy/act-smoke/rollouts-bounded.json
```

Recording and training refuse existing output directories. The local workspace already has the smoke assets; rerun evaluation or choose new output paths for a new experiment. `make policy-env`, `make policy-record`, `make policy-train`, and `make policy-evaluate` expose the same workflows.

The first ACT constructor may download the official torchvision ResNet-18 weights. After caching, the training/inference path is offline. Loading a saved checkpoint constructs with `pretrained_backbone_weights=None` and restores the exact saved weights.

## Dataset and checkpoint contracts

Each episode is a pickle-free NPZ containing timestamp, RGB images, observed joint state and commanded action. The JSON manifest records episode split, seed, units, joint order, task, source versions, success and SHA-256. Validation rejects duplicate episodes/seeds/hashes, changed archives, missing synchronization and invalid numeric/image shapes. Normalization statistics use training episodes only. Action chunks never cross episode boundaries; padded terminal actions are masked from the training loss.

These are **synthetic scripted-teacher demonstrations**, not human demonstrations. They are not presented as LeRobot-compatible files or as a Studio UI import. `policy/train_act.py` adapts them to the official `Observation` and eager ACT APIs, using a small PyTorch optimization loop. `policy.pt` is our local experiment bundle with model parameters, feature statistics and configuration—not a Studio Lightning checkpoint. A LeRobot conversion/Studio recording workflow remains to be integrated before team-scale policy training or UI imports. The policy's available export API alone does not make this unvalidated controller hardware-ready.

`policy/evaluate_act.py` communicates with an isolated MuJoCo worker over private stdin/stdout. After reset, the worker writes only predicted joint targets to the actuators and steps contact physics. It never calls the scripted IK teacher during action execution. Ground-truth simulator state is used only for the evaluation success check, not policy observations. Rollouts have bounded steps and response deadlines. Saturation is disabled by default and always reported when enabled.

## Five-face requirement and real object scope

`configs/coverage-five-face.yaml` requires front, right, back, left and top; bottom is outside the agreed scope. The top is the **studded object face**, irrespective of camera/world orientation. `CoverageLedger` accepts GOOD only when all required faces have distinct, identity-matched, valid calibrated results. A detected defect stays latched. Invalid/uncertain/missing views require reinspection; errors prevent automatic sorting.

This is a tested **aggregation contract**, not five-face capture. The current live demo still captures one face and does not invoke this ledger. Before enabling the five-face path, implement reachable presentations/re-grasps, verify that the fingers do not hide inspection regions, capture fresh face-tagged frames, and route each through a suitable calibrated model. One camera name does not prove a face was visible. Fixed multi-view raw scores should not simply be averaged; validate object-level decisions over the complete capture sequence.

The user's photo shows square and long/narrow bricks in multiple colors/orientations. Exact dimensions are unmeasured; the user estimates roughly 20–40 mm. `configs/objects.yaml` records this uncertainty and the initial marker-mark/surface-contamination defect scope. The simulated 32 mm cube is a single fixture, not a CAD reconstruction of those bricks. Record family/color/face metadata. Start with one family; expand after normal shape/color/view variation stops triggering false defects. Shape-specific or view-specific detector calibration may be necessary.

## What to do next

1. Review recorded camera frames and action alignment; choose a single task and measured brick family. Improve framing if the brick is too small in the policy camera.
2. Record a larger, diverse teacher dataset (for example 50 episodes), with explicit train/validation/test partitions by episode/seed. Currently the generator reserves just one episode each for validation/test; extend the split allocation for a meaningful larger experiment before using it to claim generalization.
3. Train longer and use **development** rollouts to diagnose failures. Keep fresh final evaluation seeds separate. Track grasp success, held presentation, limit saturation and cycle completion, not just training loss.
4. Only promote a learned controller after reliable closed-loop results. Preserve the scripted baseline for comparison and debugging.
5. On the physical station, collect actual leader/follower demonstrations and real inspection images. Synthetic motion does not replace hardware calibration or those demonstrations.

No cloud GPU is required for these smoke tests. Once the data recipe and evaluation are stable, use the organizers' supported GPU/Studio recipe for ACT or SmolVLA training. Prepare a reviewed dataset, pinned GPU environment, output/checkpoint storage, a short pilot with a spending/runtime cap, and approved account access. This task provisioned no cloud machine or paid job. Keep credentials outside Git. The current policy lock deliberately selects CPU Torch on Linux; do not assume it is a CUDA training environment.
