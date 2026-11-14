# Executed validation — 13 November 2026

All results below were produced locally in this repository. No remote Intel server, SSH login, real camera, physical arm, or cloud training was used during implementation.

## Host and versions

| Item | Observed value |
| --- | --- |
| CPU | Apple M4 Max (`sysctl -n machdep.cpu.brand_string`) |
| OS / architecture | macOS 15.6 / native arm64 |
| Python | 3.12.11 (uv-managed ARM64) |
| uv | 0.10.6 |
| Anomalib | **2.7.0** |
| torch / torchvision | **2.10.0 / 0.25.0** |
| OpenVINO | **2026.4.1**, runtime build `2026.4.1-22982-e213a147257-releases/2026/4` |
| OpenVINO devices | **`['CPU']`** |
| PyTorch MPS / CUDA | Available / unavailable; training explicitly CPU |
| OpenCV / Pillow | 5.0.0.93 / 13.0.0; conversion probe passed |

`artifacts/doctor.json` contains actual probes. `uv.lock` resolves 89 packages across target platforms; 87 were installed/audited on this Mac. `uv sync --locked --offline` succeeded after initial installation. No CUDA/XPU/NPU packages were added. Linux x86_64 resolution exists, but installation/execution on Linux has not been tested.

## Executed outcomes

- Dataset generator and validator succeeded: 12 good train, 6 good + 6 defective val, 6 good + 6 defective test. All 36 images are explicitly synthetic. Duplicate/missing/corrupt/leakage cases are tested separately.
- Actual Anomalib `Patchcore` / `Engine.fit` succeeded on CPU using official pretrained `resnet18.tv_in1k`. Fit + calibration took about **6.96 seconds** after imports. This small-fixture timing is not a real-dataset training estimate.
- Downloaded backbone cache revision: `timm/resnet18.tv_in1k`, snapshot `bbd144b3e5565108aad885f145491d11bc6ce807`; metadata also records the torchvision source URL and timm 1.0.31. The full learned checkpoint hash is recorded with the model.
- Checkpoint model version: `98a85da202d7df09`; threshold version: `f66e03aa01039932`. Threshold was selected solely from 12 validation scores: raw distance **1.8165923953056335**, uncertainty margin **0.041223672866821284**. This threshold is **not suitable for real camera deployment**.
- Uncompressed OpenVINO export succeeded. Input `[1,3,128,128]`, `pred_score` `[1]`, `anomaly_map` `[1,1,128,128]`. XML/BIN/JSON exist in `artifacts/models/patchcore/`.
- Separate-process `predict` succeeded on a held-out synthetic defective image; it returned `DEFECTIVE`, raw score approximately **2.85713**, and saved a map/overlay. It did not depend on a training-process Python object.
- Executed PyTorch/OpenVINO parity on the 12 **validation** images: maximum absolute score difference **1.6093254089355469e-06**, zero decision mismatches (tolerance 0.001).
- Independent synthetic test report was generated with TN=6, FP=0, FN=0, TP=6 and no uncertainty abstentions. These deliberately simple drawings are a **software smoke test, not real defect accuracy or a challenge-performance claim**. No real-data metrics exist.
- Scripted mock demo: 3/3 cycles, including good, defective and one reinspection, no errors/timeouts.
- Actual OpenVINO + MockRobot directory demos: 6/6 good cycles and 6/6 defective cycles, no errors/timeouts. Physical task success remains unverified.
- CPU benchmark: 5 warmups, 30 measured calls, one fixed validation image; mean **5.172 ms**, p50 **5.238 ms**, p95 **5.661 ms**. Measures `infer()` only, excluding decode/resize/compile/overlays. The optional simulation was running during this measurement; this is a local observation under that workload, not an isolated hardware benchmark or Intel prediction.
- Optional MuJoCo: isolated plugin 0.5.0 / runtime 0.4.0 / MuJoCo 4.0.1 installed, SO-101 scene started, browser arm/cube/target verified, wrist and overview rendering confirmed by health response. Stopped afterward. It is not integrated with Brickcell's controller.

## Test and tooling results

Final default offline run: **44 passed, 3 skipped, 0 failed** in 1.79s. The three skips are intentional model-asset tests gated by `BRICKCELL_MODEL_CONFIG`.

Final opt-in offline full run: **47 passed, 0 skipped, 0 failed** in 4.73s. Two upstream deprecation warnings (`timm.models.layers` and Anomalib's DINOv2 components) remain; neither affects PatchCore tests. Earlier suites also passed as tests were added; there were no unresolved pytest failures.

Ruff check passed and all 32 Python files passed formatting. Installed API signatures/source hashes were captured in `artifacts/api_signatures.json`. JUnit outputs are `artifacts/tests-software.xml` and `artifacts/tests.xml`.

Coverage includes directory importing, corrupt/missing images, duplicate hashes/IDs/paths, normal-only fitting, object/session overlap, deterministic grouped splits, changed-image provenance, RGB/channel/range handling, quality gates, threshold calibration/uncertainty/missing threshold, malformed input/result data, robot state transitions, good/defective sorting, retry budgets, action and inspector timeouts, stale results, failed grasp/release, safe stop after logging errors, full CLI mock integration, OpenVINO inference and numerical parity.

## Commands actually executed

Commands were run from the repository root. Agent executions used `UV_CACHE_DIR=/tmp/brickcell-uv-cache`; native OpenVINO commands used approved execution outside the restrictive agent sandbox.

```bash
uname -a
python3 -c 'import platform,sys; print(platform.machine()); print(sys.version)'
uv --version
uv python list --only-installed
sysctl -n machdep.cpu.brand_string
uv sync --python 3.12
uv sync --locked --offline
uv run python -m brickcell doctor
uv run python -m brickcell synthetic
uv run python -m brickcell validate-data --manifest data/synthetic/manifest.json
uv run python -m brickcell demo --mock
uv run python -m brickcell train --config configs/patchcore.yaml
uv run python -m brickcell export --config configs/patchcore.yaml
uv run python -m brickcell parity --config configs/patchcore.yaml
uv run python -m brickcell predict --model artifacts/models/patchcore/model.xml --image data/synthetic/test/defective/synthetic-test-defective-000.png
uv run python -m brickcell evaluate --config configs/patchcore.yaml
uv run python -m brickcell benchmark --model artifacts/models/patchcore/model.xml --image data/synthetic/val/good/synthetic-val-good-000.png --iterations 30 --warmup 5
uv run python -m brickcell demo --model artifacts/models/patchcore/model.xml --images data/synthetic/test/good --output artifacts/demo-good
uv run python -m brickcell demo --model artifacts/models/patchcore/model.xml --images data/synthetic/test/defective --output artifacts/demo-defective
uv run pytest -m 'not model' -q
uv run --offline pytest -q --junitxml=artifacts/tests-software.xml
BRICKCELL_MODEL_CONFIG=configs/patchcore.yaml uv run --offline pytest -q --junitxml=artifacts/tests.xml
uv run ruff check src tests scripts --fix
uv run ruff format src tests scripts
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run python scripts/inspect_apis.py
uv venv artifacts/mujoco-venv --python 3.12
uv pip install --python artifacts/mujoco-venv/bin/python physicalai-mujoco-plugin
artifacts/mujoco-venv/bin/physicalai-mujoco start --help
MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" PHYSICALAI_MUJOCO_CACHE_DIR="$PWD/.cache/physicalai_mujoco" artifacts/mujoco-venv/bin/physicalai-mujoco start --profile so101 --name brickcell-so101-smoke --seed 42 --status-json --http-port 18080 --viser-port 19090 --exit-with-parent
curl --fail --max-time 5 -sS http://127.0.0.1:18080/health -o artifacts/simulation/health.json
uv pip freeze --python artifacts/mujoco-venv/bin/python
```

Grouped import/split CLI functions and failure cases were executed by pytest using temporary fixtures. The native device probe, source/signature inspections and Git hygiene checks were also executed. The simulation startup command was attempted twice and then stopped with Ctrl+C after verification.

## Issues encountered and resolved

1. **Restricted sandbox OpenVINO crash:** a native `Xbyak_aarch64::Error: internal error` terminated the initial device probe. Approved native local execution succeeded. Doctor uses child processes so it can report this cleanly. This is not evidence that the M4 lacks OpenVINO CPU support.
2. **ONNX opset mismatch:** the first successful export inherited Anomalib's default opset 14 and logged a failed downgrade from exporter opset 18, retaining the newer graph. The wrapper now explicitly requests opset 18; re-export and numerical parity succeeded without that downgrade attempt. The initial message is retained in the append-only export log.
3. **MuJoCo first-start deadline:** initial scene startup exceeded the external plugin's 10s owner deadline. The second attempt with cached assets/imports succeeded. No plugin source patches were applied.
4. **Agent cache permissions:** the global uv cache was not writable in the sandbox. A task-specific `/tmp` cache and project-local library caches resolved this without changing `HOME`.

## Created files and remaining work

Created `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `README.md`, `AGENTS.md`; model/inspection/controller configs and an isolated MuJoCo requirements snapshot; the `src/brickcell` package (data, detector, inspection, sources, controller, metrics, CLI/doctor/utilities); six test files; `scripts/inspect_apis.py`; and architecture, collection, deployment, simulation and validation documentation. Existing `ssh.md` is unchanged and was not used for connections during this task.

Generated data, models, reports, logs, caches and environments are local and Git-ignored. No commits/pushes were made. No API keys/private keys were added.

Still unverified: real brick/camera accuracy; real-data threshold and quality settings; calibrated multi-view capture; physical grasp/release/placement; SO-101 hardware adapter, trajectories, emergency stop and joint-unit checks; Intel CPU/GPU/NPU deployment/precision/throughput; Studio policy training/export; Modal jobs; and a controller adapter for the optional MuJoCo scene. Linux compatibility is resolved but not runtime-tested. See `INTEL_DEPLOYMENT.md` for the staged hardware plan.

## Follow-up live demonstration

On request, reran `demo --mock` (3 completed cycles, 1 reinspection, 0 errors) and the OpenVINO defective-image demo (6 completed cycles, 0 errors). Restarted the SO-101 simulation, selected **Conveyor Sort** in the viewer and enabled **Autopilot → Drive the arm**. The local health endpoint confirmed `scene=conveyor_sort`, `autopilot.mode=drive`, and an active `carry` phase targeting `item_hex_red_cracked`, at approximately 50 Hz with both cameras around 30 fps. The viewer was opened and left running at `http://127.0.0.1:19090` for the user. This motion is the plugin's scripted demonstrator, not Brickcell/OpenVINO-controlled motion. Snapshot: `artifacts/simulation/live-health.json`. Use the viewer's Shutdown control to stop it.

## Camera-driven two-bowl implementation — 14 November 2026

This supersedes the earlier note that MuJoCo was not integrated. The new scene and bridge run native ARM64 on the same M4 Max. `doctor` was rerun: Python 3.12.11, Anomalib 2.7.0, torch 2.10.0, torchvision 0.25.0, OpenVINO 2026.4.1; OpenVINO devices **CPU** only, MPS available but unused. Report: `artifacts/two-bowl/doctor.json`.

Created `src/brickcell/simulation/{world,generate,server,bridge,protocol,run}.py` and package initializer; `configs/patchcore-sim.yaml`, `configs/inspection-sim.yaml`, and `tests/test_simulation.py`. Extended `tests/test_model.py` and pytest markers in `pyproject.toml`; updated README, architecture and simulation documentation. No detector or supervisor implementation changes were required. Dependencies remain locked; `uv sync --locked` succeeded after the project metadata update.

Executed results:

- Rendered 48 camera fixtures using MuJoCo contact grasps: 24 train-good, 12 validation, 12 test. Dataset validation: no missing/corrupt images, duplicates or partition overlap. Synthetic cloned geometry, not real data.
- PatchCore training succeeded; model version `8e92730963cde567`. Only good images fitted the memory bank. Validation-only threshold **1.3344924449920654**, uncertainty margin **0.03300517779588699**; threshold version `8d65f6c4bb0254c4`.
- Uncompressed OpenVINO export and independent-process CPU inference succeeded. Input `[1,3,128,128]`, score `[1]`, anomaly map `[1,1,128,128]`.
- PyTorch/OpenVINO parity on 12 calibration images: **maximum absolute difference 0.000007241964340209961**, zero decision mismatches.
- Frozen-threshold held-out synthetic test: TN=6, TP=6, FP=0, FN=0, no uncertainty abstentions. This is a software consistency check, not a real defect-performance result. Mean CPU inference 9.164 ms, p50 5.588 ms; includes first-call warmup, only 12 samples, no p95 claim.
- First live camera/physics run: **4/4 completed**, GOOD=2, DEFECTIVE=2, zero reinspection/timeouts/errors. Scores by opaque part ID: 0.513532, 2.089751, 0.513414, 2.056424. The arm used actual contact grasps, and each release was checked inside its commanded bowl. Mean cycle 17.538 s (deliberately paced simulation). Report: `artifacts/two-bowl/runs/20261113-235951-952892/summary.json`.
- Final software suite: **56 passed, 5 skipped**, zero failures, 1.97 s. Skips are four explicit model-asset tests and one live-simulator test; JUnit: `artifacts/two-bowl/tests-software.xml`.
- Opt-in model and live-simulation suite: **5 passed, 56 deselected**, zero failures, 84.03 s. Includes a full second four-object contact-physics run and a causality test that changes only image pixels while retaining an identical object ID; the selected bowl changes. Two upstream deprecation warnings remain. Report: `artifacts/two-bowl/tests/latest.json`.
- Ruff check and formatting passed for all 40 Python files. No physical hardware success or Intel acceleration was claimed.

Commands executed for this addition (from the repository root; native OpenVINO/rendering commands used approved local execution):

```bash
PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" artifacts/mujoco-venv/bin/python -m brickcell.simulation.generate --output data/two-bowl
uv run python -m brickcell train --config configs/patchcore-sim.yaml
uv run python -m brickcell export --config configs/patchcore-sim.yaml
uv run python -m brickcell parity --config configs/patchcore-sim.yaml
uv run python -m brickcell evaluate --config configs/patchcore-sim.yaml
PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" artifacts/mujoco-venv/bin/python -m brickcell.simulation.server
uv run python -m brickcell.simulation.run --reset
uv sync --locked
uv run pytest --junitxml=artifacts/two-bowl/tests-software.xml
BRICKCELL_MODEL_CONFIG=configs/patchcore-sim.yaml BRICKCELL_QUALITY_CONFIG=configs/inspection-sim.yaml BRICKCELL_SIMULATION_URL=http://127.0.0.1:18081 uv run pytest -m 'model or simulation'
uv run python -m brickcell doctor --model artifacts/models/patchcore-sim/model.xml --output artifacts/two-bowl/doctor.json
uv run python -m brickcell validate-data --manifest data/two-bowl/manifest.json --output artifacts/two-bowl/data-validation.json
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

During development, an unreachable initial IK branch was fixed by initializing a reachable ready pose; the grasp yaw was changed to expose the inspection face; separate drop offsets avoid stacking; settling is checked before completion. A browser replay launch initially resolved the venv Python symlink into the bare interpreter, producing `ModuleNotFoundError: yaml`; preserving the venv executable path fixed that issue. Failed development attempts are not counted as successful runs.

Limits: one visible synthetic defect face, three named virtual cameras but only inspection used for inference, scripted IK and known pickup poses, no learned VLA or real-camera picking. The model must be retrained/calibrated on real station images. Real arm calibration, hardware driver/policy integration, camera alignment, safe motion, emergency stop and Intel device benchmarking remain unverified.

The final browser replay button was exercised after fixing its launcher: 4/4 completed, GOOD=2, DEFECTIVE=2, zero errors/timeouts/reinspections. Report: `artifacts/two-bowl/runs/20261114-000814-770102/summary.json`. The connected viewer visibly reported completion and was left open at http://127.0.0.1:19091/. Screenshot: `artifacts/two-bowl/completed-view.jpg`.

## Five-face contract, ACT preparation and sharing — 14 November 2026

The user selected four sides plus the studded top, excluding the bottom; marker marks/surface contamination first. Their photo shows square and long/narrow colored bricks in several orientations. They estimated approximately 20–40 mm, not measured dimensions. `configs/objects.yaml` preserves that uncertainty.

Implemented and tested `CoverageLedger`: all five face results are needed for GOOD, invalid/missing/uncalibrated results require reinspection, errors stop sorting, identities and duplicate frames are checked, and detected defects remain latched. **This ledger is not yet connected to a five-face presentation/re-grasp sequence. The live demo remains single-view.**

Added an isolated official ACT lab, demonstration schema/recorder, teacher-free simulation rollout worker, training/evaluation scripts, object/coverage configs, Makefile and team handoff documentation. `policy/uv.lock` pins the training environment separately from `uv.lock` and the simulator snapshot. No paid/cloud deployment or hardware connection was made.

Executed evidence:

- `record_policy` produced six successful synthetic scripted-teacher episodes with 2,250 synchronized frames: train=4 episodes, val=1, test=1. Both cameras, six observed joints, six pre-step commanded joint targets and timestamps are archived with hashes. This is not human teleoperation or a LeRobot/Studio UI dataset.
- Official Physical AI Studio ACT CPU training completed 50 steps in 12.52 seconds; first/last training losses 7.72383/0.61693, seeded validation-batch action MAE 0.07099 radians, exact checkpoint reload prediction match. This measures a learning pipeline, not manipulation success. Metadata/report: `artifacts/policy/act-smoke/report.json`.
- Strict teacher-free rollouts: 0/2 success, rejected out-of-limit predictions at step 0. Diagnostic rollouts with explicitly enabled simulator-limit saturation: 0/2 success after 600 steps each, 600 clipped action values per episode. Reports: `rollouts.json` and `rollouts-bounded.json` alongside the checkpoint. The scripted controller was never used for rollout actions. This smoke policy is **not deployable**.
- The initial ACT constructor rejected list-shaped feature dimensions; the wrapper was corrected to the installed API's tuple requirement. macOS PyAV/OpenCV imports emit duplicate AVFoundation-class warnings; completed training/rollouts were not blocked by them.
- Final offline software tests: **69 passed, 5 skipped, zero failures**, 2.03 seconds; `artifacts/tests-policy-prep.xml`. The skips remain four explicit model tests plus the live-simulator test, whose earlier successful executions are recorded above. No full model/live rerun was needed for documentation, independent coverage/data logic, or the default no-op recording hook.
- Ruff check and format passed for 49 Python files including the separate policy scripts.
- Wheel and source archive built. Source archive inspected: 74 entries, no `ssh.md`, datasets, artifacts or virtual environments. Wheel contains the data tools and coverage ledger. Installed the wheel without dependencies in a fresh temporary environment and ran `python -I -m brickcell --help`; this tests package/CLI discovery, not a fresh installation of every dependency.
- Packaging initially included the existing unrelated `ssh.md`; the sdist now has an explicit project-file allowlist. Existing repository history/file content was not altered.

Principal executed commands:

```bash
uv sync --project policy --python 3.12
PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" artifacts/mujoco-venv/bin/python -m brickcell.simulation.record_policy
PYTHONPATH=src TORCH_HOME="$PWD/.cache/torch" policy/.venv/bin/python policy/train_act.py
PYTHONPATH=src policy/.venv/bin/python policy/evaluate_act.py
PYTHONPATH=src policy/.venv/bin/python policy/evaluate_act.py --clip-to-sim-limits --output artifacts/policy/act-smoke/rollouts-bounded.json
uv sync --locked
uv run pytest --junitxml=artifacts/tests-policy-prep.xml
uv run ruff check src tests scripts policy
uv run ruff format --check src tests scripts policy
uv build --out-dir artifacts/dist
uv venv artifacts/package-check --python 3.12
uv pip install --no-deps --python artifacts/package-check/bin/python artifacts/dist/brickcell-0.1.0-py3-none-any.whl
artifacts/package-check/bin/python -I -m brickcell --help
git switch -c codex/physical-ai-prototype
```

No hosted deployment is required to share the feature branch. The next useful work is reviewed data, meaningful learned-policy rollouts, a verified five-face presentation plan, and real station data/adapter integration. See `POLICY_LAB.md` and `TEAM_HANDOFF.md`.

## SO-101 manual geometric alignment on Intel — 14 November 2026

Executed on the team's Ubuntu Intel computer through authorized SSH access,
using Python 3.12.16. Raw acquisition used the existing LeRobot/servo-SDK uv
environment; independent kinematic checks used installed Frax 0.2.0 with JAX
64-bit precision. No root environment or Mac CPU dependency pins were changed.

Both arms' five direction signs were recorded as +1, and increasing gripper
values were observed to open the jaw. Before/after raw snapshots, rejected
isolation attempts and accepted results are retained under
`calibration/measurements/manual-direction-session/`. The leader elbow sign was
explicitly accepted by the user despite shoulder co-movement; it is not labelled
as a strict-isolation pass.

The user manually matched each arm by eye to the fixed CAD target
`[0, 93.1, -89.5, 1.5, 90]` degrees. A read-only capture derived the per-arm
encoder-zero offsets, preserving both original LeRobot calibration files and
their hashes. Both alignment flags are confirmed for this **manual visual
method**, whose physical uncertainty is unquantified. See
[`calibration/ALIGNMENT_REPORT.md`](../calibration/ALIGNMENT_REPORT.md).

Generated `models/so101-{leader,follower}-calibrated.urdf` contain five bounded
arm coordinates and fixed CAD hub/gripper coordinates. They are massless
kinematics-only models; effort/velocity, TCP and camera mounting transforms
were not measured. They are not contact-dynamics models or validated motor
command limits.

Executed checks:

- Both actual aligned models: Frax FK/Jacobian agreement, central-difference
  Jacobian checks and per-arm position-bound checks at 10 configurations each.
  Numerical errors were below 2e-8; this is software consistency, not measured
  physical pose accuracy. Machine-readable reports are in `models/`.
- Calibration/kinematics unit checks: **8 passed, 1 skipped** in each of the two installed interpreters.
  Each optional backend is tested where installed; the other backend is skipped
  in that interpreter. This includes rejection of stale calibration and CAD
  model bindings, angle-unit conversions, and Jacobians with TCP offsets.
- Latest upstream core software tests in an isolated uv environment on Intel:
  **69 passed, 5 skipped**, 2.28 seconds. Four skips require trained model assets;
  one requires a live simulator. Neither was started for this task.
- `ruff check src tests scripts` and `ruff format --check src tests scripts`
  passed (47 files). These files were the incoming prototype revision, not
  modifications to the detector/controller during calibration.

The core checks used a temporary source archive and minimal test dependencies
to preserve the working Intel XPU environment: pytest 9.2.0, Ruff 0.17.5,
NumPy 2.5.4, Pillow 13.0.0, PyYAML 6.0.3, OpenCV headless 5.0.0.93,
scikit-learn 1.9.1 and SciPy 1.18.1. Initial collection omitted scikit-learn;
adding that declared dependency resolved collection and the suite passed.
This was not a reinstallation or validation of the complete locked anomaly
model stack on Intel.

Reproducible arm checks:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tools -p 'test_so101_*.py' -v
PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 ~/frax/.venv/bin/python -m unittest discover -s tools -p 'test_so101_*.py' -v
JAX_ENABLE_X64=1 ~/frax/.venv/bin/python -B tools/so101_validate_alignment.py
```

All physical movements were made by people, with torque off. Acquisition sent
only READ instructions; no homing, limit, torque or goal-position registers
were written and no robot services were started/stopped. No automatic physical
pose command, grasp, inspection or sorting success was tested. A subsequent
supervised physical pose/TCP check is still needed to measure the alignment error.
