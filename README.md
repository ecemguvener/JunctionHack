# Brickcell — local visual inspection prototype

A working Anomalib 2.7.0 PatchCore → OpenVINO CPU inspection pipeline with a replaceable mock robot controller. Verified on native **Apple M4 Max / macOS 15.6 / ARM64 / Python 3.12.11**. No Docker, x86 emulation, CUDA, Intel XPU/NPU dependencies, physical robot, or cloud training is needed.

The examples use **synthetic drawings and rendered camera images**. They demonstrate software integration, not real brick-defect accuracy. All robot actions are simulated. See [executed validation](docs/VALIDATION.md), [architecture](docs/ARCHITECTURE.md), [real-image collection](docs/DATA_COLLECTION.md), and [Intel deployment plan](docs/INTEL_DEPLOYMENT.md).

## Watch the two-bowl inspection cell

The SO-101 now picks, presents, inspects and sorts using **actual OpenVINO decisions from a rendered camera**, with contact-based grasp/release. With the existing local simulator environment and camera model:

```bash
# Terminal 1, from this repository:
PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" \
  artifacts/mujoco-venv/bin/python -m brickcell.simulation.server

# Terminal 2 (or press the viewer's replay button):
uv run python -m brickcell.simulation.run --reset
```

Open **http://127.0.0.1:19091/**. The two bowls are GOOD and DEFECTIVE; the panel shows the camera image, score and action. Motion uses scripted IK and known pickup poses, not a trained VLA. Only the front inspection face is evaluated. [Full setup, training, tests and limitations](docs/SIMULATION.md).

## Install and verify

An experimental [policy preparation lab](docs/POLICY_LAB.md) now records synthetic
demonstrations and trains the official Physical AI Studio ACT model in a separate
locked environment. The CPU training/checkpoint smoke test passed; its short-trained
policy achieved **0/2 closed-loop successes** and is not the controller shown in the
working sorting demo. A tested five-face coverage ledger is also available, but the
live demo remains **single-view** until presentation/re-grasp motions are implemented.

For team ownership, sharing, real datasets and the next hardware milestone, see
[the handoff plan](docs/TEAM_HANDOFF.md).

From the repository root:

```bash
uv python install 3.12
uv sync --locked
uv run python -m brickcell doctor
uv run pytest
uv run python -m brickcell demo --mock
```

`brickcell` is also installed as a CLI: `uv run brickcell --help`.

The lock resolves native macOS ARM64 and Linux x86_64. Linux uses the official PyTorch CPU wheel index. Linux/Ubuntu/Intel devices have **not** been executed here. Versions are recorded in `uv.lock`; the verified core versions are Anomalib **2.7.0**, torch **2.10.0**, torchvision **0.25.0**, OpenVINO **2026.4.1**. MPS is available on this Mac, but the code explicitly trains on CPU. OpenVINO reports only **CPU**.

When running through a restricted agent sandbox, set a writable uv cache, e.g. `export UV_CACHE_DIR=/tmp/brickcell-uv-cache`. Native OpenVINO CPU JIT probing crashed inside that sandbox with an Xbyak error; it succeeded using approved local native execution. From normal Terminal, run the commands directly. `doctor` isolates native probes and reports failures rather than crashing the CLI. A missing trained model is a warning; an import/device-probe failure returns nonzero.

## Run the complete real-model smoke test

The current workspace already contains the generated dataset and trained artifacts. Use the prediction/test commands below to reuse them. To reproduce from a fresh clone:

```bash
uv run python -m brickcell synthetic --output data/synthetic
uv run python -m brickcell validate-data --manifest data/synthetic/manifest.json
uv run python -m brickcell train --config configs/patchcore.yaml
uv run python -m brickcell export --config configs/patchcore.yaml
uv run python -m brickcell parity --config configs/patchcore.yaml
uv run python -m brickcell evaluate --config configs/patchcore.yaml
```

Generation refuses to overwrite an existing fixture manifest. Training refuses to overwrite an existing checkpoint. For a new experiment, copy the YAML and choose a new `output_dir`; paths in model YAML resolve relative to the YAML file. Keep the original test split frozen. The synthetic generator creates 12 good training images, 6 good + 6 defective calibration images, and 6 good + 6 defective independent fixture images.

The baseline uses `resnet18.tv_in1k`, layers `layer2/layer3`, 128×128 RGB, batch 4, CPU float32, 5% coreset, 3 neighbors, zero data-loader workers, and seed 42. `configs/patchcore.yaml` exposes these settings. Larger images/memory banks increase runtime and memory cost. PatchCore fits a feature memory bank rather than backpropagating a defect classifier.

First training downloads official pretrained ResNet weights through timm (`timm/resnet18.tv_in1k`; original torchvision URL is recorded in metadata). Network failure is reported and logged; there is no automatic fallback to random weights or another Anomalib version. After caching, training/inference can run offline. Checkpoint restore uses `pre_trained=False` to prevent a redundant download, then restores the trained backbone and memory bank.

Anomalib's automatic `Folder` split behavior is deliberately bypassed by a small explicit `ImageItem`/`ImageBatch` loader adapter. Only `val` scores calibrate the threshold. `test` is reserved for final evaluation. See [the exact API and split rationale](docs/ARCHITECTURE.md).

## Predict an image and view its anomaly map

```bash
uv run python -m brickcell predict \
  --model artifacts/models/patchcore/model.xml \
  --image data/synthetic/test/defective/synthetic-test-defective-000.png \
  --object-id example-brick
```

The JSON includes object/frame/view identity, timestamp, validity, raw anomaly score, decision, model/threshold versions, inference latency, and map path. Predictions, raw `.npy` maps, and `.png` overlays go to `artifacts/predictions/`. Invalid views return `REINSPECT` without calling the detector. Missing calibration returns `ERROR`. **Scores are distances, not defect probabilities.** Overlays are independently scaled for visibility, not comparable calibrated heat intensities.

The exported `model.xml`, `model.bin`, and `model.json` must be kept together. The loader validates model hashes, input shape and named score output. `--checkpoint PATH` can be supplied to `export`, but its hash must match that output directory's calibrated metadata. Export is uncompressed; no FP16/INT8 optimization is applied.

## Real inference driving simulated sorting

A directory is a camera substitute, with one file per object:

```bash
uv run python -m brickcell demo \
  --model artifacts/models/patchcore/model.xml \
  --images data/synthetic/test/good \
  --output artifacts/demo-good

uv run python -m brickcell demo \
  --model artifacts/models/patchcore/model.xml \
  --images data/synthetic/test/defective \
  --output artifacts/demo-defective
```

Both use actual OpenVINO inference and **MockRobot**. `demo --mock` instead uses explicitly scripted decisions and needs no images/model. It demonstrates good, defective and invalid-view/reinspection cycles. No mock action or release verification is evidence of physical success.

`configs/controller.yaml` configures timeouts and bounded retries. `configs/inspection.yaml` configures the blur/empty/ROI heuristics. Pass `--controller-config` or `--quality-config` to override them. The directory source does not fabricate new views: if a one-image object needs reinspection, the run eventually stops. The Python `DirectorySource({object_id: [view1, view2, ...]})` interface supports multiple captured views. No live webcam access is attempted by this prototype.

Each run saves an event JSONL with a unique run ID plus `summary.json` in its output directory. Logs identify `MOCK_SCRIPTED_NOT_ML` versus `anomalib_openvino_cpu`. Summaries include attempted/completed cycles, classifications, reinspections, errors/timeouts and cycle latency.

## Import real images

For on-demand webcam/RGB capture, quality checks and real-model inspection,
follow [the real-camera walkthrough](docs/REAL_CAMERA.md). Start with
`uv run python -m brickcell.sources.camera --help`. This sends no robot commands.
For the Intel laptop's two D415s and upright wrist camera together, see
[three-camera capture and conservative decision fusion](docs/MULTICAMERA.md).

Organize captures with globally stable physical object IDs and session IDs:

```text
data/raw/
  good/session-a/brick-001/front/frame-001.png
  defective/session-b/brick-101/front/frame-001.png
```

Every image must follow `label/session/object/view/image.ext`. Then:

```bash
uv run python -m brickcell import-data --directory data/raw --output data/import.json
uv run python -m brickcell split-data --manifest data/import.json --output data/manifest.json --seed 42
uv run python -m brickcell validate-data --manifest data/manifest.json
cp configs/patchcore.yaml configs/patchcore-real.yaml
```

Edit the copied config to set `manifest: ../data/manifest.json` and a new `output_dir: ../artifacts/models/real-v1`. Train/export/parity/evaluate using `--config configs/patchcore-real.yaml`.

Splitting treats connected object/session groups as indivisible. At least three independent all-good groups and two groups containing defects are required. About 60/20/20 of all-good groups go to train/val/test; groups containing defects alternate between val/test. Group sizes can be unequal. A session shared across all objects intentionally prevents splitting. All views of one physical object must stay together, even if that object later becomes defective. An existing split cannot be silently reshuffled.

The manifest is schema-versioned JSON. Each sample has `image_id`, relative/absolute `path`, `label` (`good`/`defective`), `split` (`train`/`val`/`test`), optional `object_id`, `session_id`, `view_id`, `defect_type`, `mask_path`, `defect_regions` (`[x0,y0,x1,y1]` pixels), and `synthetic`. Validation checks missing/corrupt images/masks, dimensions, labels/counts, duplicate IDs/paths/hashes, and both object/session leakage. Annotations are preserved/validated; this baseline reports image classification, not pixel-level segmentation accuracy. [Collection plan and sample manifest](docs/DATA_COLLECTION.md).

## Metrics, latency and tests

```bash
uv run python -m brickcell benchmark \
  --model artifacts/models/patchcore/model.xml \
  --image data/synthetic/val/good/synthetic-val-good-000.png \
  --iterations 30 --warmup 5

uv run pytest
BRICKCELL_MODEL_CONFIG=configs/patchcore.yaml uv run pytest -m model
BRICKCELL_MODEL_CONFIG=configs/patchcore.yaml uv run pytest --junitxml=artifacts/tests.xml
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run python scripts/inspect_apis.py
```

Ordinary tests run offline without model assets and explicitly skip four marked real-model tests and one live-simulator test. The opt-in suite checks OpenVINO load/inference, executed numerical parity, and real inspection results driving simulated good/defective sorting. Tests also cover data corruption/leakage, preprocessing/color channels, quality/thresholds, missing calibration, stale results, timeouts (including a blocked inspector), retry limits, failed grasps/releases, and logging failures.

`evaluate` writes `test_report.json`: confusion matrix, precision/recall/F1, defect recall/miss rate, good false-rejection rate, AUROC/AUPR and per-image inference latency. Binary metrics use the frozen threshold; uncertainty-band abstentions are reported separately. View-quality gating is not included in detector-only metrics. p50 requires >=2 timings and p95 >=20; unavailable statistics are `null`. `benchmark` explicitly warms up and repeats one image to measure CPU inference latency only. It is not an accuracy test or physical robot throughput measurement.

Training/export output includes configuration, exact installed versions, checkpoint, model and threshold metadata, split manifest, image hashes, logs, validation predictions/metrics, IR, parity and test reports. Assets and datasets are Git-ignored. No secret is needed for the baseline.

## Future work

The MuJoCo `SimRobot` and `SimCamera` now implement `RobotInterface` and `FrameSource` without changing detector/supervisor logic. Real grasp/release verification, safe trajectories, joint calibration, emergency stops, live camera color checks, Intel GPU/NPU benchmarks, VLA demonstrations/training, and Modal access must wait for station access. [Intel deployment](docs/INTEL_DEPLOYMENT.md) describes this integration. [The two-bowl workflow](docs/SIMULATION.md) keeps simulator dependencies separate from the core environment.
