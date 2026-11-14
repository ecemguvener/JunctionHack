# Training on Modal

This page tells you how to run the Brickcell training jobs on [Modal](https://modal.com).
Modal runs the jobs in cloud containers. They are faster than the team laptops, and they
use mid-range resources, not the fastest possible ones.

All code is in `cloud/`. The app name is `brickcell-train`. The Volume name is
`brickcell-train`.

## Before you start

1. Install the Modal client in a separate environment. Do not add it to `uv.lock`:

   ```bash
   uv venv artifacts/modal-venv --python 3.12
   uv pip install --python artifacts/modal-venv/bin/python modal
   ```

   Or use `uvx modal ...`.
2. Log in one time: `modal setup`. Make sure that `modal profile current` works.
   Do not put tokens in files in this repository.
3. Run all commands from the repository root.

## Commands

| Make target | Modal command | What it does |
| --- | --- | --- |
| `make modal-warm` | `modal run cloud/modal_train.py::warm` | Builds all images. Use it after a lock file change. |
| `make modal-retrain-detector` | `modal run cloud/modal_train.py::pipeline --dataset two-bowl --overwrite` | New renders, then PatchCore train, export, parity and evaluate. Downloads the model. |
| `make modal-pipeline` | `modal run cloud/modal_train.py::pipeline --dataset two-bowl` | The same, but it stops if the Volume already has `data/two-bowl`. |
| `make modal-patchcore` | `modal run cloud/modal_train.py::train_patchcore --dataset two-bowl` | PatchCore only, on a dataset that is already on the Volume. |
| | `modal run cloud/modal_train.py::generate --dataset two-bowl` | Renders only. |
| `make modal-record` | `modal run cloud/modal_train.py::record_policy --demos policy-smoke --episodes 6` | Records ACT teacher demonstrations in parallel containers. |
| `make modal-act STEPS=2000` | `modal run cloud/modal_train.py::train_act --demos policy-smoke --steps 2000` | Trains ACT on a GPU. Downloads the checkpoint. |
| `make modal-act-evaluate RUN_ID=...` | `modal run cloud/modal_train.py::evaluate_act --run-id <id>` | Teacher-free rollouts on a CPU container. |
| `make modal-stop` | `modal app stop -y <app id>` and `modal app list` | Stops the app and shows that no app is running. |

Each command makes a run ID, for example `20261114T061324Z-f7cc`. Use `--help` after the
entrypoint name to see all options, for example
`modal run cloud/modal_train.py::pipeline --help`.

Useful options:

- `--model-name NAME`: the local model folder is `artifacts/models/NAME`. The default is
  `patchcore-sim`, so `configs/patchcore-sim.yaml` finds the new model.
- `--overwrite`: the old Volume dataset with the same name moves to
  `data/<name>.previous-<time>` on the Volume. Nothing is deleted.
- `--no-sharded`: use the unchanged generator in one container (slow; see below).
- `--no-gpu-train`: fit PatchCore on the CPU of the same container.
- `--scene-config configs/other.yaml`: render with a different scene config.
- `train_act --gpu A100-40GB`: use a different GPU for one run.

## Which source the jobs use

The jobs use the **current working tree**, not a Git commit. When a container starts,
Modal adds the local `src/`, `configs/`, `assets/`, `policy/` and `cloud/` folders to it.
A change in these folders does not cause an image build. Thus, if you change the scene
(`configs/sim-scene.yaml`, `world.py`), the next run renders the new scene.

Each run records the Git commit, the branch, a dirty flag, the changed files and a
SHA-256 of all uploaded source files. This makes a run from a dirty tree traceable.

## What each function does

| Function | Resources | What it does |
| --- | --- | --- |
| `generate_shard_job` | 4 CPU, 4 GB, one container per episode (12 in total) | Runs `cloud/generate_shard.py render` for one episode (4 images). MuJoCo renders with OSMesa. |
| `merge_dataset_job` | 2 CPU | Joins the 12 shards into `data/<dataset>` on the Volume, in the same row order as `generate.py`. |
| `generate_job` | 4 CPU | The unchanged `brickcell.simulation.generate` in one container (`--no-sharded`). |
| `train_patchcore_job` | GPU (L4, else A10G, else L40S), 4 CPU, 16 GB | One Python process: `validate-data`, `train` (GPU), `export` (CPU, OpenVINO), `parity` (CPU), `evaluate` (CPU). |
| `record_policy_shard` | 4 CPU per shard | `brickcell.simulation.record_policy` for 3 or more episodes. Seeds are `seed + episode index`, as in one local run. |
| `merge_policy_job` | 2 CPU | Gives global episode names and the normal splits (last two are val and test), then checks the result with `brickcell.policy.data.load_episodes`. |
| `train_act_job` | GPU (L4, else A10G, else L40S), 8 CPU, 16 GB | `policy/train_act.py` through `cloud/act_gpu.py`, on CUDA. |
| `evaluate_act_job` | 4 CPU | `policy/evaluate_act.py`. The simulator is the slow part, so it uses no GPU. |

Every function has a timeout (10 to 120 minutes).

### Images

There are three images. They are built from the lock files:

- **sim**: Python 3.12, `uv pip install -r configs/mujoco-requirements.txt`, OSMesa, and
  the SO-101 model from MuJoCo Menagerie (downloaded at build time).
- **core**: `uv sync --locked` from `uv.lock`, then the CUDA 12.8 build of the same torch
  2.10.0 and torchvision 0.25.0, then the timm `resnet18.tv_in1k` weights.
- **policy**: the sim environment, plus `uv sync --locked` from `policy/uv.lock`, the CUDA
  build of torch, and the torchvision ResNet-18 weights.

The lock files pin the CPU build of torch for Linux. The images replace only that build
with the CUDA build of the same version. All other packages are the locked versions.
The Modal image builder sets a PyPI mirror. The build removes that setting for
`uv sync --locked`, because the lock files name pypi.org.

Downloads are in the images, so a job does not download weights or robot models.
`HF_HUB_OFFLINE=1` is set at run time.

### GPU training and the Intel export

`brickcell.detector.workflow.train` sets `accelerator="cpu"`. `cloud/patchcore_steps.py`
does not change that file. It replaces `anomalib.engine.Engine` with a subclass that uses
the GPU only for the engine that has `max_epochs` (the training engine). At the end of the
fit, Lightning moves the model to the CPU. Thus the threshold calibration, the OpenVINO
export, parity and evaluation are the same CPU code as on a laptop. The export target
stays the Intel laptop CPU/NPU.

Result of the first run (model `d2078a1b39c64527`, fit on an NVIDIA A10): parity on Modal
passed with a maximum difference of 1.9e-6. The downloaded `model.xml` also passed parity
on the Intel i5-13500H laptop CPU (maximum difference 5.9e-6, zero label mismatches) and
gave TN=6, TP=6, FP=0, FN=0 on the synthetic test split.

## Where the results go

| Result | Volume path | Local path |
| --- | --- | --- |
| Rendered dataset | `data/<dataset>/` | Not downloaded (see below) |
| PatchCore model and reports | `runs/patchcore/<run id>/model/` | `artifacts/models/<model name>/` |
| Demonstrations | `data/<demos>/` | Not downloaded |
| ACT checkpoint and report | `runs/act/<run id>/policy/` | `artifacts/policy/<policy name>/` |
| Logs | `runs/<kind>/<run id>/*.log` | In the model folder (`training.log`, `export.log`) |
| Provenance | | `artifacts/modal/runs/<run id>/provenance.json` and `modal_provenance.json` in the model folder |

If the local target folder exists, it moves to `<folder>.previous-<UTC time>` first.
Nothing is deleted.

The model folder has `metadata.json`, `model.xml`, `model.bin`, `model.ckpt`,
`parity.json`, `test_report.json`, `modal_result.json` (step times, GPU, cold start) and
`modal_provenance.json` (app ID, run ID, GPU type, phase times, Git commit, dirty flag,
source SHA-256).

The data stays on the Volume, so that there is no round trip. To get the dataset (for
example to run `brickcell parity` on a laptop), use:

```bash
modal volume get brickcell-train data/two-bowl data/
```

## Retrain the detector after a scene change

1. Change the scene (`configs/sim-scene.yaml`, `world.py`). You do not need a commit.
2. Run `make modal-retrain-detector` (or the `pipeline --overwrite` command).
3. Read the summary at the end of the output. Make sure that `parity_passed` is `true`.
4. The new model is in `artifacts/models/patchcore-sim/`. The old model is in
   `artifacts/models/patchcore-sim.previous-<time>/`.
5. Optional: run `modal volume get brickcell-train data/two-bowl data/`, and then
   `uv run python -m brickcell parity --config configs/patchcore-sim.yaml` on the laptop.

If `generate.py` changed, the shard runner stops with a message (see below). Then use
`--no-sharded`, or update `cloud/generate_shard.py`.

## Parallel generation

`brickcell.simulation.generate` renders all 12 episodes in one process with one random
generator. It has no seed or shard option. `cloud/generate_shard.py` repeats its
per-episode loop with one random generator per episode
(`numpy.random.default_rng([2026, split index, episode])`), so each episode can run in a
different container. The image names, labels, splits, session IDs, defect rule and
manifest format are the same. The colour and camera jitter values are different from a
local run (same ranges, different random draws). Each shard starts with a new `World`, so
the arm starts from the ready pose at each episode, not from the end pose of the previous
episode.

The wrapper compares the SHA-256 of `generate.py` with the value that it was written for,
and stops if they differ. `tests/test_cloud_modal.py` does the same check.

A change in `generate.py` would remove the copy: add `--episodes split:episode ...`
and a per-episode seed to `generate.py`, and a `merge` command. Then the shard runner can
call `generate.py` directly.

## Timings

Measured on 2026-11-14. The local reference is the team laptop (Intel i5-13500H, 16
threads). The PatchCore dataset has 48 images (12 episodes).

| Phase | Laptop | Modal, first plan | Modal, current plan |
| --- | --- | --- | --- |
| Image build (one time) | none | core 86 s, policy 79 s, built at the first `run` | 134 s for all images (`warm`), weights and robot model included |
| Cold start | none | about 8 s | 4 to 6 s per container |
| Generate 48 images | 164.5 s | 325.7 s (one container, OSMesa) | 39.2 s (12 containers) + 7.5 s merge |
| PatchCore imports | in each step | in each step (5 processes) | 9.1 s, one time |
| validate-data | not measured | | 0.5 s |
| Train (fit + calibration) | 18.1 s process, 5.6 s fit | not measured (no L4 capacity) | 2.3 s (fit 2.0 s on an A10) |
| Export (OpenVINO) | 8.6 s process | | 6.0 s |
| Parity | 5.1 s process | | 1.1 s |
| Evaluate | 1.4 s process | | 0.3 s |
| PatchCore call, local view | 33.2 s (4 processes) | | 26.7 s (with cold start) |
| Download model | none | | 18.3 s (20 files) |
| Total, generate to local model | about 198 s | | 91.7 s |

ACT (policy-smoke: 6 demonstrations, 2,338 frames, batch 8):

| Phase | Laptop / Mac reference | Modal |
| --- | --- | --- |
| Record 6 episodes | not measured on the laptop | 195 s (2 containers of 3 episodes; 142 s and 174 s) |
| ACT 50 steps (`training_seconds`) | 12.5 s on Apple M4 Max CPU | 8.7 s on an A10 (with first-step warm-up, validation and save) |
| ACT 500 steps | about 125 s (M4 Max, 0.25 s per step) | 26.3 s; 0.039 s per step after the first 50 |
| ACT call, local view (50 steps) | | 28.0 s (cold start, imports, training) |
| Evaluate 2 seeds | | 26.5 s (both stop at step 0: the action is out of the limits, as in the local smoke run) |

For 2,000 steps, the estimate is about 85 s on an A10 and about 500 s on the M4 Max CPU.

Notes:

- The generator time is MuJoCo contact physics, not rendering. Rendering was 8.4 s of
  CPU time in total for 48 images. One Modal CPU core is slower than one laptop core, so
  one container is slower than the laptop. The gain comes from the 12 parallel
  containers. One episode is the smallest shard, because the four picks of an episode
  depend on each other.
- For this small dataset the GPU fit is 2 s. The GPU becomes important only for a
  much larger dataset (the coreset sampling grows with the number of patches).
- The first plan waited more than 10 minutes for an L4 GPU and was stopped. The current
  plan gives a fallback list (L4, A10G, L40S). The run above got an A10.

## Costs and resource choice

Approximate Modal list prices (see <https://modal.com/pricing> for current values):
L4 about USD 0.80 per hour, A10G about USD 1.10 per hour, L40S about USD 1.95 per hour,
A100-40GB about USD 2.10 per hour, H100 about USD 3.95 per hour, CPU about USD 0.05 per
core-hour.

- PatchCore uses one mid-range GPU (L4 first). The fit is small, so a larger GPU does not
  make the run shorter. The CPU steps run in the same container, so there is only one
  cold start.
- Generation uses CPU containers only. Twelve containers with 4 cores for about 40 s cost
  less than one cent.
- ACT uses the same mid-range GPU list. The ACT model is small (dim 128, batch 8). The
  batch is made with NumPy on the CPU in the training loop, so a faster GPU gives little
  gain until the script has a DataLoader with workers and a larger batch (see below).
  Use `--gpu A100-40GB` or `--gpu H100` only after a measurement shows a gain.
- A full detector retrain (generate + PatchCore) costs less than USD 0.05.

## Changes that are not made (they need changes in src/ or policy/)

- `generate.py`: per-episode seeds and an episode selection option, so that the shard
  runner does not need its copy of the loop.
- `policy/train_act.py`: a `--device` option and a `torch.utils.data.DataLoader` with
  `num_workers` and `pin_memory` for the batch assembly. Now `cloud/act_gpu.py` sets the
  torch default device to CUDA, and the batch is made in the main thread. The script also
  writes `"device": "CPU"` in `report.json`; the wrapper corrects it.
- `brickcell.detector.workflow.train`: an `accelerator` field in the model config, so
  that `cloud/patchcore_steps.py` does not need to replace the Engine class.

## Stop everything

`modal run` stops its app at the end. To make sure:

```bash
modal app list          # brickcell-train must show "stopped" and 0 tasks
modal app stop -y <app id>
```
