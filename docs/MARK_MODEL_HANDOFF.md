# Plate and wrist model handoff

There are currently no calibrated real plate/wrist export pairs in this checkout.
The local plate pilots are uncalibrated; the XML files under
`artifacts/models/patchcore-sim` are synthetic. Neither can enable real sorting.
The `anomaly` branch provides the training workflow and bundle tool, not trained
weights. Model artifacts and personal captures must remain outside Git.

The reported arm is holding a brick. This workflow does not release it, move the
arm, or resume the interrupted cycle. Coordinate recovery with the cell operator
before another deployment. The missing-model preflight now refuses deployment
before camera/arm initialization or pickup.

## If your teammate already has both calibrated exports

Each camera export directory must contain:

| File | Purpose |
|---|---|
| `model.xml`, `model.bin` | OpenVINO graph and weights; both required |
| `model.json` | Runtime sidecar: hashes, threshold, crop, camera serial |
| `metadata.json` | Original training and calibration metadata |
| `package_versions.json`, `dataset_validation.json` | Exact versions and dataset validation evidence |
| `parity.json`, `test_report.json` | Validation parity and frozen held-out test report |

`metadata.json` cannot replace `model.json`. Serial contracts must be plate
`243622060450` and wrist `sonix-wrist`, modality `rgb`, with `synthetic: false`.
Do not copy the plate model into the wrist slot or invent a threshold.

From the repository root on the machine holding those exports:

```sh
git switch anomaly
git pull --ff-only origin anomaly
uv sync --locked
uv run python scripts/bundle_mark_models.py \
  --plate /absolute/path/to/plate/model \
  --wrist /absolute/path/to/wrist/model \
  --output artifacts/mark-handoff-v1 --check-runtime
tar -czf artifacts/mark-handoff-v1.tar.gz -C artifacts/mark-handoff-v1 .
```

The tool checks both exports before copying, verifies model hashes, real sensor
contracts, calibrated thresholds and matching training/test evidence, and copies
the trained crops into a portable `configs/mark-viewer.json`. `--check-runtime`
compiles both models on OpenVINO CPU; it does not open a camera or command an arm.
Bundle checks are consistency checks, not an independent audit of supplied labels
or proof of physical accuracy. Review the test report against the application
requirements before enabling sorting. The tool does not impose a test accuracy
acceptance policy.

Transfer the archive separately from Git to the teammate/Intel machine and
extract it into a new directory, for example
`/home/intel/brickcell-deploy/artifacts/mark-handoff-v1`. Preserve the complete
directory structure. From Intel's repository root, verify its hashes:

```sh
uv run python - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('artifacts/mark-handoff-v1')
for name, expected in json.loads((root / 'bundle.json').read_text())['files'].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Bundle hash mismatch: {name}')
print('Bundle file hashes verified')
PY
```

Use the absolute path to the bundled config with the existing deployment command:
`--rgbd-viewer-config /home/intel/brickcell-deploy/artifacts/mark-handoff-v1/configs/mark-viewer.json`.
Keep the teammate's existing service URLs/ports appropriate to Intel; the template
currently has local forwarded ports 18094 and 18091. Camera endpoints in this
config are for the viewer; deploy-cell has its own camera URL arguments. Changing
URLs must not change the recorded model crop or sensor identity. The bundle tool
does not edit the repository's empty default mappings or launch deployment.

## If either model is missing

Both cameras need reviewed real captures with explicit good/defective labels,
physical object IDs, capture-session IDs, view IDs and brick type. Keep every
physical object and session in one partition across both cameras. Train on good
images only; val and test must each contain good and defective examples. Merely
having files in `/home/intel/brickcell-captures` does not establish these labels or
independent partitions. With no labeled wrist captures, collect/review them first;
automatic two-camera sorting remains blocked.

Copy `configs/mark-capture-index.example.json` to an ignored dataset directory.
Replace its example crops with reviewed crops matching the intended presentation
and add sample rows like this (paths relative to the index):

```json
{
  "camera": "plate",
  "path": "captures/clean-001.png",
  "label": "good",
  "split": "train",
  "object_id": "physical-brick-001",
  "session_id": "capture-session-001",
  "view_id": "front",
  "brick_type": "duplo_2x4_3011",
  "synthetic": false
}
```

The example is one row, not a sufficient dataset. Do not assign sequential frames
from the same session to different splits or treat a clear face of a marked brick
as a clean physical object. Fixed crops must fit the intended runtime views; the
current arm-held foreground-mask failures also need resolution before calibration.

On the training machine (Python 3.12, Anomalib 2.7.0, PyTorch CPU):

```sh
uv sync --locked
uv run python scripts/prepare_mark_data.py \
  --index data/marks/index.json --output data/marks-prepared-v1 --check-only
uv run python scripts/prepare_mark_data.py \
  --index data/marks/index.json --output data/marks-prepared-v1
```

Preparation validates joint object/session groups, duplicate hashes and per-camera,
per-brick-type partitions, then writes separate configs. For each camera run the
following in order, stopping immediately if any command fails:

```sh
uv run brickcell train --config data/marks-prepared-v1/plate/patchcore.yaml
uv run brickcell export --config data/marks-prepared-v1/plate/patchcore.yaml
uv run brickcell parity --config data/marks-prepared-v1/plate/patchcore.yaml
```

Repeat these three commands with `wrist` instead of `plate`. Calibration happens
inside `train`, using val only; training fails if its configured recall/clean
nonaccept constraints cannot be met. Initial pretrained backbone download needs
network access. No random-weight fallback or dependency downgrade is supported.

Once both models and thresholds are frozen and validation parity passes, evaluate
each once on its held-out test set:

```sh
uv run brickcell evaluate --config data/marks-prepared-v1/plate/patchcore.yaml
uv run brickcell evaluate --config data/marks-prepared-v1/wrist/patchcore.yaml
```

Do not tune models/thresholds on these results and reuse the same test set as an
independent final evaluation. Review reports, then bundle the two generated
`data/marks-prepared-v1/{plate,wrist}/model` directories as above. None of these
commands control physical hardware or establish Intel GPU/NPU performance.
