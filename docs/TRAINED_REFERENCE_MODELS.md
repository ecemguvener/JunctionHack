# Download the trained reference models

The actual trained red, blue, yellow and white plate models are transferred to
Intel separately from Git. The `anomaly` branch contains the inference code and
this runbook. No wrist checkpoint exists in this handoff.

Archive: `trained-plate-references-20261115-v1.tar.gz` (87,162,118 bytes).
SHA256: `7b06e4402c1a7b93c39dfacd7e93f6f8f97574418781c0357192f4b639d99407`.

From a teammate machine with the documented Intel SSH access:

```sh
git switch anomaly
git pull --ff-only origin anomaly
mkdir -p artifacts
scp intel:/home/intel/brickcell-deploy/artifacts/trained-plate-references-20261115-v1.tar.gz artifacts/
scp intel:/home/intel/brickcell-deploy/artifacts/trained-plate-references-20261115-v1.tar.gz.sha256 artifacts/
cd artifacts
shasum -a 256 -c trained-plate-references-20261115-v1.tar.gz.sha256
tar -xzf trained-plate-references-20261115-v1.tar.gz
cd ..
uv sync --locked
```

On Intel itself, use the already extracted directory:
`/home/intel/brickcell-deploy/artifacts/trained-plate-references-20261115-v1`.

Each color directory contains `reference.ckpt`, `pilot.json`, `references.json`,
`model.xml`, `model.bin`, `model.json`, `metadata.json`, `package_versions.json`
and `export_parity.json`. The root `manifest.json` records SHA256 hashes. Original
training images are not included or needed to restore these fitted models.

| Directory | Clean reference images used in the saved checkpoint |
|---|---:|
| `red` | 2 |
| `blue` | 12 |
| `yellow` | 24 |
| `white` | 36 |

Use the model matching the brick color. Inputs must be cropped RGB images of the
brick, not full camera frames; the models were fitted with per-image crops and
color-specific surface preprocessing. The shipped surface-mask limitations for
arm-held and silhouette-touching marks remain unresolved.

Run checkpoint scoring with the existing script, replacing the image path:

```sh
uv run python scripts/appearance_pilot.py score \
  --pilot artifacts/trained-plate-references-20261115-v1/yellow \
  --images /absolute/path/to/brick-crop.png
```

This writes `comparison.json` and heatmaps under the chosen color directory.
Alternatively run OpenVINO CPU directly with the exported runtime sidecar and
the project's preprocessing:

```sh
uv run python - artifacts/trained-plate-references-20261115-v1/yellow/model.xml /absolute/path/to/brick-crop.png <<'PY'
import sys
from pathlib import Path
from brickcell.detector.runtime import OpenVINODetector
from brickcell.inspection.images import read_rgb
detector = OpenVINODetector(Path(sys.argv[1]))
prediction = detector.predict(read_rgb(Path(sys.argv[2])))
print({'raw_anomaly_distance': prediction.score, 'threshold': None,
       'decision': 'UNCALIBRATED', 'latency_ms': prediction.latency_ms})
PY
```

These are trained reference models, not calibrated yes/no exports. Their saved
checkpoints have postprocessing disabled, no threshold and no learned binary
`pred_label`. Merely exporting them cannot recover an absent threshold. They
remain excluded from `deploy-cell` automatic sorting, and this handoff does not
change `configs/mark-viewer.json` or the missing-model preflight. See
[MARK_MODEL_HANDOFF.md](MARK_MODEL_HANDOFF.md) for paired calibrated deployment.

## Live OpenVINO reference viewer

`configs/reference-viewer.json` wires all four trained plate models into the
existing camera viewer using explicit `reference_rgb` mode. It supports
variable review crops without changing a model's calibration contract. Fresh
frame identity, expected camera serial, crop bounds and surface quality are
still checked. Valid reference results are `UNCALIBRATED`; failed views remain
REINSPECT/ERROR, and `automatic_sorting_allowed` is always false.

From the repository root after obtaining the model archive:

```sh
uv sync --locked
uv run python scripts/patchcore_depth_viewer.py \
  --config configs/reference-viewer.json \
  --source http://127.0.0.1:8094 --port 8098 \
  --events artifacts/reference-viewer/events.jsonl
```

On Intel, the existing minimal OpenVINO environment can run this without
PyTorch/Anomalib inference dependencies using
`PYTHONPATH=src .venv/bin/python scripts/patchcore_depth_viewer.py` with the same
arguments. Do not reinstall or modify the running robot environment. The viewer
reads the already running SDK snapshot service; it never opens camera devices or
commands the arm. Camera URLs in this config target Intel loopback; use SSH
forwards and an adjusted copy of the config if running on another machine.

The deployed isolated source snapshot currently runs on Intel loopback port 8098,
under `/home/intel/brickcell-deploy/artifacts/reference-viewer-runtime-v1`.
Its `viewer.log` and `events.jsonl` contain operational output and inspection
evidence. It uses the existing `.venv/bin/python` and a separate `src` snapshot;
the teammate's deployment source and existing services were not replaced.

Open the viewer through an SSH tunnel:

```sh
ssh -N -L 18098:127.0.0.1:8098 intel
```

Then visit <http://127.0.0.1:18098/>. Use the matching color card and enter a
current brick crop as `x0,y0,x1,y1`. Crops apply only until restart. For the
observed yellow brick at launch, `[89,290,191,363]` produced fresh CPU inference
and a heatmap. This crop is specific to that scene; review it if objects move.
Each API response also records a JSONL event when `--events` is configured,
including frame identity, model version, crop, score and decision state.

Export parity compares PyTorch and OpenVINO CPU on one fitting reference per
color; it is a numerical export check, not a held-out defect test or a physical
cell validation. Exact export errors and package versions accompany each model.
