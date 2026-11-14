# Read-only inspection evidence audit

This tool diagnoses the existing deployment. It does not run a detector, replace
the dashboard, publish overlays, or command the robot. It uses only Python's
standard library. Camera snapshots are new observations, not asserted to be the
frames used by a detector. The report intentionally does not draw old boxes on them.

## Capture and replay

On the Intel laptop, from a checkout containing this script:

```bash
python3 scripts/audit_station.py capture --samples 4 --interval 1 \
  --output artifacts/station-audit/capture-01
```

This GETs the existing dashboard (:8096), camera wall (:8093), aligned Top/Plate
snapshots (:8094) and arm **health only** (:8765). It opens no camera/serial device.
Services need to be running already. Missing sources are recorded as findings.
Base URLs can be overridden with `--dashboard`, `--depth`, `--camera-wall`, and
`--arm`; only loopback HTTP URLs are accepted, including local SSH tunnels.

Alternatively, using an already verified SSH alias, capture without copying or
installing anything remotely:

```bash
mkdir -p artifacts/station-audit
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes intel \
  'python3 - capture --stdout --samples 4 --interval 1' \
  < scripts/audit_station.py > artifacts/station-audit/intel-capture.json
python3 scripts/audit_station.py replay \
  --bundle artifacts/station-audit/intel-capture.json \
  --output artifacts/station-audit/review-01
```

Use the repo's pinned host keys when configuring SSH; do not bypass verification.
Open the output `index.html` directly. `bundle.json` retains captured RGB/depth,
source identities/times, response hashes and payloads; `summary.json` contains
machine-readable findings. Output directories are never overwritten. Captures
may contain people or station details: keep them in ignored artifacts and share
only with the team. The optional `--run-report PATH` retains an existing report,
labeling its success claims as self-reported, never independently verified.

## What was observed on 14 November 2026

Read-only SSH inspection found Physical AI Studio serving on :7860 and the
dashboard, camera wall, depth service and armd running. The follower was online;
the leader was offline with a missing serial-device path at that instant.
These are observations, not persistent configuration guarantees.

The running `/home/intel/brickcell-deploy` contains newer `hardware_cell.py` and
`TwinDashboardReporter` code than GitHub main at 3720e9b. Do not overwrite it with
this branch. The observed `deploy-cell` command had no `--openvino-config`.

Source inspection found:

- Default detection uses Lab blackhat/line geometry and a threshold of 16.
  It is a heuristic, not Anomalib. Calibration evidence was not found for that threshold.
- Plate-camera scores are displayed with a projected whole-object Top-camera
  box. This is not a cross-camera localization of the actual mark pixels.
- The publisher repeats cached payloads every 1.8 seconds. Dashboard TTL is
  refreshed on receipt, while acquisition timestamp/frame/camera provenance is
  absent. A connected overlay is not proof of a new inference.
- Part-category IDs are reused for dashboard object IDs; they do not establish
  identity of individual physical bricks of the same shape.
- The spin inspection passes no depth image. Its foreground-pixel count cannot
  serve as evidence of a depth-verified grasp in that path.

The audit captured four unique Top and four unique Plate frames. One anomaly
payload repeated with score 58.7704 and box [289,145,339,199] while its age reset
from 1.057 to 0.262 seconds. Later samples changed focus and cleared anomalies.
This supports the heartbeat finding, not an accuracy measurement. The saved
deployment report claimed seven cycles (one good, six defective); correct
physical grasps, destinations and classifications were not independently verified.

## Next change to coordinate with the deployment owner

Have the publisher retain the exact scored image/crop, capture timestamp, camera
serial/frame ID, physical object ID, method/model and threshold version. Keep
local mark boxes separate from projected object boxes. Expire inference evidence
by acquisition time; a transport heartbeat must not refresh it. Display the
mark on its source image until a validated cross-camera mapping exists.

`brickcell.integration.evidence.provenance_errors` supplies a small, tested
proposed binding check. It is not installed in the running controller and is
not a motion authorization gate. Threshold validation, visibility/face coverage,
grasp verification and physical outcome checking still belong to their owners.

Before tuning thresholds, collect labeled clean/marked captures under the actual
inspection pose and evaluate false rejects and misses on independent objects.
Do not infer that a low heuristic score proves a visible, correctly held good brick.
