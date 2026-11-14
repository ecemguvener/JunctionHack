# Reviewing the captured brick-001 reconstruction

The current reconstruction is an exploratory point-cloud alignment. The source
captures are real, but neither composite has validated pose alignment or complete
surface coverage. A sparse surface fit is not a solid brick model or a trained
defect detector. The cameras and capture sessions must remain identified if these
images are later used in a formal dataset.

## Reproduce the offline audit

Use the Python 3.12 locked core environment on the Mac:

```bash
uv sync --locked
uv run python scripts/reconstruction_audit.py
uv run python -m http.server 18096 --bind 127.0.0.1 --directory artifacts
```

Open `http://127.0.0.1:18096/brick-001-reconstruction-audit/review.html`.
The server is only for local review. The page embeds the separately running
live camera viewer on loopback port 18094 when that tunnel is available.
The offline results load without live hardware access.

This audit expects the five saved brick-001 capture folders with their existing
`merge-report.json` and ASCII `brick-001-rough-merged.ply` files. Those artifacts
are intentionally excluded from Git; a fresh checkout does not contain the
personal captured dataset. The exporter stores top-camera vertices followed by
plate-camera vertices; the audit checks those counts before selecting plate-only
points. The audit preserves inputs and writes derived outputs to a separate folder.

## Compare the methods

- Earlier composite: trimmed one-way fitting with no held-out points.
- Cleaned both-camera composite: largest connected surface group, bidirectional
  fitting and held-out point diagnostics, while retaining approximate camera alignment.
- Cleaned plate-only composite: the same pose-alignment procedure using only plate
  points. This avoids the cross-camera alignment error within the input scans.
- Single plate scan: avoids cross-pose registration entirely, but shows only the
  surfaces captured from that viewpoint.

Each registration tries 24 proper PCA rotation seeds. Eighty percent of points
participate in fitting; the remaining 20% measure nearest-neighbor interpolation
in both directions. Candidate selection uses fitting points only. A point holdout
does not establish independent pose accuracy, and near-equal seed scores can
indicate ambiguity. All current results retain `accepted: false`.

The known repeat underside is evaluated as a repeat, but excluded from the
composites to avoid implying additional surface coverage. No mesh is inferred
over missing surfaces. Cluster filtering can remove legitimate disconnected
surfaces; inspect retention counts and original photos before reusing the output.

## Next engineering work

The existing sparse, symmetric scans do not constrain an automatic full solid
reconstruction reliably. Improvements would require denser raw depth sequences,
better masking and stable object-pose tracking or independent correspondences.
Those are planned hardware/data improvements, not executed validations.

For the original visual-defect task, RGB PatchCore training can proceed without
a finished 3D model once sufficient independent clean bricks and explicit
train/validation/test groups exist. Keep depth as a separate measurement until
its useful geometry thresholds are calibrated and independently evaluated.
