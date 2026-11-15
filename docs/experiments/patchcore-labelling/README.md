# PatchCore training and colour labelling experiment

Executed on Intel CPU with locked Python 3.12 and Anomalib 2.7.0. See
[training details](TRAINING.md) and [validation evidence](../../VALIDATION.md).

These scripts preserve the exact experiment source. They are staged into ignored
artifact directories before execution; paths resolve from the staged script, not
from this documentation directory. Existing outputs are not overwritten.

| Source snapshot | Stage as |
|---|---|
| train_jack.py | artifacts/patchcore-jack-clean-20261115-v1/train.py |
| prepare_earlier650.py | artifacts/patchcore-combined-earlier650-v1/prepare.py |
| train_combined.py | artifacts/patchcore-combined-earlier650-v1/train.py |
| audit_calibration.py | artifacts/patchcore-calibration-audit-v1/audit.py |
| train_plate_rawgood.py | artifacts/patchcore-plate-rawgood-690-v1/train.py |
| train_plate_calibrated.py | artifacts/patchcore-plate-calibrated-707-v1/train.py |

Run preparation from the project containing the original capture records and
images. Transfer the prepared artifact collection to Intel, preserving its paths.
Run staged training/audit scripts with `uv run --locked python <staged-script>`.
Use the repository's locked CPU environment and cached pretrained backbone.
The preparation script verifies all original hashes; training verifies all inputs,
rejects duplicate hashes and preserves camera/object/session provenance.

The earlier 650 include 114 accepted clean references: 102 plate plus 12 top.
The plate fit combines its 102 with Jack's 570 clean references; top is separate.
The other 536 are retained and excluded with recorded reasons. Thresholds remain
null: current saved candidates do not provide independent labelled validation.
Diagnostic inference is not calibration evidence or independent defect accuracy.

Colour annotations are separate from defect labels. The colour summary records
720 target labels across the two collections; 476 frames with unspecified targets
and 24 rejected empty scenes remain unassigned. Labels use capture identity and
sampled visual review, not exhaustive object annotation. Per-image annotations,
source images, model weights, package snapshots and run logs remain under ignored
artifacts locally and on Intel; none are committed to Git.
