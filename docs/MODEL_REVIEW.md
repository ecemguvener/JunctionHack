# Saved-model review — 2026-11-15

The strongest real marked-image candidate found is
`artifacts/pilots/red-marks-foreground-mask-v2`. It is distinct from
`red-marks-multiview-v1`, the red model in the earlier four-color transfer.
Its corrected foreground mask places the strongest response on the gray patch
in the saved marked-brick example. It still has no calibrated threshold or
independent real accuracy evaluation and cannot currently authorize sorting.

Executed a frozen-checkpoint audit of all 16 real reference pilots, not just the
three blue/yellow/white multiview models. Every checkpoint and fitting-image hash
verified. All models loaded on PyTorch CPU. Reran 63 saved development/fitting
queries; no inference errors, and every query with a saved comparison exactly
reproduced its prior score. No training, threshold selection or test-set rerun
occurred. Source metadata and package versions remain in ignored artifacts.

| Candidate under `artifacts/pilots/` | Fitting images | Audit queries | Existing evidence and limitation |
|---|---:|---:|---|
| `red-marks-foreground-mask-v2` | 1 | 2 | Best localized real marked example; clean 2.498976, marked 3.018789; threshold absent |
| `red-marks-multiview-v1/model` | 2 | 2 | Clean 2.498634, marked 3.037864; marked maximum near studs/background; incomplete extra-reference session metadata |
| `red-marks-foreground-v1` | 1 | 2 | Clean 2.498976, marked 3.303174; marked maximum near studs/background |
| `red-marks-local-foreground-v1` | 1 | 2 | Clean 1.943396, marked 2.617894; marked maximum near studs/background |
| `red-marks-surface-fine-v1` | 1 | 2 | Clean 2.146855, marked 3.285108; marked maximum near studs/background |
| `red-marks-surface-v1` | 1 | 2 | Clean 2.397419, marked 3.110886; marked maximum near studs/background |
| `red-fixed-v1` | 1 | 3 | Fitting reference, clean repeat and tape example; diagnostic only |
| `red-reference-v1` | 1 | 3 | Fitting reference and unverified/tape-candidate queries; diagnostic only |
| `clean-block-002-multiview-20261115-v2/model` | 12 | 1 | Blue model loaded/scored on fitting reference; no saved independent marked comparison |
| `clean-block-003-multiview-20261115-v1/model` | 24 | 1 | Earlier yellow model loaded/scored on fitting reference; no saved independent marked comparison |
| `clean-block-003-multiview-20261115-v2/model` | 24 | 6 | Yellow clean-repeat distances 2.393213–2.518535; same physical fitting object; later real mark masking failures |
| `live-white-clean-multiview-20261115-v2/model` | 36 | 1 | White model loaded/scored on fitting reference; no independent marked comparison |
| `live-white-clean-20261115-v1/model` | 6 | 6 | Clean repeats 1.870613–2.068170; prior visible X partly excluded by mask |
| `plate-marks-clean-v1/model` | 6 | 6 | Blue clean repeats only, 1.093070–1.268870 |
| `plate-marks-surface-v1/model` | 6 | 6 | Blue clean repeats 1.337964–1.389529; prior synthetic line/background perturbation diagnostics |
| `brick-002-line-v1/model` | 6 | 18 | Top-camera clean-repeat/artificial-line diagnostic; not plate/wrist deployment |

Scores are raw distances, not probabilities. Distances from different models
are not comparable performance rankings. A higher marked-image distance can be
caused by background/stud response rather than the mark itself; localization
review matters. The two red examples above are already-used development images,
not an independent validation or untouched test set.

## Corrected red candidate export

Rechecked its existing uncompressed OpenVINO CPU export against both development
queries. Model/weights/checkpoint hashes matched the saved export report.
Maximum absolute score difference from PyTorch: **0.000001430511474609375**,
below the 0.001 tolerance. This verifies export fidelity, not defect accuracy.

- Checkpoint SHA256:
  `d34abd868494219a49696b08078c21d4308fed4cd937e75b9715191018a7eaa4`
- OpenVINO XML:
  `artifacts/pilots/red-marks-foreground-mask-v2/export-check/weights/openvino/reference-only.xml`
- Localization review:
  `artifacts/pilots/red-marks-foreground-mask-v2/localization-review.png`

The marked peak is at source pixel `(48.4,64.6)` on the gray patch, compared with
the earlier surface model's `(33.1,7.1)` near studs/background. The clean peak
still lies near a boundary. This is the candidate to investigate next for red
bricks; it is not a general all-color model or an arm-held inspection validation.

## Why a model may look perfect in the results

The saved `artifacts/models/patchcore-sim/test_report.json` reports 6/6 clean and
6/6 defective images correctly classified, with F1/AUROC 1.0 and a frozen
threshold. Its metadata and report explicitly mark the data **synthetic**. This
is valid synthetic pipeline evidence, not physical-brick accuracy. The test set
was not rerun during this review.

The separate `inspection/output/tool-evaluation/anomalib/results.json` is also
explicitly a synthetic smoke test (AUROC 1.0, F1 approximately 0.8). Its reported
PyTorch 2.15.0 environment differs from the locked core PyTorch 2.10.0 environment;
it was not loaded or promoted as a physical sorting artifact. The adjacent
`inspection/output/model.json` describes a grayscale mean baseline, not one of
the real PatchCore plate pilots.

## Sorting readiness

All 16 real pilots have `threshold: null`. None is a trained wrist model. The
default PatchCore deployment policy requires calibrated real plate **and wrist** RGB
models with matching sensor contracts, valid fresh views and complete scan
evidence. A frozen image model that responds to one known mark does not supply
those missing integration requirements. The explicit legacy Lab operating rule
is documented separately in [HEURISTIC_SORTER.md](HEURISTIC_SORTER.md). No arm command, camera-wall publisher
change, model replacement or gate bypass occurred during this review.

Full machine-readable rerun evidence is saved locally under ignored
`artifacts/model-review-20261115/audit.json` and `red-candidate-ir.json`.
