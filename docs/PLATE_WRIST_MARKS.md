# Plate and wrist RGB mark inspection

The selected mark-anomaly inputs are **plate RGB and wrist RGB only**. Top and
depth are excluded from the default mark viewer and real-camera mark inspector.
The top camera can still support scene/brick localization outside anomaly scoring. Existing 3D/depth utilities stay
available separately. The same Anomalib 2.7.0 PatchCore implementation/backbone is
used for both cameras; each camera has a separate clean-feature memory bank,
normalization/preprocessing contract, crop and validation-calibrated threshold.

## Why one architecture is enough

PatchCore extracts pretrained visual patch features, builds a normal-image memory
bank and scores deviations. Mark appearance is the same task for both RGB cameras;
there is no reason to implement two unrelated architectures. Camera viewpoint,
scale, illumination, distortion, fingers and background create different normal
image distributions. Separate banks avoid automatically treating a patch as
normal merely because it occurs in the other camera's view. Separate thresholds
account for different raw-distance distributions. These are two instances of one
algorithm, not a claim that a single mixed checkpoint is impossible. A mixed-bank
alternative should be compared on independent validation bricks, per camera/type,
at matched false-positive rates before adopting it. No such comparison was run.

References: [PatchCore paper](https://www.amazon.science/publications/towards-total-recall-in-industrial-anomaly-detection)
and [Anomalib 2.7 implementation](https://anomalib.readthedocs.io/en/lib-v2.7.0/markdown/guides/reference/models/image/patchcore.html).

## Actual training performed

`artifacts/pilots/plate-marks-clean-v1/` contains an actual CPU PatchCore fit using
six accepted clean plate-camera crops from brick-002, with checkpoint reload and
scores/maps for six later clean frames. Source record and original/crop hashes,
object/session IDs and exact versions are preserved. Crop: `[332,30,414,104]`;
model input: 256x256; layers: layer2/layer3; ResNet-18; 5% coreset (306 entries).
The small source crop is enlarged for the model; resizing does not invent image
detail. The earlier accepted clean capture remains ineligible as an independent
formal train/val/test dataset. This is a reference-only experiment, not calibration.

Clean-repeat raw distances range from 1.09307 to 1.26887. No marked plate example
was scored as confirmed visible defect evidence, no threshold was fitted, and no
accuracy, recall or production acceptance is established. The older confirmed
mark was facing the top camera and plate framing changed, so it is not relabeled
as a verified plate defect. No labeled wrist training set was found locally.
The wrist bank remains untrained until appropriate captures are supplied.

Reproduce into a new directory:

```bash
uv sync --locked
uv run python scripts/train_camera_reference.py \
  --record artifacts/brick-002-full-baseline-02/capture-record.json \
  --camera plate --crop 332 30 414 104 --fit-count 6 --image-size 256 \
  --output artifacts/pilots/plate-marks-clean-v2
```

The pilot has no calibrated threshold or export sensor contract and is not
silently installed into the live viewer. It must not drive good/defective sorting.

## Build a usable two-camera dataset

Keep camera placement, focus/exposure and lighting stable. Frame the brick large
enough that the smallest target mark occupies multiple original source pixels;
a far-away 50-pixel brick cannot become detailed simply by resizing to 256 pixels.
Use actual wrist/plate images and a reviewed crop that excludes fingers/clutter.
Capture normal orientations/faces for every supported brick type and confirm the
defect is visible in each image labeled defective. Pixel masks are optional for
image-level anomaly detection; human image-level good/defective labels and exact
physical brick/session identity remain necessary for calibration/evaluation.

Copy `configs/mark-capture-index.example.json` and replace the example crops with
actual source-pixel crops. Supply rows such as this **schema example** (not a
real image or automatically assigned partition):

```json
{
  "camera": "plate",
  "path": "captures/plate-brick-001-front.png",
  "object_id": "brick-001",
  "session_id": "capture-session-a",
  "view_id": "front",
  "brick_type": "duplo-2x2",
  "label": "good",
  "split": "train",
  "synthetic": false
}
```

Assign train/val/test jointly for both cameras. All faces and both cameras of a
physical brick stay in one partition; shared capture sessions also cannot cross
partitions. Train contains only good images. For each camera and brick type,
include train/good, val/good, val/defective, test/good and test/defective. Repeated
frames from one brick/session do not count as independent objects. Do not move
partitions to improve results after seeing the test scores.

```bash
uv run python scripts/prepare_mark_data.py --index data/mark-index.json \
  --output data/marks-v1 --check-only
uv run python scripts/prepare_mark_data.py --index data/mark-index.json \
  --output data/marks-v1
uv run python -m brickcell train --config data/marks-v1/plate/patchcore.yaml
uv run python -m brickcell train --config data/marks-v1/wrist/patchcore.yaml
```

Preparation rejects joint brick/session overlap, duplicate source or cropped
hashes, defective train images, missing camera/type partitions and bad crops.
It records source/crop provenance and emits the same 256x256 architecture for
both cameras. Model training verifies its camera/crop contract against the
manifest; checkpoint reload rejects a swapped contract. Train fits only train and
calibrates on val. Do not repeatedly evaluate the test set while tuning.

Newly prepared mark configurations use `minimum_defect_recall: 1.0` and
`maximum_good_nonaccept_rate: 0.1`. Calibration requires every validation defect
to receive **DEFECTIVE**, including the uncertainty margin, while no more than
10% of validation good images receive DEFECTIVE or REINSPECT. REINSPECT does not
count as a detected mark. If no threshold meets both constraints, training stops
without approving a calibrated model. A failed run may leave its diagnostic
checkpoint; use a new output directory for a new experiment. Existing configs
without the recall target retain the original F1 calibration policy.

These are initial validation acceptance targets, not achieved camera accuracy or
a guarantee about future bricks. They prevent silently accepting a threshold
that misses labeled validation marks or flags/rechecks almost everything. Improve
capture resolution, coverage or model settings on training/validation data when
the gate fails; keep the test set untouched. All-clean reference pilots cannot
pass this gate because calibration requires both classes.

Training metadata now records actual validation decisions. Test reports retain
binary metrics and additionally distinguish defects accepted as GOOD, detected
defects, REINSPECT and ERROR; they also break down decisions by brick type, face,
physical brick and capture session. Repeated frames do not count as independent
trials, and an uncertain defect is never credited toward detection recall.

After freezing model/threshold, export each model, run parity and evaluate test
once. Track misses, false alarms and REINSPECT separately for camera, brick type,
face, mark size and mark type. An isolated dark pixel, shadows, studs, molded
lettering and real scratches must be distinguished through actual validation;
no tuned morphology or pixel smoothing is claimed to solve that automatically.

## Block-surface-only PatchCore input

An optional `surface_profile` now runs identically in the training dataset,
PyTorch validation, pilot scorer and OpenVINO runtime. It finds one colored block
inside the reviewed camera crop, fills its exterior contour so enclosed black or
gray marks are retained, replaces outside pixels with constant gray, and applies
a bounded global exposure gain estimated from colored plastic. RGB surface
texture is not smoothed or inpainted. The anomaly score becomes the maximum raw
PatchCore anomaly-map distance **inside the block mask**; outside pixels are
excluded from the reported map and score. This is a different scoring contract
and requires a newly trained memory bank and new validation threshold.

Missing, clipped, too-small or ambiguous blocks, excessive exposure/clipped glare
and insufficient surface sharpness produce an unusable view/REINSPECT. The RGB
viewer and inspection service use the surface-specific checks instead of the old
uniform-background quality heuristic when this profile is configured. Existing
models without a profile retain their preprocessing and scoring behavior.

Example profiles are `configs/surface-blue.example.json` and
`configs/surface-red.example.json`. They were inspected against the existing
blue/red reference crops; they are not calibrated profiles for every brick color.
Copy the appropriate JSON object into each camera's `surface_profile` field in
the capture index before preparation. Dataset preparation carries it into the
manifest and model config; model loading checks it against training metadata.
White/black blocks, similarly colored backgrounds, a gripper obscuring the block,
edge-touching marks and arbitrary lighting are not established supported cases.
The same PatchCore architecture can serve both cameras with separate banks;
each bank must include the supported clean colors, shapes and presented views.

Actual new reference training (CPU, Anomalib 2.7.0):

```bash
uv run python scripts/train_camera_reference.py \
  --record artifacts/brick-002-full-baseline-02/capture-record.json \
  --camera plate --crop 332 30 414 104 --fit-count 6 --image-size 256 \
  --surface-profile configs/surface-blue.example.json \
  --output artifacts/pilots/plate-marks-surface-v1
uv run python scripts/audit_surface_pilot.py \
  --pilot artifacts/pilots/plate-marks-surface-v1/model \
  --image artifacts/pilots/plate-marks-clean-v1/clean-repeat/006.png \
  --output artifacts/pilots/plate-marks-surface-v1/robustness
```

These commands require new output paths if rerun. The pilot remains uncalibrated.
Three controlled synthetic background replacements produced identical input
tensors and identical scores (1.370452). Synthetic 1/2/4-pixel black line probes
scored 3.048703/3.384203/3.366593. Global exposure and smooth-shadow probes changed
the reference distance by at most 0.010163. These are diagnostics on one image,
not natural-light robustness, real defect recall or a reason to choose a threshold.
The uncalibrated blue pilot was also exported for a CPU parity diagnostic on six
clean-repeat images: maximum PyTorch/OpenVINO difference 0.000002504. No live
model configuration was changed and no automatic sorting was authorized.

The existing red-brick gray-patch image was scored as a reference diagnostic:
clean repeat 2.397419, marked image 3.110886. However, the highest response was
near the studs, not the gray patch, so localization is **not established**. A
separate layer1/layer2 feature experiment (clean 2.146855, marked 3.285108) also
peaked near the studs; it was not adopted as a production improvement. Those
development images cannot subsequently serve as an untouched test set.

### Foreground memory and improved red-mark localization

The later `red-marks-foreground-mask-v2` development candidate improves the saved
gray-patch example: its peak moves from source pixel `(33.1, 7.1)` in dark
background near the studs to `(48.4, 64.6)` on the visible gray mark. It is still
an uncalibrated same-object development result, not independent detection accuracy.
The clean repeat scores 2.498976 and the marked example 3.018789. Comparing score
values across differently trained models does not establish improved recall.

Two changes were made:

- `foreground_memory: true` selects only feature cells with at least 50% block
  occupancy for the PatchCore normal bank. Previously even the neutral replacement
  background contributed normal descriptors. The CNN, coreset algorithm and
  inference/export graph remain standard Anomalib PatchCore. Masks describe input
  cell occupancy, not the entire receptive field of each CNN feature.
- The red example profile now explicitly uses `min_value: 40`. Previously some
  nearly black exterior pixels with a red cast entered the silhouette. Enclosed
  black and gray marks remain included by contour filling. This parameter is
  specific to the development captures and must be checked on validation views.

Enable foreground memory alongside the surface profile in a camera's capture
index, or set `foreground_memory: true` in its prepared model YAML before training.
It defaults to false to preserve existing experiments. Changing this policy
requires a new fit/calibration; checkpoint loading checks the recorded policy.

```bash
uv run python scripts/appearance_pilot.py fit \
  --references data/pilot-red-fixed/reference \
  --output artifacts/pilots/red-marks-foreground-mask-v2 \
  --object-id red-brick-001 --session-id red-fixed-clean-pilot-005 \
  --crop 280 360 365 465 --camera plate --image-size 256 \
  --surface-profile configs/surface-red.example.json --foreground-memory
uv run python scripts/check_reference_export.py \
  --pilot artifacts/pilots/red-marks-foreground-mask-v2 \
  --images data/pilot-red-fixed/queries/clean-repeat.png \
           data/pilot-red/queries/flat-marked-003.png \
  --output artifacts/pilots/red-marks-foreground-mask-v2/export-check
```

Use new output paths when rerunning. This fit kept 571/1024 feature cells from
one accepted clean reference, then selected a 27-entry coreset. The original
checkpoint reloads into standard PatchCore, and CPU OpenVINO score parity on the
two development queries differs by at most 0.000001431. No threshold is exported
or installed by the reference checker.

The candidate's three synthetic exterior-background replacements again yielded
identical scores. Synthetic exposure probes changed distance by up to 0.034254;
a smooth shadow changed it by 0.115145. These limited probes do not establish
environmental immunity. A larger clean-reference experiment, layer1-only features
and disabling map smoothing were also checked; none independently fixed the
original false peak, and they are not claimed as production improvements.

Visual evidence is saved at
`artifacts/pilots/red-marks-foreground-mask-v2/localization-review.png`, alongside
source/checkpoint hashes and machine-readable comparison, robustness and parity
reports. The clean repeat still has its highest response near a boundary, and no
pass/fail threshold has been approved. More independent real captures are needed.

### Remaining capture requirements

- Plate and wrist clean captures covering every supported brick type/color and
  presented view; include normal studs, lettering and ordinary lighting changes.
- Confirmed visible marker lines/spots from both cameras in those same view
  conditions. The saved gray patch is not a substitute for marker-line examples.
- Separate physical bricks/capture sessions for training, validation and test.
  Keep every before/after pair and both cameras of a brick in one partition;
  clean training bricks cannot later become validation/test marked bricks.
- Native close-up images: the existing blue source crop is only 82x74 pixels.
  Enlarging it does not recover finer marks. Specify the smallest mark that
  must be detected so its visibility can be checked at native resolution.

Human image-level good/marked labels are needed; pixel-by-pixel masks are not
required for the normal-only PatchCore fit. No labeled wrist training set or
independent real marked plate test set is available locally yet.

## Viewer usage

```bash
uv run python scripts/patchcore_depth_viewer.py
```

The default is now `configs/mark-viewer.json`, with only plate and wrist RGB model
slots. The plate source reads RGB from the existing snapshot packet; depth bytes
are not decoded or scored. Wrist uses the raw JPEG helper and still requires
capture metadata headers for trusted inference. Missing crops, models, thresholds
or freshness/identity metadata prevent a trusted decision. Set the exported
model under each camera's `models.rgb`; never reuse a top/depth/wrong-crop model.
The older depth viewer configuration remains accessible with an explicit
`--config configs/rgbd-viewer.json` if needed for separate depth work.

The real-camera detector loader excludes top/depth models and requires matching
real sensor metadata before compilation. The mark inspector rejects top views,
passes no depth into scoring and no longer trusts its uncalibrated Lab heuristic
as a good/defective result. These source changes were not deployed to hardware.

The existing presentation scan now requires both plate and wrist models before
entering its motion sequence. It records all seven scheduled plate observations
and three wrist observations (stops 0, 3 and 6) separately. Missing/uncertain views
prevent clean completion, and repeated frame IDs cannot establish new views. A
trusted marked view stays flagged even when later views look clean. Transport or
inspector failures still abort the scan. This gate checks the existing scheduled
observations only: it does not establish that every physical face was exposed or
that the smallest mark is resolvable. Neither motion nor physical coverage was
tested in this change.

The viewer shows per-camera results. It does not automatically identify the same
brick across cameras, prove full-face coverage, rotate an arm or perform sorting.
For a future object-level decision, use trusted same-object fresh results: any
validated defective view can reject; acceptance requires both required views
and all required presented faces to be valid and calibrated. Never average raw
anomaly distances across the camera models.
