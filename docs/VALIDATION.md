# Executed validation — 13 November 2026

## PatchCore RGB/depth viewer integration — 14 November 2026

Implemented `RGBDInspector` with separate modality models and explicit serial,
crop and depth-encoding contracts. RGB keeps the existing image preprocessing;
depth uses fixed 0.15–1.5 m encoding into three repeated uint8 channels, with
invalid values zeroed. Inference is gated on frame identity/freshness, aligned
shapes, RGB quality and valid depth. Synthetic models are refused; missing
models/calibration cannot produce GOOD. Both calibrated definite channels are
required for a combined per-camera decision. No cross-camera acceptance, object
correspondence, wrist integration or controller sorting was added.

Extended the Intel SDK service with `/snapshot/top` and `/snapshot/plate`,
bundling RGB JPEG and uint16 depth PNG from one aligned frameset, with frame ID,
serial, software receive time, depth scale and intrinsics. Verified live
640×480 `I;16` depth decoding from both D415s at ages 0.034/0.024 s during the
probe. Different cameras are still not hardware synchronized. Existing point,
RGB and provisional merged endpoints remain available.

Started `scripts/patchcore_depth_viewer.py` on Mac loopback port 18097 against
the forwarded SDK service on 18094. Verified both API paths and RGB/depth input
previews, explicit REINSPECT with missing crop, crop application, and plate
valid-depth fraction 0.9946 for the diagnostic crop `[320,30,410,125]` (not a
calibrated object crop). No trained real RGB/depth model was configured; only a
synthetic exported local model was found and was not used. Real model inference,
depth training, threshold calibration and real defect accuracy remain untested.

Executed `uv sync --locked`, `uv run --locked pytest -q`: **94 passed, 5 optional
integration tests skipped**; Ruff lint/format checks passed for `src tests
scripts`. New injected tests cover distinct modality inputs, fixed encoding,
synthetic/contract rejection, missing thresholds, stale frames, missing depth,
camera mismatch, decisions and lossless depth snapshot decoding. Live viewer
JavaScript parsed successfully and the HTML was opened in the Mac browser;
automated visual interaction remains unavailable. All operations used CPU/core
dependencies on Mac and the separate existing SDK environment on Intel.

## Brick-002 opposite side and collection pause — 14 November 2026

Saved fresh RGB previews/subsampled point packets under ignored
`artifacts/brick-002-other-side-review/`. The plate view shows studs in profile
on the right, versus left in the prior side view, supporting a distinct opposite
side observation. Both views are unobstructed. Recorded the same object/session
and source hashes, with no model decision, partition or camera transform.
Five clean poses are now visually reviewed for brick-002 (studs up, studs
toward plate, hollow underside, side, opposite side); this is not verified
complete geometry or five independent objects. Updated all existing review
pages to a saved-photo gallery and next-brick studs-up diagram. No core source
changed.

## Brick-002 accepted narrow side — 14 November 2026

Saved fresh RGB previews and subsampled point packets under ignored
`artifacts/brick-002-side-review/`. Visual review confirms the upright blue
brick's narrow curved side faces the plate camera, with studs in profile and
no hand occlusion. Retained the same brick-002/session group and source hashes;
assigned no dataset partition or model decision and applied no old camera
transform. Updated the existing HTML guide and review pages with the saved
photo and an opposite-side target requiring a 180-degree turn. No core source
changed during capture.

## Brick-002 accepted back and next-pose guide — 14 November 2026

Saved fresh RGB previews and subsampled point packets under ignored
`artifacts/brick-002-back-review/`. Visual review confirms the blue brick's
hollow underside and circular tubes facing the plate camera, unobstructed.
Corrected the guide's simplified smooth-back wording to match the actual
underside. Retained brick-002 and its existing session group, with no model
decision, partition or cross-camera transform applied. Updated the existing
HTML pages to show this saved photo and a 90-degree clockwise side-view target.
The served guide returned HTTP 200; no core source changed.

## Brick-002 orientation guide and accepted studs-facing view — 14 November 2026

Captured RGB previews and subsampled point packets under ignored
`artifacts/brick-002-orientation-check/`. Plate RGB review confirms all four
studs facing the camera, without hand occlusion. Recorded this as brick-002's
human-labeled clean studs-facing pose, in the existing session group, with no
partition or detector decision. Created a local HTML guide containing native
2D SVG orientation diagrams, the actual accepted photo, a live plate preview
and explicit half-turn instructions for the next back view. The diagrams are
schematic, not measured brick geometry. Guide HTML returned HTTP 200 and inline
JavaScript syntax passed; opened it in the Mac browser. No core source changed.

## Brick-002 repeat collection review — 14 November 2026

Repeat-02 showed an empty board in both views and was excluded from the clean
brick collection. Repeat-03, saved under ignored
`artifacts/brick-002-repeat-03-review/`, visibly shows the blue brick lying flat
with studs upward. Recorded it as a clean repeat, not the requested studs-
toward-plate pose or additional surface coverage. Saved source hashes and point
receive times (0.121 s skew), retaining object `brick-002` and the same capture
session group. No model prediction, dataset split or camera transform was
applied. No source code changed for these capture reviews.

## Confirmed clean brick-002 review capture — 14 November 2026

After the operator confirmed a new clean physical brick, saved fresh top/plate
JPEG previews and subsampled SDK clouds to ignored
`artifacts/brick-002-clean-front-review/`. Visual review shows a blue square
four-stud brick lying flat, with the red brick-001 outside the board. Recorded
object `brick-002`, session `brick-002-clean-20261114-a`, human GOOD label,
source SHA-256 hashes, camera serials and source timestamps in a capture record.
No model decision or dataset partition was assigned. Both image views are
unobstructed. RGB previews and point packets were requested separately; point
receive timestamps differ by 1.67 s, not hardware synchronization. JPEG previews
are not native lossless camera captures, and clouds are subsampled rather than
full depth images. The plate framing differs from earlier calibration, so no
old camera transform was applied. No source code changed for this collection.

## Extensive saved-scan reconstruction audit — 14 November 2026

Implemented offline geometry diagnostics in `brickcell.reconstruction` and
executed `scripts/reconstruction_audit.py` on the five saved brick-001 pose
clouds. Preserved raw captures and hashes; retained exact Python/package
versions, cleanup counts, fitted transforms, random seed and split counts in
ignored `artifacts/brick-001-reconstruction-audit/audit-report.json`.

Compared largest-connected-cluster cleanup (9 mm DBSCAN radius, three-point
minimum) for two-camera clouds and plate-only clouds. Cleanup can discard real
disconnected surfaces; counts are reported rather than treated as ground truth.
Aligned each pose against the upright reference using 24 proper PCA rotation
seeds and up to 50 trimmed ICP iterations. Fitting uses 80% of source and target
points, with the other 20% withheld from fitting. Candidate selection uses
bidirectional fitting distances; held-out point distances are reported
separately. This random point holdout is NOT independent object, session or
pose validation, and no inspection dataset split was created or changed.

Both-camera held-out symmetric mean distances were 8.57 mm narrow, 10.18 mm
back, 9.29 mm repeat underside and 9.32 mm initial lying pose. Plate-only
distances were 8.72, 6.30, 5.71 and 6.00 mm respectively. Plate-only performs
better on three of four fits but not the narrow pose. Similar scores from
different seeds demonstrate unresolved orientation ambiguity. These distances
are sparse nearest-neighbor interpolation diagnostics, not physical accuracy.

Exported cleaned source clouds, per-pose aligned clouds, two exploratory
composites, comparison plots and an interactive HTML review. The new composites
contain 1,302 two-camera or 869 plate-only points, excluding the repeated
underside from their coverage. All results remain `accepted: false`; no solid
mesh, missing surface, calibrated defect result or automatic robot action is
inferred. The plate-only view avoids the approximate cross-camera transform
within each source scan but does not resolve cross-pose registration ambiguity.

Added tests for proper rigid transforms, isolated outlier removal, invalid
coordinates, bidirectional partial-surface metrics and recovery of a known
asymmetric pose with fitting/holdout accounting. All five focused tests passed.
The report/UI distinguish real scans from unverified registration. Viewer HTML
syntax and served assets are checked separately; browser automation is
unavailable, so rendered interaction has not been automated.

Final checks for this change: **85 passed, 5 optional integration tests skipped**;
Ruff lint and format checks on `src tests scripts`, and diff whitespace checks
passed. The generated viewer JavaScript parsed successfully; HTML, JSON report,
three selectable PLY assets and both plotted comparisons returned HTTP 200
from the loopback review server. Opened the new HTML in the Mac default browser.

## Brick-001 repeat underside and multi-pose candidate — 14 November 2026

Captured `artifacts/brick-001-other-side-review/` following the operator's
other-side cue. RGB review shows the broad underside again; recorded pose
`underside-repeat-operator-other-side` rather than claiming opposite-side
coverage. Existing extraction retained 129 top and 285 plate points.

Added and executed `scripts/align_pose_preview.py` on upright, narrow, back
and repeat-underside rough clouds. Tested 24 proper PCA-axis seeds per pair,
30 iterations of trimmed nearest-neighbor ICP against the first pose, retaining
80% of correspondences for fitting. Best trimmed fitting distances were 4.21,
4.37 and 3.91 mm; alternative seeds have similar scores. These are in-sample
fit measures, not physical accuracy, and the visible cloud remains noisy.
Saved candidate PLY, source-pose-colored plot and rigid transforms under ignored
`artifacts/brick-001-composite-review/`, marked `accepted: false`. No complete
mesh, verified hidden surfaces, calibrated geometry or trained model is claimed.
Raw pose clouds remain separate and preserved for later reconstruction work.

## Brick-001 back/underside pose — 14 November 2026

Saved RGB and SDK point packets under ignored `artifacts/brick-001-back-review/`.
RGB review shows the broad underside with circular tubes facing the plate
camera, consistent with the operator's back-view label. Processed top ROI
`[242,211,278,271]` and plate ROI `[360,59,452,136]` using the existing red
selection, radius filter and provisional transform. Exported a rough two-view
cloud, plot and pose report. Kept this pose separate from earlier orientations;
the cloud does not imply measured cavity geometry, complete coverage or a
cross-pose reconstruction. No source code changed; processing and diff
whitespace checks completed successfully.

## Brick-001 narrow-side pose — 14 November 2026

Saved fresh RGB/SDK point packets in ignored `artifacts/brick-001-narrow-review/`.
Visual review shows the narrow face obliquely toward the plate camera, with
part of the adjacent long face visible. Software receive timestamps differ by
0.121 s; this is not hardware synchronization. Reused the provisional camera
transform and capture-specific ROIs: top `[224,216,280,258]`, plate
`[379,62,441,143]`. Red selection plus the 60 mm review radius retained 129 top
and 147 plate points. Exported a two-camera rough PLY, source-colored preview
and report under the same folder. This pose remains separate from previous
object poses; no cross-pose reconstruction, training or defect test occurred.
No source code changed; existing processing completed successfully and diff
whitespace validation passed.

## Brick-001 upright pose — 14 November 2026

Captured RGB/point packets under `artifacts/brick-001-rotated-review/`. RGB
review shows the same red brick standing upright with studs toward the plate
camera. Kept this distinct pose separate from the earlier lying-down capture;
camera registration does not register a rotated object into its previous pose.
Exported a rough two-camera upright PLY and preview using the existing review
transform. Added configurable ROI/pose fields to `merge_brick_preview.py` and
a review-only 60 mm median-centered radius to remove distant stray depths.
This filter assumes a small brick and may discard valid surfaces; raw packets
are preserved. No new camera calibration, training or defect decision occurred.

## Brick-001 coarse merged preview — 14 November 2026

At the operator's request to proceed with coarse visualization, captured
brick-001 RGB and point packets from both running live endpoints under ignored
`artifacts/brick-001-merged-review/`. Applied the latest provisional transform
without refitting. Capture-specific pixel ROIs and red-color selection retained
136 top and 195 plate points; exported `brick-001-rough-merged.ply`, a source-
colored plot and report. Sparse visible surfaces and stray depth points remain;
this is not a complete solid reconstruction or defect measurement.

Added optional `--transform` to the live review server, with camera-serial and
freshness checks for a third merged panel. Reports retain `provisional: true`
and software frame skew. The two cameras are not hardware synchronized; keep
the object still. Coarse review transforms are not used by robot/detector code.

## Clearer labelled-board plate view — 14 November 2026

Retrieved RGB and point packets in `artifacts/tilted-board-review/` after the
operator changed the setup. The board appears flat and the plate perspective
is higher, so do not assume the previously estimated camera extrinsics remain
valid. Used visually reviewed pattern-boundary pixels in A/B/C/D order;
corrected the initial top selections to exclude the gray tape border.
Provisional corner residuals are 15.78, 4.31, 7.73 and 6.48 mm. Top plane RMS
is 1.35 mm; plate plane RMS is 0.84 mm. Saved input corner JSON, packets,
`provisional-transform.json` (`accepted: false`) and a provisional scene PLY.
This is improved coarse registration, not independent validation or calibrated
millimetre defect measurement. No live transform was applied. No source code
changed during this capture/analysis run.

## Labelled board registration review — 14 November 2026

Retrieved fresh RGB/point packets for the A/B/C/D labelled board under ignored
`artifacts/labelled-board-review/`. Labels permit manual verification of the
physical corner order. Extended the pilot script to accept explicit directory
and corresponding pixel JSON, validate shapes, and retain pixels/SDK serials.
Refitting yielded corner residuals 12.47, 22.14, 6.57 and 35.46 mm; registration
remains `accepted: false`. Estimated board edge lengths differ between cameras
by up to about 36 mm despite small plane residuals (1.31 mm top, 0.74 mm plate).
Small plane residuals alone therefore do not establish metric registration.
No cause was isolated; manual pixel selection, grazing view, uncorrected
distortion and depth error remain possible contributors. The new provisional
cloud is for review only; no live or robot transform was applied. A tilted
board observation is the next planned check.

## Provisional orange-board outline registration — 14 November 2026

After the operator repositioned the board, both full outlines were visible.
Fresh RGB and SDK point packets were saved in ignored
`artifacts/orange-corner-review-02/`. SIFT with a 0.7 ratio test found zero
matches. Testing the eight possible outline orientations with normalized
red-green image correlation produced a best coefficient of 0.664 versus
0.155 or lower for alternatives; this is supporting evidence, not calibration.

Executed capture-specific `scripts/board_alignment_pilot.py`: manually chosen
corresponding outline pixels, board depth-plane RANSAC at 3 mm, pinhole ray-plane
intersection, then a rigid transform. Top plane: 1,415/1,508 inliers, 1.36 mm
RMS. Plate plane: 2,300/2,306 inliers, 0.63 mm RMS. Corresponding corner residuals
are 15.98, 8.03, 14.57 and 9.86 mm. Exported `provisional-transform.json` with
`accepted: false` and `provisional-merged-scene.ply`. This is a rough scene
alignment only, with no independent validation, distortion correction or
defect-scale accuracy. It is not applied to the live viewer or robot control.

## Orange-board color corner candidates — 14 November 2026

Fetched fresh RGB frames from both live-depth review endpoints and executed
`scripts/orange_corners.py` on explicit manually reviewed target polygons.
Normalized `(red-green)/(red+green+blue+1)` contrast plus Shi-Tomasi detected
49 top-view and 64 plate-view corner candidates. Saved numbered overlays,
contrast channels and candidate pixel coordinates under ignored
`artifacts/orange-corner-review/`. Visual review shows pattern intersections
and extra edge candidates. The plate image clips the right target boundary.
These are unordered candidates, not cross-camera correspondences or evidence
of a successful calibration; no transform or merged cloud was generated.
The polygons are capture-specific. An irregular physical pattern can provide
shared landmarks if correspondences are verified; standard equal-square
checkerboard object coordinates are inappropriate for this target.

## Live D415 point-cloud review — 14 November 2026

Added and deployed `scripts/live_depth.py` as a separate Intel SDK process on
loopback port 8094, forwarded to Mac loopback port 18094. It owns both D415s
while the RGB bench on 8092 is stopped; the wrist helper and teammate UI are
untouched. Each camera produces aligned RGB/depth at requested 640×480/15 fps;
the review API subsamples every fourth pixel and retains depths 0.15–1.5 m.
SDK point coordinates are displayed in separate rotatable canvas views; they
are not registered or merged. Packets include camera serial, intrinsics, depth
scale and software timestamps; absent/stale (>3 s) packets return HTTP 503.
No robot or detector integration exists in this viewer. Browser visual layout
and interaction have not been verified by automation.

The new physical orange/gray checkerboard was visible in both RGB previews,
but exhaustive OpenCV SB searches for 3–8 inner corners per dimension detected
no full pattern in either view. Square counts/size are pending operator input.
No camera transform was estimated. The board appears to have uneven squares;
this observation alone does not establish the cause of failed detection.

## Brick-001 matched raised/clean experiment — 14 November 2026

Captured the operator-confirmed raised change and then its clean reference
with plate SDK serial `243622060450`, 640×480 aligned RGB/depth. Captures are
`brick-001-raised-DGMCha` and `brick-001-matched-clean-GhBs4I`, retained on Intel
under `/home/intel/brickcell-captures/` and locally in ignored artifacts with
SDK metadata and PLYs. RGB review shows gray material among the studs in the
first and its absence in the second; placement appears closely matched.

Direct raw-depth comparison in manually selected ROI `[410,302,444,324]`
has 432 paired valid pixels (57.8%); median signed difference is 0 mm and
95th-percentile absolute difference is approximately 6 mm. Front control ROI
`[385,344,439,370]` has 947 paired valid pixels (67.5%), median signed difference
0 mm, and 95th-percentile absolute difference 1 mm. This is an uncalibrated
diagnostic, with no registration, repeated noise baseline, defect threshold or
accuracy claim. Missing depth and pose variation remain confounders. Keep both
captures grouped as the same physical `brick-001` in any future partitioning.
Restored the RGB bench and verified its HTTP status endpoint after capture.

## Brick-region 3D extraction — 14 November 2026

Executed `scripts/compare_depth_scans.py` on the existing clean and later
operator-labeled defective plate captures with manually reviewed pixel ROI
`382 262 437 311`. Exported separate vertex-only PLYs, RGB crops, a static 3D
comparison and a report retaining source PLY hashes under ignored artifacts.
The clean ROI contains 1,985 points with 73.65% usable depth pixels; the later
ROI contains 2,139 points with 79.37% usable depth pixels. RGB review shows
the studs and later added material at the top. These are partial visible-surface
scans; ROI edges can contain background. No table-plane segmentation or pose
registration was performed. Intrinsics are fixed to the recorded plate SDK
640×480 probe; the parser supports that SDK PLY layout only.

Correction to the earlier conversational conclusion: a zero median depth
difference across a broad image region does NOT establish that a defect is
appearance-only. Missing depths, background pixels, pose changes and sensor
noise must be handled before attributing any difference to a defect. No depth
defect threshold or accuracy has been measured. Physical object/session identity
for these older generic clean/defective folders is unconfirmed, so these are
review artifacts and must not enter formal train/val/test partitions yet.

Software verification: `uv run --locked pytest -q` passed 80 tests with five
optional integration tests skipped; Ruff lint and format checks passed on
`src tests scripts`. No new camera or robot access occurred for this extraction.

## RealSense depth probe — 14 November 2026

Stopped the temporary RGB bench so it released both D415 devices, then ran the
hardware-only `scripts/realsense_depth_capture.py` on Intel with transient
`pyrealsense2` 2.59.1. SDK serials `350122060653` (overhead/top placement) and
`243622060450` (plate/side placement) both produced aligned 640×480 RGB and
16-bit depth frames, 1 mm depth scale metadata, human-viewable depth previews,
and RGB-colored PLY point clouds. The captures were real and non-synthetic;
depth was not passed into PatchCore or used for a defect decision. The SDK
serials differ from the earlier V4L2 by-id labels, so future hardware runs must
enumerate and record the SDK identity. The RGB bench must be restarted after
the depth process exits.

## Timed five-face capture — 14 November 2026

Ran the operator-assisted sequence on Intel for `red-brick-001`, session
`clean-sequence-001`, with faces `front right back left top` at 30-second
intervals using the side D415. All five fresh bursts completed and the sequence
record was written at `/tmp/brickcell-multicam/data/face-capture-runs/` on Intel.
Each result was REINSPECT because no real defect model/calibration is configured;
this is expected. Face labels are operator supplied and require image review.
No robot movement or automatic defect decision occurred.

Added `scripts/capture_faces.py`, a timed operator-assisted face sequence. It
requires explicit object/session IDs, face order, camera and label, captures at
the configured interval (default 30 seconds), and never commands the robot or
makes defect decisions.

## Real-image reference-only PatchCore pilot — 14 November 2026

Taped capture `21d2b3935a874fb4b60a229b5a70dc00` clearly shows the gray patch.
The frozen `red-fixed-v1` checkpoint scored it **2.485419**, versus fitting
reference **2.230126** and clean repeat **2.273086**. The brick was turned
upright and partly clipped relative to the lying-down fitting pose. This
confounds pose/framing and patch effects; no defect-specific accuracy or
calibrated decision is claimed. Maps and comparison JSON were saved.

After recentering, visually verified clean capture
`1c47f4b1c90d4892af2d20705b5bab9b` at crop `[280,360,365,465]` (85×105).
Fitted a separate clean-only reference checkpoint at
`artifacts/pilots/red-fixed-v1/`, with explicit object/session/crop provenance.
Scored its fitting image **2.230126** and a fresh same-pose clean repeat
**2.273086**. Visually verified both contain the clean brick. This is a
same-object repeatability diagnostic, not validation or test accuracy. A taped
capture matching this new camera pose is still needed. Tests remained
**80 passed / 5 optional integration tests skipped**, with Ruff checks passing.

Clean recapture `39dbc2ac3ab1440397e6ed270440176b` failed visual review:
the camera viewpoint had changed, the crop contained tabletop and the brick
was partially outside the full-frame right edge. Moved the image to review
locally and remotely, excluded it from training, and did not fit a new model.
The taped lying-down capture still lacks a matched clean reference.

Capture `cc5b203826114bf780baf55b033d7f4b` visibly shows a rectangular gray
patch on the lying-down red brick's front face. Saved and visually checked the
80×95 raw crop `[265,350,345,445]`. This pose/crop differs from the original
upright clean reference; a matched clean capture is required before attributing
any score difference to the patch. No new accuracy result was claimed.

Follow-up capture `e8f03aa228bc421bbf0938aac35df303` scored **2.327533**
with the same frozen reference checkpoint. The brick had rotated relative to
the clean reference and a dark mark was still not visually identifiable. Kept
this image in the review collection, not verified defective data. No threshold
or defect-detection success is inferred from its distance.

Added `scripts/appearance_pilot.py` for an explicitly uncalibrated same-object
appearance experiment, separate from the formal train/val/test workflow. Fitted
Anomalib 2.7.0 PatchCore on the real clean red-brick crop using CPU, ResNet-18,
128×128 input, layer2/layer3, seed 42 and 5% coreset. One fitting reference
produced 11 coreset features. Saved the checkpoint hash, image hashes, versions
and no threshold in ignored `artifacts/pilots/red-reference-v1/`.

Reloaded the checkpoint and executed real image inference: fitting reference
distance **2.060604**, later unverified capture **2.407176**. Saved anomaly maps
and overlays. No dark mark was visually identifiable in the later capture, so
it was moved from the labeled defective collection into a review collection.
Its report retains the human label and adds an UNVERIFIED review note. It is
not a verified defective test image. Pose/lighting variation can explain score
changes; these two scores establish neither defect detection nor accuracy.

This pilot has no calibration, no held-out evaluation, and no GOOD/DEFECTIVE
decision or robot integration. Both captures retain the same physical object
ID; they must not be split across formal train/val/test partitions. The normal
dataset validation and production calibration paths were not weakened.

Executed `uv run --locked pytest -q`: **80 passed, 5 optional integration tests
skipped**; `uv run --locked ruff check src tests scripts` and format check passed.
The new pilot additionally completed actual fit/checkpoint-reload/inference.

## First real brick capture — 14 November 2026

After the operator placed a clean blue brick in the side D415 view, visually
verified it and set crop `[270,310,366,434]` in the 640×480 source. Added scoped
single-camera collection without relaxing three-view acceptance requirements.
Captured a fresh side-only burst and saved one unannotated 96×124 PNG as
`blue-brick-001` / `side-session-001`, human label GOOD. Retrieved and visually
verified the saved crop. Maximum adjacent-frame motion was 2.54 grayscale levels.
The detector result remained REINSPECT because no real calibration exists.

Executed tests: **80 passed, 5 optional integration tests skipped**. Ruff lint,
format and diff whitespace checks passed. This is one human-labeled example,
not a trained/evaluated dataset or evidence of real defect accuracy. The object
ID must remain unchanged if this same brick is later marked.

## Three-camera bench and burst fusion — 14 November 2026

Deployed the loopback-only bench on Intel port 8092, forwarded locally to 18092.
Both D415 feeds and the upright wrist feed returned fresh images. D415s reported
640×480; the wrist source reported 1920×1080. Each preview JPEG endpoint returned
HTTP 200 and JPEG bytes. Added a raw `/frame.jpg` endpoint to the existing wrist
helper so focus text/rectangles do not enter collected images; increased its
focus median window from 7 to 15 samples. No arm service was accessed.

The bench reports a 15-frame median focus measure and median absolute deviation
as jitter, using light Gaussian filtering for focus measurement only. One final
status sample reported jitter of 2.06% overhead, 13.92% side and 1.27% wrist.
These are scene-dependent diagnostics, not measured defect accuracy or a claim
that sensor noise has been eliminated.

Added object-scoped 10-frame fresh bursts, explicit crop/operator-confirmation
requirements, capture timestamps, a provisional motion gate and per-camera
calibrated decision fusion. Unit tests verify isolated spikes, sustained defects,
missing/uncalibrated views, wrong/duplicate identities, same-batch RGB capture,
and that a human GOOD label cannot become an uncalibrated GOOD prediction.
The deployed service refused an unconfirmed capture with HTTP 400.

Executed `.venv/bin/python -m pytest -q`: **79 passed, 5 optional integration tests
skipped**. Ruff lint and format checks on `src tests scripts` passed. Browser
automation was unavailable, so visual browser layout was not verified; live HTTP
status and all three image endpoints were verified directly.

No real defect model is configured. Real-model burst inference, calibration of
the temporal/motion rules and real three-view defect accuracy remain untested.
The combined live result cannot be GOOD until all required models/views qualify.
Camera timestamps are receive times, not hardware synchronization, and object
correspondence still requires operator confirmation. No training photos were
captured from an unidentified scene during this verification. See `MULTICAMERA.md`.

## Intel camera connectivity — 14 November 2026

Connected to the pinned Intel SSH host and enumerated USB/video devices. Both
D415s (`318123021715`, `344223025431`) returned 640×480 BGR frames through
OpenCV/V4L2 using stable by-id paths; JPEGs were saved for visual review. The
Sonix device was busy with an existing focus-assistance process. Its metrics
reported 1920×1080 at 30 fps and one annotated JPEG was retrieved through the
existing loopback MJPEG endpoint, without opening the device again. These
snapshots were copied to ignored `artifacts/camera-check/brickcell-camera-check/`.

Visual review confirmed overhead, side and upside-down wrist views respectively.
No brick defect classification, depth capture, camera calibration, real-image
training or robot movement was performed. No remote packages/services were
installed or stopped. A local SSH tunnel exposes the existing focus preview at
`127.0.0.1:18091`; its focus score is not a defect score.

## Real-camera capture tooling — 14 November 2026

Added explicit OpenCV camera capture with BGR-to-RGB conversion, checked pixel
crops, unique PNGs, object/session/view identity and quality sidecars. Optional
inspection uses the existing service and refuses synthetic-trained models.
Added `configs/patchcore-real.yaml` and `docs/REAL_CAMERA.md` for collection,
partitioning, real-image training and new-capture inspection.

Executed `.venv/bin/python -m pytest -q`: **73 passed, 5 skipped** (four optional
real-model tests and one live-simulator test). Ruff lint and format checks on
`src tests scripts` passed. The capture module's `--help` ran successfully.
Camera tests use injected frames: RGB channel/crop correctness, failure cleanup,
dataset path validation and capture-to-manifest identity preservation.

No physical camera capture or real-image training was performed. An attempted
read-only station check failed because the `intel` SSH alias did not resolve;
no remote configuration was changed. Camera device selection, OS permissions,
focus/exposure, capture blocking behavior and quality thresholds require checks
on the selected hardware. The capture utility is on demand, not a bounded-latency
robot controller camera adapter. Existing simulated accuracy is not evidence of
real defect accuracy.

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

## Offline data/training workbench — 2026-11-14

Built on `codex/data-training-workbench`, initially from `6408aad` and fast-forwarded
to the team's non-overlapping alignment commit `3526b29` before publication.
No changes to arm-api, calibration files, or hardware-control scripts are part of
this workbench diff. No hardware connection or cloud training was performed.

Executed on the existing native M4 / Python 3.12 environment:

- Core pytest: **103 passed, 5 skipped** (34 workbench tests). Default skips are four
  model tests and one live-simulator test. Ruff lint/format checks passed.
- Separately ran the existing live MuJoCo/OpenVINO sorting integration test against
  an isolated local simulator on ports 18083/19093: **1 passed, 105 deselected,
  71.32 seconds** at that invocation. This verifies simulated sorting after UI
  changes, not physical success. The isolated test server was stopped afterward.
- Generated a new offline fixture, imported its 36 inspection images through CSV,
  preserved family/color metadata, and reviewed its three synchronized 30-frame
  demonstration-format episodes. Strict readiness passed for these software
  fixtures. They do not represent a learned grasp or realistic defect benchmark.
- Reviewed the existing six MuJoCo teacher episodes (2,250 frames) and the 48-image
  two-bowl camera dataset. Inspection remained smoke-trainable but lacked named
  five-face metadata. All six policy episodes failed the conservative joint-limit
  readiness gate. No archived data or limits were altered to conceal findings.
- Compared directly with the composed simulator model: largest wrist-flex observed
  excess per episode ranges from **0.000158 to 0.003070 rad**; commanded excess is
  about **0.00000274 rad**. These small boundary/precision/soft-constraint effects
  are not evidence of collision or an explanation for the earlier failed ACT
  rollouts. The report gives the exact frame indices, bounds, maximum excess and
  explicit numerical tolerance (1e-6 rad) for review.
- Used the emitted inspection config to train PatchCore on the imported synthetic
  fixture successfully. Used the emitted policy command with `--steps 2` to run
  an ACT CPU optimizer/checkpoint-reload smoke on the format fixture successfully
  (reload maximum difference 0). Neither measurement is a real-task success claim.
  The known PyAV/OpenCV duplicate AVFoundation warnings appeared in the isolated
  policy environment; the command completed.
- Browser verification: next-frame stepping advanced both camera frames and joint
  values; Play/Pause advanced to frame 123; selecting the validation episode reset
  playback to its own frame zero. Image label/split filters returned six defective
  validation images; selecting the absent top face returned zero. Final responsive
  layout and compact metadata cards were inspected. The simulator visibly showed
  synthetic/scripted mode and one fixed view with **0/5 confirmed named faces**.
- Wheel and source archive built successfully. The first sandbox build could not
  resolve PyPI; approved native execution completed the build. No dependency or
  lockfile changes were necessary.
- Extracted the source archive into a clean temporary directory, installed with
  `uv sync --locked` into a new environment, and successfully ran fixture generation,
  CSV import, and both strict review commands. The archive includes workbench
  modules, configs and handoff docs; datasets, artifacts and unrelated `ssh.md`
  are excluded. The wheel includes all eight workbench modules.

Representative executed commands (from repo root; output directories must be new):

```bash
uv run pytest
uv run ruff check src tests scripts policy
uv run ruff format --check src tests scripts policy
uv run python -m brickcell.workbench demo --output data/workbench-example
uv run python -m brickcell.workbench import-inspection --csv data/workbench-example/inspection/capture.csv --output data/workbench-example/inspection/imported.json
uv run python -m brickcell.workbench inspection --manifest data/workbench-example/inspection/imported.json --output artifacts/workbench/example-inspection --strict
uv run python -m brickcell.workbench policy --manifest data/workbench-example/policy/manifest.json --limits configs/workbench-sim-limits.json --output artifacts/workbench/example-policy --strict
uv run python -m brickcell train --config artifacts/workbench/example-inspection/patchcore-review.yaml
PYTHONPATH=src TORCH_HOME="$PWD/.cache/torch" policy/.venv/bin/python policy/train_act.py --manifest data/workbench-example/policy/manifest.json --output artifacts/workbench/example-policy/training-run --steps 2
BRICKCELL_MODEL_CONFIG=configs/patchcore-sim.yaml BRICKCELL_QUALITY_CONFIG=configs/inspection-sim.yaml BRICKCELL_SIMULATION_URL=http://127.0.0.1:18083 uv run pytest -m simulation -q
uv build --out-dir artifacts/workbench-dist
```

Real recording-format conversion, sensor synchronization, physical coordinate and
limit profiles, camera visibility, five-face robot presentations and reliable
learned motion still need station validation. Static reports suit small pilot
batches; they embed every preview frame and are not a large-dataset streaming UI.
See `WORKBENCH.md` and `STATION_COLLECTION.md` for the reproducible workflow and
operator handoff.
