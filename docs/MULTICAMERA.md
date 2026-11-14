# Three-camera inspection bench

The Intel laptop has two D415 RGB streams and the upright Sonix wrist camera.
The bench shows all three together, a rolling median focus score and median
absolute deviation (jitter). Focus is not a defect score. It uses light
Gaussian filtering only for focus measurement; saved crops and detector inputs
are not blurred by that filter. Camera exposure, lighting, scene movement and
focus still change measurements. The existing wrist focus helper also uses a
15-frame median. Do not freeze a score just to make it look stable.

## Running instance

The current temporary deployment is `/tmp/brickcell-multicam` on Intel,
loopback port 8092. Local SSH forwarding exposes it at
`http://127.0.0.1:18092/`. Captures go to
`/home/intel/brickcell-captures/raw/<label>/<session>/<object>/<camera>/<batch>.png`;
reports go to the adjacent `reports/` folder. All three images share a batch ID.
The raw wrist frames come from the existing helper's `/frame.jpg`, without
the focus text or measurement rectangle. The helper retains the 180° rotation.
No robot APIs are called.

1. Hold the same brick still where its intended face is visible in each camera.
2. For each view, enter its brick-face crop in original-image pixels and click
   Apply crop. The yellow rectangle shows it. Exclude fingers and background.
3. Supply a globally unique physical brick ID, capture-session ID and human
   label. Confirm that all three crops show the same stationary brick.
4. Capture. The server waits for 10 new frames from every camera, with an
   8-second acquisition deadline. It saves one unannotated crop per camera
   plus a report. A capture failure or absent calibration cannot produce GOOD.

For initial collection, the view selector can capture one camera at a time.
Only selected cameras need crops/operator confirmation. Partial collection never
establishes three-view GOOD acceptance; full inspection still requires all three.

For a faster repeated sequence, use `scripts/capture_faces.py` on the Intel
laptop after setting one crop for the selected camera:

```bash
PYTHONPATH=/tmp/brickcell-multicam/src \
  python scripts/capture_faces.py --object-id red-brick-001 \
  --session clean-seq-a --faces front right back left top \
  --interval 30 --camera side
```

The script waits for Enter on the first face, then counts down 30 seconds before
each next face, capturing a fresh burst and recording the operator's face label.
It is a collection shortcut, not automatic face recognition. Review captures
before training; 30 seconds does not synchronize sensors or verify visibility.

The cameras are sampled independently, not hardware-triggered. Timestamps are
software receive times; batch freshness and a 0.5-second inter-view timestamp
check are not proof of sensor synchronization. Operator confirmation supplies
object correspondence; there is no automatic object detection/tracking or
verification of hidden-face coverage. Keep the object stationary. Three camera
IDs do not establish five-face coverage.

## Defect inference and fusion

There are currently **no real-camera defect models** configured. The combined
decision is REINSPECT, with per-camera uncalibrated reasons. Human labels are
used for dataset organization only, never as model predictions.

For actual decisions, collect distinct physical objects/sessions, split them
together across all views, then train/calibrate a model for each camera's crop.
See `REAL_CAMERA.md` and `DATA_COLLECTION.md`. Preserve shared object/session
IDs and partitions when deriving the three per-camera manifests. Changing a
crop, camera placement or lighting requires validation/recalibration.

In a core environment containing OpenVINO, Pillow and the other locked core
dependencies, pass `--models` with a JSON file of this form (paths are examples):

```json
{
  "overhead": {"model": "/models/overhead/model.xml", "crop": [200,120,440,360]},
  "side": {"model": "/models/side/model.xml", "crop": [200,120,440,360]},
  "wrist": {"model": "/models/wrist/model.xml", "crop": [600,300,1200,900]}
}
```

The bench rejects synthetic models, requires matching model/capture crops,
checks image quality and blocks a burst with more than 5 grayscale levels of
inter-frame motion in a 64×64 thumbnail. This motion cutoff is a provisional
heuristic; validate it on real calibration sequences.

Each camera runs its own calibrated threshold on 10 valid frames. The central
80% of those scores must consistently lie above/below its uncertainty band;
otherwise REINSPECT. Single isolated outliers are trimmed, so inspect held-out
bursts with small or intermittently visible marks before relying on this rule.
A stationary, consistently visible inspection surface is required.

- A consistently defective camera rejects the object, even if another view is clean.
- GOOD requires all three camera bursts to be valid, calibrated and GOOD.
- Missing, moving, low-quality or uncertain views prevent acceptance.
- Model/inspection exceptions produce ERROR. No automated robot sorting occurs.

Raw distances from different camera models are not averaged. The decision is
fused after per-camera calibration. Temporal smoothing and fusion themselves
need evaluation on independent real sequences, not only single-image metrics.

## Reproduce

The preview/capture service needs only OpenCV/numpy and the package source:

```bash
PYTHONPATH=src python scripts/multicam_server.py --output data/multicam
```

It binds to loopback only. Start it on the Intel laptop after verifying camera
availability; the wrist helper must provide unannotated `/frame.jpg` frames.
Only one process should own each video device. Preview dependencies remain
separate from the trained-model core environment. Stop the service before
another process opens either D415.

## D415 depth and point clouds

The top and plate cameras are depth-capable D415s. The hardware-side utility
`scripts/realsense_depth_capture.py` uses the RealSense SDK on Intel to align
depth to color and save, per camera:

- the RGB image;
- a 16-bit depth PNG in millimetres;
- a human-viewable depth preview;
- an RGB-colored `.ply` point cloud; and
- JSON metadata with depth scale and valid-depth range.

Example on the Intel laptop (after stopping the RGB bench so it releases both
D415 devices):

```bash
uv run --offline --no-project --python 3.12 \
  --with pyrealsense2 --with numpy --with pillow \
  python scripts/realsense_depth_capture.py \
  --serial 350122060653 243622060450 \
  --output /tmp/brickcell-depth --frames 30
```

The current probe produced real, non-synthetic 640×480 depth and point clouds
from both cameras. Depth is not yet fed into PatchCore: that requires a
separate depth/3D feature representation and calibration set. Keep RGB and
depth files paired by capture timestamp and object/session ID; do not mix a
depth snapshot from one pose with RGB from another. The wrist Sonix remains
RGB-only.
