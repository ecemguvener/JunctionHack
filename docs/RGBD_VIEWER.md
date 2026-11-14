# PatchCore RGB + depth review

The viewer can run separate real RGB and encoded-depth PatchCore models for
each D415. It shows inputs, raw anomaly distances, per-modality heatmaps and
threshold decisions beside the existing 3D viewer. It sends no robot commands
and does not establish hidden-face coverage or object correspondence.

## Start

On Intel, run the updated SDK service (`scripts/live_depth.py`) in its separate
hardware environment. It now exposes `/snapshot/top` and `/snapshot/plate`.
Each snapshot contains RGB JPEG and lossless uint16 aligned depth PNG from the
same `rs.align(color)` frameset, together with frame ID, serial, software receive
time, intrinsics and the camera's actual metre scale. Two different cameras are
still not hardware synchronized. Existing RGB/point endpoints remain available.

On the Mac with the SSH tunnel to the SDK service on loopback port 18094:

```bash
uv sync --locked
uv run python scripts/patchcore_depth_viewer.py --config configs/rgbd-viewer.json
```

Open `http://127.0.0.1:18097`. In each camera card, enter the explicit brick crop
in full-resolution 640×480 source pixels. Crop changes are runtime-only unless
saved to the JSON configuration. Existing trained model contracts must match;
changing the crop is rejected when the configured model expects another crop.
The UI always shows which RGB/depth inputs were passed to the model.

The shipped configuration has no models and no crop. It intentionally displays
REINSPECT and an explicit setup reason. A visible depth image is not evidence
of an active depth anomaly detector. At initial implementation the only exported
local OpenVINO model was synthetic-trained and is refused for real inspection.

## Configure real models

Each camera's `models` mapping accepts `rgb` and/or `depth` paths relative to the
configuration file. Models must have the existing XML/BIN hash metadata and a
matching `sensor_contract` in the model JSON, derived from actual training
provenance. For example, a real plate depth model trained for a particular crop
must declare:

```json
{
  "serial": "243622060450",
  "crop": [320, 30, 410, 125],
  "modality": "depth",
  "depth_encoding": {
    "near_m": 0.15,
    "far_m": 1.5,
    "invalid_value": 0,
    "channels": "repeat3"
  }
}
```

For RGB omit `depth_encoding` and set `modality` to `rgb`. Do not relabel a
synthetic or RGB model as a depth-trained model. The current core export flow
does not automatically populate this camera/depth contract; producing real
depth-trained assets with trustworthy provenance is subsequent training work.

## Depth representation and calibration

`brickcell.inspection.rgbd.encode_depth` is the shared deterministic encoding:
raw uint16 multiplied by the captured metre scale, valid range 0.15–1.5 m,
valid pixels mapped to 1–255 in a fixed scale, missing/out-of-range pixels set
to zero, repeated into three uint8 channels. Never normalize each frame by its
own min/max: that changes the training/inference contract. This is a 2D depth
image representation, not point-cloud feature learning or complete 3D geometry.

Depth training requires new images encoded with exactly this function, fitting
only good objects. Preserve explicit shared object/session groups across RGB
and depth manifests; calibrate modality-specific thresholds on validation and
freeze before independent test evaluation. No such training or calibration was
performed by starting this viewer. RGB and depth raw distances are not averaged.

## Result rules

- Wrong camera, malformed frame, stale/future timestamp or inference exceptions
  prevent a trusted result. Results older than three seconds are discarded.
- An explicit crop, RGB quality checks and a minimum valid-depth fraction
  (default 60%) gate inference. Valid depth alone does not prove a brick is present.
- A missing model or threshold yields REINSPECT; no substitute score is invented.
- A per-camera combined GOOD/DEFECTIVE result requires both RGB and depth channels
  to have calibrated, definite decisions. Either definite channel rejecting then
  makes that camera view DEFECTIVE; acceptance requires both GOOD.
- Heatmap colors are normalized for display per frame, not calibrated thresholds.
- This is an on-demand review server, not the bounded robot-controller inspector
  adapter. No cross-camera GOOD decision, wrist fusion or automated sorting is
  implemented here. Native inference may block a request; other requests report
  busy and cannot reuse an old image/result as current.

## Verification limits

Unit tests use injected predictions and artificial frames for contracts, separate
modality inputs, depth encoding, gating, thresholds, alignment and decoder checks.
Live SDK snapshot retrieval/PNG decoding and the unconfigured viewer path were
verified against both D415s. No real RGB/depth PatchCore accuracy, actual
depth-model inference, native OpenVINO parity or robot behavior is established.
