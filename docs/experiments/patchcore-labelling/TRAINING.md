# Earlier 650 + Jack clean reference models on Intel

Remote project: `/home/intel/brickcell-training/jack-clean-20261115-v1`.
Run: `artifacts/patchcore-combined-earlier650-v1` within that project.

`inventory.json` accounts for exactly 650 original RGB images: 96 guided captures,
6 calibration candidates, 476 presentation-recording frames, 72 earlier captures.
Original RGB bytes are in `raw/`; source records are preserved in `records/`.
Generated crops, depth files, and heatmaps do not count toward the 650.

114 are accepted clean references (human label good and baseline accepted).
102 plate-camera images join Jack's 570 clean plate images for a new 672-image
plate-camera reference model. The other 12 top-camera images fit a separate
model. All 1,220 earlier and Jack RGB hashes are unique. Source labels and formal
training eligibility remain unchanged; accepted reference-only captures are used
only for an uncalibrated reference experiment, consistent with earlier pilots.

The other 536 earlier images are retained and excluded from fitting:

- 476 unlabeled presentation-recording frames
- 30 marked-brick images (6 guided + 24 earlier)
- 24 rejected empty-scene captures
- 6 reserved calibration candidates with unconfirmed physical identity

No good labels are inferred for unlabeled frames. Excluded marked views of a
training brick are not represented as independent held-out validation. All
accepted reference data is explicitly train-only. No calibration/test partition,
threshold or accuracy measurement is fabricated. Original captures include
plate-resting scenes while Jack's images are arm-held; this full-frame combined
reference fit has not been validated for deployed inspection.

Training runs on Intel CPU with locked Python 3.12, Anomalib 2.7.0, Torch
2.10.0+cpu, pretrained ResNet18 layer2/layer3, 128x128 images, 0.05 coreset,
3 neighbors, seed 42 and four threads. Cached backbone weights are used offline.
Each output retains manifest, exact configuration, source provenance, checkpoint
hash, and a checkpoint reload/finite fitting-image inference check. The run
retains exact package versions, all-input hash audit, training script hash and
training log. `summary.json` is written only after both fits complete.

This does not update or restart any robot/camera service, install models into
runtime, calibrate thresholds, perform OpenVINO export/parity, or authorize
sorting. Both models remain uncalibrated; automatic sorting is false.
