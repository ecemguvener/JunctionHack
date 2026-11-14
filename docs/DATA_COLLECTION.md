# Real brick data and evaluation plan

No real challenge images have been used yet. Synthetic fixture metrics measure plumbing, not generalization.

Start with one clearly defined brick family, camera, background and inspection pose. Photograph roughly 30–50 distinct good objects and 20–30 distinct defective objects if available, with 4–8 useful views per object. Include cracks/chips, damaged/missing studs, contamination, deformation and visually subtle defects; record `defect_type`. The object-level count matters more than hundreds of adjacent frames. These are initial collection targets, not a promise of statistical sufficiency.

Use several independent capture sessions with controlled variation in illumination, distance, rotation, exposure and realistic backgrounds. Reserve separate physical objects and sessions for calibration and final test before training. A practical initial allocation is roughly 60% of independent all-good groups for fitting, 20% for calibration, 20% for test. Divide defective groups between calibration/test. Ensure each defect type and expected camera view occurs in both without sharing physical objects. Do not label camera/color/pose variation as a defect unless that is part of the task specification.

The splitter joins records sharing **either** physical object ID or session ID, including transitive links. Capture each reserved group in its own sessions; repeatedly photographing all objects in every session makes the entire dataset one inseparable group. If that happens, collect independent held-out groups rather than weakening the leakage check. Use globally unique object IDs (not `object1` restarted each day). An intact brick later damaged retains its original ID and remains in one partition.

Keep focus/exposure fixed where feasible. Include separate quality-rejection examples: absent objects, gripper-only views, clipped objects, occlusion, motion blur, glare and out-of-ROI objects. These test the quality gate and reinspection behavior; do not count them as confidently good just because their anomaly score is small. Manually review/cull unusable views before fitting/calibrating. The current quality heuristics require a roughly uniform background and do not guarantee object presence or visibility.

Example hand-authored manifest:

```json
{
  "schema_version": 1,
  "samples": [
    {
      "image_id": "capture-2026-11-13-000123",
      "path": "raw/defective/day2/brick-101/front/frame-001.png",
      "label": "defective",
      "split": "val",
      "object_id": "brick-101",
      "session_id": "day2",
      "view_id": "wrist-front",
      "defect_type": "chipped_corner",
      "mask_path": "masks/capture-2026-11-13-000123.png",
      "defect_regions": [[50, 30, 90, 70]],
      "synthetic": false
    }
  ]
}
```

This one-row example illustrates fields, not a valid complete training dataset. A complete baseline manifest needs good train, good/defective val, and good/defective test partitions. Mask dimensions must equal source image dimensions. Pixel boxes use source coordinates. Store captures unrotated when attaching pixel annotations; the RGB loader applies EXIF orientation, while this image-level baseline does not train from the masks/boxes.

Freeze the train/val/test manifest and retain dataset hashes with the trained model. Review calibration errors and set a defect-recall target with the team; max-F1 is only an initial software baseline. Use calibration data to select model, image size, threshold, uncertainty margin and quality settings. After freezing these, evaluate once on held-out test data and report counts, errors by object/defect/view, abstentions and latency. Keep data collected afterward as a new test set when iterating. Bootstrapping or uncertainty intervals should resample physical objects, not correlated frames.

Treat the camera change from Mac development images to the wrist/D415/QHD station as a domain shift. Recalibrate on the actual cell; do not carry a synthetic threshold into physical operation.
