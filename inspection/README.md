# Your first inspection lesson

## Anomalib: train on real photos

`train_patchcore.py` adds real Anomalib PatchCore training. Prepare the dataset
below, then check it without installing machine learning dependencies:

```sh
python3 inspection/train_patchcore.py --data datasets/bricks --check-only
```

The check creates empty dataset folders if needed. Put your photos in them.
For this training script, half of `test/good` and `test/bad` is reserved for
validation/threshold tuning, and the remainder for testing (seed 42). Thus
these folders need multiple examples of each class. The separate
`calibration/good` folder described later is for future custom calibration;
this Anomalib script does not use it. Do not tune on the final test results.

Use a separate environment with a Python version supported by your installed
Anomalib release. The Mac's current default Python is 3.9; the documented Intel
server environment uses Python 3.12. After activating a compatible environment:

```sh
python -m pip install anomalib
python inspection/train_patchcore.py --data datasets/bricks
```

On the Intel server, follow `hardware.md` for the PyTorch XPU installation
before installing Anomalib. The script initially uses CPU for portability;
GPU acceleration is a separate verification step. The first training run can
download pretrained backbone weights and requires network access.

PatchCore builds a memory of normal image features. New image patches that
look unlike that memory receive larger anomaly scores. Marked examples help
evaluate and calibrate decisions; they are not the normal training set.
Without pixel masks, image-level evaluation is useful, but localization
accuracy cannot be measured against ground truth.

This integration has been checked for syntax and missing-dataset handling.
Training has not been run: this checkout has no real image dataset or installed
Anomalib environment. API reference:
https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/folder.py

Run this from your Desktop/JunctionHack terminal:

```sh
python3 inspection/demo.py
```

Open `inspection/output/report.html` in your browser. The left image is the
part; the right image is its difference heatmap. Clean should say **accept**;
marked should say **reject**. This runs with Python's standard library.

## What you just built

1. **Learn normal:** average 20 synthetic clean images, pixel by pixel.
2. **Calibrate:** measure 10 separate clean images and place the threshold above
   their largest score, with a five grayscale-level margin for this lesson.
3. **Inspect:** compare two new test images with the learned normal image.
4. **Score:** average the largest 1% of pixel differences. This makes a small
   dark mark visible in the image-level score.
5. **Decide:** a score at or above the threshold means reject.
6. **Record:** save the model, scores, decisions and planned bowls as JSON.

The score is a measure of visual difference, **not a defect probability**.
The report demonstrates actual computation on synthetic pixels. It is not
PatchCore, Anomalib, camera inference, or a physical sorting demonstration.
Nothing in this script connects to the robot or sends movement commands.

## Try changing it

In `demo.py`, change the marked pixels from `35` to `170`, then rerun. The
mark gets fainter and its score drops. Change them to `178`: the mark may blend
into normal variation. That is why defect size, lighting and contrast matter.

The simple mean baseline assumes every image is aligned. If the camera or
part moves, a difference can come from movement instead of damage. A learned
feature method such as PatchCore is the next model to evaluate on real images;
it still needs controlled capture and held-out evaluation.

## Real data comes next

Use the same brick type, camera position, background and lighting. Begin with
roughly 30–50 clean training photos, separate clean calibration photos, and
separate clean/Sharpie-marked test photos. These counts are a starting point,
not evidence of accuracy. Keep different physical bricks in different splits
where possible; adjacent frames from one video are not independent examples.

```text
datasets/bricks/
  train/good/
  calibration/good/
  test/good/
  test/bad/
```

For a physical workflow, the remaining pieces are: capture a D415 color frame,
crop the inspection region, run a trained image model, validate the threshold
with real defects, and then execute a taught pick/place sequence through armd.
Add an uncertain/manual-review outcome before unattended sorting. Missing
frames, model failures and motion failures must stop the sequence.

The existing `arm-api/armd/fake.py` simulates servo behavior, not images,
collisions, grasp success or whether the part reached a bowl. Real movement
needs verified joint limits and taught poses; do not invent bowl coordinates.
See `hardware.md` and `arm-api.md` before hardware work.
