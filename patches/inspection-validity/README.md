# Inspection validity patch for the deployed controller

The Plate presentation loop currently passes `None` for depth and expected depth,
then treats crop area or gripper opening as evidence of a held brick. A textured
crop without a brick can complete inspection. This patch requires usable aligned
metric depth at the projected object distance before either detector runs or a
frame counts toward scan completion.

## Scope and source

`hardware_cell.py` is present on the Intel laptop but absent from GitHub main at
`3720e9b`. This PR therefore provides a helper module and a narrow integration
patch, not a replacement copy of the teammate's entire motion stack. **Merging
this PR alone does not update the deployed controller.** The controller owner
must integrate the patch with their current source, review it and validate it at
the station before deployment.

`baseline.json` identifies the exact source read on 2026-11-14 and SHA-256 of
both the original and patched file. The fixture contains verbatim inspection
and presentation-method excerpts from that source. It is test data, not a
standalone arm controller. Its imports are never executed in the tests; AST
selection loads only the inspector definitions and presentation method, which
receive fake cameras, poses, telemetry and arm methods. The dark-mark algorithm
itself is executed unchanged.

## What changes

- Require uint8 RGB, an in-bounds integer crop, shape-aligned floating-point depth
  in metres, a positive finite expected camera distance and a valid depth band.
- Require at least 45 depth-supported pixels, at least 10% crop support, and at
  least 50 inspectable interior pixels after the scorer's existing colour
  exclusions and 7-pixel erosion. These initial gates need station validation.
- Return an explicit invalid reason before dark-mark/PatchCore scoring when
  evidence is missing. The invalid result's numeric zero is a placeholder,
  **never a GOOD classification**; callers must check `valid_view`.
- Add `depth_supported` to the inspection result. Pass the actual Plate depth and
  projected camera distance into inspection during the wrist sweep.
- Remove the dry-run, gripper-opening and total-crop-area substitutes for evidence.
  Count only depth-supported valid frames toward the existing three-frame minimum.
- Rename the evidence flag to describe depth support instead of claiming that a
  grasp has been verified. Preserve the existing failure/recovery branch.

The patch does not change the physical arm controller, poses, speeds, joint
limits, IK, dark-mark threshold/algorithm, wrist sampling schedule, destination
selection or failure-recovery motion calls. The rejection condition changes, so
existing recovery can now run more often; the motion owner must review that
behaviour before any live use. This PR does not launch hardware or install services.

## Offline setup and tests

From this branch, with Python 3.12 and uv:

```bash
uv sync --locked
uv run pytest tests/test_inspection_validity.py -q
uv run pytest --ignore=tests/test_block_inspector.py -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

Git must be available: the tests apply the actual unified patch to the frozen
excerpt in a temporary directory. No SSH, camera, robot, MuJoCo, model weights or
network access is used by these tests after environment installation.

`test_block_inspector.py` currently imports optional MuJoCo during collection;
it is excluded above because the core environment does not contain that stack.
Four model tests and one live-simulation test skip without their assets/services.

## Prepare a local review copy

Have the controller owner supply their current `hardware_cell.py` as a local
file. Keep it under ignored `artifacts/`; do not replace files in the active
Intel deployment. For the exact audited revision:

```bash
python3 scripts/prepare_inspection_validity_patch.py \
  --source artifacts/hardware_cell.original.py \
  --output artifacts/inspection-validity/hardware_cell.review.py
```

This checks the input hash, applies the patch in a temporary directory, checks
the resulting hash and syntax, and creates a new output file. It does not import
or execute that file. Unknown revisions and existing output files are refused.
If the owner has newer changes, rebase the small patch onto those changes and
rerun the offline tests; do not overwrite their source or bypass the hash check.
The integrated controller must have `src/brickcell/inspection/validity.py`
available in its environment. The helper is in the wheel; the patch, script and
fixtures are available in the checkout/source archive.

## Evidence limits and next station check

Depth within the predicted crop is **not proof of brick identity, a successful
grasp, or full face coverage**. Another object, a hand or a jaw at the expected
range can still contaminate the crop. Aligned shape does not independently prove
camera calibration. Missing depth now blocks this path; an RGB-only path would
need a separately validated presence check.

The existing scorer excludes blue plate-like and purple jaw-like pixels. Blue
or purple bricks can therefore be rejected for insufficient inspectable surface,
rather than silently classified GOOD. Colour handling, blur/occlusion checks,
per-object tracking, frame-age validation at acquisition and classifier threshold
calibration remain separate work. Existing stale-frame/duplicate-frame rejection
is regression-tested here, not rewritten.

Before a supervised arm run, manually present a clean and marked brick at the
intended Plate-camera pose and capture paired RGB/depth. Include empty jaws,
background-only crops, blue/purple bricks, occlusion and depth dropouts. Inspect
the predicted crop, expected distance and foreground mask. Adjust evidence gates
only against those labelled captures, then check recovery with the motion owner.
The current fixed depth band is preserved; it may include distractors and needs
measurement at the actual presentation pose. No real-world accuracy or physical
sorting success is established by these artificial fixtures.
