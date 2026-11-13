# SO-101 kinematics

`tools/so101_kinematics.py` provides reference forward kinematics, a base-expressed
geometric Jacobian, and a loader for the installed Frax 0.2 `Manipulator` API.
Frax is the intended FK/IK backend; the NumPy reference is an independent check.

## Coordinates and TCP

All lengths are metres and angles are radians from the exported CAD reference
pose. Joint order is:

| Coordinate | Exported joint | Role |
|---|---|---|
| q1 | `Revolute_1` | Base yaw |
| q2 | `Revolute_2` | Shoulder |
| q3 | `Revolute_3` | Elbow |
| q4 | `Revolute_2_5` | Wrist pitch |
| q5 | `Revolute_4` | Wrist roll |

The default endpoint is `Body3`, the wrist-roll child link. Supply `tcp_offset`
as an XYZ translation expressed in that link to evaluate the gripper-body TCP.
The TCP axes remain aligned with `Body3`. A calibrated full TCP rotation can be
supplied through Frax's `ee_offset` when a different tool frame is needed.

The Jacobian has shape **6x5**, with linear velocity first and angular velocity
second, both expressed in the base frame. For each revolute joint its column is
`[axis × (tcp_position - joint_origin); axis]`.

Keep gripper opening as a separate command when controlling a TCP fixed to the
gripper body. Add its coordinate when controlling a moving fingertip or contact
point. It does not generally provide a sixth independent rigid-body pose DOF.
Five arm coordinates cannot realize every arbitrary six-dimensional pose:
choose a reachable task, such as position plus tool-axis direction, or use
weighted least-squares IK with an explicitly accepted orientation residual.

The original export contains 17 independent CAD revolute joints, with no coupling
or joint-limit definitions. The five-joint model locks the remaining hub and
gripper joints at their CAD reference values. The reference `SO101Kinematics`
continues to use CAD coordinates; see the calibrated-coordinate adapter below.

## Per-arm calibration and CAD alignment

The merged leader and follower JSON files are the source of truth for each arm's
encoder interval and homing configuration. `tools/so101_calibration.py` reads them
directly. Input ticks are `Present_Position` after the stored servo homing offset
has been applied. **Do not subtract or add `homing_offset` again.**

The installed LeRobot 0.7 DEGREES mode computes:

```text
zero_ticks = (range_min + range_max) / 2
lerobot_degrees = (present_position - zero_ticks) * 360 / 4095
q_encoder_rad = (present_position - zero_ticks) * 2*pi / 4096
q_encoder_rad = radians(lerobot_degrees) * 4095/4096
```

The model uses encoder radians (4096 periods per turn), with explicit bridges
to LeRobot's `resolution - 1` degree convention. Simply calling `radians()` on
LeRobot degrees would introduce a small angle-scale mismatch.
The gripper remains a separate opening percentage. Wrist-roll's recorded full
encoder interval is not a measured collision or mechanical-stop limit.

The files do **not** record the geometric pose at each range midpoint, or the
sign relative to the CAD axes. `calibration/kinematics_alignment.json` explicitly
records those values as unknown. Obtain them from a measured, identifiable arm
pose and rotation directions before setting `confirmed` to true for an arm:

```text
q_cad = cad_at_zero_rad + cad_sign * q_encoder_rad
```

To obtain the missing mapping, record homed ticks in one physical pose whose five
CAD joint angles are known (for example, manually matching the displayed CAD
reference, if reachable). Determine whether increasing each joint's tick value
matches the corresponding CAD axis direction. For that recorded pose:

```text
cad_at_zero_rad = measured_cad_angles - cad_sign * ticks_to_radians(measured_ticks)
```

Do this separately for leader and follower. Encoder zero need not match the
folded CAD pose, and servo homing registers do not need changing. The alignment
file binds its mapping to a canonical SHA-256 of the corresponding calibration;
changed calibration data invalidates an older mapping.

`CalibratedSO101Kinematics("follower")` (or `"leader"`) requires this confirmed
mapping. It rotates each joint's home transform by the zero offset and sets its
axis sign, so FK, the Jacobian, and the Frax model use the same encoder-radian coordinates.
Its generated URDF contains bounded revolute joints with the per-arm calibrated
position intervals in radians. Effort and velocity are unbounded placeholders
because they were not measured; the model remains kinematics-only.

```bash
uv run --with numpy python tools/so101_calibration.py --arm follower
# After confirming geometric offsets and signs:
uv run --with numpy python tools/so101_calibration.py --arm follower \
  --export-urdf /path/to/so101-follower.urdf
uv run --with numpy python -m unittest discover -s tools -p 'test_so101_*.py'
```

While alignment is unknown, calibrated model construction and URDF export fail
explicitly. They do not replace the original ZIP or emit a guessed physical model.
`ServoCalibration` still provides conversions, position intervals, and explicit
limit validation. Conversion and FK queries never clamp inputs silently.
The calibrated Jacobian has columns with respect to encoder radians, including
the CAD-axis sign. A Jacobian with respect to ticks is that matrix multiplied
by `2*pi/4096`; the gripper does not add an arm column. For derivatives with
respect to LeRobot degrees, multiply it by `(pi/180)*(4095/4096)`.

### Recorded follower reference

`calibration/reference_poses/follower-arm-measurement.json` preserves a supplied
real-arm snapshot, with raw ticks as the primary numerical measurements. Its
reported calibrated values match the merged follower profile within rounding.
The user identifies wrist roll as approximately -90 CAD degrees, with the camera
parallel to the ground, and the gripper closed without overlap at raw tick 1339.
This observed closed point is 10 ticks above the existing calibrated minimum;
the recording does not rewrite servo limits or redefine LeRobot's opening scale.
The source photo is not stored in the repository.

```bash
~/frax/.venv/bin/python -B tools/so101_reference_pose.py
```

This validates the snapshot and generates `visualizations/so101-follower-reference.html`
and `so101-reference-fit.json`. The display approximates the two long members as
level, the wrist axis as horizontal, and uses zero base yaw only for display.
Those are visual hypotheses, not recovered camera geometry or measured CAD joint
angles. The wrist-roll reference comes from the user; all axis signs remain
unknown. The CAD gripper coordinate remains zero, retaining existing clearance
rather than claiming a measured contact-closure angle.

At raw wrist-roll tick 3208 the encoder coordinate is 101.9970703125 degrees.
The -90 CAD degree reference therefore gives two offset hypotheses:
`-191.9970703125` degrees for positive sign (equivalent to `168.0029296875`),
or `11.9970703125` degrees for negative sign. A second reference with a known
rotation direction distinguishes them. The script leaves confirmed alignment
and production URDFs untouched.

## Use

On the Intel computer, in `~/Desktop/tarot-club-junction`:

```bash
uv run --with numpy python tools/so101_kinematics.py --q 0 0 0 0 0 --tcp 0 0 0
uv run --with numpy python -m unittest discover -s tools -p 'test_so101_kinematics.py'
```

Use the installed Frax sandbox to exercise the adapter:

```bash
JAX_ENABLE_X64=1 ~/frax/.venv/bin/python tools/so101_kinematics.py --frax
```

In Python (with `tools` on the import path):

```python
import numpy as np
from so101_kinematics import SO101Kinematics

model = SO101Kinematics()
tcp = [0.0, 0.0, 0.0]  # Replace with the measured TCP offset in Body3.
q = np.zeros(5)
pose = model.fk(q, tcp)
J = model.jacobian(q, tcp)
robot = model.load_frax(tcp)
J_frax = robot.ee_jacobian(q)
```

The loader creates a temporary kinematics-only URDF, retains the five arm joints,
fixes the other CAD joints, and explicitly sets Frax's joint ordering. It does
not extract or load visual meshes, open a serial port, or send motor commands.
Frax/JAX should use 64-bit precision for the reference comparisons.
The CAD export has no inertial data; the adapter adds massless inertial elements
required by Frax's parser. This model supports FK/IK, not dynamics, force control,
or simulation of contact forces. Those require measured mass and inertia data.

## Export validation and visualisation

The `.agl` references CAD faces. Resolved link origins, rotations, axes, and meshes
come from `kinematics.json` in `SO-101.zip`. The builder checks AGL link/part and
joint topology against that data, checks mesh transforms, and compares C++ FK
with the independently parsed URDF across 21 configurations.

```bash
uv run --with fast-simplification python tools/so101_visualize.py
```

This regenerates `visualizations/so101-kinematics.html` and its validation report.
Meshes are simplified for display; kinematic transforms retain full precision.
The HTML uses pinned Three.js CDN scripts and requires WebGL. Its six main
sliders are CAD coordinates, including gripper articulation; all 17 exported
joints are also available. Slider ranges are exploratory display ranges.

`tools/so101_fk.cpp` is a dependency-free C++17 reference. It accepts an ordered
frame count followed by, per frame: parent index, 16 row-major home-transform
entries, three local-axis components, and angle. A parent of -1 denotes the root.
It prints one row-major base-to-link transform per frame. The visualisation
builder compiles and checks this implementation in a temporary directory.
