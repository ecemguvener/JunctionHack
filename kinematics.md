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

The export contains 17 independent CAD revolute joints, with no coupling or
joint-limit definitions. The five-joint model locks the remaining hub and gripper
joints at their CAD reference values. Joint signs, motor zero offsets, TCP,
physical limits, and the actual hub couplings still require hardware calibration
before using planned configurations as motor commands.

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
