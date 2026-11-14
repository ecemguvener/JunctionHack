# Calibrated SO-101 kinematic models

`so101-follower-calibrated.urdf` and `so101-leader-calibrated.urdf` use each arm's
midpoint-zero encoder radians, five measured direction signs, and recorded
zero offsets. Position bounds come from that arm's LeRobot calibration.

The source CAD archive remains unchanged. Regenerate through
`CalibratedSO101Kinematics.export_arm_urdf`; use `load_frax` for the installed
Frax 0.2 interface. See [the alignment report](../calibration/ALIGNMENT_REPORT.md)
and [kinematics guide](../kinematics.md).

These are kinematics-only, massless models. Their effort and velocity limits are
unbounded placeholders, not measured control limits. The TCP defaults to the
Body3 wrist-roll frame; a measured tool offset must be supplied for tip control.
The visual reference matches were made by eye, with unquantified physical error.
Numerical validation reports demonstrate software consistency only.

The leader elbow sign was accepted explicitly by the user after coupled motion;
the original isolation failures and the acceptance are retained with the raw
measurement evidence. No automatic physical pose test has been executed.
