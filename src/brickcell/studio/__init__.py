"""Modified Physical AI Studio — SO-101 Model-Based Pick-Inspect-Place Workbench.

Provides a zero-extra-dependency analytical robotics & architecture synthesis engine
and web studio UI for any SO-101 arm (Kinematics, 6x5 Geometric Jacobians, 5-DOF IK,
Published CAD Dynamics & Gravity Compensation, Multi-Camera Feeds, Leader-Arm Teaching,
and One-Click Pick-and-Place Architecture Generation without requiring VLA training).
"""

from brickcell.studio.engine import SO101StudioEngine

__all__ = ["SO101StudioEngine"]
