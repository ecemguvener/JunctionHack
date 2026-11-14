"""Autonomous 3D DUPLO block localization and all-8-corner inspection architecture.

Implements the non-VLM components of ARCHITECTURE.md:
- Canonical +X-forward world frame, 25-deg canted wrist camera rig, and DUPLO 8-corner geometry
- Calibrated 5-DOF SO-101 Frax URDF, spherized collisions, cone-admissible Lambert Look-At IK,
  and C2/D4 symmetric grasp/reorientation IK
- Material-anchored EightCornerCoverageLedger preventing D4/C4 symmetry gauge swaps
- Dual-RGB-D tabletop cuboid localization and prior-constrained post-manipulation ICP
- SymSkill-native Python 3.12 CogMan bilevel STRIPS planner and ExpectedAtomsMonitor
- MuJoCo 4.0.1 digital twin with live interactive 3D desktop viewer and multi-camera HUD
- Zero-copy babyros/loopback bus, armd v0.1.0 hardware bridge, and multicam/camui adapters
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from brickcell.block_inspector.armd_bridge import (
    SAFE_GRIPPER_MAX_RAW,
    SAFE_GRIPPER_MIN_RAW,
    ArmdFollowerBridge,
    ArmFaultError,
    DecodedArmState,
)
from brickcell.block_inspector.cogman import (
    CogManInspectionAgent,
    ExpectedAtomsMonitor,
    GroundedOperator,
    InspectionRunReport,
    SymbolicGoal,
    SymbolicState,
    compile_instruction,
    plan_strips_sequence,
)
from brickcell.block_inspector.geometry import (
    CORNER_SIGNS,
    DUPLO_CATALOG,
    T_BASE_WORLD,
    T_WORLD_BASE,
    BlockSpec,
    CameraExtrinsicsRig,
    CameraModel,
    corner_bisectors_local,
    corner_facet_normals_local,
    corner_positions_world,
    corner_upper_hemisphere_mask,
    cv_cam_to_mujoco_cam,
    mujoco_cam_to_cv_cam,
    nominal_camera_rig,
    nominal_canted_wrist_cam_transform,
)
from brickcell.block_inspector.ledger import (
    CornerObservation,
    CornerRecord,
    EightCornerCoverageLedger,
    LedgerConfig,
)
from brickcell.block_inspector.real2sim import (
    PerceivedBrick,
    PerceivedPlate,
    Real2SimClosedLoopReport,
    Real2SimPipeline,
    Real2SimSceneState,
    Real2SimSceneTracker,
    RejectedCluster,
    load_duplo_stable_poses,
    save_point_cloud_ply,
    synthesize_cluttered_scene_point_cloud,
)
from brickcell.block_inspector.runtime_bus import (
    CANONICAL_TOPICS,
    BabyRosRuntimeBus,
    InProcessRuntimeBus,
    validate_babyros_topic,
)

__all__ = [
    "ArmFaultError",
    "ArmdFollowerBridge",
    "BabyRosRuntimeBus",
    "BlockSpec",
    "CANONICAL_TOPICS",
    "CORNER_SIGNS",
    "CameraExtrinsicsRig",
    "CameraModel",
    "CogManInspectionAgent",
    "CornerObservation",
    "CornerRecord",
    "DUPLO_CATALOG",
    "DecodedArmState",
    "EightCornerCoverageLedger",
    "ExpectedAtomsMonitor",
    "GroundedOperator",
    "InProcessRuntimeBus",
    "InspectionRunReport",
    "LedgerConfig",
    "PerceivedBrick",
    "PerceivedPlate",
    "Real2SimClosedLoopReport",
    "Real2SimPipeline",
    "Real2SimSceneState",
    "Real2SimSceneTracker",
    "RejectedCluster",
    "SAFE_GRIPPER_MAX_RAW",
    "SAFE_GRIPPER_MIN_RAW",
    "SymbolicGoal",
    "SymbolicState",
    "T_BASE_WORLD",
    "T_WORLD_BASE",
    "compile_instruction",
    "corner_bisectors_local",
    "corner_facet_normals_local",
    "corner_positions_world",
    "corner_upper_hemisphere_mask",
    "cv_cam_to_mujoco_cam",
    "load_duplo_stable_poses",
    "mujoco_cam_to_cv_cam",
    "nominal_camera_rig",
    "nominal_canted_wrist_cam_transform",
    "plan_strips_sequence",
    "save_point_cloud_ply",
    "synthesize_cluttered_scene_point_cloud",
    "validate_babyros_topic",
]
