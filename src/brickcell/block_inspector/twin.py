"""MuJoCo 4.0.1 Inspection Cell Digital Twin with Pure Contact-Physics Grasping."""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from brickcell.block_inspector.cogman import (
    GroundedOperator,
    SkillOutcome,
    SymbolicState,
)
from brickcell.block_inspector.frax_ik import (
    SO101_ARM_JOINTS,
    ClutterGraspChoice,
    SO101FraxIK,
)
from brickcell.block_inspector.geometry import (
    DUPLO_CATALOG,
    PLATE_DROP_SLOT_OFFSETS_XY,
    REPO_ROOT,
    T_WORLD_BASE,
    BlockSpec,
    CameraExtrinsicsRig,
    CameraModel,
    corner_bisectors_local,
    cv_cam_to_mujoco_cam,
    look_at_cv_transform,
    station_camera_rig,
    station_plates_config,
)
from brickcell.block_inspector.ledger import EightCornerCoverageLedger
from brickcell.block_inspector.perception import (
    SOURCE_PLATE_FLOOR_Z_M,
    SOURCE_PLATE_XY,
    localize_clustered_blocks_in_source_plate,
    localize_resting_block,
    relocalize_with_prior,
)

OBSERVATION_HOME_Q = np.array([0.0, -0.85, 0.90, -0.60, 0.0], dtype=np.float64)

# Low "Plate" camera (configs/sim-scene.yaml `side`) presentation waypoint & 3 KALAS plates
PLATE_CAMERA_SCAN_POS = np.array([0.285, -0.215, 0.072], dtype=np.float64)
SOURCE_PILE_XY = SOURCE_PLATE_XY.copy()  # Aquamarine blue source plate at (0.23, 0.00)
GOOD_PILE_XY = np.array([0.10, -0.22], dtype=np.float64)
DEFECTIVE_PILE_XY = np.array([0.10, +0.22], dtype=np.float64)
JOINT5_CCW_START_RAD = -2.95
JOINT5_CCW_END_RAD = +2.95

# Safe hardware gripper floor (1339..2829 raw ticks) mapped to Revolute_5 jaw angle
SAFE_GRIPPER_MIN_RAW = 1339
SAFE_GRIPPER_MAX_RAW = 2829
JAW_CLOSED_RAD = +0.12
JAW_OPEN_RAD = -0.92


def gripper_pct_to_jaw_rad(pct: float) -> float:
    p = float(np.clip(pct, 0.0, 1.0))
    return JAW_CLOSED_RAD + p * (JAW_OPEN_RAD - JAW_CLOSED_RAD)


def gripper_pct_to_safe_raw_ticks(pct: float) -> int:
    p = float(np.clip(pct, 0.0, 1.0))
    return int(round(SAFE_GRIPPER_MIN_RAW + p * (SAFE_GRIPPER_MAX_RAW - SAFE_GRIPPER_MIN_RAW)))


def _fmt_vec(vals) -> str:
    return " ".join(f"{float(v):.8g}" for v in vals)


def _rotmat_to_mj_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to MuJoCo (w, x, y, z) unit quaternion."""
    xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _extract_so101_meshes(cache_dir: Path) -> tuple[Path, ET.Element]:
    """Extract official SO-101 STL visual meshes from SO-101.zip once into cache_dir."""
    mesh_dir = cache_dir / "so101_meshes"
    archive_path = REPO_ROOT / "SO-101.zip"
    with zipfile.ZipFile(archive_path) as zf:
        urdf_root = ET.fromstring(zf.read("robotics_ws_model/robot.urdf"))
        if not mesh_dir.exists() or not any(mesh_dir.glob("*.stl")):
            mesh_dir.mkdir(parents=True, exist_ok=True)
            for member in zf.namelist():
                if member.startswith("robotics_ws_model/meshes/") and member.endswith(".stl"):
                    target = mesh_dir / Path(member).name
                    target.write_bytes(zf.read(member))
    return mesh_dir, urdf_root


def _write_kalas_plate_obj(path: Path, segments: int = 64) -> None:
    if path.exists():
        return
    profile_mm = [
        (0, 3.7),
        (30, 3.6),
        (60, 3.3),
        (75, 3.1),
        (79, 4.0),
        (82, 6.5),
        (85, 11.0),
        (88, 16.5),
        (90.5, 21.5),
        (91.3, 23.5),
        (93.0, 23.5),
        (92.8, 21.5),
        (90.5, 17.0),
        (87.5, 11.5),
        (84.5, 6.5),
        (81.5, 3.0),
        (80, 1.2),
        (79, 0.0),
        (66, 0.0),
        (63, 0.8),
        (60, 1.5),
        (30, 1.8),
        (0, 1.9),
    ]
    profile = [(r / 1000.0, z / 1000.0) for r, z in profile_mm]
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    verts: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for r, z in profile:
        if r == 0.0:
            verts.append((0.0, 0.0, z))
            rings.append([len(verts)] * segments)
        else:
            rings.append(list(range(len(verts) + 1, len(verts) + segments + 1)))
            verts.extend((r * math.cos(a), r * math.sin(a), z) for a in angles)
    faces = []
    for ra, rb in zip(rings, rings[1:]):
        for j in range(segments):
            k = (j + 1) % segments
            quad = [ra[j], rb[j], rb[k], ra[k]]
            for tri in (quad[:3], [quad[0], quad[2], quad[3]]):
                if len(set(tri)) == 3:
                    faces.append(tri)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class SpawnedBlockInfo:
    """Initial specification for one simulated DUPLO brick in the inspection cell."""

    block_id: str
    spec: BlockSpec
    initial_xy_yaw: tuple[float, float, float]
    has_visual_defect: bool = False
    initial_T_world_block: np.ndarray | None = None


def _obb_2d_corners(cx: float, cy: float, lx: float, wy: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = 0.5 * lx, 0.5 * wy
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (local @ R.T) + np.array([cx, cy], dtype=np.float64)


def _obb_2d_separated(
    c1: tuple[float, float],
    d1: tuple[float, float],
    y1: float,
    c2: tuple[float, float],
    d2: tuple[float, float],
    y2: float,
    margin_m: float,
) -> bool:
    """Exact 2D Separating Axis Theorem check for two oriented rectangles inflated by margin_m/2."""
    pts1 = _obb_2d_corners(c1[0], c1[1], d1[0] + margin_m, d1[1] + margin_m, y1)
    pts2 = _obb_2d_corners(c2[0], c2[1], d2[0] + margin_m, d2[1] + margin_m, y2)
    for yaw in (y1, y2):
        for ax in (
            np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64),
            np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64),
        ):
            p1 = pts1 @ ax
            p2 = pts2 @ ax
            if float(p1.max()) < float(p2.min()) or float(p2.max()) < float(p1.min()):
                return True
    return False


def sample_clustered_blocks_in_source_plate(
    num_blocks: int = 8,
    seed: int | None = 42,
    source_plate_xy: np.ndarray = SOURCE_PILE_XY,
    enable_random_defects: bool = True,
) -> list[SpawnedBlockInfo]:
    """Randomly generate `num_blocks` mixed DUPLO bricks clustered inside the aquamarine plate."""
    rng = np.random.default_rng(seed)
    catalog_keys = sorted(DUPLO_CATALOG.keys())
    if num_blocks >= len(catalog_keys):
        chosen_keys = list(catalog_keys) + [
            catalog_keys[int(rng.integers(0, len(catalog_keys)))]
            for _ in range(num_blocks - len(catalog_keys))
        ]
    else:
        chosen_keys = [
            catalog_keys[int(i)]
            for i in rng.choice(len(catalog_keys), size=num_blocks, replace=False)
        ]
    rng.shuffle(chosen_keys)

    defect_flags = [False] * num_blocks
    if enable_random_defects and num_blocks >= 2:
        n_def = max(
            1, min(num_blocks - 1, int(rng.integers(max(1, num_blocks // 3), num_blocks // 2 + 2)))
        )
        for idx in rng.choice(num_blocks, size=n_def, replace=False):
            defect_flags[int(idx)] = True

    px, py = float(source_plate_xy[0]), float(source_plate_xy[1])
    placed: list[SpawnedBlockInfo] = []
    for i, key in enumerate(chosen_keys):
        spec = DUPLO_CATALOG[key]
        lx, wy, _ = spec.dims_m
        best_pose: tuple[float, float, float] | None = None
        for gap_m in (0.0125, 0.0108, 0.0094, 0.0080):
            best_dist = 1e9
            for _ in range(640):
                if i == 0:
                    r = float(rng.uniform(0.0, 0.012))
                else:
                    r = float(rng.uniform(0.014, 0.066))
                ang = float(rng.uniform(-math.pi, math.pi))
                cand_x = px + r * math.cos(ang)
                cand_y = py + r * math.sin(ang)
                cand_yaw = float(rng.uniform(-math.pi, math.pi))
                corners = _obb_2d_corners(cand_x, cand_y, lx, wy, cand_yaw)
                max_r_plate = float(np.max(np.linalg.norm(corners - np.array([[px, py]]), axis=1)))
                if max_r_plate > 0.0740:
                    continue
                if all(
                    _obb_2d_separated(
                        (cand_x, cand_y),
                        (lx, wy),
                        cand_yaw,
                        (prev.initial_xy_yaw[0], prev.initial_xy_yaw[1]),
                        (prev.spec.dims_m[0], prev.spec.dims_m[1]),
                        prev.initial_xy_yaw[2],
                        margin_m=gap_m,
                    )
                    for prev in placed
                ):
                    # Pack tightly toward the cluster centroid so blocks sit clustered together
                    if placed:
                        cx_m = sum(p.initial_xy_yaw[0] for p in placed) / len(placed)
                        cy_m = sum(p.initial_xy_yaw[1] for p in placed) / len(placed)
                        d_c = math.hypot(cand_x - cx_m, cand_y - cy_m)
                    else:
                        d_c = r
                    if d_c < best_dist:
                        best_dist = d_c
                        best_pose = (cand_x, cand_y, cand_yaw)
            if best_pose is not None:
                break
        if best_pose is None:
            ang = 2.0 * math.pi * i / max(1, num_blocks)
            best_pose = (px + 0.048 * math.cos(ang), py + 0.048 * math.sin(ang), ang)
        placed.append(
            SpawnedBlockInfo(
                block_id=f"block_{i}",
                spec=spec,
                initial_xy_yaw=best_pose,
                has_visual_defect=defect_flags[i],
            )
        )
    return placed


def build_inspection_cell_mjcf(
    scene_dir: Path,
    block_spec: BlockSpec,
    rig: CameraExtrinsicsRig,
    initial_block_xy_yaw: tuple[float, float, float] = (0.21, 0.00, 0.20),
    spawned_blocks: list[SpawnedBlockInfo] | None = None,
    plate_positions: dict[str, np.ndarray] | None = None,
) -> Path:
    """Generate unified MuJoCo 4.0.1 MJCF synchronized 1-to-1 with CalibratedSO101Kinematics."""
    scene_dir = Path(scene_dir).resolve()
    scene_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir, urdf_root = _extract_so101_meshes(scene_dir)
    plate_obj = scene_dir / "kalas_plate.obj"
    _write_kalas_plate_obj(plate_obj)
    blocks_to_spawn = spawned_blocks or [
        SpawnedBlockInfo(
            block_id="block_0",
            spec=block_spec,
            initial_xy_yaw=initial_block_xy_yaw,
            has_visual_defect=False,
        )
    ]

    from so101_kinematics import CalibratedSO101Kinematics

    cal_kin = CalibratedSO101Kinematics("follower")
    priors_file = (
        REPO_ROOT / "configs" / "published_priors_verified.json"
        if (REPO_ROOT / "configs" / "published_priors_verified.json").is_file()
        else REPO_ROOT / "artifacts" / "dynamics" / "published_priors_verified.json"
    )
    priors_doc = json.loads(priors_file.read_text(encoding="utf-8")) if priors_file.exists() else {}
    priors = priors_doc.get("links", {})
    physics_priors = priors_doc.get("physics", {})
    j_damp = str(physics_priors.get("joint_damping_nm_s_per_rad", 0.6))
    j_arm = str(physics_priors.get("armature_kg_m2", 0.028))
    j_fric = str(physics_priors.get("frictionloss_nm", 0.052))
    grav_vec = physics_priors.get("gravity_m_s2", [0.0, 0.0, -9.80665])
    motor_tau = float(physics_priors.get("motor_torque_limit_nm", 2.94))

    urdf_links = {lnk.get("name"): lnk for lnk in urdf_root.findall("link")}

    root = ET.Element("mujoco", model="so101_eight_corner_inspection_cell")
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        autolimits="true",
        fusestatic="false",
        inertiafromgeom="false",
    )
    opt = ET.SubElement(
        root,
        "option",
        timestep="0.002",
        gravity=_fmt_vec(grav_vec),
        integrator="implicitfast",
        solver="Newton",
        iterations="80",
        cone="elliptic",
        impratio="25",
    )
    ET.SubElement(opt, "flag", speculative="enable")
    ET.SubElement(root, "statistic", center="0.19 0.00 0.07", extent="0.38")

    vis = ET.SubElement(root, "visual")
    ET.SubElement(
        vis,
        "global",
        offwidth=str(max(640, rig.realsense_1.width, rig.realsense_2.width, rig.wrist.width)),
        offheight=str(max(480, rig.realsense_1.height, rig.realsense_2.height, rig.wrist.height)),
        azimuth="138",
        elevation="-25",
    )
    ET.SubElement(
        vis,
        "headlight",
        ambient="0.42 0.44 0.48",
        diffuse="0.55 0.55 0.55",
        specular="0.15 0.15 0.15",
    )
    ET.SubElement(vis, "quality", shadowsize="2048")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="table_grid",
        type="2d",
        builtin="checker",
        rgb1="0.16 0.19 0.24",
        rgb2="0.20 0.24 0.30",
        width="512",
        height="512",
        mark="edge",
        markrgb="0.28 0.33 0.40",
    )
    ET.SubElement(
        asset,
        "material",
        name="table_mat",
        texture="table_grid",
        texrepeat="8 8",
        texuniform="true",
        reflectance="0.04",
    )
    ET.SubElement(asset, "mesh", name="kalas_plate", file=str(plate_obj), inertia="shell")

    registered_duplo_meshes: set[str] = set()
    for sp in [b.spec for b in blocks_to_spawn]:
        if (
            sp.name not in registered_duplo_meshes
            and sp.stl_path is not None
            and sp.stl_path.exists()
        ):
            ET.SubElement(
                asset,
                "mesh",
                name=sp.name,
                file=str(sp.stl_path),
                inertia="convex",
            )
            registered_duplo_meshes.add(sp.name)

    # Register all 59 SO-101 CAD STL visual meshes
    registered_meshes: set[str] = set()
    for lname, ulink in urdf_links.items():
        for v_idx, vis_el in enumerate(ulink.findall("visual")):
            mesh_el = vis_el.find("geometry/mesh")
            if mesh_el is None:
                continue
            fname = Path(mesh_el.get("filename", "")).name
            m_key = Path(fname).stem
            if m_key not in registered_meshes and (mesh_dir / fname).exists():
                scale_str = mesh_el.get("scale", "1 1 1")
                ET.SubElement(
                    asset,
                    "mesh",
                    name=m_key,
                    file=str(mesh_dir / fname),
                    scale=scale_str,
                )
                registered_meshes.add(m_key)

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping=j_damp, armature=j_arm, frictionloss=j_fric)
    ET.SubElement(
        default,
        "geom",
        condim="4",
        friction="1.6 0.02 0.002",
        solref="0.008 1",
        solimp="0.95 0.99 0.001",
    )

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world, "light", pos="0.35 -0.25 0.95", dir="-0.15 0.15 -1", diffuse="0.75 0.75 0.78"
    )
    ET.SubElement(
        world,
        "light",
        pos="0.35 0.30 0.85",
        dir="-0.15 -0.20 -1",
        diffuse="0.45 0.48 0.52",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="plane",
        size="0.8 0.8 0.02",
        material="table_mat",
        contype="1",
        conaffinity="1",
    )

    # 3 IKEA KALAS plates matching configs/sim-scene.yaml (`source` aquamarine, `good` green, `defective` pink)
    plates_cfg = station_plates_config()
    resolved_plate_pos: dict[str, np.ndarray] = {}
    for pname in ("source", "good", "defective"):
        pspec = plates_cfg[pname]
        p_pos = (
            np.asarray(plate_positions[pname], dtype=np.float64)
            if (plate_positions is not None and pname in plate_positions)
            else np.asarray(pspec.pos_m, dtype=np.float64)
        )
        resolved_plate_pos[pname] = p_pos
        pbody = ET.SubElement(world, "body", name=f"plate_{pname}", pos=_fmt_vec(p_pos))
        ET.SubElement(
            pbody,
            "geom",
            name=f"plate_{pname}_vis",
            type="mesh",
            mesh="kalas_plate",
            rgba=_fmt_vec(pspec.rgba),
            contype="0",
            conaffinity="0",
            group="2",
        )
        ET.SubElement(
            pbody,
            "geom",
            name=f"plate_{pname}_floor",
            type="cylinder",
            size="0.078 0.002",
            pos="0 0 0.002",
            rgba="0.4 0.4 0.4 0.25",
            contype="1",
            conaffinity="1",
            group="3",
        )
        # 16-segment collision rim around r=88.5 mm so clustered & sorted bricks stay inside the bowls
        n_rim = 16
        for seg_idx in range(n_rim):
            theta = 2.0 * math.pi * seg_idx / n_rim
            rx = 0.0885 * math.cos(theta)
            ry = 0.0885 * math.sin(theta)
            q_rim = _rotmat_to_mj_quat(Rotation.from_euler("z", theta + 0.5 * math.pi).as_matrix())
            ET.SubElement(
                pbody,
                "geom",
                name=f"plate_{pname}_rim_{seg_idx}",
                type="box",
                size="0.0185 0.0025 0.0105",
                pos=_fmt_vec([rx, ry, 0.0115]),
                quat=_fmt_vec(q_rim),
                rgba="0.4 0.4 0.4 0.20",
                contype="1",
                conaffinity="1",
                group="3",
            )

    # Static RealSense D415 cameras + visible housing bars (group 2 so mj_ray group 0 ignores camera bars!)
    for cam_model in (rig.realsense_1, rig.realsense_2):
        T_mj = cv_cam_to_mujoco_cam(cam_model.T_parent_cvcam)
        sensor_attrs = cam_model.mujoco_sensor_attrs()
        cam_attrs = (
            sensor_attrs if sensor_attrs is not None else {"fovy": f"{cam_model.fovy_deg:.4f}"}
        )
        ET.SubElement(
            world,
            "camera",
            name=cam_model.name,
            pos=_fmt_vec(T_mj[:3, 3]),
            quat=_fmt_vec(_rotmat_to_mj_quat(T_mj[:3, :3])),
            **cam_attrs,
        )
        cbody = ET.SubElement(
            world,
            "body",
            name=f"{cam_model.name}_mount",
            pos=_fmt_vec(T_mj[:3, 3]),
            quat=_fmt_vec(_rotmat_to_mj_quat(T_mj[:3, :3])),
        )
        ET.SubElement(
            cbody,
            "geom",
            name=f"{cam_model.name}_housing",
            type="box",
            size="0.048 0.012 0.012",
            pos="0 0 0.014",
            rgba="0.76 0.79 0.84 1",
            contype="0",
            conaffinity="0",
            group="2",
        )

    # Hero cinematic camera for screenshots / overview
    T_hero_mj = cv_cam_to_mujoco_cam(
        look_at_cv_transform(np.array([0.52, -0.38, 0.34]), np.array([0.18, 0.0, 0.06]))
    )
    ET.SubElement(
        world,
        "camera",
        name="hero",
        pos=_fmt_vec(T_hero_mj[:3, 3]),
        quat=_fmt_vec(_rotmat_to_mj_quat(T_hero_mj[:3, :3])),
        fovy="44",
    )

    # Root link Body5_1 registered to +X-forward world frame at calibrated table height T_WORLD_BASE
    base_body = ET.SubElement(
        world,
        "body",
        name=cal_kin.base,
        pos=_fmt_vec(T_WORLD_BASE[:3, 3]),
        quat=_fmt_vec(_rotmat_to_mj_quat(T_WORLD_BASE[:3, :3])),
    )
    bodies: dict[str, ET.Element] = {cal_kin.base: base_body}

    for jname, parent, child, home, axis in cal_kin.joints:
        quat = _rotmat_to_mj_quat(home[:3, :3])
        b_el = ET.SubElement(
            bodies[parent],
            "body",
            name=child,
            pos=_fmt_vec(home[:3, 3]),
            quat=_fmt_vec(quat),
        )
        bodies[child] = b_el
        if jname in SO101_ARM_JOINTS:
            idx = SO101_ARM_JOINTS.index(jname)
            ET.SubElement(
                b_el,
                "joint",
                name=jname,
                type="hinge",
                axis=_fmt_vec(axis),
                range=_fmt_vec([cal_kin.calibration.lower[idx], cal_kin.calibration.upper[idx]]),
            )
        elif jname == "Revolute_5":
            ET.SubElement(
                b_el,
                "joint",
                name="Revolute_5",
                type="hinge",
                axis=_fmt_vec(axis),
                range="-0.95 0.15",
                damping="0.20",
                armature="0.005",
            )

    # Populate verified published CAD inertials (matching model-published-verified/arm.xml)
    for lname, b_el in bodies.items():
        p_row = priors.get(lname)
        if p_row and float(p_row.get("mass_kg", 0.0)) > 1e-6:
            raw_m = float(p_row["mass_kg"])
            # Split moving jaw mass (0.012 kg on Body5_2) out of Body3 (0.099 -> 0.087 kg) to conserve 0.632006 kg
            scale_m = ((raw_m - 0.012) / raw_m) if (lname == "Body3" and raw_m > 0.02) else 1.0
            mat = (
                p_row.get("inertia_matrix_kg_m2")
                or np.diag(p_row["diagonal_inertia_kg_m2"]).tolist()
            )
            ET.SubElement(
                b_el,
                "inertial",
                mass=f"{raw_m * scale_m:.8g}",
                pos=_fmt_vec(p_row["com_m"]),
                fullinertia=_fmt_vec(
                    [
                        mat[0][0] * scale_m,
                        mat[1][1] * scale_m,
                        mat[2][2] * scale_m,
                        mat[0][1] * scale_m,
                        mat[0][2] * scale_m,
                        mat[1][2] * scale_m,
                    ]
                ),
            )
        elif lname == "Body5_2":
            ET.SubElement(
                b_el,
                "inertial",
                mass="0.012",
                pos="0.03938 -0.01424 -0.018355",
                diaginertia="6.61e-6 1.89e-6 5.29e-6",
            )

        ulink = urdf_links.get(lname)
        if ulink is not None:
            is_hub = "hub" in lname
            rgba_vis = "0.18 0.20 0.24 1" if is_hub else "0.93 0.90 0.83 1"
            for v_idx, vis_el in enumerate(ulink.findall("visual")):
                mesh_el = vis_el.find("geometry/mesh")
                if mesh_el is None:
                    continue
                m_key = Path(mesh_el.get("filename", "")).stem
                if m_key not in registered_meshes:
                    continue
                orig = vis_el.find("origin")
                xyz = (
                    np.fromstring(orig.get("xyz", "0 0 0"), sep=" ")
                    if orig is not None
                    else np.zeros(3)
                )
                rpy = (
                    np.fromstring(orig.get("rpy", "0 0 0"), sep=" ")
                    if orig is not None
                    else np.zeros(3)
                )
                q_vis = _rotmat_to_mj_quat(Rotation.from_euler("xyz", rpy).as_matrix())
                ET.SubElement(
                    b_el,
                    "geom",
                    name=f"vis_{lname}_{v_idx}",
                    type="mesh",
                    mesh=m_key,
                    pos=_fmt_vec(xyz),
                    quat=_fmt_vec(q_vis),
                    rgba=rgba_vis,
                    contype="0",
                    conaffinity="0",
                    group="1",
                )

        # Ray-occlusion proxy boxes on structural links (group 3: hidden in Filament/Renderer, active in mj_ray)
        if (
            p_row
            and "box_size_m" in p_row
            and lname in {"Body5_1", "Body1_2", "Group 1", "Body1_3", "Group 1_2"}
        ):
            half_sz = 0.45 * np.asarray(p_row["box_size_m"], dtype=np.float64)
            ET.SubElement(
                b_el,
                "geom",
                name=f"col_{lname}",
                type="box",
                pos=_fmt_vec(p_row.get("collision_center_m", p_row["com_m"])),
                size=_fmt_vec(np.maximum(half_sz, 0.004)),
                rgba="0.4 0.4 0.4 0.25",
                contype="0",
                conaffinity="0",
                group="3",
            )

    # Exact CAD STL-aligned high-friction gripper jaw pads on Body3 (Arm_Design_6) & Body5_2 (Arm_Design_7)
    ET.SubElement(
        bodies["Body3"],
        "geom",
        name="jaw_pad_fixed",
        type="box",
        pos="-0.0115 0.0 -0.0880",
        size="0.0035 0.0085 0.0150",
        rgba="0.15 0.15 0.18 1",
        friction="3.6 0.08 0.01",
        solref="0.004 1",
        solimp="0.97 0.995 0.0005",
        contype="1",
        conaffinity="1",
        group="0",
    )
    ET.SubElement(
        bodies["Body3"],
        "site",
        name="jaw_fixed_site",
        pos="-0.0080 0.0 -0.0880",
        size="0.002",
        rgba="0 0 0 0",
    )
    ET.SubElement(
        bodies["Body3"],
        "site",
        name="tcp_site",
        pos="0.015 0.0 -0.088",
        size="0.003",
        rgba="0.2 0.8 1.0 0.5",
    )

    # Moving jaw pad expressed in Body5_2 frame (mapped from Body3 [-0.0018, 0.0, -0.0880] => inner face -0.0053 m)
    f0 = cal_kin.frames(np.zeros(5))
    T_b52_b3 = np.linalg.inv(f0["Body5_2"]) @ f0["Body3"]
    p_pad_b52 = (T_b52_b3 @ np.array([-0.0018, 0.0, -0.0880, 1.0]))[:3]
    q_pad_b52 = _rotmat_to_mj_quat(T_b52_b3[:3, :3])
    ET.SubElement(
        bodies["Body5_2"],
        "geom",
        name="jaw_pad_moving",
        type="box",
        pos=_fmt_vec(p_pad_b52),
        quat=_fmt_vec(q_pad_b52),
        size="0.0035 0.0085 0.0150",
        rgba="0.15 0.15 0.18 1",
        friction="3.6 0.08 0.01",
        solref="0.004 1",
        solimp="0.97 0.995 0.0005",
        contype="1",
        conaffinity="1",
        group="0",
    )
    ET.SubElement(
        bodies["Body5_2"],
        "site",
        name="jaw_moving_site",
        pos=_fmt_vec(p_pad_b52),
        size="0.002",
        rgba="0 0 0 0",
    )

    # Wrist UVC camera registered to Body3 + visible camera module housing (group 2)
    T_wrist_mj = cv_cam_to_mujoco_cam(rig.wrist.T_parent_cvcam)
    wrist_sensor_attrs = rig.wrist.mujoco_sensor_attrs()
    wrist_cam_attrs = (
        wrist_sensor_attrs
        if wrist_sensor_attrs is not None
        else {"fovy": f"{rig.wrist.fovy_deg:.4f}"}
    )
    ET.SubElement(
        bodies["Body3"],
        "camera",
        name="wrist",
        pos=_fmt_vec(T_wrist_mj[:3, 3]),
        quat=_fmt_vec(_rotmat_to_mj_quat(T_wrist_mj[:3, :3])),
        **wrist_cam_attrs,
    )
    ET.SubElement(
        bodies["Body3"],
        "geom",
        name="wrist_cam_housing",
        type="box",
        pos=_fmt_vec(T_wrist_mj[:3, 3]),
        quat=_fmt_vec(_rotmat_to_mj_quat(T_wrist_mj[:3, :3])),
        size="0.015 0.012 0.008",
        rgba="0.12 0.55 0.92 1",
        contype="0",
        conaffinity="0",
        group="2",
    )

    # Spawn 1..N DUPLO block bodies ("block_0" .. "block_{N-1}") resting on table / source plate
    src_xy = np.asarray(resolved_plate_pos["source"][:2], dtype=np.float64)
    for b_idx, spawn in enumerate(blocks_to_spawn):
        b_name = spawn.block_id
        sp = spawn.spec
        bx, by, byaw = spawn.initial_xy_yaw
        lx, wy, hz = sp.dims_m
        in_src = float(math.hypot(bx - float(src_xy[0]), by - float(src_xy[1]))) <= 0.080
        z_floor = SOURCE_PLATE_FLOOR_Z_M if in_src else 0.0
        if spawn.initial_T_world_block is not None:
            T_init = np.asarray(spawn.initial_T_world_block, dtype=np.float64)
            half_world_z = float(
                np.sum(np.abs(T_init[2, :3]) * (0.5 * np.asarray(sp.dims_m, dtype=np.float64)))
            )
            pos_init = [
                float(T_init[0, 3]),
                float(T_init[1, 3]),
                max(float(T_init[2, 3]), z_floor + half_world_z + 0.0005),
            ]
            q_block = _rotmat_to_mj_quat(T_init[:3, :3])
        else:
            pos_init = [bx, by, z_floor + 0.5 * hz + 0.0005]
            q_block = _rotmat_to_mj_quat(Rotation.from_euler("z", byaw).as_matrix())
        block_body = ET.SubElement(
            world,
            "body",
            name=b_name,
            pos=_fmt_vec(pos_init),
            quat=_fmt_vec(q_block),
        )
        ET.SubElement(block_body, "freejoint", name=f"{b_name}_joint")
        m_blk = sp.mass_kg
        ixx_b = max((1.0 / 12.0) * m_blk * (wy * wy + hz * hz), 1e-7)
        iyy_b = max((1.0 / 12.0) * m_blk * (lx * lx + hz * hz), 1e-7)
        izz_b = max((1.0 / 12.0) * m_blk * (lx * lx + wy * wy), 1e-7)
        ET.SubElement(
            block_body,
            "inertial",
            pos="0 0 0",
            mass=f"{m_blk:.6f}",
            diaginertia=_fmt_vec([ixx_b, iyy_b, izz_b]),
        )

        r, g, b_col, a = sp.rgba
        has_stl = sp.stl_path is not None and sp.stl_path.exists()
        ET.SubElement(
            block_body,
            "geom",
            name=f"{b_name}_cuboid",
            type="box",
            size=_fmt_vec([0.5 * lx, 0.5 * wy, 0.5 * hz]),
            mass=f"{sp.mass_kg:.6f}",
            friction="3.2 0.06 0.008",
            solref="0.004 1",
            solimp="0.97 0.995 0.0005",
            rgba="0.5 0.5 0.5 0.25" if has_stl else f"{r} {g} {b_col} {a}",
            contype="1",
            conaffinity="1",
            group="3" if has_stl else "0",
        )
        if has_stl:
            ET.SubElement(
                block_body,
                "geom",
                name=f"{b_name}_mesh_vis",
                type="mesh",
                mesh=sp.name,
                pos=_fmt_vec([0.0, 0.0, -0.5 * hz]),
                rgba=f"{r} {g} {b_col} {a}",
                contype="0",
                conaffinity="0",
                group="1",
            )

        # Synthetic visual surface defect decals on all 4 vertical walls so whichever axis is grasped,
        # the uncovered faces clearly expose the blemish to the Plate camera during the Joint-5 CCW spin.
        if spawn.has_visual_defect:
            half_h_patch = min(0.0055, 0.28 * hz)
            half_w_y = min(0.0062, 0.34 * wy)
            half_w_x = min(0.0062, 0.34 * lx)
            for s_idx, sgn in enumerate((+1.0, -1.0)):
                ET.SubElement(
                    block_body,
                    "geom",
                    name=f"{b_name}_defect_x_{s_idx}",
                    type="box",
                    pos=_fmt_vec([sgn * (0.5 * lx + 0.0004), 0.0, -0.001]),
                    size=_fmt_vec([0.0004, half_w_y, half_h_patch]),
                    rgba="0.96 0.08 0.90 1",
                    contype="0",
                    conaffinity="0",
                    group="1",
                )
                ET.SubElement(
                    block_body,
                    "geom",
                    name=f"{b_name}_defect_x_core_{s_idx}",
                    type="box",
                    pos=_fmt_vec([sgn * (0.5 * lx + 0.0007), 0.0, -0.001]),
                    size=_fmt_vec([0.0003, 0.58 * half_w_y, 0.58 * half_h_patch]),
                    rgba="0.99 0.96 1.00 1",
                    contype="0",
                    conaffinity="0",
                    group="1",
                )
                ET.SubElement(
                    block_body,
                    "geom",
                    name=f"{b_name}_defect_y_{s_idx}",
                    type="box",
                    pos=_fmt_vec([0.0, sgn * (0.5 * wy + 0.0004), -0.001]),
                    size=_fmt_vec([half_w_x, 0.0004, half_h_patch]),
                    rgba="0.96 0.08 0.90 1",
                    contype="0",
                    conaffinity="0",
                    group="1",
                )
                ET.SubElement(
                    block_body,
                    "geom",
                    name=f"{b_name}_defect_y_core_{s_idx}",
                    type="box",
                    pos=_fmt_vec([0.0, sgn * (0.5 * wy + 0.0007), -0.001]),
                    size=_fmt_vec([0.58 * half_w_x, 0.0003, 0.58 * half_h_patch]),
                    rgba="0.99 0.96 1.00 1",
                    contype="0",
                    conaffinity="0",
                    group="1",
                )

        # Corner sites (preserve exact "corner_0..7" names on block_0 for legacy single-block eight_corner mode)
        prefix = "" if b_idx == 0 else f"{b_name}_"
        for c_idx, c_local in enumerate(sp.corners_local()):
            ET.SubElement(
                block_body,
                "site",
                name=f"{prefix}corner_{c_idx}",
                pos=_fmt_vec(c_local),
                size="0.0022",
                rgba="1 0.85 0.1 0.9",
            )
            inset = c_local * 0.88
            ET.SubElement(
                block_body,
                "geom",
                name=f"{prefix}corner_detail_{c_idx}",
                type="box",
                pos=_fmt_vec(inset),
                size="0.0018 0.0018 0.0018",
                rgba=f"{min(1.0, r + 0.28):.2f} {min(1.0, g + 0.28):.2f} {min(1.0, b_col + 0.28):.2f} 1",
                contype="0",
                conaffinity="0",
                group="1",
            )

    # Position servo actuators matching STS3215 closed-loop stiffness & verified torque cap
    actuators = ET.SubElement(root, "actuator")
    kp_gains = [85.0, 110.0, 95.0, 65.0, 45.0]
    kv_gains = [4.2, 5.8, 4.8, 3.0, 2.0]
    for idx, jname in enumerate(SO101_ARM_JOINTS):
        ET.SubElement(
            actuators,
            "general",
            name=f"act_{jname}",
            joint=jname,
            biastype="affine",
            gainprm=f"{kp_gains[idx]} 0 0",
            biasprm=f"0 -{kp_gains[idx]} -{kv_gains[idx]}",
            forcelimited="true",
            forcerange=f"-{motor_tau:.4g} {motor_tau:.4g}",
            ctrllimited="true",
            ctrlrange=_fmt_vec([cal_kin.calibration.lower[idx], cal_kin.calibration.upper[idx]]),
        )
    ET.SubElement(
        actuators,
        "general",
        name="act_gripper",
        joint="Revolute_5",
        biastype="affine",
        gainprm="24.0 0 0",
        biasprm="0 -24.0 -1.0",
        forcelimited="true",
        forcerange="-1.4 1.4",
        ctrllimited="true",
        ctrlrange="-0.95 0.15",
    )

    xml_path = scene_dir / f"cell_{block_spec.name}.xml"
    ET.indent(root)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


@dataclass
class TwinStepTelemetry:
    q_enc_rad: np.ndarray
    qvel_rad_s: np.ndarray
    gripper_pct: float
    T_world_body3: np.ndarray
    T_world_block_gt: np.ndarray


def _bring_win32_process_windows_to_front() -> list[tuple[int, str, bool]]:
    """Override Windows STARTUPINFO SW_HIDE inheritance so MuJoCo Studio window is visible on screen."""
    if sys.platform != "win32":
        return []
    user32 = ctypes.windll.user32
    scr_w = max(1280, int(user32.GetSystemMetrics(0)))
    scr_h = max(800, int(user32.GetSystemMetrics(1)))
    mj_w = max(860, int(scr_w * 0.65))
    mj_h = max(660, int(scr_h * 0.88))
    pid_target = os.getpid()
    found: list[tuple[int, str, bool]] = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def _cb(hwnd, _lparam):
        proc_id = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
        if proc_id.value == pid_target:
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            if "MuJoCo" in title or "Studio" in title:
                # Call ShowWindow twice to clear STARTUPINFO.wShowWindow=SW_HIDE, then raise window
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.ShowWindow(hwnd, 5)  # SW_SHOW
                user32.SetWindowPos(
                    hwnd, -1, 16, 24, mj_w, mj_h, 0x0040
                )  # HWND_TOPMOST | SWP_SHOWWINDOW
                user32.SetWindowPos(
                    hwnd, -2, 0, 0, 0, 0, 0x0003
                )  # HWND_NOTOPMOST | NOMOVE | NOSIZE
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
                vis = bool(user32.IsWindowVisible(hwnd))
                found.append((int(hwnd), title, vis))
        return True

    cb_func = WNDENUMPROC(_cb)
    for _ in range(18):
        found.clear()
        user32.EnumWindows(cb_func, 0)
        if any(("MuJoCo" in title or "Studio" in title) and vis for _, title, vis in found):
            break
        time.sleep(0.08)
    return found


class SO101InspectionTwin:
    """MuJoCo 4.0.1 physics + RGB-D + Frax IK digital twin with live interactive 3D viewer."""

    def __init__(
        self,
        block_spec: BlockSpec | str = "duplo_2x2_3437",
        rig: CameraExtrinsicsRig | None = None,
        frax_ik: SO101FraxIK | None = None,
        ledger: EightCornerCoverageLedger | None = None,
        scene_dir: str | Path | None = None,
        initial_block_xy_yaw: tuple[float, float, float] = (0.21, 0.00, 0.20),
        show_viewer: bool = True,
        realtime_pacing: bool = True,
        anomaly_scorer=None,
        anomaly_threshold: float | None = None,
        anomaly_message_fn=None,
        num_blocks: int = 1,
        cluster_seed: int | None = 42,
        enable_random_defects: bool = True,
        spawned_blocks: list[SpawnedBlockInfo] | None = None,
        plate_positions: dict[str, np.ndarray] | None = None,
        show_camera_windows: bool | None = None,
    ) -> None:
        if mujoco.__version__ != "4.0.1":
            raise RuntimeError(f"Expected mujoco==4.0.1, got {mujoco.__version__}")

        self.block_spec = DUPLO_CATALOG[block_spec] if isinstance(block_spec, str) else block_spec
        self.rig = rig or station_camera_rig()
        self.ik = frax_ik or SO101FraxIK(T_body3_cam_np=self.rig.wrist.T_parent_cvcam)
        if spawned_blocks is not None:
            self.spawned_blocks = list(spawned_blocks)
        elif num_blocks > 1:
            self.spawned_blocks = sample_clustered_blocks_in_source_plate(
                num_blocks=num_blocks,
                seed=cluster_seed,
                enable_random_defects=enable_random_defects,
            )
            self.block_spec = self.spawned_blocks[0].spec
        else:
            self.spawned_blocks = [
                SpawnedBlockInfo(
                    block_id="block_0",
                    spec=self.block_spec,
                    initial_xy_yaw=initial_block_xy_yaw,
                    has_visual_defect=False,
                )
            ]

        self.ledger = ledger or EightCornerCoverageLedger(
            object_id="block_0", block_spec=self.block_spec, rig=self.rig
        )
        self.anomaly_scorer = anomaly_scorer
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_message_fn = anomaly_message_fn
        self.scene_dir = (
            Path(scene_dir).resolve()
            if scene_dir
            else (REPO_ROOT / "artifacts" / "block_inspector" / "mjcf").resolve()
        )
        xml_tag = (
            f"cluster_{len(self.spawned_blocks)}"
            if len(self.spawned_blocks) > 1
            else self.block_spec.name
        )
        self.xml_path = build_inspection_cell_mjcf(
            self.scene_dir,
            block_spec=self.block_spec,
            rig=self.rig,
            initial_block_xy_yaw=initial_block_xy_yaw,
            spawned_blocks=self.spawned_blocks,
            plate_positions=plate_positions,
        )
        if xml_tag != self.block_spec.name:
            tagged_xml = self.scene_dir / f"cell_{xml_tag}.xml"
            tagged_xml.write_text(self.xml_path.read_text(encoding="utf-8"), encoding="utf-8")
            self.xml_path = tagged_xml
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self._arm_jnt_ids = [self.model.joint(n).id for n in SO101_ARM_JOINTS]
        self._arm_qpos_adr = np.array([self.model.jnt_qposadr[j] for j in self._arm_jnt_ids])
        self._arm_qvel_adr = np.array([self.model.jnt_dofadr[j] for j in self._arm_jnt_ids])
        self._jaw_qpos_adr = int(self.model.jnt_qposadr[self.model.joint("Revolute_5").id])
        self.block_body_ids = {
            sp.block_id: int(self.model.body(sp.block_id).id) for sp in self.spawned_blocks
        }
        self._block_jnt_id = self.model.joint("block_0_joint").id
        self._block_qpos_adr = int(self.model.jnt_qposadr[self._block_jnt_id])
        self._block_qvel_adr = int(self.model.jnt_dofadr[self._block_jnt_id])
        self._block_body_id = self.block_body_ids["block_0"]
        self._body3_id = self.model.body("Body3").id
        self._jaw_fixed_site = self.model.site("jaw_fixed_site").id
        self._jaw_moving_site = self.model.site("jaw_moving_site").id

        self.q_cmd = OBSERVATION_HOME_Q.copy()
        self.gripper_pct = 0.85
        self.hand_empty = True
        self.at_rest = False
        self.active_target_corner: int | None = None
        self._active_clutter_choice: ClutterGraspChoice | None = None
        self.source_plate_cleared: bool = False
        self.remaining_source_blocks: int = len(self.spawned_blocks)
        self.sorted_counts: dict[str, int] = {"good": 0, "defective": 0}
        self.batch_sorted_records: list[dict] = []
        self.plate_scan_active: bool = False
        self.plate_scan_completed: bool = False
        self.plate_scan_valid_frames: int = 0
        self.plate_scan_keyframes: list[np.ndarray] = []
        self.anomaly_flagged: bool = False
        self.anomaly_messages: list[str] = []
        self.anomaly_untrusted_error: str | None = None
        self.max_plate_anomaly_distance: float | None = None
        self.sorted_pile: str | None = None
        self._T_body3_block_held: np.ndarray = np.eye(4, dtype=np.float64)
        self.status_banner: str = (
            "Initializing 3-Plate SO-101 Cell (Aquamarine Source -> Good / Defective)"
        )
        self._reachability_cache: dict[int, object] = {}
        self._reference_hsv: tuple[float, float, float] | None = None

        self._rgb_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._depth_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.realtime_pacing = realtime_pacing
        self.viewer_handle = None
        self.frame_listeners: list = []
        self.real2sim_state = None
        self.show_real2sim_ignored_overlay: bool = True
        self.show_3d_camera_frustums: bool = True
        self._perceived_cluster_candidates: list = []
        self._perceived_fg_pts_world: np.ndarray | None = None
        self._cam_win_step: int = 0
        self._cam_windows = None
        self.show_camera_windows: bool = (
            (os.environ.get("BRICKCELL_CAMERA_WINDOWS", "1") != "0")
            if show_camera_windows is None
            else bool(show_camera_windows)
        )

        self.reset_arm_to_home()
        if len(self.spawned_blocks) > 1:
            for _ in range(45):
                mujoco.mj_step(self.model, self.data)
        if show_viewer:
            self.launch_viewer()

    @property
    def viewer(self):
        return self.viewer_handle

    def is_viewer_running(self) -> bool:
        return self.viewer_handle is not None and bool(self.viewer_handle.is_running())

    def launch_viewer(self) -> None:
        """Launch native MuJoCo 4.0.1 desktop Simulate application (`mujoco.viewer.launch_passive`)."""
        if self.viewer_handle is not None:
            return
        self.viewer_handle = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=True,
            show_right_ui=True,
        )
        with self.viewer_handle.lock():
            self.viewer_handle.cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
            self.viewer_handle.cam.lookat[:] = [0.18, 0.00, 0.06]
            self.viewer_handle.cam.distance = 0.62
            self.viewer_handle.cam.azimuth = 136.0
            self.viewer_handle.cam.elevation = -26.0
            self.viewer_handle.opt.geomgroup[3] = 0
        shown = _bring_win32_process_windows_to_front()
        if shown:
            print(f"[MuJoCo Desktop] Foreground windows: {shown}", flush=True)
        if self.show_camera_windows and self._cam_windows is None:
            from brickcell.block_inspector.studio_vision import VirtualCameraWindowsManager

            self._cam_windows = VirtualCameraWindowsManager()
            self._cam_windows.start()
            self._cam_windows.update_from_twin(self)
        self._sync_viewer()

    def _sync_viewer(self) -> None:
        if not self.is_viewer_running():
            return
        v = self.viewer_handle
        assert v is not None
        from brickcell.block_inspector.studio_vision import populate_studio_3d_scene

        with v.lock():
            populate_studio_3d_scene(self, v.user_scn)

            anom_str = "ANOMALY FLAGGED -> DEFECTIVE" if self.anomaly_flagged else "CLEAN -> GOOD"
            if len(self.spawned_blocks) > 1:
                left_txt = "SO-101 3-Plate Cell\nPlate Counts\nActive Brick"
                right_txt = (
                    f"{self.status_banner}\n"
                    f"Source left: {self.remaining_source_blocks} | "
                    f"Good: {self.sorted_counts['good']} | Defective: {self.sorted_counts['defective']}\n"
                    f"{self.block_spec.part_id} -> {anom_str} (views: {self.plate_scan_valid_frames})"
                )
            else:
                pile_str = (self.sorted_pile or "pending").upper()
                left_txt = "SO-101 Cell\nAnomaly Signal\nSorted Pile"
                right_txt = (
                    f"{self.status_banner}\n"
                    f"{anom_str} (valid Plate views: {self.plate_scan_valid_frames})\n"
                    f"{pile_str}"
                )

        v.set_texts(
            [
                (
                    int(mujoco.mjtFontScale.mjFONTSCALE_150),
                    int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
                    left_txt,
                    right_txt,
                )
            ]
        )
        v.sync()
        if self._cam_windows is not None:
            self._cam_win_step += 1
            if self._cam_win_step % 3 == 1:
                self._cam_windows.update_from_twin(self)

    def close(self) -> None:
        if self._cam_windows is not None:
            self._cam_windows.close()
            self._cam_windows = None
        for r in self._rgb_renderers.values():
            r.close()
        self._rgb_renderers.clear()
        for r in self._depth_renderers.values():
            r.close()
        self._depth_renderers.clear()
        if self.viewer_handle is not None:
            self.viewer_handle.close()
            self.viewer_handle = None

    def reset_arm_to_home(self) -> None:
        self.q_cmd = OBSERVATION_HOME_Q.copy()
        self.gripper_pct = 0.85
        self.data.qpos[self._arm_qpos_adr] = self.q_cmd
        self.data.qvel[self._arm_qvel_adr] = 0.0
        self.data.qpos[self._jaw_qpos_adr] = gripper_pct_to_jaw_rad(self.gripper_pct)
        self.data.ctrl[:5] = self.q_cmd
        self.data.ctrl[5] = gripper_pct_to_jaw_rad(self.gripper_pct)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(40):
            mujoco.mj_step(self.model, self.data)

    def current_arm_q(self) -> np.ndarray:
        return self.data.qpos[self._arm_qpos_adr].copy()

    def current_arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self._arm_qvel_adr].copy()

    def body3_world_pose(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.xmat[self._body3_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[self._body3_id]
        return T

    def ground_truth_block_pose(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.xmat[self._block_body_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[self._block_body_id]
        return T

    def jaw_positions_world(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.site_xpos[self._jaw_fixed_site].copy(),
            self.data.site_xpos[self._jaw_moving_site].copy(),
        )

    def step_control_30hz(self, q_target: np.ndarray, gripper_pct: float | None = None) -> None:
        """Execute one 30 Hz control period (17 physics substeps @ 2 ms)."""
        t0 = time.monotonic()
        self.q_cmd = np.asarray(q_target, dtype=np.float64).copy()
        if gripper_pct is not None:
            self.gripper_pct = float(np.clip(gripper_pct, 0.0, 1.0))
        self.data.ctrl[:5] = self.q_cmd
        self.data.ctrl[5] = gripper_pct_to_jaw_rad(self.gripper_pct)

        for listener in self.frame_listeners:
            listener(self)

        mujoco.mj_step(self.model, self.data, nstep=17)
        self._sync_viewer()
        if self.realtime_pacing and self.is_viewer_running():
            rem = (1.0 / 30.0) - (time.monotonic() - t0)
            if rem > 0.0:
                time.sleep(rem)

    def move_joint_trajectory(
        self,
        q_goal: np.ndarray,
        duration_s: float = 0.75,
        gripper_pct: float | None = None,
        settle_steps: int = 8,
    ) -> None:
        """Minimum-jerk cosine-interpolated joint trajectory at 30 Hz."""
        q_start = self.current_arm_q()
        q_end = np.asarray(q_goal, dtype=np.float64)
        n_steps = max(4, int(round(duration_s * 30.0)))
        for k in range(1, n_steps + 1):
            tau = k / n_steps
            s = 0.5 * (1.0 - math.cos(math.pi * tau))
            q_interp = q_start + s * (q_end - q_start)
            self.step_control_30hz(q_interp, gripper_pct=gripper_pct)
        for _ in range(settle_steps):
            self.step_control_30hz(q_end, gripper_pct=gripper_pct)

    def receive_anomaly_message(self, detail: str = "external_anomaly_signal") -> None:
        """Record an incoming anomaly message so the block routes to the defective pile."""
        self.anomaly_flagged = True
        self.anomaly_messages.append(str(detail))

    def reset_episode_state(self) -> None:
        """Clear per-episode scan, anomaly, and sorting state between standalone rollouts."""
        self.hand_empty = True
        self.at_rest = False
        self.active_target_corner = None
        self._active_clutter_choice = None
        self.source_plate_cleared = False
        self.remaining_source_blocks = len(self.spawned_blocks)
        self.sorted_counts = {"good": 0, "defective": 0}
        self.batch_sorted_records.clear()
        self.plate_scan_active = False
        self.plate_scan_completed = False
        self.plate_scan_valid_frames = 0
        self.plate_scan_keyframes.clear()
        self.anomaly_flagged = False
        self.anomaly_messages.clear()
        self.anomaly_untrusted_error = None
        self.max_plate_anomaly_distance = None
        self.sorted_pile = None
        self._T_body3_block_held = np.eye(4, dtype=np.float64)
        self.ledger.reset_all()

    def render_rgbd(self, camera_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        """Render camera RGB and metric depth (for D415s) at each camera's configured resolution."""
        cam: CameraModel = self.rig.by_name(camera_name)
        key = (cam.width, cam.height)
        rgb_r = self._rgb_renderers.get(key)
        if rgb_r is None:
            rgb_r = mujoco.Renderer(self.model, height=cam.height, width=cam.width)
            self._rgb_renderers[key] = rgb_r
        rgb_r.update_scene(self.data, camera=cam.name)
        rgb = rgb_r.render().copy()

        depth = None
        if cam.has_depth:
            dep_r = self._depth_renderers.get(key)
            if dep_r is None:
                dep_r = mujoco.Renderer(self.model, height=cam.height, width=cam.width)
                dep_r.enable_depth_rendering()
                self._depth_renderers[key] = dep_r
            dep_r.update_scene(self.data, camera=cam.name)
            depth = dep_r.render().copy()
        return rgb, depth

    def compute_corner_ray_errors_m(self, camera_name: str) -> np.ndarray:
        """Cast 3D rays via mujoco.mj_ray (groups 0 & 3) from camera optical center to all 8 corners."""
        cam = self.rig.by_name(camera_name)
        T_wc = cam.world_transform(self.body3_world_pose())
        p_cam = T_wc[:3, 3].astype(np.float64)
        corners_w = self.ledger.corners_world()

        geomgroup = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)
        geomid = np.array([-1], dtype=np.int32)
        errors = np.zeros(8, dtype=np.float64)
        for i in range(8):
            ray = corners_w[i] - p_cam
            dist_expected = float(np.linalg.norm(ray))
            vec = (ray / max(dist_expected, 1e-9)).astype(np.float64)
            hit_dist = float(
                mujoco.mj_ray(self.model, self.data, p_cam, vec, geomgroup, 1, -1, geomid)
            )
            if hit_dist < 0.0:
                errors[i] = 0.0
            else:
                # Positive occlusion if another surface intercepts strictly in front of the corner
                errors[i] = max(0.0, dist_expected - hit_dist)
        return errors

    def _refresh_reachability(self) -> None:
        if not self.ledger.localized:
            self._reachability_cache = {}
            return
        self._reachability_cache = self.ik.evaluate_upper_corners_reachability(
            self.ledger.T_world_block, self.block_spec
        )

    def corner_seed_q(self, corner_index: int) -> np.ndarray | None:
        sol = self._reachability_cache.get(corner_index)
        return sol.q_enc_rad.copy() if (sol is not None and sol.reachable) else None

    def perceive_symbolic_state(self, block_id: str = "block_0") -> SymbolicState:
        if not self.ledger.localized:
            return SymbolicState(
                block_id=block_id,
                hand_empty=self.hand_empty,
                block_localized=False,
                block_on_table=True,
                block_held=not self.hand_empty,
                block_at_rest=self.at_rest,
                corner_upper=(False,) * 8,
                corner_wrist_reachable=(False,) * 8,
                corner_inspected=(False,) * 8,
                plate_scan_completed=self.plate_scan_completed,
                anomaly_flagged=self.anomaly_flagged,
                sorted_pile=self.sorted_pile,
                source_plate_cleared=self.source_plate_cleared,
                remaining_source_blocks=self.remaining_source_blocks,
                sorted_good_count=self.sorted_counts["good"],
                sorted_defective_count=self.sorted_counts["defective"],
            )
        upper = self.ledger.upper_hemisphere_mask()
        inspected = self.ledger.inspected_mask()
        reachable = tuple(
            bool(
                upper[i] and i in self._reachability_cache and self._reachability_cache[i].reachable
            )
            for i in range(8)
        )
        return SymbolicState(
            block_id=block_id,
            hand_empty=self.hand_empty,
            block_localized=True,
            block_on_table=self.hand_empty and (self.sorted_pile is None),
            block_held=not self.hand_empty,
            block_at_rest=self.at_rest,
            corner_upper=tuple(bool(x) for x in upper),
            corner_wrist_reachable=reachable,
            corner_inspected=tuple(bool(x) for x in inspected),
            plate_scan_completed=self.plate_scan_completed,
            anomaly_flagged=self.anomaly_flagged,
            sorted_pile=self.sorted_pile,
            source_plate_cleared=self.source_plate_cleared,
            remaining_source_blocks=self.remaining_source_blocks,
            sorted_good_count=self.sorted_counts["good"],
            sorted_defective_count=self.sorted_counts["defective"],
        )

    def _capture_dual_d415_frames(self) -> list[tuple[np.ndarray, np.ndarray, CameraModel]]:
        rgb1, d1 = self.render_rgbd("realsense_1")
        rgb2, d2 = self.render_rgbd("realsense_2")
        assert d1 is not None and d2 is not None
        return [(d1, rgb1, self.rig.realsense_1), (d2, rgb2, self.rig.realsense_2)]

    def execute_operator(
        self, op: GroundedOperator, cone_perturbation: np.ndarray | None = None
    ) -> SkillOutcome:
        self.at_rest = False
        if op.name == "LocalizeNextClusterBlock":
            return self._skill_localize_next_cluster_block()
        if op.name == "LocalizeBlock":
            return self._skill_localize()
        if op.name == "PickBlock":
            return self._skill_pick_block()
        if op.name == "PresentAndSpinAtPlateCamera":
            return self._skill_present_and_spin_at_plate_camera()
        if op.name == "SortBlockToPile":
            return self._skill_sort_block_to_pile(str(op.params.get("pile", "good")))
        if op.name == "InspectExposedCorner":
            return self._skill_inspect_corner(int(op.params["corner_index"]), cone_perturbation)
        if op.name == "ReorientBlockYaw":
            return self._skill_reorient_yaw(
                float(op.params.get("delta_yaw_rad", math.pi)),
                target_corner=op.params.get("target_corner"),
            )
        if op.name == "GraspPresentAndFlipBlock":
            return self._skill_grasp_present_and_flip()
        if op.name == "ParkAtRest":
            return self._skill_park_at_rest()
        return SkillOutcome(False, reason=f"Unknown operator: {op.name}")

    def _match_nearest_sim_block_body(self, pos_world: np.ndarray) -> str:
        """Bind `_block_body_id` strictly for physical lift/settle verification (never defect labels)."""
        best_name = "block_0"
        best_dist = float("inf")
        for bname, bid in self.block_body_ids.items():
            d = float(np.linalg.norm(self.data.xpos[bid] - pos_world))
            if d < best_dist:
                best_dist = d
                best_name = bname
        self._block_body_id = self.block_body_ids[best_name]
        return best_name

    def _skill_localize_next_cluster_block(self) -> SkillOutcome:
        """Perceive clustered DUPLO bricks in aquamarine `plate_source` & pick clutter-safe grasp."""
        self.status_banner = (
            "LocalizeNextClusterBlock: Scanning aquamarine source plate via Dual-D415 RGB-D"
        )
        self.active_target_corner = None
        self._active_clutter_choice = None
        self.plate_scan_active = False
        self.plate_scan_completed = False
        self.plate_scan_valid_frames = 0
        self.plate_scan_keyframes.clear()
        self.anomaly_flagged = False
        self.anomaly_messages.clear()
        self.anomaly_untrusted_error = None
        self.max_plate_anomaly_distance = None
        self.sorted_pile = None
        self.ledger.reset_all()

        self.move_joint_trajectory(
            OBSERVATION_HOME_Q, duration_s=0.55, gripper_pct=0.72, settle_steps=10
        )
        frames = self._capture_dual_d415_frames()
        candidates, fg_pts = localize_clustered_blocks_in_source_plate(
            frames, source_plate_xy=SOURCE_PILE_XY, stride=2
        )
        if not candidates:
            candidates, fg_pts = localize_clustered_blocks_in_source_plate(
                frames, source_plate_xy=SOURCE_PILE_XY, stride=1
            )
        self._perceived_cluster_candidates = list(candidates)
        self._perceived_fg_pts_world = (
            fg_pts.copy() if fg_pts is not None and len(fg_pts) > 0 else None
        )
        self.remaining_source_blocks = len(candidates)
        if not candidates:
            self.source_plate_cleared = True
            self.status_banner = (
                f"Source Plate Empty! Sorted Good: {self.sorted_counts['good']} | "
                f"Defective: {self.sorted_counts['defective']}"
            )
            self._sync_viewer()
            if self._cam_windows is not None:
                self._cam_windows.update_from_twin(self)
            return SkillOutcome(
                True,
                reason="source_plate_empty",
                metadata={
                    "remaining_blocks": 0,
                    "sorted_counts": dict(self.sorted_counts),
                },
            )

        choice = self.ik.select_clutter_safe_topdown_grasp(
            candidates, fg_pts, source_plate_xy=SOURCE_PILE_XY
        )
        chosen = candidates[choice.candidate_index]
        self.source_plate_cleared = False
        self._active_clutter_choice = choice
        self.block_spec = chosen.block_spec
        self.ledger.block_spec = chosen.block_spec
        self.ledger.set_initial_pose(chosen.localized.T_world_block)
        self._reference_hsv = chosen.localized.mean_hsv
        self._match_nearest_sim_block_body(chosen.localized.T_world_block[:3, 3])
        self._refresh_reachability()
        self._sync_viewer()
        if self._cam_windows is not None:
            self._cam_windows.update_from_twin(self)

        pos_err_mm = 1000.0 * float(
            np.linalg.norm(
                chosen.localized.T_world_block[:3, 3] - self.ground_truth_block_pose()[:3, 3]
            )
        )
        return SkillOutcome(
            True,
            reason="cluster_block_localized",
            metadata={
                "remaining_blocks": len(candidates),
                "inferred_part": chosen.block_spec.name,
                "jaw_conflict_pts": choice.neighbor_intrusion_pts,
                "open_gripper_pct": round(choice.open_gripper_pct, 2),
                "icp_fitness": round(chosen.localized.icp_fitness, 3),
                "pos_err_vs_sim_mm": round(pos_err_mm, 2),
            },
        )

    def _skill_localize(self) -> SkillOutcome:
        self.status_banner = "LocalizeBlock: Retracting arm & fusing Dual-D415 RGB-D"
        self.active_target_corner = None
        self.move_joint_trajectory(
            OBSERVATION_HOME_Q, duration_s=0.65, gripper_pct=0.85, settle_steps=10
        )
        frames = self._capture_dual_d415_frames()
        loc = localize_resting_block(frames, block_spec=self.block_spec)
        self._perceived_fg_pts_world = (
            loc.inlier_points_world.copy()
            if getattr(loc, "inlier_points_world", None) is not None
            else None
        )
        self.ledger.set_initial_pose(loc.T_world_block)
        self._reference_hsv = loc.mean_hsv
        self._refresh_reachability()
        self._sync_viewer()
        if self._cam_windows is not None:
            self._cam_windows.update_from_twin(self)
        pos_err_mm = 1000.0 * float(
            np.linalg.norm(loc.T_world_block[:3, 3] - self.ground_truth_block_pose()[:3, 3])
        )
        return SkillOutcome(
            True,
            reason="localized",
            metadata={
                "icp_fitness": round(loc.icp_fitness, 3),
                "icp_rmse_mm": round(loc.icp_rmse_m * 1000.0, 2),
                "pos_err_vs_sim_mm": round(pos_err_mm, 2),
            },
        )

    def _skill_inspect_corner(
        self, corner_index: int, cone_perturbation: np.ndarray | None = None
    ) -> SkillOutcome:
        self.active_target_corner = corner_index
        self.status_banner = f"InspectExposedCorner(c_{corner_index}): Frax 5-DOF Cone Look-At"
        cached = self._reachability_cache.get(corner_index)
        if cone_perturbation is None and cached is not None and cached.reachable:
            sol = cached
        else:
            sol, _ = self.ik.solve_corner_lookat(
                corner_index,
                self.ledger.T_world_block,
                self.block_spec,
                q_seed=None,
                cone_perturbation=cone_perturbation,
            )
        if not sol.reachable:
            self.active_target_corner = None
            return SkillOutcome(False, reason=f"ik_unreachable(cost={sol.cost:.4f})")

        # Transit via safe overhead arch whenever changing arm configuration substantially
        q_goal = sol.q_enc_rad.copy()
        q_cur = self.current_arm_q()
        if float(np.linalg.norm(q_goal - q_cur)) > 0.65:
            q_arch = 0.5 * (q_cur + q_goal)
            q_arch[1] = min(float(q_cur[1]), float(q_goal[1]), -0.95)
            q_arch[2] = max(float(q_cur[2]), float(q_goal[2]), 0.80)
            q_arch[3] = min(float(q_arch[3]), 0.60)
            self.move_joint_trajectory(q_arch, duration_s=0.40, gripper_pct=0.22, settle_steps=2)

        # Smooth approach with tucked empty gripper (0.22) so moving jaw clears 25-deg optical cone
        self.move_joint_trajectory(q_goal, duration_s=0.60, gripper_pct=0.22, settle_steps=8)
        # Bounded integral servo trim eliminating steady-state gravity sag before rock-steady dwell
        q_trim = np.zeros(5, dtype=np.float64)
        for _ in range(8):
            q_trim = np.clip(q_trim + 0.45 * (q_goal - self.current_arm_q()), -0.06, 0.06)
            self.step_control_30hz(q_goal + q_trim, gripper_pct=0.22)

        # Hold fixed servo command rock-steady for dwell_frames verification (>= 3 consecutive crisp frames)
        q_hold = q_goal + q_trim
        for _ in range(6):
            self.step_control_30hz(q_hold, gripper_pct=0.22)
        for _ in range(10):
            self.step_control_30hz(q_hold, gripper_pct=0.22)
            if (
                float(np.max(np.abs(self.current_arm_qvel())))
                < self.ledger.config.max_joint_speed_rad_s
            ):
                rgb_w, _ = self.render_rgbd("wrist")
                ray_w = self.compute_corner_ray_errors_m("wrist")
                obs_w = self.ledger.evaluate_camera_frame(
                    "wrist",
                    rgb_w,
                    self.body3_world_pose(),
                    self.current_arm_qvel(),
                    ray_w,
                    block_held=False,
                    anomaly_scorer=self.anomaly_scorer,
                )
                self.ledger.update_step(wrist_obs=obs_w)
                if self.ledger.records[corner_index].inspected:
                    break

        ok = self.ledger.records[corner_index].inspected
        reason = "corner_verified" if ok else self.ledger.records[corner_index].last_reason
        self.active_target_corner = None
        self._sync_viewer()
        return SkillOutcome(
            ok,
            reason=reason,
            metadata={
                "standoff_cm": round(sol.standoff_m * 100.0, 2),
                "pointing_cos": round(sol.pointing_cos, 4),
            },
        )

    def _clamp_gripper_smooth(self, q_hold: np.ndarray) -> None:
        """Smoothly close jaws around block via contact friction without impact bounce."""
        start_pct = float(self.gripper_pct)
        for k in range(1, 11):
            self.step_control_30hz(q_hold, gripper_pct=start_pct * (1.0 - k / 10.0))
        for _ in range(10):
            self.step_control_30hz(q_hold, gripper_pct=0.0)

    def _release_gripper_smooth(self, q_hold: np.ndarray, open_pct: float = 0.96) -> None:
        """Smoothly open jaws so block settles cleanly under gravity."""
        for k in range(1, 9):
            self.step_control_30hz(q_hold, gripper_pct=open_pct * (k / 8.0))
        for _ in range(8):
            self.step_control_30hz(q_hold, gripper_pct=open_pct)

    def _execute_contact_topdown_yaw(
        self,
        delta_yaw_rad: float,
        target_xy: np.ndarray | None = None,
        target_jaw_yaw: float | None = None,
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """100% pure contact-physics top-down grasp, lift, yaw reorientation, and placement."""
        T_wb = self.ledger.T_world_block
        p_b = T_wb[:3, 3].copy()
        z_grasp = max(float(p_b[2]) + 0.004, 0.0175)
        grasp_sol = self.ik.select_best_symmetric_topdown_grasp(
            T_wb, self.block_spec, z_offset_m=z_grasp - float(p_b[2])
        )
        pre_sol = self.ik.solve_grasp_pose(
            np.array([p_b[0], p_b[1], z_grasp + 0.080]),
            jaw_yaw_world=grasp_sol.jaw_yaw_world,
            pitch_down_deg=90.0,
            q_posture=grasp_sol.q_grasp,
        )

        # 1. Retract via safe overhead home, open gripper wide, move to pre-grasp, descend, clamp jaws
        self.move_joint_trajectory(
            OBSERVATION_HOME_Q, duration_s=0.45, gripper_pct=0.96, settle_steps=4
        )
        self.move_joint_trajectory(
            pre_sol.q_grasp, duration_s=0.55, gripper_pct=0.96, settle_steps=6
        )
        self.move_joint_trajectory(
            grasp_sol.q_grasp, duration_s=0.50, gripper_pct=0.96, settle_steps=6
        )
        self._clamp_gripper_smooth(grasp_sol.q_grasp)
        self.hand_empty = False
        T_body3_grasp = self.body3_world_pose()

        # 2. Lift block strictly via rigid-body jaw contact friction (zero teleports!)
        self.move_joint_trajectory(
            pre_sol.q_grasp, duration_s=0.55, gripper_pct=0.0, settle_steps=8
        )
        lifted = float(self.ground_truth_block_pose()[2, 3]) > (float(p_b[2]) + 0.035)
        if not lifted:
            self._release_gripper_smooth(pre_sol.q_grasp)
            self.hand_empty = True
            return False, T_body3_grasp, T_body3_grasp

        # 3. Solve exact re-centered placement IK at target_xy and goal_jaw_yaw
        place_xy = p_b[:2] if target_xy is None else np.asarray(target_xy, dtype=np.float64)
        goal_jaw_yaw = (
            float(grasp_sol.jaw_yaw_world + delta_yaw_rad)
            if target_jaw_yaw is None
            else float(target_jaw_yaw)
        )
        place_pre_sol = self.ik.solve_grasp_pose(
            np.array([place_xy[0], place_xy[1], z_grasp + 0.080]),
            jaw_yaw_world=goal_jaw_yaw,
            pitch_down_deg=90.0,
            q_posture=pre_sol.q_grasp,
        )
        place_sol = self.ik.solve_grasp_pose(
            np.array([place_xy[0], place_xy[1], z_grasp + 0.003]),
            jaw_yaw_world=goal_jaw_yaw,
            pitch_down_deg=90.0,
            q_posture=place_pre_sol.q_grasp,
        )

        self.move_joint_trajectory(
            place_pre_sol.q_grasp, duration_s=0.65, gripper_pct=0.0, settle_steps=6
        )
        self.move_joint_trajectory(
            place_sol.q_grasp, duration_s=0.55, gripper_pct=0.0, settle_steps=6
        )
        T_body3_place = self.body3_world_pose()
        self._release_gripper_smooth(place_sol.q_grasp)
        self.hand_empty = True
        self.move_joint_trajectory(
            place_pre_sol.q_grasp, duration_s=0.45, gripper_pct=0.96, settle_steps=6
        )
        return True, T_body3_grasp, T_body3_place

    def _set_settled_block_pose(self, pos_world: np.ndarray, R_world: np.ndarray) -> None:
        """Episode-reset helper used only between standalone training episodes (never in skills)."""
        pos_adj = np.asarray(pos_world, dtype=np.float64).copy()
        if float(np.linalg.norm(pos_adj[:2] - SOURCE_PILE_XY)) <= 0.082:
            pos_adj[2] = max(
                float(pos_adj[2]),
                SOURCE_PLATE_FLOOR_Z_M + 0.5 * float(self.block_spec.dims_m[2]),
            )
        a, v = self._block_qpos_adr, self._block_qvel_adr
        self.data.qpos[a : a + 3] = pos_adj
        self.data.qpos[a + 3 : a + 7] = _rotmat_to_mj_quat(R_world)
        self.data.qvel[v : v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(15):
            mujoco.mj_step(self.model, self.data)
        self._sync_viewer()

    def sync_real2sim_scene(self, scene_state: object, settle_steps: int = 15) -> None:
        """Synchronize MuJoCo 4.0.1 brick & plate poses live from a `Real2SimSceneState`."""
        self.real2sim_state = scene_state
        plates = getattr(scene_state, "plates", {})
        for role, p_obj in plates.items():
            bname = f"plate_{role}"
            try:
                bid = int(self.model.body(bname).id)
                self.model.body_pos[bid] = np.asarray(p_obj.pos_world, dtype=np.float64)
            except Exception:
                pass

        src_xy = (
            np.asarray(plates["source"].pos_world[:2], dtype=np.float64)
            if "source" in plates
            else SOURCE_PILE_XY
        )
        bricks = getattr(scene_state, "bricks", [])
        for idx, b in enumerate(bricks):
            bname = f"block_{idx}"
            if bname not in self.block_body_ids:
                continue
            j_id = int(self.model.joint(f"{bname}_joint").id)
            q_adr = int(self.model.jnt_qposadr[j_id])
            v_adr = int(self.model.jnt_dofadr[j_id])
            T_wb = np.asarray(b.T_world_block, dtype=np.float64)
            pos = T_wb[:3, 3].copy()
            in_src = float(np.linalg.norm(pos[:2] - src_xy)) <= 0.082
            z_floor = SOURCE_PLATE_FLOOR_Z_M if in_src else 0.0
            half_world_z = float(
                np.sum(np.abs(T_wb[2, :3]) * (0.5 * np.asarray(b.spec.dims_m, dtype=np.float64)))
            )
            pos[2] = max(float(pos[2]), z_floor + half_world_z + 0.0005)
            self.data.qpos[q_adr : q_adr + 3] = pos
            self.data.qpos[q_adr + 3 : q_adr + 7] = _rotmat_to_mj_quat(T_wb[:3, :3])
            self.data.qvel[v_adr : v_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(max(0, int(settle_steps))):
            mujoco.mj_step(self.model, self.data)

        d_counts = (
            scene_state.distractor_counts_by_category()
            if hasattr(scene_state, "distractor_counts_by_category")
            else {}
        )
        ignored_total = sum(d_counts.values())
        self.status_banner = (
            f"Real2Sim Synced: {len(bricks)} DUPLO bricks & 3 KALAS plates "
            f"(Ignored {ignored_total} distractors: arm={d_counts.get('human_arm', 0)}, "
            f"wire={d_counts.get('wire', 0)}, trash={d_counts.get('trash', 0)})"
        )
        self._sync_viewer()

    def _skill_reorient_yaw(
        self, delta_yaw_rad: float, target_corner: int | None = None
    ) -> SkillOutcome:
        self.active_target_corner = None
        if target_corner is not None and self.ledger.localized:
            T_wb = self.ledger.T_world_block
            p_b = T_wb[:3, 3]
            b_w = corner_bisectors_local()[int(target_corner)] @ T_wb[:3, :3].T
            psi_robot = math.atan2(-float(p_b[1]), -float(p_b[0]))
            psi_corner = math.atan2(float(b_w[1]), float(b_w[0]))
            delta_yaw_rad = math.atan2(
                math.sin(psi_robot - psi_corner), math.cos(psi_robot - psi_corner)
            )
            if abs(delta_yaw_rad) < 0.35:
                delta_yaw_rad = 0.5 * math.pi
        self.status_banner = (
            f"ReorientBlockYaw({math.degrees(delta_yaw_rad):.0f} deg): Pure-contact lift & turn"
        )
        ok, T_grasp, T_place = self._execute_contact_topdown_yaw(delta_yaw_rad)
        self.move_joint_trajectory(
            OBSERVATION_HOME_Q, duration_s=0.60, gripper_pct=0.85, settle_steps=10
        )
        if not ok:
            return SkillOutcome(False, reason="contact_grasp_slipped")

        T_prior = self.ledger.propagate_manipulation_prior(T_grasp, T_place)
        frames = self._capture_dual_d415_frames()
        loc = relocalize_with_prior(
            frames, T_prior, self.block_spec.dims_m, reference_hsv=self._reference_hsv
        )
        locked = self.ledger.commit_post_manipulation_pose(
            loc.T_world_block, icp_fitness=loc.icp_fitness, patch_ncc=loc.color_consistency
        )
        self._refresh_reachability()
        self._sync_viewer()
        return SkillOutcome(
            True,
            reason="yaw_reoriented" if locked else "yaw_reoriented_with_slip_resweep",
            metadata={"icp_fitness": round(loc.icp_fitness, 3), "branch_locked": locked},
        )

    def _relocalize_and_commit_step(self, T_grasp: np.ndarray, T_place: np.ndarray) -> bool:
        """Retract to observation home and re-lock 6D block pose from Dual-D415 RGB-D ICP."""
        self.move_joint_trajectory(
            OBSERVATION_HOME_Q, duration_s=0.60, gripper_pct=0.85, settle_steps=10
        )
        T_prior = self.ledger.propagate_manipulation_prior(T_grasp, T_place)
        frames = self._capture_dual_d415_frames()
        loc = relocalize_with_prior(
            frames, T_prior, self.block_spec.dims_m, reference_hsv=self._reference_hsv
        )
        return self.ledger.commit_post_manipulation_pose(
            loc.T_world_block, icp_fitness=loc.icp_fitness, patch_ncc=loc.color_consistency
        )

    def _execute_pitched_90deg_tumble(
        self,
        place_xy: np.ndarray,
        new_half_z: float,
        stage_banner: str,
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """Pure-contact 45-deg radial grasp, mid-air presentation, 180-deg wrist-roll (=90-deg block tumble)."""
        self.status_banner = stage_banner
        T_wb = self.ledger.T_world_block
        p_b = T_wb[:3, 3].copy()
        z_grasp = max(float(p_b[2]) + 0.006, 0.019)

        g_sol = self.ik.solve_grasp_pose(
            np.array([p_b[0], p_b[1], z_grasp]),
            jaw_yaw_world=+0.5 * math.pi,
            pitch_down_deg=45.0,
        )
        pre_g_sol = self.ik.solve_grasp_pose(
            np.array([p_b[0] - 0.048, p_b[1], z_grasp + 0.055]),
            jaw_yaw_world=+0.5 * math.pi,
            pitch_down_deg=45.0,
            q_posture=g_sol.q_grasp,
        )
        pres_a = self.ik.solve_grasp_pose(
            np.array([0.210, 0.000, 0.125]),
            jaw_yaw_world=+0.5 * math.pi,
            pitch_down_deg=45.0,
            q_posture=g_sol.q_grasp,
        )
        pres_b = self.ik.solve_grasp_pose(
            np.array([0.210, 0.000, 0.125]),
            jaw_yaw_world=-0.5 * math.pi,
            pitch_down_deg=45.0,
        )
        z_place = max(float(new_half_z) + 0.006, 0.020)
        p_sol = self.ik.solve_grasp_pose(
            np.array([place_xy[0], place_xy[1], z_place]),
            jaw_yaw_world=-0.5 * math.pi,
            pitch_down_deg=45.0,
            q_posture=pres_b.q_grasp,
        )
        pre_p_sol = self.ik.solve_grasp_pose(
            np.array([place_xy[0] - 0.048, place_xy[1], z_place + 0.055]),
            jaw_yaw_world=-0.5 * math.pi,
            pitch_down_deg=45.0,
            q_posture=p_sol.q_grasp,
        )

        self.move_joint_trajectory(
            OBSERVATION_HOME_Q, duration_s=0.45, gripper_pct=0.96, settle_steps=4
        )
        self.move_joint_trajectory(
            pre_g_sol.q_grasp, duration_s=0.55, gripper_pct=0.96, settle_steps=6
        )
        self.move_joint_trajectory(g_sol.q_grasp, duration_s=0.50, gripper_pct=0.96, settle_steps=6)
        self._clamp_gripper_smooth(g_sol.q_grasp)
        self.hand_empty = False
        T_grasp = self.body3_world_pose()

        # Lift into mid-air presentation pose strictly via contact friction
        self.move_joint_trajectory(pres_a.q_grasp, duration_s=0.60, gripper_pct=0.0, settle_steps=8)
        lifted = float(self.ground_truth_block_pose()[2, 3]) > 0.065
        if not lifted:
            self._release_gripper_smooth(pres_a.q_grasp)
            self.hand_empty = True
            return False, T_grasp, T_grasp

        # Roll 180 deg around 45-deg inclined axis => presents underside & tumbles block 90 deg!
        self.move_joint_trajectory(pres_b.q_grasp, duration_s=0.70, gripper_pct=0.0, settle_steps=8)
        self.move_joint_trajectory(
            pre_p_sol.q_grasp, duration_s=0.50, gripper_pct=0.0, settle_steps=4
        )
        self.move_joint_trajectory(p_sol.q_grasp, duration_s=0.50, gripper_pct=0.0, settle_steps=6)
        T_place = self.body3_world_pose()
        self._release_gripper_smooth(p_sol.q_grasp)
        self.hand_empty = True
        self.move_joint_trajectory(
            pre_p_sol.q_grasp, duration_s=0.45, gripper_pct=0.96, settle_steps=6
        )
        return True, T_grasp, T_place

    def _skill_grasp_present_and_flip(self) -> SkillOutcome:
        """100% pure contact-physics 2-stage 45-deg tumble inverting the block 180 deg onto its studs."""
        self.active_target_corner = None
        work_xy = np.array([0.210, 0.000], dtype=np.float64)
        ref_45 = self.ik.solve_grasp_pose(
            np.array([work_xy[0], work_xy[1], 0.020]),
            jaw_yaw_world=+0.5 * math.pi,
            pitch_down_deg=45.0,
        )
        desired_jaw_yaw = ref_45.jaw_yaw_world
        jaw_vec = np.array([math.cos(desired_jaw_yaw), math.sin(desired_jaw_yaw), 0.0])

        # Step 0: Pre-align graspable width with 45-deg jaw axis & center at work_xy if needed
        T_wb = self.ledger.T_world_block
        best_dot = -1.0
        best_yaw = desired_jaw_yaw
        for idx, span in enumerate(self.block_spec.dims_m):
            ax = T_wb[:3, idx]
            if abs(float(ax[2])) < 0.25 and span <= 0.048:
                for sgn in (1.0, -1.0):
                    d = float(np.dot(sgn * ax, jaw_vec))
                    if d > best_dot:
                        best_dot = d
                        best_yaw = math.atan2(float(sgn * ax[1]), float(sgn * ax[0]))
        xy_dist = float(np.linalg.norm(T_wb[:2, 3] - work_xy))
        if best_dot < math.cos(math.radians(8.0)) or xy_dist > 0.015:
            self.status_banner = (
                "GraspPresentAndFlipBlock: Centering & aligning block for contact tumble"
            )
            dyaw_align = math.atan2(
                math.sin(desired_jaw_yaw - best_yaw), math.cos(desired_jaw_yaw - best_yaw)
            )
            ok_al, T_ga, T_pa = self._execute_contact_topdown_yaw(
                dyaw_align, target_xy=work_xy, target_jaw_yaw=desired_jaw_yaw
            )
            if not ok_al:
                self.move_joint_trajectory(
                    OBSERVATION_HOME_Q, duration_s=0.55, gripper_pct=0.85, settle_steps=8
                )
                return SkillOutcome(False, reason="pre_align_grasp_slipped")
            self._relocalize_and_commit_step(T_ga, T_pa)

        # Compute radial horizontal brick dimension (becomes vertical height after Stage 1 90-deg roll)
        T_s1 = self.ledger.T_world_block
        radial_vec = np.array([-jaw_vec[1], jaw_vec[0], 0.0], dtype=np.float64)
        radial_idx = int(np.argmax(np.abs(T_s1[:3, :3].T @ radial_vec)))
        side_half_z = 0.5 * float(self.block_spec.dims_m[radial_idx])

        # Stage 1: 45-deg grasp -> mid-air underside presentation -> 90-deg roll onto side wall
        ok1, T_g1, T_p1 = self._execute_pitched_90deg_tumble(
            place_xy=work_xy,
            new_half_z=side_half_z,
            stage_banner="GraspPresentAndFlipBlock (1/3): Pure-contact lift, present & 90-deg side roll",
        )
        if not ok1:
            self.move_joint_trajectory(
                OBSERVATION_HOME_Q, duration_s=0.55, gripper_pct=0.85, settle_steps=8
            )
            return SkillOutcome(False, reason="stage1_tumble_slipped")
        self._relocalize_and_commit_step(T_g1, T_p1)

        # Stage 2: 180-deg pure-contact top-down yaw turn of side-resting brick
        self.status_banner = "GraspPresentAndFlipBlock (2/3): Pure-contact 180-deg yaw turn on side"
        ok_m, T_gm, T_pm = self._execute_contact_topdown_yaw(math.pi, target_xy=work_xy)
        if not ok_m:
            self.move_joint_trajectory(
                OBSERVATION_HOME_Q, duration_s=0.55, gripper_pct=0.85, settle_steps=8
            )
            return SkillOutcome(False, reason="mid_yaw_slipped")
        self._relocalize_and_commit_step(T_gm, T_pm)

        # Stage 3: 45-deg grasp -> mid-air presentation -> second 90-deg roll onto top studs (-Z_B up!)
        ok2, T_g2, T_p2 = self._execute_pitched_90deg_tumble(
            place_xy=np.array([0.205, 0.000], dtype=np.float64),
            new_half_z=0.5 * float(self.block_spec.dims_m[2]),
            stage_banner="GraspPresentAndFlipBlock (3/3): Completing 180-deg pure-contact inversion",
        )
        if not ok2:
            self.move_joint_trajectory(
                OBSERVATION_HOME_Q, duration_s=0.55, gripper_pct=0.85, settle_steps=8
            )
            return SkillOutcome(False, reason="stage2_tumble_slipped")

        locked = self._relocalize_and_commit_step(T_g2, T_p2)
        self._refresh_reachability()
        self._sync_viewer()
        inverted_ok = float(self.ledger.T_world_block[2, 2]) < -0.5
        return SkillOutcome(
            inverted_ok,
            reason=(
                ("flipped_and_relocalized" if locked else "flipped_with_slip_resweep")
                if inverted_ok
                else "inversion_incomplete"
            ),
            metadata={
                "branch_locked": locked,
                "z_up_cos": round(float(self.ledger.T_world_block[2, 2]), 3),
            },
        )

    def _skill_pick_block(self) -> SkillOutcome:
        """100% pure contact-physics top-down pick and lift of the localized block."""
        self.active_target_corner = None
        self.plate_scan_active = False
        T_wb = self.ledger.T_world_block
        p_b = T_wb[:3, 3].copy()

        choice = self._active_clutter_choice
        self._active_clutter_choice = None
        if choice is not None:
            grasp_sol = choice.grasp_sol
            open_pct = float(choice.open_gripper_pct)
            z_grasp = float(grasp_sol.p_tcp_world[2])
            p_pre_xy = grasp_sol.p_tcp_world[:2]
        else:
            z_grasp = max(float(p_b[2]) + 0.004, 0.0175)
            grasp_sol = self.ik.select_best_symmetric_topdown_grasp(
                T_wb, self.block_spec, z_offset_m=z_grasp - float(p_b[2])
            )
            open_pct = 0.96
            p_pre_xy = grasp_sol.p_tcp_world[:2]

        pre_sol = self.ik.solve_grasp_pose(
            np.array([p_pre_xy[0], p_pre_xy[1], z_grasp + 0.095]),
            jaw_yaw_world=grasp_sol.jaw_yaw_world,
            pitch_down_deg=90.0,
            q_posture=grasp_sol.q_grasp,
        )

        self.status_banner = (
            f"PickBlock({self.block_spec.part_id}): Descending & clamping via contact friction"
        )
        self.move_joint_trajectory(
            OBSERVATION_HOME_Q, duration_s=0.40, gripper_pct=open_pct, settle_steps=4
        )
        self.move_joint_trajectory(
            pre_sol.q_grasp, duration_s=0.55, gripper_pct=open_pct, settle_steps=6
        )
        self.move_joint_trajectory(
            grasp_sol.q_grasp, duration_s=0.50, gripper_pct=open_pct, settle_steps=6
        )
        self._clamp_gripper_smooth(grasp_sol.q_grasp)
        T_body3_grasp = self.body3_world_pose()
        self._T_body3_block_held = np.linalg.inv(T_body3_grasp) @ T_wb

        self.status_banner = f"PickBlock({self.block_spec.part_id}): Lifting cleanly off plate"
        self.move_joint_trajectory(
            pre_sol.q_grasp, duration_s=0.55, gripper_pct=0.0, settle_steps=8
        )

        # In multi-block clutter, lock `_block_body_id` to whichever physical body lifted with the jaws
        jaw_f, jaw_m = self.jaw_positions_world()
        jaw_mid = 0.5 * (jaw_f + jaw_m)
        if len(self.block_body_ids) > 1:
            best_bid = self._block_body_id
            best_d = float("inf")
            for bid in self.block_body_ids.values():
                pb_sim = self.data.xpos[bid]
                if float(pb_sim[2]) > 0.050:
                    d_jaw = float(np.linalg.norm(pb_sim - jaw_mid))
                    if d_jaw < 0.050 and d_jaw < best_d:
                        best_d = d_jaw
                        best_bid = bid
            self._block_body_id = best_bid

        z_now = float(self.ground_truth_block_pose()[2, 3])
        lifted = z_now > (float(p_b[2]) + 0.035)
        if not lifted:
            self._release_gripper_smooth(pre_sol.q_grasp, open_pct=open_pct)
            self.hand_empty = True
            if len(self.spawned_blocks) > 1:
                self.ledger.reset_all()
            return SkillOutcome(False, reason="pick_grasp_slipped")

        self.hand_empty = False
        self._sync_viewer()
        return SkillOutcome(
            True,
            reason="block_picked",
            metadata={
                "lifted_z_cm": round(z_now * 100.0, 2),
                "open_gripper_pct": round(open_pct, 2),
            },
        )

    def _skill_present_and_spin_at_plate_camera(self) -> SkillOutcome:
        """Move held block to low Plate camera (`side`) and rotate Joint 5 counter-clockwise."""
        if self.hand_empty:
            return SkillOutcome(False, reason="block_not_held")

        # Validate untrusted detector configuration upfront (AGENTS.md safety rule)
        if self.anomaly_scorer is not None and self.anomaly_threshold is None:
            self.anomaly_untrusted_error = "missing_anomaly_threshold"
            return SkillOutcome(False, reason="missing_anomaly_threshold")

        radial_az = math.atan2(float(PLATE_CAMERA_SCAN_POS[1]), float(PLATE_CAMERA_SCAN_POS[0]))
        pres_sol = self.ik.solve_grasp_pose(
            PLATE_CAMERA_SCAN_POS,
            jaw_yaw_world=radial_az + 0.5 * math.pi,
            pitch_down_deg=55.0,
        )
        q_scan_start = pres_sol.q_grasp.copy()
        q_scan_start[4] = max(
            float(self.ik.cal_kin.calibration.lower[4]) + 0.05, JOINT5_CCW_START_RAD
        )
        q_scan_end = pres_sol.q_grasp.copy()
        q_scan_end[4] = min(float(self.ik.cal_kin.calibration.upper[4]) - 0.05, JOINT5_CCW_END_RAD)

        # Smooth elevated transit from pick lift over to low Plate camera (`side`)
        q_carry = 0.5 * (self.current_arm_q() + q_scan_start)
        q_carry[1] = min(float(self.current_arm_q()[1]), float(q_scan_start[1]), -0.45)
        q_carry[4] = 0.5 * float(self.current_arm_q()[4] + q_scan_start[4])
        self.status_banner = "PresentAndSpinAtPlateCamera: Moving to low Plate camera (`side`)"
        self.move_joint_trajectory(q_carry, duration_s=0.50, gripper_pct=0.0, settle_steps=2)
        self.move_joint_trajectory(q_scan_start, duration_s=0.65, gripper_pct=0.0, settle_steps=8)

        # Rotate Joint 5 (`Revolute_4` / wrist_roll) strictly counter-clockwise (+delta q_4 > 0)
        self.plate_scan_active = True
        self.plate_scan_valid_frames = 0
        self.plate_scan_keyframes.clear()
        n_spin_steps = 64
        plate_cam = self.rig.realsense_2
        for k in range(n_spin_steps + 1):
            tau = k / n_spin_steps
            q_step = q_scan_start.copy()
            q_step[4] = float(q_scan_start[4] + tau * (q_scan_end[4] - q_scan_start[4]))
            swept_deg = math.degrees(q_step[4] - q_scan_start[4])
            self.status_banner = (
                f"PresentAndSpinAtPlateCamera: Joint 5 CCW +{swept_deg:.0f} deg | "
                f"{'ANOMALY RECEIVED' if self.anomaly_flagged else 'No anomaly'}"
            )
            self.step_control_30hz(q_step, gripper_pct=0.0)

            if k % 4 == 0:
                rgb_p, depth_p = self.render_rgbd("realsense_2")
                T_held = self.body3_world_pose() @ self._T_body3_block_held
                uv_c, z_c = plate_cam.project_world_points(T_held[:3, 3][None, :])
                u_px, v_px, z_exp = float(uv_c[0, 0]), float(uv_c[0, 1]), float(z_c[0])
                in_frame = (
                    24.0 <= u_px <= (plate_cam.width - 24.0)
                    and 24.0 <= v_px <= (plate_cam.height - 24.0)
                    and 0.12 <= z_exp <= 0.60
                )
                if in_frame and depth_p is not None:
                    u0 = max(0, int(round(u_px)) - 48)
                    u1 = min(plate_cam.width, int(round(u_px)) + 48)
                    v0 = max(0, int(round(v_px)) - 48)
                    v1 = min(plate_cam.height, int(round(v_px)) + 48)
                    roi_rgb = rgb_p[v0:v1, u0:u1]
                    roi_dep = depth_p[v0:v1, u0:u1]
                    fg_ok = bool(np.any((roi_dep > 0.10) & (roi_dep < 0.55)))
                    if fg_ok and roi_rgb.size > 0 and float(roi_rgb.std()) > 4.0:
                        self.plate_scan_valid_frames += 1
                        if len(self.plate_scan_keyframes) < 4 and k in {0, 20, 40, 60}:
                            self.plate_scan_keyframes.append(roi_rgb.copy())
                        if self.anomaly_scorer is not None:
                            try:
                                fg_m = (roi_dep > max(0.08, z_exp - 0.065)) & (
                                    roi_dep < (z_exp + 0.065)
                                )
                                bg_iso = np.array([28, 32, 38], dtype=np.uint8)
                                roi_isolated = np.where(fg_m[:, :, None], roi_rgb, bg_iso)
                                raw_dist = float(self.anomaly_scorer(roi_isolated))
                                if not math.isfinite(raw_dist):
                                    raise ValueError("Non-finite anomaly distance")
                                self.max_plate_anomaly_distance = (
                                    raw_dist
                                    if self.max_plate_anomaly_distance is None
                                    else max(self.max_plate_anomaly_distance, raw_dist)
                                )
                                assert self.anomaly_threshold is not None
                                if raw_dist > float(self.anomaly_threshold):
                                    self.receive_anomaly_message(
                                        f"plate_detector_distance={raw_dist:.4f}>{self.anomaly_threshold:.4f}"
                                    )
                            except Exception as exc:
                                self.anomaly_untrusted_error = f"detector_error:{exc}"
                                break

                if self.anomaly_message_fn is not None:
                    try:
                        msg = self.anomaly_message_fn(self, k, float(q_step[4]), rgb_p)
                        if msg:
                            self.receive_anomaly_message(
                                msg if isinstance(msg, str) else "anomaly_callback_triggered"
                            )
                    except Exception as exc:
                        self.anomaly_untrusted_error = f"message_fn_error:{exc}"
                        break

        self.plate_scan_active = False
        still_held = float(self.ground_truth_block_pose()[2, 3]) > 0.030
        if not still_held:
            self.hand_empty = True
            self._sync_viewer()
            return SkillOutcome(False, reason="dropped_during_plate_spin")
        if self.anomaly_untrusted_error is not None:
            self._sync_viewer()
            return SkillOutcome(False, reason=self.anomaly_untrusted_error)
        if self.plate_scan_valid_frames < 8:
            self._sync_viewer()
            return SkillOutcome(
                False,
                reason=f"invalid_plate_camera_views({self.plate_scan_valid_frames})",
            )

        self.plate_scan_completed = True
        self._sync_viewer()
        return SkillOutcome(
            True,
            reason="plate_ccw_scan_completed",
            metadata={
                "joint5_ccw_deg": round(math.degrees(q_scan_end[4] - q_scan_start[4]), 1),
                "valid_plate_views": self.plate_scan_valid_frames,
                "anomaly_flagged": self.anomaly_flagged,
                "anomaly_messages": list(self.anomaly_messages),
            },
        )

    def _skill_sort_block_to_pile(self, pile: str) -> SkillOutcome:
        """Drop held block into `good` (green, y=-0.22) or `defective` (pink, y=+0.22) pile."""
        if self.hand_empty or not self.plate_scan_completed or self.anomaly_untrusted_error:
            return SkillOutcome(False, reason="untrusted_or_unscanned_sort_refused")
        if pile not in {"good", "defective"}:
            return SkillOutcome(False, reason=f"invalid_pile:{pile}")

        base_xy = DEFECTIVE_PILE_XY if pile == "defective" else GOOD_PILE_XY
        slot_idx = int(self.sorted_counts[pile]) % len(PLATE_DROP_SLOT_OFFSETS_XY)
        if len(self.spawned_blocks) > 1:
            dx, dy = PLATE_DROP_SLOT_OFFSETS_XY[slot_idx]
            target_xy = np.array([base_xy[0] + dx, base_xy[1] + dy], dtype=np.float64)
            z_release = 0.052
            release_open_pct = 0.55
        else:
            target_xy = base_xy.copy()
            z_release = 0.038
            release_open_pct = 0.65

        drop_yaw = math.atan2(float(target_xy[1]), float(target_xy[0])) + 0.5 * math.pi
        above_sol = self.ik.solve_grasp_pose(
            np.array([target_xy[0], target_xy[1], 0.120]),
            jaw_yaw_world=drop_yaw,
            pitch_down_deg=90.0,
        )
        release_sol = self.ik.solve_grasp_pose(
            np.array([target_xy[0], target_xy[1], z_release]),
            jaw_yaw_world=drop_yaw,
            pitch_down_deg=90.0,
            q_posture=above_sol.q_grasp,
        )

        self.status_banner = (
            f"SortBlockToPile({pile.upper()} slot #{slot_idx + 1}): Releasing into KALAS plate"
        )
        # Gently center wrist roll before swinging across workspace from Plate camera (-Y)
        q_unwind = self.current_arm_q()
        q_unwind[4] = float(above_sol.q_grasp[4])
        self.move_joint_trajectory(q_unwind, duration_s=0.40, gripper_pct=0.0, settle_steps=3)
        if pile == "defective":
            # Transit high (z=16.5 cm) over center workspace from Plate camera (-Y) to Defective plate (+Y)
            mid_sol = self.ik.solve_grasp_pose(
                np.array([0.16, 0.00, 0.165], dtype=np.float64),
                jaw_yaw_world=drop_yaw,
                pitch_down_deg=90.0,
                q_posture=above_sol.q_grasp,
            )
            self.move_joint_trajectory(
                mid_sol.q_grasp, duration_s=0.60, gripper_pct=0.0, settle_steps=4
            )

        self.move_joint_trajectory(
            above_sol.q_grasp, duration_s=0.60, gripper_pct=0.0, settle_steps=6
        )
        self.move_joint_trajectory(
            release_sol.q_grasp, duration_s=0.45, gripper_pct=0.0, settle_steps=6
        )
        self._release_gripper_smooth(release_sol.q_grasp, open_pct=release_open_pct)
        self.hand_empty = True
        self.move_joint_trajectory(
            above_sol.q_grasp, duration_s=0.45, gripper_pct=0.72, settle_steps=10
        )

        p_final = self.ground_truth_block_pose()[:3, 3]
        xy_err = float(np.linalg.norm(p_final[:2] - base_xy))
        settled_on_plate = xy_err <= 0.088 and float(p_final[2]) < 0.075
        if not settled_on_plate:
            self._sync_viewer()
            return SkillOutcome(
                False,
                reason=f"missed_plate(xy_err_mm={xy_err * 1000.0:.1f})",
            )

        self.sorted_pile = pile
        self.sorted_counts[pile] += 1
        self.remaining_source_blocks = max(0, self.remaining_source_blocks - 1)

        # Record post-hoc batch audit entry (never read by perception, planner, or detector)
        matched_spawn = next(
            (
                sb
                for sb in self.spawned_blocks
                if self.block_body_ids.get(sb.block_id) == self._block_body_id
            ),
            None,
        )
        self.batch_sorted_records.append(
            {
                "order": len(self.batch_sorted_records) + 1,
                "sim_block_name": matched_spawn.block_id if matched_spawn else "block_0",
                "sim_part_name": (
                    matched_spawn.spec.name if matched_spawn else self.block_spec.name
                ),
                "inferred_part_name": self.block_spec.name,
                "sim_has_visual_defect": (
                    bool(matched_spawn.has_visual_defect) if matched_spawn else False
                ),
                "anomaly_flagged": bool(self.anomaly_flagged),
                "max_plate_anomaly_distance": (
                    round(float(self.max_plate_anomaly_distance), 4)
                    if self.max_plate_anomaly_distance is not None
                    else None
                ),
                "sorted_pile": pile,
                "slot_idx": slot_idx,
                "plate_xy_err_mm": round(xy_err * 1000.0, 2),
                "keyframe_rgb": (
                    self.plate_scan_keyframes[min(1, len(self.plate_scan_keyframes) - 1)].copy()
                    if self.plate_scan_keyframes
                    else None
                ),
            }
        )
        self._sync_viewer()
        return SkillOutcome(
            True,
            reason=f"sorted_to_{pile}",
            metadata={
                "pile": pile,
                "slot_idx": slot_idx,
                "plate_xy_err_mm": round(xy_err * 1000.0, 2),
                "sorted_counts": dict(self.sorted_counts),
            },
        )

    def _skill_park_at_rest(self) -> SkillOutcome:
        self.active_target_corner = None
        self.plate_scan_active = False
        if len(self.spawned_blocks) > 1:
            self.status_banner = (
                f"ParkAtRest: All {sum(self.sorted_counts.values())} blocks sorted "
                f"(Good: {self.sorted_counts['good']} | Defective: {self.sorted_counts['defective']})"
            )
        else:
            self.status_banner = f"ParkAtRest: Sorted -> {(self.sorted_pile or 'table').upper()} — retracting to rest"
        self.move_joint_trajectory(
            OBSERVATION_HOME_Q, duration_s=0.60, gripper_pct=0.50, settle_steps=10
        )
        self.at_rest = True
        self._sync_viewer()
        return SkillOutcome(True, reason="parked_at_rest")
