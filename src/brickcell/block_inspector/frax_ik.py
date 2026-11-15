"""Calibrated 5-DOF SO-101 Frax URDF, spherized collisions, and Look-At / Grasp IK."""

from __future__ import annotations

import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# Configure XLA CPU precision & low-latency flags before importing jax/frax
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED",
)

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from frax import Manipulator
from frax.ik import (
    DiffIKSolver,
    FloorConstraint,
    GlobalIKSolver,
    JointLimits,
    JointTask,
    ObstacleConstraint,
    ResidualTask,
    SelfCollisionConstraint,
    SphereObstacles,
)
from frax.utils.collision_utils import bubblify_to_mine

from brickcell.block_inspector.geometry import (
    NOMINAL_TCP_OFFSET_BODY3,
    REPO_ROOT,
    T_BASE_WORLD,
    T_WORLD_BASE,
    BlockSpec,
    corner_facet_normals_world,
    corner_positions_world,
    corner_upper_hemisphere_mask,
    nominal_canted_wrist_cam_transform,
)

jax.config.update("jax_enable_x64", True)
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from so101_kinematics import ARM_JOINTS, CalibratedSO101Kinematics  # noqa: E402

SO101_ARM_JOINTS = list(ARM_JOINTS)
SO101_CHILD_MAP = {
    "Revolute_1": "Body1_2",
    "Revolute_2": "Group 1",
    "Revolute_3": "Body1_3",
    "Revolute_2_5": "Group 1_2",
    "Revolute_4": "Body3",
}
SO101_SC_DATA = (
    ("Body5_1", 0, "Body1_3", 0, 0.0),
    ("Body5_1", 0, "Group 1_2", 0, 0.0),
    ("Body5_1", 0, "Body3", 0, 0.0),
    ("Body5_1", 0, "Body3", 1, 0.0),
    ("Body1_2", 0, "Body3", 0, 0.0),
)

T_WORLD_BASE_JAX = jnp.asarray(T_WORLD_BASE, dtype=jnp.float64)
R_BASE_WORLD_JAX = jnp.asarray(T_BASE_WORLD[:3, :3], dtype=jnp.float64)


def export_canonical_frax_assets(
    output_dir: str | Path | None = None,
    T_body3_cam_np: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Export calibrated 5-DOF SO-101 URDF (finite velocity/effort/mass) and bubblify spheres YAML."""
    out = Path(output_dir) if output_dir else (REPO_ROOT / "artifacts" / "block_inspector")
    out.mkdir(parents=True, exist_ok=True)
    urdf_path = out / "so101_follower_frax.urdf"
    spheres_path = out / "so101_spheres.yaml"

    cal_kin = CalibratedSO101Kinematics("follower")
    cal_kin.export_arm_urdf(urdf_path)

    priors_file = (
        REPO_ROOT / "configs" / "published_priors_verified.json"
        if (REPO_ROOT / "configs" / "published_priors_verified.json").is_file()
        else REPO_ROOT / "artifacts" / "dynamics" / "published_priors_verified.json"
    )
    priors = (
        json.loads(priors_file.read_text(encoding="utf-8"))["links"] if priors_file.exists() else {}
    )

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        lname = link.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        row = priors.get(lname)
        if row and float(row.get("mass_kg", 0.0)) > 1e-6:
            mass = float(row["mass_kg"])
            com = row["com_m"]
            mat = row.get("inertia_matrix_kg_m2") or np.diag(row["diagonal_inertia_kg_m2"]).tolist()
            inertial.find("origin").set("xyz", f"{com[0]:.12g} {com[1]:.12g} {com[2]:.12g}")
            inertial.find("mass").set("value", f"{mass:.12g}")
            ine = inertial.find("inertia")
            ine.set("ixx", f"{mat[0][0]:.12g}")
            ine.set("iyy", f"{mat[1][1]:.12g}")
            ine.set("izz", f"{mat[2][2]:.12g}")
            ine.set("ixy", f"{mat[0][1]:.12g}")
            ine.set("ixz", f"{mat[0][2]:.12g}")
            ine.set("iyz", f"{mat[1][2]:.12g}")
        else:
            inertial.find("mass").set("value", "0.05")
            ine = inertial.find("inertia")
            for k in ("ixx", "iyy", "izz"):
                ine.set(k, "2e-5")

    for joint in root.findall("joint"):
        if joint.get("name") in SO101_ARM_JOINTS:
            joint.set("type", "revolute")
            limit = joint.find("limit")
            if limit is not None:
                limit.set("velocity", "1.534")  # 1000 ticks/s * 2*pi/4096
                limit.set("effort", "2.94")

    ET.indent(root)
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

    # Calibrated collision spheres in each link's local frame (jaw-tip sphere kept last for grasp_floor_idxs)
    body3_spheres: list[dict] = [
        {"center": [0.0, 0.005, -0.025], "radius": 0.024},
        {"center": [0.0, 0.000, -0.068], "radius": 0.020},
    ]
    if T_body3_cam_np is not None:
        cam_p = np.asarray(T_body3_cam_np, dtype=np.float64)[:3, 3]
        if float(np.linalg.norm(cam_p)) > 0.08:
            body3_spheres.append({"center": [float(v) for v in cam_p], "radius": 0.022})
    body3_spheres.append({"center": [0.0, 0.000, -0.092], "radius": 0.022})

    spheres_dict = {
        "collision_spheres": {
            "Body5_1": [{"center": [0.0, 0.0, 0.032], "radius": 0.042}],
            "Body1_2": [{"center": [0.0, 0.0, 0.025], "radius": 0.028}],
            "Group 1": [
                {"center": [0.0, -0.045, 0.0], "radius": 0.024},
                {"center": [0.0, -0.095, 0.0], "radius": 0.024},
            ],
            "Body1_3": [
                {"center": [0.0, -0.045, 0.0], "radius": 0.022},
                {"center": [0.0, -0.095, 0.0], "radius": 0.022},
            ],
            "Group 1_2": [{"center": [0.0, 0.0, -0.015], "radius": 0.022}],
            "Body3": body3_spheres,
        }
    }
    spheres_path.write_text(yaml.safe_dump(spheres_dict, sort_keys=False), encoding="utf-8")
    return urdf_path, spheres_path


export_calibrated_frax_assets = export_canonical_frax_assets


def so101_manipulability_5d(robot: Manipulator, q: jax.Array) -> jax.Array:
    """Full-rank 5x5 Gram manipulability sqrt(det(J^T J)) for 5-DOF SO-101 (nv=5)."""
    J = robot.ee_jacobian(q)  # (6, 5)
    gram_5x5 = J.T @ J + 1e-9 * jnp.eye(5, dtype=J.dtype)
    return jnp.sqrt(jnp.linalg.det(gram_5x5))


def so101_antipodal_safe_lambert_2d(u_cam: jax.Array) -> jax.Array:
    """South-pole-guarded 2D Lambert pointing: 0 iff +Z_cam == +u_ray, ||r|| >= 1.999999 at [0, 0, -1]."""
    one_plus_uz = 1.0 + u_cam[2]
    u_xy_safe = jnp.where(
        one_plus_uz >= 1e-6,
        u_cam[:2],
        jnp.array([jnp.sqrt(1e-6 * (2.0 - 1e-6)), 0.0], dtype=u_cam.dtype),
    )
    return jnp.sqrt(2.0 / jnp.maximum(one_plus_uz, 1e-6)) * u_xy_safe


def world_spheres_to_base(
    centers_world: jax.Array | np.ndarray,
    radii: jax.Array | np.ndarray,
    active: jax.Array | np.ndarray,
) -> SphereObstacles:
    """Transform +X-forward world-frame obstacle sphere centers into Body5_1 URDF root coordinates."""
    centers_base = (
        jnp.asarray(centers_world, dtype=jnp.float64) - T_WORLD_BASE_JAX[:3, 3]
    ) @ R_BASE_WORLD_JAX.T
    return SphereObstacles(
        centers=centers_base,
        radii=jnp.asarray(radii, dtype=jnp.float64),
        active=jnp.asarray(active, dtype=bool),
    )


def build_corner_cone_ref(
    p_corner_world: np.ndarray,
    n_facets_world: np.ndarray,
    block_centers_world: np.ndarray,
    block_radii: np.ndarray | None = None,
    d_target: float = 0.125,
    q_posture: np.ndarray | None = None,
    cone_perturbation: np.ndarray | None = None,
) -> dict[str, jax.Array]:
    """Project corner->shoulder ray into 3-facet visibility cone (min(alpha) >= 0.28 => v.b_hat >= 0.80 > 0.65)."""
    r_xy = max(float(np.hypot(p_corner_world[0], p_corner_world[1])), 1e-6)
    u_rad = np.array([-p_corner_world[0], -p_corner_world[1], 0.65 * r_xy], dtype=np.float64)
    u_rad /= np.linalg.norm(u_rad)
    c_raw = np.asarray(n_facets_world, dtype=np.float64) @ u_rad
    c_clamped = np.array(
        [
            np.clip(c_raw[0], 0.35, 0.82),
            np.clip(c_raw[1], 0.35, 0.82),
            np.clip(c_raw[2], 0.55, 0.82),
        ],
        dtype=np.float64,
    )
    if cone_perturbation is not None:
        c_clamped = np.clip(c_clamped + np.asarray(cone_perturbation, dtype=np.float64), 0.30, 0.84)
    n_obs = len(block_centers_world)
    radii = (
        np.full(n_obs, 0.020, dtype=np.float64)
        if block_radii is None
        else np.asarray(block_radii, dtype=np.float64)
    )
    q_post = (
        np.array([0.0, -1.05, 0.95, 1.15, 0.0], dtype=np.float64)
        if q_posture is None
        else np.asarray(q_posture, dtype=np.float64)
    )
    return {
        "p_corner": jnp.asarray(p_corner_world, dtype=jnp.float64),
        "n_facets": jnp.asarray(n_facets_world, dtype=jnp.float64),
        "alpha_ref": jnp.asarray(c_clamped / np.linalg.norm(c_clamped), dtype=jnp.float64),
        "d_target": jnp.asarray(d_target, dtype=jnp.float64),
        "q_posture": jnp.asarray(q_post, dtype=jnp.float64),
        "block_spheres": world_spheres_to_base(
            block_centers_world,
            radii,
            np.ones(n_obs, dtype=bool),
        ),
    }


def make_5dof_cone_lookat_residual(
    robot: Manipulator,
    T_body3_cam: jax.Array,
    d_min: float = 0.088,
    d_max: float = 0.192,
):
    """5-row cone-admissible, antipodal-free Lambert Look-At residual matching SO-101 nv=5."""

    def _softplus(x: jax.Array, beta: float = 50.0) -> jax.Array:
        return jax.nn.softplus(beta * x) / beta

    def residual_fn(q: jax.Array, ref: dict[str, jax.Array]) -> jax.Array:
        T_world_cam = T_WORLD_BASE_JAX @ robot.ee_transform(q) @ T_body3_cam
        p_cam = T_world_cam[:3, 3]
        R_cam = T_world_cam[:3, :3]

        v = p_cam - ref["p_corner"]
        d = jnp.sqrt(jnp.dot(v, v) + 1e-12)
        v_hat = v / d
        u_ray = -v_hat

        u_cam = R_cam.T @ u_ray
        r_look = 4.0 * so101_antipodal_safe_lambert_2d(u_cam)

        r_dist = 2.0 * (d - ref["d_target"]) + 15.0 * (_softplus(d - d_max) - _softplus(d_min - d))

        alpha = ref["n_facets"] @ v_hat
        a_ref = ref["alpha_ref"]
        r_az = 0.35 * ((alpha[0] - alpha[1]) - (a_ref[0] - a_ref[1]))
        cone_pen = (
            20.0 * _softplus(0.30 - alpha[0])
            + 20.0 * _softplus(0.30 - alpha[1])
            + 20.0 * _softplus(0.30 - alpha[2])
        )
        r_el = 0.60 * (alpha[2] - a_ref[2]) + cone_pen
        return jnp.array([r_look[0], r_look[1], r_dist, r_az, r_el])

    return residual_fn


def make_5dof_grasp_residual(robot: Manipulator, tcp_offset_body3: jax.Array):
    """7-row exact 5-DOF-consistent grasp/placement residual for SO-101."""
    T_b3_tcp = jnp.eye(4, dtype=jnp.float64).at[:3, 3].set(tcp_offset_body3)

    def _softplus(x: jax.Array, beta: float = 25.0) -> jax.Array:
        return jax.nn.softplus(beta * x) / beta

    def residual_fn(q: jax.Array, ref: dict[str, jax.Array]) -> jax.Array:
        T_world_tcp = T_WORLD_BASE_JAX @ robot.ee_transform(q) @ T_b3_tcp
        p_tcp = T_world_tcp[:3, 3]
        R_tcp = T_world_tcp[:3, :3]

        # Position error in world meters (rows 0..2)
        r_pos = 15.0 * (p_tcp - ref["p_target"])
        neg_z = -R_tcp[:, 2]
        jaw_x = R_tcp[:, 0]

        # Top-down mode (-Z_Body3 == [0, 0, -1] + exact signed jaw_axis_world)
        R_tool = R_tcp @ jnp.diag(jnp.array([1.0, -1.0, -1.0], dtype=jnp.float64))
        r_td_pitch = 3.0 * so101_antipodal_safe_lambert_2d(R_tool.T @ ref["approach_dir"])
        jaw_perp = jnp.cross(ref["approach_dir"], ref["jaw_axis_world"])
        r_td_roll = 3.0 * jnp.array(
            [1.0 - jnp.dot(jaw_x, ref["jaw_axis_world"]), jnp.dot(jaw_x, jaw_perp)]
        )

        # Pitched radial mode (exact pitch neg_z[2] + exact horizontal jaw_x[2] == 0 + sign guards)
        horiz_ref = ref["approach_dir"][:2] / jnp.maximum(
            jnp.linalg.norm(ref["approach_dir"][:2]), 1e-6
        )
        r_p_pitch = jnp.array(
            [
                4.0 * (neg_z[2] - ref["approach_dir"][2]),
                3.0 * _softplus(0.45 - jnp.dot(neg_z[:2], horiz_ref)),
            ]
        )
        r_p_roll = jnp.array(
            [
                4.0 * jaw_x[2],
                3.0 * _softplus(0.65 - jnp.dot(jaw_x, ref["jaw_axis_world"])),
            ]
        )

        is_td = ref["is_topdown"] > 0.5
        r_pitch = jnp.where(is_td, r_td_pitch, r_p_pitch)
        r_roll = jnp.where(is_td, r_td_roll, r_p_roll)
        return jnp.concatenate([r_pos, r_pitch, r_roll])

    return residual_fn


@dataclass(frozen=True)
class CornerIKSolution:
    corner_index: int
    reachable: bool
    q_enc_rad: np.ndarray
    cost: float
    standoff_m: float
    pointing_cos: float
    min_facet_cos: float
    manipulability_5d: float


@dataclass(frozen=True)
class GraspIKSolution:
    feasible: bool
    q_grasp: np.ndarray
    p_tcp_world: np.ndarray
    jaw_yaw_world: float
    cost: float
    manipulability_5d: float


@dataclass(frozen=True)
class ClutterGraspChoice:
    candidate_index: int
    grasp_sol: GraspIKSolution
    open_gripper_pct: float
    neighbor_intrusion_pts: int
    rim_clearance_m: float


class SO101FraxIK:
    """JIT-compiled Frax 0.2.0 Look-At, Global Reachability, and C2/D4 Grasp IK engine."""

    def __init__(
        self,
        T_body3_cam_np: np.ndarray | None = None,
        tcp_offset_body3: np.ndarray | None = None,
        assets_dir: str | Path | None = None,
        num_global_samples: int = 48,
    ) -> None:
        self.cal_kin = CalibratedSO101Kinematics("follower")
        self.T_body3_cam_np = (
            nominal_canted_wrist_cam_transform(25.0)
            if T_body3_cam_np is None
            else np.asarray(T_body3_cam_np, dtype=np.float64)
        )
        self.urdf_path, self.spheres_path = export_canonical_frax_assets(
            assets_dir, T_body3_cam_np=self.T_body3_cam_np
        )
        self.tcp_offset_np = (
            NOMINAL_TCP_OFFSET_BODY3.copy()
            if tcp_offset_body3 is None
            else np.asarray(tcp_offset_body3, dtype=np.float64)
        )
        z_jaw_cam = float(
            (np.linalg.inv(self.T_body3_cam_np) @ np.array([-0.009, 0.0, -0.103, 1.0]))[2]
        )
        if z_jaw_cam > 0.11:
            self.default_d_target = z_jaw_cam + 0.055
            self.d_min_m = z_jaw_cam + 0.015
            self.d_max_m = z_jaw_cam + 0.145
        else:
            self.default_d_target = 0.125
            self.d_min_m = 0.085
            self.d_max_m = 0.195

        collision_data = bubblify_to_mine(
            self.spheres_path,
            child_map=SO101_CHILD_MAP,
            root_link_name="Body5_1",
            add_floating_base=False,
            sc_data=SO101_SC_DATA,
            verbose=False,
        )
        self.robot = Manipulator(
            str(self.urdf_path),
            collision_data=collision_data,
            joint_ordering=SO101_ARM_JOINTS,
        )
        self.T_body3_cam = jnp.asarray(self.T_body3_cam_np, dtype=jnp.float64)
        self.q_lo = self.robot.actuated_joint_lower_limits + 0.02
        self.q_hi = self.robot.actuated_joint_upper_limits - 0.02
        # Safe elbow-up overhead inspection joint envelope (prevents inverted-elbow / backward-wrist table sweeps)
        self.q_look_lo = jnp.maximum(
            self.q_lo, jnp.array([-1.65, -1.75, 0.12, -1.62, -3.10], dtype=jnp.float64)
        )
        self.q_look_hi = jnp.minimum(
            self.q_hi, jnp.array([1.65, -0.10, 1.65, 1.62, 3.10], dtype=jnp.float64)
        )
        # Exclude base Body5_1 (idx 0) and turret Body1_2 (idx 1) from tabletop floor margin
        moving_sphere_idxs = jnp.arange(2, len(self.robot.flat_collision_radii))
        base_z = float(T_WORLD_BASE[2, 3])

        init_obstacles = SphereObstacles(
            centers=jnp.zeros((4, 3), dtype=jnp.float64),
            radii=jnp.full(4, 0.020, dtype=jnp.float64),
            active=jnp.ones(4, dtype=bool),
        )
        look_constraints = [
            JointLimits(
                self.robot,
                q_min=self.q_look_lo,
                q_max=self.q_look_hi,
                v_max=jnp.full(5, 3.00, dtype=jnp.float64),
                v_min=jnp.full(5, -3.00, dtype=jnp.float64),
            ),
            FloorConstraint(self.robot, sphere_idxs=moving_sphere_idxs, margin=0.020 - base_z),
            SelfCollisionConstraint(self.robot, margin=0.010),
            ObstacleConstraint(
                self.robot, obstacles=init_obstacles, ref_key="block_spheres", margin=0.014
            ),
        ]
        look_tasks = [
            ResidualTask(
                self.robot,
                make_5dof_cone_lookat_residual(
                    self.robot,
                    self.T_body3_cam,
                    d_min=self.d_min_m + 0.003,
                    d_max=self.d_max_m - 0.003,
                ),
                num_rows=5,
                gain=20.0,
                weight=1.0,
            ),
            JointTask(self.robot, idxs=jnp.arange(5), ref_key="q_posture", gain=5.0, weight=1e-2),
        ]
        self.diff_ik = DiffIKSolver(
            self.robot, tasks=look_tasks, constraints=look_constraints, damping=1e-2
        )
        self.global_ik = GlobalIKSolver(
            self.diff_ik,
            num_samples=num_global_samples,
            q_min=self.q_look_lo,
            q_max=self.q_look_hi,
            max_iters=100,
            tol=1e-4,
            cost_tol=None,
        )
        self._jit_diff_step = jax.jit(self.diff_ik.step)
        self._jit_global_solve = jax.jit(self.global_ik.solve)
        self._jit_manip5 = jax.jit(lambda q: so101_manipulability_5d(self.robot, q))
        self._jit_look_violation = jax.jit(
            lambda q, ref: jnp.max(jnp.maximum(-self.diff_ik._qp(q, ref)[3], 0.0))
        )

        # Grasp solver: keep arm/wrist links above table while allowing jaw-tip sphere (last idx) to descend alongside block
        grasp_floor_idxs = jnp.arange(2, len(self.robot.flat_collision_radii) - 1)
        grasp_constraints = [
            JointLimits(
                self.robot,
                q_min=self.q_lo,
                q_max=self.q_hi,
                v_max=jnp.full(5, 3.00, dtype=jnp.float64),
                v_min=jnp.full(5, -3.00, dtype=jnp.float64),
            ),
            FloorConstraint(self.robot, sphere_idxs=grasp_floor_idxs, margin=0.012 - base_z),
            SelfCollisionConstraint(self.robot, margin=0.008),
        ]
        grasp_tasks = [
            ResidualTask(
                self.robot,
                make_5dof_grasp_residual(
                    self.robot, jnp.asarray(self.tcp_offset_np, dtype=jnp.float64)
                ),
                num_rows=7,
                gain=20.0,
                weight=1.0,
            ),
            JointTask(self.robot, idxs=jnp.arange(5), ref_key="q_posture", gain=4.0, weight=2e-3),
        ]
        self.grasp_diff_ik = DiffIKSolver(
            self.robot, tasks=grasp_tasks, constraints=grasp_constraints, damping=1e-2
        )
        self.grasp_global_ik = GlobalIKSolver(
            self.grasp_diff_ik,
            num_samples=num_global_samples,
            q_min=self.q_lo,
            q_max=self.q_hi,
            max_iters=100,
            tol=1e-4,
            cost_tol=None,
        )
        self._jit_grasp_solve = jax.jit(self.grasp_global_ik.solve)

        def _run_grasp_diff(q_init: jax.Array, ref: dict[str, jax.Array]) -> jax.Array:
            st0 = self.grasp_diff_ik.init(q_init)
            st_final = jax.lax.fori_loop(
                0,
                14,
                lambda _i, st: self.grasp_diff_ik.step(st, ref, 1.0 / 25.0)[0],
                st0,
            )
            return st_final.q

        self._jit_grasp_diff_solve = jax.jit(_run_grasp_diff)
        self._rng_key = jax.random.PRNGKey(42)

    def next_key(self) -> jax.Array:
        self._rng_key, sub = jax.random.split(self._rng_key)
        return sub

    def fk_world_body3(self, q_enc_rad: np.ndarray) -> np.ndarray:
        """Exact calibrated Body3 SE(3) pose in +X-forward world frame."""
        return T_WORLD_BASE @ self.cal_kin.fk(q_enc_rad)

    def fk_world_tcp(self, q_enc_rad: np.ndarray) -> np.ndarray:
        """Exact calibrated jaw-center TCP SE(3) pose in +X-forward world frame."""
        return T_WORLD_BASE @ self.cal_kin.fk(q_enc_rad, tcp_offset=self.tcp_offset_np)

    def fk_world_wrist_cam(self, q_enc_rad: np.ndarray) -> np.ndarray:
        """Exact calibrated wrist camera SE(3) pose in +X-forward world frame (+Z optical forward)."""
        return self.fk_world_body3(q_enc_rad) @ self.T_body3_cam_np

    def step_lookat_diff_ik(
        self, q_current: np.ndarray, ref: dict[str, jax.Array], dt: float = 1.0 / 30.0
    ):
        state = self.diff_ik.init(jnp.asarray(q_current, dtype=jnp.float64))
        next_state, v, info = self._jit_diff_step(state, ref, dt)
        return np.asarray(next_state.q, dtype=np.float64), np.asarray(v, dtype=np.float64), info

    def solve_corner_lookat(
        self,
        corner_index: int,
        T_world_block: np.ndarray,
        block_spec: BlockSpec,
        d_target: float | None = None,
        q_seed: np.ndarray | None = None,
        cone_perturbation: np.ndarray | None = None,
    ) -> tuple[CornerIKSolution, dict[str, jax.Array]]:
        """Solve collision-free 5-DOF cone Look-At posture for corner c_i using GlobalIKSolver."""
        d_tgt = self.default_d_target if d_target is None else float(d_target)
        corners_w = corner_positions_world(T_world_block, block_spec.dims_m)
        facets_w = corner_facet_normals_world(T_world_block)
        obs_local, obs_radii = block_spec.obstacle_spheres_local()
        obs_world = (obs_local @ T_world_block[:3, :3].T) + T_world_block[:3, 3]

        ref = build_corner_cone_ref(
            p_corner_world=corners_w[corner_index],
            n_facets_world=facets_w[corner_index],
            block_centers_world=obs_world,
            block_radii=obs_radii,
            d_target=d_tgt,
            q_posture=q_seed,
            cone_perturbation=cone_perturbation,
        )
        res = self._jit_global_solve(ref["q_posture"], ref, self.next_key())
        q_sol = np.asarray(res.q, dtype=np.float64)
        cost = float(res.cost)

        T_w_cam = self.fk_world_wrist_cam(q_sol)
        v = T_w_cam[:3, 3] - corners_w[corner_index]
        standoff = float(np.linalg.norm(v))
        v_hat = v / max(standoff, 1e-9)
        u_ray = -v_hat
        pointing_cos = float(np.dot(T_w_cam[:3, 2], u_ray))
        min_facet_cos = float(np.min(facets_w[corner_index] @ v_hat))
        manip5 = float(self._jit_manip5(jnp.asarray(q_sol)))

        viol = float(self._jit_look_violation(jnp.asarray(q_sol), ref))
        bisector_w = np.sum(facets_w[corner_index], axis=0) / math.sqrt(3.0)
        bisector_cos = float(np.dot(v_hat, bisector_w))
        reachable = (
            viol <= 2e-3
            and pointing_cos >= 0.985
            and self.d_min_m <= standoff <= self.d_max_m
            and min_facet_cos >= 0.25
            and bisector_cos >= 0.70
        )
        return (
            CornerIKSolution(
                corner_index=corner_index,
                reachable=reachable,
                q_enc_rad=q_sol,
                cost=cost,
                standoff_m=standoff,
                pointing_cos=pointing_cos,
                min_facet_cos=min_facet_cos,
                manipulability_5d=manip5,
            ),
            ref,
        )

    def evaluate_upper_corners_reachability(
        self, T_world_block: np.ndarray, block_spec: BlockSpec
    ) -> dict[int, CornerIKSolution]:
        """Batch-evaluate CornerWristReachable(b, c_i) on all upper-hemisphere corners."""
        upper = corner_upper_hemisphere_mask(T_world_block)
        out: dict[int, CornerIKSolution] = {}
        for i in range(8):
            if upper[i]:
                sol, _ = self.solve_corner_lookat(i, T_world_block, block_spec)
                out[i] = sol
        return out

    def solve_grasp_pose(
        self,
        p_target_world: np.ndarray,
        jaw_yaw_world: float,
        pitch_down_deg: float = 90.0,
        q_posture: np.ndarray | None = None,
        local_only: bool = False,
    ) -> GraspIKSolution:
        """Solve 5-DOF collision-free grasp/placement posture.

        - pitch_down_deg == 90.0: vertical top-down grasp (-Z_Body3 = [0, 0, -1], arbitrary jaw_yaw_world).
        - pitch_down_deg < 90.0 (e.g. 45.0): radial pitched grasp in the arm's vertical plane
          so wrist_roll (Revolute_4) rotates around a 45-deg inclined axis for table-safe 90/180-deg flips.
        """
        p_tgt = np.asarray(p_target_world, dtype=np.float64)
        radial_az = math.atan2(p_tgt[1], p_tgt[0])
        pitch_rad = math.radians(pitch_down_deg)
        approach = np.array(
            [
                math.cos(pitch_rad) * math.cos(radial_az),
                math.cos(pitch_rad) * math.sin(radial_az),
                -math.sin(pitch_rad),
            ],
            dtype=np.float64,
        )
        approach /= np.linalg.norm(approach)

        is_topdown = abs(pitch_down_deg - 90.0) < 1.0
        if is_topdown:
            jaw_axis = np.array(
                [math.cos(jaw_yaw_world), math.sin(jaw_yaw_world), 0.0], dtype=np.float64
            )
        else:
            # Horizontal axis perpendicular to radial plane (or radial in-plane if yaw aligns)
            perp_horiz = np.array(
                [-math.sin(radial_az), math.cos(radial_az), 0.0], dtype=np.float64
            )
            in_plane = np.cross(perp_horiz, approach)
            in_plane /= np.linalg.norm(in_plane)
            rel_yaw = jaw_yaw_world - (radial_az + 0.5 * math.pi)
            jaw_axis = math.cos(rel_yaw) * perp_horiz + math.sin(rel_yaw) * in_plane
            jaw_axis /= np.linalg.norm(jaw_axis)

        q_post = (
            np.array([radial_az, -0.85, 1.45, 0.20, 0.0], dtype=np.float64)
            if q_posture is None
            else np.asarray(q_posture, dtype=np.float64)
        )
        ref = {
            "p_target": jnp.asarray(p_tgt, dtype=jnp.float64),
            "approach_dir": jnp.asarray(approach, dtype=jnp.float64),
            "jaw_axis_world": jnp.asarray(jaw_axis, dtype=jnp.float64),
            "is_topdown": jnp.asarray(1.0 if is_topdown else 0.0, dtype=jnp.float64),
            "q_posture": jnp.asarray(q_post, dtype=jnp.float64),
        }
        if local_only and q_posture is not None:
            q_sol = np.asarray(self._jit_grasp_diff_solve(ref["q_posture"], ref), dtype=np.float64)
            cost = float(np.linalg.norm(q_sol - q_post))
            T_tcp = self.fk_world_tcp(q_sol)
            pos_err = float(np.linalg.norm(T_tcp[:3, 3] - p_tgt))
            if is_topdown:
                orient_ok = (
                    float(np.dot(-T_tcp[:3, 2], approach)) >= 0.96
                    and float(np.dot(T_tcp[:3, 0], jaw_axis)) >= 0.92
                )
            else:
                orient_ok = (
                    abs(float(-T_tcp[2, 2] - approach[2])) <= 0.08
                    and abs(float(T_tcp[2, 0])) <= 0.08
                    and float(np.dot(T_tcp[:3, 0], jaw_axis)) >= 0.85
                )
            if not (pos_err <= 0.010 and orient_ok):
                res = self._jit_grasp_solve(ref["q_posture"], ref, self.next_key())
                q_sol = np.asarray(res.q, dtype=np.float64)
                cost = float(res.cost)
                T_tcp = self.fk_world_tcp(q_sol)
                pos_err = float(np.linalg.norm(T_tcp[:3, 3] - p_tgt))
        else:
            res = self._jit_grasp_solve(ref["q_posture"], ref, self.next_key())
            q_sol = np.asarray(res.q, dtype=np.float64)
            cost = float(res.cost)
            T_tcp = self.fk_world_tcp(q_sol)
            pos_err = float(np.linalg.norm(T_tcp[:3, 3] - p_tgt))
        if is_topdown:
            orient_ok = (
                float(np.dot(-T_tcp[:3, 2], approach)) >= 0.96
                and float(np.dot(T_tcp[:3, 0], jaw_axis)) >= 0.92
            )
            achieved_yaw = jaw_yaw_world
        else:
            orient_ok = (
                abs(float(-T_tcp[2, 2] - approach[2])) <= 0.08
                and abs(float(T_tcp[2, 0])) <= 0.08
                and float(np.dot(T_tcp[:3, 0], jaw_axis)) >= 0.85
            )
            achieved_yaw = math.atan2(float(T_tcp[1, 0]), float(T_tcp[0, 0]))
        manip5 = float(self._jit_manip5(jnp.asarray(q_sol)))
        feasible = pos_err <= 0.010 and orient_ok
        return GraspIKSolution(
            feasible=feasible,
            q_grasp=q_sol,
            p_tcp_world=T_tcp[:3, 3].copy(),
            jaw_yaw_world=achieved_yaw,
            cost=cost,
            manipulability_5d=manip5,
        )

    def select_best_symmetric_topdown_grasp(
        self, T_world_block: np.ndarray, block_spec: BlockSpec, z_offset_m: float = 0.0
    ) -> GraspIKSolution:
        """Score all C2/D4 graspable width axes via GlobalIKSolver + 5D Yoshikawa manipulability."""
        p_block = T_world_block[:3, 3]
        R_wb = T_world_block[:3, :3]
        # Strongly prefer flat 32 mm / 16 mm side walls (<= 0.036 m) over curved 48 mm arches
        candidates: list[tuple[float, float]] = []
        for max_w in (0.036, 0.052):
            for local_idx, span in enumerate(block_spec.dims_m):
                axis_w = R_wb[:, local_idx]
                if abs(axis_w[2]) < 0.25 and span <= max_w:
                    yaw_base = math.atan2(axis_w[1], axis_w[0])
                    candidates.extend([(yaw_base, span), (yaw_base + math.pi, span)])
            if candidates:
                break
        if not candidates:
            yaw_base = math.atan2(R_wb[1, 1], R_wb[0, 1])
            candidates = [(yaw_base, 0.032), (yaw_base + math.pi, 0.032)]

        p_base = np.array([p_block[0], p_block[1], p_block[2] + z_offset_m], dtype=np.float64)
        best: GraspIKSolution | None = None
        best_score = -1e9
        for yaw, span in candidates:
            # Shift TCP along -jaw_axis for narrow 16 mm bricks so fixed jaw sits 6 mm from side wall
            shift_m = -0.007 if span <= 0.020 else 0.0
            jaw_dir = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
            p_target = p_base + shift_m * jaw_dir
            sol = self.solve_grasp_pose(p_target, jaw_yaw_world=yaw, pitch_down_deg=90.0)
            score = (10.0 if sol.feasible else 0.0) - 5.0 * sol.cost + 2.0 * sol.manipulability_5d
            if score > best_score:
                best_score = score
                best = sol
        assert best is not None
        return best

    def select_clutter_safe_topdown_grasp(
        self,
        candidates: list,
        all_fg_pts_world: np.ndarray,
        source_plate_xy: np.ndarray | None = None,
    ) -> ClutterGraspChoice:
        """Select the brick + jaw orientation in a clustered plate with maximum 3D jaw clearance."""
        plate_xy = (
            np.array([0.23, 0.00], dtype=np.float64)
            if source_plate_xy is None
            else np.asarray(source_plate_xy, dtype=np.float64)
        )
        best_choice: ClutterGraspChoice | None = None
        best_score = -1e12

        for idx, cand in enumerate(candidates):
            spec: BlockSpec = cand.block_spec
            T_wb = cand.localized.T_world_block
            p_b = T_wb[:3, 3]
            R_wb = T_wb[:3, :3]
            half_own = 0.5 * np.asarray(spec.dims_m, dtype=np.float64) + 0.0045

            # Filter neighbor points strictly outside this candidate brick's own expanded cuboid
            if len(all_fg_pts_world) > 0:
                rel_own = np.abs((all_fg_pts_world - p_b) @ R_wb)
                other_mask = np.any(rel_own > half_own[None, :], axis=1)
                neighbor_pts = all_fg_pts_world[other_mask]
            else:
                neighbor_pts = np.zeros((0, 3), dtype=np.float64)

            # Grasp upper wall of tall 1x2x2 bricks so jaws stay above 23.6 mm neighbors
            hz = float(spec.dims_m[2])
            z_grasp = float(p_b[2]) + (0.011 if hz > 0.035 else 0.0042)
            z_tip = z_grasp - 0.0155

            yaw_spans: list[tuple[float, float]] = []
            for local_idx, span in enumerate(spec.dims_m):
                axis_w = R_wb[:, local_idx]
                if abs(axis_w[2]) < 0.25 and span <= 0.036:
                    y0 = math.atan2(axis_w[1], axis_w[0])
                    yaw_spans.extend([(y0, span), (y0 + math.pi, span)])
            if not yaw_spans:
                y0 = math.atan2(R_wb[1, 1], R_wb[0, 1])
                yaw_spans = [(y0, 0.032), (y0 + math.pi, 0.032)]

            along_offsets = (0.0, -0.015, +0.015) if max(spec.dims_m[:2]) >= 0.058 else (0.0,)
            for yaw, span in yaw_spans:
                jaw_dir = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
                perp_dir = np.array([-jaw_dir[1], jaw_dir[0]], dtype=np.float64)
                is_narrow = span <= 0.020
                open_pct = 0.48 if is_narrow else 0.68
                shift_m = -0.007 if is_narrow else 0.0
                fix_span = (-0.031, -0.021)
                mov_span = (+0.012, +0.024) if is_narrow else (+0.021, +0.032)

                for s_along in along_offsets:
                    p_tcp = np.array(
                        [
                            p_b[0] + shift_m * jaw_dir[0] + s_along * perp_dir[0],
                            p_b[1] + shift_m * jaw_dir[1] + s_along * perp_dir[1],
                            z_grasp,
                        ],
                        dtype=np.float64,
                    )

                    # Check plate sloped rim clearance for both jaw outer tips
                    p_fix_tip = p_tcp[:2] + fix_span[0] * jaw_dir
                    p_mov_tip = p_tcp[:2] + mov_span[1] * jaw_dir
                    r_max_jaw = max(
                        float(np.linalg.norm(p_fix_tip - plate_xy)),
                        float(np.linalg.norm(p_mov_tip - plate_xy)),
                    )
                    rim_clearance = 0.079 - r_max_jaw

                    # Count neighbor brick points colliding with either descending jaw pad
                    intrusions = 0
                    if len(neighbor_pts) > 0:
                        above_tip = neighbor_pts[:, 2] >= z_tip
                        if np.any(above_tip):
                            d_xy = neighbor_pts[above_tip, :2] - p_tcp[None, :2]
                            u_jaw = d_xy @ jaw_dir
                            v_perp = np.abs(d_xy @ perp_dir)
                            in_fix = (
                                (u_jaw >= fix_span[0]) & (u_jaw <= fix_span[1]) & (v_perp <= 0.011)
                            )
                            in_mov = (
                                (u_jaw >= mov_span[0]) & (u_jaw <= mov_span[1]) & (v_perp <= 0.011)
                            )
                            intrusions = int(np.sum(in_fix | in_mov))

                    sol = self.solve_grasp_pose(p_tcp, jaw_yaw_world=yaw, pitch_down_deg=90.0)
                    # Strictly prefer 0-intrusion outer bricks first (peels cluster outside-in!)
                    r_from_cluster = float(np.linalg.norm(p_b[:2] - plate_xy))
                    score = (
                        (25.0 if sol.feasible else -50.0)
                        - 6.0 * float(intrusions)
                        - 18.0 * abs(s_along)
                        + (25.0 * rim_clearance if rim_clearance < 0.0 else 2.0 * rim_clearance)
                        + 18.0 * hz
                        + 5.0 * r_from_cluster
                        - 5.0 * sol.cost
                        + 2.0 * sol.manipulability_5d
                    )
                    if score > best_score:
                        best_score = score
                        best_choice = ClutterGraspChoice(
                            candidate_index=idx,
                            grasp_sol=sol,
                            open_gripper_pct=open_pct,
                            neighbor_intrusion_pts=intrusions,
                            rim_clearance_m=rim_clearance,
                        )
        assert best_choice is not None
        return best_choice
