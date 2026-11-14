"""Material-anchored 8-corner coverage ledger preventing D4/C4 symmetry gauge swaps."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from brickcell.block_inspector.geometry import (
    BlockSpec,
    CameraExtrinsicsRig,
    corner_bisectors_local,
    corner_facet_normals_local,
    corner_positions_world,
    corner_upper_hemisphere_mask,
    so3_geodesic_angle_rad,
)


@dataclass(frozen=True)
class LedgerConfig:
    """Geometric and optical verification thresholds from ARCHITECTURE.md §4.3."""

    min_table_clearance_m: float = 0.004
    min_jaw_clearance_m: float = 0.015
    max_ray_occlusion_error_m: float = 0.004
    min_bisector_cos: float = 0.65
    min_facet_cos: float = 0.20
    wrist_standoff_min_m: float = 0.08
    wrist_standoff_max_m: float = 0.20
    d415_corroboration_max_m: float = 0.35
    allow_dual_d415_corroboration: bool = False
    frame_margin_px: int = 16
    max_joint_speed_rad_s: float = 0.05
    min_laplacian_var: float = 25.0
    roi_half_size_px: int = 32
    required_dwell_frames: int = 3
    max_prior_rotation_error_deg: float = 35.0
    min_icp_fitness: float = 0.65
    min_patch_ncc: float = 0.60


@dataclass(frozen=True)
class CornerObservation:
    """Single-frame optical + geometric verification evidence for one material corner."""

    corner_index: int
    camera_name: str
    valid: bool
    reason: str
    standoff_m: float
    bisector_cos: float
    min_facet_cos: float
    pixel_uv: tuple[float, float]
    laplacian_var: float
    ray_occlusion_error_m: float
    raw_anomaly_distance: float | None = None
    roi_rgb: np.ndarray | None = None


@dataclass
class CornerRecord:
    """Persistent state for one material corner c_i (anchored to the physical block)."""

    corner_index: int
    inspected: bool = False
    dwell_count: int = 0
    best_sharpness: float = 0.0
    best_bisector_cos: float = 0.0
    best_min_facet_cos: float = 0.0
    best_standoff_m: float = 0.0
    best_camera: str = ""
    best_pixel_uv: tuple[float, float] = (0.0, 0.0)
    raw_anomaly_distance: float | None = None
    score_type: str = "raw_patchcore_distance_not_probability"
    best_roi_rgb: np.ndarray | None = field(default=None, repr=False)
    last_reason: str = "unvisited"


def compute_roi_laplacian_var(
    rgb: np.ndarray, u: float, v: float, half_size: int = 32
) -> tuple[float, np.ndarray]:
    """Extract clamped 64x64 grayscale ROI around (u, v) and return Laplacian variance + RGB crop."""
    h, w = rgb.shape[:2]
    cx = int(round(u))
    cy = int(round(v))
    x0 = max(0, min(w - 2 * half_size, cx - half_size))
    y0 = max(0, min(h - 2 * half_size, cy - half_size))
    x1 = min(w, x0 + 2 * half_size)
    y1 = min(h, y0 + 2 * half_size)
    crop = rgb[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return 0.0, crop
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return lap_var, crop


class EightCornerCoverageLedger:
    """Tracks all 8 material corners c_0..c_7 with left-SE(3) pose propagation across manipulations."""

    def __init__(
        self,
        object_id: str,
        block_spec: BlockSpec,
        rig: CameraExtrinsicsRig,
        config: LedgerConfig | None = None,
    ) -> None:
        if not object_id:
            raise ValueError("object_id must be nonempty")
        self.object_id = object_id
        self.block_spec = block_spec
        self.rig = rig
        if config is None:
            cfg = LedgerConfig()
            z_jaw_cam = float(
                (np.linalg.inv(rig.wrist.T_parent_cvcam) @ np.array([-0.009, 0.0, -0.103, 1.0]))[2]
            )
            if z_jaw_cam > 0.11:
                cfg = replace(
                    cfg, wrist_standoff_max_m=max(cfg.wrist_standoff_max_m, z_jaw_cam + 0.16)
                )
            if float(rig.wrist.K[0, 0]) < 350.0:
                cfg = replace(cfg, min_laplacian_var=12.0)
            self.config = cfg
        else:
            self.config = config

        self.localized: bool = False
        self.T_world_block: np.ndarray = np.eye(4, dtype=np.float64)
        self.T_prior_next: np.ndarray | None = None
        self.slip_ambiguity_detected: bool = False
        self.records: list[CornerRecord] = [CornerRecord(i) for i in range(8)]
        self._bisectors_local = corner_bisectors_local()
        self._facets_local = corner_facet_normals_local()

    def set_initial_pose(self, T_world_block: np.ndarray) -> None:
        """Anchor corner IDs c_0..c_7 to the initially localized block pose."""
        self.T_world_block = np.array(T_world_block, dtype=np.float64, copy=True)
        self.localized = True
        self.slip_ambiguity_detected = False

    def reset_all(self) -> None:
        """Reset localization and all 8 corner records for a fresh episode."""
        self.localized = False
        self.T_world_block = np.eye(4, dtype=np.float64)
        self.T_prior_next = None
        self.slip_ambiguity_detected = False
        self.records = [CornerRecord(i) for i in range(8)]

    def propagate_manipulation_prior(
        self, T_world_body3_grasp: np.ndarray, T_world_body3_place: np.ndarray
    ) -> np.ndarray:
        """Compute left-multiplied SE(3) kinematic reorientation prior T_prior^(t+).

        T_prior = T_world_Body3(q_place) @ inv(T_world_Body3(q_grasp)) @ T_world_B^(t-)
        """
        if not self.localized:
            raise RuntimeError("Cannot propagate manipulation prior before initial localization")
        T_grasp = np.asarray(T_world_body3_grasp, dtype=np.float64)
        T_place = np.asarray(T_world_body3_place, dtype=np.float64)
        delta_world = T_place @ np.linalg.inv(T_grasp)
        self.T_prior_next = delta_world @ self.T_world_block
        return self.T_prior_next.copy()

    def commit_post_manipulation_pose(
        self,
        T_world_block_refined: np.ndarray,
        icp_fitness: float = 1.0,
        patch_ncc: float = 1.0,
    ) -> bool:
        """Lock post-manipulation pose onto the material branch without D4/C4 corner-ID swaps.

        Returns True if locked cleanly; returns False (SlipAmbiguity) and conservatively
        marks newly exposed upper-hemisphere corners uninspected if slip > 35 deg or fitness fails.
        """
        if self.T_prior_next is None:
            raise RuntimeError("propagate_manipulation_prior must be called before committing pose")
        T_ref = np.array(T_world_block_refined, dtype=np.float64, copy=True)
        rot_err_rad = so3_geodesic_angle_rad(self.T_prior_next[:3, :3], T_ref[:3, :3])
        max_rot_rad = math.radians(self.config.max_prior_rotation_error_deg)

        clean_lock = (
            rot_err_rad < max_rot_rad
            and icp_fitness >= self.config.min_icp_fitness
            and patch_ncc >= self.config.min_patch_ncc
        )
        self.T_world_block = T_ref
        self.T_prior_next = None

        if not clean_lock:
            self.slip_ambiguity_detected = True
            upper = self.upper_hemisphere_mask()
            for idx in range(8):
                if upper[idx]:
                    self.records[idx].inspected = False
                    self.records[idx].dwell_count = 0
                    self.records[idx].last_reason = "reset_due_to_slip_ambiguity"
            return False

        self.slip_ambiguity_detected = False
        return True

    def corners_world(self) -> np.ndarray:
        return corner_positions_world(self.T_world_block, self.block_spec.dims_m)

    def upper_hemisphere_mask(self) -> np.ndarray:
        return corner_upper_hemisphere_mask(self.T_world_block)

    def inspected_mask(self) -> np.ndarray:
        return np.array([r.inspected for r in self.records], dtype=bool)

    def all_inspected(self) -> bool:
        return all(r.inspected for r in self.records)

    def evaluate_camera_frame(
        self,
        camera_name: str,
        rgb: np.ndarray,
        T_world_body3: np.ndarray,
        qvel_arm: np.ndarray,
        ray_errors_m: np.ndarray,
        block_held: bool = False,
        jaw_positions_world: tuple[np.ndarray, np.ndarray] | None = None,
        anomaly_scorer=None,
    ) -> list[CornerObservation]:
        """Evaluate one camera frame against all 8 corners (single-camera check)."""
        if not self.localized:
            raise RuntimeError("Block must be localized before evaluating corner coverage")
        cam = self.rig.by_name(camera_name)
        T_world_cam = cam.world_transform(T_world_body3)
        p_cam = T_world_cam[:3, 3]

        corners_w = self.corners_world()
        R_wb = self.T_world_block[:3, :3]
        bisectors_w = self._bisectors_local @ R_wb.T
        facets_w = np.einsum("ij,kfj->kfi", R_wb, self._facets_local)
        uv_all, z_cam_all = cam.project_world_points(corners_w, T_world_body3)
        joint_speed = float(np.max(np.abs(np.asarray(qvel_arm, dtype=np.float64))))

        observations: list[CornerObservation] = []
        for i in range(8):
            p_c = corners_w[i]
            v_vec = p_cam - p_c
            standoff = float(np.linalg.norm(v_vec))
            v_hat = v_vec / max(standoff, 1e-9)
            b_cos = float(np.dot(v_hat, bisectors_w[i]))
            f_cos = float(np.min(facets_w[i] @ v_hat))
            u, v = float(uv_all[i, 0]), float(uv_all[i, 1])
            ray_err = float(ray_errors_m[i])

            reason = "ok"
            valid = True
            if p_c[2] < self.config.min_table_clearance_m:
                valid, reason = False, "table_occluded"
            elif (
                block_held
                and jaw_positions_world is not None
                and any(
                    float(np.linalg.norm(p_c - jp)) <= self.config.min_jaw_clearance_m
                    for jp in jaw_positions_world
                )
            ):
                valid, reason = False, "jaw_occluded"
            elif z_cam_all[i] <= 0.02:
                valid, reason = False, "behind_camera"
            elif not (
                self.config.frame_margin_px <= u < cam.width - self.config.frame_margin_px
                and self.config.frame_margin_px <= v < cam.height - self.config.frame_margin_px
            ):
                valid, reason = False, "outside_frame_margin"
            elif cam.is_wrist and not (
                self.config.wrist_standoff_min_m <= standoff <= self.config.wrist_standoff_max_m
            ):
                valid, reason = False, f"wrist_standoff_{standoff:.3f}m_out_of_bounds"
            elif (not cam.is_wrist) and standoff > self.config.d415_corroboration_max_m:
                valid, reason = False, f"d415_standoff_{standoff:.3f}m_too_far"
            elif b_cos < self.config.min_bisector_cos or f_cos < self.config.min_facet_cos:
                valid, reason = False, f"outside_3facet_cone(b={b_cos:.2f},f={f_cos:.2f})"
            elif ray_err > self.config.max_ray_occlusion_error_m:
                valid, reason = False, f"ray_occluded(err={ray_err:.4f}m)"
            elif joint_speed >= self.config.max_joint_speed_rad_s:
                valid, reason = False, f"arm_moving({joint_speed:.3f}rad/s)"

            lap_var = 0.0
            crop = None
            raw_dist = None
            if valid:
                lap_var, crop = compute_roi_laplacian_var(
                    rgb, u, v, half_size=self.config.roi_half_size_px
                )
                if lap_var < self.config.min_laplacian_var:
                    valid, reason = False, f"blurry_roi(lap={lap_var:.1f})"
                elif anomaly_scorer is not None and crop is not None:
                    raw_dist = float(anomaly_scorer(crop))

            observations.append(
                CornerObservation(
                    corner_index=i,
                    camera_name=cam.name,
                    valid=valid,
                    reason=reason,
                    standoff_m=standoff,
                    bisector_cos=b_cos,
                    min_facet_cos=f_cos,
                    pixel_uv=(u, v),
                    laplacian_var=lap_var,
                    ray_occlusion_error_m=ray_err,
                    raw_anomaly_distance=raw_dist,
                    roi_rgb=crop,
                )
            )
        return observations

    def update_step(
        self,
        wrist_obs: list[CornerObservation] | None = None,
        rs1_obs: list[CornerObservation] | None = None,
        rs2_obs: list[CornerObservation] | None = None,
    ) -> np.ndarray:
        """Advance consecutive dwell counters across synchronized multi-camera observations."""
        for i in range(8):
            rec = self.records[i]
            accepted: CornerObservation | None = None
            if wrist_obs is not None and wrist_obs[i].valid:
                accepted = wrist_obs[i]
            elif (
                self.config.allow_dual_d415_corroboration
                and rs1_obs is not None
                and rs2_obs is not None
                and rs1_obs[i].valid
                and rs2_obs[i].valid
            ):
                accepted = (
                    rs1_obs[i]
                    if rs1_obs[i].laplacian_var >= rs2_obs[i].laplacian_var
                    else rs2_obs[i]
                )

            if accepted is not None:
                rec.dwell_count += 1
                rec.last_reason = "accepted_frame"
                if accepted.laplacian_var >= rec.best_sharpness:
                    rec.best_sharpness = accepted.laplacian_var
                    rec.best_bisector_cos = accepted.bisector_cos
                    rec.best_min_facet_cos = accepted.min_facet_cos
                    rec.best_standoff_m = accepted.standoff_m
                    rec.best_camera = accepted.camera_name
                    rec.best_pixel_uv = accepted.pixel_uv
                    rec.best_roi_rgb = accepted.roi_rgb
                    if accepted.raw_anomaly_distance is not None:
                        rec.raw_anomaly_distance = accepted.raw_anomaly_distance
                if rec.dwell_count >= self.config.required_dwell_frames:
                    rec.inspected = True
            else:
                if not rec.inspected:
                    rec.dwell_count = 0
                if wrist_obs is not None:
                    rec.last_reason = wrist_obs[i].reason
        return self.inspected_mask()

    def summary_dict(self) -> dict:
        corners_w = self.corners_world()
        upper = self.upper_hemisphere_mask()
        return {
            "object_id": self.object_id,
            "part_id": self.block_spec.part_id,
            "block_name": self.block_spec.name,
            "symmetry_group": self.block_spec.symmetry_group,
            "localized": self.localized,
            "all_8_inspected": self.all_inspected(),
            "inspected_count": int(sum(r.inspected for r in self.records)),
            "slip_ambiguity_detected": self.slip_ambiguity_detected,
            "T_world_block": self.T_world_block.tolist(),
            "dims_m": list(self.block_spec.dims_m),
            "corners": [
                {
                    "corner_index": r.corner_index,
                    "position_world_m": corners_w[r.corner_index].tolist(),
                    "upper_hemisphere": bool(upper[r.corner_index]),
                    "inspected": r.inspected,
                    "dwell_count": r.dwell_count,
                    "best_sharpness_laplacian_var": r.best_sharpness,
                    "best_bisector_cos": r.best_bisector_cos,
                    "best_min_facet_cos": r.best_min_facet_cos,
                    "best_standoff_m": r.best_standoff_m,
                    "best_camera": r.best_camera,
                    "best_pixel_uv": list(r.best_pixel_uv),
                    "raw_anomaly_distance": r.raw_anomaly_distance,
                    "score_type": r.score_type,
                    "last_reason": r.last_reason,
                }
                for r in self.records
            ],
        }
