"""Non-VLM Dual-RGB-D 3D Cuboid Localization and Prior-Constrained Surface ICP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN, KMeans

from brickcell.block_inspector.geometry import (
    DUPLO_CATALOG,
    BlockSpec,
    CameraModel,
    so3_geodesic_angle_rad,
)

# Canonical +X-forward tabletop inspection arena between sorting plates at y = +-0.22 m
WORLD_ROI_MIN = np.array([0.11, -0.115, -0.01], dtype=np.float64)
WORLD_ROI_MAX = np.array([0.38, +0.115, +0.18], dtype=np.float64)

# Aquamarine source KALAS plate geometry (center at x=0.23, y=0.00; collision floor top at z=4.0 mm)
SOURCE_PLATE_XY = np.array([0.23, 0.00], dtype=np.float64)
SOURCE_PLATE_FLOOR_Z_M = 0.0040
SOURCE_PLATE_FLAT_RADIUS_M = 0.0755
SOURCE_PLATE_OUTER_RADIUS_M = 0.0980
SOURCE_PLATE_RGBA_RGB = np.array([0.28, 0.82, 0.80], dtype=np.float64)


@dataclass(frozen=True)
class LocalizedBlock:
    """Result of 3D RGB-D tabletop localization or post-manipulation prior-locked ICP."""

    T_world_block: np.ndarray  # (4, 4) SE(3) pose of body-centered cuboid frame B
    dims_m: tuple[float, float, float]
    icp_fitness: float
    icp_rmse_m: float
    color_consistency: float
    mean_hsv: tuple[float, float, float]
    num_cluster_points: int


@dataclass(frozen=True)
class ClusterBlockCandidate:
    """Segmented & 6D-localized individual DUPLO brick inside the aquamarine source plate."""

    block_spec: BlockSpec
    localized: LocalizedBlock
    cluster_pts_world: np.ndarray


def unproject_rgbd_to_world(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    camera: CameraModel,
    T_world_body3: np.ndarray | None = None,
    stride: int = 1,
    z_min_m: float = 0.06,
    z_max_m: float = 1.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject aligned metric depth + RGB into canonical +X-forward world coordinates."""
    d = np.asarray(depth_m, dtype=np.float64)[::stride, ::stride]
    c = np.asarray(rgb, dtype=np.uint8)[::stride, ::stride]
    rows, cols = d.shape
    u_grid, v_grid = np.meshgrid(
        np.arange(0, camera.width, stride)[:cols],
        np.arange(0, camera.height, stride)[:rows],
    )
    valid = np.isfinite(d) & (d >= z_min_m) & (d <= z_max_m)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    z = d[valid]
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    colors = c[valid]

    fx, fy = camera.K[0, 0], camera.K[1, 1]
    cx, cy = camera.K[0, 2], camera.K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    pts_cam = np.column_stack([x_cam, y_cam, z])

    T_wc = camera.world_transform(T_world_body3)
    pts_world = (pts_cam @ T_wc[:3, :3].T) + T_wc[:3, 3]

    in_roi = np.all((pts_world >= WORLD_ROI_MIN) & (pts_world <= WORLD_ROI_MAX), axis=1)
    return pts_world[in_roi], colors[in_roi]


def fuse_dual_rgbd_clouds(
    frames: list[tuple[np.ndarray, np.ndarray, CameraModel]],
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse multi-camera RGB-D frames into a single world-frame colored point cloud."""
    pts_list = []
    col_list = []
    for depth_m, rgb, cam in frames:
        pts, cols = unproject_rgbd_to_world(depth_m, rgb, cam, stride=stride)
        if len(pts) > 0:
            pts_list.append(pts)
            col_list.append(cols)
    if not pts_list:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(pts_list), np.vstack(col_list)


def _foreground_mask_ignoring_source_plate(
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    table_z_threshold_m: float = 0.0035,
    source_plate_xy: np.ndarray = SOURCE_PLATE_XY,
) -> np.ndarray:
    """Mask foreground object points while rejecting z=0 table and aquamarine source KALAS bowl."""
    if len(points_world) == 0:
        return np.zeros(0, dtype=bool)
    r_plate = np.linalg.norm(points_world[:, :2] - source_plate_xy[None, :], axis=1)
    z = points_world[:, 2]

    in_flat = r_plate <= SOURCE_PLATE_FLAT_RADIUS_M
    in_rim = (r_plate > SOURCE_PLATE_FLAT_RADIUS_M) & (r_plate <= SOURCE_PLATE_OUTER_RADIUS_M)
    rim_wall_z = SOURCE_PLATE_FLOOR_Z_M + 0.0205 * np.clip(
        (r_plate - 0.068) / (SOURCE_PLATE_OUTER_RADIUS_M - 0.068), 0.0, 1.0
    )

    if len(colors_rgb) == len(points_world):
        hsv = cv2.cvtColor(colors_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hue = hsv[:, 0].astype(np.float64)
        sat = hsv[:, 1].astype(np.float64)
        is_aqua_hue = (hue >= 79.0) & (hue <= 99.0) & (sat >= 38.0)
        is_brick_chromatic = (~is_aqua_hue) & (sat >= 70.0)
    else:
        is_aqua_hue = np.zeros(len(points_world), dtype=bool)
        is_brick_chromatic = np.zeros(len(points_world), dtype=bool)

    rim_thresh = np.where(is_brick_chromatic, rim_wall_z + 0.0035, 0.0265)
    above_support = np.where(
        in_flat,
        z > (SOURCE_PLATE_FLOOR_Z_M + 0.0028),
        np.where(in_rim, z > rim_thresh, z > table_z_threshold_m),
    )
    is_plate_plastic = is_aqua_hue & (r_plate <= SOURCE_PLATE_OUTER_RADIUS_M + 0.004) & (z <= 0.028)
    return above_support & (~is_plate_plastic)


def _support_floor_z_at_xy(
    cx: float, cy: float, source_plate_xy: np.ndarray = SOURCE_PLATE_XY
) -> float:
    if float(math.hypot(cx - float(source_plate_xy[0]), cy - float(source_plate_xy[1]))) <= 0.078:
        return SOURCE_PLATE_FLOOR_Z_M
    return 0.0


def segment_foreground_block(
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    table_z_threshold_m: float = 0.0035,
    eps_m: float = 0.012,
    min_samples: int = 20,
    target_selector_fn: Callable[[list[np.ndarray], list[np.ndarray]], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract table & source KALAS plate, then select foreground block cluster via DBSCAN."""
    above_table = _foreground_mask_ignoring_source_plate(
        points_world, colors_rgb, table_z_threshold_m=table_z_threshold_m
    )
    fg_pts = points_world[above_table]
    fg_cols = colors_rgb[above_table]
    if len(fg_pts) < min_samples:
        raise RuntimeError(
            f"Insufficient foreground points above table ({len(fg_pts)} < {min_samples})"
        )

    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(fg_pts)
    unique_labels = [lbl for lbl in sorted(set(labels)) if lbl >= 0]
    if not unique_labels:
        raise RuntimeError("DBSCAN found no foreground object cluster in tabletop ROI")

    clusters_pts = [fg_pts[labels == lbl] for lbl in unique_labels]
    clusters_cols = [fg_cols[labels == lbl] for lbl in unique_labels]

    if target_selector_fn is not None:
        chosen = int(target_selector_fn(clusters_pts, clusters_cols))
    else:
        workspace_center = np.array([0.23, 0.0], dtype=np.float64)
        chosen = max(
            range(len(clusters_pts)),
            key=lambda idx: (
                len(clusters_pts[idx])
                - 250.0
                * float(np.linalg.norm(clusters_pts[idx][:, :2].mean(axis=0) - workspace_center))
            ),
        )
    return clusters_pts[chosen], clusters_cols[chosen]


def _cuboid_surface_residuals_and_jacobian(
    pts_world: np.ndarray, T_wb: np.ndarray, half_extents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance from world points to an oriented cuboid surface and SE(3) Jacobian."""
    R = T_wb[:3, :3]
    p = T_wb[:3, 3]
    # Points in cuboid local frame B: p_local = R^T (p_world - p)
    p_local = (pts_world - p) @ R
    q = np.abs(p_local) - half_extents[None, :]

    outside = np.maximum(q, 0.0)
    outside_norm = np.linalg.norm(outside, axis=1)
    inside_dist = np.minimum(np.max(q, axis=1), 0.0)
    signed_dist = outside_norm + inside_dist

    # Local gradient d(dist)/d(p_local)
    grad_local = np.zeros_like(p_local)
    is_out = outside_norm > 1e-9
    if np.any(is_out):
        grad_local[is_out] = np.sign(p_local[is_out]) * outside[is_out] / outside_norm[is_out, None]
    is_in = ~is_out
    if np.any(is_in):
        nearest_face = np.argmax(q[is_in], axis=1)
        signs = np.sign(p_local[is_in])
        signs[signs == 0.0] = 1.0
        idx_rows = np.arange(np.sum(is_in))
        grad_local[is_in, nearest_face] = signs[idx_rows, nearest_face]

    # Right perturbation T_wb <- T_wb @ exp([delta_p_B; delta_omega_B]):
    # d(dist) = grad . (-delta_p_B - delta_omega_B x p_local) = -grad . delta_p_B + (grad x p_local) . delta_omega_B
    J = np.empty((len(pts_world), 6), dtype=np.float64)
    J[:, :3] = -grad_local
    J[:, 3:] = np.cross(grad_local, p_local)
    return signed_dist, J


def refine_cuboid_icp(
    cluster_pts_world: np.ndarray,
    T_init: np.ndarray,
    dims_m: tuple[float, float, float],
    max_iters: int = 15,
    huber_delta_m: float = 0.006,
    lock_table_resting: bool = True,
    max_rot_from_init_rad: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Huber-loss Gauss-Newton point-to-cuboid-surface ICP refinement."""
    pts = np.asarray(cluster_pts_world, dtype=np.float64)
    if len(pts) > 800:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), size=800, replace=False)]

    T_cur = np.array(T_init, dtype=np.float64, copy=True)
    R_init = T_init[:3, :3].copy()
    half = 0.5 * np.asarray(dims_m, dtype=np.float64)

    for _ in range(max_iters):
        res, J = _cuboid_surface_residuals_and_jacobian(pts, T_cur, half)
        abs_r = np.abs(res)
        weights = np.where(abs_r <= huber_delta_m, 1.0, huber_delta_m / np.maximum(abs_r, 1e-9))
        weights = np.where(res < -0.0035, 0.10 * weights, weights)

        if lock_table_resting:
            # Restrict rotation to yaw about world Z and keep bottom face on z_world = 0
            # Find which local axis aligns with world +Z so yaw in local frame is about that axis:
            z_in_local = T_cur[:3, :3].T @ np.array([0.0, 0.0, 1.0])
            J_yaw = J[:, 3:] @ z_in_local
            # Restrict translation to world XY:
            xy_in_local = T_cur[:3, :3].T[:, :2]  # (3, 2)
            J_xy = J[:, :3] @ xy_in_local  # (N, 2)
            J_red = np.column_stack([J_xy, J_yaw])  # (N, 3)
            W_J = J_red * weights[:, None]
            H_mat = J_red.T @ W_J + 1e-4 * np.eye(3)
            g_vec = J_red.T @ (weights * res)
            step = -np.linalg.solve(H_mat, g_vec)
            step[:2] = np.clip(step[:2], -0.015, 0.015)
            step[2] = float(np.clip(step[2], -0.15, 0.15))

            T_cand = T_cur.copy()
            T_cand[:2, 3] += step[:2]
            T_cand[:3, :3] = (
                Rotation.from_rotvec(np.array([0.0, 0.0, step[2]])).as_matrix() @ T_cur[:3, :3]
            )
        else:
            W_J = J * weights[:, None]
            H_mat = J.T @ W_J + 1e-3 * np.eye(6)
            g_vec = J.T @ (weights * res)
            step = -np.linalg.solve(H_mat, g_vec)
            step[:3] = np.clip(step[:3], -0.012, 0.012)
            step[3:] = np.clip(step[3:], -0.12, 0.12)
            T_delta = np.eye(4, dtype=np.float64)
            T_delta[:3, :3] = Rotation.from_rotvec(step[3:]).as_matrix()
            T_delta[:3, 3] = step[:3]
            T_cand = T_cur @ T_delta

        if max_rot_from_init_rad is not None:
            if so3_geodesic_angle_rad(R_init, T_cand[:3, :3]) >= max_rot_from_init_rad:
                break

        T_cur = T_cand
        if float(np.linalg.norm(step)) < 1e-5:
            break

    final_res, _ = _cuboid_surface_residuals_and_jacobian(pts, T_cur, half)
    inliers = np.abs(final_res) <= 0.0065
    fitness = float(np.mean(inliers)) if len(final_res) else 0.0
    rmse = float(np.sqrt(np.mean(final_res**2))) if len(final_res) else 0.0
    return T_cur, fitness, rmse


def _mean_hsv(colors_rgb: np.ndarray) -> tuple[float, float, float]:
    if len(colors_rgb) == 0:
        return (0.0, 0.0, 0.0)
    hsv = cv2.cvtColor(colors_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    m = hsv.mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def _fit_resting_cuboid_to_cluster(
    cluster_pts: np.ndarray,
    cluster_cols: np.ndarray,
    block_spec: BlockSpec | None = None,
    support_z_m: float | None = None,
) -> LocalizedBlock:
    pts_xy = cluster_pts[:, :2].astype(np.float32)
    (cx, cy), (w_rect, h_rect), angle_deg = cv2.minAreaRect(pts_xy)
    yaw_rad = math.radians(angle_deg)
    if w_rect < h_rect:
        w_rect, h_rect = h_rect, w_rect
        yaw_rad += 0.5 * math.pi

    floor_z = _support_floor_z_at_xy(float(cx), float(cy)) if support_z_m is None else support_z_m
    z_top = float(np.percentile(cluster_pts[:, 2], 96))
    if block_spec is not None:
        dims_m = block_spec.dims_m
        lx, wy, hz = dims_m
        if (lx < wy and w_rect >= h_rect) or (lx > wy and w_rect < h_rect):
            yaw_rad += 0.5 * math.pi
        z_center = floor_z + 0.5 * hz
    else:
        est_h = max(z_top - floor_z, 0.014)
        dims_m = (max(float(w_rect), 0.014), max(float(h_rect), 0.014), est_h)
        z_center = floor_z + 0.5 * dims_m[2]

    best_T = np.eye(4, dtype=np.float64)
    best_fit = -1.0
    best_rmse = 1e9
    for k_quarter in (
        (0, 2)
        if (block_spec and abs(block_spec.dims_m[0] - block_spec.dims_m[1]) > 0.006)
        else (0, 1, 2, 3)
    ):
        cand_yaw = yaw_rad + k_quarter * 0.5 * math.pi
        T0 = np.eye(4, dtype=np.float64)
        T0[:3, :3] = Rotation.from_euler("z", cand_yaw).as_matrix()
        T0[:3, 3] = [float(cx), float(cy), z_center]
        T_opt, fit, rmse = refine_cuboid_icp(cluster_pts, T0, dims_m, lock_table_resting=True)
        score = fit - 20.0 * rmse
        if score > best_fit - 20.0 * best_rmse:
            best_T, best_fit, best_rmse = T_opt, fit, rmse

    return LocalizedBlock(
        T_world_block=best_T,
        dims_m=dims_m,
        icp_fitness=best_fit,
        icp_rmse_m=best_rmse,
        color_consistency=1.0,
        mean_hsv=_mean_hsv(cluster_cols),
        num_cluster_points=len(cluster_pts),
    )


def localize_resting_block(
    frames: list[tuple[np.ndarray, np.ndarray, CameraModel]],
    block_spec: BlockSpec | None = None,
    target_selector_fn: Callable[[list[np.ndarray], list[np.ndarray]], int] | None = None,
) -> LocalizedBlock:
    """Localize a resting tabletop block from Dual-D415 RGB-D views in +X-forward world frame."""
    pts_world, cols_rgb = fuse_dual_rgbd_clouds(frames, stride=2)
    cluster_pts, cluster_cols = segment_foreground_block(
        pts_world, cols_rgb, target_selector_fn=target_selector_fn
    )
    return _fit_resting_cuboid_to_cluster(cluster_pts, cluster_cols, block_spec=block_spec)


def infer_duplo_spec_from_cluster(
    cluster_pts: np.ndarray,
    cluster_cols: np.ndarray,
    support_z_m: float = SOURCE_PLATE_FLOOR_Z_M,
) -> BlockSpec:
    """Identify which DUPLO_CATALOG brick a 3D RGB-D cluster matches from geometry + median RGB."""
    pts_xy = cluster_pts[:, :2].astype(np.float32)
    _, (w_rect, h_rect), _ = cv2.minAreaRect(pts_xy)
    obs_major = max(float(w_rect), float(h_rect))
    obs_minor = min(float(w_rect), float(h_rect))
    obs_h = max(float(np.percentile(cluster_pts[:, 2], 96)) - support_z_m, 0.015)

    # Robust body RGB ignoring small dark/magenta defect blemish patches
    cols_norm = cluster_cols.astype(np.float64) / 255.0
    med_rgb = np.median(cols_norm, axis=0)

    best_spec: BlockSpec | None = None
    best_cost = 1e9
    for spec in DUPLO_CATALOG.values():
        lx, wy, hz = spec.dims_m
        exp_major = max(lx, wy)
        exp_minor = min(lx, wy)
        spec_rgb = np.asarray(spec.rgba[:3], dtype=np.float64)
        # Note: rendered MuJoCo shading scales diffuse brightness; compare chromaticity + height + span
        obs_c = med_rgb / max(float(np.sum(med_rgb)), 1e-6)
        exp_c = spec_rgb / max(float(np.sum(spec_rgb)), 1e-6)
        chrom_err = float(np.linalg.norm(obs_c - exp_c))
        geom_err = (
            2.5 * abs(obs_h - hz)
            + 1.6 * abs(obs_major - exp_major)
            + 1.2 * abs(obs_minor - exp_minor)
        )
        cost = chrom_err + 8.0 * geom_err
        if cost < best_cost:
            best_cost = cost
            best_spec = spec
    assert best_spec is not None
    return best_spec


def _split_same_color_if_oversize(
    pts_sub: np.ndarray, cols_sub: np.ndarray, density_scale: float = 1.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split touching same-color bricks using 2D footprint / point-count KMeans."""
    min_pts = max(36, int(round(38.0 * density_scale)))
    if len(pts_sub) < min_pts:
        return []
    pts_xy = pts_sub[:, :2].astype(np.float32)
    _, (w_r, h_r), _ = cv2.minAreaRect(pts_xy)
    max_s = max(float(w_r), float(h_r))
    min_s = min(float(w_r), float(h_r))
    obs_h = float(np.percentile(pts_sub[:, 2], 96)) - SOURCE_PLATE_FLOOR_Z_M
    tall_pair = obs_h > 0.034 and (min_s > 0.029 or max_s > 0.044)
    if (
        len(pts_sub) <= int(1180 * density_scale)
        and max_s <= 0.070
        and min_s <= 0.041
        and not tall_pair
    ):
        return [(pts_sub, cols_sub)]
    n_k = max(
        2,
        min(
            4,
            int(
                round(
                    max(
                        len(pts_sub) / (520.0 * density_scale),
                        (max_s * min_s) / (0.032 * 0.034),
                    )
                )
            ),
        ),
    )
    km = KMeans(n_clusters=n_k, n_init=5, random_state=0).fit_predict(pts_sub[:, :2])
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for u in range(n_k):
        if int(np.sum(km == u)) >= min_pts:
            out.append((pts_sub[km == u], cols_sub[km == u]))
    return out or [(pts_sub, cols_sub)]


def _split_touching_cluster_if_needed(
    c_pts: np.ndarray, c_cols: np.ndarray, density_scale: float = 1.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split an oversize merged cluster into individual bricks by body hue or XY KMeans."""
    cols_f = c_cols.astype(np.float64) / 255.0
    c_max = np.max(cols_f, axis=1)
    c_min = np.min(cols_f, axis=1)
    sat = (c_max - c_min) / np.maximum(c_max, 1e-6)
    is_magenta_decal = (cols_f[:, 0] > 0.42) & (cols_f[:, 2] > 0.35) & (cols_f[:, 1] < 0.24)
    is_body_pixel = (sat >= 0.32) & (~is_magenta_decal)

    cat_chroms = []
    for spec in DUPLO_CATALOG.values():
        rgb = np.asarray(spec.rgba[:3], dtype=np.float64)
        cat_chroms.append(rgb / max(float(np.sum(rgb)), 1e-6))
    cat_mat = np.vstack(cat_chroms)  # (4, 3)

    sums = np.maximum(np.sum(cols_f, axis=1, keepdims=True), 1e-6)
    chrom = cols_f / sums
    dists = np.linalg.norm(chrom[:, None, :] - cat_mat[None, :, :], axis=2)
    nearest_cat = np.argmin(dists, axis=1)
    min_dist = np.min(dists, axis=1)

    min_group_pts = max(int(round(45 * density_scale)), int(0.16 * int(np.sum(is_body_pixel))))
    valid_groups = [
        k
        for k in range(len(cat_chroms))
        if int(np.sum(is_body_pixel & (nearest_cat == k) & (min_dist < 0.14))) >= min_group_pts
    ]

    if len(valid_groups) <= 1:
        return _split_same_color_if_oversize(c_pts, c_cols, density_scale=density_scale)

    subclusters: list[tuple[np.ndarray, np.ndarray]] = []
    clean_mask = is_body_pixel & np.isin(nearest_cat, valid_groups) & (min_dist < 0.16)
    assigned_cat = nearest_cat.copy()
    if np.any(~clean_mask) and np.any(clean_mask):
        clean_pts = c_pts[clean_mask]
        clean_lbl = nearest_cat[clean_mask]
        d2 = np.sum(
            (c_pts[~clean_mask, None, :] - clean_pts[None, :, :]) ** 2,
            axis=2,
        )
        assigned_cat[~clean_mask] = clean_lbl[np.argmin(d2, axis=1)]
    for k in valid_groups:
        m = assigned_cat == k
        pts_k, cols_k = c_pts[m], c_cols[m]
        lbls = DBSCAN(eps=0.0060, min_samples=12).fit_predict(pts_k)
        for u in sorted(set(lbls)):
            if u >= 0 and int(np.sum(lbls == u)) >= max(36, int(round(38 * density_scale))):
                subclusters.extend(
                    _split_same_color_if_oversize(
                        pts_k[lbls == u], cols_k[lbls == u], density_scale=density_scale
                    )
                )
    return subclusters or _split_same_color_if_oversize(c_pts, c_cols, density_scale=density_scale)


def localize_clustered_blocks_in_source_plate(
    frames: list[tuple[np.ndarray, np.ndarray, CameraModel]],
    source_plate_xy: np.ndarray = SOURCE_PLATE_XY,
    stride: int = 2,
) -> tuple[list[ClusterBlockCandidate], np.ndarray]:
    """Segment and 6D-localize all remaining DUPLO blocks inside the aquamarine source plate."""
    pts_world, cols_rgb = fuse_dual_rgbd_clouds(frames, stride=stride)
    if len(pts_world) == 0:
        return [], np.zeros((0, 3), dtype=np.float64)

    density_scale = (2.0 / max(1, int(stride))) ** 2
    min_cluster_pts = max(36, int(round(38.0 * density_scale)))
    in_source_bowl = (
        np.linalg.norm(pts_world[:, :2] - source_plate_xy[None, :], axis=1)
        <= SOURCE_PLATE_OUTER_RADIUS_M + 0.006
    )
    fg_mask = in_source_bowl & _foreground_mask_ignoring_source_plate(
        pts_world, cols_rgb, source_plate_xy=source_plate_xy
    )
    fg_pts = pts_world[fg_mask]
    fg_cols = cols_rgb[fg_mask]
    if len(fg_pts) < min_cluster_pts:
        return [], np.zeros((0, 3), dtype=np.float64)

    labels = DBSCAN(eps=0.0058, min_samples=max(12, int(round(14 * (2.0 / stride))))).fit_predict(
        fg_pts
    )
    raw_clusters = [
        (fg_pts[labels == lbl], fg_cols[labels == lbl])
        for lbl in sorted(set(labels))
        if lbl >= 0 and int(np.sum(labels == lbl)) >= min_cluster_pts
    ]

    candidates: list[ClusterBlockCandidate] = []
    for r_pts, r_cols in raw_clusters:
        for s_pts, s_cols in _split_touching_cluster_if_needed(
            r_pts, r_cols, density_scale=density_scale
        ):
            if len(s_pts) < min_cluster_pts:
                continue
            spec = infer_duplo_spec_from_cluster(s_pts, s_cols, support_z_m=SOURCE_PLATE_FLOOR_Z_M)
            loc = _fit_resting_cuboid_to_cluster(
                s_pts, s_cols, block_spec=spec, support_z_m=SOURCE_PLATE_FLOOR_Z_M
            )
            candidates.append(
                ClusterBlockCandidate(
                    block_spec=spec,
                    localized=loc,
                    cluster_pts_world=s_pts,
                )
            )
    return candidates, fg_pts


def _snap_rotation_to_resting_face(R_prior: np.ndarray) -> np.ndarray:
    """Level small post-release gripper tilt (<15 deg) so the nearest cuboid face rests flat on z=0."""
    R = np.asarray(R_prior, dtype=np.float64)
    z_local = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    k = int(np.argmax(np.abs(z_local)))
    sign = 1.0 if z_local[k] >= 0.0 else -1.0
    n_world = sign * R[:, k]
    axis = np.cross(n_world, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.clip(n_world[2], -1.0, 1.0))
    if sin_a > 1e-8 and math.atan2(sin_a, cos_a) < math.radians(28.0):
        rotvec = (math.atan2(sin_a, cos_a) / sin_a) * axis
        R = Rotation.from_rotvec(rotvec).as_matrix() @ R
    return R


def relocalize_with_prior(
    frames: list[tuple[np.ndarray, np.ndarray, CameraModel]],
    T_prior: np.ndarray,
    dims_m: tuple[float, float, float],
    reference_hsv: tuple[float, float, float] | None = None,
    max_rot_deg: float = 35.0,
) -> LocalizedBlock:
    """Post-yaw / post-flip ICP strictly initialized at T_prior and bounded within 35 deg."""
    pts_world, cols_rgb = fuse_dual_rgbd_clouds(frames, stride=2)
    cluster_pts, cluster_cols = segment_foreground_block(pts_world, cols_rgb)

    # Seed XY centroid at observed cluster and level settled resting face while preserving R_prior branch!
    T_seed = np.array(T_prior, dtype=np.float64, copy=True)
    T_seed[:3, :3] = _snap_rotation_to_resting_face(T_seed[:3, :3])
    cx = float(np.median(cluster_pts[:, 0]))
    cy = float(np.median(cluster_pts[:, 1]))
    T_seed[0, 3] = cx
    T_seed[1, 3] = cy
    # Resting center height from leveled R_seed applied to half-extents + plate/table support floor:
    half_world_z = float(np.sum(np.abs(T_seed[2, :3]) * (0.5 * np.asarray(dims_m))))
    T_seed[2, 3] = _support_floor_z_at_xy(cx, cy) + half_world_z

    T_opt, fitness, rmse = refine_cuboid_icp(
        cluster_pts,
        T_seed,
        dims_m,
        max_iters=15,
        lock_table_resting=True,
        max_rot_from_init_rad=math.radians(max_rot_deg),
    )

    obs_hsv = _mean_hsv(cluster_cols)
    color_consistency = 1.0
    if reference_hsv is not None and len(cluster_cols) > 0:
        dh = (
            min(abs(obs_hsv[0] - reference_hsv[0]), 180.0 - abs(obs_hsv[0] - reference_hsv[0]))
            / 90.0
        )
        ds = abs(obs_hsv[1] - reference_hsv[1]) / 255.0
        color_consistency = float(np.clip(1.0 - 0.6 * dh - 0.4 * ds, 0.0, 1.0))

    return LocalizedBlock(
        T_world_block=T_opt,
        dims_m=dims_m,
        icp_fitness=fitness,
        icp_rmse_m=rmse,
        color_consistency=color_consistency,
        mean_hsv=obs_hsv,
        num_cluster_points=len(cluster_pts),
    )
