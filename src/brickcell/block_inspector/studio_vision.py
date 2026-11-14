"""MuJoCo Studio 3D Vision Overlays & Live Virtual Camera Windows for SO-101 Inspection Cell.

Provides two synchronized visual layers without interrupting background training:
1. Main 3D Scene Overlays (`populate_studio_3d_scene` in `viewer.user_scn`):
   Renders in the interactive 3D MuJoCo Studio environment (where you can freely orbit, pan,
   and zoom) what the arm and cameras are seeing and doing in 3D:
   - Live 3D viewing frustums (FOV pyramids, image-plane frames, and optical sightlines) for:
     * `wrist` (25-deg canted Sonix UVC eye-in-hand camera moving with `Body3` on the arm)
     * `realsense_2` (Low Plate D415 camera aimed at the Joint-5 CCW spin waypoint)
     * `realsense_1` (Overhead Top D415 camera covering the 3 IKEA KALAS plates)
   - Live 3D perceived point cloud spheres from Dual-D415 RGB-D foreground segmentation
   - 3D 12-edge oriented wireframe bounding boxes (OBBs), 6D pose triads, and 8 corner markers
     (`c_0..c_7`) for localized and held DUPLO bricks
   - 3D manipulation intent ray from the gripper TCP to the active pick/spin/sort target
2. Native Desktop Virtual Camera Windows (`VirtualCameraWindowsManager`):
   Opens 3 live native desktop windows alongside MuJoCo Studio showing what each virtual camera
   (`Wrist / Gripper`, `Plate / Side D415`, and `Overhead / Top D415`) is seeing in real time,
   complete with metric depth colormaps, projected 3D brick wireframes, 96x96 inspection ROI
   crops, and raw anomaly distance telemetry.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
import threading
import time
from typing import Any

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageTk

from brickcell.block_inspector.geometry import (
    CORNER_SIGNS,
    PLATE_DROP_SLOT_OFFSETS_XY,
    REPO_ROOT,
    BlockSpec,
    CameraModel,
)

# 12 edges of a canonical 8-corner cuboid indexed by CORNER_SIGNS (0..3 bottom, 4..7 top)
_CUBOID_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _add_capsule(
    scn: Any,
    p_from: np.ndarray,
    p_to: np.ndarray,
    radius: float,
    rgba: np.ndarray,
    eye_mat: np.ndarray,
) -> bool:
    """Append one 3D capsule segment between `p_from` and `p_to` into `viewer.user_scn`."""
    if scn.ngeom >= scn.maxgeom - 2:
        return False
    p0 = np.asarray(p_from, dtype=np.float64)
    p1 = np.asarray(p_to, dtype=np.float64)
    if float(np.linalg.norm(p1 - p0)) < 1e-6:
        return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        eye_mat,
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), float(radius), p0, p1)
    scn.ngeom += 1
    return True


def _add_sphere(
    scn: Any,
    pos: np.ndarray,
    radius: float,
    rgba: np.ndarray,
    eye_mat: np.ndarray,
) -> bool:
    """Append one 3D sphere marker at `pos` into `viewer.user_scn`."""
    if scn.ngeom >= scn.maxgeom - 2:
        return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([float(radius), 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        eye_mat,
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1
    return True


def _camera_frustum_world_points(
    cam: CameraModel,
    T_world_body3: np.ndarray,
    frustum_depth_m: float,
    axis_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute world optical center, 4 frustum image-plane corners, top-up tick, and optical tip."""
    T_wc = cam.world_transform(T_world_body3)
    R_wc = T_wc[:3, :3]
    p_cam = T_wc[:3, 3].astype(np.float64)
    fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
    cx, cy = float(cam.K[0, 2]), float(cam.K[1, 2])
    w, h = float(cam.width), float(cam.height)
    z = float(frustum_depth_m)

    uv_corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    corners_cam = np.array(
        [[(u - cx) * z / fx, (v - cy) * z / fy, z] for u, v in uv_corners],
        dtype=np.float64,
    )
    corners_world = (corners_cam @ R_wc.T) + p_cam

    # Upward orientation tick at top-center of image plane (-Y_cvcam is image top)
    top_mid_cam = 0.5 * (corners_cam[0] + corners_cam[1])
    up_tick_cam = top_mid_cam + np.array([0.0, -0.14 * z, 0.0], dtype=np.float64)
    top_tick_world = np.vstack([(top_mid_cam @ R_wc.T) + p_cam, (up_tick_cam @ R_wc.T) + p_cam])

    axis_tip_world = p_cam + R_wc[:, 2] * float(axis_depth_m)
    return p_cam, corners_world, top_tick_world, axis_tip_world


def _draw_camera_frustum_3d(
    scn: Any,
    cam: CameraModel,
    T_world_body3: np.ndarray,
    frustum_depth_m: float,
    axis_depth_m: float,
    rgba_pyramid: np.ndarray,
    rgba_axis: np.ndarray,
    eye_mat: np.ndarray,
    radius_m: float = 0.0012,
) -> None:
    """Render a 3D camera frustum pyramid + image frame + optical sightline ray in `user_scn`."""
    p_cam, corners_w, top_tick_w, axis_tip_w = _camera_frustum_world_points(
        cam, T_world_body3, frustum_depth_m, axis_depth_m
    )
    for idx in range(4):
        _add_capsule(scn, p_cam, corners_w[idx], radius_m, rgba_pyramid, eye_mat)
        _add_capsule(
            scn,
            corners_w[idx],
            corners_w[(idx + 1) % 4],
            radius_m * 1.15,
            rgba_pyramid,
            eye_mat,
        )
    _add_capsule(scn, top_tick_w[0], top_tick_w[1], radius_m * 1.25, rgba_axis, eye_mat)
    _add_capsule(scn, p_cam, axis_tip_w, radius_m * 0.95, rgba_axis, eye_mat)


def _draw_obb_wireframe_3d(
    scn: Any,
    T_world_block: np.ndarray,
    dims_m: tuple[float, float, float],
    rgba: np.ndarray,
    eye_mat: np.ndarray,
    radius_m: float = 0.0013,
    top_only: bool = False,
    draw_axes: bool = False,
) -> np.ndarray:
    """Render a 3D oriented bounding box (12 edges or 4 top edges) around a perceived brick."""
    T = np.asarray(T_world_block, dtype=np.float64)
    half = 0.5 * np.asarray(dims_m, dtype=np.float64)
    corners_local = CORNER_SIGNS * half[None, :]
    corners_world = (corners_local @ T[:3, :3].T) + T[:3, 3]

    edges = ((4, 5), (5, 6), (6, 7), (7, 4)) if top_only else _CUBOID_EDGES
    for i0, i1 in edges:
        _add_capsule(scn, corners_world[i0], corners_world[i1], radius_m, rgba, eye_mat)

    if draw_axes:
        p_c = T[:3, 3]
        R_c = T[:3, :3]
        axis_len = 0.024
        _add_capsule(
            scn,
            p_c,
            p_c + R_c[:, 0] * axis_len,
            0.0014,
            np.array([1.0, 0.22, 0.22, 0.92], dtype=np.float32),
            eye_mat,
        )
        _add_capsule(
            scn,
            p_c,
            p_c + R_c[:, 1] * axis_len,
            0.0014,
            np.array([0.22, 0.95, 0.32, 0.92], dtype=np.float32),
            eye_mat,
        )
        _add_capsule(
            scn,
            p_c,
            p_c + R_c[:, 2] * axis_len,
            0.0014,
            np.array([0.28, 0.62, 1.00, 0.92], dtype=np.float32),
            eye_mat,
        )
    return corners_world


def populate_studio_3d_scene(twin: Any, scn: Any) -> None:
    """Populate `viewer.user_scn` with 3D camera frustums, perceived point cloud, and brick OBBs."""
    from brickcell.block_inspector.twin import (
        DEFECTIVE_PILE_XY,
        GOOD_PILE_XY,
        PLATE_CAMERA_SCAN_POS,
    )

    scn.ngeom = 0
    eye_mat = np.eye(3, dtype=np.float64).ravel()
    T_b3 = twin.body3_world_pose()

    # 1. 3D Viewing Frustums & Sightlines for all 3 Virtual Cameras
    if getattr(twin, "show_3d_camera_frustums", True):
        # 1a. Arm Wrist / Gripper Camera (moves live with Body3!)
        _draw_camera_frustum_3d(
            scn,
            twin.rig.wrist,
            T_b3,
            frustum_depth_m=0.145,
            axis_depth_m=0.210,
            rgba_pyramid=np.array([1.00, 0.82, 0.16, 0.85], dtype=np.float32),
            rgba_axis=np.array([1.00, 0.92, 0.35, 0.92], dtype=np.float32),
            eye_mat=eye_mat,
            radius_m=0.0013,
        )

        # 1b. Low Plate Camera (`realsense_2` / `side`)
        if twin.plate_scan_active:
            plate_pyr_rgba = (
                np.array([1.00, 0.25, 0.22, 0.90], dtype=np.float32)
                if twin.anomaly_flagged
                else np.array([0.18, 0.96, 0.52, 0.88], dtype=np.float32)
            )
        else:
            plate_pyr_rgba = np.array([0.20, 0.86, 0.95, 0.58], dtype=np.float32)
        _draw_camera_frustum_3d(
            scn,
            twin.rig.realsense_2,
            T_b3,
            frustum_depth_m=0.220,
            axis_depth_m=0.310,
            rgba_pyramid=plate_pyr_rgba,
            rgba_axis=plate_pyr_rgba,
            eye_mat=eye_mat,
            radius_m=0.0011,
        )

        # 3D wireframe target inspection zone around PLATE_CAMERA_SCAN_POS
        T_scan_zone = np.eye(4, dtype=np.float64)
        T_scan_zone[:3, 3] = PLATE_CAMERA_SCAN_POS
        _draw_obb_wireframe_3d(
            scn,
            T_scan_zone,
            dims_m=(0.052, 0.052, 0.052),
            rgba=plate_pyr_rgba,
            eye_mat=eye_mat,
            radius_m=0.0009,
            top_only=False,
            draw_axes=False,
        )

        # 1c. Overhead Camera (`realsense_1` / `overhead`)
        _draw_camera_frustum_3d(
            scn,
            twin.rig.realsense_1,
            T_b3,
            frustum_depth_m=0.240,
            axis_depth_m=0.340,
            rgba_pyramid=np.array([0.38, 0.72, 1.00, 0.48], dtype=np.float32),
            rgba_axis=np.array([0.48, 0.80, 1.00, 0.62], dtype=np.float32),
            eye_mat=eye_mat,
            radius_m=0.0010,
        )

    # 2. Active Plate-camera optical beam during Joint-5 CCW spin
    if twin.plate_scan_active:
        p_cam = twin.rig.realsense_2.T_parent_cvcam[:3, 3].astype(np.float64)
        T_held = T_b3 @ twin._T_body3_block_held
        p_blk = T_held[:3, 3].astype(np.float64)
        beam_rgba = (
            np.array([1.0, 0.25, 0.20, 0.85], dtype=np.float32)
            if twin.anomaly_flagged
            else np.array([0.20, 0.95, 0.55, 0.78], dtype=np.float32)
        )
        _add_capsule(scn, p_cam, p_blk, 0.0020, beam_rgba, eye_mat)

    # 3. Target sorting pile beacon cylinder
    if twin.plate_scan_completed or twin.sorted_pile is not None:
        dest = twin.sorted_pile or ("defective" if twin.anomaly_flagged else "good")
        xy = DEFECTIVE_PILE_XY if dest == "defective" else GOOD_PILE_XY
        ring_rgba = (
            np.array([1.0, 0.30, 0.45, 0.85], dtype=np.float32)
            if dest == "defective"
            else np.array([0.18, 0.92, 0.45, 0.85], dtype=np.float32)
        )
        if scn.ngeom < scn.maxgeom - 2:
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                np.array([0.086, 0.0025, 0.0], dtype=np.float64),
                np.array([xy[0], xy[1], 0.026], dtype=np.float64),
                eye_mat,
                ring_rgba,
            )
            scn.ngeom += 1

    # 4. Perceived Cluster Candidates & Foreground 3D Point Cloud in Source Plate
    cands = getattr(twin, "_perceived_cluster_candidates", None) or []
    for cand in cands:
        T_c = cand.localized.T_world_block
        if (
            twin.ledger.localized
            and float(np.linalg.norm(T_c[:3, 3] - twin.ledger.T_world_block[:3, 3])) < 0.012
        ):
            continue
        c_rgba = np.array([0.30, 0.88, 0.98, 0.72], dtype=np.float32)
        _draw_obb_wireframe_3d(
            scn,
            T_c,
            cand.block_spec.dims_m,
            rgba=c_rgba,
            eye_mat=eye_mat,
            radius_m=0.0010,
            top_only=True,
            draw_axes=False,
        )

    fg_pts = getattr(twin, "_perceived_fg_pts_world", None)
    if fg_pts is not None and len(fg_pts) > 0:
        stride = max(1, len(fg_pts) // 90)
        pt_rgba = np.array([0.28, 0.95, 1.00, 0.62], dtype=np.float32)
        for pt in fg_pts[::stride]:
            if scn.ngeom >= scn.maxgeom - 48:
                break
            _add_sphere(scn, pt, 0.0022, pt_rgba, eye_mat)

    # 5. Active Localized or Held Brick 3D Wireframe OBB, 6D Pose Axes, 8 Corners & Intent Ray
    jaw_f, jaw_m = twin.jaw_positions_world()
    p_tcp = 0.5 * (jaw_f + jaw_m)

    if not twin.hand_empty:
        T_active = T_b3 @ twin._T_body3_block_held
        active_rgba = (
            np.array([1.00, 0.25, 0.25, 0.95], dtype=np.float32)
            if twin.anomaly_flagged
            else np.array([0.22, 0.96, 0.52, 0.92], dtype=np.float32)
        )
        corners_w = _draw_obb_wireframe_3d(
            scn,
            T_active,
            twin.block_spec.dims_m,
            rgba=active_rgba,
            eye_mat=eye_mat,
            radius_m=0.0014,
            top_only=False,
            draw_axes=True,
        )
        for i in range(8):
            _add_sphere(scn, corners_w[i], 0.0032, active_rgba, eye_mat)

        # Draw manipulation intent ray from held brick to next waypoint (Plate cam or target plate)
        if not twin.plate_scan_completed:
            _add_capsule(
                scn,
                T_active[:3, 3],
                PLATE_CAMERA_SCAN_POS,
                0.0014,
                np.array([0.35, 0.92, 1.00, 0.72], dtype=np.float32),
                eye_mat,
            )
        else:
            dest = twin.sorted_pile or ("defective" if twin.anomaly_flagged else "good")
            base_xy = DEFECTIVE_PILE_XY if dest == "defective" else GOOD_PILE_XY
            slot_idx = int(twin.sorted_counts[dest]) % len(PLATE_DROP_SLOT_OFFSETS_XY)
            dx, dy = (
                PLATE_DROP_SLOT_OFFSETS_XY[slot_idx] if len(twin.spawned_blocks) > 1 else (0.0, 0.0)
            )
            p_drop = np.array([base_xy[0] + dx, base_xy[1] + dy, 0.025], dtype=np.float64)
            _add_capsule(scn, T_active[:3, 3], p_drop, 0.0016, active_rgba, eye_mat)

    elif twin.ledger.localized and twin.sorted_pile is None:
        T_active = twin.ledger.T_world_block
        p_sel = T_active[:3, 3]
        if len(twin.spawned_blocks) > 1 and scn.ngeom < scn.maxgeom - 4:
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                np.array([0.024, 0.0020, 0.0], dtype=np.float64),
                np.array([p_sel[0], p_sel[1], p_sel[2] + 0.024], dtype=np.float64),
                eye_mat,
                np.array([1.00, 0.86, 0.15, 0.88], dtype=np.float32),
            )
            scn.ngeom += 1

        _draw_obb_wireframe_3d(
            scn,
            T_active,
            twin.block_spec.dims_m,
            rgba=np.array([1.00, 0.84, 0.16, 0.92], dtype=np.float32),
            eye_mat=eye_mat,
            radius_m=0.0014,
            top_only=False,
            draw_axes=True,
        )

        # Draw intent ray from arm TCP to localized target brick
        _add_capsule(
            scn,
            p_tcp,
            p_sel,
            0.0013,
            np.array([1.00, 0.86, 0.18, 0.68], dtype=np.float32),
            eye_mat,
        )

        corners_w = twin.ledger.corners_world()
        upper = twin.ledger.upper_hemisphere_mask()
        inspected = twin.ledger.inspected_mask()
        for i in range(8):
            if scn.ngeom >= scn.maxgeom - 4:
                break
            if inspected[i]:
                rgba = np.array([0.15, 0.92, 0.35, 0.95], dtype=np.float32)
                rad = 0.0042
            elif twin.active_target_corner == i:
                rgba = np.array([1.00, 0.82, 0.10, 0.98], dtype=np.float32)
                rad = 0.0052
            elif upper[i]:
                rgba = np.array([0.20, 0.78, 0.98, 0.90], dtype=np.float32)
                rad = 0.0036
            else:
                rgba = np.array([0.55, 0.58, 0.62, 0.65], dtype=np.float32)
                rad = 0.0030
            _add_sphere(scn, corners_w[i], rad, rgba, eye_mat)

        if twin.active_target_corner is not None:
            T_wc = twin.rig.wrist.world_transform(T_b3)
            p_from = T_wc[:3, 3].astype(np.float64)
            p_to = corners_w[int(twin.active_target_corner)].astype(np.float64)
            _add_capsule(
                scn,
                p_from,
                p_to,
                0.0014,
                np.array([1.0, 0.85, 0.15, 0.75], dtype=np.float32),
                eye_mat,
            )

    # 6. Real2Sim perceived bricks (3D OBB wireframes, 6D pose triads, 8 corners & 3D point clouds)
    if twin.real2sim_state is not None:
        for b_idx, rb in enumerate(getattr(twin.real2sim_state, "bricks", ())):
            if scn.ngeom >= scn.maxgeom - 32:
                break
            disp_rgba = (
                np.asarray(rb.display_rgba(), dtype=np.float32)
                if hasattr(rb, "display_rgba")
                else np.array([0.25, 0.92, 0.98, 0.90], dtype=np.float32)
            )
            wire_rgba = np.clip(
                np.array(
                    [
                        disp_rgba[0] * 0.8 + 0.2,
                        disp_rgba[1] * 0.8 + 0.2,
                        disp_rgba[2] * 0.8 + 0.2,
                        0.92,
                    ],
                    dtype=np.float32,
                ),
                0.0,
                1.0,
            )
            corners_rb = _draw_obb_wireframe_3d(
                scn,
                rb.T_world_block,
                rb.spec.dims_m,
                rgba=wire_rgba,
                eye_mat=eye_mat,
                radius_m=0.0012 if b_idx == 0 else 0.0010,
                top_only=False,
                draw_axes=True,
            )
            for c_pt in corners_rb:
                if scn.ngeom >= scn.maxgeom - 16:
                    break
                _add_sphere(scn, c_pt, 0.0026, wire_rgba, eye_mat)
            c_pts = getattr(rb, "cluster_pts_world", None)
            if c_pts is not None and len(c_pts) > 0:
                stride_rb = max(1, len(c_pts) // 14)
                pt_col = np.array(
                    [disp_rgba[0], disp_rgba[1], disp_rgba[2], 0.72], dtype=np.float32
                )
                for pt in c_pts[::stride_rb]:
                    if scn.ngeom >= scn.maxgeom - 16:
                        break
                    _add_sphere(scn, pt, 0.0020, pt_col, eye_mat)

    # 7. Real2Sim ignored distractors (human arms, wires, trash)
    if twin.real2sim_state is not None and twin.show_real2sim_ignored_overlay:
        cat_rgba = {
            "human_arm": np.array([1.00, 0.32, 0.22, 0.42], dtype=np.float32),
            "wire": np.array([0.95, 0.65, 0.15, 0.42], dtype=np.float32),
            "trash": np.array([0.75, 0.30, 0.85, 0.38], dtype=np.float32),
        }
        for rc in getattr(twin.real2sim_state, "rejected_clusters", ()):
            if rc.category not in cat_rgba or scn.ngeom >= scn.maxgeom - 16:
                continue
            c_rgba = cat_rgba[rc.category]
            for pt in rc.sample_pts_world[::3]:
                if scn.ngeom >= scn.maxgeom - 8:
                    break
                _add_sphere(scn, pt, 0.0045, c_rgba, eye_mat)


def _project_cuboid_2d(
    cam: CameraModel,
    T_world_block: np.ndarray,
    dims_m: tuple[float, float, float],
    T_world_body3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 8 corners of a 3D cuboid into camera pixel coordinates `(u, v)` and depth `z`."""
    T = np.asarray(T_world_block, dtype=np.float64)
    half = 0.5 * np.asarray(dims_m, dtype=np.float64)
    corners_w = ((CORNER_SIGNS * half[None, :]) @ T[:3, :3].T) + T[:3, 3]
    return cam.project_world_points(corners_w, T_world_body3=T_world_body3)


def _draw_projected_obb_on_rgb(
    canvas: np.ndarray,
    cam: CameraModel,
    T_world_block: np.ndarray,
    spec: BlockSpec,
    T_world_body3: np.ndarray,
    color_rgb: tuple[int, int, int],
    thickness: int = 2,
    label: str | None = None,
    draw_corner_ids: bool = False,
    draw_axes: bool = True,
) -> None:
    """Draw projected 12-edge 3D wireframe box and 6D coordinate triad on an RGB image."""
    uv, z = _project_cuboid_2d(cam, T_world_block, spec.dims_m, T_world_body3)
    if not np.all(z > 0.03):
        return
    pts_i = np.round(uv).astype(np.int32)
    h, w = canvas.shape[:2]
    for i0, i1 in _CUBOID_EDGES:
        p0 = (int(pts_i[i0, 0]), int(pts_i[i0, 1]))
        p1 = (int(pts_i[i1, 0]), int(pts_i[i1, 1]))
        if (-200 <= p0[0] <= w + 200) and (-200 <= p0[1] <= h + 200):
            cv2.line(canvas, p0, p1, color_rgb, thickness, cv2.LINE_AA)

    if draw_axes:
        T = np.asarray(T_world_block, dtype=np.float64)
        p_c = T[:3, 3]
        R_b = T[:3, :3]
        axis_len = 0.018
        axis_pts_w = np.vstack(
            [
                p_c,
                p_c + axis_len * R_b[:, 0],
                p_c + axis_len * R_b[:, 1],
                p_c + axis_len * R_b[:, 2],
            ]
        )
        uv_ax, z_ax = cam.project_world_points(axis_pts_w, T_world_body3=T_world_body3)
        if np.all(z_ax > 0.03):
            ax_i = np.round(uv_ax).astype(np.int32)
            c0 = (int(ax_i[0, 0]), int(ax_i[0, 1]))
            for ax_idx, ax_col in ((1, (255, 65, 65)), (2, (65, 245, 95)), (3, (70, 165, 255))):
                p_end = (int(ax_i[ax_idx, 0]), int(ax_i[ax_idx, 1]))
                cv2.line(canvas, c0, p_end, ax_col, max(1, thickness), cv2.LINE_AA)

    if draw_corner_ids:
        for idx in range(8):
            px, py = int(pts_i[idx, 0]), int(pts_i[idx, 1])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), 3, (255, 230, 40), -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    f"c{idx}",
                    (px + 4, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (255, 245, 160),
                    1,
                    cv2.LINE_AA,
                )

    if label:
        top_pt = pts_i[np.argmin(pts_i[:, 1])]
        lx = int(np.clip(top_pt[0] - 28, 6, w - 150))
        ly = int(np.clip(top_pt[1] - 6, 18, h - 10))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.rectangle(canvas, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 3), (12, 18, 32), -1)
        cv2.putText(
            canvas,
            label,
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color_rgb,
            1,
            cv2.LINE_AA,
        )


def _draw_projected_plate_ring_on_rgb(
    canvas: np.ndarray,
    cam: CameraModel,
    pos_world: np.ndarray,
    radius_m: float,
    T_world_body3: np.ndarray,
    color_rgb: tuple[int, int, int],
    label: str | None = None,
) -> None:
    """Project a 3D circular plate rim onto an RGB camera frame."""
    angles = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    rim_pts = np.column_stack(
        [
            float(pos_world[0]) + radius_m * np.cos(angles),
            float(pos_world[1]) + radius_m * np.sin(angles),
            np.full_like(angles, float(pos_world[2]) + 0.014),
        ]
    )
    uv, z = cam.project_world_points(rim_pts, T_world_body3=T_world_body3)
    if not np.all(z > 0.05):
        return
    pts_i = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        canvas, [pts_i], isClosed=True, color=color_rgb, thickness=1, lineType=cv2.LINE_AA
    )
    if label:
        h, w = canvas.shape[:2]
        bot_pt = pts_i.reshape(-1, 2)[np.argmax(pts_i[:, 0, 1])]
        lx = int(np.clip(bot_pt[0] - 42, 6, w - 140))
        ly = int(np.clip(bot_pt[1] + 15, 22, h - 8))
        cv2.putText(
            canvas,
            label,
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color_rgb,
            1,
            cv2.LINE_AA,
        )


def render_real2sim_hud_on_live_frame(
    rgb_frame: np.ndarray,
    cam: CameraModel,
    scene_state: Any,
    T_world_body3: np.ndarray | None = None,
    title: str = "LIVE CAMERA FEED + REAL2SIM 3D HUD OVERLAY",
    draw_point_cloud: bool = True,
) -> np.ndarray:
    """Render the simulation 3D HUD overlays directly onto an actual live camera RGB frame."""
    canvas = np.asarray(rgb_frame, dtype=np.uint8).copy()
    h, w = canvas.shape[:2]
    T_b3 = (
        np.eye(4, dtype=np.float64)
        if T_world_body3 is None
        else np.asarray(T_world_body3, dtype=np.float64)
    )

    # 1. Draw localized 3D KALAS plate rings & labels
    plate_colors = {
        "source": (80, 235, 230),
        "good": (95, 245, 135),
        "defective": (255, 125, 165),
    }
    plates = getattr(scene_state, "plates", {}) or {}
    bricks = getattr(scene_state, "bricks", ()) or ()
    for role, p_obj in plates.items():
        cnt = sum(1 for b in bricks if getattr(b, "support_region", "") == role)
        p_col = plate_colors.get(role, (180, 220, 255))
        _draw_projected_plate_ring_on_rgb(
            canvas,
            cam,
            np.asarray(p_obj.pos_world, dtype=np.float64),
            0.0795,
            T_b3,
            color_rgb=p_col,
            label=f"{role.upper()} ({cnt})",
        )

    # 2. Optionally project perceived 3D point cloud inliers onto the live camera feed
    if draw_point_cloud:
        for rb in bricks:
            c_pts = getattr(rb, "cluster_pts_world", None)
            if c_pts is None or len(c_pts) == 0:
                continue
            uv_pts, z_pts = cam.project_world_points(c_pts[::2], T_world_body3=T_b3)
            disp_rgba = rb.display_rgba() if hasattr(rb, "display_rgba") else rb.spec.rgba
            pt_col = (
                int(round(min(255.0, disp_rgba[0] * 255.0 + 35.0))),
                int(round(min(255.0, disp_rgba[1] * 255.0 + 35.0))),
                int(round(min(255.0, disp_rgba[2] * 255.0 + 35.0))),
            )
            for (u_f, v_f), z_f in zip(uv_pts, z_pts):
                if z_f > 0.05:
                    px, py = int(round(float(u_f))), int(round(float(v_f)))
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(canvas, (px, py), 1, pt_col, -1, cv2.LINE_AA)

    # 3. Draw 3D wireframe OBBs, 6D pose triads, corners, and labels for all perceived DUPLO bricks
    for rb in bricks:
        disp_rgba = rb.display_rgba() if hasattr(rb, "display_rgba") else rb.spec.rgba
        col_rgb = (
            int(round(min(255.0, disp_rgba[0] * 220.0 + 35.0))),
            int(round(min(255.0, disp_rgba[1] * 220.0 + 35.0))),
            int(round(min(255.0, disp_rgba[2] * 220.0 + 35.0))),
        )
        c_name = rb.color_name() if hasattr(rb, "color_name") else ""
        pose_short = (
            str(getattr(rb, "stable_pose_name", "upright"))
            .replace("wide_side_", "wide_")
            .replace("narrow_side_", "narrow_")
        )
        lbl = f"{c_name} {rb.spec.part_id} [{pose_short}]".strip()
        _draw_projected_obb_on_rgb(
            canvas,
            cam,
            rb.T_world_block,
            rb.spec,
            T_b3,
            color_rgb=col_rgb,
            thickness=2,
            label=lbl,
            draw_corner_ids=False,
            draw_axes=True,
        )

    # 4. Header banner with Real2Sim telemetry
    cv2.rectangle(canvas, (0, 0), (w, 48), (15, 23, 42), -1)
    cv2.putText(
        canvas,
        title,
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (95, 225, 255),
        1,
        cv2.LINE_AA,
    )
    lat = getattr(scene_state, "latency_ms", 0.0)
    fg_n = getattr(scene_state, "foreground_point_count", 0)
    cv2.putText(
        canvas,
        f"Bricks: {len(bricks)} | Plates: {len(plates)} | FG Cloud: {fg_n} pts | ICP+OBB Latency: {lat:.0f} ms",
        (10, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (215, 232, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _depth_to_rgb_pip(
    depth_m: np.ndarray | None,
    pip_wh: tuple[int, int] = (156, 116),
    z_min: float = 0.10,
    z_max: float = 0.85,
) -> np.ndarray | None:
    """Convert metric float32 depth map into a colorized TURBO RGB picture-in-picture inset."""
    if depth_m is None:
        return None
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.04) & (d < 1.60)
    norm = np.zeros_like(d, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(d, z_min, z_max)
        norm_f = 255.0 * (1.0 - (clipped - z_min) / max(z_max - z_min, 1e-5))
        norm = np.where(valid, norm_f, 0.0).astype(np.uint8)
    bgr_cmap = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    rgb_cmap = cv2.cvtColor(bgr_cmap, cv2.COLOR_BGR2RGB)
    rgb_cmap[~valid] = (14, 18, 28)
    pip = cv2.resize(rgb_cmap, pip_wh, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(pip, (0, 0), (pip_wh[0] - 1, pip_wh[1] - 1), (120, 200, 255), 1)
    cv2.putText(
        pip,
        "Metric Depth",
        (6, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (245, 250, 255),
        1,
        cv2.LINE_AA,
    )
    return pip


def render_annotated_virtual_camera_frames(
    twin: Any,
) -> dict[str, np.ndarray]:
    """Render and annotate all 3 virtual cameras (`wrist`, `realsense_2`, `realsense_1`) on main thread."""
    from brickcell.block_inspector.twin import (
        DEFECTIVE_PILE_XY,
        GOOD_PILE_XY,
        PLATE_CAMERA_SCAN_POS,
        SOURCE_PILE_XY,
    )

    T_b3 = twin.body3_world_pose()
    q_deg = np.degrees(twin.current_arm_q())
    T_active = (
        (T_b3 @ twin._T_body3_block_held)
        if not twin.hand_empty
        else (twin.ledger.T_world_block if twin.ledger.localized else None)
    )

    # --- 1. WRIST CAMERA (Arm Eye-in-Hand) ---
    rgb_w, _ = twin.render_rgbd("wrist")
    w_img = rgb_w.copy()
    wh, ww = w_img.shape[:2]
    cx_w, cy_w = ww // 2, wh // 2
    cv2.line(w_img, (cx_w - 16, cy_w), (cx_w + 16, cy_w), (255, 215, 40), 1, cv2.LINE_AA)
    cv2.line(w_img, (cx_w, cy_w - 16), (cx_w, cy_w + 16), (255, 215, 40), 1, cv2.LINE_AA)
    cv2.circle(w_img, (cx_w, cy_w), 24, (255, 215, 40), 1, cv2.LINE_AA)

    if T_active is not None:
        _draw_projected_obb_on_rgb(
            w_img,
            twin.rig.wrist,
            T_active,
            twin.block_spec,
            T_b3,
            color_rgb=(50, 240, 120) if not twin.anomaly_flagged else (255, 70, 70),
            thickness=2,
            label=f"{twin.block_spec.part_id} ({'HELD' if not twin.hand_empty else 'TARGET'})",
            draw_corner_ids=True,
        )

    cv2.rectangle(w_img, (0, 0), (ww, 46), (15, 23, 42), -1)
    cv2.putText(
        w_img,
        "VIRTUAL CAM 1: ARM WRIST / GRIPPER (25-deg Eye-in-Hand)",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 220, 65),
        1,
        cv2.LINE_AA,
    )
    q_txt = " ".join(f"{v:+.0f}d" for v in q_deg)
    cv2.putText(
        w_img,
        f"q=[{q_txt}]  Grip:{twin.gripper_pct * 100:.0f}%  {twin.status_banner[:48]}",
        (10, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (210, 230, 255),
        1,
        cv2.LINE_AA,
    )

    # --- 2. LOW PLATE CAMERA (realsense_2 / side) ---
    rgb_p, dep_p = twin.render_rgbd("realsense_2")
    p_img = rgb_p.copy()
    ph, pw = p_img.shape[:2]
    plate_cam = twin.rig.realsense_2

    # Project PLATE_CAMERA_SCAN_POS inspection ROI box
    uv_scan, z_scan = plate_cam.project_world_points(PLATE_CAMERA_SCAN_POS[None, :], T_b3)
    if z_scan[0] > 0.05:
        u_s, v_s = int(round(float(uv_scan[0, 0]))), int(round(float(uv_scan[0, 1])))
        roi_col = (
            (255, 65, 65)
            if twin.anomaly_flagged
            else ((50, 245, 130) if twin.plate_scan_active else (90, 210, 250))
        )
        cv2.rectangle(p_img, (u_s - 48, v_s - 48), (u_s + 48, v_s + 48), roi_col, 2)
        cv2.putText(
            p_img,
            "96x96 CCW Spin ROI",
            (max(6, u_s - 48), max(60, v_s - 54)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            roi_col,
            1,
            cv2.LINE_AA,
        )

    if T_active is not None:
        _draw_projected_obb_on_rgb(
            p_img,
            plate_cam,
            T_active,
            twin.block_spec,
            T_b3,
            color_rgb=(255, 75, 75) if twin.anomaly_flagged else (60, 250, 140),
            thickness=2,
            label=f"{twin.block_spec.part_id}",
            draw_corner_ids=False,
        )

    pip_p = _depth_to_rgb_pip(dep_p, pip_wh=(152, 114))
    if pip_p is not None:
        p_img[ph - 122 : ph - 8, pw - 160 : pw - 8] = pip_p

    if twin.plate_scan_keyframes:
        kf = cv2.resize(twin.plate_scan_keyframes[-1], (96, 96), interpolation=cv2.INTER_NEAREST)
        p_img[ph - 104 : ph - 8, 8:104] = kf
        cv2.rectangle(
            p_img,
            (8, ph - 104),
            (104, ph - 8),
            (255, 75, 75) if twin.anomaly_flagged else (55, 240, 125),
            2,
        )

    cv2.rectangle(p_img, (0, 0), (pw, 50), (15, 23, 42), -1)
    verdict_txt = (
        "ANOMALY FLAGGED -> DEFECTIVE"
        if twin.anomaly_flagged
        else ("SCANNING JOINT-5 CCW" if twin.plate_scan_active else "CLEAN -> GOOD")
    )
    verdict_col = (255, 85, 85) if twin.anomaly_flagged else (80, 245, 150)
    dist_str = (
        f"raw_dist={twin.max_plate_anomaly_distance:.4f}"
        if twin.max_plate_anomaly_distance is not None
        else "raw_dist=--"
    )
    cv2.putText(
        p_img,
        f"VIRTUAL CAM 2: LOW PLATE D415 (Side CCW Spin) | {verdict_txt}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        verdict_col,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        p_img,
        f"Valid Spin Views: {twin.plate_scan_valid_frames} | {dist_str} | J5={q_deg[4]:+.1f} deg",
        (10, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (215, 232, 255),
        1,
        cv2.LINE_AA,
    )

    # --- 3. OVERHEAD CAMERA (realsense_1 / overhead) ---
    rgb_o, dep_o = twin.render_rgbd("realsense_1")
    o_img = rgb_o.copy()
    oh, ow = o_img.shape[:2]
    over_cam = twin.rig.realsense_1

    # Draw Real2Sim perceived bricks & 6D poses on Overhead D415 and Low Plate D415
    if getattr(twin, "real2sim_state", None) is not None:
        for rb in getattr(twin.real2sim_state, "bricks", ()):
            disp_rgba = rb.display_rgba() if hasattr(rb, "display_rgba") else rb.spec.rgba
            col_rgb = (
                int(round(min(255.0, disp_rgba[0] * 220.0 + 35.0))),
                int(round(min(255.0, disp_rgba[1] * 220.0 + 35.0))),
                int(round(min(255.0, disp_rgba[2] * 220.0 + 35.0))),
            )
            c_name = rb.color_name() if hasattr(rb, "color_name") else ""
            pose_short = (
                str(getattr(rb, "stable_pose_name", "upright"))
                .replace("wide_side_", "wide_")
                .replace("narrow_side_", "narrow_")
            )
            lbl = f"{c_name} {rb.spec.part_id} [{pose_short}]".strip()
            _draw_projected_obb_on_rgb(
                o_img,
                over_cam,
                rb.T_world_block,
                rb.spec,
                T_b3,
                color_rgb=col_rgb,
                thickness=2,
                label=lbl,
                draw_corner_ids=False,
                draw_axes=True,
            )
            _draw_projected_obb_on_rgb(
                p_img,
                plate_cam,
                rb.T_world_block,
                rb.spec,
                T_b3,
                color_rgb=col_rgb,
                thickness=2,
                label=lbl,
                draw_corner_ids=False,
                draw_axes=True,
            )

    # Draw perceived cluster candidate OBBs in source plate
    cands = getattr(twin, "_perceived_cluster_candidates", None) or []
    for cand in cands:
        _draw_projected_obb_on_rgb(
            o_img,
            over_cam,
            cand.localized.T_world_block,
            cand.block_spec,
            T_b3,
            color_rgb=(70, 225, 255),
            thickness=1,
            label=cand.block_spec.part_id,
            draw_corner_ids=False,
        )

    if T_active is not None:
        _draw_projected_obb_on_rgb(
            o_img,
            over_cam,
            T_active,
            twin.block_spec,
            T_b3,
            color_rgb=(255, 220, 45),
            thickness=2,
            label=f"ACTIVE:{twin.block_spec.part_id}",
            draw_corner_ids=False,
        )

    # Project 3 KALAS plate labels
    for pname, pxy, pcol in (
        (f"SOURCE ({twin.remaining_source_blocks})", SOURCE_PILE_XY, (80, 235, 230)),
        (f"GOOD ({twin.sorted_counts['good']})", GOOD_PILE_XY, (95, 245, 135)),
        (f"DEFECTIVE ({twin.sorted_counts['defective']})", DEFECTIVE_PILE_XY, (255, 125, 165)),
    ):
        uv_pl, z_pl = over_cam.project_world_points(
            np.array([[pxy[0], pxy[1], 0.015]], dtype=np.float64), T_b3
        )
        if z_pl[0] > 0.05:
            ux = int(np.clip(round(float(uv_pl[0, 0])) - 42, 6, ow - 130))
            uy = int(np.clip(round(float(uv_pl[0, 1])) + 52, 60, oh - 10))
            cv2.putText(
                o_img,
                pname,
                (ux, uy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                pcol,
                1,
                cv2.LINE_AA,
            )

    pip_o = _depth_to_rgb_pip(dep_o, pip_wh=(152, 114))
    if pip_o is not None:
        o_img[oh - 122 : oh - 8, ow - 160 : ow - 8] = pip_o

    cv2.rectangle(o_img, (0, 0), (ow, 50), (15, 23, 42), -1)
    cv2.putText(
        o_img,
        "VIRTUAL CAM 3: OVERHEAD D415 (Top 3-Plate Scene & RGB-D ICP)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (110, 205, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        o_img,
        f"Source Left: {twin.remaining_source_blocks} | Sorted Good: {twin.sorted_counts['good']} | Defective: {twin.sorted_counts['defective']}",
        (10, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (215, 232, 255),
        1,
        cv2.LINE_AA,
    )

    return {
        "wrist": w_img,
        "realsense_2": p_img,
        "realsense_1": o_img,
    }


def _force_win32_hwnd_visible(hwnd: int) -> None:
    """Ensure a Win32 window handle is shown and not suppressed by STARTUPINFO SW_HIDE."""
    if sys.platform != "win32" or not hwnd:
        return
    try:
        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.ShowWindow(hwnd, 5)  # SW_SHOW
        user32.BringWindowToTop(hwnd)
    except Exception:
        pass


class VirtualCameraWindowsManager:
    """Opens 3 live native desktop windows displaying the virtual cameras alongside MuJoCo Studio."""

    def __init__(self, display_width: int = 480) -> None:
        self.display_width = int(display_width)
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] = {}
        self._running = False
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._gui_thread_main,
            name="BrickcellVirtualCameraWindows",
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait(timeout=3.0)

    def push_frames(self, frames: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._frames = {k: v.copy() for k, v in frames.items()}

    def update_from_twin(self, twin: Any) -> None:
        if not self._running:
            return
        frames = render_annotated_virtual_camera_frames(twin)
        self.push_frames(frames)

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def _gui_thread_main(self) -> None:
        try:
            import tkinter as tk
        except Exception:
            self._running = False
            self._ready_event.set()
            return

        try:
            root = tk.Tk()
            scr_w = max(1280, int(root.winfo_screenwidth()))
            scr_h = max(800, int(root.winfo_screenheight()))
            win_w = min(self.display_width, max(380, int(scr_w * 0.31)))
            x_col = max(40, scr_w - win_w - 24)

            h_wrist = int(round(win_w * 360.0 / 640.0))
            h_d415 = int(round(win_w * 480.0 / 640.0))
            total_h = h_wrist + 2 * h_d415 + 96
            if total_h > scr_h - 40:
                scale = (scr_h - 96) / float(h_wrist + 2 * h_d415)
                win_w = max(340, int(round(win_w * scale)))
                h_wrist = int(round(win_w * 360.0 / 640.0))
                h_d415 = int(round(win_w * 480.0 / 640.0))
                x_col = max(40, scr_w - win_w - 20)

            y1 = 18
            y2 = y1 + h_wrist + 34
            y3 = y2 + h_d415 + 34

            # Window 1: Arm Wrist / Gripper Camera (root Tk window)
            root.title("Virtual Camera 1 — Arm Wrist / Gripper (Eye-in-Hand)")
            root.configure(bg="#0b1120")
            root.geometry(f"{win_w}x{h_wrist}+{x_col}+{y1}")
            lbl_wrist = tk.Label(root, bg="#0b1120", bd=0, highlightthickness=0)
            lbl_wrist.pack(fill=tk.BOTH, expand=True)

            # Window 2: Low Plate D415 Camera (Toplevel)
            win_plate = tk.Toplevel(root)
            win_plate.title("Virtual Camera 2 — Low Plate D415 (CCW Spin & Anomaly)")
            win_plate.configure(bg="#0b1120")
            win_plate.geometry(f"{win_w}x{h_d415}+{x_col}+{y2}")
            lbl_plate = tk.Label(win_plate, bg="#0b1120", bd=0, highlightthickness=0)
            lbl_plate.pack(fill=tk.BOTH, expand=True)

            # Window 3: Overhead D415 Camera (Toplevel)
            win_over = tk.Toplevel(root)
            win_over.title("Virtual Camera 3 — Overhead D415 (Top 3-Plate Scene)")
            win_over.configure(bg="#0b1120")
            win_over.geometry(f"{win_w}x{h_d415}+{x_col}+{y3}")
            lbl_over = tk.Label(win_over, bg="#0b1120", bd=0, highlightthickness=0)
            lbl_over.pack(fill=tk.BOTH, expand=True)

            root.update_idletasks()
            root.update()
            for w_obj in (root, win_plate, win_over):
                try:
                    w_obj.deiconify()
                    w_obj.lift()
                    _force_win32_hwnd_visible(int(w_obj.winfo_id()))
                except Exception:
                    pass

            photo_refs: dict[str, Any] = {}
            self._ready_event.set()

            def _refresh() -> None:
                if not self._running:
                    try:
                        root.destroy()
                    except Exception:
                        pass
                    return
                with self._lock:
                    snap = dict(self._frames)
                for key, lbl, target_wh in (
                    ("wrist", lbl_wrist, (win_w, h_wrist)),
                    ("realsense_2", lbl_plate, (win_w, h_d415)),
                    ("realsense_1", lbl_over, (win_w, h_d415)),
                ):
                    arr = snap.get(key)
                    if arr is not None and arr.size > 0:
                        resized = cv2.resize(arr, target_wh, interpolation=cv2.INTER_AREA)
                        img_tk = ImageTk.PhotoImage(Image.fromarray(resized))
                        lbl.configure(image=img_tk)
                        photo_refs[key] = img_tk
                root.after(45, _refresh)

            root.after(20, _refresh)
            root.mainloop()
        except Exception:
            self._running = False
            self._ready_event.set()


def run_live_studio_session(
    num_blocks: int = 8,
    cluster_seed: int = 42,
    hold_seconds: float = 180.0,
    loop_episodes: bool = True,
    hud_port: int = 8095,
) -> int:
    """Launch an isolated MuJoCo Studio + 3 Virtual Camera Windows session without touching training files."""
    from brickcell.block_inspector.camera_adapters import LiveInspectionHudServer
    from brickcell.block_inspector.cogman import CogManInspectionAgent
    from brickcell.block_inspector.openvino_anomaly import ensure_default_plate_openvino_detector
    from brickcell.block_inspector.twin import SO101InspectionTwin

    detector = ensure_default_plate_openvino_detector()
    assert detector.threshold is not None
    thr_val = float(detector.threshold.value)

    isolated_scene_dir = REPO_ROOT / "artifacts" / "block_inspector" / "mjcf_studio_live"
    isolated_reports_dir = REPO_ROOT / "artifacts" / "block_inspector" / "reports_studio_live"

    twin = SO101InspectionTwin(
        num_blocks=num_blocks,
        cluster_seed=cluster_seed,
        scene_dir=isolated_scene_dir,
        show_viewer=True,
        show_camera_windows=True,
        realtime_pacing=True,
        anomaly_scorer=lambda roi: detector.predict(roi).score,
        anomaly_threshold=thr_val,
    )
    hud = LiveInspectionHudServer(port=hud_port)
    try:
        hud_url = hud.start()
        print(
            f"[MuJoCo Studio Live] 3D Scene + 3 Virtual Camera Windows active | Web HUD: {hud_url}",
            flush=True,
        )
    except OSError:
        hud = None  # type: ignore[assignment]

    step_box = [0]

    def _on_step(tw: SO101InspectionTwin) -> None:
        if hud is not None and step_box[0] % 8 == 0:
            hud.update_from_twin(tw, step_counter=step_box[0])
        step_box[0] += 1

    twin.frame_listeners.append(_on_step)

    try:
        ep = 0
        while twin.is_viewer_running():
            ep += 1
            if ep > 1:
                # Respawn blocks back in the aquamarine source plate for continuous live viewing
                twin.reset_episode_state()
                twin.reset_arm_to_home()
                for b in twin.spawned_blocks:
                    j_id = int(twin.model.joint(f"{b.block_id}_joint").id)
                    q_adr = int(twin.model.jnt_qposadr[j_id])
                    v_adr = int(twin.model.jnt_dofadr[j_id])
                    bx, by, byaw = b.initial_xy_yaw
                    hz = float(b.spec.dims_m[2])
                    twin.data.qpos[q_adr : q_adr + 3] = [bx, by, 0.004 + 0.5 * hz]
                    twin.data.qpos[q_adr + 3 : q_adr + 7] = [
                        math.cos(0.5 * byaw),
                        0.0,
                        0.0,
                        math.sin(0.5 * byaw),
                    ]
                    twin.data.qvel[v_adr : v_adr + 6] = 0.0
                for _ in range(35):
                    mujoco.mj_step(twin.model, twin.data)
                twin._sync_viewer()

            agent = CogManInspectionAgent(executor=twin, ledger=twin.ledger)
            agent.run(
                instruction=f"pick and classify all {num_blocks} blocks from source plate",
                block_id="block_0",
                output_dir=isolated_reports_dir,
                mode="clear_source_plate" if num_blocks > 1 else "plate_spin_sort",
            )
            if not loop_episodes:
                break

        deadline = time.monotonic() + float(hold_seconds)
        while twin.is_viewer_running() and time.monotonic() < deadline:
            twin._sync_viewer()
            time.sleep(0.05)
        return 0
    finally:
        if hud is not None:
            hud.stop()
        twin.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch MuJoCo Studio 3D Scene + 3 Live Virtual Camera Windows"
    )
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hold-seconds", type=float, default=180.0)
    parser.add_argument("--once", action="store_true", help="Run a single episode then hold viewer")
    args = parser.parse_args(argv)
    return run_live_studio_session(
        num_blocks=args.num_blocks,
        cluster_seed=args.seed,
        hold_seconds=args.hold_seconds,
        loop_episodes=not args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
