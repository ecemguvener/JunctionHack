"""Offline refinement of the D415 camera poses (overhead and side) for the station twin.

Run: ``python -m brickcell.twin.calibrate --fixture DIR [--scene-config PATH] [--out DIR]``.
The tool reads the fixture (depth and colour), the present camera poses and, if given, a
colour image of the clear board for each camera. It writes report.json, proposal.yaml and
diagnostic images. ``--write`` puts the new overhead and side poses into the scene config.

Method
1. Table plane per camera. RANSAC and least squares on the depth points (not on the board)
   give the table plane in the camera frame. The pose correction makes this plane z = 0 in
   the world. This sets z, roll and pitch of each camera (3 DoF). x, y and yaw stay.
2. Board grid. The board is a checkerboard of orange and tan cells. The rows and the columns
   are straight lines, but the cells do not have the same size. The board plane (from depth)
   and the levelled pose give a top-down metric view of the board. In this view the lines of
   each family are almost parallel: a 1D peak search finds them. Each line is then fitted
   again on the edge pixels of the original image. Corners are line intersections. A ray from
   a corner pixel meets the board plane: this gives the 3D corner.
3. Relative pose. The side camera keeps its plane correction (z, roll, pitch). A 2D rigid fit
   on the table (x, y, yaw) moves the side corners onto the overhead corners. The grid index
   offset between the cameras is the one with the smallest residual; the uneven cells make
   it unique.
4. Absolute anchor. The overhead camera keeps the present x, y and yaw (they come from the
   SO-101 model fit). See the report for how well the data can check this.

Frames: OpenCV camera frame (+X right, +Y down, +Z forward). A pose is camera-to-world:
x_world = R x_cam + p. World = robot base frame of configs/sim-scene.yaml.
Lengths: metres in the code, millimetres in the report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from brickcell.simulation.scene_config import DEFAULT_PATH as SCENE_CONFIG_PATH
from brickcell.simulation.scene_config import load_scene_config
from brickcell.twin.state import load_fixture

REPO = Path(__file__).resolve().parents[3]
TWIN_CONFIG = REPO / "configs" / "twin.yaml"
CAMERAS = ("overhead", "side")
DEFAULTS: dict = {
    "max_range_m": 1.3,  # depth points farther than this are not used
    "table_threshold_m": 0.004,  # RANSAC inlier distance for the table plane
    "board_band_m": [-0.002, 0.007],  # board pixels: this far above the table plane
    "overhead_pivot_xy": [0.0, 0.0],  # world (x, y) that the overhead levelling keeps
    "object_z_m": [0.006, 0.12],  # object points (plates, bricks) for the side registration
    "object_box_m": [[0.0, 0.6], [-0.45, 0.25]],  # world x and y range of the object points
    "voxel_m": 0.002,
    "board_roi": {},  # {camera: [[u, v], ...]} board polygon in the image (optional)
}

# ------------------------------------------------------------------------------- geometry


def rot_axis_angle(axis, angle: float) -> np.ndarray:
    """Rotation matrix about an axis (Rodrigues)."""
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(angle) < 1e-15:
        return np.eye(3)
    return cv2.Rodrigues((axis / norm * angle).reshape(3, 1))[0]


def rotation_between(a, b) -> np.ndarray:
    """Smallest rotation that turns the vector a into the vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    axis = np.cross(a, b)
    return rot_axis_angle(axis, math.atan2(np.linalg.norm(axis), float(a @ b)))


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def angle_deg(a, b) -> float:
    """Angle between two rotation matrices in degrees."""
    r = np.asarray(a, float).T @ np.asarray(b, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def rays(pixels, fx, fy, cx, cy) -> np.ndarray:
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    return np.column_stack(
        ((pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels)))
    )


def depth_points(depth_m, intrinsics, mask=None, step: int = 1):
    """Camera-frame points and their pixels for the valid depth pixels (optional mask)."""
    valid = depth_m > 0
    if mask is not None:
        valid = valid & mask
    if step > 1:
        grid = np.zeros_like(valid)
        grid[::step, ::step] = True
        valid &= grid
    v, u = np.nonzero(valid)
    pixels = np.column_stack((u, v)).astype(float)
    return rays(pixels, *intrinsics) * depth_m[v, u][:, None], pixels


def project_cam(points_cam, fx, fy, cx, cy) -> np.ndarray:
    p = np.atleast_2d(points_cam)
    return np.column_stack((fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy))


@dataclass
class Pose:
    """Camera-to-world pose, OpenCV camera frame: x_world = rotation @ x_cam + position."""

    rotation: np.ndarray
    position: np.ndarray

    def to_world(self, points_cam) -> np.ndarray:
        return np.atleast_2d(points_cam) @ self.rotation.T + self.position

    def to_camera(self, points_world) -> np.ndarray:
        return (np.atleast_2d(points_world) - self.position) @ self.rotation

    def project(self, points_world, intrinsics) -> np.ndarray:
        return project_cam(self.to_camera(points_world), *intrinsics)

    def turned(self, yaw: float, shift_xy) -> Pose:
        """This pose after a turn about the world z axis and a shift on the table."""
        r = rot_z(yaw)
        shift = np.array([shift_xy[0], shift_xy[1], 0.0])
        return Pose(r @ self.rotation, r @ self.position + shift)

    def as_dict(self) -> dict:
        return {
            "pos": [round(float(v), 5) for v in self.position],
            "rotation_cv": [[round(float(v), 6) for v in row] for row in self.rotation],
        }


# --------------------------------------------------------------------------------- planes


@dataclass
class Plane:
    """Plane n . x + d = 0 in the camera frame. n points to the camera side (d > 0)."""

    normal: np.ndarray
    offset: float
    inliers: int = 0
    total: int = 0
    rms_mm: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def distance(self, points) -> np.ndarray:
        return np.asarray(points) @ self.normal + self.offset

    def intersect(self, pixels, intrinsics) -> np.ndarray:
        """Camera-frame points where the pixel rays meet the plane."""
        r = rays(pixels, *intrinsics)
        return r * (-self.offset / (r @ self.normal))[:, None]

    def summary(self) -> dict:
        return {"inliers": self.inliers, "points": self.total, "rms_mm": round(self.rms_mm, 2)}


def _plane_lsq(points) -> tuple[np.ndarray, float]:
    centroid = points.mean(0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    offset = -float(normal @ centroid)
    if offset < 0:  # the camera (the origin) is on the positive side
        normal, offset = -normal, -offset
    return normal, offset


def fit_plane(points, threshold: float = 0.004, iterations: int = 500, seed: int = 0) -> Plane:
    """RANSAC plane, then least squares on the inliers (three rounds)."""
    points = np.asarray(points, float)
    if len(points) < 3:
        raise ValueError("A plane fit needs at least 3 points.")
    rng = np.random.default_rng(seed)
    sample = points if len(points) <= 20000 else points[rng.choice(len(points), 20000, False)]
    best, best_count = None, -1
    for _ in range(iterations):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        count = int((np.abs((sample - a) @ normal) < threshold).sum())
        if count > best_count:
            best, best_count = (normal, -float(normal @ a)), count
    if best is None:
        raise ValueError("The plane fit found no plane (the points are on a line).")
    normal, offset = best
    for _ in range(3):
        keep = np.abs(points @ normal + offset) < threshold
        normal, offset = _plane_lsq(points[keep])
    keep = np.abs(points @ normal + offset) < threshold
    residual = points[keep] @ normal + offset
    return Plane(
        normal=normal,
        offset=offset,
        inliers=int(keep.sum()),
        total=len(points),
        rms_mm=float(np.sqrt(np.mean(residual**2)) * 1000),
        centroid=points[keep].mean(0),
    )


def plane_in_world(plane: Plane, pose: Pose) -> tuple[np.ndarray, float]:
    """World normal (pointing up) and offset (n . x + d = 0) of a camera-frame plane."""
    normal = pose.rotation @ plane.normal
    offset = plane.offset - float(normal @ pose.position)
    if normal[2] < 0:
        normal, offset = -normal, -offset
    return normal, offset


def plane_error(plane: Plane, pose: Pose, at=(0.0, 0.0)) -> dict:
    """Tilt (deg) of a camera plane against world z = 0, and its height (mm) at a point."""
    normal, offset = plane_in_world(plane, pose)
    tilt = math.degrees(math.acos(min(1.0, float(normal[2]))))
    height = -(offset + normal[0] * at[0] + normal[1] * at[1]) / normal[2]
    return {"tilt_deg": round(tilt, 3), "height_mm": round(height * 1000, 2)}


def level_pose(pose: Pose, plane: Plane, pivot=(0.0, 0.0), height: float = 0.0) -> Pose:
    """Correct a pose so that the camera-frame plane becomes z = height in the world.

    The correction is the smallest rotation about the point where the plane meets the
    vertical line through ``pivot`` (x, y), then a vertical shift. The x and y of that point
    do not change, and the yaw changes only a little.
    """
    normal, offset = plane_in_world(plane, pose)
    x, y = pivot
    anchor = np.array([x, y, -(offset + normal[0] * x + normal[1] * y) / normal[2]])
    turn = rotation_between(normal, [0.0, 0.0, 1.0])
    return Pose(turn @ pose.rotation, turn @ (pose.position - anchor) + np.array([x, y, height]))


# ----------------------------------------------------------------------------- rigid fits


def rigid_fit_3d(source, target):
    """R, t that make R source + t nearest to target (Kabsch, least squares)."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    cs, ct = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    fix = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ fix @ u.T
    return rotation, ct - rotation @ cs


def rigid_fit_2d(source, target) -> tuple[float, np.ndarray]:
    """Yaw and shift that make Rz(yaw) source + shift nearest to target (x, y only)."""
    source, target = np.asarray(source, float)[:, :2], np.asarray(target, float)[:, :2]
    cs, ct = source.mean(0), target.mean(0)
    a, b = source - cs, target - ct
    yaw = math.atan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])), float(np.sum(a * b)))
    return yaw, ct - rot_z(yaw)[:2, :2] @ cs


# ------------------------------------------------------------------------------- colours


def orange_channel(rgb) -> np.ndarray:
    """Lab a* (red-green) channel, float. Orange cells are high, tan cells are lower."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB)
    return cv2.GaussianBlur(lab[..., 1].astype(np.float32), (0, 0), 0.7)


def _otsu(values) -> float:
    values = np.asarray(values, float)
    lo, hi = np.percentile(values, [1, 99])
    scaled = np.clip((values - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(scaled.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(lo + t / 255 * (hi - lo))


def board_roi(a_star, roi_override=None, min_cells: int = 6, valid=None):
    """Board region (bool mask), the a* levels (tan, orange) and a typical cell size (px).

    The orange cells are the brightest a* regions. The board is the cluster of bright blobs
    with the most separate cells; the mask is its convex hull. ``roi_override`` (a pixel
    polygon) replaces the automatic mask.
    """
    h, w = a_star.shape
    if roi_override is not None:
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.asarray(roi_override, np.int32)], 1)
        mask = mask > 0
        if valid is not None:
            mask &= valid
        inside = a_star[mask]
        split = _otsu(inside)
        levels = (
            float(np.median(inside[inside <= split])),
            float(np.median(inside[inside > split])),
        )
        return mask, levels, 15.0
    high, mid = np.percentile(a_star, [99.5, 50])
    bright = a_star > 0.5 * (high + mid)
    cells = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(cells, connectivity=4)
    keep = [i for i in range(1, count) if 20 <= stats[i, cv2.CC_STAT_AREA] <= 0.02 * h * w]
    if not keep:
        raise ValueError("No orange cells found.")
    cell = float(np.median(np.sqrt([stats[i, cv2.CC_STAT_AREA] for i in keep])))
    gap = int(max(5, cell * 0.8))
    joined = cv2.dilate(np.isin(labels, keep).astype(np.uint8), np.ones((gap, gap), np.uint8))
    _, labels2 = cv2.connectedComponents(joined, connectivity=8)
    clusters: dict[int, list[int]] = {}
    for i in keep:
        u, v = np.round(centroids[i]).astype(int)
        clusters.setdefault(int(labels2[v, u]), []).append(i)
    best = max(clusters.values(), key=len)
    if len(best) < min_cells:
        raise ValueError(f"No board found: the best cluster has {len(best)} orange cells.")
    mask = np.zeros((h, w), np.uint8)
    points = np.column_stack(np.nonzero(np.isin(labels, best)))[:, ::-1]
    cv2.fillPoly(mask, [cv2.convexHull(points.astype(np.int32))], 1)
    if valid is not None:
        mask &= valid.astype(np.uint8)
    inside = a_star[mask > 0]
    split = _otsu(inside)
    levels = (float(np.median(inside[inside <= split])), float(np.median(inside[inside > split])))
    return mask > 0, levels, cell


# ------------------------------------------------------------------------- board detection


@dataclass
class GridLine:
    """One grid line. ``theta``/``offset``: world line x cos(t) + y sin(t) = offset.

    ``image``: unit normal (a, b) and c of the image line a u + b v = c.
    ``extent``: along-line range (m) on the board where the line is visible.
    """

    family: int
    theta: float
    offset: float
    extent: tuple[float, float]
    image: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position: float = 0.0  # m, where the line cuts the transversal through the board centre
    edge_pixels: int = 0
    image_rms_px: float = float("nan")


@dataclass
class BoardDetection:
    camera: str
    pixels: np.ndarray  # (N, 2) corner pixels (line intersections in the image)
    index: np.ndarray  # (N, 2) int, line numbers (family 0, family 1)
    lines: list[list[GridLine]]  # per family, sorted by offset
    levels: tuple[float, float]  # a* of tan and orange
    roi: np.ndarray  # board mask in the image
    board_height_m: float = 0.0
    rectified: np.ndarray | None = None  # BGR top-down view (diagnostic)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "corners": int(len(self.pixels)),
            "lines": [len(f) for f in self.lines],
            "line_image_rms_px": [
                round(float(np.nanmedian([g.image_rms_px for g in f])), 3) if f else None
                for f in self.lines
            ],
            "a_star_levels": [round(v, 1) for v in self.levels],
            "notes": self.notes,
        }


class Rectifier:
    """Top-down view of the board plane z = height in world x-y, ``res`` m per pixel.

    View column i, row j is the world point (x0 + i res, y0 + j res, height).
    """

    def __init__(self, pose: Pose, intrinsics, height: float, bounds, res: float):
        self.pose, self.intrinsics, self.height, self.res = pose, intrinsics, height, res
        (x0, y0), (x1, y1) = bounds
        self.x0, self.y0 = float(x0), float(y0)
        self.shape = (int(math.ceil((y1 - y0) / res)) + 1, int(math.ceil((x1 - x0) / res)) + 1)
        jj, ii = np.mgrid[0 : self.shape[0], 0 : self.shape[1]]
        world = np.column_stack(
            (x0 + ii.ravel() * res, y0 + jj.ravel() * res, np.full(ii.size, height))
        )
        cam = pose.to_camera(world)
        uv = project_cam(cam, *intrinsics)
        uv[cam[:, 2] <= 0] = -1
        self.map_u = uv[:, 0].reshape(self.shape).astype(np.float32)
        self.map_v = uv[:, 1].reshape(self.shape).astype(np.float32)

    def warp(self, image, interpolation=cv2.INTER_LINEAR, border=0.0) -> np.ndarray:
        return cv2.remap(
            image,
            self.map_u,
            self.map_v,
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    def to_view(self, xy) -> np.ndarray:
        return (np.atleast_2d(xy) - [self.x0, self.y0]) / self.res


def _line_points(theta: float, offset: float, s) -> np.ndarray:
    """Points (x, y) on the line x cos(t) + y sin(t) = offset at along-line values s."""
    n = np.array([math.cos(theta), math.sin(theta)])
    d = np.array([-n[1], n[0]])
    return n * offset + np.outer(np.atleast_1d(np.asarray(s, float)), d)


def _board_angles(a_rect, mask) -> tuple[float, float]:
    """The two main gradient (line normal) angles in the top-down view, radians in [0, pi)."""
    gx = cv2.Sobel(a_rect, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a_rect, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)[mask]
    ang = np.mod(np.arctan2(gy, gx)[mask], np.pi)
    bins = 360
    hist, _ = np.histogram(ang, bins=bins, range=(0, np.pi), weights=mag)
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    hist = np.convolve(np.r_[hist[-6:], hist, hist[:6]], kernel, mode="same")[6:-6]
    first = int(np.argmax(hist))
    gap = np.abs(((np.arange(bins) - first + bins // 2) % bins) - bins // 2)
    second = int(np.argmax(np.where(gap > bins // 6, hist, 0)))
    centres = (np.arange(bins) + 0.5) * np.pi / bins

    def refine(k):  # weighted mean angle of the strong gradients near the peak
        near = np.abs(((ang - centres[k] + np.pi / 2) % np.pi) - np.pi / 2) < math.radians(4)
        doubled = np.angle(np.sum(mag[near] * np.exp(2j * ang[near]))) / 2
        return float(np.mod(doubled, np.pi))

    return refine(first), refine(second)


def _classify(values, levels) -> np.ndarray:
    """+1 orange, -1 tan, 0 not clear (NaN gives 0)."""
    tan, orange = levels
    mid, margin = 0.5 * (tan + orange), 0.15 * (orange - tan)
    out = np.zeros(len(values), int)
    out[values > mid + margin] = 1
    out[values < mid - margin] = -1
    return out


def _sample_float(image, points) -> np.ndarray:
    """Bilinear samples of a float image at (col, row) points; outside gives NaN."""
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    count = len(pts)
    width = 4096
    padded = np.full((-(-count // width) * width, 2), -10.0, np.float32)
    padded[:count] = pts
    grid = padded.reshape(-1, width, 2)
    out = cv2.remap(
        image.astype(np.float32),
        grid[..., 0].copy(),
        grid[..., 1].copy(),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    return out.reshape(-1)[:count]


def _vanishing_point(points, normals, weights, tol_deg=2.0, iterations=800, seed=0):
    """Homogeneous point where most local edge lines meet (RANSAC, then least squares).

    ``points`` (N, 2) are edge pixels and ``normals`` (N, 2) their unit gradient directions.
    The local line at a pixel is perpendicular to its gradient.
    """
    lines = np.column_stack((normals, -(normals * points).sum(1)))  # a u + b v + c = 0
    rng = np.random.default_rng(seed)
    sin_tol = math.sin(math.radians(tol_deg))

    def consistent(vp):
        if abs(vp[2]) < 1e-9:
            to_vp = np.broadcast_to(vp[:2], points.shape)
        else:
            to_vp = vp[:2] / vp[2] - points
        norm = np.maximum(np.linalg.norm(to_vp, axis=1), 1e-9)
        return np.abs((to_vp * normals).sum(1)) / norm < sin_tol  # along the local line

    best, best_score = None, -1.0
    for _ in range(iterations):
        i, j = rng.choice(len(points), 2, replace=False)
        vp = np.cross(lines[i], lines[j])
        if np.linalg.norm(vp) < 1e-12:
            continue
        score = float(weights[consistent(vp)].sum())
        if score > best_score:
            best, best_score = vp, score
    if best is None:
        raise ValueError("No vanishing point.")
    for _ in range(3):
        keep = consistent(best)
        w = np.sqrt(weights[keep])[:, None]
        _, _, vt = np.linalg.svd(lines[keep] * w, full_matrices=False)
        best = vt[-1]
    return best / np.linalg.norm(best), float(consistent(best).mean())


def _pencil_lines(vp, gx, gy, mask, direction, min_gap_px, step=0.25):
    """Lines through ``vp`` with strong edges across them, as (unit normal, c) in the view.

    The scan moves a point along a transversal through the mask centre (along ``direction``,
    the mean line normal) and scores the line from ``vp`` through that point.
    """
    rows, cols = np.nonzero(mask)
    centre = np.array([cols.mean(), rows.mean()])
    e = np.asarray(direction, float) / np.linalg.norm(direction)
    reach = float(np.abs((np.column_stack((cols, rows)) - centre) @ e).max()) + 2
    ts = np.arange(-reach, reach, step)
    h, w = mask.shape
    span = np.arange(-(h + w), h + w, 1.0)
    maskf = mask.astype(np.float32)
    scores = np.zeros(len(ts))
    normals = np.zeros((len(ts), 2))
    offsets = np.zeros(len(ts))
    for k0 in range(0, len(ts), 64):
        t = ts[k0 : k0 + 64]
        q = centre + np.outer(t, e)
        lines = np.cross(np.broadcast_to(vp, (len(t), 3)), np.column_stack((q, np.ones(len(t)))))
        n = lines[:, :2] / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
        n *= np.sign((n @ e))[:, None] + (n @ e == 0)[:, None]
        d = np.column_stack((-n[:, 1], n[:, 0]))
        pts = q[:, None, :] + span[None, :, None] * d[:, None, :]
        flat = pts.reshape(-1, 2)
        inside = _sample_float(maskf, flat) > 0.5
        g = _sample_float(gx, flat) * np.repeat(n[:, 0], len(span)) + _sample_float(
            gy, flat
        ) * np.repeat(n[:, 1], len(span))
        g = np.where(inside & np.isfinite(g), np.abs(g), 0.0).reshape(len(t), -1)
        scores[k0 : k0 + 64] = g.sum(1)
        normals[k0 : k0 + 64] = n
        offsets[k0 : k0 + 64] = (n * q).sum(1)
    scores = np.convolve(scores, np.ones(3) / 3, mode="same")
    width = max(1, int(round(min_gap_px / step)))
    floor = 0.1 * float(np.percentile(scores, 99))
    local_max = scores == _running_max(scores, width)
    peaks = [i for i in np.nonzero(local_max & (scores >= floor))[0]]
    return [(normals[i], float(offsets[i]), float(ts[i])) for i in peaks]


def _running_max(values, half: int) -> np.ndarray:
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return windows.max(axis=1)


def _check_line(a_rect, res, theta, offset, levels, min_len, step=0.5e-3):
    """Along-line extent (m, view frame) if this is an inner grid line, else None.

    An inner grid line has orange on one side and tan on the other side, and each side
    changes between orange and tan along the line. The tape border fails this check.
    """
    h, w = a_rect.shape
    span = (h + w) * res
    s = np.arange(-span, span, step)
    n = np.array([math.cos(theta), math.sin(theta)])
    base = _line_points(theta, offset, s)
    side = 4.0e-3
    left, right = (
        _classify(_sample_float(a_rect, (base + sign * side * n) / res), levels) for sign in (1, -1)
    )
    differ = (left * right) == -1
    idx = np.nonzero(differ)[0]
    if len(idx) < min_len / step:
        return None
    runs, start = [], idx[0]
    gap = int(15e-3 / step)  # allow 15 mm gaps (the brick, a blurred corner)
    for a, b in zip(idx[:-1], idx[1:], strict=False):
        if b - a > gap:
            runs.append((start, a))
            start = b
    runs.append((start, idx[-1]))
    # A line can be visible in several pieces (objects on the board). Keep pieces >= 5 mm.
    runs = [r for r in runs if (r[1] - r[0]) * step >= 5e-3]
    if not runs or sum(r[1] - r[0] for r in runs) * step < min_len:
        return None
    first, last = runs[0][0], runs[-1][1]
    keep = np.zeros(len(s), bool)
    for a, b in runs:
        keep[a : b + 1] = True
    lseg = left[keep & differ]
    flips = int(np.count_nonzero(np.diff(lseg)))
    orange_left = float((lseg == 1).mean())
    clear = keep & (left != 0) & (right != 0)
    if differ[clear].mean() < 0.6 or flips < 1 or not 0.1 < orange_left < 0.9:
        return None
    return float(s[first]), float(s[last])


def _refine_in_image(end_a, end_b, edges, gx, gy):
    """Total-least-squares line on the edge pixels near the segment a-b.

    Returns (unit normal, c, pixel count, rms px) or None.
    """
    a, b = np.asarray(end_a, float), np.asarray(end_b, float)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 5:
        return None
    d /= length
    n = np.array([-d[1], d[0]])
    v, u = np.nonzero(edges)
    pts = np.column_stack((u, v)).astype(float)
    t = (pts - a) @ d
    near = (np.abs(pts @ n - n @ a) < 2.0) & (t > 0) & (t < length)
    pts = pts[near]
    if len(pts) < 8:
        return None
    ui, vi = pts[:, 0].astype(int), pts[:, 1].astype(int)
    g = np.column_stack((gx[vi, ui], gy[vi, ui]))
    along = np.abs(g @ d) / np.maximum(np.linalg.norm(g, axis=1), 1e-6)
    pts = pts[along < 0.5]  # the gradient must cross the line
    if len(pts) < 8:
        return None
    for _ in range(4):
        centre = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
        n2 = vt[-1] if vt[-1] @ n > 0 else -vt[-1]
        c2 = float(n2 @ centre)
        r = pts @ n2 - c2
        keep = np.abs(r) < max(0.75, 2.5 * 1.4826 * float(np.median(np.abs(r))))
        if keep.all() or keep.sum() < 8:
            break
        pts = pts[keep]
    r = pts @ n2 - c2
    return n2, c2, len(pts), float(np.sqrt(np.mean(r**2)))


def _edges(a_star, region):
    lo, hi = np.percentile(a_star[region], [2, 98])
    scaled = np.clip((a_star - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    edges = (cv2.Canny(scaled, 30, 90, L2gradient=True) > 0) & region
    gx = cv2.Sobel(a_star, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a_star, cv2.CV_32F, 0, 1, ksize=3)
    return edges, gx, gy


def _angle_gap(a: float, b: float) -> float:
    """Difference of two line-normal angles (modulo pi), radians in [0, pi/2]."""
    return abs(((a - b + math.pi / 2) % math.pi) - math.pi / 2)


def detect_board(
    rgb,
    intrinsics,
    pose: Pose,
    board_plane: Plane,
    camera: str = "",
    roi_override=None,
    res: float = 0.5e-3,
    min_gap: float = 8e-3,
    reference_angles: tuple[float, float] = (0.0, math.pi / 2),
    valid=None,
) -> BoardDetection:
    """Find the inner grid lines and corners of the orange/tan board.

    ``pose`` must be levelled on the board plane (the board is horizontal in the world).
    Family 0 holds the lines whose normal angle (world x-y) is nearest to
    ``reference_angles[0]``. Lines are numbered in the order of their world offset.
    ``valid`` (bool image) marks the pixels that can be board (for example: near the table
    in the depth image); other pixels are not used.
    """
    notes: list[str] = []
    a_star = orange_channel(rgb)
    if valid is not None:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    roi, levels, cell_px = board_roi(a_star, roi_override=roi_override, valid=valid)
    if valid is not None:
        a_star = np.where(valid, a_star, np.nan).astype(np.float32)
    normal_w, offset_w = plane_in_world(board_plane, pose)
    height = float(-offset_w / normal_w[2])
    # Board region made larger by 1.5 cells, so the small dim cells at the edge are in it.
    grow = int(max(5, 1.5 * cell_px)) | 1
    big = (
        cv2.dilate(roi.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
        > 0
    )
    if valid is not None:
        big &= valid
    v, u = np.nonzero(big)
    sample = np.column_stack((u, v))[:: max(1, len(u) // 4000)]
    world = pose.to_world(board_plane.intersect(sample, intrinsics))
    rect = Rectifier(pose, intrinsics, height, (world[:, :2].min(0), world[:, :2].max(0)), res)
    a_rect = rect.warp(a_star, border=float("nan"))
    kernel = np.ones((7, 7), np.uint8)
    valid_rect = cv2.erode(np.isfinite(a_rect).astype(np.uint8), kernel) > 0
    mask_rect = (rect.warp(big.astype(np.uint8), cv2.INTER_NEAREST) > 0) & valid_rect
    core_rect = (rect.warp(roi.astype(np.uint8), cv2.INTER_NEAREST) > 0) & valid_rect
    middle = 0.5 * (levels[0] + levels[1])
    a_fill = np.where(np.isfinite(a_rect), a_rect, middle).astype(np.float32)
    angles = list(_board_angles(a_fill, core_rect))
    if _angle_gap(angles[0], reference_angles[0]) > _angle_gap(angles[1], reference_angles[0]):
        angles = angles[::-1]
    # Use the representative of each angle (mod pi) nearest to the reference angle, so that
    # the line offsets of the two cameras have the same sense.
    angles = [
        a if math.cos(a - r) >= 0 else a - math.pi
        for a, r in zip(angles, reference_angles, strict=True)
    ]
    a_img = np.where(np.isfinite(a_star), a_star, middle).astype(np.float32)
    inner = cv2.erode(np.isfinite(a_star).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edges, gx, gy = _edges(a_img, big & inner)
    smooth = cv2.GaussianBlur(a_fill, (0, 0), 1.0)
    vgx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    vgy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    view_edges, _, _ = _edges(a_fill, mask_rect)
    er, ec = np.nonzero(view_edges)
    eg = np.column_stack((vgx[er, ec], vgy[er, ec]))
    emag = np.linalg.norm(eg, axis=1)
    eang = np.arctan2(eg[:, 1], eg[:, 0])
    families: list[list[GridLine]] = []
    for fam, family_theta in enumerate(angles):
        near = np.array([_angle_gap(x, family_theta) < math.radians(30) for x in eang], bool)
        near &= emag > 0
        if near.sum() < 20:
            families.append([])
            notes.append(f"family {fam}: too few edge pixels")
            continue
        vp, share = _vanishing_point(
            np.column_stack((ec[near], er[near])).astype(float),
            eg[near] / emag[near, None],
            emag[near],
        )
        notes.append(f"family {fam}: vanishing point fits {share:.0%} of the edge pixels")
        lines = []
        direction = (math.cos(family_theta), math.sin(family_theta))
        for n, c, t in _pencil_lines(vp, vgx, vgy, mask_rect, direction, min_gap / res):
            theta = math.atan2(n[1], n[0])
            extent = _check_line(a_rect, res, theta, c * res, levels, 3 * min_gap)
            if extent is None:
                continue
            # View metres -> world: (x, y) = view + (x0, y0).
            offset = c * res + rect.x0 * math.cos(theta) + rect.y0 * math.sin(theta)
            shift = -rect.x0 * math.sin(theta) + rect.y0 * math.cos(theta)
            ext = (extent[0] + shift, extent[1] + shift)
            ends = _line_points(theta, offset, ext)
            ends_px = pose.project(np.column_stack((ends, [height, height])), intrinsics)
            line = GridLine(fam, theta, offset, ext, position=t * res)
            fit = _refine_in_image(ends_px[0], ends_px[1], edges, gx, gy)
            if fit is None:
                d = ends_px[1] - ends_px[0]
                n_img = np.array([-d[1], d[0]]) / np.linalg.norm(d)
                line.image = (float(n_img[0]), float(n_img[1]), float(n_img @ ends_px[0]))
                notes.append(f"family {fam} line at {line.position * 1000:.1f} mm: no image fit")
            else:
                n_img, c_img, count, rms = fit
                line.image = (float(n_img[0]), float(n_img[1]), c_img)
                line.edge_pixels, line.image_rms_px = count, rms
            lines.append(line)
        families.append(sorted(lines, key=lambda g: g.position))
    pixels, index = [], []
    for i, a in enumerate(families[0]):
        for j, b in enumerate(families[1]):
            m = np.array([a.image[:2], b.image[:2]])
            if abs(np.linalg.det(m)) < 1e-3:
                continue
            p = np.linalg.solve(m, np.array([a.image[2], b.image[2]]))
            ui, vi = int(round(p[0])), int(round(p[1]))
            if not (0 <= vi < big.shape[0] and 0 <= ui < big.shape[1] and big[vi, ui]):
                continue
            xyz = pose.to_world(board_plane.intersect(p[None], intrinsics))[0]
            if _checker_ok(a_star, xyz, a, b, height, pose, intrinsics, levels, families):
                pixels.append(p)
                index.append((i, j))
    det = BoardDetection(
        camera=camera,
        pixels=np.array(pixels, float).reshape(-1, 2),
        index=np.array(index, int).reshape(-1, 2),
        lines=families,
        levels=levels,
        roi=roi,
        board_height_m=height,
        notes=notes,
    )
    view = rect.warp(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    det.rectified = _draw_rectified(view, rect, det)
    return det


def _checker_ok(a_star, xyz, a, b, height, pose, intrinsics, levels, families) -> bool:
    """The four cells around a corner alternate orange/tan.

    The test samples a small cluster at the centre of each of the four cells (half way to
    the next lines), so the blurred cell edges do not change the result.
    """

    def half_gaps(fam, line):
        lines = families[fam]
        k = lines.index(line)
        out = []
        for step in (1, -1):
            j = k + step
            gap = abs(lines[j].position - line.position) if 0 <= j < len(lines) else 0.012
            out.append(min(max(0.5 * gap, 3e-3), 15e-3))
        return out  # (+ direction, - direction)

    na = np.array([math.cos(a.theta), math.sin(a.theta)])
    nb = np.array([math.cos(b.theta), math.sin(b.theta)])
    ha, hb = half_gaps(0, a), half_gaps(1, b)
    cluster = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) * 1.5e-3
    pts = []
    for ia, sa in enumerate((1, -1)):
        for ib, sb in enumerate((1, -1)):
            centre = np.asarray(xyz)[:2] + sa * ha[ia] * na + sb * hb[ib] * nb
            pts.extend(centre + cluster)
    pts = np.column_stack((np.array(pts), np.full(len(pts), height)))
    cls = _classify(_sample_float(a_star, pose.project(pts, intrinsics)), levels).reshape(4, -1)
    if (cls == 0).any() or not (cls == cls[:, :1]).all():
        return False
    q = cls[:, 0]  # order: (+,+), (+,-), (-,+), (-,-)
    return bool(q[0] == q[3] and q[1] == q[2] and q[0] != q[1])


def _draw_rectified(view, rect: Rectifier, det: BoardDetection) -> np.ndarray:
    image = view.copy()
    colours = [(0, 0, 255), (255, 0, 0)]
    for fam, lines in enumerate(det.lines):
        for k, line in enumerate(lines):
            ends = rect.to_view(_line_points(line.theta, line.offset, line.extent))
            a, b = (tuple(int(x) for x in np.round(e)) for e in ends)
            cv2.line(image, a, b, colours[fam], 1, cv2.LINE_AA)
            cv2.putText(
                image, f"{'ab'[fam]}{k}", a, cv2.FONT_HERSHEY_SIMPLEX, 0.35, colours[fam], 1
            )
    return image


def draw_detection(rgb, det: BoardDetection, extra=None) -> np.ndarray:
    """BGR image: ROI outline, grid lines, corners. ``extra``: [(pixels, BGR colour, label)]."""
    image = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(
        det.roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (255, 255, 0), 1)
    colours = [(0, 0, 255), (255, 0, 0)]
    h = image.shape[0]
    for fam, lines in enumerate(det.lines):
        for k, line in enumerate(lines):
            mine = det.pixels[det.index[:, fam] == k] if len(det.pixels) else np.zeros((0, 2))
            if len(mine) == 0:
                continue
            n = np.array(line.image[:2])
            d = np.array([-n[1], n[0]])
            t = mine @ d
            a = n * line.image[2] + d * (t.min() - 8)
            b = n * line.image[2] + d * (t.max() + 8)
            cv2.line(
                image,
                tuple(int(x) for x in np.round(a)),
                tuple(int(x) for x in np.round(b)),
                colours[fam],
                1,
                cv2.LINE_AA,
            )
    for p in det.pixels:
        cv2.circle(image, tuple(int(x) for x in np.round(p)), 2, (0, 255, 0), -1)
    for row, (pixels, colour, label) in enumerate(extra or []):
        for p in pixels:
            cv2.drawMarker(
                image, tuple(int(x) for x in np.round(p)), colour, cv2.MARKER_CROSS, 7, 1
            )
        cv2.putText(image, label, (8, h - 10 - 16 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    return image


# ---------------------------------------------------------------------- table and board


def load_settings(path: Path | None = TWIN_CONFIG) -> dict:
    """The ``calibration`` section of configs/twin.yaml over the defaults."""
    settings = json.loads(json.dumps(DEFAULTS))
    if path is not None and Path(path).is_file():
        raw = yaml.safe_load(Path(path).read_text()) or {}
        settings.update(raw.get("calibration") or {})
    return settings


def polygon_mask(shape, polygon) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if polygon:
        cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 1)
    return mask > 0


def fit_table(depth_m, intrinsics, exclude=None, max_range=1.3, threshold=0.004) -> Plane:
    """Table plane: the largest plane in the depth image (the board region is not used)."""
    keep = None if exclude is None else ~exclude
    points, _ = depth_points(depth_m, intrinsics, keep, step=2)
    return fit_plane(points[points[:, 2] < max_range], threshold)


def height_map(depth_m, intrinsics, pose: Pose) -> np.ndarray:
    """World z (m) of each depth pixel; NaN where there is no depth."""
    points, pixels = depth_points(depth_m, intrinsics)
    out = np.full(depth_m.shape, np.nan, np.float32)
    out[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] = pose.to_world(points)[:, 2]
    return out


# --------------------------------------------------------------------- object registration


def voxel_down(points, size) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def object_points(frame, pose: Pose, settings) -> np.ndarray:
    """World points of the objects on the table (plates, bricks) in the shared work area."""
    intr = (frame.fx, frame.fy, frame.cx, frame.cy)
    points, _ = depth_points(frame.depth_m, intr)
    world = pose.to_world(points[points[:, 2] < settings["max_range_m"]])
    (x0, x1), (y0, y1) = settings["object_box_m"]
    z0, z1 = settings["object_z_m"]
    keep = (
        (world[:, 2] > z0)
        & (world[:, 2] < z1)
        & (world[:, 0] > x0)
        & (world[:, 0] < x1)
        & (world[:, 1] > y0)
        & (world[:, 1] < y1)
    )
    return voxel_down(world[keep], settings["voxel_m"])


def _normals(points, tree, k=12) -> np.ndarray:
    _, idx = tree.query(points, k=k)
    nb = points[idx] - points[idx].mean(1, keepdims=True)
    _, _, vt = np.linalg.svd(nb, full_matrices=False)
    return vt[:, -1, :]


def register_xy_yaw(source, target, iterations=40, pivot=None, start=(0.0, 0.0, 0.0)):
    """Point-to-plane ICP with 3 DoF: turn about z at ``pivot`` and shift in x, y.

    Returns (yaw, shift_xy, stats). The source moves onto the target. The turn is about the
    vertical line through ``pivot`` (default: source centre); ``shift_xy`` includes the
    effect of the pivot, so the full motion is p -> Rz(yaw) p + (shift_x, shift_y, 0).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    normals = _normals(target, tree)
    pivot = source[:, :2].mean(0) if pivot is None else np.asarray(pivot, float)
    yaw, tx, ty = start
    for it in range(iterations):
        r = rot_z(yaw)
        moved = (source - [*pivot, 0]) @ r.T + [*pivot, 0] + [tx, ty, 0]
        dist, idx = tree.query(moved)
        limit = max(0.004, 0.02 * (1 - it / iterations))
        pair = dist < limit
        if pair.sum() < 30:
            break
        p, n = moved[pair], normals[idx[pair]]
        res = ((p - target[idx[pair]]) * n).sum(1)
        rel = p[:, :2] - pivot
        jac = np.column_stack((n[:, 0] * -rel[:, 1] + n[:, 1] * rel[:, 0], n[:, 0], n[:, 1]))
        scale = 1.4826 * np.median(np.abs(res)) + 1e-4
        w = 1.0 / np.maximum(1.0, np.abs(res) / (1.5 * scale))  # Huber weights
        step = np.linalg.lstsq(jac * w[:, None], -res * w, rcond=None)[0]
        yaw, tx, ty = yaw + step[0], tx + step[1], ty + step[2]
        if np.abs(step).max() < 1e-6:
            break
    r = rot_z(yaw)
    moved = (source - [*pivot, 0]) @ r.T + [*pivot, 0] + [tx, ty, 0]
    dist, idx = tree.query(moved)
    pair = dist < 0.004
    res = ((moved[pair] - target[idx[pair]]) * normals[idx[pair]]).sum(1)
    rel = moved[pair][:, :2] - pivot
    n = normals[idx[pair]]
    jac = np.column_stack((n[:, 0] * -rel[:, 1] + n[:, 1] * rel[:, 0], n[:, 0], n[:, 1]))
    sigma = float(np.sqrt(np.mean(res**2))) if len(res) else float("nan")
    try:
        cov = np.linalg.inv(jac.T @ jac) * sigma**2
        std = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        std = np.full(3, np.nan)
    shift = np.array([tx, ty]) + pivot - r[:2, :2] @ pivot
    stats = {
        "pairs": int(pair.sum()),
        "source_points": int(len(source)),
        "rms_mm": round(sigma * 1000, 2),
        "std_yaw_deg": round(math.degrees(std[0]), 4),
        "std_xy_mm": [round(v * 1000, 3) for v in std[1:]],
    }
    return yaw, shift, stats


def cloud_agreement(source, target, limit=0.015) -> dict:
    """Distances (mm) from source points to the nearest target points (pairs < limit only)."""
    from scipy.spatial import cKDTree

    if len(source) == 0 or len(target) == 0:
        return {"pairs": 0}
    dist, _ = cKDTree(target).query(source)
    near = dist < limit
    d = dist[near] * 1000
    return {
        "pairs": int(near.sum()),
        "fraction": round(float(near.mean()), 3),
        "median_mm": round(float(np.median(d)), 2) if near.any() else None,
        "p90_mm": round(float(np.percentile(d, 90)), 2) if near.any() else None,
    }


def correction_size(source, target) -> dict:
    """3-DoF ICP from the source to the target: the size of the misfit that is still there."""
    yaw, shift, stats = register_xy_yaw(source, target)
    return {
        "yaw_deg": round(math.degrees(yaw), 3),
        "shift_mm": [round(v * 1000, 2) for v in shift],
        "shift_norm_mm": round(float(np.linalg.norm(shift)) * 1000, 2),
        **stats,
    }


# ------------------------------------------------------------------------- board corners


def match_corners(det_a: BoardDetection, det_b: BoardDetection, world_a, world_b, max_shift=3):
    """Pairs of corner rows (in a, in b) with the same grid index after the best index shift.

    The shift (di, dj) means: index in a = index in b + (di, dj). The shift with the
    smallest 2D rigid-fit residual wins. The uneven cells make it unique.
    Returns (pairs, shift, rms_mm, second_best_rms_mm).
    """
    results = []
    for di in range(-max_shift, max_shift + 1):
        for dj in range(-max_shift, max_shift + 1):
            key_b = {(int(i) + di, int(j) + dj): k for k, (i, j) in enumerate(det_b.index)}
            pairs = [
                (k, key_b[(int(i), int(j))])
                for k, (i, j) in enumerate(det_a.index)
                if (int(i), int(j)) in key_b
            ]
            if len(pairs) < 6:
                continue
            ia, ib = np.array(pairs).T
            yaw, shift = rigid_fit_2d(world_b[ib], world_a[ia])
            moved = world_b[ib][:, :2] @ rot_z(yaw)[:2, :2].T + shift
            rms = float(np.sqrt(np.mean(np.sum((moved - world_a[ia][:, :2]) ** 2, 1))))
            results.append((rms, (di, dj), pairs))
    if not results:
        return [], None, float("nan"), float("nan")
    results.sort(key=lambda r: r[0])
    second = results[1][0] * 1000 if len(results) > 1 else float("nan")
    return results[0][2], results[0][1], results[0][0] * 1000, second


def corner_world(det: BoardDetection, intrinsics, pose: Pose, board: Plane) -> np.ndarray:
    return pose.to_world(board.intersect(det.pixels, intrinsics))


def camera_motion(image_a, image_b) -> dict:
    """Median image shift (px) between two colour images of the same camera (DIS flow).

    Only pixels that look the same in both images count, so moved objects do not count.
    """
    ga = cv2.cvtColor(np.ascontiguousarray(image_a), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(np.ascontiguousarray(image_b), cv2.COLOR_RGB2GRAY)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(ga, gb, None)
    h, w = ga.shape
    grid = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(gb, grid[1] + flow[..., 0], grid[0] + flow[..., 1], cv2.INTER_LINEAR)
    texture = cv2.Laplacian(cv2.GaussianBlur(ga, (0, 0), 1.0), cv2.CV_32F)
    same = (np.abs(warped.astype(float) - ga) < 12) & (np.abs(texture) > 4)
    f = flow[same]
    return {
        "static_pixels": int(same.sum()),
        "median_shift_px": [round(float(v), 3) for v in np.median(f, 0)],
        "p90_shift_norm_px": round(float(np.percentile(np.linalg.norm(f, axis=1), 90)), 3),
    }


# ----------------------------------------------------------------------------- pipeline


def camera_pose(scene, name) -> tuple[Pose, tuple]:
    camera = scene.cameras[name]
    return Pose(camera.rotation_cv(), np.array(camera.pos, float)), camera.pinhole()


def level_cameras(frames, scene, settings) -> dict:
    """Table plane per camera, present and levelled pose, plane errors before and after."""
    out = {}
    for name in CAMERAS:
        frame = frames[name]
        pose0, cfg_intr = camera_pose(scene, name)
        intr = (frame.fx, frame.fy, frame.cx, frame.cy)
        roi = polygon_mask(frame.depth_m.shape, settings["board_roi"].get(name))
        exclude = cv2.dilate(roi.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        table = fit_table(
            frame.depth_m, intr, exclude, settings["max_range_m"], settings["table_threshold_m"]
        )
        pivot = settings["overhead_pivot_xy"] if name == "overhead" else (0.3, 0.0)
        pose1 = level_pose(pose0, table, pivot)
        board = None
        if roi.any():
            pts, _ = depth_points(frame.depth_m, intr, roi)
            d = table.distance(pts)
            lo, hi = settings["board_band_m"]
            near = (d > lo) & (d < hi)
            if near.sum() > 200:
                thick = float(np.median(d[near]))
                board = Plane(table.normal, table.offset - thick, int(near.sum()), len(pts))
        out[name] = {
            "frame": frame,
            "intrinsics": intr,
            "config_intrinsics_diff_px": round(float(np.abs(np.subtract(cfg_intr, intr)).max()), 3),
            "table": table,
            "board": board or Plane(table.normal, table.offset - 0.003),
            "pose0": pose0,
            "pose1": pose1,
        }
    return out


def plane_report(cam) -> dict:
    t, p0, p1 = cam["table"], cam["pose0"], cam["pose1"]
    return {
        "fit": t.summary(),
        "before": {"origin": plane_error(t, p0, (0, 0)), "x0.3": plane_error(t, p0, (0.3, 0))},
        "after": {"origin": plane_error(t, p1, (0, 0)), "x0.3": plane_error(t, p1, (0.3, 0))},
        "pose_change": pose_change(p0, p1),
    }


def pose_change(a: Pose, b: Pose) -> dict:
    return {
        "rotation_deg": round(angle_deg(a.rotation, b.rotation), 3),
        "position_mm": [round(float(v) * 1000, 2) for v in (b.position - a.position)],
    }


def calibrate(
    fixture,
    scene_config=None,
    settings=None,
    test_fixtures=(),
    board_images=None,
    out_dir: Path | None = None,
) -> dict:
    """Full refinement. Returns the report (with the new poses under "proposal")."""
    settings = settings or load_settings()
    scene = load_scene_config(scene_config)
    frames = {f.camera: f for f in load_fixture(fixture)}
    cams = level_cameras(frames, scene, settings)
    report: dict = {
        "tool": "brickcell.twin.calibrate",
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixture": str(fixture),
        "scene_config": scene.source,
        "intrinsics": "factory (fixture meta.json); not changed",
        "planes": {name: plane_report(cams[name]) for name in CAMERAS},
    }
    for name in CAMERAS:
        report["planes"][name]["config_intrinsics_diff_px"] = cams[name][
            "config_intrinsics_diff_px"
        ]
        b = cams[name]["board"]
        report["planes"][name]["board_above_table_mm"] = round(
            (cams[name]["table"].offset - b.offset) * 1000, 2
        )

    over, side = cams["overhead"], cams["side"]
    # Diagnostic only: register side object points (plates, bricks) onto overhead points.
    src = object_points(side["frame"], side["pose1"], settings)
    dst = object_points(over["frame"], over["pose1"], settings)
    yaw, shift, stats = register_xy_yaw(src, dst)
    report["object_registration"] = {
        "use": "diagnostic only: the two halves give different answers (round plates, "
        "different visible faces), so it does not fix x, y, yaw",
        "yaw_deg": round(math.degrees(yaw), 3),
        "shift_mm": [round(v * 1000, 2) for v in shift],
        **stats,
        "split_check": split_check(src, dst, side["pose1"]),
    }
    proposal = {"overhead": over["pose1"], "side": side["pose1"].turned(yaw, shift)}
    side_source = "object ICP (weak)"
    if board_images:
        board = board_fit(cams, board_images, settings, out_dir)
        report["board_check"] = board
        motion_ok = all(
            max(abs(v) for v in m["median_shift_px"]) < settings.get("max_motion_px", 0.5)
            for m in board["camera_motion_vs_fixture"].values()
        )
        if "_fit" in board and motion_ok and board["shared_corners"] >= 8:
            yaw_b, shift_b = board["_fit"]
            proposal["side"] = side["pose1"].turned(yaw_b, shift_b)
            side_source = "board corners (2D rigid fit on the levelled poses)"
        elif not motion_ok:
            board["warning"] = "a camera moved between the board image and the fixture"
    report["side_xy_yaw_source"] = side_source
    variants = {
        "present": (over["pose0"], side["pose0"]),
        "levelled_only": (over["pose1"], side["pose1"]),
        "proposal": (proposal["overhead"], proposal["side"]),
    }
    if "board_check" in report and "_pairs" in report["board_check"]:
        evaluate_board(report["board_check"], cams, variants)
    report["cross_camera"] = {str(fixture): agreement(frames, variants, settings)}
    for test in test_fixtures:
        tframes = {f.camera: f for f in load_fixture(test)}
        entry = agreement(tframes, variants, settings)
        entry["camera_motion_vs_fit_fixture"] = {
            name: camera_motion(frames[name].rgb, tframes[name].rgb) for name in CAMERAS
        }
        tcams = level_cameras(tframes, scene, settings)
        entry["table_under_proposal"] = {
            name: {
                "origin": plane_error(tcams[name]["table"], proposal[name], (0, 0)),
                "x0.3": plane_error(tcams[name]["table"], proposal[name], (0.3, 0)),
            }
            for name in CAMERAS
        }
        report["cross_camera"][str(test)] = entry

    report["proposal"] = {
        name: {**pose.as_dict(), "change": pose_change(cams[name]["pose0"], pose)}
        for name, pose in proposal.items()
    }
    report["absolute_anchor"] = {
        "kept": "overhead x, y, yaw from the present config (SO-101 model fit, 05:09 UTC)",
        "levelling_pivot_xy": settings["overhead_pivot_xy"],
        "note": "The fixtures have no arm joint angles, and the arm and a hand hide the base. "
        "This data cannot check x, y or yaw of the robot base frame.",
    }
    report["_poses"] = proposal
    report["_cams"] = cams
    return report


def split_check(src, dst, side_pose=None) -> dict:
    median_y = float(np.median(src[:, 1]))
    out = {}
    for label, fit_mask in (
        ("fit_low_y", src[:, 1] < median_y),
        ("fit_high_y", src[:, 1] >= median_y),
    ):
        yaw, shift, _ = register_xy_yaw(src[fit_mask], dst)
        test = src[~fit_mask] @ rot_z(yaw).T + [shift[0], shift[1], 0]
        out[label] = {
            "yaw_deg": round(math.degrees(yaw), 3),
            "shift_mm": [round(v * 1000, 2) for v in shift],
            "held_out_half": cloud_agreement(test, dst),
        }
    return out


def agreement(frames, variants, settings) -> dict:
    out = {}
    for label, (pose_o, pose_s) in variants.items():
        a = object_points(frames["side"], pose_s, settings)
        b = object_points(frames["overhead"], pose_o, settings)
        # Same object box for both; the clouds come from each variant's own poses.
        out[label] = {"side_to_overhead": cloud_agreement(a, b)}
    return out


def board_fit(cams, board_images, settings, out_dir) -> dict:
    """Board corners in clear-board images (same camera poses as the fixture).

    Gives the side x, y, yaw from a 2D rigid fit of the side corners onto the overhead corners
    (both lifted on their own board plane with the levelled poses), and a hold-out check.
    """
    dets, worlds, motion = {}, {}, {}
    for name in CAMERAS:
        rgb = cv2.imread(str(board_images[name]))
        if rgb is None:
            raise FileNotFoundError(board_images[name])
        rgb = rgb[..., ::-1]
        cam = cams[name]
        motion[name] = camera_motion(rgb, cam["frame"].rgb)
        det = detect_board(
            rgb,
            cam["intrinsics"],
            cam["pose1"],
            cam["board"],
            name,
            roi_override=settings["board_roi"].get(name),
        )
        dets[name] = det
        worlds[name] = corner_world(det, cam["intrinsics"], cam["pose1"], cam["board"])
        if out_dir is not None:
            cv2.imwrite(str(Path(out_dir) / f"board-corners-{name}.png"), draw_detection(rgb, det))
            cv2.imwrite(str(Path(out_dir) / f"board-topdown-{name}.png"), det.rectified)
    out: dict = {
        "images": {k: str(v) for k, v in board_images.items()},
        "camera_motion_vs_fixture": motion,
        "detections": {k: d.summary() for k, d in dets.items()},
    }
    pairs, shift, rms, second = match_corners(
        dets["overhead"], dets["side"], worlds["overhead"], worlds["side"]
    )
    if not pairs:
        out["result"] = "not enough shared corners"
        out["shared_corners"] = 0
        return out
    ia, ib = np.array(pairs).T
    out["shared_corners"] = len(pairs)
    out["index_shift"] = list(shift)
    out["index_rms_mm_best_vs_second"] = [round(rms, 2), round(second, 2)]
    out["board_lines_mm"] = {
        name: [[round(g.position * 1000, 1) for g in fam] for fam in dets[name].lines]
        for name in CAMERAS
    }
    src, dst = worlds["side"][ib], worlds["overhead"][ia]
    yaw, sh = rigid_fit_2d(src, dst)
    moved = src[:, :2] @ rot_z(yaw)[:2, :2].T + sh
    out["fit_residual_mm"] = _stats(np.linalg.norm(moved - dst[:, :2], axis=1) * 1000)
    out["fit_yaw_deg"] = round(math.degrees(yaw), 3)
    out["fit_shift_mm"] = [round(v * 1000, 2) for v in sh]
    # Hold-out: fit on the corners of one half of the board, test on the other half.
    index_a = dets["overhead"].index[ia]
    holdout = {}
    for fam in (0, 1):
        cut = np.median(index_a[:, fam])
        for label, train in (
            (f"family{fam}_low", index_a[:, fam] <= cut),
            (f"family{fam}_high", index_a[:, fam] > cut),
        ):
            if train.sum() < 4 or (~train).sum() < 2:
                continue
            y2, s2 = rigid_fit_2d(src[train], dst[train])
            test = src[~train][:, :2] @ rot_z(y2)[:2, :2].T + s2
            holdout[label] = {
                "train": int(train.sum()),
                "test_error_mm": _stats(np.linalg.norm(test - dst[~train][:, :2], axis=1) * 1000),
                "yaw_deg": round(math.degrees(y2), 3),
            }
    out["holdout"] = holdout
    out["_fit"] = (yaw, sh)
    out["_pairs"] = (dets, ia, ib)
    return out


def evaluate_board(out, cams, variants) -> None:
    """Corner distances and reprojection between the cameras for each pose variant."""
    dets, ia, ib = out["_pairs"]
    cam_o, cam_s = cams["overhead"], cams["side"]
    for label, (pose_o, pose_s) in variants.items():
        wo = pose_o.to_world(
            cam_o["board"].intersect(dets["overhead"].pixels[ia], cam_o["intrinsics"])
        )
        ws = pose_s.to_world(cam_s["board"].intersect(dets["side"].pixels[ib], cam_s["intrinsics"]))
        diff = (ws - wo) * 1000
        reproj = pose_s.project(wo, cam_s["intrinsics"]) - dets["side"].pixels[ib]
        out[label] = {
            "corner_distance_mm": _stats(np.linalg.norm(diff, axis=1)),
            "corner_dz_mm_median": round(float(np.median(diff[:, 2])), 2),
            "reprojection_overhead_into_side_px": _stats(np.linalg.norm(reproj, axis=1)),
        }


def _stats(values) -> dict:
    v = np.asarray(values, float)
    return {
        "median": round(float(np.median(v)), 2),
        "p90": round(float(np.percentile(v, 90)), 2),
        "max": round(float(v.max()), 2),
        "n": int(len(v)),
    }


# ------------------------------------------------------------------------------- outputs


def _fmt(values, digits) -> str:
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in values) + "]"


def camera_entry_lines(name, pose: Pose, intrinsics, size=(640, 480), indent="  ") -> list[str]:
    fx, fy, cx, cy = intrinsics
    rows = ", ".join(_fmt(row, 5) for row in pose.rotation)
    return [
        f"{indent}{name}:",
        f"{indent}  enabled: true",
        f"{indent}  pos: {_fmt(pose.position, 4)}",
        f"{indent}  rotation_cv: [{rows}]",
        f"{indent}  fx: {fx:.2f}",
        f"{indent}  fy: {fy:.2f}",
        f"{indent}  cx: {cx:.2f}",
        f"{indent}  cy: {cy:.2f}",
        f"{indent}  width: {size[0]}",
        f"{indent}  height: {size[1]}",
    ]


def write_scene_poses(path: Path, poses: dict, note: str) -> None:
    """Replace pos and rotation_cv of the named camera entries; keep all other text.

    ``note`` goes into one comment line just above each entry (a line that starts with
    "# REFINED" is replaced). Other comments, entries and the key order do not change.
    """
    lines = Path(path).read_text().splitlines(keepends=True)
    for name, pose in poses.items():
        start = next(
            (
                i
                for i, line in enumerate(lines)
                if re.match(rf"^  {re.escape(name)}:\s*(#.*)?$", line)
            ),
            None,
        )
        if start is None:
            raise ValueError(f"{path}: camera entry '{name}' not found.")
        end = start + 1
        while end < len(lines) and (lines[end].startswith("    ") or not lines[end].strip()):
            end += 1
        done = set()
        for i in range(start + 1, end):
            match = re.match(r"^(\s+)(pos|rotation_cv):", lines[i])
            if not match:
                continue
            key = match.group(2)
            value = (
                _fmt(pose.position, 4)
                if key == "pos"
                else "[" + ", ".join(_fmt(row, 5) for row in pose.rotation) + "]"
            )
            lines[i] = f"{match.group(1)}{key}: {value}\n"
            done.add(key)
        if done != {"pos", "rotation_cv"}:
            raise ValueError(f"{path}: entry '{name}' needs pos and rotation_cv lines.")
        comment = f"  # REFINED {note}\n"
        if start > 0 and lines[start - 1].startswith("  # REFINED"):
            lines[start - 1] = comment
        else:
            lines.insert(start, comment)
    Path(path).write_text("".join(lines))


def proposal_yaml(report, cams) -> str:
    poses = report["_poses"]
    planes = report["planes"]
    fit = report["cross_camera"]
    lines = [
        "# PROPOSED camera poses for configs/sim-scene.yaml (overhead and side entries).",
        f"# Tool: python -m brickcell.twin.calibrate, {report['time_utc']}.",
        f"# Fixture: {report['fixture']}. Intrinsics: factory (not changed).",
        "# Method: (1) table plane per camera from depth (RANSAC + least squares) -> table at",
        "#   z = 0: sets z, roll, pitch of each camera.",
        f"#   (2) side x, y, yaw: {report['side_xy_yaw_source']}.",
        "#   (3) overhead x, y, yaw: kept from the present SO-101 model fit (not checked here).",
    ]
    for name in CAMERAS:
        b = planes[name]["before"]["origin"]
        lines.append(
            f"# {name}: table before: tilt {b['tilt_deg']} deg, height at origin "
            f"{b['height_mm']} mm; after: 0 deg, 0 mm. Plane rms "
            f"{planes[name]['fit']['rms_mm']} mm."
        )
    for key, entry in fit.items():
        lines.append(
            f"# Cross-camera objects {Path(key).name}: median NN "
            f"{entry['present']['side_to_overhead'].get('median_mm')} mm (present) -> "
            f"{entry['proposal']['side_to_overhead'].get('median_mm')} mm (proposal)."
        )
    if "board_check" in report and "proposal" in report["board_check"]:
        bc = report["board_check"]
        lines.append(
            f"# Board corner check ({bc['shared_corners']} corners): median distance "
            f"{bc['present']['corner_distance_mm']['median']} mm (present) -> "
            f"{bc['proposal']['corner_distance_mm']['median']} mm (proposal)."
        )
    lines.append("cameras:")
    for name in CAMERAS:
        lines += camera_entry_lines(name, poses[name], cams[name]["intrinsics"])
    return "\n".join(lines) + "\n"


def overlay_image(frames, variants, settings, title) -> np.ndarray:
    """Top-down image of the object points: overhead in magenta, side in green, per variant."""
    panels = []
    (x0, x1), (y0, y1) = settings["object_box_m"]
    res = 0.002
    w, h = int((y1 - y0) / res), int((x1 - x0) / res)
    for label, (pose_o, pose_s) in variants.items():
        panel = np.full((h, w, 3), 255, np.uint8)
        for frame, pose, colour in (
            (frames["overhead"], pose_o, (200, 0, 200)),
            (frames["side"], pose_s, (0, 160, 0)),
        ):
            pts = object_points(frame, pose, settings)
            col = ((y1 - pts[:, 1]) / res).astype(int)  # image right = world -y
            row = ((x1 - pts[:, 0]) / res).astype(int)  # image down = world -x
            ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
            panel[row[ok], col[ok]] = colour
        cv2.putText(panel, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        panels.append(panel)
    image = np.hstack([np.pad(p, ((0, 0), (0, 4), (0, 0))) for p in panels])
    bar = np.full((22, image.shape[1], 3), 255, np.uint8)
    cv2.putText(
        bar,
        title + "  (magenta: overhead, green: side; top = +x, right = -y)",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
    )
    return np.vstack([bar, image])


def save_outputs(report, out_dir: Path, settings, test_fixtures) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = report.pop("_cams")
    poses = report.pop("_poses")
    report["_poses"] = poses
    (out_dir / "proposal.yaml").write_text(proposal_yaml(report, cams))
    report.pop("_poses")
    board = report.get("board_check", {})
    pairs = board.pop("_pairs", None)
    variants = {
        "present": (cams["overhead"]["pose0"], cams["side"]["pose0"]),
        "proposal": (poses["overhead"], poses["side"]),
    }
    for fixture in [report["fixture"], *map(str, test_fixtures)]:
        frames = {f.camera: f for f in load_fixture(fixture)}
        image = overlay_image(frames, variants, settings, Path(fixture).name)
        cv2.imwrite(str(out_dir / f"overlay-{Path(fixture).name}.png"), image)
    if pairs is not None:
        dets, ia, ib = pairs
        rgb = cv2.imread(board["images"]["side"])[..., ::-1]
        cam_o, cam_s = cams["overhead"], cams["side"]
        wo = cam_o["board"].intersect(dets["overhead"].pixels[ia], cam_o["intrinsics"])
        extra = []
        for label, pose_o, pose_s, colour in (
            ("present", cam_o["pose0"], cam_s["pose0"], (0, 0, 255)),
            ("proposal", poses["overhead"], poses["side"], (255, 0, 255)),
        ):
            px = pose_s.project(pose_o.to_world(wo), cam_s["intrinsics"])
            extra.append((px, colour, f"overhead corners in side image: {label}"))
        cv2.imwrite(
            str(out_dir / "board-reprojection-side.png"), draw_detection(rgb, dets["side"], extra)
        )
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=_json) + "\n")


def _json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m brickcell.twin.calibrate", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--fixture", type=Path, required=True, help="fixture for the fit")
    parser.add_argument(
        "--test-fixture",
        type=Path,
        action="append",
        default=[],
        help="held-out fixture (same camera poses); may repeat",
    )
    parser.add_argument("--scene-config", type=Path, default=SCENE_CONFIG_PATH)
    parser.add_argument("--twin-config", type=Path, default=TWIN_CONFIG)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/twin-dev/calibration")
    parser.add_argument(
        "--board-image",
        action="append",
        default=[],
        metavar="CAMERA=PATH",
        help="colour image of the clear board from the same camera pose; "
        "used only as an independent check, not for the proposal",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the new overhead and side poses into --scene-config",
    )
    args = parser.parse_args(argv)
    settings = load_settings(args.twin_config)
    board_images = None
    if args.board_image:
        board_images = dict(item.split("=", 1) for item in args.board_image)
        if set(board_images) != set(CAMERAS):
            parser.error("--board-image needs overhead=PATH and side=PATH")
    args.out.mkdir(parents=True, exist_ok=True)
    report = calibrate(
        args.fixture, args.scene_config, settings, args.test_fixture, board_images, args.out
    )
    poses = report["_poses"]
    save_outputs(report, args.out, settings, args.test_fixture)
    if args.write:
        note = (
            f"{report['time_utc']} by brickcell.twin.calibrate from "
            f"{Path(report['fixture']).name}: table planes + "
            f"{report['side_xy_yaw_source']}; "
            "see artifacts/twin-dev/calibration/report.json."
        )
        write_scene_poses(args.scene_config, poses, note)
    summary = {
        "planes": {
            k: {"before": v["before"]["origin"], "pose_change": v["pose_change"]}
            for k, v in report["planes"].items()
        },
        "side_xy_yaw_source": report["side_xy_yaw_source"],
        "board_fit_residual_mm": report.get("board_check", {}).get("fit_residual_mm"),
        "cross_camera": {
            Path(k).name: {
                lab: e[lab]["side_to_overhead"].get("median_mm")
                for lab in ("present", "levelled_only", "proposal")
            }
            for k, e in report["cross_camera"].items()
        },
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
