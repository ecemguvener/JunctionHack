"""Geometry helpers for the station twin perception.

Imports only numpy, OpenCV and the standard library. No SciPy, no MuJoCo: this module must run
in the core environment, the MuJoCo environment and the RealSense environment on the laptop.

Frames: world = robot base frame, +Z up, metres. Camera frame = OpenCV (+X right, +Y down, the
camera looks along +Z). A camera pose is (R, t): R is the camera-to-world rotation (its columns
are the camera axes in the world), t is the camera position in the world.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------- rotations


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rodrigues(omega) -> np.ndarray:
    """Rotation matrix of the rotation vector ``omega`` (axis times angle, radians)."""
    omega = np.asarray(omega, float)
    angle = float(np.linalg.norm(omega))
    if angle < 1e-12:
        return np.eye(3)
    k = omega / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def rot_z_batch(angles: np.ndarray) -> np.ndarray:
    """Rotations (N, 3, 3) about Z by the angles (N,) in radians."""
    c, s = np.cos(angles), np.sin(angles)
    out = np.zeros((len(c), 3, 3))
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1], out[:, 2, 2] = c, -s, s, c, 1.0
    return out


def rodrigues_batch(omega: np.ndarray) -> np.ndarray:
    """Rotation matrices (N, 3, 3) of the rotation vectors (N, 3)."""
    omega = np.asarray(omega, float)
    angle = np.linalg.norm(omega, axis=1)
    k = omega / np.maximum(angle, 1e-12)[:, None]
    kx = np.zeros((len(omega), 3, 3))
    kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    kx[:, 1, 0], kx[:, 2, 0], kx[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(angle)[:, None, None], (1 - np.cos(angle))[:, None, None]
    return np.eye(3)[None] + s * kx + c * (kx @ kx)


def rotation_angle_deg(a, b) -> float:
    """Angle of the rotation between two rotation matrices, degrees."""
    r = np.asarray(a, float).T @ np.asarray(b, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def yaw_of(rotation) -> float:
    """Yaw (rad) of the rotation about world Z that is nearest to ``rotation``."""
    r = np.asarray(rotation, float)
    return math.atan2(r[1, 0] - r[0, 1], r[0, 0] + r[1, 1])


def symmetric_rotation_error_deg(r_true, r_est, symmetries) -> float:
    """Smallest angle between ``r_est`` and ``r_true @ S`` for S in the symmetry group.

    ``symmetries`` are rotations in the part frame that map the part onto itself (identity
    included). Two poses that differ by such a rotation give the same shape in the world.
    """
    return min(rotation_angle_deg(np.asarray(r_true) @ s, r_est) for s in symmetries)


# ----------------------------------------------------------------------------- meshes


def read_stl(path: str | Path) -> np.ndarray:
    """Triangles (n, 3, 3) of a binary STL, in the units of the file."""
    raw = Path(path).read_bytes()
    count = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
    records = np.frombuffer(
        raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count, 84
    )
    return records["v"].astype(float)


def sample_surface(
    triangles: np.ndarray, spacing: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Points and unit normals on the mesh surface, about one point per ``spacing`` squared.

    The number of points on a triangle is proportional to its area. The normals come from the
    triangle winding; the sign is not used by the callers.
    """
    rng = np.random.default_rng(seed)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = area > 1e-14
    a, b, c, cross, area = a[keep], b[keep], c[keep], cross[keep], area[keep]
    expected = area / spacing**2
    counts = np.floor(expected + rng.random(len(expected))).astype(int)
    tri = np.repeat(np.arange(len(area)), counts)
    r1, r2 = rng.random(len(tri)), rng.random(len(tri))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    points = a[tri] + r1[:, None] * (b[tri] - a[tri]) + r2[:, None] * (c[tri] - a[tri])
    normals = cross[tri] / np.linalg.norm(cross[tri], axis=1, keepdims=True)
    return points, normals


def nearest_neighbours(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest ``reference`` point and the distance for each ``query`` point.

    Uses an OpenCV FLANN k-d tree (approximate, but near exact for the sizes used here).
    """
    reference = np.ascontiguousarray(reference, np.float32)
    query = np.ascontiguousarray(query, np.float32)
    if len(reference) == 0 or len(query) == 0:
        return np.zeros(len(query), int), np.full(len(query), np.inf)
    index = cv2.flann_Index(reference, {"algorithm": 1, "trees": 4})
    idx, dist2 = index.knnSearch(query, 1, params={"checks": 64})
    return idx[:, 0].astype(int), np.sqrt(np.maximum(dist2[:, 0], 0.0)).astype(float)


@dataclass
class ClosestPointGrid:
    """Voxel grid in the part frame: each voxel holds the index of the nearest surface sample.

    With this grid, one nearest-neighbour query is one array look-up, so ICP is fast without a
    k-d tree. Points outside the grid have no neighbour (index -1).
    """

    origin: np.ndarray  # (3,) part-frame position of voxel (0, 0, 0)
    resolution: float
    index: np.ndarray  # (nx, ny, nz) int32

    @classmethod
    def build(cls, points: np.ndarray, resolution: float, padding: float) -> ClosestPointGrid:
        lo = points.min(0) - padding
        hi = points.max(0) + padding
        shape = np.ceil((hi - lo) / resolution).astype(int) + 1
        axes = [lo[i] + resolution * np.arange(shape[i]) for i in range(3)]
        centres = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        idx, _ = nearest_neighbours(points, centres)
        return cls(lo, resolution, idx.reshape(tuple(shape)).astype(np.int32))

    def lookup(self, points: np.ndarray) -> np.ndarray:
        """Nearest sample index for each point (part frame); -1 outside the grid."""
        shape = self.index.shape
        cell = np.floor((points - (self.origin - 0.5 * self.resolution)) / self.resolution)
        inside = (
            (cell[:, 0] >= 0)
            & (cell[:, 0] < shape[0])
            & (cell[:, 1] >= 0)
            & (cell[:, 1] < shape[1])
            & (cell[:, 2] >= 0)
            & (cell[:, 2] < shape[2])
        )
        c = cell.astype(np.int64)
        flat = (c[:, 0] * shape[1] + c[:, 1]) * shape[2] + c[:, 2]
        flat[~inside] = 0
        out = self.index.reshape(-1)[flat].astype(np.int64)
        out[~inside] = -1
        return out


def rotational_symmetries(
    points: np.ndarray, tolerance: float = 0.0008, steps: int = 4
) -> list[np.ndarray]:
    """Rotations about the part Z axis (through the origin) that map the surface onto itself.

    Tries the angles k * 360 / steps. A rotation is a symmetry if the mean distance from each
    turned point to the nearest original point is less than ``tolerance`` (metres).
    """
    rng = np.random.default_rng(1)
    probe = points[rng.choice(len(points), min(3000, len(points)), replace=False)]
    result = [np.eye(3)]
    for k in range(1, steps):
        r = rot_z(2 * math.pi * k / steps)
        _, dist = nearest_neighbours(points, probe @ r.T)
        if float(np.mean(dist)) < tolerance:
            result.append(r)
    return result


def file_digest(*paths: str | Path) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- cameras


def deproject(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 1,
    max_depth: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points (N, 3) and their flat pixel indices for the valid depth pixels."""
    h, w = depth_m.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[vs, us].astype(float)
    valid = (z > 0) & (z < max_depth) & np.isfinite(z)
    us, vs, z = us[valid], vs[valid], z[valid]
    points = np.column_stack(((us - cx) / fx * z, (vs - cy) / fy * z, z))
    return points, (vs * w + us).astype(np.int64)


def project(points_world: np.ndarray, rotation: np.ndarray, position, fx, fy, cx, cy):
    """Pixels (N, 2) and depth along the optical axis (N,) of world points."""
    local = (np.asarray(points_world, float) - np.asarray(position, float)) @ np.asarray(rotation)
    z = local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack((fx * local[:, 0] / z + cx, fy * local[:, 1] / z + cy))
    return uv, z


# ----------------------------------------------------------------------------- planes


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float,
    iterations: int = 200,
    seed: int = 0,
    refine_points: int | None = None,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Plane n . p + d = 0 (unit n with n_z >= 0) with the most inliers, refined by SVD.

    ``refine_points``: if set, the SVD refinement uses a random sample of this many points
    (faster for large clouds). Returns (n, d, inlier mask over all points) or None if there
    are fewer than 3 points.
    """
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    triples = rng.integers(0, len(points), size=(iterations, 3))
    a, b, c = points[triples[:, 0]], points[triples[:, 1]], points[triples[:, 2]]
    normals = np.cross(b - a, c - a)
    norm = np.linalg.norm(normals, axis=1)
    ok = norm > 1e-9
    if not ok.any():
        return None
    normals = normals[ok] / norm[ok, None]
    ds = -np.einsum("ij,ij->i", normals, a[ok])
    probe = np.take(points, rng.choice(len(points), min(len(points), 4000), replace=False), axis=0)
    counts = (np.abs(probe @ normals.T + ds) < threshold).sum(0)
    best = int(np.argmax(counts))
    n, d = normals[best], ds[best]
    refine = points
    if refine_points is not None and len(points) > refine_points:
        refine = np.take(points, rng.integers(0, len(points), refine_points), axis=0)
    for _ in range(3):
        inliers = np.abs(refine @ n + d) < threshold
        if inliers.sum() < 3:
            break
        sel = np.take(refine, np.flatnonzero(inliers), axis=0).astype(float)
        centre = sel.mean(0)
        q = sel - centre
        # Normal = eigenvector of the smallest eigenvalue of the 3x3 scatter matrix (the same
        # as the last right singular vector of q, but much faster for many points).
        _, vectors = np.linalg.eigh(q.T @ q)
        n = vectors[:, 0]
        d = -float(n @ centre)
    if n[2] < 0:
        n, d = -n, -d
    return n, float(d), np.abs(points @ n + d) < threshold


def plane_to_z0(normal: np.ndarray, d: float, pivot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid motion (R, t), p' = R p + t, that moves the plane n . p + d = 0 onto z = 0.

    The rotation is the smallest one that turns n onto +Z, about the point of the plane
    below ``pivot``. So points near the pivot move mostly in Z, not in X and Y.
    """
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    axis = np.cross(n, [0.0, 0.0, 1.0])
    s = np.linalg.norm(axis)
    angle = math.atan2(s, n[2])
    r = rodrigues(axis / s * angle) if s > 1e-12 else np.eye(3)
    pivot = np.asarray(pivot, float)
    foot = pivot - (n @ pivot + d) * n  # point of the plane nearest the pivot
    target = np.array([foot[0], foot[1], 0.0])
    return r, target - r @ foot


# ----------------------------------------------------------------------------- circles


def circle_hough(
    xy: np.ndarray,
    radii: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    cell: float,
    angles: int = 64,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hough votes for circle centres when each point has a known radius.

    A point at ``xy`` with radius ``r`` votes for all centres on the circle of radius ``r``
    around it. Returns the accumulator (ny, nx) and the cell centre of index (0, 0).
    A circle that is only partly in view still gives a clear peak.
    """
    lo = np.asarray(lo, float)
    shape = (np.ceil((np.asarray(hi) - lo) / cell).astype(int) + 1)[::-1]  # (ny, nx)
    theta = np.linspace(0, 2 * math.pi, angles, endpoint=False)
    cx = xy[:, 0:1] + radii[:, None] * np.cos(theta)[None]
    cy = xy[:, 1:2] + radii[:, None] * np.sin(theta)[None]
    ix = np.rint((cx - lo[0]) / cell).astype(np.int64)
    iy = np.rint((cy - lo[1]) / cell).astype(np.int64)
    ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    w = np.ones_like(cx) if weights is None else np.repeat(weights[:, None], angles, 1)
    acc = np.bincount((iy * shape[1] + ix)[ok], w[ok], minlength=int(shape[0] * shape[1]))
    return acc.reshape(tuple(shape)).astype(np.float32), lo


def fit_circle_known_radius(
    xy: np.ndarray,
    radii: np.ndarray,
    centre,
    iterations: int = 10,
    huber: float = 0.003,
    scale_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Gauss-Newton fit of a centre c (and a scale s) so that |p - c| = s * r(p) (Huber weights).

    ``scale_range`` None keeps s = 1. Otherwise s is fitted and clipped to the range; this
    accepts a plate that is a little smaller or larger than the nominal one.
    """
    c = np.asarray(centre, float).copy()
    s = 1.0
    fit_scale = scale_range is not None
    for _ in range(iterations):
        diff = xy - c
        dist = np.linalg.norm(diff, axis=1)
        ok = dist > 1e-9
        if ok.sum() < 4:
            break
        res = dist[ok] - s * radii[ok]
        jac = -diff[ok] / dist[ok, None]
        if fit_scale:
            jac = np.column_stack((jac, -radii[ok]))
        w = np.where(np.abs(res) < huber, 1.0, huber / np.abs(res))
        h = jac.T @ (jac * w[:, None])
        if abs(np.linalg.det(h)) < 1e-18:
            break
        step = -np.linalg.solve(h, jac.T @ (w * res))
        c += step[:2]
        if fit_scale:
            s = float(np.clip(s + step[2], *scale_range))
        if np.linalg.norm(step[:2]) < 1e-5:
            break
    return c, s


def angular_coverage_deg(xy: np.ndarray, centre, bins: int = 36, min_count: int = 3) -> float:
    """Degrees of the circle around ``centre`` that have at least ``min_count`` points."""
    angles = np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])
    hist = np.bincount(((angles + math.pi) / (2 * math.pi) * bins).astype(int) % bins, None, bins)
    return float((hist >= min_count).sum() * 360.0 / bins)


# ----------------------------------------------------------------------------- polygons


def convex_hull_2d(xy: np.ndarray) -> np.ndarray:
    """Vertices (K, 2) of the convex hull of 2D points, in order."""
    hull = cv2.convexHull(np.ascontiguousarray(xy, np.float32))
    return hull[:, 0, :].astype(float)


def polygon_overlap(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    """Overlap depth of two convex polygons and the direction to move ``q`` out of ``p``.

    Separating axis test on the edge normals of both polygons. Returns (depth, unit
    direction). depth > 0: the polygons overlap, and moving ``q`` by depth along the
    direction separates them (the smallest such move). depth <= 0: they do not overlap;
    -depth is the gap along the direction.
    """
    edges = np.concatenate((np.roll(p, -1, 0) - p, np.roll(q, -1, 0) - q))
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    length = np.linalg.norm(normals, axis=1)
    normals = normals[length > 1e-12] / length[length > 1e-12, None]
    pp = p @ normals.T
    qq = q @ normals.T
    forward = pp.max(0) - qq.min(0)  # move q along +normal by this
    backward = qq.max(0) - pp.min(0)  # move q along -normal by this
    depth = np.minimum(forward, backward)
    k = int(np.argmin(depth))
    direction = normals[k] if forward[k] <= backward[k] else -normals[k]
    return float(depth[k]), direction


# ----------------------------------------------------------------------------- colours


def rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """CIE Lab (L 0..100, a and b about -128..127) of RGB uint8 values with any leading shape."""
    flat = np.ascontiguousarray(np.asarray(rgb_uint8, np.uint8).reshape(-1, 1, 3))
    lab = cv2.cvtColor(flat.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return lab.reshape(*np.asarray(rgb_uint8).shape[:-1], 3)


def color_distance(lab_a: np.ndarray, lab_b: np.ndarray, lightness_weight: float = 0.35):
    """Lab distance with less weight on lightness, so that shade changes count less than hue."""
    d = np.asarray(lab_a, np.float32) - np.asarray(lab_b, np.float32)
    return np.sqrt((lightness_weight * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
