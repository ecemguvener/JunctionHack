"""Perception for the station twin: RGB-D frames of the D415 cameras -> StationState.

Imports only numpy, OpenCV, PyYAML and the standard library (no SciPy, no MuJoCo), so it runs
in the core, MuJoCo and RealSense environments.

Method (``locate`` = one cold ``PerceptionTracker.update``)
 1. Views. Each depth image becomes world points with the camera pose of the scene config
    (OpenCV camera frame). Points outside the workspace box are removed.
 2. Table check. A RANSAC plane fit on the points near z = 0 gives the table plane of each
    camera. The offset and the tilt go to ``cameras``. Small errors are corrected: the camera
    pose is moved so that its table plane is z = 0.
 3. Plates. The KALAS wall has a known radius at each height. A Hough vote with that radius
    finds the plate centres, also for a plate that is partly out of view. A Gauss-Newton fit
    refines each centre. The mean rim colour gives the name ("good" / "defective" / "spare").
 4. Segments. Points above the table (or above a plate floor) go into a 2 mm world-XY grid.
    Connected cells are one cluster. A cluster is split where the colour changes, because two
    bricks that touch usually have different colours. Segments that are too large, too high or
    that float are rejected (arm, can, laptop, box).
 5. Fit, coarse to fine. All hypotheses (segment x part x stable pose x start yaw) get a short
    ICP (point-to-plane, 4 DoF: x, y, z, yaw) and a coarse score in one vectorised pass per
    part. The score is (fraction of segment points on the model) x (fraction of the model
    points that a camera must see and that agree with the depth image), times a colour
    silhouette term. The second term is a render-and-compare check: it uses only model points
    that face a camera and are not hidden by the model itself, and it does not count points
    that something in front hides. Segments with no good coarse hypothesis stop here. The best
    few hypotheses of the other segments get the full ICP and the full score, again in one
    pass per part.
 6. Bricks. Fits that share points: the best one stays. The sim has INSTANCES_PER_PART bodies
    per part: per part, the fits in score order get the copies part-00N, part-00N-2, ...; more
    fits go to ``diagnostics["duplicates"]``. Bricks that go into each other or into a plate
    wall are moved apart in the support plane (smallest move, with a small clearance). Each
    brick gets the name of the nearest palette colour (configs/brick-palette.yaml).

``PerceptionTracker`` keeps data between frames (live mode): the plates (found again after a
time, or when their walls move), the bricks whose segment did not change (only a short local
refinement), and the segments that were rejected and did not change (no new fit). Only new or
changed segments get the full search.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from brickcell.twin import geometry as geo
from brickcell.twin.state import (
    INSTANCES_PER_PART,
    PARTS,
    BrickState,
    PlateState,
    RGBDFrame,
    StationState,
    instance_id,
    matrix_to_quat,
    part_of,
    quat_to_matrix,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "twin.yaml"
ASSETS = ROOT / "assets" / "duplo"

# KALAS plate wall: radius (m) of the middle of the wall at a height z (m). From the inner and
# the outer surface of world.PLATE_PROFILE (tests/test_twin_perception.py checks it).
PLATE_WALL_Z = np.array([0.004, 0.0065, 0.011, 0.0165, 0.0215, 0.0235])
PLATE_WALL_R = np.array([0.0802, 0.08325, 0.08625, 0.0891, 0.09165, 0.09215])
# Inner and outer wall surface (z, r) in m, from world.PLATE_PROFILE (the test checks it).
# A brick in a plate must stay inside the inner surface; a brick on the table outside it.
PLATE_INNER_Z = np.array([0.0031, 0.0040, 0.0065, 0.0110, 0.0165, 0.0215, 0.0235])
PLATE_INNER_R = np.array([0.0750, 0.0790, 0.0820, 0.0850, 0.0880, 0.0905, 0.0913])
PLATE_OUTER_Z = np.array([0.0, 0.0012, 0.0030, 0.0065, 0.0115, 0.0170, 0.0215, 0.0235])
PLATE_OUTER_R = np.array([0.0790, 0.0800, 0.0815, 0.0845, 0.0875, 0.0905, 0.0928, 0.0930])
PLATE_RIM_Z = 0.0235

DEFAULT_CONFIG: dict = {
    # Box in the world frame (m). Points outside it are not used.
    "workspace": {"min": [-0.10, -0.65, -0.04], "max": [0.85, 0.60, 0.25]},
    "depth": {"max_depth": 2.5, "stride": 1},
    "cameras": {
        "reference": "overhead",  # the other cameras are checked against this one
        # A camera whose plate centres differ from the reference camera by more than this (m)
        # is not fused (its pose is wrong). The offsets go to diagnostics["camera_agreement"].
        "max_disagreement": 0.02,
        "exclude": [],  # camera names that are never used
    },
    "table": {
        "fit_band": 0.025,  # use points with |z| < this for the plane fit
        "ransac_threshold": 0.003,
        "ransac_iterations": 200,
        "refine_points": None,  # random points for the plane refinement (None = all points)
        "correct": True,  # move the camera pose so that its table plane is z = 0
        "max_tilt_deg": 5.0,  # do not correct a plane with more tilt or offset than this
        "max_offset": 0.03,
    },
    "plates": {
        "detect": True,
        "z_band": [0.006, 0.026],  # wall points used for the circle vote
        "hough_cell": 0.004,
        "inlier_tol": 0.004,
        "min_inliers": 150,
        "min_arc_deg": 160.0,  # minimum angle of the wall that must be seen (brick walls give less)
        "max_plates": 4,
        "min_arc_unnamed_deg": 270.0,  # a plate with no configured colour needs more wall
        "min_separation": 0.17,
        "scale_range": [0.9, 1.1],  # allowed plate size / nominal KALAS size
        "floor_z": 0.0031,  # top of the floor (sim collision floor); a brick on a plate is here
        "inner_radius": 0.077,  # bricks are segmented inside this radius
        "outer_radius": 0.100,  # points out to this radius are the plate (below rim_top)
        "support_radius": 0.100,  # a brick whose centre is nearer than this is on the plate
        "rim_top": 0.030,
        # Colours: one or more per name, "#rrggbb" or [r, g, b] 0..1. Nearest colour wins.
        # The first values are the real KALAS plates of fixture-001, then the sim colours.
        "colors": {
            "good": ["#17825c", [0.62, 0.86, 0.76]],
            "defective": ["#c28fa1", [0.90, 0.72, 0.78]],
            "spare": ["#0296bf", "#0ab0d4"],
        },
        "max_color_distance": 20.0,  # Lab distance (lightness weight 0.35)
        # A named plate whose floor colour is farther than this from the plate colour is not a
        # plate (for example a ring of blue brick sides round the table). 0 = no check.
        "floor_color_max": 25.0,
    },
    "segment": {
        "cell": 0.002,
        "min_height": 0.005,  # above the table
        "min_height_plate": 0.006,  # above a plate floor
        "floor_color_tol": 14.0,  # in a plate, points with the floor colour are floor (0 = off)
        "min_cells": 60,
        "local_margin": 0.025,  # fit cloud = cluster points within extent / 2 + this
        "color_split": 18.0,  # Lab distance between neighbour cells that splits a cluster
        "max_extent": 0.14,  # longest side of a segment (m)
        "max_top": 0.08,  # highest point of a segment above its support (m)
        "max_bottom": 0.03,  # a segment whose lowest point is higher than this floats
    },
    "fit": {
        "sample_spacing": 0.001,  # model points for the ICP correspondences
        "score_spacing": 0.002,  # model points for the visibility check
        "grid_resolution": 0.001,
        "grid_padding": 0.012,
        "max_points": 600,
        "coarse_iterations": 4,
        "coarse_points": 200,
        "coarse_score_points": 400,  # segment points for the explained term of the coarse score
        # A segment whose best coarse hypothesis has a lower quality stops before the fine
        # stage (it can not reach min_score). 0 = all segments go to the fine stage.
        "coarse_min_quality": 0.28,
        "fine_min_ratio": 0.5,  # fine stage only for coarse scores >= this x the best one
        "score_points": 2000,  # segment points for the explained term of the final score
        "fine_iterations": 20,
        "keep_coarse": 4,
        "huber": 0.002,
        "max_corr": 0.010,
        "inlier": 0.003,
        "z_prior": 0.05,
        "height_tol_above": 0.006,
        "height_tol_below": 0.010,
        "extent_tol": 0.012,
        "visibility_bin_px": 2,
        "depth_tol": 0.004,
        "depth_tol_z2": 0.003,  # extra depth tolerance per m^2 of depth (stereo noise)
        "snap_z": True,
        "keep_better_start": True,  # keep the start pose if the ICP makes the score worse
        "snap_z_tol": 0.006,
        "free_refine": False,
        "free_min_gain": 0.05,
        "free_min_angle_deg": 4.0,
        "min_explained": 0.6,
        "min_coverage": 0.55,
        "min_score": 0.4,  # on explained x coverage
        # Colour silhouette: score *= IoU(model outline, brick-colour mask) ** weight (0 = off).
        "silhouette_weight": 0.5,
        "min_silhouette_iou": 0.2,
        "silhouette_color_tol": 22.0,
        "silhouette_margin_px": 25,
        "max_residual_mm": 4.0,
        "min_visible": 60,
        "same_brick_m": 0.015,  # two fits with nearer footprint centres are one brick
        # Fits are chosen jointly: segments in score order, each takes its best accepted
        # hypothesis that goes less than this into a brick that is already chosen.
        "max_overlap_m": 0.004,
        # No overlap. The penetration of bricks into each other and into the plate walls is
        # always measured (diagnostics["overlap"]). With "separate", bricks are also moved
        # apart in x, y until each brick is min_clearance_m from each other brick and each
        # plate wall. Off by default: sim_sync has the same guard.
        "separate": False,
        "min_clearance_m": 0.0005,
        "separate_rounds": 30,
    },
    # PerceptionTracker (live mode).
    "tracker": {
        "plate_period_s": 10.0,  # find the plates again after this time (s)
        "plate_wall_fraction": 0.6,  # ... or when less of a plate wall is where it was
        "full_period_s": 30.0,  # a full search of all segments after this time (0 = never)
        "match_iou": 0.6,  # grid-cell IoU of a segment and the last segment: "not changed"
        "match_top": 0.003,  # m; and the top height changed less than this
        "match_color": 12.0,  # and the colour changed less than this (Lab distance)
        "refine_iterations": 6,  # ICP iterations of the local refinement
        "min_quality_ratio": 0.85,  # the refinement must keep this fraction of the quality
        "id_distance": 0.02,  # m; a new fit this near a lost brick of the part gets its id
    },
    "palette": None,  # brick palette for the colour names; None = configs/brick-palette.yaml
    "cache_dir": "~/.cache/brickcell/twin-perception",
}


def _merge(base: dict, extra: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_perception_config(path: str | Path | None = None) -> dict:
    """The ``perception:`` section of configs/twin.yaml, with the defaults for missing keys."""
    path = Path(path) if path is not None else CONFIG_PATH
    section = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text()) or {}
        section = raw.get("perception") or {}
    unknown = sorted(set(section) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"perception config {path}: unknown keys {unknown}")
    return _merge(DEFAULT_CONFIG, section)


# ----------------------------------------------------------------------------- part models


@dataclass
class PoseHypothesis:
    pose_id: str
    rotation: np.ndarray  # mesh frame -> world, yaw 0
    rest_height: float
    top: float  # height of the highest point above the support
    long: float  # footprint along world X (the long side)
    short: float
    centre_xy: np.ndarray  # footprint centre minus mesh origin, world XY, yaw 0
    yaw_period: float  # yaws that differ by this give the same shape
    probability: float


@dataclass
class PartModel:
    object_id: str
    part: str
    mesh: str
    triangles: np.ndarray
    samples: np.ndarray
    normals: np.ndarray
    grid: geo.ClosestPointGrid
    score_samples: np.ndarray
    symmetries: list[np.ndarray]
    poses: list[PoseHypothesis] = field(default_factory=list)
    hull: np.ndarray | None = None  # mesh vertices on the convex hull (approximately)
    centre: np.ndarray | None = None  # bounding box centre, mesh frame


_MODELS: dict[tuple, dict[str, PartModel]] = {}


def _yaw_period(rotation: np.ndarray, symmetries: list[np.ndarray]) -> float:
    """Smallest yaw > 0 that maps the posed part onto itself (2 pi if there is none)."""
    best = 2 * math.pi
    for s in symmetries[1:]:
        m = rotation @ s @ rotation.T
        if m[2, 2] > 1 - 1e-6:
            angle = abs(math.atan2(m[1, 0], m[0, 0]))
            if angle > 1e-3:
                best = min(best, angle)
    return best


def _hull_vertices(triangles: np.ndarray, directions: int = 1500) -> np.ndarray:
    """Mesh vertices that are extreme along one of many directions (the convex hull vertices)."""
    vertices = np.unique(triangles.reshape(-1, 3).round(6), axis=0)
    k = np.arange(directions) + 0.5
    phi = np.arccos(1 - 2 * k / directions)
    theta = math.pi * (1 + 5**0.5) * k
    dirs = np.column_stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)))
    return vertices[np.unique(np.argmax(vertices @ dirs.T, axis=0))]


def _grid_for(mesh: str, samples, cfg_fit: dict, cache_dir: str | None) -> geo.ClosestPointGrid:
    res, pad = cfg_fit["grid_resolution"], cfg_fit["grid_padding"]
    key = f"{mesh}-{geo.file_digest(ASSETS / (mesh + '.stl'))}-{cfg_fit['sample_spacing']}-{res}-{pad}-v1"
    path = None
    if cache_dir:
        path = Path(os.path.expanduser(cache_dir)) / f"{key}.npz"
        if path.is_file():
            try:
                data = np.load(path)
                return geo.ClosestPointGrid(
                    data["origin"], float(data["resolution"]), data["index"]
                )
            except (OSError, ValueError, KeyError):
                pass
    grid = geo.ClosestPointGrid.build(samples, res, pad)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp, origin=grid.origin, resolution=grid.resolution, index=grid.index
            )
            tmp.replace(path)
        except OSError:
            pass
    return grid


def part_models(config: dict | None = None) -> dict[str, PartModel]:
    """Sampled meshes, closest-point grids and stable poses of the four parts (cached)."""
    cfg = config or load_perception_config()
    f = cfg["fit"]
    key = (f["sample_spacing"], f["score_spacing"], f["grid_resolution"], f["grid_padding"])
    if key in _MODELS:
        return _MODELS[key]
    stable = json.loads((ASSETS / "stable_poses.json").read_text())["bricks"]
    models = {}
    for object_id, part in PARTS.items():
        triangles = geo.read_stl(ASSETS / f"{part.mesh}.stl")
        samples, normals = geo.sample_surface(triangles, f["sample_spacing"], seed=0)
        score_samples, _ = geo.sample_surface(triangles, f["score_spacing"], seed=1)
        symmetries = geo.rotational_symmetries(samples)
        grid = _grid_for(part.mesh, samples, f, cfg.get("cache_dir"))
        model = PartModel(
            object_id,
            part.part,
            part.mesh,
            triangles,
            samples,
            normals,
            grid,
            score_samples,
            symmetries,
            hull=_hull_vertices(triangles),
            centre=(samples.min(0) + samples.max(0)) / 2,
        )
        entries = sorted(stable[object_id]["poses"], key=lambda p: -p["probability"])
        for entry in entries:
            r = quat_to_matrix(entry["quat"])
            # Same shape as a pose that is already in the list (a symmetry and a yaw)?
            if any(
                (r @ s @ h.rotation.T)[2, 2] > 1 - 1e-6 for h in model.poses for s in symmetries
            ):
                continue
            pts = samples @ r.T
            lo, hi = pts.min(0), pts.max(0)
            model.poses.append(
                PoseHypothesis(
                    pose_id=entry["id"],
                    rotation=r,
                    rest_height=float(entry["rest_height"]),
                    top=float(hi[2] - lo[2]),
                    long=float(hi[0] - lo[0]),
                    short=float(hi[1] - lo[1]),
                    centre_xy=(lo[:2] + hi[:2]) / 2,
                    yaw_period=_yaw_period(r, symmetries),
                    probability=float(entry["probability"]),
                )
            )
        models[object_id] = model
    _MODELS[key] = models
    return models


# ----------------------------------------------------------------------------- views


@dataclass
class View:
    name: str
    frame: RGBDFrame
    rotation: np.ndarray  # camera -> world (OpenCV camera frame), after the table correction
    position: np.ndarray
    points: np.ndarray  # (N, 3) float32 world points in the workspace
    pixels: np.ndarray  # (N,) flat pixel index
    rgb: np.ndarray  # (N, 3) uint8
    info: dict
    _lab: np.ndarray | None = None

    def lab_image(self) -> np.ndarray:
        """CIE Lab of the full colour image (computed once)."""
        if self._lab is None:
            self._lab = geo.rgb_to_lab(self.frame.rgb)
        return self._lab


_RAYS: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _ray_grid(shape, fx, fy, cx, cy, stride) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame rays (x/z, y/z, 1) as float32 (N, 3) and the flat pixel index (cached)."""
    key = (shape, fx, fy, cx, cy, stride)
    hit = _RAYS.get(key)
    if hit is None:
        h, w = shape
        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        rays = np.stack(((us - cx) / fx, (vs - cy) / fy, np.ones(us.shape)), -1)
        hit = (rays.reshape(-1, 3).astype(np.float32), (vs * w + us).reshape(-1).astype(np.int64))
        if len(_RAYS) > 16:
            _RAYS.clear()
        _RAYS[key] = hit
    return hit


def _take(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Rows ``idx`` of ``a`` (np.take is much faster than fancy indexing of (N, 3) arrays)."""
    return np.take(a, idx, axis=0)


def _prepare_view(frame: RGBDFrame, camera, cfg: dict, seed: int) -> View:
    rotation = np.asarray(camera.rotation_cv(), float)
    position = np.asarray(camera.pos, float)
    d = cfg["depth"]
    stride = int(d["stride"])
    depth = frame.depth_m
    full_valid = (depth > 0) & (depth < d["max_depth"])  # NaN compares False
    valid_fraction = float(np.count_nonzero(full_valid)) / full_valid.size
    rays, pix = _ray_grid(depth.shape, frame.fx, frame.fy, frame.cx, frame.cy, stride)
    z = np.ascontiguousarray(depth[::stride, ::stride], np.float32).reshape(-1)
    cam = rays * z[:, None]  # all pixels; the invalid ones are removed below
    world = cam @ rotation.T.astype(np.float32) + position.astype(np.float32)
    lo, hi = np.asarray(cfg["workspace"]["min"]), np.asarray(cfg["workspace"]["max"])
    inside = (
        full_valid[::stride, ::stride].reshape(-1)
        & (world[:, 0] >= lo[0])
        & (world[:, 0] <= hi[0])
        & (world[:, 1] >= lo[1])
        & (world[:, 1] <= hi[1])
        & (world[:, 2] >= lo[2])
        & (world[:, 2] <= hi[2])
    )
    idx = np.flatnonzero(inside)
    world, cam, pixels = _take(world, idx), _take(cam, idx), pix[idx]
    info: dict = {
        "serial": frame.serial,
        "timestamp": frame.timestamp,
        "synthetic": frame.synthetic,
        "valid_depth_fraction": round(valid_fraction, 4),
        "points_in_workspace": int(len(world)),
    }
    t = cfg["table"]
    band_pts = _take(world, np.flatnonzero(np.abs(world[:, 2]) < t["fit_band"]))
    plane = geo.fit_plane_ransac(
        band_pts, t["ransac_threshold"], t["ransac_iterations"], seed, t.get("refine_points")
    )
    if plane is not None:
        n, dist, inliers = plane
        sel = _take(band_pts, np.flatnonzero(inliers))
        if len(sel) > 20000:
            sel = _take(sel, np.random.default_rng(seed).integers(0, len(sel), 20000))
        pivot = np.median(sel, axis=0).astype(float)
        offset = float(-(n[0] * pivot[0] + n[1] * pivot[1] + dist) / n[2])  # plane z at the pivot
        tilt = math.degrees(math.acos(min(1.0, float(n[2]))))
        corrected = bool(
            t["correct"] and tilt <= t["max_tilt_deg"] and abs(offset) <= t["max_offset"]
        )
        info["table_plane"] = {
            "normal": [round(float(v), 5) for v in n],
            "offset_mm": round(offset * 1000, 2),
            "tilt_deg": round(tilt, 3),
            "pivot": [round(float(v), 4) for v in pivot],
            "inliers": int(inliers.sum()),
            "inlier_fraction": round(float(inliers.mean()), 3),
            "corrected": corrected,
        }
        if corrected:
            r_c, t_c = geo.plane_to_z0(n, dist, pivot)
            rotation = r_c @ rotation
            position = r_c @ position + t_c
            world = cam @ rotation.T.astype(np.float32) + position.astype(np.float32)
            info["corrected_pose"] = {
                "pos": [round(float(v), 5) for v in position],
                "rotation_cv": [[round(float(v), 6) for v in row] for row in rotation],
            }
    else:
        info["table_plane"] = None
    rgb = _take(frame.rgb.reshape(-1, 3), pixels)
    return View(frame.camera, frame, rotation, position, world, pixels, rgb, info)


# ----------------------------------------------------------------------------- plates


def _plate_wall_radius(z: np.ndarray) -> np.ndarray:
    return np.interp(z, PLATE_WALL_Z, PLATE_WALL_R)


_COLOR_LABS: dict[str, list[np.ndarray]] = {}


def _color_list(values) -> list[np.ndarray]:
    key = repr(values)
    if key not in _COLOR_LABS:
        _COLOR_LABS[key] = _color_list_uncached(values)
    return _COLOR_LABS[key]


def _color_list_uncached(values) -> list[np.ndarray]:
    out = []
    for v in (
        values
        if isinstance(values, (list, tuple)) and values and not isinstance(values[0], (int, float))
        else [values]
    ):
        if isinstance(v, str):
            text = v.strip().lstrip("#")
            rgb = [int(text[i : i + 2], 16) for i in (0, 2, 4)]
        else:
            rgb = [round(255 * float(c)) for c in v]
        out.append(geo.rgb_to_lab(np.array([rgb], np.uint8))[0])
    return out


@dataclass
class PlateFit:
    centre: np.ndarray
    inliers: int
    arc_deg: float
    color: np.ndarray  # RGB 0..1
    name: str | None
    color_distance: float
    confidence: float
    scale: float = 1.0  # fitted radius / nominal KALAS radius
    floor_lab: np.ndarray | None = None  # median Lab colour of the floor (found once)


def _wall_points(views: list[View], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Points in the wall height band (at most 15000, a fixed random choice) and their RGB."""
    p = cfg["plates"]
    pts = np.concatenate([v.points for v in views])
    rgb = np.concatenate([v.rgb for v in views])
    band = np.flatnonzero((pts[:, 2] > p["z_band"][0]) & (pts[:, 2] < p["z_band"][1]))
    if len(band) > 15000:
        band = band[np.random.default_rng(0).choice(len(band), 15000, replace=False)]
    return _take(pts, band).astype(float), _take(rgb, band)


def _find_plates(views: list[View], cfg: dict) -> tuple[list[PlateFit], list[dict]]:
    p = cfg["plates"]
    if not p["detect"]:
        return [], []
    pts, rgb = _wall_points(views, cfg)
    rejected: list[dict] = []
    if len(pts) < p["min_inliers"]:
        return [], rejected
    xy, radii = pts[:, :2], _plate_wall_radius(pts[:, 2])
    lo = np.asarray(cfg["workspace"]["min"][:2]) - 0.1
    hi = np.asarray(cfg["workspace"]["max"][:2]) + 0.1
    cell = p["hough_cell"]
    acc, origin = geo.circle_hough(xy, radii, lo, hi, cell, angles=32)
    acc = cv2.GaussianBlur(acc, (5, 5), 1.0)
    colors = {name: _color_list(values) for name, values in p["colors"].items()}
    all_pts = all_rgb = None  # all points, made only when a floor check needs them
    plates: list[PlateFit] = []
    for _ in range(p["max_plates"] * 6):
        if len(plates) >= p["max_plates"]:
            break
        iy, ix = np.unravel_index(int(np.argmax(acc)), acc.shape)
        if acc[iy, ix] <= 0:
            break
        centre = origin + cell * np.array([ix, iy], float)
        # Suppress a small area round this peak before any "continue". An accepted plate
        # suppresses a larger area round its fitted centre (below).
        sup = int(round(0.03 / cell))
        acc[max(0, iy - sup) : iy + sup + 1, max(0, ix - sup) : ix + sup + 1] = 0
        near = np.abs(np.linalg.norm(xy - centre, axis=1) - radii) < 3 * p["inlier_tol"]
        n_near = int(near.sum())
        if n_near < p["min_inliers"]:
            if len(plates) == 0 and n_near < 10:
                break
            rejected.append(
                {
                    "centre": centre.round(4).tolist(),
                    "reason": "too few wall points",
                    "inliers": n_near,
                }
            )
            continue
        # Quick check before the fit: the wide band round the peak must already hold most of
        # the wall arc (the fit makes the band narrower, so the arc does not become larger).
        arc_near = geo.angular_coverage_deg(xy[near], centre)
        if arc_near < 0.8 * p["min_arc_deg"]:
            rejected.append(
                {
                    "centre": centre.round(4).tolist(),
                    "inliers": n_near,
                    "arc_deg": arc_near,
                    "reason": "wall not complete enough",
                }
            )
            continue
        centre, scale = geo.fit_circle_known_radius(
            xy[near], radii[near], centre, scale_range=tuple(p["scale_range"])
        )
        inl = np.abs(np.linalg.norm(xy - centre, axis=1) - scale * radii) < p["inlier_tol"]
        if inl.sum() < p["min_inliers"] or (
            geo.angular_coverage_deg(xy[inl], centre) < 0.9 * p["min_arc_deg"]
        ):
            rejected.append(
                {
                    "centre": centre.round(4).tolist(),
                    "inliers": int(inl.sum()),
                    "reason": "wall not complete enough",
                }
            )
            continue
        centre, scale = geo.fit_circle_known_radius(
            xy[inl], radii[inl], centre, scale_range=tuple(p["scale_range"])
        )
        inl = np.abs(np.linalg.norm(xy - centre, axis=1) - scale * radii) < p["inlier_tol"]
        arc = geo.angular_coverage_deg(xy[inl], centre)
        entry = {
            "centre": centre.round(4).tolist(),
            "inliers": int(inl.sum()),
            "arc_deg": arc,
            "scale": round(scale, 3),
        }
        if inl.sum() < p["min_inliers"] or arc < p["min_arc_deg"]:
            rejected.append({**entry, "reason": "wall not complete enough"})
            continue
        if any(np.linalg.norm(centre - q.centre) < p["min_separation"] for q in plates):
            continue
        med = np.median(rgb[inl], axis=0)
        lab = geo.rgb_to_lab(med.astype(np.uint8)[None])[0]
        best_name, best_d = None, float("inf")
        for name, labs in colors.items():
            for c in labs:
                dist = float(geo.color_distance(lab, c))
                if dist < best_d:
                    best_name, best_d = name, dist
        name = best_name if best_d <= p["max_color_distance"] else None
        if name is None and arc < p["min_arc_unnamed_deg"]:
            rejected.append({**entry, "reason": "no plate colour and wall not complete enough"})
            continue
        if name is not None and p["floor_color_max"] > 0:
            # A real plate has a floor of its own colour inside the rim. A ring of brick sides
            # with the plate colour has the table or other bricks inside.
            if all_pts is None:
                all_pts = np.concatenate([v.points for v in views])
                all_rgb = np.concatenate([v.rgb for v in views])
            probe = PlateFit(centre, int(inl.sum()), arc, med / 255.0, name, best_d, 0.0, scale)
            floor_lab = _floor_color(all_pts, all_rgb, probe, p)
            if floor_lab is not None:
                floor_d = min(float(geo.color_distance(floor_lab, c)) for c in colors[name])
                if floor_d > p["floor_color_max"]:
                    rejected.append(
                        {
                            **entry,
                            "name": name,
                            "floor_color_distance": round(floor_d, 1),
                            "reason": "the floor inside the rim does not have the plate colour",
                        }
                    )
                    continue
        jx, jy = np.rint((centre - origin) / cell).astype(int)
        sup = int(round(p["min_separation"] / cell))
        acc[max(0, jy - sup) : jy + sup + 1, max(0, jx - sup) : jx + sup + 1] = 0
        conf = min(1.0, arc / 180.0) * (1.0 if name else 0.5)
        plates.append(PlateFit(centre, int(inl.sum()), arc, med / 255.0, name, best_d, conf, scale))
    return plates, rejected


def _plates_still_there(views: list[View], plates: list[PlateFit], cfg: dict, fraction: float):
    """True if the wall of each known plate is still at its place (tracker)."""
    p = cfg["plates"]
    pts, _ = _wall_points(views, cfg)
    if len(pts) == 0:
        return not plates
    xy, radii = pts[:, :2], _plate_wall_radius(pts[:, 2])
    for plate in plates:
        inl = (
            np.abs(np.linalg.norm(xy - plate.centre, axis=1) - plate.scale * radii)
            < p["inlier_tol"]
        )
        if inl.sum() < max(p["min_inliers"], fraction * plate.inliers):
            return False
        if geo.angular_coverage_deg(xy[inl], plate.centre) < min(
            p["min_arc_deg"], fraction * plate.arc_deg
        ):
            return False
    return True


def _camera_centre(view: View, plate: PlateFit, cfg: dict) -> np.ndarray | None:
    """Centre of a known plate from the wall points of one camera only (or None)."""
    p = cfg["plates"]
    pts, _ = _wall_points([view], cfg)
    if len(pts) == 0:
        return None
    xy, radii = pts[:, :2], _plate_wall_radius(pts[:, 2]) * plate.scale
    near = np.abs(np.linalg.norm(xy - plate.centre, axis=1) - radii) < 3 * p["inlier_tol"]
    if near.sum() < p["min_inliers"] / 3 or geo.angular_coverage_deg(xy[near], plate.centre) < 90:
        return None
    centre, _ = geo.fit_circle_known_radius(xy[near], radii[near], plate.centre)
    inl = np.abs(np.linalg.norm(xy - centre, axis=1) - radii) < p["inlier_tol"]
    if inl.sum() < p["min_inliers"] / 3 or geo.angular_coverage_deg(xy[inl], centre) < 90:
        return None
    centre, _ = geo.fit_circle_known_radius(xy[inl], radii[inl], centre)
    return centre


def _detect_plates(
    all_views: list[View], cfg: dict
) -> tuple[list[View], list[PlateFit], list[dict], dict]:
    """Plates from the fused cameras, and the check that the cameras agree.

    A camera with a wrong pose puts the plates at a different place. Such a camera is not
    fused (the bricks would get a mix of two positions). For each plate, each camera fits the
    centre with its own wall points; the median offset to the reference camera must be less
    than ``cameras.max_disagreement``. Returns (views to use, plates, rejected, report).
    """
    c = cfg["cameras"]
    report: dict = {"reference": c["reference"], "cameras": {}}
    views = [v for v in all_views if v.name not in c["exclude"]]
    plates, rejected = _find_plates(views, cfg)
    ref = next((v for v in views if v.name == c["reference"]), None)
    if ref is None or len(views) < 2:
        report["note"] = "no check: the reference camera or a second camera is missing"
        return views, plates, rejected, report
    ref_centres = [_camera_centre(ref, plate, cfg) for plate in plates]
    used = [ref]
    for v in views:
        if v is ref:
            continue
        offsets = []
        for plate, rc in zip(plates, ref_centres):
            if rc is None:
                continue
            own = _camera_centre(v, plate, cfg)
            if own is not None:
                offsets.append((own - rc).round(4).tolist())
        entry: dict = {"plate_offsets": offsets}
        if offsets:
            median = np.median(np.asarray(offsets), axis=0)
            entry["median_offset_mm"] = (median * 1000).round(1).tolist()
            entry["used"] = bool(np.linalg.norm(median) <= c["max_disagreement"])
            if not entry["used"]:
                entry["reason"] = (
                    f"plates differ by {np.linalg.norm(median) * 1000:.0f} mm from the "
                    f"'{ref.name}' camera; check the camera pose"
                )
        else:
            entry["used"] = True
            entry["note"] = "no plate seen by both cameras; not checked"
        report["cameras"][v.name] = entry
        if entry["used"]:
            used.append(v)
    if len(used) < len(views):  # find the plates again without the cameras that disagree
        plates, rejected = _find_plates(used, cfg)
    return used, plates, rejected, report


# ----------------------------------------------------------------------------- segments


@dataclass
class Segment:
    points: np.ndarray  # (N, 3) world, float64
    rgb: np.ndarray  # (N, 3) uint8
    view_index: np.ndarray  # (N,) index into views
    cell_ids: np.ndarray  # sorted ids of the 2 mm workspace grid cells of the segment
    support_z: float
    plate: int | None  # index into the plates, or None
    cluster: int
    local_points: np.ndarray  # cluster points near the segment (the fit cloud)
    local_view_index: np.ndarray
    lab: np.ndarray  # median Lab colour of the points
    top: float  # 98th percentile height above the support
    info: dict = field(default_factory=dict)


def _quantile(values: np.ndarray, q: float) -> float:
    """The q-quantile of the values (nearest rank; faster than np.percentile)."""
    k = int(round(q * (len(values) - 1)))
    return float(np.partition(values, k)[k])


def _groups(labels: np.ndarray) -> dict[int, np.ndarray]:
    """{label: indices} for an integer label array (one sort, not one pass per label)."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    ends = np.r_[starts[1:], len(order)]
    return {int(sorted_labels[a]): order[a:b] for a, b in zip(starts, ends)}


def _floor_color(all_pts: np.ndarray, all_rgb: np.ndarray, plate: PlateFit, p: dict):
    """Median Lab colour of the plate floor (points inside the floor radius, near floor_z)."""
    sub = np.arange(0, len(all_pts), 2)
    xy = all_pts[sub, :2]
    near = sub[
        (np.abs(xy[:, 0] - plate.centre[0]) < 0.08) & (np.abs(xy[:, 1] - plate.centre[1]) < 0.08)
    ]
    sp = _take(all_pts, near)
    floor = (
        np.linalg.norm(sp[:, :2] - plate.centre.astype(np.float32), axis=1) / plate.scale
        < p["inner_radius"]
    ) & (sp[:, 2] < p["floor_z"] + 0.003)
    if floor.sum() <= 25:
        return None
    return np.median(geo.rgb_to_lab(_take(all_rgb, near[floor])), axis=0)


def _segments(
    views: list[View], plates: list[PlateFit], cfg: dict
) -> tuple[list[Segment], list[dict], dict]:
    s, p = cfg["segment"], cfg["plates"]
    all_pts = np.concatenate([v.points for v in views])
    all_rgb = np.concatenate([v.rgb for v in views])
    n_view = [len(v.points) for v in views]
    # Cheap height test first: Lab and the plate tests only for points that can be a brick.
    lowest = s["min_height"]
    if plates:
        lowest = min(lowest, p["floor_z"] + s["min_height_plate"])
    cand = np.flatnonzero(all_pts[:, 2] > lowest)
    pts = _take(all_pts, cand)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    support = np.zeros(len(pts), np.float32)
    plate_of = np.full(len(pts), -1, np.int16)
    keep = np.ones(len(pts), bool)
    for k, plate in enumerate(plates):
        dx = x - np.float32(plate.centre[0])
        dy = y - np.float32(plate.centre[1])
        r2 = (dx * dx + dy * dy) / np.float32(plate.scale**2)
        inner = r2 < p["inner_radius"] ** 2
        support[inner] = p["floor_z"]
        plate_of[inner] = k
        keep &= ~(
            (r2 >= p["inner_radius"] ** 2) & (r2 <= p["outer_radius"] ** 2) & (z < p["rim_top"])
        )
    above = np.where(plate_of >= 0, s["min_height_plate"], s["min_height"]).astype(np.float32)
    keep &= z > support + above
    sel = np.flatnonzero(keep)
    pts, support, plate_of = _take(pts, sel), support[sel], plate_of[sel]
    cand = cand[sel]
    rgb = _take(all_rgb, cand)
    lab = geo.rgb_to_lab(rgb)
    # Depth edges are soft: the floor next to a brick can look raised. Points with the colour
    # of the plate floor are floor, not brick.
    if plates and s["floor_color_tol"] > 0:
        drop = np.zeros(len(pts), bool)
        for k, plate in enumerate(plates):
            if plate.floor_lab is None:
                plate.floor_lab = _floor_color(all_pts, all_rgb, plate, p)
            if plate.floor_lab is None:
                continue
            test = np.flatnonzero(plate_of == k)
            drop[test[geo.color_distance(lab[test], plate.floor_lab) < s["floor_color_tol"]]] = True
        sel = np.flatnonzero(~drop)
        pts, support, plate_of, rgb, lab, cand = (
            _take(pts, sel),
            support[sel],
            plate_of[sel],
            _take(rgb, sel),
            _take(lab, sel),
            cand[sel],
        )
    vidx = np.searchsorted(np.cumsum(n_view), cand, side="right").astype(np.int16)
    stats = {"points_above_support": int(len(pts))}
    if len(pts) == 0:
        return [], [], stats
    cell = s["cell"]
    lo = np.asarray(cfg["workspace"]["min"][:2])
    hi = np.asarray(cfg["workspace"]["max"][:2])
    n_global = int(math.ceil((hi[0] - lo[0]) / cell)) + 4
    ij_global = np.floor((pts[:, :2] - lo) / cell).astype(np.int64)
    g0 = ij_global.min(0) - 2
    ij = ij_global - g0
    nx = int(ij[:, 0].max()) + 3
    ny = int(ij[:, 1].max()) + 3
    flat = ij[:, 1] * nx + ij[:, 0]
    count = np.bincount(flat, minlength=nx * ny).reshape(ny, nx).astype(np.float32)
    occ = (count > 0).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n_clusters, cluster_map = cv2.connectedComponents(occ, connectivity=8)
    # Mean colour per cell over a 3x3 window (less noise, but a colour edge stays sharp).
    sm_count = cv2.boxFilter(count, -1, (3, 3), normalize=False)
    mean_lab = np.empty((ny * nx, 3), np.float32)
    inv = 1.0 / np.maximum(sm_count, 1e-6).reshape(-1)
    for c in range(3):
        lab_sum = np.bincount(flat, lab[:, c], minlength=nx * ny).reshape(ny, nx)
        sm = cv2.boxFilter(lab_sum.astype(np.float32), -1, (3, 3), normalize=False)
        mean_lab[:, c] = sm.reshape(-1) * inv
    has = count.reshape(-1) > 0
    occupied = np.flatnonzero(has)
    # A cell is on a boundary if its colour differs from a cell 1 or 2 steps away. The grid
    # has a margin of 2 cells, so the neighbour index is always in the grid.
    boundary = np.zeros(nx * ny, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)):
        nb = occupied + dy * nx + dx
        both = has[nb]
        a, b = occupied[both], nb[both]
        diff = geo.color_distance(mean_lab[a], mean_lab[b]) > s["color_split"]
        boundary[a[diff]] = True
        boundary[b[diff]] = True
    core = (has & ~boundary).reshape(ny, nx).astype(np.uint8)
    n_core, seg_map = cv2.connectedComponents(core, connectivity=4)
    seg_flat = seg_map.reshape(-1).astype(np.int32)
    # Boundary cells and closed gaps go to the neighbour segment with the nearest colour.
    core_cells = np.flatnonzero(seg_flat > 0)
    seg_n = np.bincount(seg_flat[core_cells], minlength=n_core).astype(np.float32)
    seg_lab = np.zeros((n_core, 3), np.float32)
    for c in range(3):
        seg_lab[:, c] = np.bincount(seg_flat[core_cells], mean_lab[core_cells, c], minlength=n_core)
    seg_lab /= np.maximum(seg_n, 1)[:, None]
    occ_flat = occ.reshape(-1) > 0
    for _ in range(4):
        todo = np.flatnonzero((seg_flat == 0) & occ_flat)
        if not len(todo):
            break
        best = np.zeros(len(todo), np.int32)
        best_d = np.full(len(todo), np.inf, np.float32)
        todo_lab = mean_lab[todo]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cand_seg = seg_flat[todo + dy * nx + dx]
                d = geo.color_distance(todo_lab, seg_lab[cand_seg])
                better = (cand_seg > 0) & (d < best_d)
                best[better] = cand_seg[better]
                best_d[better] = d[better]
        assign = best > 0
        seg_flat[todo[assign]] = best[assign]
    point_seg = seg_flat[flat]
    point_cluster = cluster_map.reshape(-1)[flat]
    # Cells with points per segment; small segments (noise, leftovers) are not reported.
    n_cells = np.bincount(seg_flat[occupied], minlength=int(seg_flat.max()) + 1)
    big = n_cells >= s["min_cells"]
    big[0] = False
    point_seg = np.where(big[point_seg], point_seg, 0)
    seg_groups = _groups(point_seg)
    cluster_groups = _groups(point_cluster)
    segments: list[Segment] = []
    rejected: list[dict] = []
    for label, sel in seg_groups.items():
        if label == 0:
            continue
        cells = np.unique(flat[sel])
        cx, cy = cells % nx, cells // nx
        sp = _take(pts, sel).astype(float)
        centre = sp[:, :2].mean(0)
        (_, _), (w, h), _ = cv2.minAreaRect((np.column_stack((cx, cy)) * cell).astype(np.float32))
        extent = float(max(w, h))
        # Cluster points near the segment: the same brick can have more than one colour
        # segment (shade), so the fit and the support test use these points.
        cluster = int(np.bincount(point_cluster[sel]).argmax())
        near = cluster_groups[cluster]
        near_pts = _take(pts, near)
        dxy = near_pts[:, :2] - centre.astype(np.float32)
        close = np.flatnonzero(
            np.einsum("ij,ij->i", dxy, dxy) < (extent / 2 + s["local_margin"]) ** 2
        )
        near, near_pts = near[close], _take(near_pts, close)
        sup = float(np.median(support[sel]))
        top = _quantile(sp[:, 2], 0.98) - sup
        bottom = _quantile(near_pts[:, 2], 0.02) - sup
        seg_rgb = _take(rgb, sel)
        info = {
            "centre": centre.round(4).tolist(),
            "cells": int(len(cells)),
            "points": int(len(sel)),
            "top_mm": round(top * 1000, 1),
            "bottom_mm": round(bottom * 1000, 1),
            "extent_mm": round(extent * 1000, 1),
            "color": (np.median(seg_rgb[:: max(1, len(sel) // 500)], axis=0) / 255)
            .round(3)
            .tolist(),
        }
        reason = None
        if extent > s["max_extent"]:
            reason = "segment too large for a brick"
        elif top > s["max_top"]:
            reason = "segment too high for a brick"
        elif bottom > s["max_bottom"]:
            reason = "segment does not stand on a support"
        if reason:
            rejected.append({**info, "reason": reason})
            continue
        plates_here = plate_of[sel]
        plate = (
            int(np.bincount(plates_here[plates_here >= 0]).argmax())
            if (plates_here >= 0).mean() > 0.5
            else None
        )
        cell_ids = np.sort((cy + g0[1]) * n_global + (cx + g0[0]))
        segments.append(
            Segment(
                sp,
                seg_rgb,
                vidx[sel],
                cell_ids,
                sup,
                plate,
                cluster,
                near_pts.astype(float),
                vidx[near],
                np.median(_take(lab, sel), axis=0),
                top,
                info,
            )
        )
    stats["clusters"] = int(n_clusters - 1)
    stats["segments"] = int(len(segments))
    return segments, rejected, stats


# ----------------------------------------------------------------------------- fitting


@dataclass
class Fit:
    object_id: str  # part id (the copy id is given later)
    pose: PoseHypothesis | None  # None after a free 6-DoF fit
    rotation: np.ndarray  # mesh -> world
    position: np.ndarray  # mesh origin in the world
    explained: float = 0.0
    coverage: float = 0.0
    visible: int = 0
    score: float = 0.0
    residual_mm: float | None = None
    rms_mm: float = 0.0
    cameras: list[str] = field(default_factory=list)
    inlier_rgb: np.ndarray | None = None  # colours of the segment points on the model
    segment: Segment | None = None
    quality: float = 0.0  # explained x coverage: used for acceptance and confidence
    iou: float | None = None  # colour silhouette IoU: used only to rank the hypotheses
    track_id: str | None = None  # brick copy id from the last frame (tracker)
    alternatives: list = field(default_factory=list)  # other fine fits of the segment


@dataclass
class _Job:
    """Fit data of one segment."""

    seg: Segment
    cloud: np.ndarray  # ICP points (at most max_points), random order
    coarse_pts: np.ndarray  # segment points for the coarse explained term
    fine_pts: np.ndarray  # segment points for the final explained term
    fine_rgb: np.ndarray
    which: list[int]  # views that see the segment
    rect: tuple
    masks: dict | None = None


def _subsample(points: np.ndarray, n: int, rng) -> np.ndarray:
    if len(points) > n:
        return points[rng.choice(len(points), n, replace=False)]
    return rng.permutation(points)


def _make_job(seg: Segment, f: dict, rng) -> _Job:
    cloud = _subsample(seg.local_points, f["max_points"], rng)
    coarse = _subsample(seg.points, f["coarse_score_points"], rng)
    if len(seg.points) > f["score_points"]:
        k = rng.choice(len(seg.points), f["score_points"], replace=False)
        fine, fine_rgb = seg.points[k], seg.rgb[k]
    else:
        fine, fine_rgb = seg.points, seg.rgb
    which = sorted(set(np.unique(seg.local_view_index).tolist()))
    rect = cv2.minAreaRect(seg.points[:, :2].astype(np.float32))
    return _Job(seg, cloud, coarse, fine, fine_rgb, which, rect)


def _pad(arrays: list[np.ndarray], n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Stack (k_i, 3) arrays into (K, n, 3) with a (K, n) mask of the real rows."""
    n = n or max(1, max(len(a) for a in arrays))
    out = np.zeros((len(arrays), n, 3))
    mask = np.zeros((len(arrays), n), bool)
    for k, a in enumerate(arrays):
        m = min(len(a), n)
        out[k, :m] = a[:m]
        mask[k, :m] = True
    return out, mask


def _icp_batch(
    model: PartModel,
    clouds: np.ndarray,
    mask: np.ndarray,
    rotations: np.ndarray,
    positions: np.ndarray,
    iterations: int,
    f: dict,
    z0: np.ndarray | None = None,
    dof: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust point-to-plane ICP of H poses of one part, each to its own cloud.

    clouds (H, C, 3) with mask (H, C); dof 4 = x, y, z, yaw; dof 6 = free. One numpy
    operation does all H poses. A pose stops when its step is very small.
    Returns (R (H, 3, 3), t (H, 3), mean weighted residual (H,)).
    """
    r, t = rotations.astype(float).copy(), positions.astype(float).copy()
    n_h = len(r)
    huber, max_corr = f["huber"], f["max_corr"]
    count = np.maximum(mask.sum(1), 1)
    pivot = (clouds * mask[..., None]).sum(1) / count[:, None]
    rms = np.full(n_h, np.inf)
    active = np.ones(n_h, bool)
    n_dof = 4 if dof == 4 else 6
    eye = np.eye(n_dof)
    for _ in range(iterations):
        hs = np.flatnonzero(active)
        if not len(hs):
            break
        ra, ta, pv = r[hs], t[hs], pivot[hs]
        cl, mk = clouds[hs], mask[hs]
        local = np.matmul(cl - ta[:, None], ra)
        idx = model.grid.lookup(local.reshape(-1, 3)).reshape(len(hs), -1)
        ok = (idx >= 0) & mk
        idx0 = np.where(ok, idx, 0)
        rt = ra.transpose(0, 2, 1)
        c = np.matmul(model.samples[idx0], rt) + ta[:, None]
        n = np.matmul(model.normals[idx0], rt)
        diff = cl - c
        dist2 = np.einsum("hci,hci->hc", diff, diff)
        res = np.einsum("hci,hci->hc", n, diff)
        ares = np.abs(res)
        w = (ok & (dist2 < max_corr**2)) * np.where(
            ares < huber, 1.0, huber / np.maximum(ares, 1e-12)
        )
        wsum = w.sum(1)
        good = (ok.sum(1) >= 10) & (wsum >= 10)
        a = c - pv[:, None]
        if dof == 4:
            axz = a[..., 0] * n[..., 1] - a[..., 1] * n[..., 0]
            jac = np.concatenate((-axz[..., None], -n), -1)
        else:
            jac = np.concatenate((-np.cross(a, n), -n), -1)
        jw = jac * w[..., None]
        h = np.matmul(jw.transpose(0, 2, 1), jac)
        g = np.einsum("hcj,hc->hj", jw, res)
        if dof == 4 and z0 is not None and f["z_prior"] > 0:
            lam = f["z_prior"] * wsum
            h[:, 3, 3] += lam
            g[:, 3] += lam * (ta[:, 2] - z0[hs])
        h += eye[None] * (1e-9 * np.maximum(1.0, np.trace(h, axis1=1, axis2=2)))[:, None, None]
        h[~good] = eye
        g[~good] = 0.0
        step = -np.linalg.solve(h, g[..., None])[..., 0]
        if dof == 4:
            dr = geo.rot_z_batch(step[:, 0])
            v = step[:, 1:]
        else:
            dr = geo.rodrigues_batch(step[:, :3])
            v = step[:, 3:]
        new_r = np.matmul(dr, ra)
        new_t = np.matmul(dr, (ta - pv)[..., None])[..., 0] + pv + v
        upd = hs[good]
        r[upd], t[upd] = new_r[good], new_t[good]
        rms[upd] = np.sqrt((w * res**2).sum(1) / np.maximum(wsum, 1e-12))[good]
        active[hs[~good | (np.linalg.norm(step, axis=1) < 1e-5)]] = False
    # Keep the rotations exact.
    u, _, vt = np.linalg.svd(r)
    return u @ vt, t, rms


@dataclass
class _Scores:
    quality: np.ndarray
    score: np.ndarray
    explained: np.ndarray
    coverage: np.ndarray
    seen: np.ndarray
    iou: np.ndarray  # NaN where there is no colour mask
    inliers: np.ndarray  # (H, S) mask over the scored segment points
    per_view: np.ndarray  # (H, V) matched model points per view
    visible: dict  # view index -> (hyp indices, (k, M) matched mask); only if asked


def _score_batch(
    model: PartModel,
    rot: np.ndarray,
    pos: np.ndarray,
    pts: np.ndarray,
    pts_mask: np.ndarray,
    jobs: list[_Job],
    views: list[View],
    f: dict,
    stride: int = 1,
    want_visible: bool = False,
) -> _Scores:
    """Score of H poses of one part; pose k belongs to ``jobs[k]``. See the module docstring.

    pts (H, S, 3) with pts_mask: the segment points for the explained term.
    """
    n_h = len(rot)
    local = np.matmul(pts - pos[:, None], rot)
    idx = model.grid.lookup(local.reshape(-1, 3)).reshape(n_h, -1)
    ok = (idx >= 0) & pts_mask
    d = local - model.samples[np.where(ok, idx, 0)]
    inl = ok & (np.einsum("hsi,hsi->hs", d, d) < f["inlier"] ** 2)
    explained = inl.sum(1) / np.maximum(pts_mask.sum(1), 1)
    samples = model.score_samples[::stride].astype(np.float32)
    matched = np.zeros(n_h, np.int64)
    violated = np.zeros(n_h, np.int64)
    per_view = np.zeros((n_h, len(views)), np.int64)
    iou_sum = np.zeros(n_h)
    iou_n = np.zeros(n_h)
    visible: dict = {}
    b = f["visibility_bin_px"]
    k_close = 3 if stride == 1 else 5
    kernel = np.ones((k_close, k_close), np.uint8)
    for i, v in enumerate(views):
        hs = np.array([k for k, job in enumerate(jobs) if i in job.which], np.int64)
        if not len(hs):
            continue
        fr = v.frame
        height, width = fr.depth_m.shape
        # Camera frame of the model points: loc = A_h s + b_h, all poses in one matrix
        # product. Layout (M, k): one column per pose.
        a = np.matmul(v.rotation.T[None], rot[hs])  # (k, 3, 3)
        off = np.matmul(pos[hs] - v.position, v.rotation)  # (k, 3)
        k = len(hs)
        loc = samples @ a.transpose(2, 1, 0).reshape(3, 3 * k).astype(np.float32)
        loc = loc.reshape(len(samples), 3, k)
        z = loc[:, 2] + off[:, 2].astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            uf = fr.fx * (loc[:, 0] + off[:, 0].astype(np.float32)) / z + fr.cx
            vf = fr.fy * (loc[:, 1] + off[:, 1].astype(np.float32)) / z + fr.cy
            ok_px = (
                (z > 0.05) & (uf > -0.5) & (uf < width - 0.5) & (vf > -0.5) & (vf < height - 0.5)
            )
        ui = np.where(ok_px, np.rint(uf), 0).astype(np.int64)
        vi = np.where(ok_px, np.rint(vf), 0).astype(np.int64)
        # Z-buffer per pose over the pixel bins of its own bounding box.
        bu, bv = ui // b, vi // b
        big = np.int64(1) << 40
        bu0 = np.where(ok_px, bu, big).min(0)
        bv0 = np.where(ok_px, bv, big).min(0)
        bw = np.maximum(np.where(ok_px, bu, -1).max(0) - bu0 + 1, 0)
        bh = np.maximum(np.where(ok_px, bv, -1).max(0) - bv0 + 1, 0)
        size = bw * bh
        offset = np.concatenate(([0], np.cumsum(size)[:-1]))
        key = np.where(ok_px, offset + (bv - bv0) * bw + bu - bu0, 0)
        zbuf = np.full(max(int(size.sum()), 1), np.inf, np.float32)
        np.minimum.at(zbuf, key[ok_px], z[ok_px])
        front = ok_px & (z <= zbuf[key] + 0.002)
        dm = fr.depth_m[vi, ui]
        has = front & (dm > 0)
        tol = f["depth_tol"] + f["depth_tol_z2"] * z * z
        diff = dm - z
        m = has & (np.abs(diff) <= tol)
        x = has & (diff > tol)
        mc = m.sum(0)
        matched[hs] += mc
        violated[hs] += x.sum(0)
        per_view[hs, i] = mc
        if want_visible:
            visible[i] = (hs, m)
        # Colour silhouette: the projected model, closed, against the brick-colour mask.
        with_mask = [
            (row, jobs[j].masks[i])
            for row, j in enumerate(hs)
            if jobs[j].masks and i in jobs[j].masks
        ]
        if not with_mask:
            continue
        rows = np.array([row for row, _ in with_mask])
        mh = max(mm[2].shape[0] for _, mm in with_mask)
        mw = max(mm[2].shape[1] for _, mm in with_mask)
        tile_h = mh + k_close
        canvas = np.zeros((len(rows) * tile_h, mw + k_close), np.uint8)
        mask_stack = np.zeros((len(rows), mh, mw), bool)
        win_stack = np.zeros((len(rows), mh, mw), bool)
        x0 = np.zeros(len(rows), np.int64)
        y0 = np.zeros(len(rows), np.int64)
        wmax = np.zeros(len(rows), np.int64)
        hmax = np.zeros(len(rows), np.int64)
        for j, (_, (wx, wy, cmask)) in enumerate(with_mask):
            x0[j], y0[j] = wx, wy
            hmax[j], wmax[j] = cmask.shape
            mask_stack[j, : cmask.shape[0], : cmask.shape[1]] = cmask
            win_stack[j, : cmask.shape[0], : cmask.shape[1]] = True
        with np.errstate(invalid="ignore"):
            su = np.rint(uf[:, rows]) - x0
            sv = np.rint(vf[:, rows]) - y0
            inside = (z[:, rows] > 0) & (su >= 0) & (su < wmax) & (sv >= 0) & (sv < hmax)
        pm, pj = np.nonzero(inside)
        canvas[pj * tile_h + sv[pm, pj].astype(np.int64), su[pm, pj].astype(np.int64)] = 1
        closed = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        sil = closed.reshape(len(rows), tile_h, mw + k_close)[:, :mh, :mw] > 0
        sil &= win_stack
        union = (sil | mask_stack).sum((1, 2))
        inter = (sil & mask_stack).sum((1, 2))
        good = union > 0
        iou_sum[hs[rows[good]]] += inter[good] / union[good]
        iou_n[hs[rows[good]]] += 1
    seen = matched + violated
    coverage = np.where(seen > 0, matched / np.maximum(seen, 1), 0.0)
    quality = np.where(seen * stride >= f["min_visible"], explained * coverage, 0.0)
    iou = np.where(iou_n > 0, iou_sum / np.maximum(iou_n, 1), np.nan)
    score = quality.copy()
    has_iou = np.isfinite(iou)
    score[has_iou] *= iou[has_iou] ** f["silhouette_weight"]
    return _Scores(quality, score, explained, coverage, seen, iou, inl, per_view, visible)


def _color_masks(job: _Job, views: list[View], f: dict) -> dict:
    """Per view: a window of the colour image around the segment and the brick-colour mask.

    The colour image is sharp, but the depth of a small brick is soft. The mask gives the
    brick outline: pixels near the median colour of the segment.
    """
    masks = {}
    if f["silhouette_weight"] <= 0:
        return masks
    seg = job.seg
    for i in job.which:
        v = views[i]
        fr = v.frame
        uv, z = geo.project(seg.local_points, v.rotation, v.position, fr.fx, fr.fy, fr.cx, fr.cy)
        ok = z > 0
        if ok.sum() < 10:
            continue
        h, w = fr.rgb.shape[:2]
        m = f["silhouette_margin_px"]
        x0, y0 = (np.floor(uv[ok].min(0)) - m).astype(int)
        x1, y1 = (np.ceil(uv[ok].max(0)) + m).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        lab = v.lab_image()[y0:y1, x0:x1]
        mask = geo.color_distance(lab, seg.lab) < f["silhouette_color_tol"]
        mask = (
            cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        )
        masks[i] = (x0, y0, mask)
    return masks


def _initial_poses(
    model: PartModel, seg: Segment, f: dict, rect
) -> list[tuple[PoseHypothesis, float, float]]:
    """(pose, yaw, z) start values that agree with the segment height and footprint."""
    (cxy, (w, h), angle) = rect
    long_angle = math.radians(angle) if w >= h else math.radians(angle + 90)
    seg_long = max(w, h)
    seg_short = min(w, h)
    top = seg.top
    out = []
    square = seg_long < 1.25 * max(seg_short, 1e-6)
    steps = 8 if square else 4
    for pose in model.poses:
        if top > pose.top + f["height_tol_above"] or top < pose.top - f["height_tol_below"]:
            continue
        if (
            seg_long > math.hypot(pose.long, pose.short) + f["extent_tol"]
            or seg_short > pose.long + f["extent_tol"]
        ):
            continue
        seen: list[float] = []
        for k in range(steps):
            yaw = long_angle + k * 2 * math.pi / steps
            yaw_m = yaw % pose.yaw_period
            if any(min(abs(yaw_m - s), pose.yaw_period - abs(yaw_m - s)) < 0.05 for s in seen):
                continue
            seen.append(yaw_m)
            out.append((pose, yaw, seg.support_z + pose.rest_height))
    return out


def _make_fit(model, pose, rot, pos, sc: _Scores, k: int, job: _Job, views, f, rms=0.0) -> Fit:
    cams = [views[i].name for i in job.which if sc.per_view[k, i] > 0.3 * f["min_visible"]]
    iou = float(sc.iou[k]) if np.isfinite(sc.iou[k]) else None
    inl = sc.inliers[k, : len(job.fine_pts)]
    return Fit(
        model.object_id,
        pose,
        rot[k],
        pos[k],
        float(sc.explained[k]),
        float(sc.coverage[k]),
        int(sc.seen[k]),
        float(sc.score[k]),
        None,
        float(rms),
        cams,
        job.fine_rgb[inl] if inl.any() else job.fine_rgb,
        job.seg,
        float(sc.quality[k]),
        iou,
    )


def _fine_fits(
    models: dict[str, PartModel],
    jobs: list[_Job],
    views: list[View],
    f: dict,
    starts: list[tuple[int, str, PoseHypothesis | None, np.ndarray, np.ndarray, float]],
    iterations: int,
) -> list[tuple[int, Fit]]:
    """ICP and full score of start poses (job index, part, pose, R, t, z0), one pass per part."""
    out: list[tuple[int, Fit]] = []
    by_part: dict[str, list] = {}
    for s in starts:
        by_part.setdefault(s[1], []).append(s)
    for object_id, group in by_part.items():
        model = models[object_id]
        jidx = [g[0] for g in group]
        gj = [jobs[j] for j in jidx]
        clouds, cmask = _pad([j.cloud for j in gj])
        rot = np.stack([g[3] for g in group])
        pos = np.stack([g[4] for g in group])
        z0 = np.array([g[5] for g in group])
        r2, t2, rms = _icp_batch(model, clouds, cmask, rot, pos, iterations, f, z0)
        if f["snap_z"]:
            snap = (np.abs(t2[:, 2] - z0) < f["snap_z_tol"]) & np.array(
                [g[2] is not None for g in group]
            )
            t2[snap, 2] = z0[snap]
        pts, pmask = _pad([j.fine_pts for j in gj])
        sc = _score_batch(model, r2, t2, pts, pmask, gj, views, f, 1)
        if f["keep_better_start"]:
            # Points that are not on this brick (soft depth at the edge of a touching brick,
            # a plate wall) can pull the ICP to a worse pose. Score the start pose too and
            # keep the better of the two.
            t0 = pos.copy()
            if f["snap_z"]:
                t0[:, 2] = np.where([g[2] is not None for g in group], z0, t0[:, 2])
            sc0 = _score_batch(model, rot, t0, pts, pmask, gj, views, f, 1)
            better = sc0.score > sc.score
            if better.any():
                r2, t2 = r2.copy(), t2.copy()
                r2[better], t2[better] = rot[better], t0[better]
                sc = _score_batch(model, r2, t2, pts, pmask, gj, views, f, 1)
        for k, g in enumerate(group):
            fit = _make_fit(model, g[2], r2, t2, sc, k, gj[k], views, f, rms[k] * 1000)
            out.append((g[0], fit))
    return out


def _yaw_diff(a: np.ndarray, b: np.ndarray, period: float) -> float:
    d = (geo.yaw_of(a) - geo.yaw_of(b)) % period
    return min(d, period - d)


def _search(
    models: dict[str, PartModel], jobs: list[_Job], views: list[View], f: dict
) -> list[tuple[Fit | None, list[dict]]]:
    """Best part, stable pose and 4-DoF pose for each job (full coarse-to-fine search)."""
    n = len(jobs)
    results: list[tuple[Fit | None, list[dict]]] = [(None, []) for _ in range(n)]
    if not n:
        return results
    starts_by_part: dict[str, list] = {}
    for j, job in enumerate(jobs):
        centre = np.array(job.rect[0], float)
        for model in models.values():
            for pose, yaw, z in _initial_poses(model, job.seg, f, job.rect):
                rz = geo.rot_z(yaw)
                t = np.array([*(centre - rz[:2, :2] @ pose.centre_xy), z])
                starts_by_part.setdefault(model.object_id, []).append(
                    (j, pose, rz @ pose.rotation, t, z)
                )
    has_start = {s[0] for group in starts_by_part.values() for s in group}
    for j in has_start:
        jobs[j].masks = _color_masks(jobs[j], views, f)
    coarse_clouds, coarse_mask = _pad([job.cloud[: f["coarse_points"]] for job in jobs])
    coarse_pts, coarse_pmask = _pad([job.coarse_pts for job in jobs])
    candidates: list[list] = [[] for _ in range(n)]
    for object_id, group in starts_by_part.items():
        model = models[object_id]
        jidx = np.array([g[0] for g in group])
        rot = np.stack([g[2] for g in group])
        pos = np.stack([g[3] for g in group])
        z0 = np.array([g[4] for g in group])
        r1, t1, _ = _icp_batch(
            model, coarse_clouds[jidx], coarse_mask[jidx], rot, pos, f["coarse_iterations"], f, z0
        )
        sc = _score_batch(
            model,
            r1,
            t1,
            coarse_pts[jidx],
            coarse_pmask[jidx],
            [jobs[j] for j in jidx],
            views,
            f,
            3,
        )
        for k, g in enumerate(group):
            candidates[g[0]].append(
                (float(sc.score[k]), float(sc.quality[k]), object_id, g[1], r1[k], t1[k], g[4])
            )
    fine_starts = []
    for j in range(n):
        cands = candidates[j]
        if not cands:
            results[j] = (None, [{"reason": "no part has a stable pose of this height and size"}])
            continue
        cands.sort(key=lambda c: -c[0])
        best_q = max(c[1] for c in cands)
        if best_q < f["coarse_min_quality"]:
            c = cands[0]
            results[j] = (
                None,
                [
                    {
                        "reason": "low score (coarse stage)",
                        "best": {
                            "object_id": c[2],
                            "pose": c[3].pose_id,
                            "quality": round(best_q, 3),
                        },
                    }
                ],
            )
            continue
        chosen: list = []
        for c in cands:
            if len(chosen) >= f["keep_coarse"] or c[0] < f["fine_min_ratio"] * cands[0][0]:
                break
            # Start poses that converged to the same pose: only the best one goes on.
            if any(
                o[2] == c[2]
                and o[3] is c[3]
                and np.linalg.norm(o[5][:2] - c[5][:2]) < 0.003
                and _yaw_diff(o[4], c[4], c[3].yaw_period) < math.radians(5)
                for o in chosen
            ):
                continue
            chosen.append(c)
            fine_starts.append((j, c[2], c[3], c[4], c[5], c[6], c[1]))
    fits = _fine_fits(models, jobs, views, f, fine_starts, f["fine_iterations"])
    per_job: dict[int, list[Fit]] = {}
    for j, fit in fits:
        per_job.setdefault(j, []).append(fit)
    for j, group in per_job.items():
        group.sort(key=lambda x: -x.score)
        best = group[0]
        best.alternatives = group[1:]
        tried = [_fit_summary(x) for x in group]
        if f["free_refine"]:
            best, extra = _free_refine(models, jobs[j], best, views, f)
            tried += extra
        results[j] = (best, tried)
    return results


def _free_refine(models, job: _Job, best: Fit, views, f) -> tuple[Fit, list[dict]]:
    """Optional 6-DoF refinement: accept a pose that is not stable if it is much better."""
    model = models[best.object_id]
    clouds, cmask = _pad([job.cloud])
    r3, t3, rms = _icp_batch(
        model,
        clouds,
        cmask,
        best.rotation[None],
        best.position[None],
        f["fine_iterations"],
        f,
        dof=6,
    )
    pts, pmask = _pad([job.fine_pts])
    sc = _score_batch(model, r3, t3, pts, pmask, [job], views, f, 1)
    free = _make_fit(model, None, r3, t3, sc, 0, job, views, f, rms[0] * 1000)
    angle = geo.rotation_angle_deg(r3[0], best.rotation)
    if free.score > best.score + f["free_min_gain"] and angle > f["free_min_angle_deg"]:
        return free, [{**_fit_summary(free), "free": True}]
    return best, []


def _fit_summary(fit: Fit) -> dict:
    return {
        "object_id": fit.object_id,
        "pose": fit.pose.pose_id if fit.pose else None,
        "pos": fit.position.round(4).tolist(),
        "quat": matrix_to_quat(fit.rotation).round(5).tolist(),
        "explained": round(fit.explained, 3),
        "coverage": round(fit.coverage, 3),
        "visible": fit.visible,
        "score": round(fit.score, 3),
        "quality": round(fit.quality, 3),
        "silhouette_iou": None if fit.iou is None else round(fit.iou, 3),
    }


def _reject_reason(fit: Fit, f: dict) -> str | None:
    if fit.visible < f["min_visible"]:
        return "too few model points in view"
    if fit.explained < f["min_explained"]:
        return "points do not lie on the part"
    if fit.coverage < f["min_coverage"]:
        return "part does not agree with the depth image"
    if fit.quality < f["min_score"]:
        return "low score"
    if fit.iou is not None and fit.iou < f["min_silhouette_iou"]:
        return "outline does not match the brick colour in the image"
    return None


def _residual(model: PartModel, fit: Fit, views: list[View], f: dict) -> float | None:
    """Median distance (mm) of the visible model points to the segment cloud."""
    job = _Job(
        fit.segment,
        fit.segment.local_points,
        fit.segment.points,
        fit.segment.points,
        fit.segment.rgb,
        sorted(set(np.unique(fit.segment.local_view_index).tolist())),
        (),
    )
    pts, pmask = _pad([np.zeros((1, 3))])
    sc = _score_batch(
        model,
        fit.rotation[None],
        fit.position[None],
        pts,
        pmask,
        [job],
        views,
        f,
        1,
        want_visible=True,
    )
    samples = model.score_samples @ fit.rotation.T + fit.position
    vis = [samples[m[:, 0]] for _, m in sc.visible.values()]
    pts = np.concatenate(vis) if vis else np.zeros((0, 3))
    if len(pts) < 5:
        return None
    cloud = fit.segment.local_points
    pts = pts[:: max(1, len(pts) // 400)]  # the median needs only a few hundred points
    cloud = cloud[:: max(1, len(cloud) // 3000)]
    _, dist = geo.nearest_neighbours(cloud, pts)
    return float(np.median(dist) * 1000)


# ----------------------------------------------------------------------------- no overlap


def _footprint(model: PartModel, rot: np.ndarray, pos: np.ndarray):
    v = model.hull @ rot.T + pos
    return v, geo.convex_hull_2d(v[:, :2])


def _overlap_depth(va, pa, vb, pb) -> float:
    """XY overlap depth (m) of two posed bricks (hull points, footprint); <= 0: no overlap."""
    if va[:, 2].min() >= vb[:, 2].max() - 1e-4 or vb[:, 2].min() >= va[:, 2].max() - 1e-4:
        return -1.0
    return geo.polygon_overlap(pa, pb)[0]


def _wall_violation(
    v: np.ndarray, plate: PlateFit, inside: bool, clearance: float
) -> tuple[float, np.ndarray]:
    """Depth of hull points ``v`` in a plate wall, and the unit XY direction out of it."""
    d = v[:, :2] - plate.centre
    r = np.linalg.norm(d, axis=1)
    if inside:  # the brick is on the plate floor: stay inside the inner wall surface
        limit = np.interp(v[:, 2], PLATE_INNER_Z, PLATE_INNER_R)
        excess = r - (limit - clearance)
        k = int(np.argmax(excess))
        return float(excess[k]), -d[k] / max(r[k], 1e-9)
    low = v[:, 2] < PLATE_RIM_Z
    if not low.any():
        return -1.0, np.zeros(2)
    limit = np.interp(v[low, 2], PLATE_OUTER_Z, PLATE_OUTER_R)
    excess = (limit + clearance) - r[low]
    k = int(np.argmax(excess))
    centre = v[:, :2].mean(0) - plate.centre
    return float(excess[k]), centre / max(np.linalg.norm(centre), 1e-9)


def _penetration(items: list[dict], plates: list[PlateFit]) -> tuple[float, list]:
    """Largest penetration (m) between bricks and between a brick and a plate wall."""
    worst, pairs = 0.0, []
    for a in range(len(items)):
        ia = items[a]
        va, pa = _footprint(ia["model"], ia["rot"], ia["pos"])
        for b in range(a + 1, len(items)):
            ib = items[b]
            vb, pb = _footprint(ib["model"], ib["rot"], ib["pos"])
            if va[:, 2].min() >= vb[:, 2].max() - 1e-4 or vb[:, 2].min() >= va[:, 2].max() - 1e-4:
                continue
            depth, _ = geo.polygon_overlap(pa, pb)
            if depth > 0:
                pairs.append((ia["id"], ib["id"], round(depth * 1000, 2)))
                worst = max(worst, depth)
        for plate in plates:
            if np.linalg.norm(va[:, :2].mean(0) - plate.centre) > 0.2:
                continue
            depth, _ = _wall_violation(va, plate, ia["plate"] is plate, 0.0)
            if depth > 0:
                pairs.append((ia["id"], "plate", round(depth * 1000, 2)))
                worst = max(worst, depth)
    return worst, pairs


def _separate(items: list[dict], plates: list[PlateFit], f: dict) -> dict:
    """Move bricks apart in x, y so that no brick goes into another brick or a plate wall.

    Each round: for each pair that overlaps (or is nearer than the clearance), the smallest
    move that separates the two footprints, shared so that the brick with the lower
    confidence moves more; for each brick in a wall, the move out of the wall. Height and
    orientation do not change. Returns a report (penetration before and after, moves).
    """
    clearance = float(f["min_clearance_m"])
    before, pairs = _penetration(items, plates)
    report = {"max_penetration_mm_before": round(before * 1000, 2), "overlaps_before": pairs}
    if not f["separate"] or clearance < 0 or not items:
        report["max_penetration_mm_after"] = report["max_penetration_mm_before"]
        return report
    start = [it["pos"].copy() for it in items]
    eps = 2e-5
    for _ in range(int(f["separate_rounds"])):
        moved = False
        shapes = [_footprint(it["model"], it["rot"], it["pos"]) for it in items]
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (va, pa), (vb, pb) = shapes[a], shapes[b]
                if (
                    va[:, 2].min() >= vb[:, 2].max() - 1e-4
                    or vb[:, 2].min() >= va[:, 2].max() - 1e-4
                ):
                    continue
                depth, direction = geo.polygon_overlap(pa, pb)
                need = depth + clearance
                if need <= 0:
                    continue
                need += eps
                ca, cb = max(items[a]["conf"], 1e-3), max(items[b]["conf"], 1e-3)
                share_b = ca / (ca + cb)
                move_b = direction * need * share_b
                move_a = -direction * need * (1 - share_b)
                items[a]["pos"][:2] += move_a
                items[b]["pos"][:2] += move_b
                shapes[a] = (va + [*move_a, 0], pa + move_a)
                shapes[b] = (vb + [*move_b, 0], pb + move_b)
                moved = True
        for a, it in enumerate(items):
            v = shapes[a][0]
            for plate in plates:
                if np.linalg.norm(v[:, :2].mean(0) - plate.centre) > 0.2:
                    continue
                if it["support"] == "other":
                    continue
                depth, direction = _wall_violation(v, plate, it["plate"] is plate, clearance)
                if depth > 0:
                    move = direction * (depth + eps)
                    it["pos"][:2] += move
                    v = v + [*move, 0]
                    moved = True
        if not moved:
            break
    after, pairs = _penetration(items, plates)
    report["max_penetration_mm_after"] = round(after * 1000, 2)
    if pairs:
        report["overlaps_after"] = pairs
    moves = {
        it["id"]: round(float(np.linalg.norm(it["pos"][:2] - s[:2])) * 1000, 2)
        for it, s in zip(items, start)
    }
    report["moves_mm"] = {k: v for k, v in moves.items() if v > 0}
    return report


# ----------------------------------------------------------------------------- bricks


_PALETTE_CACHE: dict = {}


def _palette(cfg: dict):
    """The brick palette (brickcell.palette), or None if it can not be loaded."""
    key = cfg.get("palette") or ""
    if key not in _PALETTE_CACHE:
        try:
            from brickcell.palette import load_palette

            _PALETTE_CACHE[key] = load_palette(key or None)
        except Exception:  # noqa: BLE001 - colour names are optional
            _PALETTE_CACHE[key] = None
    return _PALETTE_CACHE[key]


def _footprint_centre(models, fit: Fit) -> np.ndarray:
    return fit.position[:2] + fit.rotation[:2] @ models[fit.object_id].centre


def _assign_ids(fits: list[Fit], f: dict, models) -> tuple[list[tuple[str, Fit]], list[Fit]]:
    """Copy ids per part: tracked ids stay; the others get free copies in score order."""
    chosen: list[tuple[str, Fit]] = []
    extra: list[Fit] = []
    by_part: dict[str, list[Fit]] = {}
    for fit in fits:
        by_part.setdefault(fit.object_id, []).append(fit)
    for part_id, group in by_part.items():
        group.sort(key=lambda x: -x.score)
        used: set[str] = set()
        for fit in group:
            if fit.track_id and fit.track_id not in used:
                used.add(fit.track_id)
                chosen.append((fit.track_id, fit))
        for fit in group:
            if fit.track_id and any(fit is c[1] for c in chosen):
                continue
            free = [
                instance_id(part_id, k)
                for k in range(1, INSTANCES_PER_PART + 1)
                if instance_id(part_id, k) not in used
            ]
            if free:
                used.add(free[0])
                chosen.append((free[0], fit))
            else:
                extra.append(fit)
    return chosen, extra


def _build_bricks(
    fits: list[Fit], plates: list[PlateFit], views: list[View], cfg: dict, models
) -> tuple[list[BrickState], list[dict], dict, list[tuple[str, Fit]]]:
    f = cfg["fit"]
    # Two fits that explain the same points are the same brick: keep the better one.
    fits = sorted(fits, key=lambda x: -x.score)
    kept: list[Fit] = []
    for fit in fits:
        c = _footprint_centre(models, fit)
        if all(np.linalg.norm(c - _footprint_centre(models, o)) >= f["same_brick_m"] for o in kept):
            kept.append(fit)
    chosen, extra = _assign_ids(kept, f, models)
    items = []
    for object_id, fit in chosen:
        model = models[fit.object_id]
        support, plate_fit = "table", None
        centre_xy = _footprint_centre(models, fit)
        for plate in plates:  # the footprint centre is inside the plate wall (or on it)
            if (
                np.linalg.norm(centre_xy - plate.centre)
                < cfg["plates"]["support_radius"] * plate.scale
            ):
                support = f"plate_{plate.name}" if plate.name else "other"
                plate_fit = plate if fit.segment.plate is not None else None
                break
        bottom = float((model.hull @ fit.rotation.T + fit.position)[:, 2].min())
        if bottom > fit.segment.support_z + 0.008:
            support = "other"
        items.append(
            {
                "id": object_id,
                "fit": fit,
                "model": model,
                "rot": fit.rotation,
                "pos": fit.position.astype(float).copy(),
                "conf": fit.quality,
                "support": support,
                "plate": plate_fit,
            }
        )
    overlap = _separate(items, plates, f)
    palette = _palette(cfg)
    bricks: list[BrickState] = []
    for it in items:
        fit, model = it["fit"], it["model"]
        fit.position = it["pos"]
        color = (np.median(fit.inlier_rgb, axis=0) / 255.0).round(4)
        name = palette.nearest(color)[0] if palette is not None else None
        residual = _residual(model, fit, views, f)
        bricks.append(
            BrickState(
                object_id=it["id"],
                part=model.part,
                pos=[round(float(v), 5) for v in fit.position],
                quat=[round(float(v), 6) for v in matrix_to_quat(fit.rotation)],
                color=color.tolist(),
                stable_pose=fit.pose.pose_id if fit.pose else None,
                support=it["support"],
                confidence=round(float(fit.quality), 3),
                residual_mm=None if residual is None else round(residual, 2),
                cameras=fit.cameras,
                color_name=name,
            )
        )
    bricks.sort(key=lambda b: (b.object_id.count("-"), b.object_id))
    duplicates = []
    for fit in extra:
        color = (np.median(fit.inlier_rgb, axis=0) / 255.0).round(4).tolist()
        duplicates.append(
            {
                **_fit_summary(fit),
                "color": color,
                "stable_pose": fit.pose.pose_id if fit.pose else None,
                "reason": f"more than {INSTANCES_PER_PART} bricks of this part",
            }
        )
    return bricks, duplicates, overlap, [(it["id"], it["fit"]) for it in items]


# ----------------------------------------------------------------------------- tracker


def _cell_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / max(len(a) + len(b) - inter, 1)


@dataclass
class _Memory:
    """A segment of the last frame and what perception found for it."""

    cell_ids: np.ndarray
    top: float
    lab: np.ndarray
    plate: int | None
    fit: Fit | None = None  # accepted brick
    object_id: str | None = None  # its copy id
    rejected: dict | None = None  # or the rejection entry


class PerceptionTracker:
    """Perception with memory between frames (live mode). ``update`` returns a StationState.

    - Plates: found again after ``tracker.plate_period_s`` or when a plate wall is no longer
      at its place; else the last plates are used.
    - A segment that did not change (cell IoU, top height and colour) keeps its result: a
      brick gets a short local refinement, a rejected segment stays rejected (no fit).
    - New or changed segments get the full search. A brick keeps its copy id while it is
      tracked; a new fit near a lost brick of the same part gets the id of that brick.
    - A full search of all segments after ``tracker.full_period_s``.
    ``locate`` is one update of a new tracker.
    """

    def __init__(self, scene, config: dict | None = None):
        self.scene = scene
        self.cfg = load_perception_config() if config is None else _merge(DEFAULT_CONFIG, config)
        self.reset()

    def reset(self) -> None:
        self._plates: list[PlateFit] | None = None
        self._plates_rejected: list[dict] = []
        self._agreement: dict = {}
        self._used_cameras: list[str] = []
        self._plate_time = -math.inf
        self._full_time = -math.inf
        self._memory: list[_Memory] = []
        self.updates = 0

    def update(self, frames: list[RGBDFrame], scene=None, now: float | None = None) -> StationState:
        if scene is not None and scene is not self.scene:
            self.scene = scene
            self.reset()
        now = time.monotonic() if now is None else now
        cfg, f, tr = self.cfg, self.cfg["fit"], self.cfg["tracker"]
        times: dict[str, float] = {}
        t_start = time.perf_counter()
        models = part_models(cfg)
        times["models"] = time.perf_counter() - t_start

        t0 = time.perf_counter()
        all_views = []
        for i, frame in enumerate(frames):
            camera = self.scene.cameras.get(frame.camera)
            if camera is None:
                raise ValueError(f"Camera '{frame.camera}' is not in the scene config")
            all_views.append(_prepare_view(frame, camera, cfg, seed=i))
        times["views"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        names = sorted(v.name for v in all_views)
        plates_mode = "detected"
        views = None
        if (
            self._plates is not None
            and now - self._plate_time < tr["plate_period_s"]
            and set(self._used_cameras) <= set(names)
        ):
            views = [v for v in all_views if v.name in self._used_cameras]
            if _plates_still_there(views, self._plates, cfg, tr["plate_wall_fraction"]):
                plates_mode = "kept"
            else:
                views = None
        if views is None:
            views, self._plates, self._plates_rejected, self._agreement = _detect_plates(
                all_views, cfg
            )
            self._used_cameras = [v.name for v in views]
            self._plate_time = now
        plates = self._plates
        for v in all_views:
            v.info["used"] = any(v is u for u in views)
        times["plates"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        segments, seg_rejected, seg_stats = _segments(views, plates, cfg)
        times["segments"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        full = tr["full_period_s"] > 0 and now - self._full_time >= tr["full_period_s"]
        if not self._memory:
            full = True
        if full:
            self._full_time = now
        jobs = [
            _make_job(seg, f, np.random.default_rng([len(seg.points), int(seg.cell_ids[0])]))
            for seg in segments
        ]
        to_search: list[int] = []
        to_refine: list[tuple[int, _Memory]] = []
        results: dict[int, tuple[Fit | None, list[dict]]] = {}
        free_memory = list(self._memory)
        for j, seg in enumerate(segments):
            match = None if full else self._match(seg, free_memory, tr)
            if match is None:
                to_search.append(j)
            elif match.fit is not None:
                free_memory.remove(match)
                to_refine.append((j, match))
            else:
                free_memory.remove(match)
                results[j] = (None, [match.rejected])
        if to_refine:
            starts = []
            for j, mem in to_refine:
                prior = mem.fit
                z0 = (
                    segments[j].support_z + prior.pose.rest_height
                    if prior.pose is not None
                    else float(prior.position[2])
                )
                starts.append((j, prior.object_id, prior.pose, prior.rotation, prior.position, z0))
            refined = dict(_fine_fits(models, jobs, views, f, starts, tr["refine_iterations"]))
            for j, mem in to_refine:
                fit = refined[j]
                if (
                    _reject_reason(fit, f) is None
                    and fit.quality >= tr["min_quality_ratio"] * mem.fit.quality
                ):
                    fit.track_id = mem.object_id
                    results[j] = (fit, [_fit_summary(fit)])
                else:
                    to_search.append(j)
        searched = _search(models, [jobs[j] for j in to_search], views, f)
        for j, res in zip(to_search, searched):
            results[j] = res
        times["fit"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        fits: list[Fit] = []
        rejected = list(seg_rejected)
        reject_of: dict[int, dict] = {}
        options: dict[int, list[Fit]] = {}
        for j, seg in enumerate(segments):
            fit, tried = results[j]
            info = {
                "centre": seg.info["centre"],
                "points": seg.info["points"],
                "color": seg.info["color"],
            }
            if fit is None:
                entry = {**info, **tried[0]} if "centre" not in tried[0] else tried[0]
                rejected.append(entry)
                reject_of[j] = entry
                continue
            ok = [x for x in [fit, *fit.alternatives] if _reject_reason(x, f) is None]
            if not ok:
                entry = {**info, "reason": _reject_reason(fit, f), "best": _fit_summary(fit)}
                rejected.append(entry)
                reject_of[j] = entry
                continue
            options[j] = ok
        # Joint choice: a fit must not go into a brick that a better segment has.
        shapes: list = []
        for j in sorted(options, key=lambda j: -options[j][0].score):
            pick = None
            for x in options[j]:
                v, poly = _footprint(models[x.object_id], x.rotation, x.position)
                if all(_overlap_depth(v, poly, sv, sp) <= f["max_overlap_m"] for sv, sp in shapes):
                    pick = (x, v, poly)
                    break
            if pick is None:
                seg = segments[j]
                entry = {
                    "centre": seg.info["centre"],
                    "points": seg.info["points"],
                    "color": seg.info["color"],
                    "reason": "goes into a better brick",
                    "best": _fit_summary(options[j][0]),
                }
                rejected.append(entry)
                reject_of[j] = entry
                continue
            if pick[0] is not results[j][0]:
                pick[0].track_id = results[j][0].track_id
                results[j] = (pick[0], results[j][1])
            shapes.append(pick[1:])
            fits.append(pick[0])
        # A new fit near a lost brick of the same part gets the id of that brick.
        lost = [m for m in self._memory if m.fit is not None]
        kept_ids = {fit.track_id for fit in fits if fit.track_id}
        lost = [m for m in lost if m.object_id not in kept_ids]
        for fit in fits:
            if fit.track_id:
                continue
            c = _footprint_centre(models, fit)
            best, best_d = None, tr["id_distance"]
            for m in lost:
                if m.fit.object_id != fit.object_id:
                    continue
                d = float(np.linalg.norm(c - _footprint_centre(models, m.fit)))
                if d < best_d:
                    best, best_d = m, d
            if best is not None:
                fit.track_id = best.object_id
                lost.remove(best)
        bricks, duplicates, overlap, chosen = _build_bricks(fits, plates, views, cfg, models)
        times["bricks"] = time.perf_counter() - t0

        # Memory for the next frame.
        chosen_by_fit = {id(fit): oid for oid, fit in chosen}
        memory = []
        for j, seg in enumerate(segments):
            fit = results[j][0]
            if j in reject_of:
                memory.append(
                    _Memory(seg.cell_ids, seg.top, seg.lab, seg.plate, rejected=reject_of[j])
                )
            elif fit is not None and id(fit) in chosen_by_fit:
                memory.append(
                    _Memory(seg.cell_ids, seg.top, seg.lab, seg.plate, fit, chosen_by_fit[id(fit)])
                )
        self._memory = memory
        self.updates += 1

        plate_states: list[PlateState] = []
        plates_unnamed = []
        for plate in sorted(plates, key=lambda q: -q.confidence):
            entry = {
                "centre": plate.centre.round(4).tolist(),
                "inliers": plate.inliers,
                "arc_deg": plate.arc_deg,
                "color": plate.color.round(3).tolist(),
                "color_distance": round(plate.color_distance, 1),
            }
            if plate.name is None:
                plates_unnamed.append(
                    {**entry, "reason": "colour does not match a configured plate"}
                )
            elif any(p.name == plate.name for p in plate_states):
                plates_unnamed.append({**entry, "reason": f"second '{plate.name}' plate"})
            else:
                plate_states.append(
                    PlateState(
                        plate.name,
                        [round(float(plate.centre[0]), 4), round(float(plate.centre[1]), 4), 0.0],
                        round(plate.confidence, 3),
                    )
                )
        times["total"] = time.perf_counter() - t_start
        return StationState(
            timestamp=max((fr.timestamp for fr in frames), default=time.time()),
            bricks=bricks,
            plates=plate_states,
            arm=None,
            cameras={v.name: v.info for v in all_views},
            diagnostics={
                "timings_ms": {k: round(v * 1000, 1) for k, v in times.items()},
                "camera_agreement": self._agreement,
                "segments": seg_stats,
                "rejected": rejected,
                "duplicates": duplicates,
                "overlap": overlap,
                "tracker": {
                    "update": self.updates,
                    "plates": plates_mode,
                    "full_search": full,
                    "segments_refined": len(to_refine),
                    "segments_kept_rejected": len(segments)
                    - len(to_refine)
                    - len(to_search)
                    + sum(1 for j, _ in to_refine if j in to_search),
                    "segments_searched": len(to_search),
                },
                "plates": [
                    {
                        "name": p.name,
                        "centre": p.centre.round(4).tolist(),
                        "scale": round(p.scale, 3),
                        "inliers": p.inliers,
                        "arc_deg": p.arc_deg,
                        "color": p.color.round(3).tolist(),
                        "color_distance": round(p.color_distance, 1),
                    }
                    for p in plates
                ],
                "plates_rejected": self._plates_rejected + plates_unnamed,
            },
        )

    @staticmethod
    def _match(seg: Segment, memory: list[_Memory], tr: dict) -> _Memory | None:
        best, best_iou = None, tr["match_iou"]
        for m in memory:
            if m.plate != seg.plate or abs(m.top - seg.top) > tr["match_top"]:
                continue
            if float(geo.color_distance(m.lab, seg.lab)) > tr["match_color"]:
                continue
            iou = _cell_iou(seg.cell_ids, m.cell_ids)
            if iou >= best_iou:
                best, best_iou = m, iou
        return best


# ----------------------------------------------------------------------------- main


def locate(frames: list[RGBDFrame], scene, config: dict | None = None) -> StationState:
    """Find the bricks and the plates in RGB-D frames. Returns a StationState (arm = None).

    frames: one RGBDFrame per camera; ``frame.camera`` must be a camera of ``scene``.
    scene: SceneConfig (brickcell.simulation.scene_config) with the camera poses.
    config: the perception config (``load_perception_config()``); None loads configs/twin.yaml.
    One update of a new PerceptionTracker (no memory of other frames).
    """
    return PerceptionTracker(scene, config).update(frames)


# ----------------------------------------------------------------------------- overlays


def draw_overlay(
    frame: RGBDFrame,
    state: StationState,
    scene,
    config: dict | None = None,
    show_duplicates: bool = True,
) -> np.ndarray:
    """RGB image with the fitted bricks (filled outline of the mesh) and plate circles.

    Duplicates (more bricks of a part than the sim has copies) are drawn too, with "dup".
    """
    cfg = load_perception_config() if config is None else _merge(DEFAULT_CONFIG, config)
    models = part_models(cfg)
    info = state.cameras.get(frame.camera, {})
    if "corrected_pose" in info:
        rotation = np.asarray(info["corrected_pose"]["rotation_cv"], float)
        position = np.asarray(info["corrected_pose"]["pos"], float)
    else:
        cam = scene.cameras[frame.camera]
        rotation, position = np.asarray(cam.rotation_cv(), float), np.asarray(cam.pos, float)
    img = frame.rgb.copy()
    layer = img.copy()
    h, w = img.shape[:2]

    def proj(pts):
        uv, z = geo.project(pts, rotation, position, frame.fx, frame.fy, frame.cx, frame.cy)
        return uv, z

    for plate in state.plates:
        theta = np.linspace(0, 2 * math.pi, 120)
        for radius, z in ((0.093, 0.0235), (0.079, 0.0)):
            ring = np.column_stack(
                (
                    plate.pos[0] + radius * np.cos(theta),
                    plate.pos[1] + radius * np.sin(theta),
                    np.full_like(theta, z),
                )
            )
            uv, zc = proj(ring)
            ok = zc > 0
            pts = np.round(uv[ok] * 4).astype(np.int32)
            color = (40, 220, 40) if plate.name == "good" else (255, 60, 160)
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA, shift=2)
        uv, _ = proj(np.array([[plate.pos[0], plate.pos[1], 0.0]]))
        cv2.drawMarker(img, tuple(int(v) for v in uv[0]), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(
            img,
            plate.name,
            (int(uv[0, 0]) + 6, int(uv[0, 1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    extra = [
        BrickState(
            d["object_id"],
            models[part_of(d["object_id"]).object_id].part,
            d["pos"],
            d["quat"],
            [0, 0, 0],
            d.get("stable_pose"),
        )
        for d in (state.diagnostics.get("duplicates", []) if show_duplicates else [])
        if "quat" in d
    ]
    for b in [*state.bricks, *extra]:
        model = models[part_of(b.object_id).object_id]
        r = quat_to_matrix(b.quat)
        tri = model.triangles @ r.T + np.asarray(b.pos)
        uv, z = proj(tri.reshape(-1, 3))
        if (z <= 0).any():
            continue
        polys = np.round(uv.reshape(-1, 3, 2) * 4).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, list(polys), 255, cv2.LINE_8, shift=2)
        layer[mask > 0] = (0, 255, 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
        origin_uv, _ = proj(np.asarray(b.pos)[None])
        tag = b.object_id if b not in extra else f"dup {b.part}"
        label = f"{tag} {b.stable_pose or 'free'} {b.color_name or ''}".strip()
        cv2.putText(
            img,
            label,
            (int(origin_uv[0, 0]) + 8, int(origin_uv[0, 1]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(layer, 0.25, img, 0.75, 0)
