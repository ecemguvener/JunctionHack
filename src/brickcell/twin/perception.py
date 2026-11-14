"""Perception for the station twin: RGB-D frames of the D415 cameras -> StationState.

Imports only numpy, OpenCV, PyYAML and the standard library (no SciPy, no MuJoCo), so it runs
in the core, MuJoCo and RealSense environments.

Method (``locate``)
 1. Views. Each depth image becomes world points with the camera pose of the scene config
    (OpenCV camera frame). Points outside the workspace box are removed.
 2. Table check. A RANSAC plane fit on the points near z = 0 gives the table plane of each
    camera. The offset and the tilt go to ``cameras``. Small errors are corrected: the camera
    pose is moved so that its table plane is z = 0.
 3. Plates. The KALAS wall has a known radius at each height. A Hough vote with that radius
    finds the plate centres, also for a plate that is partly out of view. A Gauss-Newton fit
    refines each centre. The mean rim colour gives the name ("good" / "defective").
 4. Segments. Points above the table (or above a plate floor) go into a 2 mm world-XY grid.
    Connected cells are one cluster. A cluster is split where the colour changes, because two
    bricks that touch usually have different colours. Segments that are too large, too high or
    that float are rejected (arm, can, laptop, box).
 5. Fit. For each segment, each part and each stable pose of stable_poses.json, an ICP fit
    (point-to-plane, 4 DoF: x, y, z, yaw) starts from the segment footprint. The score is
    (fraction of segment points on the model) x (fraction of the model points that a camera
    must see and that agree with the depth image). The second term is a render-and-compare
    check: it uses only model points that face a camera and are not hidden by the model
    itself, and it does not count points that something in front hides.
    An optional 6-DoF refinement accepts a pose that is not stable if it is much better.
 6. Bricks. Fits that share points: the best one stays. The sim has one body per part, so the
    best fit for each part is the brick; the others go to ``diagnostics["duplicates"]``.
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
    PARTS,
    BrickState,
    PlateState,
    RGBDFrame,
    StationState,
    matrix_to_quat,
    quat_to_matrix,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "twin.yaml"
ASSETS = ROOT / "assets" / "duplo"

# KALAS plate wall: radius (m) of the middle of the wall at a height z (m). From the inner and
# the outer surface of world.PLATE_PROFILE (tests/test_twin_perception.py checks it).
PLATE_WALL_Z = np.array([0.004, 0.0065, 0.011, 0.0165, 0.0215, 0.0235])
PLATE_WALL_R = np.array([0.0802, 0.08325, 0.08625, 0.0891, 0.09165, 0.09215])

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
        },
        "max_color_distance": 20.0,  # Lab distance (lightness weight 0.35)
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
        "max_points": 300,
        "coarse_iterations": 4,
        "coarse_points": 120,
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
    },
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
    points: np.ndarray  # (N, 3) world points in the workspace
    pixels: np.ndarray  # (N,) flat pixel index
    rgb: np.ndarray  # (N, 3) uint8
    info: dict


def _prepare_view(frame: RGBDFrame, camera, cfg: dict, seed: int) -> View:
    rotation = np.asarray(camera.rotation_cv(), float)
    position = np.asarray(camera.pos, float)
    d = cfg["depth"]
    cam_points, pixels = geo.deproject(
        frame.depth_m, frame.fx, frame.fy, frame.cx, frame.cy, d["stride"], d["max_depth"]
    )
    valid_fraction = float(np.mean((frame.depth_m > 0) & (frame.depth_m < d["max_depth"])))
    world = cam_points @ rotation.T + position
    lo, hi = np.asarray(cfg["workspace"]["min"]), np.asarray(cfg["workspace"]["max"])
    inside = np.all((world >= lo) & (world <= hi), axis=1)
    world, pixels = world[inside], pixels[inside]
    info: dict = {
        "serial": frame.serial,
        "timestamp": frame.timestamp,
        "synthetic": frame.synthetic,
        "valid_depth_fraction": round(valid_fraction, 4),
        "points_in_workspace": int(len(world)),
    }
    t = cfg["table"]
    band = np.abs(world[:, 2]) < t["fit_band"]
    plane = geo.fit_plane_ransac(world[band], t["ransac_threshold"], t["ransac_iterations"], seed)
    if plane is not None:
        n, dist, inliers = plane
        pivot = np.median(world[band][inliers], axis=0)
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
            world = world @ r_c.T + t_c
            rotation = r_c @ rotation
            position = r_c @ position + t_c
            info["corrected_pose"] = {
                "pos": [round(float(v), 5) for v in position],
                "rotation_cv": [[round(float(v), 6) for v in row] for row in rotation],
            }
    else:
        info["table_plane"] = None
    rgb = frame.rgb.reshape(-1, 3)[pixels]
    return View(frame.camera, frame, rotation, position, world, pixels, rgb, info)


# ----------------------------------------------------------------------------- plates


def _plate_wall_radius(z: np.ndarray) -> np.ndarray:
    return np.interp(z, PLATE_WALL_Z, PLATE_WALL_R)


def _color_list(values) -> list[np.ndarray]:
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


def _find_plates(views: list[View], cfg: dict) -> tuple[list[PlateFit], list[dict]]:
    p = cfg["plates"]
    if not p["detect"]:
        return [], []
    pts = np.concatenate([v.points for v in views])
    rgb = np.concatenate([v.rgb for v in views])
    band = (pts[:, 2] > p["z_band"][0]) & (pts[:, 2] < p["z_band"][1])
    pts, rgb = pts[band], rgb[band]
    rejected: list[dict] = []
    if len(pts) < p["min_inliers"]:
        return [], rejected
    rng = np.random.default_rng(0)
    if len(pts) > 15000:
        keep = rng.choice(len(pts), 15000, replace=False)
        pts, rgb = pts[keep], rgb[keep]
    xy, radii = pts[:, :2], _plate_wall_radius(pts[:, 2])
    lo = np.asarray(cfg["workspace"]["min"][:2]) - 0.1
    hi = np.asarray(cfg["workspace"]["max"][:2]) + 0.1
    cell = p["hough_cell"]
    acc, origin = geo.circle_hough(xy, radii, lo, hi, cell, angles=32)
    acc = cv2.GaussianBlur(acc, (5, 5), 1.0)
    colors = {name: _color_list(values) for name, values in p["colors"].items()}
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
        if near.sum() < p["min_inliers"]:
            if len(plates) == 0 and near.sum() < 10:
                break
            rejected.append(
                {
                    "centre": centre.round(4).tolist(),
                    "reason": "too few wall points",
                    "inliers": int(near.sum()),
                }
            )
            continue
        centre, scale = geo.fit_circle_known_radius(
            xy[near], radii[near], centre, scale_range=tuple(p["scale_range"])
        )
        inl = np.abs(np.linalg.norm(xy - centre, axis=1) - scale * radii) < p["inlier_tol"]
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
        jx, jy = np.rint((centre - origin) / cell).astype(int)
        sup = int(round(p["min_separation"] / cell))
        acc[max(0, jy - sup) : jy + sup + 1, max(0, jx - sup) : jx + sup + 1] = 0
        conf = min(1.0, arc / 180.0) * (1.0 if name else 0.5)
        plates.append(PlateFit(centre, int(inl.sum()), arc, med / 255.0, name, best_d, conf, scale))
    return plates, rejected


def _check_agreement(views: list[View], cfg: dict) -> tuple[list[View], dict]:
    """Compare the plate centres that each camera finds with those of the reference camera.

    A camera with a wrong pose puts the plates at a different place. Such a camera is not
    fused (the bricks would get a mix of two positions). Returns the views to use and a report.
    """
    c = cfg["cameras"]
    report: dict = {"reference": c["reference"], "cameras": {}}
    views = [v for v in views if v.name not in c["exclude"]]
    ref = next((v for v in views if v.name == c["reference"]), None)
    if ref is None or len(views) < 2:
        report["note"] = "no check: the reference camera or a second camera is missing"
        return views, report
    ref_plates, _ = _find_plates([ref], cfg)
    used = [ref]
    for v in views:
        if v is ref:
            continue
        own, _ = _find_plates([v], cfg)
        offsets = []
        for p in own:
            if not ref_plates:
                break
            dist = [np.linalg.norm(p.centre - q.centre) for q in ref_plates]
            k = int(np.argmin(dist))
            if dist[k] < 0.15:
                offsets.append((p.centre - ref_plates[k].centre).round(4).tolist())
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
    return used, report


# ----------------------------------------------------------------------------- segments


@dataclass
class Segment:
    points: np.ndarray  # (N, 3) world
    rgb: np.ndarray  # (N, 3) uint8
    view_index: np.ndarray  # (N,) index into views
    cells: np.ndarray  # (M, 2) grid cells (ix, iy)
    support_z: float
    plate: int | None  # index into the plates, or None
    cluster: int
    local_points: np.ndarray  # cluster points near the segment (the fit cloud)
    local_view_index: np.ndarray


def _groups(labels: np.ndarray) -> dict[int, np.ndarray]:
    """{label: indices} for an integer label array (one sort, not one pass per label)."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    ends = np.r_[starts[1:], len(order)]
    return {int(sorted_labels[a]): order[a:b] for a, b in zip(starts, ends)}


def _segments(
    views: list[View], plates: list[PlateFit], cfg: dict
) -> tuple[list[Segment], list[dict], dict]:
    s, p = cfg["segment"], cfg["plates"]
    pts = np.concatenate([v.points for v in views])
    rgb = np.concatenate([v.rgb for v in views])
    vidx = np.concatenate([np.full(len(v.points), i) for i, v in enumerate(views)])
    support = np.zeros(len(pts))
    plate_of = np.full(len(pts), -1)
    keep = np.ones(len(pts), bool)
    lab_all = geo.rgb_to_lab(rgb)
    for k, plate in enumerate(plates):
        r = np.linalg.norm(pts[:, :2] - plate.centre, axis=1) / plate.scale
        inner = r < p["inner_radius"]
        support[inner] = p["floor_z"]
        plate_of[inner] = k
        keep &= ~((r >= p["inner_radius"]) & (r <= p["outer_radius"]) & (pts[:, 2] < p["rim_top"]))
        # Depth edges are soft: the floor next to a brick can look raised. Points with the
        # colour of this plate floor are floor, not brick.
        floor = inner & (pts[:, 2] < p["floor_z"] + 0.003)
        if floor.sum() > 50 and s["floor_color_tol"] > 0:
            floor_lab = np.median(lab_all[floor], axis=0)
            keep &= ~(inner & (geo.color_distance(lab_all, floor_lab) < s["floor_color_tol"]))
    above = np.where(plate_of >= 0, s["min_height_plate"], s["min_height"])
    keep &= pts[:, 2] > support + above
    pts, rgb, vidx, support, plate_of = (
        pts[keep],
        rgb[keep],
        vidx[keep],
        support[keep],
        plate_of[keep],
    )
    stats = {"points_above_support": int(len(pts))}
    if len(pts) == 0:
        return [], [], stats
    cell = s["cell"]
    lo = np.asarray(cfg["workspace"]["min"][:2])
    ij = np.floor((pts[:, :2] - lo) / cell).astype(np.int64)
    nx = int(ij[:, 0].max()) + 2
    ny = int(ij[:, 1].max()) + 2
    flat = ij[:, 1] * nx + ij[:, 0]
    count = np.bincount(flat, minlength=nx * ny).reshape(ny, nx).astype(np.float32)
    occ = (count > 0).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n_clusters, cluster_map = cv2.connectedComponents(occ, connectivity=8)
    lab = geo.rgb_to_lab(rgb)
    lab_sum = np.stack(
        [np.bincount(flat, lab[:, c], minlength=nx * ny).reshape(ny, nx) for c in range(3)], -1
    ).astype(np.float32)
    # Mean colour per cell over a 3x3 window (less noise, but a colour edge stays sharp).
    sm_count = cv2.boxFilter(count, -1, (3, 3), normalize=False)
    sm_lab = np.stack(
        [cv2.boxFilter(lab_sum[..., c], -1, (3, 3), normalize=False) for c in range(3)], -1
    )
    mean_lab = sm_lab / np.maximum(sm_count, 1e-6)[..., None]
    has = count > 0
    # A cell is on a boundary if its colour differs from a cell 1 or 2 steps away.
    boundary = np.zeros_like(has)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)):
        ys = slice(0, ny - dy), slice(dy, ny)
        xs = (slice(0, nx - dx), slice(dx, nx)) if dx >= 0 else (slice(-dx, nx), slice(0, nx + dx))
        a = mean_lab[ys[0], xs[0]]
        b = mean_lab[ys[1], xs[1]]
        both = has[ys[0], xs[0]] & has[ys[1], xs[1]]
        diff = both & (geo.color_distance(a, b) > s["color_split"])
        boundary[ys[0], xs[0]] |= diff
        boundary[ys[1], xs[1]] |= diff
    core = (has & ~boundary).astype(np.uint8)
    n_core, seg_map = cv2.connectedComponents(core, connectivity=4)
    seg_map = seg_map.astype(np.int32)
    # Boundary cells and closed gaps go to the neighbour segment with the nearest colour.
    seg_lab = np.zeros((n_core, 3), np.float32)
    seg_n = np.bincount(seg_map.reshape(-1), minlength=n_core).astype(np.float32)
    for c in range(3):
        seg_lab[:, c] = np.bincount(
            seg_map.reshape(-1), mean_lab[..., c].reshape(-1), minlength=n_core
        )
    seg_lab /= np.maximum(seg_n, 1)[:, None]
    for _ in range(4):
        todo = (seg_map == 0) & (occ > 0)
        if not todo.any():
            break
        best = np.zeros_like(seg_map)
        best_d = np.full(seg_map.shape, np.inf, np.float32)
        padded = np.pad(seg_map, 1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cand = padded[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx]
                d = geo.color_distance(mean_lab, seg_lab[cand]).astype(np.float32)
                better = todo & (cand > 0) & (d < best_d)
                best[better] = cand[better]
                best_d[better] = d[better]
        seg_map[todo & (best > 0)] = best[todo & (best > 0)]
    point_seg = seg_map.reshape(-1)[flat]
    point_cluster = cluster_map.reshape(-1)[flat]
    seg_groups = _groups(point_seg)
    cluster_groups = _groups(point_cluster)
    segments: list[Segment] = []
    rejected: list[dict] = []
    for label, sel in seg_groups.items():
        if label == 0:
            continue
        cells = np.unique(ij[sel], axis=0)
        if len(cells) < s["min_cells"]:
            continue  # noise and small leftovers: not reported
        sp = pts[sel]
        centre = sp[:, :2].mean(0)
        (_, _), (w, h), _ = cv2.minAreaRect((cells * cell).astype(np.float32))
        extent = float(max(w, h))
        # Cluster points near the segment: the same brick can have more than one colour
        # segment (shade), so the fit and the support test use these points.
        cluster = int(np.bincount(point_cluster[sel]).argmax())
        near = cluster_groups[cluster]
        near = near[np.linalg.norm(pts[near, :2] - centre, axis=1) < extent / 2 + s["local_margin"]]
        sup = float(np.median(support[sel]))
        top = float(np.percentile(sp[:, 2], 98)) - sup
        bottom = float(np.percentile(pts[near, 2], 2)) - sup
        info = {
            "centre": centre.round(4).tolist(),
            "cells": int(len(cells)),
            "points": int(len(sel)),
            "top_mm": round(top * 1000, 1),
            "bottom_mm": round(bottom * 1000, 1),
            "extent_mm": round(extent * 1000, 1),
            "color": (np.median(rgb[sel], axis=0) / 255).round(3).tolist(),
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
        segments.append(
            Segment(sp, rgb[sel], vidx[sel], cells, sup, plate, cluster, pts[near], vidx[near])
        )
    stats["clusters"] = int(n_clusters - 1)
    stats["segments"] = int(len(segments))
    return segments, rejected, stats


# ----------------------------------------------------------------------------- fitting


@dataclass
class Fit:
    object_id: str
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
    inliers: np.ndarray | None = None  # mask over the segment points
    segment: Segment | None = None
    quality: float = 0.0  # explained x coverage: used for acceptance and confidence
    iou: float | None = None  # colour silhouette IoU: used only to rank the hypotheses


def _icp(
    model: PartModel,
    cloud: np.ndarray,
    rotation: np.ndarray,
    position: np.ndarray,
    iterations: int,
    f: dict,
    dof: int = 4,
    z0: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Robust point-to-plane ICP of the model to the cloud. Returns (R, t, mean residual)."""
    r, t = rotation.copy(), position.copy()
    pivot = cloud.mean(0)
    huber, max_corr = f["huber"], f["max_corr"]
    rms = float("inf")
    for _ in range(iterations):
        local = (cloud - t) @ r
        idx = model.grid.lookup(local)
        ok = idx >= 0
        if ok.sum() < 10:
            break
        c = model.samples[idx[ok]] @ r.T + t
        n = model.normals[idx[ok]] @ r.T
        p = cloud[ok]
        diff = p - c
        dist = np.linalg.norm(diff, axis=1)
        res = np.einsum("ij,ij->i", n, diff)
        w = np.where(dist < max_corr, 1.0, 0.0) * np.where(
            np.abs(res) < huber, 1.0, huber / np.maximum(np.abs(res), 1e-12)
        )
        if w.sum() < 10:
            break
        a = c - pivot
        axn = np.column_stack(
            (
                a[:, 1] * n[:, 2] - a[:, 2] * n[:, 1],
                a[:, 2] * n[:, 0] - a[:, 0] * n[:, 2],
                a[:, 0] * n[:, 1] - a[:, 1] * n[:, 0],
            )
        )
        if dof == 4:
            jac = np.column_stack((-axn[:, 2], -n))
        else:
            jac = np.column_stack((-axn, -n))
        h = jac.T @ (jac * w[:, None])
        g = jac.T @ (w * res)
        if dof == 4 and z0 is not None and f["z_prior"] > 0:
            lam = f["z_prior"] * w.sum()
            h[3, 3] += lam
            g[3] += lam * (t[2] - z0)
        h += np.eye(len(h)) * 1e-9 * max(1.0, np.trace(h))
        try:
            step = -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        if dof == 4:
            dr = geo.rot_z(step[0])
            v = step[1:]
        else:
            dr = geo.rodrigues(step[:3])
            v = step[3:]
        r = dr @ r
        t = dr @ (t - pivot) + pivot + v
        rms = float(np.sqrt(np.sum(w * res**2) / w.sum()))
        if np.linalg.norm(step) < 1e-6:
            break
    # Keep the rotation exact.
    u, _, vt = np.linalg.svd(r)
    return u @ vt, t, rms


def _visible_check(
    model: PartModel, r, t, views: list[View], f: dict, which: list[int], stride: int = 1
) -> tuple[int, int, dict, list[np.ndarray]]:
    """Render-and-compare: model points a camera must see, against the measured depth.

    Returns (matched, violated, per-camera counts, visible world points per camera). A model
    point is visible if it is the nearest model point in its pixel bin (bin of
    ``visibility_bin_px``). It is matched if the depth image agrees within the tolerance,
    violated if the camera sees through it (measured depth is farther), and not counted if
    something nearer hides it or there is no depth.
    """
    world = model.score_samples[::stride] @ r.T + t
    matched = violated = 0
    per_cam = {}
    visible_pts = []
    b = f["visibility_bin_px"]
    for i in which:
        v = views[i]
        fr = v.frame
        uv, z = geo.project(world, v.rotation, v.position, fr.fx, fr.fy, fr.cx, fr.cy)
        h, w = fr.depth_m.shape
        u = np.rint(uv[:, 0]).astype(np.int64)
        vv = np.rint(uv[:, 1]).astype(np.int64)
        ok = (z > 0.05) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        if ok.sum() == 0:
            per_cam[v.name] = (0, 0)
            visible_pts.append(np.zeros((0, 3)))
            continue
        u, vv, z, pts = u[ok], vv[ok], z[ok], world[ok]
        bins = (vv // b) * (w // b + 1) + (u // b)
        order = np.lexsort((z, bins))
        sb = bins[order]
        first = np.r_[True, sb[1:] != sb[:-1]]
        zmin = np.repeat(z[order][first], np.diff(np.r_[np.flatnonzero(first), len(sb)]))
        front = np.zeros(len(z), bool)
        front[order] = z[order] <= zmin + 0.002
        u, vv, z, pts = u[front], vv[front], z[front], pts[front]
        d = fr.depth_m[vv, u].astype(float)
        has = d > 0
        tol = f["depth_tol"] + f["depth_tol_z2"] * z**2
        m = has & (np.abs(z - d) <= tol)
        x = has & (z < d - tol)
        matched += int(m.sum())
        violated += int(x.sum())
        per_cam[v.name] = (int(m.sum()), int(x.sum()))
        visible_pts.append(pts[m])
    return matched, violated, per_cam, visible_pts


def _color_masks(seg: Segment, views: list[View], f: dict, which: list[int]) -> dict:
    """Per view: a window of the colour image around the segment and the brick-colour mask.

    The colour image is sharp, but the depth of a small brick is soft. The mask gives the
    brick outline: pixels near the median colour of the segment.
    """
    masks = {}
    if f["silhouette_weight"] <= 0:
        return masks
    seg_lab = np.median(geo.rgb_to_lab(seg.rgb), axis=0)
    for i in which:
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
        lab = geo.rgb_to_lab(fr.rgb[y0:y1, x0:x1])
        mask = geo.color_distance(lab, seg_lab) < f["silhouette_color_tol"]
        mask = (
            cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        )
        masks[i] = (x0, y0, mask)
    return masks


def _silhouette_iou(
    model: PartModel, r, t, views: list[View], masks: dict, stride: int
) -> float | None:
    """Mean IoU of the projected model and the brick-colour mask over the views."""
    ious = []
    world = model.score_samples[::stride] @ r.T + t
    for i, (x0, y0, mask) in masks.items():
        v = views[i]
        fr = v.frame
        uv, z = geo.project(world, v.rotation, v.position, fr.fx, fr.fy, fr.cx, fr.cy)
        u = np.rint(uv[:, 0]).astype(int) - x0
        vv = np.rint(uv[:, 1]).astype(int) - y0
        ok = (z > 0) & (u >= 0) & (u < mask.shape[1]) & (vv >= 0) & (vv < mask.shape[0])
        sil = np.zeros(mask.shape, np.uint8)
        sil[vv[ok], u[ok]] = 1
        k = 3 if stride == 1 else 5
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
        union = (sil | mask).sum()
        if union:
            ious.append((sil & mask).sum() / union)
    return float(np.mean(ious)) if ious else None


def _score(
    model: PartModel, r, t, seg: Segment, views, f, which, stride: int = 1, masks=None
) -> Fit:
    local = (seg.points - t) @ r
    idx = model.grid.lookup(local)
    dist = np.full(len(local), np.inf)
    ok = idx >= 0
    dist[ok] = np.linalg.norm(local[ok] - model.samples[idx[ok]], axis=1)
    inl = dist < f["inlier"]
    explained = float(inl.mean()) if len(inl) else 0.0
    matched, violated, per_cam, _ = _visible_check(model, r, t, views, f, which, stride)
    seen = matched + violated
    coverage = matched / seen if seen else 0.0
    quality = explained * coverage if seen * stride >= f["min_visible"] else 0.0
    score = quality
    iou = _silhouette_iou(model, r, t, views, masks, stride) if masks else None
    if iou is not None:
        score *= iou ** f["silhouette_weight"]
    cams = [
        views[i].name
        for i in which
        if per_cam.get(views[i].name, (0, 0))[0] > 0.3 * f["min_visible"]
    ]
    return Fit(
        model.object_id,
        None,
        r,
        t,
        explained,
        coverage,
        seen,
        score,
        None,
        0.0,
        cams,
        inl,
        seg,
        quality,
        iou,
    )


def _initial_poses(
    model: PartModel, seg: Segment, f: dict, rect
) -> list[tuple[PoseHypothesis, float, float]]:
    """(pose, yaw, z) start values that agree with the segment height and footprint."""
    (cxy, (w, h), angle) = rect
    long_angle = math.radians(angle) if w >= h else math.radians(angle + 90)
    seg_long = max(w, h)
    seg_short = min(w, h)
    top = float(np.percentile(seg.points[:, 2], 98)) - seg.support_z
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


def _fit_segment(
    models: dict[str, PartModel], seg: Segment, views: list[View], f: dict, rng
) -> tuple[Fit | None, list[dict]]:
    pts = seg.local_points
    if len(pts) > f["max_points"]:
        cloud = pts[rng.choice(len(pts), f["max_points"], replace=False)]
    else:
        cloud = rng.permutation(pts)
    coarse_cloud = cloud[: f["coarse_points"]]
    which = sorted(set(seg.local_view_index.tolist()))
    masks = _color_masks(seg, views, f, which)
    rect = cv2.minAreaRect(seg.points[:, :2].astype(np.float32))
    centre = np.array(rect[0], float)
    candidates = []
    for model in models.values():
        for pose, yaw, z in _initial_poses(model, seg, f, rect):
            r = geo.rot_z(yaw) @ pose.rotation
            t = np.array([*(centre - geo.rot_z(yaw)[:2, :2] @ pose.centre_xy), z])
            r1, t1, _ = _icp(model, coarse_cloud, r, t, f["coarse_iterations"], f, 4, z)
            fit = _score(model, r1, t1, seg, views, f, which, stride=3, masks=masks)
            fit.pose = pose
            candidates.append((fit, model, z))
    if not candidates:
        return None, [{"reason": "no part has a stable pose of this height and size"}]
    candidates.sort(key=lambda c: -c[0].score)
    best: Fit | None = None
    tried = []
    for fit, model, z in candidates[: f["keep_coarse"]]:
        r2, t2, rms = _icp(model, cloud, fit.rotation, fit.position, f["fine_iterations"], f, 4, z)
        if f["snap_z"] and abs(t2[2] - z) < f["snap_z_tol"]:
            t2 = np.array([t2[0], t2[1], z])
        final = _score(model, r2, t2, seg, views, f, which, masks=masks)
        final.pose, final.rms_mm = fit.pose, rms * 1000
        tried.append(_fit_summary(final))
        if best is None or final.score > best.score:
            best = final
    if best is not None and f["free_refine"]:
        model = models[best.object_id]
        r3, t3, rms = _icp(model, cloud, best.rotation, best.position, f["fine_iterations"], f, 6)
        free = _score(model, r3, t3, seg, views, f, which, masks=masks)
        angle = geo.rotation_angle_deg(r3, best.rotation)
        if free.score > best.score + f["free_min_gain"] and angle > f["free_min_angle_deg"]:
            free.rms_mm = rms * 1000
            tried.append({**_fit_summary(free), "free": True})
            best = free
    return best, tried


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


def _model_to_cloud_residual(
    model: PartModel, fit: Fit, views: list[View], f: dict
) -> float | None:
    which = sorted(set(fit.segment.local_view_index.tolist()))
    _, _, _, visible = _visible_check(model, fit.rotation, fit.position, views, f, which)
    pts = np.concatenate(visible) if visible else np.zeros((0, 3))
    if len(pts) < 5:
        return None
    _, dist = geo.nearest_neighbours(fit.segment.local_points, pts)
    return float(np.median(dist) * 1000)


# ----------------------------------------------------------------------------- main


def locate(frames: list[RGBDFrame], scene, config: dict | None = None) -> StationState:
    """Find the bricks and the plates in RGB-D frames. Returns a StationState (arm = None).

    frames: one RGBDFrame per camera; ``frame.camera`` must be a camera of ``scene``.
    scene: SceneConfig (brickcell.simulation.scene_config) with the camera poses.
    config: the perception config (``load_perception_config()``); None loads configs/twin.yaml.
    """
    times: dict[str, float] = {}
    t_start = time.perf_counter()
    cfg = load_perception_config() if config is None else _merge(DEFAULT_CONFIG, config)
    f = cfg["fit"]
    models = part_models(cfg)
    times["models"] = time.perf_counter() - t_start

    t0 = time.perf_counter()
    views = []
    for i, frame in enumerate(frames):
        camera = scene.cameras.get(frame.camera)
        if camera is None:
            raise ValueError(f"Camera '{frame.camera}' is not in the scene config")
        views.append(_prepare_view(frame, camera, cfg, seed=i))
    times["views"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    all_views = views
    views, agreement = _check_agreement(views, cfg)
    for v in all_views:
        v.info["used"] = any(v is u for u in views)
    plates, plates_rejected = _find_plates(views, cfg)
    times["plates"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    segments, seg_rejected, seg_stats = _segments(views, plates, cfg)
    times["segments"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    fits: list[Fit] = []
    rejected = list(seg_rejected)
    for seg in segments:
        fit, tried = _fit_segment(models, seg, views, f, rng)
        info = {
            "centre": seg.points[:, :2].mean(0).round(4).tolist(),
            "points": int(len(seg.points)),
            "color": (np.median(seg.rgb, axis=0) / 255).round(3).tolist(),
        }
        if fit is None:
            rejected.append({**info, **tried[0]})
            continue
        reason = None
        if fit.visible < f["min_visible"]:
            reason = "too few model points in view"
        elif fit.explained < f["min_explained"]:
            reason = "points do not lie on the part"
        elif fit.coverage < f["min_coverage"]:
            reason = "part does not agree with the depth image"
        elif fit.quality < f["min_score"]:
            reason = "low score"
        elif fit.iou is not None and fit.iou < f["min_silhouette_iou"]:
            reason = "outline does not match the brick colour in the image"
        if reason:
            rejected.append({**info, "reason": reason, "best": _fit_summary(fit)})
            continue
        fits.append(fit)
    times["fit"] = time.perf_counter() - t0

    # Two fits that explain the same points are the same brick: keep the better one.
    t0 = time.perf_counter()
    fits.sort(key=lambda x: -x.score)
    kept: list[Fit] = []
    for fit in fits:
        same = False
        for other in kept:
            if (
                np.linalg.norm(
                    fit.position[:2]
                    + fit.rotation[:2] @ _centre(models, fit)
                    - other.position[:2]
                    - other.rotation[:2] @ _centre(models, other)
                )
                < 0.015
            ):
                same = True
                break
        if not same:
            kept.append(fit)
    bricks: list[BrickState] = []
    duplicates: list[dict] = []
    used: set[str] = set()
    for fit in kept:
        model = models[fit.object_id]
        support = "table"
        centre_xy = fit.position[:2] + fit.rotation[:2] @ _centre(models, fit)
        for plate in plates:  # the footprint centre is inside the plate wall (or on it)
            if (
                np.linalg.norm(centre_xy - plate.centre)
                < cfg["plates"]["support_radius"] * plate.scale
            ):
                support = f"plate_{plate.name}" if plate.name else "other"
                break
        bottom = float((model.score_samples @ fit.rotation.T + fit.position)[:, 2].min())
        if bottom > fit.segment.support_z + 0.008:
            support = "other"
        inl = (
            fit.inliers
            if fit.inliers is not None and fit.inliers.any()
            else np.ones(len(fit.segment.points), bool)
        )
        color = (np.median(fit.segment.rgb[inl], axis=0) / 255.0).round(4).tolist()
        residual = _model_to_cloud_residual(model, fit, views, f)
        state = BrickState(
            object_id=fit.object_id,
            part=model.part,
            pos=[round(float(v), 5) for v in fit.position],
            quat=[round(float(v), 6) for v in matrix_to_quat(fit.rotation)],
            color=color,
            stable_pose=fit.pose.pose_id if fit.pose else None,
            support=support,
            confidence=round(float(fit.quality), 3),
            residual_mm=None if residual is None else round(residual, 2),
            cameras=fit.cameras,
        )
        if fit.object_id in used:
            duplicates.append(
                {
                    **_fit_summary(fit),
                    "color": color,
                    "support": support,
                    "stable_pose": state.stable_pose,
                }
            )
            continue
        used.add(fit.object_id)
        bricks.append(state)
    bricks.sort(key=lambda b: b.object_id)
    times["bricks"] = time.perf_counter() - t0

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
            plates_unnamed.append({**entry, "reason": "colour does not match a configured plate"})
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
            "camera_agreement": agreement,
            "segments": seg_stats,
            "rejected": rejected,
            "duplicates": duplicates,
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
            "plates_rejected": plates_rejected + plates_unnamed,
        },
    )


def _centre(models, fit: Fit) -> np.ndarray:
    """Centre of the part (mesh frame) that the NMS compares."""
    s = models[fit.object_id].samples
    return (s.min(0) + s.max(0)) / 2


# ----------------------------------------------------------------------------- overlays


def draw_overlay(
    frame: RGBDFrame,
    state: StationState,
    scene,
    config: dict | None = None,
    show_duplicates: bool = True,
) -> np.ndarray:
    """RGB image with the fitted bricks (filled outline of the mesh) and plate circles.

    Duplicates (a second brick of a part that is already used) are drawn too, with "dup".
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
            models[d["object_id"]].part,
            d["pos"],
            d["quat"],
            [0, 0, 0],
            d.get("stable_pose"),
        )
        for d in (state.diagnostics.get("duplicates", []) if show_duplicates else [])
        if "quat" in d
    ]
    for b in [*state.bricks, *extra]:
        model = models[b.object_id]
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
        label = f"{'dup ' if b in extra else ''}{b.part} {b.stable_pose or 'free'}"
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
