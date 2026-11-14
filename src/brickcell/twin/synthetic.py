"""Synthetic RGB-D frames from the MuJoCo world, for perception tests with a known truth.

Needs the MuJoCo environment (``brickcell.simulation.world``). The frames use the cameras of
the scene config (``overhead`` and ``side`` are the two real D415s), so a test sees the same
views as the real station.

Depth: the MuJoCo renderer gives depth in metres along the optical axis (z-depth), which is
the RealSense convention. ``check_depth_convention`` measures this on a render of the table.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from brickcell.simulation.world import OBJECTS, World
from brickcell.twin.state import (
    RGBDFrame,
    body_pose_from_mesh_pose,
    matrix_to_quat,
    quat_to_matrix,
    rot_z,
)

PARK = np.array(
    [3.0, 3.0, 0.5]
)  # far from every camera; bricks and plates not in the scene go here


@dataclass(frozen=True)
class DepthNoise:
    """A simple D415-like depth error model.

    sigma_at_1m: standard deviation of the depth noise at 1 m (metres). It grows with z^2,
      as for any stereo camera (D415: baseline 55 mm, subpixel about 0.08 px -> ~1.5 mm at 1 m
      for one frame; the median of 20 frames has less noise).
    edge_holes: probability that a pixel at a depth edge has no depth.
    edge_step: depth change (m) to a neighbour pixel that marks a depth edge.
    dropout: probability that any pixel has no depth.
    quantum: depth step (m); the D415 depth unit is 1 mm.
    min_depth: pixels nearer than this have no depth (D415 at 1280x720: about 0.3-0.45 m).
    """

    sigma_at_1m: float = 0.0015
    edge_holes: float = 0.5
    edge_step: float = 0.01
    dropout: float = 0.01
    quantum: float = 0.001
    min_depth: float = 0.3

    def apply(self, depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        d = depth.astype(np.float64).copy()
        valid = d > 0
        d[valid] += rng.normal(size=int(valid.sum())) * self.sigma_at_1m * d[valid] ** 2
        if self.quantum > 0:
            d = np.round(d / self.quantum) * self.quantum
        step = np.zeros(d.shape, bool)
        dx = np.abs(np.diff(depth, axis=1)) > self.edge_step
        dy = np.abs(np.diff(depth, axis=0)) > self.edge_step
        step[:, 1:] |= dx
        step[:, :-1] |= dx
        step[1:, :] |= dy
        step[:-1, :] |= dy
        holes = (step & (rng.random(d.shape) < self.edge_holes)) | (
            rng.random(d.shape) < self.dropout
        )
        d[holes | (d < self.min_depth)] = 0.0
        return d.astype(np.float32)


NOISE_LEVELS = {
    "none": None,
    "d415": DepthNoise(),
    "d415x2": DepthNoise(sigma_at_1m=0.003, edge_holes=0.7, dropout=0.03),
}


def _renderer(world: World, width: int, height: int) -> mujoco.Renderer:
    key = (width, height)
    renderer = world.renderers.get(key)
    if renderer is None:
        renderer = mujoco.Renderer(world.model, height=height, width=width)
        world.renderers[key] = renderer
    return renderer


def set_scene(
    world: World,
    bricks: Mapping[str, tuple] | None = None,
    plates: Mapping[str, tuple] | None = None,
    arm_qpos: Mapping[str, float] | None = None,
) -> None:
    """Put the bricks (mesh-frame poses) and the plates (xy) in the world, then mj_forward.

    Bricks and plates that are not given go to a parking place far from the cameras.
    """
    bricks = dict(bricks or {})
    plates = dict(plates or {})
    m, d = world.model, world.data
    for i, object_id in enumerate(OBJECTS):
        joint = m.joint(f"{object_id}-joint")
        a, v = joint.qposadr[0], joint.dofadr[0]
        if object_id in bricks:
            mesh_pos, mesh_quat = bricks[object_id]
            pos, quat = body_pose_from_mesh_pose(object_id, mesh_pos, mesh_quat)
        else:
            pos, quat = PARK + [0.2 * i, 0, 0], np.array([1.0, 0, 0, 0])
        d.qpos[a : a + 3] = pos
        d.qpos[a + 3 : a + 7] = quat
        d.qvel[v : v + 6] = 0
    for name in ("good", "defective"):
        body = m.body(f"plate_{name}").id
        if name in plates:
            x, y = plates[name][:2]
            m.body_pos[body] = [x, y, 0.0]
        else:
            m.body_pos[body] = PARK + [0, 0.5 if name == "good" else 1.0, -0.5]
    for joint_name, value in (arm_qpos or {}).items():
        d.qpos[m.joint(joint_name).qposadr[0]] = value
    mujoco.mj_forward(m, d)


def render_frames(
    world: World,
    cameras=("overhead", "side"),
    noise: DepthNoise | str | None = None,
    seed: int = 0,
) -> list[RGBDFrame]:
    """RGB-D frames of the current world state from the named model cameras."""
    if isinstance(noise, str):
        noise = NOISE_LEVELS[noise]
    rng = np.random.default_rng(seed)
    info = {c["name"]: c for c in world.cameras()}
    frames = []
    now = time.time()
    for name in cameras:
        if name not in info:
            raise ValueError(f"Unknown camera '{name}'. Cameras: {sorted(info)}")
        c = info[name]
        renderer = _renderer(world, int(c["width"]), int(c["height"]))
        renderer.update_scene(world.data, camera=name)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        try:
            renderer.update_scene(world.data, camera=name)
            depth = renderer.render().astype(np.float32).copy()
        finally:
            renderer.disable_depth_rendering()
        far = depth >= 0.99 * float(world.model.stat.extent * world.model.vis.map.zfar)
        depth[far | ~np.isfinite(depth)] = 0.0
        if noise is not None:
            depth = noise.apply(depth, rng)
        frames.append(
            RGBDFrame(
                camera=name,
                rgb=rgb,
                depth_m=depth,
                fx=float(c["fx"]),
                fy=float(c["fy"]),
                cx=float(c["cx"]),
                cy=float(c["cy"]),
                timestamp=now,
                serial=f"sim-{name}",
                synthetic=True,
            )
        )
    return frames


def render_scene(
    world: World,
    bricks: Mapping[str, tuple] | None = None,
    plates: Mapping[str, tuple] | None = None,
    cameras=("overhead", "side"),
    noise: DepthNoise | str | None = None,
    seed: int = 0,
    arm_qpos: Mapping[str, float] | None = None,
) -> list[RGBDFrame]:
    """Render RGB-D frames of a known scene.

    bricks: {object_id: (mesh_pos, mesh_quat)}, the pose of the MESH frame in the world (the
      convention of state.BrickState). plates: {"good" | "defective": (x, y)}.
    noise: None, a DepthNoise, or a name in NOISE_LEVELS ("none", "d415", "d415x2").
    """
    set_scene(world, bricks, plates, arm_qpos)
    return render_frames(world, cameras, noise, seed)


STABLE_POSES = Path(__file__).resolve().parents[3] / "assets" / "duplo" / "stable_poses.json"
PLATE_FLOOR_Z = 0.0031  # top of the plate collision floor in world.plate_xml
# Arm pose that keeps the SO-101 out of the brick area (it points to +Y, near the base).
ARM_AWAY = {"shoulder_pan": 1.4, "shoulder_lift": -1.6, "elbow_flex": 1.5, "wrist_flex": 1.0}


def _mesh_vertices(object_id: str) -> np.ndarray:
    from brickcell.twin.geometry import read_stl
    from brickcell.twin.state import PARTS

    return read_stl(STABLE_POSES.parent / f"{PARTS[object_id].mesh}.stl").reshape(-1, 3)


def random_scene(
    rng: np.random.Generator,
    region=((0.22, 0.50), (-0.22, 0.12)),
    n_in_plate: int = 1,
    gap: float = 0.012,
    attempts: int = 200,
) -> tuple[dict, dict, dict]:
    """A random scene with the four parts in random stable poses and random yaws.

    Returns (bricks, plates, truth). bricks: {object_id: (mesh_pos, mesh_quat)}; plates:
    {"good": (x, y), "defective": (x, y)}; truth: {object_id: {"pose": id, "plate": name|None}}.
    The first ``n_in_plate`` bricks stand on a plate floor, the others on the table. Bricks do
    not touch each other or a plate (``gap`` between their footprint circles).
    """
    stable = json.loads(STABLE_POSES.read_text())["bricks"]
    (x0, x1), (y0, y1) = region
    for _ in range(attempts):
        a = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        b = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        if math.dist(a, b) < 0.21:
            continue
        plates = {"good": a, "defective": b}
        ids = list(OBJECTS)
        rng.shuffle(ids)
        bricks, truth, placed = {}, {}, []  # placed: (centre xy, radius)
        for k, object_id in enumerate(ids):
            poses = stable[object_id]["poses"]
            pose = poses[int(rng.integers(len(poses)))]
            r = rot_z(rng.uniform(-math.pi, math.pi)) @ quat_to_matrix(pose["quat"])
            v = _mesh_vertices(object_id) @ r.T
            lo, hi = v.min(0), v.max(0)
            centre_offset = (lo[:2] + hi[:2]) / 2
            radius = float(np.linalg.norm(hi[:2] - lo[:2]) / 2)
            name = ("good" if k % 2 == 0 else "defective") if k < n_in_plate else None
            for _ in range(attempts):
                if name:
                    ang, rad = rng.uniform(0, 2 * math.pi), rng.uniform(0, max(0.0, 0.074 - radius))
                    c = (
                        plates[name][0] + rad * math.cos(ang),
                        plates[name][1] + rad * math.sin(ang),
                    )
                else:
                    c = (rng.uniform(x0, x1), rng.uniform(y0, y1))
                    if any(math.dist(c, p) < 0.094 + radius + gap for p in plates.values()):
                        continue
                if all(math.dist(c, q) > radius + rq + gap for q, rq in placed):
                    break
            else:
                break  # no place for this brick: try a new layout
            placed.append((c, radius))
            z = pose["rest_height"] + (PLATE_FLOOR_Z if name else 0.0)
            origin = np.array([c[0] - centre_offset[0], c[1] - centre_offset[1], z])
            bricks[object_id] = (origin, matrix_to_quat(r))
            truth[object_id] = {"pose": pose["id"], "plate": name}
        if len(bricks) == len(OBJECTS):
            return bricks, plates, truth
    raise RuntimeError("random_scene: no layout found; use a larger region")


def check_depth_convention(world: World, camera: str = "overhead") -> dict:
    """Compare the rendered depth of the empty table with the z-depth of the table plane.

    Returns the median and the maximum absolute error (m) for z-depth and for ray distance.
    The z-depth error must be near zero.
    """
    set_scene(world, {}, {})
    frame = render_frames(world, (camera,))[0]
    cam = {c["name"]: c for c in world.cameras()}[camera]
    r = np.asarray(cam["world_rotation_cv"])
    p = np.asarray(cam["world_pos"])
    h, w = frame.depth_m.shape
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack(
        ((u - frame.cx) / frame.fx, (v - frame.cy) / frame.fy, np.ones_like(u, float)), -1
    )
    world_dir = rays @ r.T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -p[2] / world_dir[..., 2]  # z-depth to the table plane z = 0
    ok = (frame.depth_m > 0) & (s > 0) & np.isfinite(s)
    ray_len = s * np.linalg.norm(rays, axis=-1)
    dz = np.abs(frame.depth_m[ok] - s[ok])
    dr = np.abs(frame.depth_m[ok] - ray_len[ok])
    return {
        "pixels": int(ok.sum()),
        "zdepth_median_err": float(np.median(dz)),
        "zdepth_max_err": float(np.max(dz)),
        "ray_median_err": float(np.median(dr)),
    }
