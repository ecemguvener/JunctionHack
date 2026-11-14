"""Stable resting poses of the DUPLO bricks, checked with MuJoCo, with top-down grasp data.

A pose is stable when a brick that is put on a flat surface in that pose stays at rest.

Method:
1. Centre of mass (COM): a real brick is a thin moulded shell, so the COM is the
   area-weighted centroid of the mesh triangles (uniform thin shell). The inertia is the
   thin-shell inertia. The volume centroid of the convex hull is recorded for comparison.
2. Analytical candidates: MuJoCo collides with the convex hull of each mesh, so we use the
   same hull. A hull facet is stable when the COM projects inside it. The probability is the
   quasi-static estimate: the solid angle of each facet, seen from the COM, flows to the
   stable facet that the brick topples to.
3. Empirical candidates: MuJoCo drops from random orientations, clustered by the body
   direction that points down.
4. Verification: each pose is put on a table plane and on a plate floor (a flat box) and is
   simulated; then tilted by a small angle in many directions to find marginal poses.
5. Grasp data for a top-down parallel gripper with jaws that close horizontally.

The friction, solver and mass values come from ``world.scene_xml`` at run time.

    MUJOCO_GL=egl PYTHONPATH=src python -m brickcell.simulation.stable_poses \\
        --output assets/duplo/stable_poses.json --image assets/duplo/stable_poses.png
"""

from __future__ import annotations

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "duplo"
DEFAULT_JSON = ASSETS / "stable_poses.json"

DROPS = 200  # random drops per brick
DROP_HEIGHT = (0.010, 0.020)  # lowest hull point above the surface at release (m)
CLUSTER_DEG = 3.0  # drops with down directions closer than this are one pose
HOLD_SECONDS = 5.0  # verification time
MAX_DRIFT = 0.0005  # m
MAX_ROTATION_DEG = 1.0
MAX_SPEED = 0.0001  # m/s net over the last 0.5 s; angular limit 0.01 rad/s
PERTURB_DEG = 2.0
PERTURB_DIRECTIONS = 12
GRIP_WIDTH = (0.010, 0.045)  # jaw opening limits (m)
GRIP_MIN_HEIGHT = 0.008  # grasp height above the table (m)
JAW_HALF_HEIGHT = 0.003  # half height of the jaw contact band (m)

# Body axis -> words for the face that is down. Keys are body axes of the face normal.
FACE_NAMES = {
    "3437": {"-Z": "upright", "+Z": "upside down on studs", "X": "on side", "Y": "on side"},
    "3011": {"-Z": "upright", "+Z": "upside down on studs", "X": "on end", "Y": "on long side"},
    "76371": {
        "-Z": "upright",
        "+Z": "upside down on studs",
        "X": "on narrow side (16 mm face)",
        "Y": "on wide side (32 mm face)",
    },
    "2302": {
        "-Z": "upright",
        "+Z": "upside down on studs",
        "X": "on side (curved profile)",
        "+Y": "on flat end",
        "-Y": "on curved end",
    },
}
AXES = {
    "+X": (1, 0, 0),
    "-X": (-1, 0, 0),
    "+Y": (0, 1, 0),
    "-Y": (0, -1, 0),
    "+Z": (0, 0, 1),
    "-Z": (0, 0, -1),
}


# --------------------------------------------------------------------------- loading


@dataclass(frozen=True)
class Grasp:
    """Top-down parallel-jaw grasp in the stored pose.

    ``yaw`` is the angle of the jaw closing axis from world +X (degrees, 0..180) when the
    brick has the stored quaternion. ``axis_body`` is the same axis in the brick frame, so it
    stays correct for any yaw of the brick on the table.
    """

    feasible: bool
    width: float  # m, brick width between the jaws
    yaw: float  # degrees
    height: float  # m, grasp point above the table
    axis_body: tuple[float, float, float]
    contact_length: float  # m, shortest of the two flat contact lines
    faces: tuple[str, str]  # body faces that touch the jaws
    note: str


@dataclass(frozen=True)
class StablePose:
    id: str
    description: str
    quat: tuple[float, float, float, float]  # wxyz; body frame = mesh frame
    rest_height: float  # m, mesh origin above the table
    tilt: float  # degrees between body +Z and world +Z
    probability: float  # quasi-static estimate
    drop_frequency: float  # fraction of the random MuJoCo drops
    tip_angle: float  # degrees of tilt that tips the brick off this face (analytical)
    verified_table: bool
    verified_plate: bool
    marginal: bool
    grasp: Grasp | None


def load_stable_poses(path: Path | str = DEFAULT_JSON) -> dict[str, list[StablePose]]:
    """Read the JSON from the CLI. Keys are the ``world.BRICKS`` ids (``part-001`` ...)."""
    data = json.loads(Path(path).read_text())
    result: dict[str, list[StablePose]] = {}
    for brick_id, brick in data["bricks"].items():
        poses = []
        for p in brick["poses"]:
            g = p.get("grasp")
            grasp = None
            if g is not None:
                grasp = Grasp(
                    g["feasible"],
                    g["width"],
                    g["yaw"],
                    g["height"],
                    tuple(g["axis_body"]),
                    g["contact_length"],
                    tuple(g["faces"]),
                    g["note"],
                )
            poses.append(
                StablePose(
                    p["id"],
                    p["description"],
                    tuple(p["quat"]),
                    p["rest_height"],
                    p["tilt"],
                    p["probability"],
                    p["drop_frequency"],
                    p["tip_angle"],
                    p["verified"]["table"]["ok"],
                    p["verified"]["plate"]["ok"],
                    p["marginal"],
                    grasp,
                )
            )
        result[brick_id] = poses
    return result


# --------------------------------------------------------------------------- physics


def world_physics() -> dict:
    """Read option, defaults, surfaces, brick geoms and masses from ``world.scene_xml``."""
    from brickcell.simulation import world

    with tempfile.TemporaryDirectory() as tmp:
        root = ET.fromstring(world.scene_xml(Path(tmp)))
    geoms = {g.get("name"): g for g in root.iter("geom")}
    floor = next(g for name, g in geoms.items() if name and name.endswith("-floor"))
    meshes = {m.get("name"): m for m in root.iter("mesh")}
    bricks = {}
    for brick_id, b in world.BRICKS.items():
        body = next(x for x in root.iter("body") if x.get("name") == brick_id)
        geom = next(g for g in body.iter("geom") if g.get("mesh") == b.mesh)
        inertial = body.find("inertial")
        if inertial is not None:
            mass = float(inertial.get("mass"))
        else:
            mass = sum(float(g.get("mass", 0)) for g in body.iter("geom")) or b.mass
        bricks[brick_id] = {
            "part": b.part,
            "mesh": b.mesh,
            "rgba": list(b.rgba),
            "mass": mass,
            "geom": dict(geom.attrib),
            "mesh_asset": dict(meshes[b.mesh].attrib),
        }
    return {
        "compiler": dict(root.find("compiler").attrib) if root.find("compiler") is not None else {},
        "option": ET.tostring(root.find("option"), encoding="unicode"),
        "default": ET.tostring(root.find("default"), encoding="unicode")
        if root.find("default") is not None
        else "",
        "table": dict(geoms["table"].attrib),
        "plate_floor": dict(floor.attrib),
        "bricks": bricks,
        "block": json.loads(json.dumps(getattr(world, "PHYSICS", {}))),
    }


def physics_summary(phys: dict) -> dict:
    """Short record of the values used, for the JSON."""
    option = ET.fromstring(phys["option"])
    default = ET.fromstring(phys["default"]) if phys["default"] else None
    default_geom = default.find("geom") if default is not None else None
    return {
        "world_physics_block": phys["block"],
        "option": dict(option.attrib),
        "default_geom": dict(default_geom.attrib) if default_geom is not None else {},
        "table": {k: v for k, v in phys["table"].items() if k not in {"name", "rgba", "size"}},
        "plate_floor": {
            k: v for k, v in phys["plate_floor"].items() if k not in {"name", "rgba", "size", "pos"}
        },
        "bricks": {
            k: {
                "mass": b["mass"],
                **{
                    a: v
                    for a, v in b["geom"].items()
                    if a not in {"name", "mesh", "pos", "euler", "rgba", "mass", "type"}
                },
            }
            for k, b in phys["bricks"].items()
        },
    }


def _attrs(attrs: dict, drop: set[str]) -> str:
    return " ".join(f'{k}="{v}"' for k, v in attrs.items() if k not in drop)


SURFACE_DROP = {"name", "type", "size", "pos", "quat", "euler", "rgba"}


def pose_model_xml(phys: dict, brick_id: str, shell: dict, floor: str = "table") -> str:
    """One free brick with the shell COM and inertia on a table plane or a plate box."""
    b = phys["bricks"][brick_id]
    compiler = {**phys["compiler"], "inertiafromgeom": "auto"}
    mesh = _attrs(b["mesh_asset"], set())
    geom = _attrs(b["geom"], {"name", "pos", "quat", "euler", "axisangle", "mass", "density"})
    if floor == "table":
        surface = f'<geom name="floor" type="plane" size="1 1 .01" {_attrs(phys["table"], SURFACE_DROP)}/>'
    else:
        surface = (
            f'<geom name="floor" type="box" size=".15 .15 .005" pos="0 0 -.005" '
            f"{_attrs(phys['plate_floor'], SURFACE_DROP)}/>"
        )
    c, inertia = shell["com"], shell["inertia"] * b["mass"]
    full = " ".join(
        f"{inertia[i, j]:.6e}" for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    )
    return f"""<mujoco model="stable pose test">
      <compiler {_attrs(compiler, set())}/>
      {phys["option"]}
      {phys["default"]}
      <asset><mesh {mesh}/></asset>
      <worldbody>
        {surface}
        <body name="brick" pos="0 0 .1">
          <freejoint name="free"/>
          <inertial pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" mass="{b["mass"]}" fullinertia="{full}"/>
          <geom name="brick" {geom}/>
        </body>
      </worldbody></mujoco>"""


# --------------------------------------------------------------------------- geometry


def read_stl(path: Path) -> np.ndarray:
    """Triangles (n, 3, 3) from a binary STL."""
    raw = path.read_bytes()
    count = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
    records = np.frombuffer(
        raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count, 84
    )
    return records["v"].astype(float)


def shell_properties(triangles: np.ndarray) -> dict:
    """Area, centroid and inertia per unit mass (about the centroid) of a uniform thin shell."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    com = (area[:, None] * (a + b + c) / 3).sum(0) / total
    s = a + b + c
    second = sum(np.einsum("n,ni,nj->ij", area, v, v) for v in (a, b, c))
    second = (second + np.einsum("n,ni,nj->ij", area, s, s)) / 12 / total
    second -= np.outer(com, com)
    inertia = np.trace(second) * np.eye(3) - second
    return {"area": total, "com": com, "inertia": inertia}


def hull_volume_centroid(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    origin = vertices.mean(0)
    a, b, c = (vertices[faces[:, i]] - origin for i in range(3))
    vol = np.einsum("ni,ni->n", a, np.cross(b, c)) / 6
    return origin + (vol[:, None] * (a + b + c) / 4).sum(0) / vol.sum()


def mujoco_hull(phys: dict, brick_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull that MuJoCo collides with, in the mesh frame: (vertices, outward faces)."""
    asset = _attrs(phys["bricks"][brick_id]["mesh_asset"], set())
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><mesh {asset}/></asset><worldbody><geom type="mesh" mesh="'
        f'{phys["bricks"][brick_id]["mesh"]}"/></worldbody></mujoco>'
    )
    g = model.mesh_graph[model.mesh_graphadr[0] :]
    nv, nf = int(g[0]), int(g[1])
    global_ids = g[2 + nv : 2 + 2 * nv]
    faces = g[2 + 3 * nv + 3 * nf : 2 + 3 * nv + 6 * nf].reshape(nf, 3)
    start = model.mesh_vertadr[0]
    verts = model.mesh_vert[start : start + model.mesh_vertnum[0]].astype(float)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.mesh_quat[0])
    verts = verts @ rot.reshape(3, 3).T + model.mesh_pos[0]
    index = {int(gid): i for i, gid in enumerate(global_ids)}
    vertices = verts[global_ids]
    faces = np.vectorize(index.get)(faces)
    centre = vertices.mean(0)
    for f in faces:
        n = np.cross(vertices[f[1]] - vertices[f[0]], vertices[f[2]] - vertices[f[0]])
        if np.dot(n, vertices[f[0]] - centre) < 0:
            f[[1, 2]] = f[[2, 1]]
    return vertices, faces


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise hull of 2D points (monotone chain)."""
    pts = np.unique(np.round(points, 9), axis=0)
    if len(pts) < 3:
        return pts

    def half(seq):
        out: list[np.ndarray] = []
        for p in seq:
            while len(out) >= 2 and cross2(out[-1] - out[-2], p - out[-2]) <= 1e-12:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


@dataclass
class Facet:
    normal: np.ndarray  # outward, mesh frame
    triangles: list[int]
    edges: list[tuple[int, int]]  # boundary edges, ordered with the facet on the left
    neighbours: dict[tuple[int, int], int]  # boundary edge -> neighbouring facet


def hull_facets(vertices: np.ndarray, faces: np.ndarray) -> list[Facet]:
    """Merge coplanar hull triangles into facets."""
    normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    edge_face = {}
    for i, f in enumerate(faces):
        for k in range(3):
            edge_face[(int(f[k]), int(f[(k + 1) % 3]))] = i
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (u, w), i in edge_face.items():
        j = edge_face[(w, u)]
        if np.dot(normals[i], normals[j]) > 1 - 1e-6:
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(faces)):
        groups.setdefault(find(i), []).append(i)
    facet_of = {}
    facets = []
    for k, tris in enumerate(groups.values()):
        area = [
            np.linalg.norm(np.cross(*(vertices[faces[t, 1:]] - vertices[faces[t, 0]])))
            for t in tris
        ]
        normal = np.average(normals[tris], axis=0, weights=area)
        facets.append(Facet(normal / np.linalg.norm(normal), tris, [], {}))
        for t in tris:
            facet_of[t] = k
    for (u, w), i in edge_face.items():
        j = edge_face[(w, u)]
        if facet_of[i] != facet_of[j]:
            facets[facet_of[i]].edges.append((u, w))
            facets[facet_of[i]].neighbours[(u, w)] = facet_of[j]
    return facets


def solid_angles(vertices: np.ndarray, faces: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Solid angle of each triangle seen from ``centre`` (Van Oosterom and Strackee)."""
    a, b, c = (vertices[faces[:, i]] - centre for i in range(3))
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    num = np.abs(np.einsum("ni,ni->n", a, np.cross(b, c)))
    den = (
        la * lb * lc
        + np.einsum("ni,ni->n", a, b) * lc
        + np.einsum("ni,ni->n", a, c) * lb
        + np.einsum("ni,ni->n", b, c) * la
    )
    return 2 * np.arctan2(num, den)


def analyse_facets(vertices, faces, facets, com) -> list[dict]:
    """Stability, tip angle and toppling target of each facet."""
    out = []
    for facet in facets:
        n = facet.normal
        v0 = vertices[facet.edges[0][0]]
        height = float(np.dot(v0 - com, n))  # COM to plane distance
        p = com + height * n  # COM projection on the facet plane
        signed = []
        for u, w in facet.edges:
            e = vertices[w] - vertices[u]
            m = np.cross(e, n)  # in-plane, points out of the facet (facet is on the left)
            m /= np.linalg.norm(m)
            s = float(np.dot(p - vertices[u], m))
            t = float(np.dot(p - vertices[u], e) / np.dot(e, e))
            signed.append((s, t))
        # Collinear boundary pieces share a line; the margin is the smallest inside distance.
        margin = -max(s for s, _ in signed)
        stable = margin > 1e-7
        target = None
        if not stable:
            outside = [(s, k) for k, (s, t) in enumerate(signed) if s > 0 and 0 <= t <= 1]
            if not outside:
                outside = [(s, k) for k, (s, _) in enumerate(signed) if s > 0]
            k = max(outside)[1]
            target = facet.neighbours[facet.edges[k]]
        tip = float(np.degrees(np.arctan2(margin, height))) if stable else 0.0
        out.append(
            {
                "stable": stable,
                "height": height,
                "margin": margin,
                "tip_angle": tip,
                "target": target,
            }
        )
    return out


def quasi_static_probabilities(vertices, faces, facets, info, com) -> dict[int, float]:
    angles = solid_angles(vertices, faces, com)
    prob: dict[int, float] = {}
    for k, facet in enumerate(facets):
        j, seen = k, set()
        while not info[j]["stable"]:
            if j in seen:  # numerical loop; should not happen on a convex hull
                break
            seen.add(j)
            j = info[j]["target"]
        prob[j] = prob.get(j, 0.0) + angles[facet.triangles].sum() / (4 * np.pi)
    return prob


# --------------------------------------------------------------------------- rotations


def mat_to_quat(rot: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.reshape(9))
    return q if q[0] >= 0 else -q


def quat_to_mat(q) -> np.ndarray:
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(q, float))
    return rot.reshape(3, 3)


def axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def rotation_to_down(normal: np.ndarray) -> np.ndarray:
    """Shortest rotation that turns ``normal`` (mesh frame) to world -Z."""
    down = np.array([0.0, 0.0, -1.0])
    axis = np.cross(normal, down)
    if np.linalg.norm(axis) < 1e-9:
        return np.eye(3) if normal[2] < 0 else axis_angle([1, 0, 0], np.pi)
    return axis_angle(axis, np.arccos(np.clip(np.dot(normal, down), -1, 1)))


def canonical_rotation(normal: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Rest rotation with the footprint box on the world axes, long side on X, near identity."""
    r0 = rotation_to_down(normal)
    xy = convex_hull_2d((vertices @ r0.T)[:, :2])
    best = None
    for i in range(len(xy)):
        e = xy[(i + 1) % len(xy)] - xy[i]
        psi = -np.arctan2(e[1], e[0])
        p = xy @ axis_angle([0, 0, 1], psi)[:2, :2].T
        size = p.max(0) - p.min(0)
        if best is None or size.prod() < best[0] - 1e-12:
            best = (size.prod(), psi, size)
    _, psi, size = best
    if size[1] > size[0] + 5e-4:
        psi += np.pi / 2
    turns = (0, 1, 2, 3) if abs(size[0] - size[1]) <= 5e-4 else (0, 2)
    options = [axis_angle([0, 0, 1], psi + t * np.pi / 2) @ r0 for t in turns]
    return max(options, key=lambda r: round(mat_to_quat(r)[0], 6))


def angle_between(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def axis_label(direction: np.ndarray, tolerance: float = 3.0) -> str | None:
    for name, axis in AXES.items():
        if angle_between(direction, axis) < tolerance:
            return name
    return None


SUFFIX = {"+X": "px", "-X": "nx", "+Y": "py", "-Y": "ny", "+Z": "pz", "-Z": "nz"}


def describe(part: str, down: np.ndarray, rot: np.ndarray) -> tuple[str, str]:
    """(id, words) for a pose from the outward normal of the face that is down (mesh frame)."""
    names = FACE_NAMES[part]
    label = axis_label(down)
    if label in ("-Z", "+Z"):
        return ("upright" if label == "-Z" else "upside_down"), names[label]
    if label is not None:
        words = names.get(label, names.get(label[1]))
        slug = words.split(" (")[0].removeprefix("on ").replace(" ", "_")
        return f"{slug}_{SUFFIX[label]}", f"{words}, {label} face down"
    from_upside_down = 180 - angle_between(rot.T @ np.array([0, 0, 1.0]), [0, 0, 1])
    side = axis_label(np.array([down[0], down[1], 0]) / np.linalg.norm(down[:2]), 10) or "?"
    if part == "2302" and side == "-Y":
        return "curved_face", (
            f"on curved face and stud edges, tilted {from_upside_down:.0f} deg from upside down"
        )
    return f"stud_edge_{SUFFIX.get(side, 'x')}", (
        f"on stud edges and {side} top edge, tilted {from_upside_down:.0f} deg from upside down"
    )


# --------------------------------------------------------------------------- simulation


class Sim:
    def __init__(self, xml: str):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.body = self.model.body("brick").id

    def place(self, rot: np.ndarray, z: float, xy=(0.0, 0.0)) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = [xy[0], xy[1], z, *mat_to_quat(rot)]
        mujoco.mj_forward(self.model, self.data)

    def rot(self) -> np.ndarray:
        return self.data.xmat[self.body].reshape(3, 3).copy()

    def com(self) -> np.ndarray:
        return self.data.xipos[self.body].copy()

    def speeds(self) -> tuple[float, float]:
        return float(np.linalg.norm(self.data.qvel[:3])), float(np.linalg.norm(self.data.qvel[3:6]))

    def run(self, seconds: float) -> None:
        mujoco.mj_step(
            self.model, self.data, nstep=max(1, round(seconds / self.model.opt.timestep))
        )

    def settle(self, max_seconds: float = 6.0) -> bool:
        """Step until the brick is at rest for 0.25 s. Return False on time out."""
        still, t = 0.0, 0.0
        while t < max_seconds:
            self.run(0.05)
            t += 0.05
            v, w = self.speeds()
            still = still + 0.05 if t > 0.3 and v < 0.002 and w < 0.05 else 0.0
            if still >= 0.25:
                return True
        return False


def rest_height(rot: np.ndarray, vertices: np.ndarray) -> float:
    return float(-(vertices @ rot.T)[:, 2].min())


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return quat_to_mat(q / np.linalg.norm(q))


def drop_brick(args) -> dict:
    """Random drops on the table for one brick; returns the final down directions."""
    phys, brick_id, shell, vertices, seed, count = args
    sim = Sim(pose_model_xml(phys, brick_id, shell, "table"))
    rng = np.random.default_rng(seed)
    downs, unsettled = [], 0
    for _ in range(count):
        rot = random_rotation(rng)
        sim.place(rot, rest_height(rot, vertices) + rng.uniform(*DROP_HEIGHT))
        if not sim.settle():
            unsettled += 1
            continue
        downs.append(-sim.rot()[2])  # world -Z in the body frame
    return {"downs": downs, "unsettled": unsettled}


def hold_test(sim: Sim, rot: np.ndarray, vertices: np.ndarray, seconds: float) -> dict:
    """Put the brick in the pose, simulate, measure drift, rotation and final speed.

    The final speed is the net motion over the last 0.5 s, because the contact solver keeps a
    small vibration (about 0.5 mm/s, < 0.01 mm amplitude) on some poses.
    """
    sim.place(rot, rest_height(rot, vertices))
    c0, r0 = sim.com(), sim.rot()
    sim.run(seconds - 0.5)
    c1, r1 = sim.com(), sim.rot()
    sim.run(0.5)
    v_now, w_now = sim.speeds()
    v = float(np.linalg.norm(sim.com() - c1)) / 0.5
    w = np.radians(rotation_angle(sim.rot() @ r1.T)) / 0.5
    rotation = rotation_angle(sim.rot() @ r0.T)
    drift = float(np.linalg.norm(sim.com() - c0))
    ok = drift < MAX_DRIFT and rotation < MAX_ROTATION_DEG and v < MAX_SPEED and w < 0.01
    return {
        "ok": bool(ok),
        "drift_mm": round(drift * 1000, 4),
        "rotation_deg": round(rotation, 4),
        "speed_mm_s": round(v * 1000, 5),
        "angular_speed_rad_s": round(float(w), 5),
        "vibration_mm_s": round(v_now * 1000, 4),
    }


def rotation_angle(rel: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1))))


def perturb_test(sim: Sim, rot: np.ndarray, vertices: np.ndarray, down: np.ndarray) -> dict:
    """Tilt the pose by PERTURB_DEG in many directions and see if it comes back."""
    back, worst = 0, 0.0
    for k in range(PERTURB_DIRECTIONS):
        phi = 2 * np.pi * k / PERTURB_DIRECTIONS
        tilted = axis_angle([np.cos(phi), np.sin(phi), 0], np.radians(PERTURB_DEG)) @ rot
        sim.place(tilted, rest_height(tilted, vertices))
        sim.settle(max_seconds=4.0)
        error = angle_between(-sim.rot()[2], down)
        worst = max(worst, error)
        back += error < 1.0
    return {
        "tilt_deg": PERTURB_DEG,
        "trials": PERTURB_DIRECTIONS,
        "returned": int(back),
        "worst_error_deg": round(worst, 3),
    }


# --------------------------------------------------------------------------- grasp


def slice_hull(vertices: np.ndarray, faces: np.ndarray, z: float) -> np.ndarray:
    """2D convex section of a convex hull at height z (world frame)."""
    pts = []
    for f in faces:
        p = vertices[f]
        for i in range(3):
            a, b = p[i], p[(i + 1) % 3]
            if (a[2] - z) * (b[2] - z) <= 0 and a[2] != b[2]:
                t = (z - a[2]) / (b[2] - a[2])
                pts.append((a + t * (b - a))[:2])
            elif a[2] == z:
                pts.append(a[:2])
    return convex_hull_2d(np.array(pts)) if len(pts) >= 3 else np.zeros((0, 2))


def contact_length(poly: np.ndarray, a: np.ndarray, extreme: float) -> float:
    """Length of the longest polygon edge at the extreme along a, facing along a (within 3 deg)."""
    best = 0.0
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        e = q - p
        length = np.linalg.norm(e)
        if length < 1e-6:
            continue
        normal = np.array([e[1], -e[0]]) / length  # outward for a CCW polygon
        if np.dot(normal, a) > np.cos(np.radians(3)) and extreme - p @ a < 3e-4:
            best = max(best, float(length))
    return best


def face_label(direction_body: np.ndarray) -> str:
    return axis_label(direction_body, tolerance=10.0) or "oblique"


def grasps_for_pose(rot: np.ndarray, z0: float, vertices: np.ndarray, faces: np.ndarray) -> list:
    """Feasible top-down grasps, best first. One entry per distinct jaw axis."""
    world = vertices @ rot.T + [0, 0, z0]
    top = float(world[:, 2].max())
    h0 = max(GRIP_MIN_HEIGHT, top / 2)
    heights = sorted(
        (h for h in np.arange(GRIP_MIN_HEIGHT, top - JAW_HALF_HEIGHT - 5e-4, 0.0005)),
        key=lambda h: abs(h - h0),
    )
    for h in heights:
        found = []
        mid = slice_hull(world, faces, h)
        if len(mid) < 3:
            continue
        low = slice_hull(world, faces, h - JAW_HALF_HEIGHT)
        high = slice_hull(world, faces, h + JAW_HALF_HEIGHT)
        above = np.vstack([mid, world[world[:, 2] >= h][:, :2]])
        axes = []
        for i in range(len(mid)):
            e = mid[(i + 1) % len(mid)] - mid[i]
            if np.linalg.norm(e) > 1e-6:
                yaw = np.degrees(np.arctan2(-e[0], e[1])) % 180  # edge normal, mod 180
                if all(min(abs(yaw - y), 180 - abs(yaw - y)) > 0.5 for y in axes):
                    axes.append(yaw)
        for yaw in axes:
            a = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
            ext = (
                [(s @ a).max() for s in (low, mid, high)],
                [(s @ a).min() for s in (low, mid, high)],
            )
            hi, lo = ext[0][1], ext[1][1]
            width = hi - lo
            flat = np.ptp(ext[0]) < 7.5e-4 and np.ptp(ext[1]) < 7.5e-4
            overhang = (above @ a).max() > hi + 5e-4 or (above @ a).min() < lo - 5e-4
            lengths = contact_length(mid, a, hi), contact_length(mid, -a, -lo)
            axis_body = rot.T @ np.array([a[0], a[1], 0.0])
            labels = (face_label(axis_body), face_label(-axis_body))
            feasible = (
                GRIP_WIDTH[0] <= width <= GRIP_WIDTH[1]
                and flat
                and not overhang
                and min(lengths) >= 0.003
            )
            if not feasible:
                continue
            notes = []
            if "+Z" in labels:
                notes.append("one jaw on the stud tops")
            if "-Z" in labels:
                notes.append("one jaw on the open underside (wall rim only)")
            score = min(lengths) * (0.5 if notes else 1.0)
            found.append(
                (
                    score,
                    {
                        "feasible": True,
                        "width": round(float(width), 5),
                        "yaw": round(float(yaw), 2),
                        "height": round(float(h), 5),
                        "axis_body": [round(float(x), 4) for x in axis_body],
                        "contact_length": round(float(min(lengths)), 5),
                        "faces": list(labels),
                        "note": "; ".join(notes) or "flat parallel side walls",
                    },
                )
            )
        if found:
            found.sort(key=lambda s: -s[0])
            return [g for _, g in found]
    return []


def footprint(rot: np.ndarray, z0: float, vertices: np.ndarray) -> dict:
    world = vertices @ rot.T + [0, 0, z0]
    size = world.max(0) - world.min(0)
    return {
        "x": round(float(size[0]), 5),
        "y": round(float(size[1]), 5),
        "height": round(float(size[2]), 5),
    }


# --------------------------------------------------------------------------- analysis


def analyse_brick(phys: dict, brick_id: str, drops: int = DROPS, seed: int = 0) -> dict:
    b = phys["bricks"][brick_id]
    shell = shell_properties(read_stl(Path(b["mesh_asset"]["file"])))
    vertices, faces = mujoco_hull(phys, brick_id)
    com = shell["com"]
    facets = hull_facets(vertices, faces)
    info = analyse_facets(vertices, faces, facets, com)
    prob = quasi_static_probabilities(vertices, faces, facets, info, com)

    # Empirical drops, in parallel chunks.
    chunks = 8
    jobs = [
        (phys, brick_id, shell, vertices, seed * 1000 + k, drops // chunks + (k < drops % chunks))
        for k in range(chunks)
    ]
    with ProcessPoolExecutor(chunks) as pool:
        results = list(pool.map(drop_brick, jobs))
    downs = [d for r in results for d in r["downs"]]
    unsettled = sum(r["unsettled"] for r in results)

    # Candidates: stable facets, then drop clusters that match no stable facet.
    candidates = [
        {"normal": facets[k].normal, "facet": k, "count": 0}
        for k in range(len(facets))
        if info[k]["stable"]
    ]
    unmatched = []
    for d in downs:
        match = min(candidates, key=lambda c: angle_between(c["normal"], d))
        if angle_between(match["normal"], d) < CLUSTER_DEG:
            match["count"] += 1
        else:
            unmatched.append(d)
    for d in unmatched:
        for c in candidates:
            if c["facet"] is None and angle_between(c["normal"], d) < CLUSTER_DEG:
                c["count"] += 1
                c["members"].append(d)
                mean = np.mean(c["members"], axis=0)
                c["normal"] = mean / np.linalg.norm(mean)
                break
        else:
            candidates.append({"normal": d, "facet": None, "count": 1, "members": [d]})

    table = Sim(pose_model_xml(phys, brick_id, shell, "table"))
    plate = Sim(pose_model_xml(phys, brick_id, shell, "plate"))
    poses = []
    for c in candidates:
        rot = canonical_rotation(c["normal"], vertices)
        z0 = rest_height(rot, vertices)
        k = c["facet"]
        verified = {
            "table": hold_test(table, rot, vertices, HOLD_SECONDS),
            "plate": hold_test(plate, rot, vertices, HOLD_SECONDS),
        }
        perturb = {
            "table": perturb_test(table, rot, vertices, c["normal"]),
            "plate": perturb_test(plate, rot, vertices, c["normal"]),
        }
        marginal = any(p["returned"] < p["trials"] for p in perturb.values())
        grasps = grasps_for_pose(rot, z0, vertices, faces)
        pose_id, words = describe(b["part"], c["normal"], rot)
        tilt = angle_between(rot.T @ [0, 0, 1], [0, 0, 1])
        poses.append(
            {
                "id": pose_id,
                "description": words,
                "quat": [round(float(x), 6) for x in mat_to_quat(rot)],
                "rest_height": round(z0, 6),
                "tilt": round(tilt, 2),
                "down_face_normal_body": [round(float(x), 4) for x in c["normal"]],
                "probability": round(prob.get(k, 0.0), 4) if k is not None else 0.0,
                "drop_frequency": round(c["count"] / max(1, len(downs)), 4),
                "drop_count": c["count"],
                "analytic_stable": k is not None,
                "tip_angle": round(info[k]["tip_angle"], 2) if k is not None else 0.0,
                "com_height": round(float(com @ rot.T[:, 2] + z0), 5),
                "footprint": footprint(rot, z0, vertices),
                "verified": verified,
                "perturbation": perturb,
                "marginal": bool(marginal),
                "grasp": grasps[0]
                if grasps
                else {
                    "feasible": False,
                    "width": 0.0,
                    "yaw": 0.0,
                    "height": 0.0,
                    "axis_body": [0.0, 0.0, 0.0],
                    "contact_length": 0.0,
                    "faces": ["", ""],
                    "note": "no flat parallel jaw contact 10-45 mm wide",
                },
                "alternative_grasps": grasps[1:],
            }
        )
    poses.sort(
        key=lambda p: (-p["verified"]["table"]["ok"], -p["probability"] - p["drop_frequency"])
    )
    used: dict[str, int] = {}
    for p in poses:
        used[p["id"]] = used.get(p["id"], 0) + 1
        if used[p["id"]] > 1:
            p["id"] += f"_{used[p['id']]}"
    return {
        "part": b["part"],
        "mesh": b["mesh"],
        "mass": b["mass"],
        "com_shell": [round(float(x), 6) for x in com],
        "com_hull_volume": [round(float(x), 6) for x in hull_volume_centroid(vertices, faces)],
        "hull": {
            "vertices": len(vertices),
            "triangles": len(faces),
            "facets": len(facets),
            "stable_facets": int(sum(i["stable"] for i in info)),
        },
        "drops": drops,
        "drops_unsettled": unsettled,
        "drops_off_facet": len(unmatched),
        "poses": poses,
    }


def analyse(drops: int = DROPS, seed: int = 0) -> dict:
    phys = world_physics()
    return {
        "version": 1,
        "units": (
            "metres, degrees; quaternions wxyz; body frame = mesh frame (origin bottom centre); "
            "each pose yaw puts the long side of its footprint on world X"
        ),
        "method": {
            "com": "area-weighted centroid of the mesh triangles (uniform thin shell)",
            "collision": "MuJoCo convex hull of each mesh",
            "probability": "quasi-static: solid angle of hull facets seen from the COM",
            "drops": f"{drops} random orientations, {DROP_HEIGHT[0] * 1000:.0f}-"
            f"{DROP_HEIGHT[1] * 1000:.0f} mm above the table, simulated to rest",
            "verify": f"{HOLD_SECONDS:.0f} s hold: drift < {MAX_DRIFT * 1000} mm, rotation < "
            f"{MAX_ROTATION_DEG} deg, speed < {MAX_SPEED * 1000} mm/s; plate = flat box with the "
            "KALAS floor geom values",
            "marginal": f"tipped after a {PERTURB_DEG} deg tilt in one of {PERTURB_DIRECTIONS} "
            "directions on the table or the plate",
            "grasp": f"top-down parallel jaws closing horizontally; width {GRIP_WIDTH[0] * 1000:.0f}"
            f"-{GRIP_WIDTH[1] * 1000:.0f} mm; flat parallel contact over +/-"
            f"{JAW_HALF_HEIGHT * 1000:.0f} mm; no overhang above the grasp; height >= "
            f"{GRIP_MIN_HEIGHT * 1000:.0f} mm, near half the part height; yaw = closing axis "
            "from world +X in the stored pose",
        },
        "physics": physics_summary(phys),
        "bricks": {bid: analyse_brick(phys, bid, drops, seed) for bid in phys["bricks"]},
    }


# --------------------------------------------------------------------------- render


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_pose(b: dict, p: dict, size: int):
    """Offscreen image of one pose; red pads show the jaws of the best grasp."""
    from PIL import Image

    g = p["grasp"]
    jaws = ""
    if g["feasible"]:
        yaw = np.radians(g["yaw"])
        a = np.array([np.cos(yaw), np.sin(yaw), 0])
        for side in (1, -1):
            c = side * a * (g["width"] / 2 + 0.002) + [0, 0, g["height"]]
            jaws += (
                f'<geom type="box" size=".0015 .009 {JAW_HALF_HEIGHT}" pos="{c[0]} {c[1]} {c[2]}" '
                f'euler="0 0 {yaw}" rgba=".95 .1 .1 .55"/>'
            )
    rgba = " ".join(str(x) for x in b["rgba"])
    xml = f"""<mujoco><visual><global offwidth="{size}" offheight="{size}"/>
      <headlight ambient=".45 .45 .45" diffuse=".55 .55 .55" specular="0 0 0"/></visual>
      <asset><mesh {_attrs(b["mesh_asset"], set())}/>
      <texture name="grid" type="2d" builtin="checker" rgb1=".93 .93 .93" rgb2=".84 .84 .84"
        width="64" height="64"/>
      <material name="grid" texture="grid" texrepeat="100 100"/></asset>
      <worldbody><light pos=".1 -.1 .4" dir="-.2 .2 -1"/>
      <geom type="plane" size="1 1 .01" material="grid"/>
      <body pos="0 0 {p["rest_height"]}" quat="{" ".join(map(str, p["quat"]))}">
      <geom type="mesh" mesh="{b["mesh"]}" rgba="{rgba} 1"/></body>{jaws}
      </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, p["footprint"]["height"] / 2]
    cam.distance, cam.azimuth, cam.elevation = 0.15, 225, -35
    with mujoco.Renderer(model, size, size) as renderer:
        renderer.update_scene(data, camera=cam)
        return Image.fromarray(renderer.render())


def render_grid(result: dict, path: Path, size: int = 300, cols: int = 5) -> None:
    """Grid of every pose that passed the table test, with a text label above each image."""
    import textwrap

    from PIL import Image, ImageDraw

    phys = world_physics()
    font, band = _font(13), 92
    cells = []
    for bid, brick in result["bricks"].items():
        b = phys["bricks"][bid]
        for p in brick["poses"]:
            if not p["verified"]["table"]["ok"]:
                continue
            cell = Image.new("RGB", (size, size + band), "white")
            cell.paste(render_pose(b, p, size), (0, band))
            g = p["grasp"]
            flags = "table ok, " + ("plate ok" if p["verified"]["plate"]["ok"] else "PLATE FAIL")
            flags += ", MARGINAL" if p["marginal"] else ""
            grasp = (
                f"grasp {g['width'] * 1000:.1f} mm, yaw {g['yaw']:.0f} deg, h {g['height'] * 1000:.1f} mm"
                if g["feasible"]
                else "no top-down grasp"
            )
            lines = textwrap.wrap(f"{b['part']} {p['id']}: {p['description']}", 46)[:2]
            lines += [
                f"p={p['probability']:.3f}  drops={p['drop_frequency']:.3f}  tip={p['tip_angle']:.1f} deg",
                flags,
                grasp,
            ]
            draw = ImageDraw.Draw(cell)
            for i, line in enumerate(lines):
                draw.text((6, 3 + 17 * i), line, fill=(0, 0, 0), font=font)
            draw.rectangle([0, 0, size - 1, size + band - 1], outline=(160, 160, 160))
            cells.append(cell)
    rows = (len(cells) + cols - 1) // cols
    grid = Image.new("RGB", (cols * size, rows * (size + band)), "white")
    for i, cell in enumerate(cells):
        grid.paste(cell, ((i % cols) * size, (i // cols) * (size + band)))
    grid.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--image", type=Path, default=None, help="optional PNG render grid")
    parser.add_argument("--drops", type=int, default=DROPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = analyse(args.drops, args.seed)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for bid, brick in result["bricks"].items():
        print(f"{bid} {brick['part']}: {len(brick['poses'])} poses")
        for p in brick["poses"]:
            print(
                f"  {p['id']:28s} p={p['probability']:.3f} drop={p['drop_frequency']:.3f} "
                f"table={p['verified']['table']['ok']} plate={p['verified']['plate']['ok']} "
                f"marginal={p['marginal']} grasp={p['grasp']['feasible']}"
            )
    if args.image:
        render_grid(result, args.image)


if __name__ == "__main__":
    main()
