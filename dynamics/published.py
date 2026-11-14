"""Import pinned Robot Studio CAD inertials by registering rigid link geometry.

Downloaded source models remain under artifacts. Source masses are CAD-derived
nominals, not measurements of the user's printed arm. All frame transforms,
registration errors, hashes and the unchanged calibration are recorded.
"""

from __future__ import annotations

import itertools
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from model import (
    ARM_JOINTS,
    CalibratedSO101Kinematics,
    digest,
    display_geometry,
    dump,
    new_priors,
)

COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"
PREFIX = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{COMMIT}/Simulation/SO101/"
EXPECTED_URDF_SHA256 = "3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c"
LINKS = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
)


def origin(node):
    result = np.eye(4)
    if node is not None:
        result[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return result


def transform(triangles, frame):
    return triangles @ frame[:3, :3].T + frame[:3, 3]


def sample(triangles, count, seed=101):
    rng = np.random.default_rng(seed)
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    chosen = triangles[rng.choice(len(triangles), count, p=area / area.sum())]
    a, b = rng.random((2, count))
    a = np.sqrt(a)
    return (
        (1 - a[:, None]) * chosen[:, 0]
        + (a * (1 - b))[:, None] * chosen[:, 1]
        + (a * b)[:, None] * chosen[:, 2]
    )


def fit_rigid(source, target):
    sc, tc = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - sc).T @ (target - tc))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return r, tc - r @ sc


def register(source, target, incoming=True):
    """PCA multi-start, trimmed surface ICP, incoming-axis consistency check."""
    tree = cKDTree(target)
    if incoming:
        candidates = []
        # Both URDF/CAD link origins lie on the incoming joint's local Z axis.
        # Restrict the registration to that exact axis; axial origin gauges may
        # differ. This prevents surface sampling from tilting an inertia frame.
        for flip in (np.eye(3), np.diag([1.0, -1.0, -1.0])):
            points0 = source @ flip.T
            for phi0 in np.linspace(-np.pi, np.pi, 36, endpoint=False):
                phi = phi0
                tz = target[:, 2].mean() - points0[:, 2].mean()
                for _ in range(40):
                    r = Rotation.from_rotvec([0, 0, phi]).as_matrix()
                    points = points0 @ r.T + [0, 0, tz]
                    distance, index = tree.query(points)
                    keep = distance <= np.quantile(distance, 0.95)
                    a, b = points0[keep], target[index[keep]]
                    new_phi = np.arctan2(
                        np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]),
                        np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]),
                    )
                    new_tz = np.mean(b[:, 2] - a[:, 2])
                    if abs(new_phi - phi) < 1e-6 and abs(new_tz - tz) < 1e-7:
                        phi, tz = new_phi, new_tz
                        break
                    phi, tz = new_phi, new_tz
                r = Rotation.from_rotvec([0, 0, phi]).as_matrix() @ flip
                t = np.array([0.0, 0.0, tz])
                distances = np.sort(tree.query(source @ r.T + t)[0])
                rms = np.sqrt(np.mean(distances[: int(0.95 * len(distances))] ** 2))
                candidates.append((rms, r, t))
        rms, r, t = min(candidates, key=lambda row: row[0])
        return (
            r,
            t,
            {
                "trimmed_surface_rms_m": float(rms),
                "incoming_axis_error_deg": 0.0,
                "incoming_axis_line_distance_m": 0.0,
                "method": "exact incoming-axis constrained surface registration",
            },
        )
    sc, tc = source.mean(0), target.mean(0)
    _, sv = np.linalg.eigh(np.cov(source.T))
    _, tv = np.linalg.eigh(np.cov(target.T))
    candidates = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            r = tv[:, order] @ np.diag(signs) @ sv.T
            if np.linalg.det(r) < 0.99:
                continue
            t = tc - r @ sc
            for _ in range(35):
                points = source @ r.T + t
                distance, index = tree.query(points)
                keep = distance <= np.quantile(distance, 0.95)
                dr, dt = fit_rigid(points[keep], target[index[keep]])
                r, t = dr @ r, dr @ t + dt
                if np.linalg.norm(dt) < 1e-7 and np.linalg.norm(dr - np.eye(3)) < 1e-5:
                    break
            distances = tree.query(source @ r.T + t)[0]
            rms = np.sqrt(np.mean(np.sort(distances)[: int(0.95 * len(distances))] ** 2))
            axis_error = 1 - abs(r[2, 2]) if incoming else 0
            lateral = np.linalg.norm(t[:2]) if incoming else 0
            candidates.append(
                (rms + 0.05 * axis_error + 3 * lateral, rms, r, t, axis_error, lateral)
            )
    _, rms, r, t, axis_error, lateral = min(candidates, key=lambda c: c[0])
    return (
        r,
        t,
        {
            "trimmed_surface_rms_m": float(rms),
            "incoming_axis_error_deg": float(np.rad2deg(np.arccos(np.clip(abs(r[2, 2]), 0, 1))))
            if incoming
            else 0,
            "incoming_axis_line_distance_m": float(lateral),
        },
    )


def fetch(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "robotstudio_so101_new_calib.urdf"
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(PREFIX + "so101_new_calib.urdf", timeout=30).read())
    if digest(path) != EXPECTED_URDF_SHA256:
        raise ValueError("Published URDF does not match the pinned source hash")
    root = ET.parse(path).getroot()
    manifest = {path.name: {"url": PREFIX + "so101_new_calib.urdf", "sha256": digest(path)}}
    for mesh in root.findall(".//visual/geometry/mesh"):
        name = mesh.get("filename")
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(urllib.request.urlopen(PREFIX + name, timeout=30).read())
        manifest[name] = {"url": PREFIX + name, "sha256": digest(destination)}
    dump(directory / "mesh_sources.json", {"commit": COMMIT, "files": manifest})
    return root


def inertia(link):
    node = link.find("inertial")
    tensor = node.find("inertia")
    a = {key: float(value) for key, value in tensor.attrib.items()}
    matrix = np.array(
        [
            [a["ixx"], a["ixy"], a["ixz"]],
            [a["ixy"], a["iyy"], a["iyz"]],
            [a["ixz"], a["iyz"], a["izz"]],
        ]
    )
    frame = origin(node.find("origin"))
    return (
        float(node.find("mass").get("value")),
        frame[:3, 3],
        frame[:3, :3] @ matrix @ frame[:3, :3].T,
    )


def merge(first, second, frame):
    m1, c1, i1 = first
    m2, c2, i2 = second
    c2 = frame[:3, :3] @ c2 + frame[:3, 3]
    i2 = frame[:3, :3] @ i2 @ frame[:3, :3].T
    c = (m1 * c1 + m2 * c2) / (m1 + m2)

    def shift(m, d):
        return m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    return m1 + m2, c, i1 + i2 + shift(m1, c1 - c) + shift(m2, c2 - c)


def import_priors(directory, output):
    directory = Path(directory)
    source = fetch(directory)
    src_links = {link.get("name"): link for link in source.findall("link")}
    geometric = CalibratedSO101Kinematics("follower")
    frames = geometric.frames(np.zeros(5))
    owners = {geometric.base: geometric.base}
    moving = []
    for name, parent, child, _, _ in geometric.joints:
        owners[child] = child if name in ARM_JOINTS else owners[parent]
        if name in ARM_JOINTS:
            moving.append(child)
    local_owners = [geometric.base, *moving]
    local_triangles = {n: [] for n in local_owners}
    rows = display_geometry()
    for row in rows:
        owner = owners[row["name"]]
        relative = np.linalg.inv(frames[owner]) @ frames[row["name"]]
        for mesh in row["meshes"]:
            vertices = np.array(mesh["v"]).reshape(-1, 3) * 1e-5
            triangles = vertices[np.array(mesh["f"]).reshape(-1, 3)]
            local_triangles[owner].append(transform(triangles, relative))
    priors = new_priors()
    for name, row in priors["links"].items():
        row["collision_center_m"] = row["com_m"][:]
        row.update(
            mass_kg=0.0,
            com_m=[0, 0, 0],
            diagonal_inertia_kg_m2=[0, 0, 0],
            inertia_matrix_kg_m2=np.zeros((3, 3)).tolist(),
            absorbed_into=owners[name],
            source="Mass consolidated into rigid owner from published CAD",
        )
    registrations = {}
    for group_index, (local, src) in enumerate(zip(local_owners, LINKS)):
        triangles = []
        for visual in src_links[src].findall("visual"):
            mesh = trimesh.load_mesh(
                directory / visual.find("geometry/mesh").get("filename"), process=False
            )
            triangles.append(transform(np.array(mesh.triangles), origin(visual.find("origin"))))
        physical = inertia(src_links[src])
        if src == "gripper_link":
            joint = next(j for j in source.findall("joint") if j.get("name") == "gripper")
            frame = origin(joint.find("origin"))
            jaw = src_links["moving_jaw_so101_v1_link"]
            physical = merge(physical, inertia(jaw), frame)
            for visual in jaw.findall("visual"):
                mesh = trimesh.load_mesh(
                    directory / visual.find("geometry/mesh").get("filename"), process=False
                )
                triangles.append(
                    transform(np.array(mesh.triangles), frame @ origin(visual.find("origin")))
                )
        source_points = sample(np.concatenate(triangles), 1800)
        target_points = sample(np.concatenate(local_triangles[local]), 3500)
        r, t, evidence = register(source_points, target_points, incoming=src != "base_link")
        if group_index < 5:
            outgoing = next(
                j
                for j in source.findall("joint")
                if j.find("parent").get("link") == src
                and j.find("child").get("link") == LINKS[group_index + 1]
            )
            source_out = origin(outgoing.find("origin"))
            local_out = np.linalg.inv(frames[local]) @ frames[local_owners[group_index + 1]]
            local_axis = local_out[:3, 2]
            source_axis = r @ source_out[:3, 2]
            distance = np.linalg.norm(
                np.cross(r @ source_out[:3, 3] + t - local_out[:3, 3], local_axis)
            )
            angle = np.rad2deg(np.arccos(np.clip(abs(np.dot(source_axis, local_axis)), 0, 1)))
            evidence.update(
                outgoing_axis_line_distance_m=float(distance), outgoing_axis_error_deg=float(angle)
            )
            if group_index > 0 and (distance > 0.008 or angle > 3):
                raise ValueError(
                    f"Published/local rigid-link geometry mismatch: {local}: {evidence}"
                )
        mass, com, matrix = physical
        transformed_com = r @ com + t
        transformed_inertia = r @ matrix @ r.T
        priors["links"][local].update(
            mass_kg=mass,
            com_m=transformed_com.tolist(),
            diagonal_inertia_kg_m2=np.diag(transformed_inertia).tolist(),
            inertia_matrix_kg_m2=transformed_inertia.tolist(),
            measured=False,
            source=f"Robot Studio SO101 CAD {COMMIT}: {src}; rigid surface registration",
            source_link=src,
            source_mass_kg=mass,
            absorbed_into=local,
        )
        registrations[local] = {
            "source_link": src,
            "R_source_to_local": r.tolist(),
            "t_source_to_local_m": t.tolist(),
            **evidence,
        }
        print(local, src, mass, evidence, flush=True)
    priors.update(
        status="Published SO101 CAD inertials registered to calibrated local geometry; not weighed on this arm",
        source_commit=COMMIT,
        registrations=registrations,
        source_urdf_sha256=digest(directory / "robotstudio_so101_new_calib.urdf"),
        source_url=PREFIX + "so101_new_calib.urdf",
        gripper_pose_assumption="source jaw fixed at zero",
    )
    priors["physics"].update(
        joint_damping=0.60,
        joint_frictionloss=0.052,
        joint_armature=0.028,
        motor_torque_limit_nm=2.94,
        joint_dynamics_source="MuJoCo Menagerie sts3215 defaults, not individually measured",
        joint_dynamics_source_url="https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101/so101.xml",
    )
    dump(output, priors)
    return priors
