"""Build matching positive-inertia URDF/MJCF from the calibrated CAD tree.

Mass, COM, inertia and contact values are explicit UNMEASURED priors.
This module never opens a robot, camera or network connection.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from so101_calibration import MOTOR_NAMES  # noqa: E402
from so101_calibration import ServoCalibration as ServoCalibration  # noqa: E402
from so101_kinematics import ARM_JOINTS, CalibratedSO101Kinematics  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def display_geometry():
    text = (ROOT / "visualizations/so101-kinematics.html").read_text()
    match = re.search(r"const data=(\[.*?\]);", text)
    if not match:
        raise ValueError("Missing CAD display geometry")
    return json.loads(match.group(1))


def new_priors():
    links = {}
    for row in display_geometry():
        vertices = np.concatenate([np.array(m["v"]).reshape(-1, 3) * 1e-5 for m in row["meshes"]])
        size = np.maximum(np.ptp(vertices, axis=0), 0.003)
        # Deliberately simple initial guesses; never presented as identified mass.
        mass = 0.08 if row["name"].startswith("Body") else 0.02
        inertia = (
            mass
            / 12
            * np.array(
                [
                    size[1] ** 2 + size[2] ** 2,
                    size[0] ** 2 + size[2] ** 2,
                    size[0] ** 2 + size[1] ** 2,
                ]
            )
        )
        links[row["name"]] = {
            "mass_kg": mass,
            "com_m": ((vertices.min(0) + vertices.max(0)) / 2).tolist(),
            "diagonal_inertia_kg_m2": inertia.tolist(),
            "box_size_m": size.tolist(),
            "measured": False,
            "source": "UNMEASURED heuristic mass and uniform display-bounding-box COM/inertia",
        }
    return {
        "schema": 1,
        "status": "unmeasured inertial/contact priors",
        "links": links,
        "physics": {
            "gravity": [0, 0, -9.80665],
            "timestep": 0.001,
            "floor_z_m": -0.10,
            "sliding_friction": 0.7,
            "contact_measured": False,
            "self_collision_validated": False,
        },
    }


def validate_prior(row):
    m = float(row["mass_kg"])
    matrix = np.asarray(
        row.get("inertia_matrix_kg_m2", np.diag(row["diagonal_inertia_kg_m2"])), float
    )
    com = np.asarray(row["com_m"], float)
    if not np.isfinite(m) or m < 0 or matrix.shape != (3, 3) or com.shape != (3,):
        raise ValueError("Invalid mass/COM/inertia")
    if m == 0 and np.all(matrix == 0) and row.get("absorbed_into"):
        return
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(com).all()
        or not np.allclose(matrix, matrix.T, atol=1e-12)
    ):
        raise ValueError("Inertia must be finite and symmetric")
    inertia = np.linalg.eigvalsh(matrix)
    if np.any(inertia <= 0):
        raise ValueError("Inertias must be positive and finite")
    if 2 * inertia.max() > inertia.sum() * (1 + 1e-9):
        raise ValueError("Inertia triangle inequality violated")


def numbers(values):
    return " ".join(f"{float(v):.17g}" for v in values)


def build(destination, priors_path=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    priors = json.loads(Path(priors_path).read_text()) if priors_path else new_priors()
    model = CalibratedSO101Kinematics("follower")
    urdf = destination / "arm.urdf"
    model.export_arm_urdf(urdf)
    tree = ET.parse(urdf)
    for link in tree.getroot().findall("link"):
        prior = priors["links"][link.get("name")]
        validate_prior(prior)
        inertial = link.find("inertial")
        inertial.find("origin").set("xyz", numbers(prior["com_m"]))
        inertial.find("mass").set("value", str(prior["mass_kg"]))
        i = inertial.find("inertia")
        matrix = np.asarray(
            prior.get("inertia_matrix_kg_m2", np.diag(prior["diagonal_inertia_kg_m2"]))
        )
        for key, a, b in (
            ("ixx", 0, 0),
            ("iyy", 1, 1),
            ("izz", 2, 2),
            ("ixy", 0, 1),
            ("ixz", 0, 2),
            ("iyz", 1, 2),
        ):
            i.set(key, str(matrix[a, b]))
    tree.write(urdf, encoding="utf-8", xml_declaration=True)
    root = ET.Element("mujoco", model="so101_follower_inertial_priors")
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        autolimits="true",
        fusestatic="false",
        inertiafromgeom="false",
    )
    physics = priors["physics"]
    ET.SubElement(
        root,
        "option",
        timestep=str(physics["timestep"]),
        gravity=numbers(physics["gravity"]),
        integrator="implicitfast",
        solver="Newton",
        iterations="100",
        tolerance="1e-10",
        cone="elliptic",
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "joint",
        damping=str(physics.get("joint_damping", 0.01)),
        armature=str(physics.get("joint_armature", 0.00002)),
        frictionloss=str(physics.get("joint_frictionloss", 0.005)),
    )
    # Collision boxes interact with the floor, not one another. Self collision
    # cannot be validated from this CAD tree's overlapping joint housings.
    ET.SubElement(
        default,
        "geom",
        contype="1",
        conaffinity="2",
        condim="3",
        friction=f"{physics['sliding_friction']} .005 .0001",
        solref=".01 1",
        solimp=".95 .99 .001",
        rgba=".4 .5 .6 1",
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="1 1 .02",
        pos=f"0 0 {physics['floor_z_m']}",
        contype="2",
        conaffinity="1",
    )
    bodies = {model.base: ET.SubElement(world, "body", name=model.base)}
    for name, parent, child, home, axis in model.joints:
        quat = Rotation.from_matrix(home[:3, :3]).as_quat()[[3, 0, 1, 2]]
        body = ET.SubElement(
            bodies[parent], "body", name=child, pos=numbers(home[:3, 3]), quat=numbers(quat)
        )
        bodies[child] = body
        if name in ARM_JOINTS:
            index = ARM_JOINTS.index(name)
            ET.SubElement(
                body,
                "joint",
                name=name,
                type="hinge",
                axis=numbers(axis),
                range=numbers([model.calibration.lower[index], model.calibration.upper[index]]),
            )
    for name, body in bodies.items():
        row = priors["links"][name]
        if row["mass_kg"] > 0:
            matrix = np.asarray(
                row.get("inertia_matrix_kg_m2", np.diag(row["diagonal_inertia_kg_m2"]))
            )
            ET.SubElement(
                body,
                "inertial",
                mass=str(row["mass_kg"]),
                pos=numbers(row["com_m"]),
                fullinertia=numbers(
                    [
                        matrix[0, 0],
                        matrix[1, 1],
                        matrix[2, 2],
                        matrix[0, 1],
                        matrix[0, 2],
                        matrix[1, 2],
                    ]
                ),
            )
        ET.SubElement(
            body,
            "geom",
            name=f"collision_{name}",
            type="box",
            pos=numbers(row.get("collision_center_m", row["com_m"])),
            size=numbers(np.array(row["box_size_m"]) / 2),
        )
    actuators = ET.SubElement(root, "actuator")
    sensors = ET.SubElement(root, "sensor")
    for name in ARM_JOINTS:
        ET.SubElement(
            actuators,
            "motor",
            name=f"motor_{name}",
            joint=name,
            gear="1",
            ctrllimited="true",
            ctrlrange="-1.5 1.5",
        )
        ET.SubElement(sensors, "jointpos", name=f"encoder_{name}", joint=name)
    xml = destination / "arm.xml"
    ET.indent(root)
    ET.ElementTree(root).write(xml, encoding="utf-8", xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_path(str(xml))
    if compiled.nq != 5:
        raise ValueError("Unexpected model joint count")
    dump(destination / "inertial_priors.json", priors)
    dump(
        destination / "provenance.json",
        {
            "cad_sha256": digest(ROOT / "SO-101.zip"),
            "alignment_sha256": digest(ROOT / "calibration/kinematics_alignment.json"),
            "calibration_sha256": model.calibration.fingerprint,
            "urdf_sha256": digest(urdf),
            "mjcf_sha256": digest(xml),
            "inertial_priors_sha256": digest(destination / "inertial_priors.json"),
            "joint_order": list(MOTOR_NAMES),
            "coordinates": "midpoint-zero encoder radians",
            "mass_identified": False,
            "contacts_calibrated": False,
        },
    )
    return compiled


def indices(model):
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
    return np.array([model.jnt_qposadr[j] for j in joints]), np.array(
        [model.jnt_dofadr[j] for j in joints]
    )


def load(path):
    model = mujoco.MjModel.from_xml_path(str(Path(path) / "arm.xml"))
    if mujoco.__version__ != "4.0.0":
        raise RuntimeError("This pipeline is pinned and checked with MuJoCo 4.0.0")
    return model
