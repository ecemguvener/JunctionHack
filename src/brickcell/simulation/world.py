"""MuJoCo scene used only by the separately installed simulation environment.

Motion uses scripted inverse kinematics and known pickup poses, not a learned policy.
Bricks are grasped through friction/contact: no welds or carried-object teleportation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from physicalai_mujoco_plugin.compose import compose_scene
from physicalai_mujoco_plugin.conveyor_demo import ARM_JOINTS, ArmIK
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.sim import place_home

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "duplo"

# All physical parameters in one place. Every value is an ESTIMATE until it is measured on the
# real cell: weigh each brick on a kitchen scale, and do a tilt test for each pair (put a brick
# on the surface, tilt it slowly, read the slip angle; mu = tan(angle)).
#
# Friction triples are MuJoCo (sliding, torsional [m], rolling [m]). MuJoCo has one coefficient
# per contact (no static/kinetic split); the sliding values are the middle of the dry static
# ranges in common engineering friction tables (e.g. The Engineering ToolBox, "Friction and
# Friction Coefficients"; plastics supplier tribology sheets). Torsional ~= mu x contact patch
# radius (~3 mm); rolling is near zero for hard plastic. How a pair gets its value: MuJoCo uses
# the geom with the higher `priority`; on a tie it uses the larger friction. Bricks have priority
# 0, the SO-101 jaws 1 (Menagerie default), plates and table 2. So brick-brick uses
# "brick", brick-plate uses "plate", brick-table uses "table" and brick-jaw uses "jaw".
PHYSICS = {
    "gravity": (0.0, 0.0, -9.81),  # m/s^2
    "timestep": 0.002,  # s; the scene option wins over the SO-101 profile (0.005)
    "iterations": 30,
    "ls_iterations": 50,
    "integrator": "implicitfast",
    "cone": "elliptic",
    "impratio": 10,  # elliptic cones + impratio > 1 reduce slip in grasps
    "friction": {
        "brick": (0.35, 0.001, 0.0001),  # ABS on ABS, dry: 0.3-0.4
        "plate": (0.30, 0.001, 0.0001),  # ABS on polypropylene (KALAS), dry: 0.25-0.35
        "table": (0.40, 0.001, 0.0001),  # ABS on wood / laminate, dry: ~0.4 (assumed table)
        # ABS on 3D-printed PLA jaws, dry: 0.3-0.5. Rubber pads on the jaws: use ~0.8.
        "jaw": (0.40, 0.001, 0.0001),
    },
    "condim": 6,  # sliding + torsional + rolling friction
    # Contact softness for every brick, plate, table and jaw contact. solref time constant must be
    # >= 2 x timestep. Stiffer than the MuJoCo default (0.02) to limit penetration.
    "solref": (0.004, 1.0),
    "solimp": (0.95, 0.99, 0.001, 0.5, 2.0),
    "margin": 0.0,
    # Brick masses in kg: LDraw mesh volume x ABS density 1.05 g/cm^3 (6.08, 10.70, 7.04 and
    # 7.21 cm^3). The LDraw meshes are not checked to be watertight, so weigh the real bricks.
    "brick_mass": {"part-001": 0.0064, "part-002": 0.0112, "part-003": 0.0074, "part-004": 0.0076},
}


def _vec(values) -> str:
    return " ".join(f"{v:g}" for v in values)


def _contact(name: str, priority: int) -> str:
    """Geom attributes for a contact surface, from PHYSICS."""
    return (
        f'friction="{_vec(PHYSICS["friction"][name])}" condim="{PHYSICS["condim"]}" '
        f'priority="{priority}" solref="{_vec(PHYSICS["solref"])}" '
        f'solimp="{_vec(PHYSICS["solimp"])}" margin="{PHYSICS["margin"]}"'
    )


# Sorting targets are IKEA KALAS plates (Ø186 x 23.5 mm; floor top 3.1 mm). The xy value is the
# plate centre. The z value is the brick bottom height at release: a 6 mm drop onto the floor.
BOWLS = {"good": np.array([0.10, -0.22, 0.009]), "defective": np.array([0.10, 0.22, 0.009])}
# Release slots on a plate, in placement order: two rows along X, diagonal first, so that the
# next brick does not land on an earlier one and the opening jaw does not touch a neighbour.
SLOTS = [(-1, -0.024), (1, 0.024), (1, -0.024), (-1, 0.024)]
RELEASE_GRIP = 0.3  # open the moving jaw only a little at release; full open after the lift
PLATE_RGBA = {"good": "0.62 0.86 0.76 1", "defective": "0.90 0.72 0.78 1"}
INSPECTION = np.array([0.25, 0.0, 0.15])


@dataclass(frozen=True)
class Brick:
    """A LEGO DUPLO part. Mesh origin: bottom centre. Body origin: grasp point."""

    part: str
    mesh: str
    yaw: float  # mesh yaw so the jaws close across world Y (32 mm on every part)
    half_x: float  # half length along world X; the +X face looks at the inspection camera
    body_height: float  # height without studs
    grip_z: float  # grasp point above the table
    mass: float  # from PHYSICS["brick_mass"]; weigh the real bricks to confirm
    rgba: tuple[float, float, float]


_MASS = PHYSICS["brick_mass"]
BRICKS = {
    "part-001": Brick(
        "3437", "duplo_2x2_3437", 0.0, 0.016, 0.0192, 0.0105, _MASS["part-001"], (0.80, 0.08, 0.08)
    ),
    "part-002": Brick(
        "3011", "duplo_2x4_3011", 0.0, 0.032, 0.0192, 0.0105, _MASS["part-002"], (0.05, 0.33, 0.80)
    ),
    "part-003": Brick(
        "76371",
        "duplo_1x2x2_76371",
        np.pi / 2,
        0.008,
        0.0384,
        0.017,
        _MASS["part-003"],
        (0.97, 0.76, 0.05),
    ),
    "part-004": Brick(
        "2302",
        "duplo_2x3_curved_top_2302",
        -np.pi / 2,
        0.024,
        0.0192,
        0.0105,
        _MASS["part-004"],
        (0.10, 0.60, 0.28),
    ),
}
OBJECTS = list(BRICKS)

# KALAS plate cross-section (radius, height) in mm, measured from IKEA's product model.
# Only these numbers are used; the mesh is generated here, so no IKEA file is redistributed.
PLATE_PROFILE = [
    (0, 3.7), (30, 3.6), (60, 3.3), (75, 3.1), (79, 4.0), (82, 6.5), (85, 11.0), (88, 16.5),
    (90.5, 21.5), (91.3, 23.5), (93.0, 23.5), (92.8, 21.5), (90.5, 17.0), (87.5, 11.5),
    (84.5, 6.5), (81.5, 3.0), (80, 1.2), (79, 0.0), (66, 0.0), (63, 0.8), (60, 1.5),
    (30, 1.8), (0, 1.9),
]  # fmt: skip


def write_plate_obj(path: Path, segments: int = 96) -> None:
    """Revolve PLATE_PROFILE around Z into a closed OBJ mesh in metres."""
    profile = [(r / 1000, z / 1000) for r, z in PLATE_PROFILE]
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    vertices: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for r, z in profile:
        if r == 0:
            vertices.append((0.0, 0.0, z))
            rings.append([len(vertices)] * segments)
        else:
            rings.append(list(range(len(vertices) + 1, len(vertices) + segments + 1)))
            vertices.extend((r * np.cos(a), r * np.sin(a), z) for a in angles)
    faces = []
    for a, b in zip(rings, rings[1:]):
        for j in range(segments):
            k = (j + 1) % segments
            quad = [a[j], b[j], b[k], a[k]]
            for tri in (quad[:3], [quad[0], quad[2], quad[3]]):
                if len(set(tri)) == 3:
                    faces.append(tri)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")


def _quat(yaw: float, tilt: float) -> str:
    """Quaternion for a rotation about Z by yaw, then about the new Y by tilt."""
    cz, sz, cy, sy = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(tilt / 2), np.sin(tilt / 2)
    return f"{cz * cy:.6f} {-sz * sy:.6f} {cz * sy:.6f} {sz * cy:.6f}"


def plate_xml(name: str, pos: np.ndarray) -> str:
    """Visual mesh plus primitive collision: a floor disc and a ring of sloped wall boxes."""
    tilt = np.arctan2(91.3 - 79, 23.5 - 4.0)
    contact = _contact("plate", 2)
    walls = "".join(
        f'<geom type="box" size=".001 .0092 .0116" quat="{_quat(a, tilt)}" '
        f'pos="{0.0860 * np.cos(a):.5f} {0.0860 * np.sin(a):.5f} .0132" rgba="0 0 0 0" {contact}/>'
        for a in np.linspace(0, 2 * np.pi, 32, endpoint=False)
    )
    return (
        f'<body name="plate_{name}" pos="{pos[0]} {pos[1]} 0">'
        f'<geom name="plate_{name}-visual" type="mesh" mesh="kalas_plate" '
        f'rgba="{PLATE_RGBA[name]}" contype="0" conaffinity="0"/>'
        f'<geom name="plate_{name}-floor" type="cylinder" size=".079 .00155" pos="0 0 .00155" '
        f'rgba="0 0 0 0" {contact}/>{walls}</body>'
    )


def brick_xml(index: int, name: str, b: Brick) -> str:
    face_z = b.body_height / 2 - b.grip_z
    mark_z = min(0.005, b.body_height / 2 - 0.003)
    return (
        f'<body name="{name}" pos=".25 {(index - 1.5) * 0.065} {b.grip_z}">'
        f'<freejoint name="{name}-joint"/>'
        f'<geom name="{name}-solid" type="mesh" mesh="{b.mesh}" pos="0 0 {-b.grip_z}" '
        f'euler="0 0 {b.yaw}" mass="{b.mass}" {_contact("brick", 0)} '
        f'rgba="{b.rgba[0]} {b.rgba[1]} {b.rgba[2]} 1"/>'
        f'<geom name="{name}-mark" type="box" pos="{b.half_x + 0.0001} 0 {face_z:.5f}" '
        f'size=".0001 .007 {mark_z:.4f}" rgba=".015 .015 .015 0" contype="0" conaffinity="0" '
        f'mass="0"/></body>'
    )


def scene_xml(directory: Path) -> str:
    """Self-contained scene; Menagerie supplies only the SO-101 geometry."""
    plate = directory / "kalas_plate.obj"
    write_plate_obj(plate)
    assets = f'<mesh name="kalas_plate" file="{plate}" inertia="shell"/>'
    assets += "".join(
        f'<mesh name="{b.mesh}" file="{ASSETS / (b.mesh + ".stl")}" inertia="convex"/>'
        for b in BRICKS.values()
    )
    plates = "".join(plate_xml(name, pos) for name, pos in BOWLS.items())
    bricks = "".join(brick_xml(i, name, b) for i, (name, b) in enumerate(BRICKS.items()))
    p = PHYSICS
    return f"""<mujoco model="Brickcell two plate inspection">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="{p["timestep"]}" gravity="{_vec(p["gravity"])}" integrator="{p["integrator"]}" cone="{p["cone"]}" iterations="{p["iterations"]}" ls_iterations="{p["ls_iterations"]}" impratio="{p["impratio"]}"/>
      <visual><global offwidth="640" offheight="480"/><headlight ambient=".4 .4 .4" diffuse=".5 .5 .5" specular="0 0 0"/></visual>
      <default><geom rgba=".9 .48 .08 1" condim="4" friction="1 .01 .001"/></default>
      <asset>{assets}</asset>
      <worldbody>
        <light pos=".4 -.3 1" dir="0 0 -1" diffuse=".7 .7 .7" castshadow="false"/>
        <geom name="table" type="plane" size="1 1 .01" rgba=".15 .18 .22 1" {_contact("table", 2)}/>
        {plates}{bricks}<frame name="robot_mount"/>
        <camera name="inspection" pos=".55 0 .24" xyaxes="0 1 0 -.287348 0 .957826" fovy="22"/>
        <camera name="overview" pos=".3 0 1.1" xyaxes="1 0 0 0 1 0" fovy="50"/>
      </worldbody></mujoco>"""


def apply_jaw_physics(model: mujoco.MjModel) -> None:
    """Give the colliding SO-101 jaw geoms the PHYSICS jaw friction and contact softness.

    The jaws come from Menagerie, so they are set on the compiled model, not in scene_xml.
    """
    jaws = {model.body(name).id for name in ("gripper", "moving_jaw_so101_v1")}
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in jaws and (model.geom_contype[g] or model.geom_conaffinity[g]):
            model.geom_priority[g] = 1
            model.geom_friction[g] = PHYSICS["friction"]["jaw"]
            model.geom_condim[g] = PHYSICS["condim"]
            model.geom_solref[g] = PHYSICS["solref"]
            model.geom_solimp[g] = PHYSICS["solimp"]
            model.geom_margin[g] = PHYSICS["margin"]


class World:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "two_plate.xml"
        path.write_text(scene_xml(directory))
        composed = compose_scene(path, get_profile("so101"))
        self.model = composed.model
        apply_jaw_physics(self.model)
        self.data = mujoco.MjData(self.model)
        for robot in composed.robots:
            place_home(self.model, self.data, robot)
        mujoco.mj_forward(self.model, self.data)
        self.ik = ArmIK(self.model)
        self.actuators = [self.model.actuator(name).id for name in (*ARM_JOINTS, "gripper")]
        self.q = self.data.qpos[self.ik.qpos_adr].copy()
        self.target = self.ik.tcp(self.data).copy()
        self.grip = 0.5
        self.yaw = np.pi / 2
        self.renderer = None
        self.active: str | None = None
        self.destination: str | None = None
        self.tick_callback = lambda: None
        self.before_step = lambda: None
        self.reset()
        self.target = np.array([0.25, 0.0, 0.12])
        for _ in range(100):
            self.q, _ = self.ik.solve(self.q, self.target, self.yaw)
        self.data.qpos[self.ik.qpos_adr] = self.q
        self.data.ctrl[self.actuators] = [*self.q, self.grip]
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """Reset fixtures explicitly between runs, never while carrying an object."""
        for i, name in enumerate(OBJECTS):
            joint = self.model.joint(f"{name}-joint")
            a, v = joint.qposadr[0], joint.dofadr[0]
            self.data.qpos[a : a + 7] = [0.25, (i - 1.5) * 0.065, BRICKS[name].grip_z, 1, 0, 0, 0]
            self.data.qvel[v : v + 6] = 0
            # Fixture truth controls appearance only; it is absent from the camera/action API.
            self.model.geom(f"{name}-mark").rgba[3] = float(i % 2)
        self.active = self.destination = None
        self.counts = {"good": 0, "defective": 0}
        mujoco.mj_forward(self.model, self.data)

    def position(self, object_id: str) -> np.ndarray:
        return self.data.xpos[self.model.body(object_id).id].copy()

    def tick(self) -> None:
        self.q, _ = self.ik.solve(self.q, self.target, self.yaw, iters=32)
        self.data.ctrl[self.actuators] = [*self.q, self.grip]
        # Recorder observes the state and commanded action before applying physics.
        self.before_step()
        mujoco.mj_step(self.model, self.data, nstep=10)
        self.tick_callback()

    def hold(self, seconds: float) -> None:
        for _ in range(round(seconds / 0.02)):
            self.tick()

    def move(self, target: np.ndarray, speed: float = 0.12) -> None:
        start = self.target.copy()
        steps = max(1, int(np.linalg.norm(target - start) / speed / 0.02))
        for i in range(1, steps + 1):
            self.target = start + (target - start) * i / steps
            self.tick()
        self.hold(0.4)
        if np.linalg.norm(self.ik.tcp(self.data) - target) > 0.025:
            raise RuntimeError("Simulated arm failed to reach commanded pose")

    def action(self, operation: str, object_id: str) -> None:
        if object_id not in OBJECTS:
            raise ValueError("Unknown fixture object")
        if operation == "pick":
            if self.active is not None and self.active != object_id:
                raise ValueError("Another object is active")
            self.active, self.destination = object_id, None
            self.grip = 0.5
            p = self.position(object_id)
            self.move(np.array([p[0], p[1], 0.16]))
            self.move(p + [0, 0, 0.005], speed=0.06)
            self.grip = -0.15
            self.hold(0.5)
            self.move(np.array([p[0], p[1], 0.16]), speed=0.08)
            if self.position(object_id)[2] < 0.10:
                raise RuntimeError("Contact grasp failed: object did not lift")
        elif object_id != self.active:
            raise ValueError("Action object does not match active object")
        elif operation in {"present", "adjust_view"}:
            self.move(INSPECTION + [0, 0.003 if operation == "adjust_view" else 0, 0])
            self.hold(0.5)
        elif operation in {"move_good", "move_defective"}:
            self.destination = operation.removeprefix("move_")
            # Goal for the grasp point, converted to a TCP goal with the measured grasp offset.
            brick = BRICKS[object_id]
            side, y = SLOTS[self.counts[self.destination] % len(SLOTS)]
            x = side * min(0.008 + brick.half_x, 0.062 - brick.half_x)
            goal = BOWLS[self.destination] + [x, y, brick.grip_z]
            goal += self.ik.tcp(self.data) - self.position(object_id)
            self.move(np.array([goal[0], goal[1], 0.18]))
            self.move(goal, speed=0.06)
        elif operation == "release":
            if self.destination is None:
                raise ValueError("Cannot release without a selected plate")
            # Open slowly and only a little, so the jaws do not push the brick, then lift clear.
            start = self.grip
            for i in range(1, 16):
                self.grip = start + (RELEASE_GRIP - start) * i / 15
                self.tick()
            self.hold(0.5)
            self.move(self.target + [0, 0, 0.05], speed=0.06)
            self.grip = 0.5
            self.hold(0.3)
        elif operation == "verify_release":
            self.hold(0.5)
            p = self.position(object_id)
            velocity = self.model.joint(f"{object_id}-joint").dofadr[0]
            if (
                self.destination is None
                or np.max(np.abs(p[:2] - BOWLS[self.destination][:2])) > 0.055
                or p[2] > 0.08
                or np.linalg.norm(self.data.qvel[velocity : velocity + 3]) > 0.01
            ):
                raise RuntimeError("Simulated object is not settled on the commanded plate")
            self.counts[self.destination] += 1
            self.active = None
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def camera(self, camera: str = "inspection") -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


def inspection_crop(rgb: np.ndarray) -> np.ndarray:
    """Fixed camera ROI shared by rendered training and live capture, never truth-dependent."""
    if rgb.shape != (256, 256, 3):
        raise ValueError("Inspection camera must produce 256x256 RGB")
    return rgb[90:220, 48:208].copy()
