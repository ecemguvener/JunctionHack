"""MuJoCo scene used only by the separately installed simulation environment.

Motion uses scripted inverse kinematics and known pickup poses, not a learned policy.
Bricks are grasped through friction/contact: no welds or carried-object teleportation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
from physicalai_mujoco_plugin.compose import compose_scene, compose_scene_spec
from physicalai_mujoco_plugin.conveyor_demo import ARM_JOINTS, ArmIK
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.sim import place_home

from brickcell.simulation.scene_config import (
    CV_TO_MUJOCO,
    DEFAULT_BRICK_COLORS,
    DEFAULT_PLATE_COLORS,
    Camera,
    SceneConfig,
    focal_from_fovy,
    load_scene_config,
    parse_color,
)

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
# Default colours. configs/sim-scene.yaml (see scene_config.py) can change them for a World.
PLATE_RGBA = {name: f"{_vec(rgb)} 1" for name, rgb in DEFAULT_PLATE_COLORS.items()}
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
    rgba: tuple[float, float, float]  # default colour; World.bricks has the configured colour


_MASS = PHYSICS["brick_mass"]
_RGB = DEFAULT_BRICK_COLORS
BRICKS = {
    "part-001": Brick(
        "3437", "duplo_2x2_3437", 0.0, 0.016, 0.0192, 0.0105, _MASS["part-001"], _RGB["part-001"]
    ),
    "part-002": Brick(
        "3011", "duplo_2x4_3011", 0.0, 0.032, 0.0192, 0.0105, _MASS["part-002"], _RGB["part-002"]
    ),
    "part-003": Brick(
        "76371",
        "duplo_1x2x2_76371",
        np.pi / 2,
        0.008,
        0.0384,
        0.017,
        _MASS["part-003"],
        _RGB["part-003"],
    ),
    "part-004": Brick(
        "2302",
        "duplo_2x3_curved_top_2302",
        -np.pi / 2,
        0.024,
        0.0192,
        0.0105,
        _MASS["part-004"],
        _RGB["part-004"],
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


def plate_xml(name: str, pos: np.ndarray, rgb: tuple[float, float, float] | None = None) -> str:
    """Visual mesh plus primitive collision: a floor disc and a ring of sloped wall boxes."""
    rgba = PLATE_RGBA[name] if rgb is None else f"{rgb[0]} {rgb[1]} {rgb[2]} 1"
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
        f'rgba="{rgba}" contype="0" conaffinity="0"/>'
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


def scene_xml(directory: Path, scene: SceneConfig | None = None) -> str:
    """Self-contained scene; Menagerie supplies only the SO-101 geometry.

    ``scene`` gives the world-fixed cameras and the colours; None loads the default scene config.
    """
    scene = load_scene_config(scene)
    plate = (
        directory.resolve() / "kalas_plate.obj"
    )  # absolute: MuJoCo resolves relative paths from the XML directory
    write_plate_obj(plate)
    assets = f'<mesh name="kalas_plate" file="{plate}" inertia="shell"/>'
    assets += "".join(
        f'<mesh name="{b.mesh}" file="{ASSETS / (b.mesh + ".stl")}" inertia="convex"/>'
        for b in BRICKS.values()
    )
    plates = "".join(plate_xml(name, pos, scene.plate_colors[name]) for name, pos in BOWLS.items())
    bricks = "".join(
        brick_xml(i, name, replace(b, rgba=scene.brick_colors[name]))
        for i, (name, b) in enumerate(BRICKS.items())
    )
    cameras = "".join(camera.xml() for camera in scene.world_cameras())
    width, height = scene.offscreen_size()  # a Renderer cannot be larger than this buffer
    p = PHYSICS
    return f"""<mujoco model="Brickcell two plate inspection">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="{p["timestep"]}" gravity="{_vec(p["gravity"])}" integrator="{p["integrator"]}" cone="{p["cone"]}" iterations="{p["iterations"]}" ls_iterations="{p["ls_iterations"]}" impratio="{p["impratio"]}"/>
      <visual><global offwidth="{width}" offheight="{height}"/><headlight ambient=".4 .4 .4" diffuse=".5 .5 .5" specular="0 0 0"/></visual>
      <default><geom rgba=".9 .48 .08 1" condim="4" friction="1 .01 .001"/></default>
      <asset>{assets}</asset>
      <worldbody>
        <light pos=".4 -.3 1" dir="0 0 -1" diffuse=".7 .7 .7" castshadow="false"/>
        <geom name="table" type="plane" size="1 1 .01" rgba=".15 .18 .22 1" {_contact("table", 2)}/>
        {plates}{bricks}<frame name="robot_mount"/>
        {cameras}
      </worldbody></mujoco>"""


def attach_cameras(path: Path, profile, cameras: list[Camera]) -> mujoco.MjModel:
    """Compile the composed scene again with cameras mounted on bodies (wrist camera).

    compose_scene gives only the compiled model, so this gets the same scene as an MjSpec, adds
    the cameras and compiles it. A camera with the name of a robot camera replaces that camera.
    """
    spec = compose_scene_spec(path, profile)
    for camera in cameras:
        body = spec.body(camera.attach)
        if body is None:
            names = [b.name for b in spec.bodies if b.name]
            raise ValueError(
                f"Camera '{camera.name}': attach body '{camera.attach}' is not in the model. "
                f"Bodies: {names}"
            )
        existing = spec.camera(camera.name)
        if existing is not None:
            spec.delete(existing)
        added = body.add_camera(name=camera.name, pos=list(camera.pos), quat=camera.quat())
        sensor = camera.sensor()
        if sensor is None:
            added.fovy = camera.fovy
        else:
            added.resolution = sensor["resolution"]
            added.sensor_size = sensor["sensorsize"]
            added.focal_pixel = sensor["focalpixel"]
            added.principal_pixel = sensor["principalpixel"]
    return spec.compile()


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
    def __init__(self, directory: Path, scene_config: object = None):
        """``scene_config``: SceneConfig, mapping or YAML path; None loads the default file."""
        self.scene = load_scene_config(scene_config)
        # Bricks with the configured colours. set_brick_color keeps this up to date.
        self.bricks = {
            name: replace(b, rgba=self.scene.brick_colors[name]) for name, b in BRICKS.items()
        }
        profile = get_profile("so101")
        robot_cameras = {c.name: c.body for c in profile.cameras}
        for camera in self.scene.world_cameras():
            if camera.name in robot_cameras:
                raise ValueError(
                    f"Camera '{camera.name}' is a camera of the robot. To replace it, give "
                    f"`attach: <body>` (the robot has it on '{robot_cameras[camera.name]}')."
                )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "two_plate.xml"
        path.write_text(scene_xml(directory, self.scene))
        composed = compose_scene(path, profile)
        self.model = composed.model
        attached = self.scene.attached_cameras()
        if attached:
            self.model = attach_cameras(path, profile, attached)
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
        self.renderers: dict[tuple[int, int], mujoco.Renderer] = {}  # one per (width, height)
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

    def camera_names(self) -> list[str]:
        return [self.model.camera(i).name for i in range(self.model.ncam)]

    def camera(self, camera: str = "inspection") -> np.ndarray:
        """RGB image of a model camera at its configured size (256x256 if not configured)."""
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError(f"Unknown camera '{camera}'. Cameras: {self.camera_names()}")
        size = self.scene.camera_size(camera)
        renderer = self.renderers.get(size)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=size[1], width=size[0])
            self.renderers[size] = renderer
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def cameras(self) -> list[dict]:
        """Name, pose and pinhole intrinsics of each model camera and each disabled config entry.

        ``pos``, ``quat`` (MuJoCo camera frame, wxyz) and ``rotation_cv`` (OpenCV camera frame)
        are in the parent frame: the world, or the ``attach`` body. ``world_pos`` and
        ``world_rotation_cv`` are the pose in the world at this time. fx, fy, cx, cy are in
        OpenCV pixels for the rendered ``width`` x ``height`` image.
        """
        result = []
        for i in range(self.model.ncam):
            name = self.model.camera(i).name
            width, height = self.scene.camera_size(name)
            body = int(self.model.cam_bodyid[i])
            sensor, intrinsic = self.model.cam_sensorsize[i], self.model.cam_intrinsic[i]
            fovy = float(self.model.cam_fovy[i])
            if sensor[1] > 0:
                fx = float(intrinsic[0] / sensor[0]) * width
                fy = float(intrinsic[1] / sensor[1]) * height
                cx = (width - 1) / 2 - float(intrinsic[2] / sensor[0]) * width
                cy = (height - 1) / 2 - float(intrinsic[3] / sensor[1]) * height
            else:
                fx = fy = focal_from_fovy(fovy, height)
                cx, cy = (width - 1) / 2, (height - 1) / 2
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, self.model.cam_quat[i])
            result.append(
                {
                    "name": name,
                    "enabled": True,
                    "source": "scene config" if name in self.scene.cameras else "robot profile",
                    "attach": self.model.body(body).name if body else None,
                    "pos": self.model.cam_pos[i].tolist(),
                    "quat": self.model.cam_quat[i].tolist(),
                    "rotation_cv": (rotation.reshape(3, 3) @ CV_TO_MUJOCO).tolist(),
                    "world_pos": self.data.cam_xpos[i].tolist(),
                    "world_rotation_cv": (
                        self.data.cam_xmat[i].reshape(3, 3) @ CV_TO_MUJOCO
                    ).tolist(),
                    "fovy": fovy,
                    "width": width,
                    "height": height,
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                }
            )
        result += [
            {**camera.as_dict(), "source": "scene config"}
            for camera in self.scene.cameras.values()
            if not camera.enabled
        ]
        return result

    def set_brick_color(self, part_id: str, color: object) -> tuple[float, float, float]:
        """Change a brick colour in the running model. ``color``: "#rrggbb" or [r, g, b] 0-1."""
        if part_id not in self.bricks:
            raise ValueError(f"Unknown part id '{part_id}'. Known: {list(self.bricks)}")
        rgb = parse_color(color, f"brick colour {part_id}")
        self.model.geom(f"{part_id}-solid").rgba[:3] = rgb
        self.bricks[part_id] = replace(self.bricks[part_id], rgba=rgb)
        return rgb

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()


def inspection_crop(rgb: np.ndarray) -> np.ndarray:
    """Fixed camera ROI shared by rendered training and live capture, never truth-dependent."""
    if rgb.shape != (256, 256, 3):
        raise ValueError("Inspection camera must produce 256x256 RGB")
    return rgb[90:220, 48:208].copy()
