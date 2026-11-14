"""MuJoCo scene used only by the separately installed simulation environment.

Motion uses scripted inverse kinematics and known pickup poses, not a learned policy.
Bricks are grasped through friction/contact: no welds or carried-object teleportation.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from physicalai_mujoco_plugin.compose import compose_scene
from physicalai_mujoco_plugin.conveyor_demo import ARM_JOINTS, ArmIK
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.sim import place_home

BOWLS = {"good": np.array([0.10, -0.22, 0.12]), "defective": np.array([0.10, 0.22, 0.12])}
INSPECTION = np.array([0.25, 0.0, 0.15])
OBJECTS = [f"part-{i + 1:03d}" for i in range(4)]


def scene_xml() -> str:
    """Self-contained scene; Menagerie supplies only the SO-101 geometry."""
    bowls = ""
    for name, pos in BOWLS.items():
        color = "0.18 0.65 0.35 1" if name == "good" else "0.8 0.22 0.20 1"
        walls = f'<geom type="box" size=".065 .065 .004" pos="0 0 .004" rgba="{color}"/>'
        for x, y, sx, sy in [
            (0.065, 0, 0.004, 0.065),
            (-0.065, 0, 0.004, 0.065),
            (0, 0.065, 0.065, 0.004),
            (0, -0.065, 0.065, 0.004),
        ]:
            walls += f'<geom type="box" size="{sx} {sy} .025" pos="{x} {y} .025" rgba="{color}"/>'
        bowls += f'<body name="bowl_{name}" pos="{pos[0]} {pos[1]} 0"><geom type="box" size=".069 .069 .002" rgba="{color}"/>{walls}</body>'
    bricks = ""
    for i, name in enumerate(OBJECTS):
        studs = "".join(
            f'<geom type="cylinder" pos="{x} {y} .018" size=".004 .003" contype="0" conaffinity="0" mass="0"/>'
            for x in [-0.008, 0.008]
            for y in [-0.008, 0.008]
        )
        bricks += f'''<body name="{name}" pos=".25 {(i - 1.5) * 0.065} .017">
          <freejoint name="{name}-joint"/>
          <geom name="{name}-solid" type="box" size=".016 .016 .016" mass=".02" friction="3 .01 .001" rgba=".9 .48 .08 1"/>
          {studs}
          <geom name="{name}-mark" type="box" pos=".0161 0 0" size=".0001 .012 .009" rgba=".015 .015 .015 0" contype="0" conaffinity="0" mass="0"/>
        </body>'''
    return f"""<mujoco model="Brickcell two bowl inspection">
      <compiler angle="radian" autolimits="true"/>
      <option timestep=".002" integrator="implicitfast" cone="elliptic" iterations="30" impratio="10"/>
      <visual><global offwidth="640" offheight="480"/><headlight ambient=".4 .4 .4" diffuse=".5 .5 .5" specular="0 0 0"/></visual>
      <default><geom rgba=".9 .48 .08 1" condim="4" friction="1 .01 .001"/></default>
      <worldbody>
        <light pos=".4 -.3 1" dir="0 0 -1" diffuse=".7 .7 .7" castshadow="false"/>
        <geom name="table" type="plane" size="1 1 .01" rgba=".15 .18 .22 1"/>
        {bowls}{bricks}<frame name="robot_mount"/>
        <camera name="inspection" pos=".55 0 .24" xyaxes="0 1 0 -.287348 0 .957826" fovy="22"/>
        <camera name="overview" pos=".3 0 1.1" xyaxes="1 0 0 0 1 0" fovy="50"/>
      </worldbody></mujoco>"""


class World:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "two_bowl.xml"
        path.write_text(scene_xml())
        composed = compose_scene(path, get_profile("so101"))
        self.model = composed.model
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
            self.data.qpos[a : a + 7] = [0.25, (i - 1.5) * 0.065, 0.017, 1, 0, 0, 0]
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
            offset = self.ik.tcp(self.data) - self.position(object_id)
            goal = BOWLS[self.destination].copy()
            goal[:2] += offset[:2]
            goal[0] += -0.024 if self.counts[self.destination] % 2 == 0 else 0.024
            self.move(np.array([goal[0], goal[1], 0.18]))
            self.move(goal)
        elif operation == "release":
            if self.destination is None:
                raise ValueError("Cannot release without a selected bowl")
            self.grip = 0.5
            self.hold(1)
            self.move(self.target + [0, 0, 0.05])
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
                raise RuntimeError("Simulated object is not settled inside the commanded bowl")
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
