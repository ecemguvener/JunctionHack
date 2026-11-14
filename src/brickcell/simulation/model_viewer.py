"""Browser viewer to compare the simulation's 3D models with the real objects.

Shows the four LEGO DUPLO bricks and one IKEA KALAS plate at true scale on a 1 cm grid.
You can move each model with a gizmo, show the collision shapes that MuJoCo uses, measure
between two points, and drop the bricks into the plate with MuJoCo physics. A station map shows
the configured cameras (configs/sim-scene.yaml) relative to the arm base, the plates and the bricks.

    MUJOCO_GL=egl PYTHONPATH=src python -m brickcell.simulation.model_viewer --port 19092

The server binds to 127.0.0.1 only. Use an SSH tunnel to see it from another computer.
"""

from __future__ import annotations

import argparse
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
import viser.transforms as vtf

from brickcell.simulation.scene_config import SceneConfig, color_hex, load_scene_config
from brickcell.simulation.world import (
    ASSETS,
    BOWLS,
    BRICKS,
    INSPECTION,
    PLATE_RGBA,
    brick_xml,
    plate_xml,
    scene_xml,
    write_plate_obj,
)

PLATE = "plate"
FLOOR_RADIUS = 0.079  # KALAS inner floor radius (PLATE_PROFILE r = 79 mm at the bottom)
SNAP_DISTANCE = 0.003
# Browser camera (position, look_at): close to the models, and far enough to see the cameras.
CHECK_VIEW = ((0.17, -0.33, 0.17), (0.0, -0.07, 0.0))
STATION_VIEW = ((0.9, -0.8, 0.8), (0.25, 0.0, 0.3))
GIZMO_MODES = ("Selected model", "All models", "Hidden")
# The user's plates. "good" and "defective" are the sim colours from the scene config (green and
# pink by default); light blue is only for this viewer.
LIGHT_BLUE = (0.60, 0.80, 0.93)
PLATE_OUTER_RADIUS = 0.093  # KALAS rim radius, for the station map
# Drop-test layout over the plate centre (x, y) in metres: two rows of two bricks.
DROP_LAYOUT = [(0.038, -0.022), (-0.038, -0.022), (-0.038, 0.022), (0.038, 0.022)]


def rgb255(rgba: str | tuple[float, ...]) -> tuple[int, int, int]:
    values = [float(v) for v in rgba.split()] if isinstance(rgba, str) else list(rgba)
    return tuple(int(round(255 * min(max(v, 0.0), 1.0))) for v in values[:3])  # type: ignore[return-value]


def add_camera_frustums(scene, cameras: list[dict], root: str = "/cameras") -> list:
    """Draw the world-fixed cameras as frustums with labels. Returns the handles.

    ``cameras``: dicts with name, enabled, attach, pos, rotation_cv, fovy, width, height
    (World.cameras() or Camera.as_dict()). A camera on a body (``attach``) is not drawn.
    """
    handles = []
    for camera in cameras:
        if camera["attach"] is not None:
            continue  # it moves with a robot body; the pose is not fixed in the world
        rotation = np.array(camera["rotation_cv"])  # viser frustums use the OpenCV camera frame
        handles.append(
            scene.add_camera_frustum(
                f"{root}/{camera['name']}",
                fov=float(np.radians(camera["fovy"])),
                aspect=camera["width"] / camera["height"],
                scale=0.06,
                thickness=1.5,
                thickness_units="screen",
                color=(255, 200, 40) if camera["enabled"] else (120, 120, 120),
                wxyz=vtf.SO3.from_matrix(rotation).wxyz,
                position=camera["pos"],
                cast_shadow=False,
                receive_shadow=False,
            )
        )
        text = camera["name"] if camera["enabled"] else f"{camera['name']} (off)"
        handles.append(
            scene.add_label(
                f"{root}/{camera['name']}-label",
                f"CAM {text}",
                position=camera["pos"],
                anchor="bottom-center",
                font_screen_scale=0.8,
            )
        )
    return handles


def yaw_wxyz(yaw: float) -> np.ndarray:
    return vtf.SO3.from_z_radians(yaw).wxyz


def mat_from_mj_quat(quat: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=float))
    return out.reshape(3, 3)


@dataclass
class Model:
    key: str
    title: str
    detail: str
    mesh: trimesh.Trimesh  # model frame: metres, Z up, origin at the bottom centre
    collision: trimesh.Trimesh
    collision_note: str
    colour: tuple[int, int, int]
    home_pos: np.ndarray
    home_wxyz: np.ndarray
    smooth: bool = False
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    handles: dict = field(default_factory=dict)

    def rotation(self) -> np.ndarray:
        return vtf.SO3(self.wxyz).as_matrix()

    def world_vertices(self) -> np.ndarray:
        return self.mesh.vertices @ self.rotation().T + self.pos

    def world_triangles(self) -> np.ndarray:
        return self.mesh.triangles @ self.rotation().T + self.pos


def build_drop_model(
    directory: Path, plate_name: str, scene: SceneConfig | None = None
) -> mujoco.MjModel:
    """Plate + four bricks + table, from the world.py helpers. No robot."""
    # scene_xml also writes kalas_plate.obj into directory
    root = ET.fromstring(scene_xml(directory, scene))
    keep = "".join(
        ET.tostring(root.find(tag), encoding="unicode")
        for tag in ("compiler", "option", "visual", "default", "asset")
        if root.find(tag) is not None
    )
    bricks = "".join(brick_xml(i, name, b) for i, (name, b) in enumerate(BRICKS.items()))
    xml = f"""<mujoco model="Brickcell model viewer drop test">{keep}
      <worldbody>
        <light pos="0 -.3 1" dir="0 .3 -1" diffuse=".7 .7 .7" castshadow="false"/>
        <geom name="table" type="plane" size="1 1 .01" rgba=".15 .18 .22 1"/>
        {plate_xml(plate_name, np.zeros(3))}{bricks}
        <camera name="check" pos=".22 -.30 .22" xyaxes=".81 .59 0 -.30 .41 .86" fovy="40"/>
      </worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def plate_collision_mesh(model: mujoco.MjModel, plate_name: str) -> trimesh.Trimesh:
    """Collision primitives of the plate body, read from the compiled MuJoCo model."""
    body = model.body(f"plate_{plate_name}").id
    parts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        size, kind = model.geom_size[g], model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            part = trimesh.creation.box(extents=2 * size)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            part = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=96)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            part = trimesh.creation.capsule(radius=size[0], height=2 * size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            part = trimesh.creation.icosphere(radius=size[0])
        else:
            continue
        transform = np.eye(4)
        transform[:3, :3] = mat_from_mj_quat(model.geom_quat[g])
        transform[:3, 3] = model.geom_pos[g]
        part.apply_transform(transform)
        parts.append(part)
    return trimesh.util.concatenate(parts)


def body_to_mesh_frame(model: mujoco.MjModel, geom: int) -> tuple[np.ndarray, np.ndarray]:
    """Pose of the original mesh frame in the body frame (MuJoCo re-centres mesh geoms)."""
    mesh = model.geom_dataid[geom]
    rot = mat_from_mj_quat(model.geom_quat[geom]) @ mat_from_mj_quat(model.mesh_quat[mesh]).T
    return rot, model.geom_pos[geom] - rot @ model.mesh_pos[mesh]


def mujoco_hull(model: mujoco.MjModel, mesh_name: str) -> trimesh.Trimesh:
    """The convex hull MuJoCo computed for collisions, in the original mesh frame."""
    mesh = model.mesh(mesh_name).id
    adr = model.mesh_graphadr[mesh]
    if adr < 0:
        raise RuntimeError(f"MuJoCo has no convex hull for mesh {mesh_name}")
    graph = model.mesh_graph[adr:]
    nvert, nface = int(graph[0]), int(graph[1])
    faces = graph[2 + 3 * nvert + 3 * nface : 2 + 3 * nvert + 6 * nface].reshape(-1, 3)
    start = model.mesh_vertadr[mesh]
    local = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    vertices = local @ mat_from_mj_quat(model.mesh_quat[mesh]).T + model.mesh_pos[mesh]
    hull = trimesh.Trimesh(vertices, faces, process=False)
    hull.remove_unreferenced_vertices()
    return hull


def upright(rot: vtf.SO3) -> vtf.SO3:
    """Smallest rotation that makes the model's +Z axis point up again."""
    up = rot.as_matrix()[:, 2]
    axis = np.cross(up, [0.0, 0.0, 1.0])
    if np.linalg.norm(axis) < 1e-9:
        axis = rot.as_matrix()[:, 0] if up[2] < 0 else np.array([0.0, 0.0, 1.0])
        angle = np.pi if up[2] < 0 else 0.0
    else:
        angle = float(np.arccos(np.clip(up[2], -1.0, 1.0)))
    axis = axis / np.linalg.norm(axis)
    return vtf.SO3.exp(axis * angle) @ rot


def ray_triangles(origin: np.ndarray, direction: np.ndarray, tris: np.ndarray) -> float | None:
    """Distance to the nearest triangle hit (Moller-Trumbore), or None."""
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14
    inv = 1.0 / np.where(ok, det, 1.0)
    s = origin - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    return float(t[hit].min()) if hit.any() else None


class ModelViewer:
    def __init__(self, port: int, scene_config: object = None, station: bool = False):
        self.lock = threading.RLock()
        self.view = CHECK_VIEW
        self.scene_config = load_scene_config(scene_config)
        plates = self.scene_config.plate_colors
        self.plate_colours = {
            "green": plates["good"],
            "pink": plates["defective"],
            "light blue": LIGHT_BLUE,
        }
        self.tmp = tempfile.TemporaryDirectory(prefix="brickcell-model-viewer-")
        directory = Path(self.tmp.name)
        self.plate_name = "good" if "good" in PLATE_RGBA else next(iter(PLATE_RGBA))
        self.drop_model = build_drop_model(directory, self.plate_name, self.scene_config)
        self.models = self._load_models(directory)
        self.selected = next(iter(self.models))
        self.points: list[np.ndarray] = []
        self.playing = False

        self.server = viser.ViserServer(host="127.0.0.1", port=port)
        if self.server.get_port() != port:
            self.server.stop()
            raise RuntimeError(f"Port {port} is already in use")
        self._build_scene()
        self.station_handles = self._build_station_map()
        self._build_gui()
        self.reset_layout()
        if station:
            self.station_layout()

    # ------------------------------------------------------------------ models
    def _load_models(self, directory: Path) -> dict[str, Model]:
        models: dict[str, Model] = {}
        x = -0.10
        for key, b in BRICKS.items():
            mesh = trimesh.load(ASSETS / f"{b.mesh}.stl", force="mesh")
            hull = mujoco_hull(self.drop_model, b.mesh)
            rot = vtf.SO3.from_z_radians(b.yaw).as_matrix()
            half_x = float(np.abs(mesh.vertices @ rot.T)[:, 0].max())
            pos = np.array([x + half_x, -0.14, 0.0])
            x += 2 * half_x + 0.02
            models[key] = Model(
                key=key,
                title=f"{key} · DUPLO {b.part}",
                detail=(
                    f"`{b.mesh}.stl` · sim mass **{b.mass * 1000:.1f} g** · "
                    f"body height (no studs) {b.body_height * 1000:.1f} mm · "
                    f"grip {b.grip_z * 1000:.1f} mm above table"
                ),
                mesh=mesh,
                collision=hull,
                collision_note=f"convex hull, {len(hull.vertices)} vertices",
                colour=rgb255(self.scene_config.brick_colors[key]),
                home_pos=pos,
                home_wxyz=yaw_wxyz(b.yaw),
            )
        write_plate_obj(directory / "kalas_plate_view.obj")
        plate = trimesh.load(directory / "kalas_plate_view.obj", force="mesh", process=False)
        collision = plate_collision_mesh(self.drop_model, self.plate_name)
        n_geoms = sum(
            1
            for g in range(self.drop_model.ngeom)
            if self.drop_model.geom_bodyid[g] == self.drop_model.body(f"plate_{self.plate_name}").id
            and (self.drop_model.geom_contype[g] or self.drop_model.geom_conaffinity[g])
        )
        models[PLATE] = Model(
            key=PLATE,
            title="IKEA KALAS plate",
            detail="Generated from `PLATE_PROFILE` · static body in the sim (no mass)",
            mesh=plate,
            collision=collision,
            collision_note=f"{n_geoms} primitives (floor disc + wall boxes)",
            colour=rgb255(self.plate_colours["green"]),
            home_pos=np.zeros(3),
            home_wxyz=np.array([1.0, 0, 0, 0]),
            smooth=True,
        )
        return models

    # ------------------------------------------------------------------- scene
    def _build_scene(self) -> None:
        scene = self.server.scene
        scene.set_up_direction("+z")
        scene.configure_default_lights(enabled=True, cast_shadow=True)
        scene.add_grid(
            "/grid",
            width=1.2,
            height=1.2,
            cell_size=0.01,
            cell_color=(70, 74, 82),
            section_size=0.1,
            section_color=(150, 156, 168),
            section_thickness=1.6,
            plane_opacity=0.0,
            shadow_opacity=0.35,
            position=(0, 0, -0.0002),
        )
        for model in self.models.values():
            frame = scene.add_frame(f"/models/{model.key}", show_axes=False)
            mesh = scene.add_mesh_simple(
                f"/models/{model.key}/mesh",
                model.mesh.vertices,
                model.mesh.faces,
                color=model.colour,
                flat_shading=not model.smooth,
                side="double",
            )
            collision = scene.add_mesh_simple(
                f"/models/{model.key}/collision",
                model.collision.vertices,
                model.collision.faces,
                color=(255, 140, 30) if model.key != PLATE else (40, 200, 255),
                opacity=0.45,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
                visible=False,
            )
            size = float(max(model.mesh.extents))
            gizmo = scene.add_transform_controls(
                f"/gizmos/{model.key}",
                scale=max(0.05, 0.75 * size),
                line_width=3.0,
                depth_test=False,
                visible=False,
            )
            label = scene.add_label(
                f"/labels/{model.key}", model.title, anchor="bottom-center", font_screen_scale=0.8
            )
            model.handles = dict(
                frame=frame, mesh=mesh, collision=collision, gizmo=gizmo, label=label
            )
            gizmo.on_update(self._gizmo_callback(model))
            mesh.on_click(self._select_callback(model.key))

        @self.server.on_client_connect
        def configure_camera(client: viser.ClientHandle) -> None:
            client.camera.position, client.camera.look_at = self.view
            client.camera.up_direction = (0, 0, 1)
            client.camera.near = 0.005

    def _build_station_map(self) -> list:
        """Cameras, arm base, plates and brick start positions in the robot base frame."""
        scene = self.server.scene
        handles = [
            scene.add_frame("/station/base", axes_length=0.05, axes_radius=0.0015),
            scene.add_label(
                "/station/base-label",
                "ARM BASE (0, 0, 0)",
                position=(0, 0, 0.055),
                anchor="bottom-center",
                font_screen_scale=0.8,
            ),
        ]
        angles = np.linspace(0, 2 * np.pi, 49)
        ring = PLATE_OUTER_RADIUS * np.column_stack((np.cos(angles), np.sin(angles), 0 * angles))
        for name, pos in BOWLS.items():
            centre = np.array([pos[0], pos[1], 0.0005])
            points = (ring + centre).astype(np.float32)
            handles.append(
                scene.add_line_segments(
                    f"/station/plate-{name}",
                    np.stack((points[:-1], points[1:]), axis=1),
                    rgb255(self.scene_config.plate_colors[name]),
                    thickness=2.0,
                    thickness_units="screen",
                )
            )
            handles.append(
                scene.add_label(
                    f"/station/plate-{name}-label",
                    f"{name.upper()} plate",
                    position=centre,
                    anchor="bottom-center",
                    font_screen_scale=0.8,
                )
            )
        for key, pos in self._station_positions().items():
            if key == PLATE:
                continue
            a = np.array([[-0.012, 0, 0], [0, -0.012, 0]]) + pos + [0, 0, 0.0005]
            b = np.array([[0.012, 0, 0], [0, 0.012, 0]]) + pos + [0, 0, 0.0005]
            handles.append(
                scene.add_line_segments(
                    f"/station/start-{key}",
                    np.stack((a, b), axis=1).astype(np.float32),
                    self.models[key].colour,
                    thickness=2.0,
                    thickness_units="screen",
                )
            )
        handles.append(
            scene.add_icosphere(
                "/station/inspection",
                radius=0.003,
                color=(255, 200, 40),
                position=INSPECTION,
                cast_shadow=False,
            )
        )
        handles.append(
            scene.add_label(
                "/station/inspection-label",
                "inspection point",
                position=INSPECTION,
                anchor="bottom-center",
                font_screen_scale=0.8,
            )
        )
        cameras = [camera.as_dict() for camera in self.scene_config.cameras.values()]
        return handles + add_camera_frustums(scene, cameras, "/station/cameras")

    def _station_positions(self) -> dict[str, np.ndarray]:
        """Where the simulation puts each model at the start (model origin = bottom centre)."""
        positions = {key: np.array([0.25, (i - 1.5) * 0.065, 0.0]) for i, key in enumerate(BRICKS)}
        plate = BOWLS[self.plate_name]
        positions[PLATE] = np.array([plate[0], plate[1], 0.0])
        return positions

    def station_layout(self) -> None:
        """Show the station map and put the models where the simulation starts them."""
        if self.playing:
            return
        for key, pos in self._station_positions().items():
            self.set_pose(self.models[key], pos, self.models[key].home_wxyz)
        self._apply_gizmos()
        self.station.value = True  # its callback shows the map
        self._set_view(STATION_VIEW)

    def _set_view(self, view) -> None:
        self.view = view
        for client in self.server.get_clients().values():
            client.camera.position, client.camera.look_at = view

    def _camera_table(self) -> str:
        rows = ["| Camera | Position (m) | fovy | Size | State |", "|---|---|---|---|---|"]
        for camera in self.scene_config.cameras.values():
            x, y, z = camera.pos
            where = f"{x:.3f}, {y:.3f}, {z:.3f}"
            if camera.attach:
                where += f" on `{camera.attach}`"
            rows.append(
                f"| {camera.name} | {where} | {camera.fovy:.1f}° | "
                f"{camera.width} × {camera.height} | {'on' if camera.enabled else 'off'} |"
            )
        return (
            "\n".join(rows)
            + f"\n\nFrom `{Path(self.scene_config.source).name}`. Frame: arm base, Z up. Grey = "
            "off (`enabled: false`). A camera on a robot body (`attach`) is not drawn here; the "
            "simulation server viewer shows its image."
        )

    def _gizmo_callback(self, model: Model):
        def callback(event) -> None:
            if self.playing:
                return
            gizmo = model.handles["gizmo"]
            pos, wxyz = np.array(gizmo.position), np.array(gizmo.wxyz)
            if event.phase == "end" and self.keep_above.value:
                pos[2] += max(0.0, -self._min_z(model, pos, wxyz))
                self.set_pose(model, pos, wxyz)
            else:
                self.set_pose(model, pos, wxyz, move_gizmo=False)
            if model.key != self.selected:
                self.select(model.key)

        return callback

    def _select_callback(self, key: str):
        def callback(_) -> None:
            if not self.measure.value:
                self.select(key)

        return callback

    @staticmethod
    def _min_z(model: Model, pos: np.ndarray, wxyz: np.ndarray) -> float:
        return float((model.mesh.vertices @ vtf.SO3(wxyz).as_matrix().T)[:, 2].min() + pos[2])

    def set_pose(
        self, model: Model, pos: np.ndarray, wxyz: np.ndarray, move_gizmo: bool = True
    ) -> None:
        with self.lock:
            model.pos, model.wxyz = np.asarray(pos, float), np.asarray(wxyz, float)
            h = model.handles
            with self.server.atomic():
                h["frame"].position, h["frame"].wxyz = model.pos, model.wxyz
                if move_gizmo:
                    h["gizmo"].position, h["gizmo"].wxyz = model.pos, model.wxyz
                world = model.world_vertices()
                lo, hi = world.min(0), world.max(0)
                h["label"].position = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + 0.004)
            if model.key == self.selected:
                self._update_readout()

    # --------------------------------------------------------------------- GUI
    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.configure_theme(dark_mode=True, show_share_button=False)
        gui.main_panel.dock_right()
        gui.main_panel.set_width(340)
        gui.add_markdown(
            "## DUPLO + KALAS model check\nTrue scale · grid cell **1 cm**, bright lines "
            "**10 cm**. Drag to orbit, scroll to zoom, right-drag to pan. Click a model to "
            "select it."
        )

        with gui.add_folder("Layout"):
            reset = gui.add_button("Reset layout")
            reset.on_click(lambda _: self.reset_layout())
            self.colour = gui.add_dropdown(
                "Plate colour", tuple(self.plate_colours), initial_value="green"
            )
            self.colour.on_update(lambda _: self._apply_plate_colour())
            self.gizmo_mode = gui.add_dropdown(
                "Move gizmo", GIZMO_MODES, initial_value=GIZMO_MODES[0]
            )
            self.gizmo_mode.on_update(lambda _: self._apply_gizmos())
            self.keep_above = gui.add_checkbox(
                "Stop at table", True, hint="After a drag, lift the model if it went below z = 0."
            )
            self.labels = gui.add_checkbox("Show labels", True)
            self.labels.on_update(lambda _: self._apply_visibility())

        with gui.add_folder("Selected model"):
            self.pick = gui.add_dropdown("Model", tuple(self.models), initial_value=self.selected)
            self.pick.on_update(lambda _: self.select(self.pick.value))
            self.model_colour = gui.add_rgb(
                "Colour",
                self.models[self.selected].colour,
                hint="Colour of the selected model in this viewer. To keep it, copy the line "
                "below into configs/sim-scene.yaml.",
            )
            self.model_colour.on_update(lambda _: self._apply_model_colour())
            self.readout = gui.add_markdown("")
            row = gui.add_button_group("Action", ("On table", "Upright", "Turn 90°", "Tip 90°"))
            row.on_click(lambda event: self._action(event.target.value))

        with gui.add_folder("Station cameras"):
            self.station = gui.add_checkbox(
                "Show station map",
                False,
                hint="Cameras from the scene config, arm base, plate positions and brick start "
                "positions, in the robot base frame.",
            )
            self.station.on_update(lambda _: self._apply_visibility())
            place = gui.add_button("Station layout")
            place.on_click(lambda _: self.station_layout())
            gui.add_markdown(
                "**Station layout** shows the map and puts the models where the simulation "
                "starts them. **Reset layout** goes back to the model check."
            )
            gui.add_markdown(self._camera_table())

        with gui.add_folder("Collision shapes"):
            self.show_collision = gui.add_checkbox(
                "Show what the physics touches",
                False,
                hint="Bricks: convex hull of the mesh (MuJoCo uses the hull). "
                "Plate: the primitives from plate_xml.",
            )
            self.show_collision.on_update(lambda _: self._apply_visibility())
            self.opacity = gui.add_slider("Visual mesh opacity", 0.1, 1.0, 0.05, 1.0)
            self.opacity.on_update(lambda _: self._apply_visibility())
            gui.add_markdown(
                "Orange = brick convex hulls. Blue = plate floor disc and wall boxes. "
                "The visual mesh becomes see-through when the overlay is on."
            )

        with gui.add_folder("Measure"):
            self.measure = gui.add_checkbox(
                "Measure mode (click 2 points)",
                False,
                hint="Click on a mesh or on the table. A third click starts a new measurement.",
            )
            self.measure.on_update(lambda _: self._apply_measure_mode())
            self.snap = gui.add_checkbox(
                "Snap to mesh corner (≤ 3 mm)",
                True,
                hint="Use the nearest mesh vertex if it is within 3 mm of the click.",
            )
            clear = gui.add_button("Clear measurement")
            clear.on_click(lambda _: self._clear_measurement())
            self.measure_text = gui.add_markdown("Turn on measure mode, then click two points.")

        with gui.add_folder("Physics drop test"):
            self.drop_height = gui.add_slider("Drop height (mm)", 5, 100, 5, 50)
            self.drop_button = gui.add_button("Drop the 4 bricks into the plate")
            self.drop_button.on_click(lambda _: self._start_drop())
            self.drop_text = gui.add_markdown(
                "MuJoCo drops the bricks from above the plate (where the plate is now) and "
                "plays the result back here."
            )

        with gui.add_folder("Sizes (from the mesh files)"):
            gui.add_markdown(self._size_table())

    def _size_table(self) -> str:
        rows = ["| Model | Part | X × Y × Z (mm) | Mass |", "|---|---|---|---|"]
        for key, b in BRICKS.items():
            x, y, z = self.models[key].mesh.extents * 1000
            rows.append(
                f"| {key} | {b.part} | {x:.1f} × {y:.1f} × {z:.1f} | {b.mass * 1000:.1f} g |"
            )
        x, y, z = self.models[PLATE].mesh.extents * 1000
        rows.append(f"| plate | KALAS | {x:.1f} × {y:.1f} × {z:.1f} | static |")
        return (
            "\n".join(rows)
            + "\n\nSizes are the mesh bounding box in the mesh's own axes (studs included), "
            "measured at start-up. LDraw sizes are nominal: a real brick is about 0.2 mm "
            "smaller. Select a model for more data."
        )

    def _apply_plate_colour(self) -> None:
        self._set_colour(self.models[PLATE], rgb255(self.plate_colours[self.colour.value]))

    def _apply_model_colour(self) -> None:
        self._set_colour(self.models[self.selected], tuple(self.model_colour.value))

    def _set_colour(self, model: Model, colour: tuple[int, int, int]) -> None:
        if model.colour == colour:
            return
        model.colour = colour
        model.handles["mesh"].color = colour
        if model.key == self.selected:
            self.model_colour.value = colour
            self._update_readout()

    def _apply_gizmos(self) -> None:
        mode = self.gizmo_mode.value
        for model in self.models.values():
            model.handles["gizmo"].visible = not self.playing and (
                mode == "All models" or (mode == "Selected model" and model.key == self.selected)
            )

    def _apply_visibility(self) -> None:
        overlay = self.show_collision.value
        opacity = min(self.opacity.value, 0.35) if overlay else self.opacity.value
        with self.server.atomic():
            for model in self.models.values():
                model.handles["collision"].visible = overlay
                model.handles["mesh"].opacity = None if opacity >= 0.999 else float(opacity)
                model.handles["label"].visible = self.labels.value
            for handle in self.station_handles:
                handle.visible = self.station.value

    def select(self, key: str) -> None:
        with self.lock:
            self.selected = key
            if self.pick.value != key:
                self.pick.value = key
            self.model_colour.value = self.models[key].colour
            self._apply_gizmos()
            self._update_readout()

    def _update_readout(self) -> None:
        model = self.models[self.selected]
        world = model.world_vertices()
        lo, hi = world.min(0) * 1000, world.max(0) * 1000
        roll, pitch, yaw = np.degrees(vtf.SO3(model.wxyz).as_rpy_radians())
        x, y, z = model.pos * 1000
        value = color_hex([v / 255 for v in model.colour])
        entry = "plates: good" if model.key == PLATE else f"bricks: {model.key}"
        self.readout.content = (
            f"**{model.title}**\n\n{model.detail}\n\nCollision: {model.collision_note}\n\n"
            f'Colour `{value}` · scene config: `{entry}: {{color: "{value}"}}`\n\n'
            f"Origin (bottom centre): {x:.1f}, {y:.1f}, {z:.1f} mm\n\n"
            f"Roll {roll:.0f}° · pitch {pitch:.0f}° · yaw {yaw:.0f}°\n\n"
            f"World box: {hi[0] - lo[0]:.1f} × {hi[1] - lo[1]:.1f} mm, "
            f"bottom z {lo[2]:.1f} mm, **top z {hi[2]:.1f} mm**"
        )

    def _action(self, name: str) -> None:
        if self.playing:
            return
        model = self.models[self.selected]
        pos, rot = model.pos.copy(), vtf.SO3(model.wxyz)
        if name == "Upright":
            rot = upright(rot)
        elif name == "Turn 90°":
            rot = vtf.SO3.from_z_radians(np.pi / 2) @ rot
        elif name == "Tip 90°":
            rot = vtf.SO3.from_x_radians(np.pi / 2) @ rot
        # Rotate about the model's centre, not the bottom origin, then put it on the table.
        centre = model.mesh.bounds.mean(0)
        pos = pos + vtf.SO3(model.wxyz).as_matrix() @ centre - rot.as_matrix() @ centre
        pos[2] -= self._min_z(model, pos, rot.wxyz)
        self.set_pose(model, pos, rot.wxyz)

    def reset_layout(self) -> None:
        if self.playing:
            return
        for model in self.models.values():
            self.set_pose(model, model.home_pos, model.home_wxyz)
        self._apply_gizmos()
        self.station.value = False  # the map is in the robot base frame, not in this layout
        self._apply_visibility()
        self._set_view(CHECK_VIEW)

    # ----------------------------------------------------------------- measure
    def _apply_measure_mode(self) -> None:
        scene = self.server.scene
        scene.remove_click_callback("all")
        if self.measure.value:
            scene.on_click()(self._on_scene_click)

    def _on_scene_click(self, event) -> None:
        origin = np.asarray(event.ray_origin, float)
        direction = np.asarray(event.ray_direction, float)
        direction /= np.linalg.norm(direction)
        best, owner = None, None
        with self.lock:
            for model in self.models.values():
                t = ray_triangles(origin, direction, model.world_triangles())
                if t is not None and (best is None or t < best):
                    best, owner = t, model
            if direction[2] < -1e-9:
                t = -origin[2] / direction[2]
                if t > 0 and (best is None or t < best):
                    best, owner = t, None
            if best is None:
                return
            point = origin + best * direction
            if owner is not None and self.snap.value:
                world = owner.world_vertices()
                d = np.linalg.norm(world - point, axis=1)
                if d.min() <= SNAP_DISTANCE:
                    point = world[int(d.argmin())]
            if len(self.points) >= 2:
                self.points = []
            self.points.append(point)
            self._draw_measurement()

    def _draw_measurement(self) -> None:
        scene = self.server.scene
        for name in ("/measure/p0", "/measure/p1", "/measure/line", "/measure/label"):
            scene.remove_by_name(name)
        for i, p in enumerate(self.points):
            scene.add_icosphere(
                f"/measure/p{i}", radius=0.0012, color=(255, 230, 40), position=p, cast_shadow=False
            )
        if len(self.points) < 2:
            if self.points:
                x, y, z = self.points[0] * 1000
                self.measure_text.content = f"Point 1: {x:.1f}, {y:.1f}, {z:.1f} mm. Click point 2."
            else:
                self.measure_text.content = "Turn on measure mode, then click two points."
            return
        a, b = self.points
        d = b - a
        dist = float(np.linalg.norm(d)) * 1000
        scene.add_line_segments(
            "/measure/line",
            np.array([[a, b]], dtype=np.float32),
            (255, 230, 40),
            thickness=2.5,
            thickness_units="screen",
        )
        scene.add_label(
            "/measure/label", f"{dist:.1f} mm", position=(a + b) / 2, anchor="bottom-center"
        )
        dx, dy, dz = np.abs(d) * 1000
        self.measure_text.content = (
            f"## {dist:.1f} mm\n|dx| {dx:.1f} · |dy| {dy:.1f} · |dz| {dz:.1f} mm\n\n"
            "Click again to start a new measurement."
        )

    def _clear_measurement(self) -> None:
        with self.lock:
            self.points = []
            self._draw_measurement()

    # --------------------------------------------------------------- drop test
    def _start_drop(self) -> None:
        with self.lock:
            if self.playing:
                return
            self.playing = True
        self.drop_button.disabled = True
        self._apply_gizmos()
        threading.Thread(target=self._run_drop, daemon=True).start()

    def simulate_drop(self, height: float, seconds: float = 2.5, fps: int = 30):
        """Return per-frame mesh-frame poses (plate frame) for each brick, and final stats."""
        model = self.drop_model
        data = mujoco.MjData(model)
        floor = float(
            self.models[PLATE]
            .mesh.vertices[:, 2][
                np.linalg.norm(self.models[PLATE].mesh.vertices[:, :2], axis=1) < 0.03
            ]
            .max()
        )
        offsets = {}
        for (key, b), (x, y) in zip(BRICKS.items(), DROP_LAYOUT):
            joint = model.joint(f"{key}-joint")
            data.qpos[joint.qposadr[0] : joint.qposadr[0] + 7] = [
                x,
                y,
                floor + height + b.grip_z,
                1,
                0,
                0,
                0,
            ]
            offsets[key] = body_to_mesh_frame(model, model.geom(f"{key}-solid").id)
        mujoco.mj_forward(model, data)
        steps = max(1, round(1 / fps / model.opt.timestep))
        frames = []
        for i in range(round(seconds * fps) + 1):
            if i:
                mujoco.mj_step(model, data, nstep=steps)
            frame = {}
            for key in BRICKS:
                body = model.body(key).id
                rot = data.xmat[body].reshape(3, 3)
                r_off, t_off = offsets[key]
                frame[key] = (data.xpos[body] + rot @ t_off, rot @ r_off)
            frames.append(frame)
        speeds = {}
        for key in BRICKS:
            v = model.joint(f"{key}-joint").dofadr[0]
            speeds[key] = float(np.linalg.norm(data.qvel[v : v + 3]))
        return frames, speeds

    def _run_drop(self) -> None:
        try:
            plate = self.models[PLATE]
            yaw = vtf.SO3(plate.wxyz).as_rpy_radians().yaw
            plate_rot = vtf.SO3.from_z_radians(yaw)
            plate_pos = np.array([plate.pos[0], plate.pos[1], 0.0])
            self.set_pose(plate, plate_pos, plate_rot.wxyz)
            self.drop_text.content = "Simulating…"
            frames, speeds = self.simulate_drop(self.drop_height.value / 1000)
            R = plate_rot.as_matrix()
            for frame in frames:
                start = time.monotonic()
                for key, (pos, rot) in frame.items():
                    world_rot = vtf.SO3.from_matrix(R @ rot)
                    self.set_pose(self.models[key], plate_pos + R @ pos, world_rot.wxyz)
                time.sleep(max(0.0, 1 / 30 - (time.monotonic() - start)))
            self.drop_text.content = self._drop_report(frames[-1], speeds)
        except Exception as exc:  # show the error in the panel, keep the viewer alive
            self.drop_text.content = f"**Drop test failed:** {exc}"
            raise
        finally:
            self.playing = False
            self.drop_button.disabled = False
            self._apply_gizmos()

    def _drop_report(self, final: dict, speeds: dict) -> str:
        rows = [
            "| Brick | From centre | Bottom z | Tilt | On floor |",
            "|---|---|---|---|---|",
        ]
        for key, (pos, rot) in final.items():
            mesh = self.models[key].mesh
            world = mesh.vertices @ rot.T + pos
            centre = (world.min(0) + world.max(0)) / 2
            radial = float(np.linalg.norm(centre[:2]))
            bottom = float(world[:, 2].min())
            tilt = float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))
            on_floor = (
                radial + 0.5 * float(np.ptp(world[:, :2], axis=0).min()) < FLOOR_RADIUS + 0.002
            )
            on_floor = on_floor and bottom < 0.006 and tilt < 5
            rows.append(
                f"| {BRICKS[key].part} | {radial * 1000:.1f} mm | {bottom * 1000:.1f} mm | "
                f"{tilt:.1f}° | {'yes' if on_floor else '**no**'} |"
            )
        still = max(speeds.values())
        return (
            "\n".join(rows)
            + f"\n\nMax speed at the end: {still * 1000:.1f} mm/s. 'On floor' = flat, low, and "
            "inside the Ø158 mm floor. Positions are relative to the plate centre."
        )

    def run(self) -> None:
        print(f"MODEL_VIEWER_READY http://127.0.0.1:{self.server.get_port()}/", flush=True)
        try:
            while True:
                time.sleep(1)
        finally:
            self.server.stop()
            self.tmp.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=19092)
    parser.add_argument(
        "--scene-config",
        type=Path,
        help="YAML with cameras and colours (default: configs/sim-scene.yaml)",
    )
    parser.add_argument(
        "--station",
        action="store_true",
        help="start in the station layout: cameras, arm base, plates and bricks at sim positions",
    )
    args = parser.parse_args()
    try:
        ModelViewer(args.port, args.scene_config, args.station).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
