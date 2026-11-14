"""Loopback-only bridge. All MuJoCo stepping/rendering stays on the main thread.

Scene endpoints (cameras and colours, see docs/SIMULATION.md):
  GET  /scene                 cameras (pose, intrinsics) and brick / plate colours
  GET  /camera/<name>.png     full image of a sim camera, for viewing only
  POST /brick_color           {"object_id": "part-001", "color": "#ff0000"}
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import queue
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import viser
from mjviser import ViserMujocoScene
from PIL import Image

from brickcell.simulation.model_viewer import add_camera_frustums, rgb255
from brickcell.simulation.protocol import accept_inspection, authorize_sort
from brickcell.simulation.scene_config import (
    color_hex,
    load_scene_config,
    parse_brick_color_arg,
    parse_color,
)
from brickcell.simulation.world import BOWLS, OBJECTS, World, inspection_crop

CAMERA_PATH = re.compile(r"/camera/([A-Za-z0-9_][A-Za-z0-9_-]*)\.png")
THUMBNAIL_WIDTH = 240  # pixels; the GUI panel is 250 wide
THUMBNAIL_TICKS = 10  # one thumbnail for each 10 ticks of 20 ms: 5 images per second
COLOR_DELAY = 0.25  # s; a drag in the colour picker makes many values, apply only the last


def thumbnail(rgb: np.ndarray) -> np.ndarray:
    height = max(1, round(rgb.shape[0] * THUMBNAIL_WIDTH / rgb.shape[1]))
    return np.asarray(Image.fromarray(rgb).resize((THUMBNAIL_WIDTH, height), Image.BILINEAR))


class SimulationServer:
    def __init__(
        self, api_port: int, viewer_port: int, directory: Path, scene_config: object = None
    ):
        self.world = World(directory, scene_config)
        self.commands: queue.Queue = queue.Queue()
        # Scene requests (camera images, colours). The main thread serves them between physics
        # steps, also during an action. They do not change the inspection / sort state.
        self.calls: queue.Queue = queue.Queue()
        self.pending_colors: dict[str, tuple[float, float, float]] = {}
        self.color_time = 0.0
        self.statuses: dict[str, dict] = {}
        self.busy = False
        self.run_active = False
        self.stopped = threading.Event()
        self.run_id = uuid.uuid4().hex
        self.sequence = 0
        self.last_capture: dict | None = None
        self.inspected: dict | None = None
        self.process = None
        self.lock = threading.Lock()
        self.viewer = viser.ViserServer(host="127.0.0.1", port=viewer_port)
        if self.viewer.get_port() != viewer_port:
            raise RuntimeError("Viewer port is already occupied")
        self.scene = ViserMujocoScene(self.viewer, self.world.model, num_envs=1)
        self.scene.camera_tracking_enabled = False
        self.scene.set_refresh_handler(lambda: None)
        self.viewer.gui.configure_theme(dark_mode=True, show_share_button=False)
        self.viewer.gui.main_panel.dock_right()
        self.viewer.gui.main_panel.set_width(250)
        self.viewer.gui.add_markdown("## See → Detect → Sort\nTwo plates · SO-101 · MuJoCo")
        self.panel = self.viewer.gui.add_markdown(
            "Ready. Start the camera-driven inspection loop below."
        )
        self.camera_panel = self.viewer.gui.add_image(
            np.zeros((130, 160, 3), dtype=np.uint8), label="Last inspection frame → PatchCore"
        )
        self.decision_panel = self.viewer.gui.add_markdown(
            "No inspection yet. Decisions come from Anomalib / OpenVINO CPU."
        )
        self.count_panel = self.viewer.gui.add_markdown("Good: 0 · Defective: 0")
        self.viewer.gui.add_markdown(
            "**Simulation only.** Scripted IK uses known pickup poses. Camera pixels determine sorting. No learned VLA or real hardware."
        )
        button = self.viewer.gui.add_button("Run / replay full inspection loop")

        @button.on_click
        def replay(_):
            with self.lock:
                if (
                    self.busy
                    or self.run_active
                    or (self.process is not None and self.process.poll() is None)
                ):
                    return
                output = directory / "viewer-replay.log"
                with output.open("a") as log:
                    self.process = subprocess.Popen(
                        [
                            # Keep the venv symlink: resolving it selects the bare uv Python.
                            str(Path.cwd() / ".venv/bin/python"),
                            "-m",
                            "brickcell.simulation.run",
                            "--url",
                            f"http://127.0.0.1:{api_port}",
                            "--reset",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )

        gui = self.viewer.gui
        with gui.add_folder("Sim camera"):
            names = self.world.camera_names()
            self.view_select = gui.add_dropdown(
                "Camera",
                ("Off", *names),
                initial_value="overview" if "overview" in names else "Off",
                hint="Live image of a simulation camera, to compare with the real stream.",
            )
            self.view_panel = gui.add_image(
                np.zeros((THUMBNAIL_WIDTH * 3 // 4, THUMBNAIL_WIDTH, 3), dtype=np.uint8),
                label="Live view · not a detector input",
            )
            self.view_text = gui.add_markdown("")
            frustums = gui.add_checkbox("Show cameras in the 3D view", True)
            handles = add_camera_frustums(self.viewer.scene, self.world.cameras())

            @frustums.on_update
            def show_frustums(_):
                for handle in handles:
                    handle.visible = frustums.value

        with gui.add_folder("Brick colours"):
            self.color_inputs = {}
            for part in OBJECTS:
                picker = gui.add_rgb(part, rgb255(self.world.bricks[part].rgba))
                picker.on_update(self.color_callback(part))
                self.color_inputs[part] = picker
            gui.add_markdown(
                "The change is for this run only. For a permanent change, edit "
                "`configs/sim-scene.yaml`."
            )

        for name, pos in BOWLS.items():
            self.viewer.scene.add_label(
                f"/labels/{name}", name.upper(), position=(pos[0], pos[1], 0.06)
            )
        self.viewer.scene.add_label("/labels/inspection", "INSPECTION", position=(0.25, 0, 0.23))

        @self.viewer.on_client_connect
        def configure_camera(client):
            client.camera.position = (0.65, -0.60, 0.58)
            client.camera.look_at = (0.16, 0, -0.10)
            client.camera.up_direction = (0, 0, 1)

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.handle_request("GET")

            def do_POST(self):
                self.handle_request("POST")

            def handle_request(self, method):
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size > 65536:
                        raise ValueError("Request too large")
                    payload = json.loads(self.rfile.read(size)) if size else {}
                    response = owner.request(method, self.path, payload)
                    status = 200
                except Exception as exc:
                    response, status = {"error": str(exc)}, 400
                if isinstance(response, bytes):
                    raw, kind = response, "image/png"
                else:
                    raw, kind = json.dumps(response, allow_nan=False).encode(), "application/json"
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.http = ThreadingHTTPServer(("127.0.0.1", api_port), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.last_tick = time.monotonic()
        self.tick_count = 0
        self.world.tick_callback = self.tick

    def color_callback(self, part: str):
        def callback(event) -> None:
            rgb = tuple(v / 255 for v in event.target.value)
            if color_hex(rgb) != color_hex(self.world.bricks[part].rgba):
                with self.lock:
                    self.pending_colors[part] = rgb  # the main thread applies it

        return callback

    def apply_brick_colors(self, colors: dict, update_pickers: bool = True) -> None:
        """Main thread only. Change the model colours and show them in the browser viewer."""
        for part, rgb in colors.items():
            self.world.set_brick_color(part, rgb)
        # The viewer bakes geom colours into its meshes, so make the meshes again.
        self.scene.rebuild_visual_handles()
        self.scene.update_from_mjdata(self.world.data)
        if update_pickers:  # not for a change that came from a picker: the user can be dragging
            for part in colors:
                self.color_inputs[part].value = rgb255(self.world.bricks[part].rgba)

    def scene_state(self) -> dict:
        return {
            "config": self.world.scene.source,
            "inference_camera": "inspection",
            "cameras": self.world.cameras(),
            "bricks": {
                part: {"color": color_hex(b.rgba), "rgb": list(b.rgba)}
                for part, b in self.world.bricks.items()
            },
            "plates": self.world.scene.as_dict()["plates"],
        }

    def camera_png(self, name: str) -> bytes:
        buffer = io.BytesIO()
        Image.fromarray(self.world.camera(name)).save(buffer, format="PNG")
        return buffer.getvalue()

    def call_on_main(self, function):
        """Run a scene request on the main thread and wait for its result."""
        future = Future()
        self.calls.put((function, future))
        return future.result(timeout=5)

    def serve_scene(self) -> None:
        """Main thread only: scene requests, colour changes from the GUI, live camera view."""
        while True:
            try:
                function, future = self.calls.get_nowait()
            except queue.Empty:
                break
            try:
                future.set_result(function())
            except Exception as exc:
                future.set_exception(exc)
        try:
            if self.pending_colors and time.monotonic() - self.color_time > COLOR_DELAY:
                with self.lock:
                    colors, self.pending_colors = self.pending_colors, {}
                self.apply_brick_colors(colors, update_pickers=False)
                self.color_time = time.monotonic()
            name = self.view_select.value
            show = self.tick_count % THUMBNAIL_TICKS == 0 and self.viewer.get_clients()
            if name != "Off" and show:
                rgb = self.world.camera(name)
                self.view_panel.image = thumbnail(rgb)
                self.view_text.content = f"`{name}` · {rgb.shape[1]} × {rgb.shape[0]} px"
        except Exception as exc:  # a problem in the viewer panel must not stop a robot action
            self.view_select.value = "Off"
            self.view_text.content = f"**Viewer panel error:** {exc}"

    def tick(self):
        if self.stopped.is_set():
            raise RuntimeError("Simulation safely stopped; reset required")
        self.tick_count += 1
        if self.tick_count % 3 == 0:
            self.scene.update_from_mjdata(self.world.data)
        self.serve_scene()
        delay = 0.02 - (time.monotonic() - self.last_tick)
        if delay > 0:
            time.sleep(delay)
        self.last_tick = time.monotonic()

    def request(self, method: str, path: str, payload: dict) -> dict | bytes:
        if method == "GET" and path == "/scene":
            return self.call_on_main(self.scene_state)
        if method == "GET" and (match := CAMERA_PATH.fullmatch(path)):
            # For viewing only: no frame identity, so it cannot go into /inspection.
            return self.call_on_main(lambda: self.camera_png(match.group(1)))
        if method == "POST" and path == "/brick_color":
            part = payload["object_id"]
            if part not in OBJECTS:
                raise ValueError(f"Unknown fixture object. Known: {OBJECTS}")
            rgb = parse_color(payload["color"], f"brick colour {part}")
            self.call_on_main(lambda: self.apply_brick_colors({part: rgb}))
            return {"object_id": part, "color": color_hex(rgb), "rgb": list(rgb)}
        if method == "GET" and path == "/health":
            return {
                "mode": "mujoco_contact_simulation",
                "objects": OBJECTS,
                "run_id": self.run_id,
                "cameras": [self.world.model.camera(i).name for i in range(self.world.model.ncam)],
                "inference_camera": "inspection",
                "busy": self.busy,
                "stopped": self.stopped.is_set(),
            }
        if method == "GET" and path.startswith("/actions/"):
            return self.statuses[path.rsplit("/", 1)[1]].copy()
        if method == "POST" and path == "/stop":
            self.stopped.set()
            return {"stopped": True}
        if method != "POST" or path not in {
            "/action",
            "/capture",
            "/inspection",
            "/reset",
            "/summary",
        }:
            raise ValueError("Unknown endpoint")
        if path == "/action":
            with self.lock:
                if self.busy or self.stopped.is_set():
                    raise ValueError("Simulation busy or stopped")
                self.busy = True
                token = uuid.uuid4().hex
                action = {
                    "action_id": token,
                    "object_id": payload["object_id"],
                    "operation": payload["operation"],
                    "status": "PENDING",
                }
                self.statuses[token] = action
                self.commands.put((path, action, None))
            return action.copy()
        future = Future()
        self.commands.put((path, payload, future))
        return future.result(timeout=5)

    def execute(self, path: str, payload: dict) -> dict:
        if path == "/action":
            op, obj = payload["operation"], payload["object_id"]
            # Defense in depth: exact fresh camera result must authorize the plate.
            if op.startswith("move_"):
                authorize_sort(self.inspected, op, obj)
            if op == "pick":
                self.run_active = True
                self.inspected = self.last_capture = None
            self.panel.content = f"**{obj}** · {op.replace('_', ' ').upper()}"
            self.world.action(op, obj)
            self.count_panel.content = f"Good: {self.world.counts['good']} · Defective: {self.world.counts['defective']}\n\nPlate occupancy is checked in simulation; physical success remains unverified."
            return {"position": self.world.position(obj).tolist()}
        if path == "/capture":
            self.inspected = None
            obj = payload["object_id"]
            if self.world.active != obj or self.world.position(obj)[2] < 0.10:
                raise ValueError("Object is not held at the inspection camera")
            if np.linalg.norm(self.world.ik.tcp(self.world.data) - [0.25, 0, 0.15]) > 0.02:
                raise ValueError("Arm is outside inspection pose")
            rgb = inspection_crop(self.world.camera())
            self.camera_panel.image = rgb
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            self.sequence += 1
            self.last_capture = {
                "object_id": obj,
                "frame_id": f"{self.run_id}-{self.sequence}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "view_id": "inspection-fixed-roi",
            }
            self.panel.content = f"**{obj}** · INSPECTING camera pixels with OpenVINO CPU"
            return {**self.last_capture, "png_base64": base64.b64encode(buffer.getvalue()).decode()}
        if path == "/inspection":
            self.inspected = accept_inspection(self.last_capture, payload)
            score = payload["anomaly_score"]
            self.decision_panel.content = f"## {payload['decision']}\nObject: {payload['object_id']}\n\nRaw anomaly score: {score}\n\nModel: `{payload['model_version']}`\n\nOpenVINO CPU: {payload['inference_latency_ms']:.1f} ms\n\n{payload.get('reason', '')}"
            return {"accepted": True}
        if path == "/reset":
            if self.busy or (self.run_active and not self.stopped.is_set()):
                raise ValueError("Cannot reset an active run")
            self.stopped.clear()
            self.run_active = False
            self.world.reset()
            self.inspected = self.last_capture = None
            self.run_id = uuid.uuid4().hex
            self.panel.content = "Reset · ready for four camera-driven inspections"
            self.count_panel.content = "Good: 0 · Defective: 0"
            return {"run_id": self.run_id}
        self.run_active = False
        self.panel.content = f"**Run complete** · {payload['completed_cycles']}/{payload['total_attempted']} cycles\n\nReplay to watch fresh camera inference and contact physics."
        return {"accepted": True}

    def run(self):
        print("TWO_BOWL_READY", flush=True)
        try:
            while True:
                try:
                    path, payload, future = self.commands.get_nowait()
                except queue.Empty:
                    if self.stopped.is_set():
                        self.tick_count += 1
                        self.serve_scene()
                        time.sleep(0.02)
                    else:
                        try:
                            self.world.tick()
                        except RuntimeError:
                            if not self.stopped.is_set():
                                raise
                    continue
                try:
                    result = self.execute(path, payload)
                    if future is not None:
                        future.set_result(result)
                    else:
                        self.statuses[payload["action_id"]].update(status="SUCCEEDED", **result)
                except Exception as exc:
                    if future is not None:
                        future.set_exception(exc)
                    else:
                        self.statuses[payload["action_id"]].update(status="FAILED", error=str(exc))
                        self.panel.content = f"**Action failed**: {exc}"
                finally:
                    if path == "/action":
                        self.busy = False
        finally:
            self.http.shutdown()
            self.world.close()
            self.viewer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=18081)
    parser.add_argument("--viewer-port", type=int, default=19091)
    parser.add_argument("--output", type=Path, default=Path("artifacts/two-bowl"))
    parser.add_argument(
        "--scene-config",
        type=Path,
        help="YAML with cameras and colours (default: configs/sim-scene.yaml)",
    )
    parser.add_argument(
        "--brick-color",
        action="append",
        default=[],
        metavar="PART=COLOUR",
        help="brick colour for this run, e.g. part-001=#ff0000; you can give it more than once",
    )
    args = parser.parse_args()
    try:
        scene = load_scene_config(args.scene_config).with_brick_colors(
            dict(parse_brick_color_arg(value) for value in args.brick_color)
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        SimulationServer(args.api_port, args.viewer_port, args.output, scene).run()
    except KeyboardInterrupt:
        pass
