"""Loopback-only bridge. All MuJoCo stepping/rendering stays on the main thread."""

from __future__ import annotations

import argparse
import base64
import io
import json
import queue
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

from brickcell.simulation.protocol import accept_inspection, authorize_sort
from brickcell.simulation.world import BOWLS, OBJECTS, World, inspection_crop
from brickcell.workbench.status import SIMULATION_STATUS, coverage_status


class SimulationServer:
    def __init__(self, api_port: int, viewer_port: int, directory: Path):
        self.world = World(directory)
        self.commands: queue.Queue = queue.Queue()
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
        self.viewer.gui.add_markdown("## See → Detect → Sort\nTwo bowls · SO-101 · MuJoCo")
        self.viewer.gui.add_markdown(SIMULATION_STATUS)
        self.coverage_panel = self.viewer.gui.add_markdown(coverage_status(False))
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

        for name, pos in BOWLS.items():
            self.viewer.scene.add_label(
                f"/labels/{name}", name.upper(), position=tuple(pos + [0, 0, 0.02])
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
                raw = json.dumps(response, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.http = ThreadingHTTPServer(("127.0.0.1", api_port), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.last_tick = time.monotonic()
        self.tick_count = 0
        self.world.tick_callback = self.tick

    def tick(self):
        if self.stopped.is_set():
            raise RuntimeError("Simulation safely stopped; reset required")
        self.tick_count += 1
        if self.tick_count % 3 == 0:
            self.scene.update_from_mjdata(self.world.data)
        delay = 0.02 - (time.monotonic() - self.last_tick)
        if delay > 0:
            time.sleep(delay)
        self.last_tick = time.monotonic()

    def request(self, method: str, path: str, payload: dict) -> dict:
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
            # Defense in depth: exact fresh camera result must authorize the bowl.
            if op.startswith("move_"):
                authorize_sort(self.inspected, op, obj)
            if op == "pick":
                self.run_active = True
                self.inspected = self.last_capture = None
                self.coverage_panel.content = coverage_status(False)
                self.decision_panel.content = "New object: no inspection decision yet."
                self.camera_panel.image = np.zeros((130, 160, 3), dtype=np.uint8)
            self.panel.content = f"**{obj}** · {op.replace('_', ' ').upper()}"
            self.world.action(op, obj)
            self.count_panel.content = f"Good: {self.world.counts['good']} · Defective: {self.world.counts['defective']}\n\nBowl occupancy is checked in simulation; physical success remains unverified."
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
            self.coverage_panel.content = coverage_status(True)
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
            self.coverage_panel.content = coverage_status(False)
            self.decision_panel.content = "No inspection yet."
            self.camera_panel.image = np.zeros((130, 160, 3), dtype=np.uint8)
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
    args = parser.parse_args()
    try:
        SimulationServer(args.api_port, args.viewer_port, args.output).run()
    except KeyboardInterrupt:
        pass
