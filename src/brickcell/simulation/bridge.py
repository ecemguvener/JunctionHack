"""Core-side adapter: standard-library HTTP, no MuJoCo dependency or fixture labels."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from brickcell.controller.robot import Action, ActionStatus
from brickcell.inspection.types import InspectionResult
from brickcell.sources.directory import Frame
from brickcell.util import json_default


class LocalSimulation:
    def __init__(self, url: str = "http://127.0.0.1:18081"):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Simulation bridge accepts only http://127.0.0.1:<port>")
        self.url = url.rstrip("/")

    def call(self, path: str, payload: dict | None = None) -> dict:
        body = (
            json.dumps(payload, default=json_default, allow_nan=False).encode()
            if payload is not None
            else None
        )
        request = Request(self.url + path, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result


class SimRobot:
    mode = "mujoco_contact_simulation"

    def __init__(self, connection: LocalSimulation):
        self.connection = connection
        self.connected = False
        self.actions: dict[str, Action] = {}

    def connect(self) -> None:
        if self.connection.call("/health")["mode"] != self.mode:
            raise ValueError("Wrong simulation service")
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def _start(self, operation: str, object_id: str) -> Action:
        if not self.connected:
            raise RuntimeError("Simulation robot is disconnected")
        result = self.connection.call("/action", {"operation": operation, "object_id": object_id})
        if result["object_id"] != object_id or result["operation"] != operation:
            raise ValueError("Simulation returned a mismatched action")
        action = Action(result["action_id"], object_id, operation)
        self.actions[action.action_id] = action
        return action

    def start_pick(self, object_id: str) -> Action:
        return self._start("pick", object_id)

    def move_to_inspection(self, object_id: str) -> Action:
        return self._start("present", object_id)

    def adjust_view(self, object_id: str) -> Action:
        return self._start("adjust_view", object_id)

    def move_to_good(self, object_id: str) -> Action:
        return self._start("move_good", object_id)

    def move_to_defective(self, object_id: str) -> Action:
        return self._start("move_defective", object_id)

    def release(self, object_id: str) -> Action:
        return self._start("release", object_id)

    def verify_release(self, object_id: str) -> Action:
        return self._start("verify_release", object_id)

    def action_status(self, action: Action) -> ActionStatus:
        if self.actions.get(action.action_id) != action:
            raise ValueError("Unknown action token")
        result = self.connection.call(f"/actions/{action.action_id}")
        if any(result[k] != getattr(action, k) for k in ("action_id", "object_id", "operation")):
            raise ValueError("Stale/mismatched action status")
        if result.get("error"):
            raise RuntimeError(result["error"])
        return ActionStatus(result["status"])

    def stop_safely(self) -> None:
        self.connection.call("/stop", {})


class SimCamera:
    def __init__(self, connection: LocalSimulation, directory: Path):
        self.connection, self.directory = connection, directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def capture(self, object_id: str, attempt: int) -> Frame:
        result = self.connection.call("/capture", {"object_id": object_id, "attempt": attempt})
        if result["object_id"] != object_id:
            raise ValueError("Camera returned a different object's frame")
        name = hashlib.sha256(result["frame_id"].encode()).hexdigest()[:24]
        path = self.directory / f"{name}.png"
        path.write_bytes(base64.b64decode(result["png_base64"], validate=True))
        return Frame(object_id, result["frame_id"], result["timestamp"], result["view_id"], path)


class PublishedInspector:
    """Publish genuine model results for the viewer and simulator's action interlock."""

    def __init__(self, inspector, connection: LocalSimulation):
        self.inspector, self.connection = inspector, connection
        self.mode = inspector.mode

    def inspect(self, frame: Frame) -> InspectionResult:
        result = self.inspector.inspect(frame)
        self.connection.call("/inspection", asdict(result))
        print(
            f"{result.object_id}: {result.decision} | raw score={result.anomaly_score} | OpenVINO CPU {result.inference_latency_ms:.1f} ms",
            flush=True,
        )
        return result
