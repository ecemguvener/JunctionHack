"""The bridge and safety contract run offline without MuJoCo installed."""

import base64
import io
import os
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from brickcell.controller.robot import Action, ActionStatus
from brickcell.simulation.bridge import LocalSimulation, SimCamera, SimRobot
from brickcell.simulation.protocol import accept_inspection, authorize_sort


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:18081",
        "http://example.com",
        "http://user@127.0.0.1",
        "http://127.0.0.1/path",
    ],
)
def test_bridge_rejects_nonlocal_services(url):
    with pytest.raises(ValueError):
        LocalSimulation(url)


@pytest.fixture
def capture():
    return dict(object_id="opaque-1", frame_id="frame-7", timestamp="now", view_id="inspection")


@pytest.fixture
def result(capture):
    return dict(
        **capture, decision="GOOD", is_valid_view=True, inspector_mode="anomalib_openvino_cpu"
    )


@pytest.mark.parametrize("field", ["object_id", "frame_id", "timestamp", "view_id"])
def test_stale_camera_result_rejected(capture, result, field):
    result[field] = "stale"
    with pytest.raises(ValueError, match="Stale"):
        accept_inspection(capture, result)


def test_invalid_mock_and_uncertain_results_cannot_sort(capture, result):
    result["inspector_mode"] = "MOCK_SCRIPTED_NOT_ML"
    with pytest.raises(ValueError):
        accept_inspection(capture, result)
    result["inspector_mode"] = "anomalib_openvino_cpu"
    result["is_valid_view"] = False
    assert accept_inspection(capture, result) is None
    result["is_valid_view"] = True
    for decision in ["REINSPECT", "ERROR"]:
        result["decision"] = decision
        for operation in ["move_good", "move_defective"]:
            with pytest.raises(ValueError):
                authorize_sort(result, operation, "opaque-1")


def test_sort_destination_follows_result_not_object_id(capture, result):
    authorize_sort(accept_inspection(capture, result), "move_good", "opaque-1")
    with pytest.raises(ValueError):
        authorize_sort(result, "move_defective", "opaque-1")
    result["decision"] = "DEFECTIVE"
    authorize_sort(accept_inspection(capture, result), "move_defective", "opaque-1")
    with pytest.raises(ValueError):
        authorize_sort(result, "move_good", "opaque-1")
    with pytest.raises(ValueError):
        authorize_sort(result, "move_defective", "different-object")


class Transport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call(self, path, payload=None):
        self.calls.append((path, payload))
        if path == "/health":
            return {"mode": "mujoco_contact_simulation"}
        return self.response


def test_camera_preserves_rgb_and_uses_safe_filename(tmp_path, capture):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 0, 12)).save(buffer, format="PNG")
    capture["frame_id"] = "../../do-not-use-as-path"
    transport = Transport(dict(**capture, png_base64=base64.b64encode(buffer.getvalue()).decode()))
    frame = SimCamera(transport, tmp_path).capture("opaque-1", 0)
    assert frame.path.parent == tmp_path
    assert Image.open(frame.path).getpixel((0, 0)) == (255, 0, 12)
    with pytest.raises(ValueError, match="different object"):
        SimCamera(transport, tmp_path).capture("opaque-2", 0)


def test_robot_checks_action_identity():
    transport = Transport(
        dict(action_id="1", object_id="opaque-1", operation="pick", status="PENDING")
    )
    robot = SimRobot(transport)
    with pytest.raises(RuntimeError):
        robot.start_pick("opaque-1")
    robot.connect()
    action = robot.start_pick("opaque-1")
    assert robot.action_status(action) == ActionStatus.PENDING
    transport.response["status"] = "SUCCEEDED"
    assert robot.action_status(action) == ActionStatus.SUCCEEDED
    transport.response["object_id"] = "old-object"
    with pytest.raises(ValueError, match="mismatched"):
        robot.action_status(action)
    with pytest.raises(ValueError, match="Unknown"):
        robot.action_status(Action("unknown", "opaque-1", "pick"))


@pytest.mark.simulation
def test_live_camera_openvino_contact_sorting():
    url = os.environ.get("BRICKCELL_SIMULATION_URL")
    if not url:
        pytest.skip(
            "Start two-bowl simulator and set BRICKCELL_SIMULATION_URL for live integration"
        )
    from brickcell.simulation.run import run

    report = run(
        Namespace(
            url=url,
            reset=True,
            model=Path("artifacts/models/patchcore-sim/model.xml"),
            quality_config=Path("configs/inspection-sim.yaml"),
            output=Path("artifacts/two-bowl/tests"),
        )
    )
    assert report["errors"] == report["timeouts"] == 0
    assert report["completed_cycles"] == 4
    assert report["completed_sort_actions"] == {"GOOD": 2, "DEFECTIVE": 2}
    assert report["inspector_mode"] == "anomalib_openvino_cpu"
    assert not report["physical_success_verified"]
