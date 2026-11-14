from dataclasses import replace
from pathlib import Path

import pytest

from brickcell.controller.robot import ActionStatus, MockRobot
from brickcell.controller.supervisor import ControllerConfig, State, Supervisor
from brickcell.inspection.service import MockInspector
from brickcell.inspection.types import Decision
from brickcell.sources.directory import DirectorySource


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, amount):
        self.now += amount


def make(decisions, outcomes=None, views=None, config=None):
    robot = MockRobot(outcomes)
    source = DirectorySource({"o": [Path("mock")] * (views or len(decisions))})
    clock = Clock()
    supervisor = Supervisor(
        robot, source, MockInspector({"o": decisions}), config, clock=clock, sleep=clock.sleep
    )
    return robot, supervisor


@pytest.mark.parametrize(
    "decision,operation", [(Decision.GOOD, "move_good"), (Decision.DEFECTIVE, "move_defective")]
)
def test_full_cycle(decision, operation):
    robot, supervisor = make([decision])
    report = supervisor.run(["o"])
    assert report["completed_mock_cycles"] == 1
    assert not report["physical_success_verified"]
    assert report["cycles"][0].states == [
        State.WAITING,
        State.PICKING,
        State.PRESENTING,
        State.INSPECTING,
        State.SORTING,
        State.VERIFYING,
        State.COMPLETE,
    ]
    assert operation in [a.operation for a in robot.actions]
    assert not robot.connected


def test_invalid_view_reinspection():
    robot, supervisor = make([Decision.REINSPECT, Decision.GOOD])
    report = supervisor.run(["o"])
    assert report["reinspections"] == 1
    assert report["completed_mock_cycles"] == 1
    assert "adjust_view" in [a.operation for a in robot.actions]


def test_retry_limit_and_missing_frames():
    robot, supervisor = make([Decision.REINSPECT] * 4)
    report = supervisor.run(["o"])
    assert report["reinspections"] == 2
    assert report["errors"] == 1
    assert robot.stopped
    assert not any(a.operation.startswith("move_") for a in robot.actions)
    robot, supervisor = make([Decision.GOOD])
    supervisor.source.frames = {}
    report = supervisor.run(["o"])
    assert report["errors"] == 1
    assert report["reinspections"] == 2


@pytest.mark.parametrize("operation", ["pick", "present", "move_good", "release"])
def test_action_timeout(operation):
    robot, supervisor = make([Decision.GOOD], {operation: [ActionStatus.PENDING]})
    report = supervisor.run(["o"])
    assert report["timeouts"] == 1
    assert report["errors"] == 1
    assert robot.stopped


def test_failed_grasp_recovery_and_limit():
    robot, supervisor = make(
        [Decision.GOOD], {"pick": [ActionStatus.FAILED, ActionStatus.SUCCEEDED]}
    )
    assert supervisor.run(["o"])["completed_mock_cycles"] == 1
    assert sum(a.operation == "pick" for a in robot.actions) == 2
    robot, supervisor = make([Decision.GOOD], {"pick": [ActionStatus.FAILED] * 4})
    assert supervisor.run(["o"])["errors"] == 1
    assert sum(a.operation == "pick" for a in robot.actions) == 2


@pytest.mark.parametrize("operation", ["release", "verify_release_mock_only"])
def test_failed_release(operation):
    robot, supervisor = make([Decision.GOOD], {operation: [ActionStatus.FAILED] * 4})
    report = supervisor.run(["o"])
    assert report["errors"] == 1
    assert report["completed_mock_cycles"] == 0
    assert robot.stopped


@pytest.mark.parametrize(
    "field,value",
    [
        ("object_id", "previous"),
        ("frame_id", "old-frame"),
        ("view_id", "wrong"),
        ("timestamp", "old"),
    ],
)
def test_stale_result_rejected(field, value):
    robot, supervisor = make([Decision.GOOD])

    class Stale(MockInspector):
        def inspect(self, frame):
            return replace(super().inspect(frame), **{field: value})

    supervisor.inspector = Stale({"o": [Decision.GOOD]})
    report = supervisor.run(["o"])
    assert "Stale" in report["cycles"][0].error
    assert not any(a.operation.startswith("move_") for a in robot.actions)


def test_exception_safe_stop_and_no_next_object():
    robot, supervisor = make([Decision.ERROR])
    report = supervisor.run(["o", "next"])
    assert report["total_attempted"] == 1
    assert robot.stopped


def test_robot_connection_and_actions():
    robot = MockRobot()
    with pytest.raises(RuntimeError):
        robot.start_pick("o")
    robot.connect()
    action = robot.start_pick("o")
    assert robot.action_status(action) == ActionStatus.SUCCEEDED
    robot.stop_safely()
    with pytest.raises(RuntimeError):
        robot.release("o")


def test_bad_limits():
    with pytest.raises(ValueError):
        ControllerConfig(action_timeout_s=0)
    with pytest.raises(ValueError):
        ControllerConfig(max_reinspections=-1)


def test_inspection_timeout_discards_result():
    robot, supervisor = make([Decision.GOOD])

    class Slow(MockInspector):
        def inspect(self, frame):
            supervisor.clock.sleep(31)
            return super().inspect(frame)

    supervisor.inspector = Slow({"o": [Decision.GOOD]})
    report = supervisor.run(["o"])
    assert report["timeouts"] == 1
    assert not any(a.operation.startswith("move_") for a in robot.actions)


def test_blocked_inspector_has_wall_clock_deadline():
    import threading
    import time

    event = threading.Event()
    robot, supervisor = make([Decision.GOOD], config=ControllerConfig(inspection_timeout_s=0.05))

    class Blocked(MockInspector):
        def inspect(self, frame):
            event.wait(2)
            return super().inspect(frame)

    supervisor.inspector = Blocked({"o": [Decision.GOOD]})
    start = time.monotonic()
    try:
        report = supervisor.run(["o"])
        assert time.monotonic() - start < 1
        assert report["timeouts"] == 1
        assert robot.stopped
        assert not any(a.operation.startswith("move_") for a in robot.actions)
    finally:
        event.set()


def test_log_failure_still_stops_robot():
    robot, supervisor = make([Decision.GOOD])

    class BrokenLog:
        def emit(self, event):
            raise OSError("disk full")

    supervisor.log = BrokenLog()
    report = supervisor.run(["o"])
    assert report["errors"] == 1
    assert robot.stopped
    assert "disk full" in report["cycles"][0].error
