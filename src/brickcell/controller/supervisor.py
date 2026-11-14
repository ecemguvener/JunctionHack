from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Protocol

from brickcell.controller.robot import Action, ActionStatus, RobotInterface
from brickcell.inspection.types import Decision, InspectionResult
from brickcell.metrics.report import EventLog, latency
from brickcell.sources.directory import Frame, FrameSource


class Inspector(Protocol):
    mode: str

    def inspect(self, frame: Frame) -> InspectionResult: ...


class State(StrEnum):
    WAITING = "WAITING"
    PICKING = "PICKING"
    PRESENTING = "PRESENTING"
    INSPECTING = "INSPECTING"
    REINSPECTING = "REINSPECTING"
    SORTING = "SORTING"
    VERIFYING = "VERIFYING"
    COMPLETE = "COMPLETE"
    RECOVERING = "RECOVERING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ControllerConfig:
    action_timeout_s: float = 5.0
    poll_interval_s: float = 0.01
    max_pick_retries: int = 1
    max_reinspections: int = 2
    max_release_retries: int = 1
    inspection_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        import math

        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.action_timeout_s, self.poll_interval_s, self.inspection_timeout_s)
        ):
            raise ValueError("Timeouts and polling interval must be positive and finite")
        if any(
            type(n) is not int or n < 0
            for n in (self.max_pick_retries, self.max_reinspections, self.max_release_retries)
        ):
            raise ValueError("Retry counts must be nonnegative integers")


@dataclass
class Cycle:
    object_id: str
    complete: bool = False
    decision: Decision | None = None
    reinspections: int = 0
    timeouts: int = 0
    error: str | None = None
    latency_ms: float = 0.0
    states: list[State] = field(default_factory=list)


class Supervisor:
    def __init__(
        self,
        robot: RobotInterface,
        source: FrameSource,
        inspector: Inspector,
        config: ControllerConfig | None = None,
        log: EventLog | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.robot, self.source, self.inspector = robot, source, inspector
        self.config = config or ControllerConfig()
        self.log, self.clock, self.sleep = log, clock, sleep
        self.used_frames: set[str] = set()

    def emit(self, cycle: Cycle, event: str, **fields) -> None:
        if self.log:
            self.log.emit(
                {
                    "object_id": cycle.object_id,
                    "event": event,
                    "robot_mode": self.robot.mode,
                    "inspector_mode": self.inspector.mode,
                    "monotonic_s": self.clock(),
                    **fields,
                }
            )

    def transition(self, cycle: Cycle, state: State) -> None:
        cycle.states.append(state)
        self.emit(cycle, "state", state=state)

    def wait_action(self, cycle: Cycle, action: Action) -> bool:
        if action.object_id != cycle.object_id:
            raise ValueError("Stale robot action token")
        self.emit(cycle, "action", detail=action.operation, action_id=action.action_id)
        deadline = self.clock() + self.config.action_timeout_s
        while True:
            status = self.robot.action_status(action)
            if self.clock() >= deadline:
                cycle.timeouts += 1
                raise TimeoutError(f"{action.operation} timed out")
            if status == ActionStatus.SUCCEEDED:
                return True
            if status == ActionStatus.FAILED:
                return False
            if status != ActionStatus.PENDING:
                raise ValueError(f"Invalid action status: {status}")
            self.sleep(min(self.config.poll_interval_s, max(0, deadline - self.clock())))

    def retry_action(
        self, cycle: Cycle, start: Callable[[str], Action], retries: int, state: State
    ) -> None:
        for attempt in range(retries + 1):
            self.transition(cycle, state)
            if self.wait_action(cycle, start(cycle.object_id)):
                return
            self.transition(cycle, State.RECOVERING)
            self.emit(cycle, "retry", detail=f"{state}: failed attempt {attempt + 1}")
        raise RuntimeError(f"{state}: retry limit reached")

    def inspect_with_deadline(self, cycle: Cycle, frame: Frame) -> InspectionResult:
        # An inspector never receives the robot. A late worker can only put an
        # unused result in its private queue; it cannot issue a sorting command.
        result_queue: queue.Queue = queue.Queue(maxsize=1)

        def work() -> None:
            try:
                result_queue.put((self.inspector.inspect(frame), None))
            except Exception as exc:
                result_queue.put((None, exc))

        start = self.clock()
        threading.Thread(target=work, daemon=True, name="brickcell-inspect").start()
        try:
            result, error = result_queue.get(timeout=self.config.inspection_timeout_s)
        except queue.Empty as exc:
            cycle.timeouts += 1
            raise TimeoutError("Inspection exceeded deadline; late result discarded") from exc
        if self.clock() - start >= self.config.inspection_timeout_s:
            cycle.timeouts += 1
            raise TimeoutError("Inspection exceeded deadline; result discarded")
        if error:
            raise error
        return result

    def cycle(self, object_id: str) -> Cycle:
        cycle = Cycle(object_id)
        began = self.clock()
        try:
            self.transition(cycle, State.WAITING)
            self.retry_action(
                cycle, self.robot.start_pick, self.config.max_pick_retries, State.PICKING
            )
            self.transition(cycle, State.PRESENTING)
            if not self.wait_action(cycle, self.robot.move_to_inspection(object_id)):
                raise RuntimeError("Inspection movement failed")
            result = None
            for attempt in range(self.config.max_reinspections + 1):
                self.transition(cycle, State.INSPECTING)
                frame = self.source.capture(object_id, attempt)
                if frame is not None:
                    if frame.object_id != object_id or frame.frame_id in self.used_frames:
                        raise ValueError("Stale or mismatched camera frame")
                    self.used_frames.add(frame.frame_id)
                    result = self.inspect_with_deadline(cycle, frame)
                    if (result.object_id, result.frame_id, result.view_id, result.timestamp) != (
                        object_id,
                        frame.frame_id,
                        frame.view_id,
                        frame.timestamp,
                    ):
                        raise ValueError("Stale or mismatched inspection result")
                    self.emit(cycle, "inspection", detail=result.decision, result=result)
                    if result.decision == Decision.ERROR:
                        raise RuntimeError(f"Inspection error: {result.reason}")
                    if result.is_valid_view and result.decision in (
                        Decision.GOOD,
                        Decision.DEFECTIVE,
                    ):
                        break
                if attempt == self.config.max_reinspections:
                    raise RuntimeError("Inspection retry limit reached; object left unsorted")
                cycle.reinspections += 1
                self.transition(cycle, State.REINSPECTING)
                if not self.wait_action(cycle, self.robot.adjust_view(object_id)):
                    raise RuntimeError("View adjustment failed")
            if result is None:
                raise RuntimeError("No inspection result")
            cycle.decision = result.decision
            self.transition(cycle, State.SORTING)
            move = (
                self.robot.move_to_good
                if result.decision == Decision.GOOD
                else self.robot.move_to_defective
            )
            if not self.wait_action(cycle, move(object_id)):
                raise RuntimeError("Sorting movement failed")
            self.retry_action(
                cycle, self.robot.release, self.config.max_release_retries, State.VERIFYING
            )
            if not self.wait_action(cycle, self.robot.verify_release(object_id)):
                raise RuntimeError("Release verification failed")
            cycle.complete = True
            self.transition(cycle, State.COMPLETE)
        except Exception as exc:
            cycle.error = str(exc)
            try:
                self.robot.stop_safely()
            except Exception as stop_error:
                cycle.error += f"; safe stop also failed: {stop_error}"
            cycle.states.append(State.ERROR)
            try:
                self.emit(cycle, "state", state=State.ERROR)
                self.emit(cycle, "error", detail=cycle.error)
            except Exception as log_error:
                cycle.error += f"; logging also failed: {log_error}"
        finally:
            cycle.latency_ms = (self.clock() - began) * 1000
        return cycle

    def run(self, objects: list[str]) -> dict:
        if len(objects) != len(set(objects)):
            raise ValueError("Object IDs must be unique within a run")
        cycles = []
        try:
            self.robot.connect()
            for obj in objects:
                cycle = self.cycle(obj)
                cycles.append(cycle)
                if cycle.error:
                    break  # A safe stop requires human review, never silently resume.
        except Exception:
            self.robot.stop_safely()
            raise
        finally:
            self.robot.disconnect()
        return {
            "robot_mode": self.robot.mode,
            "inspector_mode": self.inspector.mode,
            "physical_success_verified": False,
            "total_requested": len(objects),
            "total_attempted": len(cycles),
            "completed_cycles": sum(c.complete for c in cycles),
            "completed_mock_cycles": sum(c.complete for c in cycles)
            if self.robot.mode == "mock"
            else 0,
            "reinspections": sum(c.reinspections for c in cycles),
            "timeouts": sum(c.timeouts for c in cycles),
            "errors": sum(c.error is not None for c in cycles),
            "classifications": dict(Counter(c.decision.value for c in cycles if c.decision)),
            "completed_sort_actions": dict(Counter(c.decision.value for c in cycles if c.complete)),
            "cycle_latency": latency([c.latency_ms for c in cycles]),
            "cycles": cycles,
        }
