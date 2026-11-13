"""Nonblocking robot actions; the supervisor owns deadlines and recovery decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ActionStatus(StrEnum):
    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Action:
    action_id: str
    object_id: str
    operation: str


class RobotInterface(Protocol):
    """Driver methods must return promptly; status must refer to the exact action token.

    SUCCEEDED means driver completion, never proof of physical grasp/release success.
    Real adapters must supply sensor-backed verification before enabling sorting.
    """

    mode: str

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def start_pick(self, object_id: str) -> Action: ...
    def move_to_inspection(self, object_id: str) -> Action: ...
    def adjust_view(self, object_id: str) -> Action: ...
    def move_to_good(self, object_id: str) -> Action: ...
    def move_to_defective(self, object_id: str) -> Action: ...
    def release(self, object_id: str) -> Action: ...
    def verify_release(self, object_id: str) -> Action: ...
    def action_status(self, action: Action) -> ActionStatus: ...
    def stop_safely(self) -> None: ...


class MockRobot:
    """Deterministic scripted outcomes. No physical hardware is controlled or verified."""

    mode = "mock"

    def __init__(self, outcomes: dict[str, list[ActionStatus]] | None = None) -> None:
        self.outcomes = {k: list(v) for k, v in (outcomes or {}).items()}
        self.connected = False
        self.stopped = False
        self.actions: list[Action] = []
        self.statuses: dict[str, ActionStatus] = {}

    def connect(self) -> None:
        self.connected = True
        self.stopped = False

    def disconnect(self) -> None:
        self.connected = False

    def _start(self, operation: str, object_id: str) -> Action:
        if not self.connected or self.stopped:
            raise RuntimeError("Robot is disconnected or safely stopped")
        action = Action(str(len(self.actions) + 1), object_id, operation)
        self.actions.append(action)
        choices = self.outcomes.get(operation, [])
        self.statuses[action.action_id] = choices.pop(0) if choices else ActionStatus.SUCCEEDED
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
        return self._start("verify_release_mock_only", object_id)

    def action_status(self, action: Action) -> ActionStatus:
        if action not in self.actions:
            raise ValueError("Unknown action token")
        return self.statuses[action.action_id]

    def stop_safely(self) -> None:
        self.stopped = True
