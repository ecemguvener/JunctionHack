"""Shared fixtures: a config with two fake arms and a FastAPI TestClient."""

from __future__ import annotations

import copy
import time
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from armd.config import config_from_dict
from armd.fake import FakeBus
from armd.server import create_app

JOINTS = {
    "shoulder_pan": {"id": 1, "min": 1000, "max": 3000},   # config limits
    "shoulder_lift": {"id": 2},
    "elbow_flex": {"id": 3},
    "wrist_flex": {"id": 4},
    "wrist_roll": {"id": 5},
    "gripper": {"id": 6},                                 # EEPROM limits (see FAKE_LIMITS)
}
FAKE_LIMITS = {6: (1500, 2500)}

BASE_CONFIG = {
    "server": {"host": "127.0.0.1", "port": 8765, "poll_hz": 100, "admin": False},
    "arms": {
        "follower": {
            "port": "/dev/fake-follower",
            "allow_motion": True,
            "allow_spin": True,
            "max_speed": 3000,
            "default_speed": 2000,
            "default_acc": 0,
            "on_shutdown": "hold",
            "block_on_errors": ["overload", "temperature", "current", "sensor"],
            "joints": JOINTS,
        },
        "leader": {
            "port": "/dev/fake-leader",
            "allow_motion": False,
            "allow_spin": False,
            "max_speed": 1000,
            "default_speed": 300,
            "default_acc": 30,
            "joints": {k: {"id": v["id"]} for k, v in JOINTS.items()},
        },
    },
}


def wait_until(pred: Callable[[], bool], timeout: float = 2.0, step: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


def make_config_dict(mutate: Callable[[dict], None] | None = None) -> dict:
    data = copy.deepcopy(BASE_CONFIG)
    if mutate:
        mutate(data)
    return data


class AppHarness:
    """Build an app with FakeBus instances that tests can change."""

    def __init__(self, mutate=None, missing_ports: tuple[str, ...] = (), retry_interval=2.0):
        self.config = config_from_dict(make_config_dict(mutate))
        self.buses: dict[str, FakeBus] = {}
        self.missing_ports = missing_ports

        def factory(arm_cfg):
            bus = FakeBus(arm_cfg.port, arm_cfg.baudrate,
                          ids=[j.id for j in arm_cfg.joints.values()], limits=FAKE_LIMITS)
            if arm_cfg.name in self.missing_ports:
                bus.port_present = False
            self.buses[arm_cfg.name] = bus
            return bus

        self.app = create_app(self.config, bus_factory=factory, retry_interval=retry_interval)


@pytest.fixture
def harness_factory():
    """Return a function that builds an AppHarness and a started TestClient."""
    clients: list[TestClient] = []

    def build(mutate=None, missing_ports=(), retry_interval=2.0):
        h = AppHarness(mutate, missing_ports, retry_interval)
        c = TestClient(h.app)
        c.__enter__()
        clients.append(c)
        return h, c

    yield build
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def harness(harness_factory):
    return harness_factory()


@pytest.fixture
def client(harness):
    return harness[1]


@pytest.fixture
def buses(harness):
    return harness[0].buses
