"""Arm, FakeBus, and config tests. No hardware, no HTTP."""

from __future__ import annotations

from pathlib import Path

import pytest

from armd.arm import Arm
from armd.config import ConfigError, config_from_dict, load_config, resolve_config_path
from armd.fake import FakeBus
from armd.types import (
    MODE_POSITION,
    MODE_WHEEL,
    BusError,
    BusReleased,
    InvalidRequest,
    JointFault,
    MotionDisabled,
    NotFound,
    ServoError,
)

from conftest import FAKE_LIMITS, make_config_dict, wait_until

REPO = Path(__file__).resolve().parent.parent


def make_arm(name="follower", mutate=None, bus=None, **kw):
    cfg = config_from_dict(make_config_dict(mutate))
    acfg = cfg.arms[name]
    bus = bus or FakeBus(acfg.port, ids=[j.id for j in acfg.joints.values()], limits=FAKE_LIMITS)
    kw.setdefault("poll_hz", 100)
    return Arm(name, acfg, bus, **kw), bus


@pytest.fixture
def arm():
    a, bus = make_arm()
    a.start()
    yield a, bus
    a.shutdown()


# --- FakeBus -----------------------------------------------------------------

def test_fakebus_moves_over_time():
    bus = FakeBus("/dev/x", ids=(1,))
    bus.open()
    bus.set_torque([1], True)
    bus.write_positions({1: (2248, 2000, 0)})
    r = bus.read_states([1])[1]
    assert r.moving and r.position < 2248
    assert wait_until(lambda: bus.read_states([1])[1].position == 2248, timeout=1)
    assert not bus.read_states([1])[1].moving


def test_fakebus_failures():
    bus = FakeBus("/dev/x", ids=(1, 2))
    with pytest.raises(BusError):
        bus.read_states([1])        # not open
    bus.port_present = False
    with pytest.raises(BusError):
        bus.open()
    bus.port_present = True
    bus.open()
    bus.servos[2].online = False
    assert set(bus.read_states([1, 2])) == {1}
    with pytest.raises(ServoError):
        bus.ping(2)
    bus.servos[1].online = False
    with pytest.raises(ServoError):
        bus.read_states([1, 2])
    bus.fail = True
    with pytest.raises(BusError):
        bus.scan([1, 2])


# --- limits, clamps, defaults ----------------------------------------------------

def test_limits_config_and_eeprom(arm):
    a, _ = arm
    assert a.limits("shoulder_pan") == (1000, 3000)   # config
    assert a.limits("gripper") == (1500, 2500)        # EEPROM
    assert a.limits("elbow_flex") == (0, 4095)


def test_bad_eeprom_limits_use_full_range():
    bus = FakeBus("/dev/x", limits={3: (0, 0)})
    a, _ = make_arm(bus=bus)
    a.start()
    try:
        assert a.limits("elbow_flex") == (0, 4095)
    finally:
        a.shutdown()


def test_joint_defaults_from_config():
    def joint_defaults(d):
        d["arms"]["follower"]["joints"]["gripper"].update(default_speed=111, default_acc=7)
    a, bus = make_arm(mutate=joint_defaults)
    a.start()
    try:
        res = a.move("gripper", position=2000)
        assert (res["speed"], res["acc"]) == (111, 7)
        res = a.move("elbow_flex", position=2000)
        assert (res["speed"], res["acc"]) == (2000, 0)
    finally:
        a.shutdown()


def test_unknown_joint_and_bad_values(arm):
    a, _ = arm
    with pytest.raises(NotFound):
        a.move("nope", position=1)
    with pytest.raises(InvalidRequest):
        a.move("gripper", position=1, speed=0)
    with pytest.raises(InvalidRequest):
        a.move("gripper", timeout=0, position=1)
    with pytest.raises(InvalidRequest):
        a.move("gripper", tolerance=-1, position=1)
    with pytest.raises(InvalidRequest):
        a.move_many({"gripper": {"position": 1, "foo": 2}})
    with pytest.raises(InvalidRequest):
        a.spin("gripper", 4000)


# --- moves -------------------------------------------------------------------

def test_move_sets_position_mode_and_torque():
    bus = FakeBus("/dev/x")
    bus.servos[4].mode = MODE_WHEEL
    bus.servos[4].torque_enabled = True
    a, _ = make_arm(bus=bus)
    a.start()
    try:
        assert a.state()["joints"]["wrist_flex"]["mode"] == MODE_WHEEL   # mode cache at connect
        res = a.move("wrist_flex", position=2100, wait=True, timeout=2)
        assert res["reached"] is True
        s = bus.servos[4]
        assert s.mode == MODE_POSITION and s.torque_enabled
        # Torque off before the mode change, then the goal, then torque on.
        names = [n for n, _ in bus.calls if n in ("set_torque", "set_mode", "write_positions")]
        assert names[-4:] == ["set_torque", "set_mode", "write_positions", "set_torque"]
        assert a.state()["joints"]["wrist_flex"]["mode"] == MODE_POSITION
    finally:
        a.shutdown()


def test_moving_flag_and_wait_timeout(arm):
    a, bus = arm
    res = a.move("elbow_flex", position=3000, speed=200, wait=True, timeout=0.15)
    assert res["reached"] is False
    assert 2048 < res["final_position"] < 3000
    assert a.joint_state("elbow_flex", fresh=True)["moving"] is True


def test_missing_servo(arm):
    a, bus = arm
    bus.servos[3].online = False
    assert wait_until(lambda: a.state()["joints"]["elbow_flex"]["online"] is False)
    st = a.state()
    assert st["status"] == "online"
    j = st["joints"]["elbow_flex"]
    assert j["position"] is None and j["healthy"] is False and j["errors"] == []
    assert st["joints"]["gripper"]["online"] is True
    with pytest.raises(ServoError):
        a.move("elbow_flex", position=2000)
    with pytest.raises(ServoError):
        a.move_many({"elbow_flex": {"position": 2000}, "gripper": {"position": 2000}})
    a.stop("hold")   # stop works with a missing servo
    a.stop("limp")


def test_block_on_errors_list_is_used():
    def only_overload(d):
        d["arms"]["follower"]["block_on_errors"] = ["overload"]
    a, bus = make_arm(mutate=only_overload)
    a.start()
    try:
        bus.servos[1].status = 1 << 2   # temperature: not in the list
        bus.servos[2].status = 1 << 5   # overload
        a.state(fresh=True)
        a.move("shoulder_pan", position=2000)
        with pytest.raises(JointFault):
            a.move("shoulder_lift", position=2000)
        with pytest.raises(JointFault):
            a.spin("shoulder_lift", 100)
        a.spin("shoulder_lift", 0)   # speed 0 (stop) is allowed
    finally:
        a.shutdown()


def test_read_only_arm():
    a, bus = make_arm("leader")
    a.start()
    try:
        with pytest.raises(MotionDisabled):
            a.move("gripper", position=2000)
        with pytest.raises(MotionDisabled):
            a.torque(True)
        with pytest.raises(MotionDisabled):
            a.spin("gripper", 10)
        assert a.torque(False)["torque"]["gripper"] is False
        assert a.stop("hold")["mode"] == "hold"
    finally:
        a.shutdown()


# --- poller, offline, release ------------------------------------------------------

def test_offline_retry_and_recover():
    bus = FakeBus("/dev/x", limits=FAKE_LIMITS)
    bus.port_present = False
    a, _ = make_arm(bus=bus, retry_interval=0.1)
    a.start()
    try:
        assert a.status == "offline"
        assert a.health()["error"]
        with pytest.raises(BusError):
            a.move("gripper", position=2000)
        bus.port_present = True
        assert wait_until(lambda: a.status == "online", timeout=2)
        assert a.limits("gripper") == (1500, 2500)
        # The link fails: offline after some failed polls, then back.
        bus.fail = True
        assert wait_until(lambda: a.status == "offline", timeout=2)
        assert a.state()["joints"]["gripper"]["online"] is False
        bus.fail = False
        assert wait_until(lambda: a.status == "online", timeout=2)
    finally:
        a.shutdown()


def test_all_servos_missing_is_offline():
    bus = FakeBus("/dev/x")
    for s in bus.servos.values():
        s.online = False
    a, _ = make_arm(bus=bus, retry_interval=0.1)
    a.start()
    try:
        assert a.status == "offline"
        for s in bus.servos.values():
            s.online = True
        assert wait_until(lambda: a.status == "online", timeout=2)
    finally:
        a.shutdown()


def test_release_pauses_poller(arm):
    a, bus = arm
    a.release()
    assert a.status == "released" and not bus.is_open
    n = len(bus.calls_of("read_states"))
    wait_until(lambda: False, timeout=0.1)
    assert len(bus.calls_of("read_states")) == n
    for call in (lambda: a.move("gripper", position=2000), lambda: a.torque(False),
                 lambda: a.stop(), lambda: a.set_speed("gripper", 100),
                 lambda: a.state(fresh=True)):
        with pytest.raises(BusReleased):
            call()
    assert a.acquire()["status"] == "online"
    assert wait_until(lambda: len(bus.calls_of("read_states")) > n + 2)


def test_on_shutdown():
    a, bus = make_arm()
    a.start()
    a.torque(True)
    a.shutdown()
    assert all(s.torque_enabled for s in bus.servos.values())   # hold

    def limp(d):
        d["arms"]["follower"]["on_shutdown"] = "limp"
    a, bus = make_arm(mutate=limp)
    a.start()
    a.torque(True)
    a.shutdown()
    assert not any(s.torque_enabled for s in bus.servos.values())
    assert not bus.is_open


def test_registers_need_admin(arm):
    a, _ = arm
    with pytest.raises(MotionDisabled):
        a.read_register("gripper", 56, 2)
    a.admin = True
    assert a.read_register("gripper", 56, 2)["value"] == 2048
    with pytest.raises(MotionDisabled):
        a.write_register("gripper", 33, MODE_WHEEL, 1)   # mode is in EEPROM
    a.write_register("gripper", 40, 1, 1)
    assert a.state()["joints"]["gripper"]["torque_enabled"] is True
    assert wait_until(lambda: a.state()["joints"]["gripper"]["mode"] == MODE_POSITION)


# --- config ------------------------------------------------------------------

def test_load_repo_config(monkeypatch):
    monkeypatch.delenv("ARMD_ADMIN", raising=False)
    monkeypatch.delenv("ARMD_FAKE", raising=False)
    cfg = load_config(REPO / "config.yaml")
    assert set(cfg.arms) == {"follower", "leader"}
    assert cfg.server.port == 8765 and cfg.server.admin is False and cfg.server.fake is False
    f = cfg.arms["follower"]
    assert f.allow_motion and not cfg.arms["leader"].allow_motion
    assert f.joints["gripper"].id == 6 and f.joints["gripper"].min is None
    monkeypatch.setenv("ARMD_ADMIN", "1")
    monkeypatch.setenv("ARMD_FAKE", "1")
    cfg = load_config(REPO / "config.yaml")
    assert cfg.server.admin and cfg.server.fake


def test_config_path_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("ARMD_CONFIG", raising=False)
    assert resolve_config_path(None) == REPO / "config.yaml"
    monkeypatch.setenv("ARMD_CONFIG", str(tmp_path / "x.yaml"))
    assert resolve_config_path(None) == tmp_path / "x.yaml"
    assert resolve_config_path("/a/b.yaml") == Path("/a/b.yaml")
    with pytest.raises(ConfigError):
        load_config(None)   # ARMD_CONFIG file does not exist


@pytest.mark.parametrize("mutate", [
    lambda d: d["arms"]["follower"].update(max_speed=0),
    lambda d: d["arms"]["follower"].update(default_acc=300),
    lambda d: d["arms"]["follower"].update(on_shutdown="drop"),
    lambda d: d["arms"]["follower"].update(block_on_errors=["bad"]),
    lambda d: d["arms"]["follower"].update(allow_motion="yes"),
    lambda d: d["arms"]["follower"].update(typo_key=1),
    lambda d: d["arms"]["follower"]["joints"]["gripper"].update(id=1),
    lambda d: d["arms"]["follower"]["joints"]["gripper"].update(min=3000, max=1000),
    lambda d: d["arms"]["follower"].pop("port"),
    lambda d: d["server"].update(poll_hz=0),
])
def test_config_validation(mutate):
    with pytest.raises(ConfigError):
        config_from_dict(make_config_dict(mutate))


def test_config_defaults():
    cfg = config_from_dict({"arms": {"a": {"port": "/dev/x", "joints": {"j": {"id": 1}}}}})
    a = cfg.arms["a"]
    assert cfg.server.host == "127.0.0.1" and cfg.server.port == 8765 and cfg.server.poll_hz == 30
    assert (a.baudrate, a.allow_motion, a.allow_spin) == (1_000_000, False, False)
    assert (a.max_speed, a.default_speed, a.default_acc, a.on_shutdown) == (1000, 300, 30, "hold")
    assert a.block_on_errors == ["overload", "temperature", "current", "sensor"]
    assert config_from_dict({}).arms == {}
