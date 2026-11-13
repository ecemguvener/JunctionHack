"""HTTP API tests with FakeBus and the FastAPI TestClient. No hardware."""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn

from armd.client import ArmClient, ArmClientError
from armd.types import MODE_POSITION, MODE_WHEEL

from conftest import AppHarness, wait_until

JOINT_KEYS = {
    "name", "id", "position", "degrees", "goal_position", "speed", "goal_speed", "acceleration",
    "load", "voltage", "temperature", "current_ma", "moving", "torque_enabled", "mode", "errors",
    "healthy", "min", "max", "default_speed", "default_acc", "online",
}


def err(resp, status, code):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["error"] == code, body
    assert isinstance(body["detail"], str)
    return body


# --- read endpoints ----------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["admin"] is False
    assert body["version"]
    assert set(body["arms"]) == {"follower", "leader"}
    f = body["arms"]["follower"]
    assert f["status"] == "online" and f["port"] == "/dev/fake-follower"
    assert f["error"] is None and isinstance(f["last_poll"], float)


def test_arms_and_arm_info(client):
    arms = client.get("/arms").json()
    assert [a["name"] for a in arms] == ["follower", "leader"]
    info = client.get("/arms/follower").json()
    assert set(info) == {"name", "port", "status", "allow_motion", "allow_spin", "max_speed",
                         "default_speed", "default_acc", "joints"}
    joints = {j["name"]: j for j in info["joints"]}
    assert joints["shoulder_pan"] == {"name": "shoulder_pan", "id": 1, "min": 1000, "max": 3000}
    assert joints["gripper"]["min"] == 1500 and joints["gripper"]["max"] == 2500  # EEPROM
    assert joints["elbow_flex"]["min"] == 0 and joints["elbow_flex"]["max"] == 4095


def test_not_found(client):
    err(client.get("/arms/nope"), 404, "not_found")
    err(client.get("/arms/follower/joints/nope"), 404, "not_found")
    err(client.post("/arms/follower/joints/nope/position", json={"position": 1}), 404, "not_found")
    err(client.get("/no/such/route"), 404, "not_found")


def test_state_and_joint(client):
    s = client.get("/arms/follower/state").json()
    assert s["arm"] == "follower" and s["status"] == "online"
    assert isinstance(s["timestamp"], float) and s["age_ms"] >= 0
    assert set(s["joints"]) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                "wrist_roll", "gripper"}
    j = s["joints"]["elbow_flex"]
    assert set(j) == JOINT_KEYS
    assert j["position"] == 2048 and j["degrees"] == 180.0 and j["online"] is True
    assert j["mode"] == MODE_POSITION and j["healthy"] is True and j["errors"] == []

    fresh = client.get("/arms/follower/state", params={"fresh": "true"}).json()
    assert fresh["age_ms"] < 100
    one = client.get("/arms/follower/joints/gripper", params={"fresh": "true"}).json()
    assert set(one) == JOINT_KEYS and one["id"] == 6 and one["min"] == 1500


def test_docs(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


# --- moves -------------------------------------------------------------------

def test_move_wait_reached(client, buses):
    r = client.post("/arms/follower/joints/elbow_flex/position",
                    json={"position": 2248, "wait": True, "timeout": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"joint": "elbow_flex", "target": 2248, "clamped": False, "speed": 2000,
                    "acc": 0, "reached": True, "final_position": body["final_position"]}
    assert abs(body["final_position"] - 2248) <= 10
    assert buses["follower"].servos[3].torque_enabled  # auto torque on


def test_move_no_wait(client, buses):
    body = client.post("/arms/follower/joints/wrist_flex/position",
                       json={"degrees": 190.0, "speed": 500, "acc": 10}).json()
    assert body["target"] == round(190 * 4096 / 360)
    assert body["reached"] is None and body["final_position"] is None
    assert body["speed"] == 500 and body["acc"] == 10
    assert buses["follower"].servos[4].goal_position == body["target"]


def test_move_clamp(client):
    body = client.post("/arms/follower/joints/shoulder_pan/position",
                       json={"position": 4000}).json()
    assert body["target"] == 3000 and body["clamped"] is True
    body = client.post("/arms/follower/joints/gripper/position", json={"degrees": 10.0}).json()
    assert body["target"] == 1500 and body["clamped"] is True


def test_speed_rules(client):
    url = "/arms/follower/joints/elbow_flex/position"
    err(client.post(url, json={"position": 2100, "speed": 0}), 400, "invalid_request")
    err(client.post(url, json={"position": 2100, "speed": 5000}), 400, "invalid_request")
    err(client.post(url, json={"position": 2100, "acc": 300}), 400, "invalid_request")
    body = client.post(url, json={"position": 2100, "speed": 3400}).json()
    assert body["speed"] == 3000  # clamped to max_speed


def test_move_bad_body(client):
    url = "/arms/follower/joints/elbow_flex/position"
    err(client.post(url, json={}), 400, "invalid_request")
    err(client.post(url, json={"position": 1, "degrees": 2.0}), 400, "invalid_request")
    err(client.post(url, json={"positon": 1}), 400, "invalid_request")
    err(client.post(url, json={"position": "abc"}), 400, "invalid_request")
    err(client.post(url, content=b"not json", headers={"content-type": "application/json"}),
        400, "invalid_request")


def test_delta_move(client, buses):
    buses["follower"].servos[5].position = 1000.0
    body = client.post("/arms/follower/joints/wrist_roll/position", json={"delta": 100}).json()
    assert body["target"] == 1100
    body = client.post("/arms/follower/joints/wrist_roll/position",
                       json={"delta_degrees": -45.0, "wait": True}).json()
    # The delta uses the present position, not the last goal.
    assert abs(body["target"] - (body["final_position"] or 0)) <= 10
    assert body["reached"] is True


def test_multi_joint_move(client, buses):
    bus = buses["follower"]
    n_before = len(bus.calls_of("write_positions"))
    r = client.post("/arms/follower/positions", json={
        "joints": {"shoulder_lift": {"position": 2148},
                   "elbow_flex": {"degrees": 170.0, "speed": 1000},
                   "shoulder_pan": {"position": 100}},
        "speed": 2500, "wait": True, "timeout": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    writes = bus.calls_of("write_positions")
    assert len(writes) == n_before + 1  # one sync write
    assert set(writes[-1][0]) == {1, 2, 3}
    t = body["targets"]
    assert t["shoulder_lift"] == {"target": 2148, "clamped": False, "speed": 2500, "acc": 0}
    assert t["elbow_flex"]["speed"] == 1000
    assert t["shoulder_pan"]["clamped"] is True and t["shoulder_pan"]["target"] == 1000
    assert body["reached"] is True
    assert set(body["final_positions"]) == {"shoulder_lift", "elbow_flex", "shoulder_pan"}

    err(client.post("/arms/follower/positions", json={"joints": {"nope": {"position": 1}}}),
        404, "not_found")
    err(client.post("/arms/follower/positions", json={"joints": {}}), 400, "invalid_request")


def test_wait_timeout_not_error(client):
    body = client.post("/arms/follower/joints/elbow_flex/position",
                       json={"position": 3000, "speed": 100, "wait": True, "timeout": 0.2}).json()
    assert body["reached"] is False and isinstance(body["final_position"], int)


def test_set_speed(client):
    url = "/arms/follower/joints/elbow_flex/speed"
    r = client.put(url, json={"speed": 700, "acc": 5})
    assert r.json() == {"joint": "elbow_flex", "default_speed": 700, "default_acc": 5}
    assert client.put(url, json={"speed": 3400}).json()["default_speed"] == 3000
    err(client.put(url, json={"speed": 0}), 400, "invalid_request")
    client.put(url, json={"speed": 700})
    body = client.post("/arms/follower/joints/elbow_flex/position", json={"position": 2000}).json()
    assert body["speed"] == 700 and body["acc"] == 5
    j = client.get("/arms/follower/joints/elbow_flex").json()
    assert j["default_speed"] == 700 and j["default_acc"] == 5


def test_spin_and_back_to_position(client, buses):
    s = buses["follower"].servos[5]
    body = client.post("/arms/follower/joints/wrist_roll/spin", json={"speed": -5000})
    err(body, 400, "invalid_request")
    body = client.post("/arms/follower/joints/wrist_roll/spin", json={"speed": -3400}).json()
    assert body == {"joint": "wrist_roll", "speed": -3000, "acc": 0, "mode": 1}
    assert s.mode == MODE_WHEEL and s.wheel_speed == -3000 and s.torque_enabled
    assert client.post("/arms/follower/joints/wrist_roll/spin", json={"speed": 0}).json()["speed"] == 0
    client.post("/arms/follower/joints/wrist_roll/position", json={"position": 2048})
    assert s.mode == MODE_POSITION


def test_spin_needs_allow_spin(harness_factory):
    def no_spin(d):
        d["arms"]["follower"]["allow_spin"] = False
    _, c = harness_factory(no_spin)
    err(c.post("/arms/follower/joints/wrist_roll/spin", json={"speed": 100}), 403, "motion_disabled")
    err(c.post("/arms/leader/joints/wrist_roll/spin", json={"speed": 100}), 403, "motion_disabled")


# --- torque, stop, safety ------------------------------------------------------

def test_torque(client, buses):
    r = client.post("/arms/follower/torque", json={"enabled": True})
    assert r.json() == {"torque": {n: True for n in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                                      "wrist_flex", "wrist_roll", "gripper"]}}
    assert all(s.torque_enabled for s in buses["follower"].servos.values())
    r = client.post("/arms/follower/torque", json={"enabled": False, "joints": ["gripper"]})
    assert r.json() == {"torque": {"gripper": False}}
    assert not buses["follower"].servos[6].torque_enabled
    r = client.post("/arms/follower/joints/gripper/torque", json={"enabled": True})
    assert r.json() == {"torque": {"gripper": True}}
    err(client.post("/arms/follower/torque", json={"enabled": True, "joints": ["x"]}), 404, "not_found")


def test_torque_on_does_not_jump(client, buses):
    s = buses["follower"].servos[2]
    s.position = 1234.0          # moved by hand while limp; goal is still 2048
    client.post("/arms/follower/joints/shoulder_lift/torque", json={"enabled": True})
    assert s.goal_position == 1234


def test_leader_read_only(client, buses):
    err(client.post("/arms/leader/joints/gripper/position", json={"position": 2000}),
        403, "motion_disabled")
    err(client.post("/arms/leader/positions", json={"joints": {"gripper": {"position": 2000}}}),
        403, "motion_disabled")
    err(client.post("/arms/leader/torque", json={"enabled": True}), 403, "motion_disabled")
    err(client.post("/arms/leader/joints/gripper/torque", json={"enabled": True}), 403,
        "motion_disabled")
    assert client.post("/arms/leader/torque", json={"enabled": False}).status_code == 200
    assert client.post("/arms/leader/joints/gripper/torque", json={"enabled": False}).status_code == 200
    assert client.post("/arms/leader/stop", json={"mode": "limp"}).status_code == 200
    assert not buses["leader"].calls_of("write_positions")


def test_block_on_errors(client, buses):
    buses["follower"].servos[6].status = 1 << 5  # overload
    assert wait_until(lambda: client.get("/arms/follower/joints/gripper").json()["errors"]
                      == ["overload"])
    j = client.get("/arms/follower/joints/gripper").json()
    assert j["healthy"] is False
    err(client.post("/arms/follower/joints/gripper/position", json={"position": 2000}),
        409, "joint_fault")
    err(client.post("/arms/follower/positions",
                    json={"joints": {"gripper": {"position": 2000}, "elbow_flex": {"position": 2000}}}),
        409, "joint_fault")
    assert client.post("/arms/follower/joints/gripper/torque", json={"enabled": False}).status_code == 200
    assert client.post("/arms/follower/stop").status_code == 200
    # A non-blocking error (voltage) does not block.
    buses["follower"].servos[3].status = 1 << 0
    assert wait_until(lambda: client.get("/arms/follower/joints/elbow_flex").json()["errors"]
                      == ["voltage"])
    assert client.post("/arms/follower/joints/elbow_flex/position",
                       json={"position": 2000}).status_code == 200


def test_stop(client, buses):
    client.post("/arms/follower/joints/elbow_flex/position", json={"position": 3000, "speed": 200})
    time.sleep(0.05)
    body = client.post("/arms/follower/stop", json={"mode": "hold"}).json()
    assert body["mode"] == "hold"
    assert set(body["positions"]) == {"shoulder_pan", "shoulder_lift", "elbow_flex",
                                      "wrist_flex", "wrist_roll", "gripper"}
    s = buses["follower"].servos[3]
    assert s.goal_position == body["positions"]["elbow_flex"] < 3000
    body = client.post("/arms/follower/stop", json={"mode": "limp"}).json()
    assert body["mode"] == "limp"
    assert not any(s.torque_enabled for s in buses["follower"].servos.values())
    assert client.post("/arms/follower/stop").json()["mode"] == "hold"  # no body = hold
    err(client.post("/arms/follower/stop", json={"mode": "fast"}), 400, "invalid_request")


def test_release_acquire(client, buses):
    r = client.post("/arms/follower/release")
    assert r.json() == {"arm": "follower", "status": "released"}
    assert not buses["follower"].is_open
    err(client.post("/arms/follower/joints/gripper/position", json={"position": 2000}),
        409, "bus_released")
    err(client.post("/arms/follower/torque", json={"enabled": False}), 409, "bus_released")
    err(client.post("/arms/follower/stop"), 409, "bus_released")
    err(client.get("/arms/follower/state", params={"fresh": "true"}), 409, "bus_released")
    # Reads from the cache still work and show "released".
    s = client.get("/arms/follower/state").json()
    assert s["status"] == "released" and s["joints"]["gripper"]["online"] is False
    assert client.get("/arms/follower").json()["status"] == "released"
    h = client.get("/health").json()
    assert h["status"] == "degraded" and h["arms"]["follower"]["status"] == "released"

    r = client.post("/arms/follower/acquire")
    assert r.json() == {"arm": "follower", "status": "online"}
    assert client.post("/arms/follower/joints/gripper/position",
                       json={"position": 2000}).status_code == 200
    assert client.get("/health").json()["status"] == "ok"


# --- registers (admin) ---------------------------------------------------------

def test_registers_admin_off(client):
    err(client.get("/arms/follower/joints/gripper/registers/56"), 403, "motion_disabled")
    err(client.put("/arms/follower/joints/gripper/registers/41", json={"value": 1}),
        403, "motion_disabled")


def test_registers_admin_on(harness_factory):
    def admin(d):
        d["server"]["admin"] = True
    h, c = harness_factory(admin)
    assert c.get("/health").json()["admin"] is True
    r = c.get("/arms/follower/joints/gripper/registers/56", params={"size": 2})
    assert r.json() == {"addr": 56, "size": 2, "value": 2048}
    r = c.put("/arms/follower/joints/gripper/registers/41", json={"value": 20, "size": 1})
    assert r.json() == {"addr": 41, "size": 1, "value": 20}
    assert h.buses["follower"].servos[6].acc == 20
    for addr, size in [(5, 1), (31, 2), (39, 2), (55, 1), (54, 2)]:
        err(c.put(f"/arms/follower/joints/gripper/registers/{addr}",
                  json={"value": 1, "size": size}), 403, "motion_disabled")
    err(c.put("/arms/follower/joints/gripper/registers/41", json={"value": 300, "size": 1}),
        400, "invalid_request")
    err(c.get("/arms/follower/joints/gripper/registers/56", params={"size": 3}),
        400, "invalid_request")
    assert not [w for w in h.buses["follower"].calls_of("write_register") if w[1] < 40 or w[1] == 55]


# --- offline -----------------------------------------------------------------

def test_offline_arm_service_starts(harness_factory):
    h, c = harness_factory(missing_ports=("follower",))
    body = c.get("/health").json()
    assert body["status"] == "degraded"
    f = body["arms"]["follower"]
    assert f["status"] == "offline" and f["error"] and f["last_poll"] is None
    assert body["arms"]["leader"]["status"] == "online"
    s = c.get("/arms/follower/state").json()
    assert s["status"] == "offline" and s["timestamp"] is None
    assert all(j["online"] is False and j["position"] is None for j in s["joints"].values())
    err(c.post("/arms/follower/joints/gripper/position", json={"position": 2000}), 503, "bus_error")
    err(c.get("/arms/follower/state", params={"fresh": "true"}), 503, "bus_error")


def test_offline_then_port_appears(harness_factory):
    h, c = harness_factory(missing_ports=("follower",), retry_interval=0.1)
    assert c.get("/health").json()["arms"]["follower"]["status"] == "offline"
    h.buses["follower"].port_present = True
    assert wait_until(lambda: c.get("/health").json()["status"] == "ok", timeout=2)


# --- websocket ---------------------------------------------------------------

def test_stream(client):
    with client.websocket_connect("/arms/follower/stream?hz=50") as ws:
        a = ws.receive_json()
        b = ws.receive_json()
        assert a["arm"] == "follower" and set(a["joints"]["gripper"]) == JOINT_KEYS
        assert b["timestamp"] >= a["timestamp"]
    with client.websocket_connect("/arms/follower/stream?hz=500") as ws:
        assert ws.receive_json()["error"] == "invalid_request"
    with client.websocket_connect("/arms/nope/stream") as ws:
        assert ws.receive_json()["error"] == "not_found"


# --- python client against a real uvicorn server -------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server():
    h = AppHarness()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(h.app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    assert wait_until(lambda: server.started, timeout=5)
    yield h, f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=5)


def test_client(live_server):
    h, url = live_server
    c = ArmClient(url, arm="follower", timeout=5)
    assert c.health()["status"] == "ok"
    assert len(c.arms()) == 2
    assert c.state()["arm"] == "follower"
    assert c.state(fresh=True)["status"] == "online"
    assert c.joint("gripper", fresh=True)["id"] == 6
    m = c.move("elbow_flex", position=2148, wait=True, timeout=2)
    assert m["reached"] is True
    m = c.move_many({"shoulder_lift": 2100, "elbow_flex": {"delta": -50}}, speed=2000, wait=True)
    assert m["reached"] is True and set(m["targets"]) == {"shoulder_lift", "elbow_flex"}
    assert c.set_speed("gripper", 500, acc=3)["default_speed"] == 500
    assert c.spin("wrist_roll", 100)["mode"] == 1
    assert c.torque(False, ["wrist_roll"]) == {"torque": {"wrist_roll": False}}
    assert c.stop()["mode"] == "hold"
    assert c.release()["status"] == "released"
    with pytest.raises(ArmClientError) as e:
        c.move("gripper", position=2000)
    assert e.value.status == 409 and e.value.code == "bus_released"
    assert c.acquire()["status"] == "online"
    with pytest.raises(ArmClientError) as e:
        c.move("gripper", position=2000, speed=0)
    assert e.value.status == 400 and e.value.code == "invalid_request"
    with pytest.raises(ArmClientError) as e:
        c.read_register("gripper", 56)
    assert e.value.status == 403
    with pytest.raises(ArmClientError) as e:
        c.write_register("gripper", 41, 1)
    assert e.value.code == "motion_disabled"
    with pytest.raises(ArmClientError) as e:
        ArmClient(url, arm="leader").move("gripper", position=2000)
    assert e.value.status == 403
    with pytest.raises(ArmClientError) as e:
        ArmClient(f"http://127.0.0.1:{_free_port()}", timeout=1).health()
    assert e.value.status == 0 and e.value.code == "connection_error"
