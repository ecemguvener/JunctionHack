"""Station service in replay mode (no hardware): HTTP end to end, record round trip, errors."""

import json
import sys
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from brickcell.simulation.scene_config import load_scene_config
from brickcell.twin import realsense, station
from brickcell.twin.state import BrickState, PlateState, StationState, load_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "artifacts" / "twin-fixtures" / "fixture-001"
SNAPSHOT = {
    "timestamp_utc": "2026-11-14T04:44:15+00:00",
    "arms": {
        "follower": {
            "joints": {
                "shoulder_pan": {"raw": 1970},
                "shoulder_lift": {"raw": 1246},
                "elbow_flex": {"raw": 2992},
                "wrist_flex": {"raw": 2263},
                "wrist_roll": {"raw": 3217},
                "gripper": {"raw": 1465},
            }
        }
    },
}

pytestmark = pytest.mark.skipif(not FIXTURE.is_dir(), reason="fixture-001 is not there")


class FakePerception(station.Perception):
    """Perception with a given behaviour: "missing", "raise" or a function."""

    def __init__(self, behaviour):
        super().__init__()
        self.behaviour = behaviour

    def _load(self):
        if self.behaviour == "missing":
            raise ImportError("No module named 'brickcell.twin.perception'")
        if self.behaviour == "raise":
            self.module = types.SimpleNamespace(locate=self._raise)
        else:
            self.module = types.SimpleNamespace(locate=self.behaviour)

    @staticmethod
    def _raise(frames, scene, config):
        raise RuntimeError("no plane found")


def one_brick(frames, scene, config):
    assert {f.camera for f in frames} == {"overhead", "side"}
    assert all(f.depth_m.dtype == np.float32 and f.rgb.shape == (480, 640, 3) for f in frames)
    return StationState(
        timestamp=0.0,
        bricks=[
            BrickState(
                "part-001",
                "3437",
                [0.25, 0.0, 0.0],
                [1, 0, 0, 0],
                [0.8, 0.1, 0.1],
                confidence=0.9,
                cameras=["overhead"],
            )
        ],
        plates=[PlateState("good", [0.25, -0.22, 0.0], 0.8)],
        diagnostics={"perception": {"ok": True}},
    )


def start(tmp_path, perception, arm_file=None, frames=None):
    config = station.load_station_config(
        None, {"port": 0, "record_dir": str(tmp_path / "rec"), "frames": frames}
    )
    arm_reader = None
    if arm_file:
        from brickcell.twin.arm import make_arm_reader

        arm_reader = make_arm_reader("file", arm_file, config)
    st = station.Station(
        realsense.FixtureSource(FIXTURE),
        load_scene_config(),
        config,
        arm_reader,
        FakePerception(perception),
    )
    st.open()
    server = station.serve(st, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def stop():
        server.shutdown()
        server.server_close()
        st.close()

    return st, url, stop


def get(url, raw=False):
    with urllib.request.urlopen(url, timeout=30) as r:
        body = r.read()
        return (r.status, r.headers["Content-Type"], body) if raw else json.loads(body)


def post(url, body):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_median_matches_the_contract():
    src = realsense.FixtureSource(FIXTURE)
    for ref in load_fixture(FIXTURE):
        frame = src.captures[ref.camera].to_frame()
        assert np.array_equal(frame.depth_m, ref.depth_m)
        assert np.array_equal(frame.rgb, ref.rgb)
        assert (frame.fx, frame.cy, frame.timestamp) == (ref.fx, ref.cy, ref.timestamp)
    depth = np.array([[[0, 100]], [[0, 0]], [[300, 200]], [[0, 400]]], np.uint16)
    out = realsense.median_depth(depth, 0.001)
    assert out[0, 0] == 0  # 1 of 4 valid: fewer than half
    assert out[0, 1] == pytest.approx(0.2)  # median of 100, 200, 400


def test_replay_http_end_to_end(tmp_path):
    arm_file = tmp_path / "joints.json"
    arm_file.write_text(json.dumps(SNAPSHOT))
    st, url, stop = start(tmp_path, one_brick, arm_file)
    try:
        health = get(url + "/health")
        assert health["mode"] == "fixture" and health["cameras"] == ["overhead", "side"]
        assert all(c["ok"] for c in health["intrinsics_check"].values())
        state = StationState.from_dict(get(url + "/state"))
        assert [b.object_id for b in state.bricks] == ["part-001"]
        assert state.arm is not None and state.arm.source == "file"
        assert set(state.cameras) == {"overhead", "side"}
        cam = state.cameras["overhead"]
        assert cam["serial"] == "350122060653" and cam["fx"] == pytest.approx(603.9367, abs=1e-3)
        assert 0.5 < cam["depth_valid_fraction"] <= 1
        assert state.timestamp == pytest.approx(max(c["timestamp"] for c in state.cameras.values()))
        diag = state.diagnostics
        assert diag["perception"] == {"ok": True} and diag["station"]["perception"] == "ok"
        assert diag["station"]["arm"]["gripper"]["percent"] == pytest.approx(9.0667, abs=1e-3)
        n = get(url + "/health")["snapshots"]
        get(url + "/state?fresh=1")
        assert get(url + "/health")["snapshots"] == n + 1
        for path, ctype in (
            ("/frame/overhead.jpg", "image/jpeg"),
            ("/depth/side.png", "image/png"),
            ("/overlay/overhead.jpg", "image/jpeg"),
            ("/", "text/html"),
        ):
            status, got, body = get(url + path, raw=True)
            assert status == 200 and got == ctype and len(body) > 100
        frame = cv2.imdecode(np.frombuffer(get(url + "/frame/overhead.jpg", True)[2], np.uint8), 1)
        overlay = cv2.imdecode(
            np.frombuffer(get(url + "/overlay/overhead.jpg", True)[2], np.uint8), 1
        )
        assert frame.shape == overlay.shape == (480, 640, 3)
        assert np.abs(frame.astype(int) - overlay.astype(int)).sum() > 1000  # something drawn
        with pytest.raises(urllib.error.HTTPError) as err:
            get(url + "/frame/nope.jpg")
        assert err.value.code == 404
        assert post(url + "/live", {"enabled": True, "hz": 5})[0] == 400
        assert post(url + "/live", {"enabled": True, "hz": 2}) == (
            200,
            {"enabled": True, "hz": 2.0},
        )
        assert post(url + "/live", {"enabled": False}) == (200, {"enabled": False, "hz": 2.0})
    finally:
        stop()


def test_record_round_trip(tmp_path):
    st, url, stop = start(tmp_path, "missing", frames=None)
    try:
        status, rec = post(url + "/record", {})
        assert status == 200 and rec["path"] == str(tmp_path / "rec" / "fixture-001")
        status, rec2 = post(url + "/record", {"fresh": True})
        assert status == 200 and rec2["path"].endswith("fixture-002")
        assert post(url + "/record", {"name": "fixture-001"})[0] == 400  # not empty
        assert post(url + "/record", {"name": "../escape"})[0] == 400
        status, rec3 = post(url + "/record", {"dir": str(tmp_path / "custom")})
        assert status == 200 and rec3["path"] == str(tmp_path / "custom")
    finally:
        stop()
    meta = json.loads((tmp_path / "rec" / "fixture-001" / "meta.json").read_text())
    original = json.loads((FIXTURE / "meta.json").read_text())
    for cam in ("overhead", "side"):
        for key in (
            "serial",
            "depth_scale_m",
            "width",
            "height",
            "fx",
            "fy",
            "cx",
            "cy",
            "distortion_model",
            "coeffs",
            "depth_stream",
            "frames",
            "firmware",
        ):
            assert meta[cam][key] == original[cam][key], (cam, key)
        a = np.load(FIXTURE / f"{cam}.npz")
        b = np.load(tmp_path / "rec" / "fixture-001" / f"{cam}.npz")
        for key in ("rgb", "depth", "timestamps_ms"):
            assert a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key])
    assert "captured_utc" in meta
    again = {f.camera: f for f in load_fixture(tmp_path / "rec" / "fixture-001")}
    for ref in load_fixture(FIXTURE):
        assert np.array_equal(again[ref.camera].depth_m, ref.depth_m)


def test_missing_or_failing_perception_still_serves(tmp_path):
    for behaviour, status, text in (
        ("missing", "missing", "perception module missing"),
        ("raise", "error", "RuntimeError: no plane found"),
    ):
        st, url, stop = start(tmp_path / behaviour, behaviour)
        try:
            state = StationState.from_dict(get(url + "/state?fresh=1"))
            assert state.bricks == [] and state.arm is None
            assert state.diagnostics["station"]["perception"] == status
            assert text in state.diagnostics["station"]["perception_error"]
            assert text in state.diagnostics["error"]
            assert set(state.cameras) == {"overhead", "side"}
            assert get(url + "/health")["ok"] is True
            assert get(url + "/overlay/side.jpg", raw=True)[0] == 200
        finally:
            stop()


def test_station_config(tmp_path):
    config = station.load_station_config()
    assert config["port"] == 8095 and config["host"] == "127.0.0.1" and config["frames"] == 10
    assert config["serials"] == {"overhead": "350122060653", "side": "243622060450"}
    bad = tmp_path / "twin.yaml"
    bad.write_text("station: {prot: 1}\n")
    with pytest.raises(ValueError, match="prot"):
        station.load_station_config(bad)
    bad.write_text("station: {live_hz: 9}\n")
    assert station.load_station_config(bad)["live_hz"] == 2.0


# --------------------------------------------------------------------------- live errors


class FakeRs(types.ModuleType):
    """Stand-in for pyrealsense2: one device, and a pipeline that is busy."""

    def __init__(self, serials, error):
        super().__init__("pyrealsense2")
        self.camera_info = types.SimpleNamespace(serial_number="sn", firmware_version="fw")
        self.stream = types.SimpleNamespace(color="color", depth="depth")
        self.format = types.SimpleNamespace(rgb8="rgb8", z16="z16")
        devices = [types.SimpleNamespace(get_info=lambda _k, s=s: s) for s in serials]
        self.context = lambda: types.SimpleNamespace(query_devices=lambda: devices)
        self.config = lambda: types.SimpleNamespace(
            enable_device=lambda s: None, enable_stream=lambda *a: None
        )

        def start(config):
            raise RuntimeError(error)

        self.pipeline = lambda: types.SimpleNamespace(start=start)


def test_busy_and_missing_camera_errors(monkeypatch, tmp_path, caplog):
    busy = "xioctl(VIDIOC_S_FMT) failed, errno=16 Last Error: Device or resource busy"
    monkeypatch.setitem(sys.modules, "pyrealsense2", FakeRs(["350122060653"], busy))
    with pytest.raises(realsense.CameraBusyError, match="busy.*8094"):
        realsense.RealSenseCamera("overhead", "350122060653").open()
    with pytest.raises(realsense.CameraNotFoundError, match="243622060450"):
        realsense.RealSenseCamera("side", "243622060450").open()
    # The service exits with code 2 and a clear message.
    code = station.main(["--port", "0", "--config", str(ROOT / "configs" / "twin.yaml")])
    assert code == 2
    assert any("busy" in r.getMessage() for r in caplog.records)
