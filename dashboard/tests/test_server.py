"""Dashboard server tests against fake upstream services (no hardware, no sim).

Run: uv run --no-project --with numpy --with opencv-python-headless --with pytest pytest dashboard/tests
"""

import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server as dash  # noqa: E402

# A camera 0.8 m above the origin, looking straight down (OpenCV: x right, y down, z forward).
DOWN = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]


def scene_payload():
    cam = dict(enabled=True, fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    return dict(
        cameras=[
            dict(name="overhead", world_pos=[0.25, -0.13, 0.8], world_rotation_cv=DOWN, **cam),
            dict(name="side", world_pos=[0.3, -0.5, 0.25], world_rotation_cv=DOWN, **cam),
        ]
    )


def jpeg(color=(0, 106, 255)) -> bytes:
    image = np.zeros((480, 640, 3), np.uint8)
    image[:] = color
    return cv2.imencode(".jpg", image)[1].tobytes()


def png() -> bytes:
    return cv2.imencode(".png", np.full((480, 640, 3), 90, np.uint8))[1].tobytes()


class Upstream:
    """Fake live depth service (8094) and sim server (18081) on one port."""

    def __init__(self):
        self.age = 0.05
        self.fail = set()
        self.points = [
            [0.0, 0.0, 0.8, 200, 100, 50],
            [0.01, 0.0, 0.7, 10, 20, 30],
            [5.0, 5.0, 0.5, 1, 1, 1],
        ]
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in upstream.fail:
                    self.send_response(503)
                    self.end_headers()
                    return
                body, kind = {
                    "/rgb/top": (jpeg(), "image/jpeg"),
                    "/camera/overhead.png": (png(), "image/png"),
                    "/scene": (json.dumps(scene_payload()).encode(), "application/json"),
                    "/health": (
                        json.dumps(
                            dict(mode="mujoco", busy=False, stopped=False, run_id="r1")
                        ).encode(),
                        "application/json",
                    ),
                    "/points/top": (
                        json.dumps(
                            dict(points=upstream.points, age_s=upstream.age, serial="T")
                        ).encode(),
                        "application/json",
                    ),
                    "/points/plate": (
                        json.dumps(
                            dict(points=[[0.0, 0.0, 0.25, 9, 9, 9]], age_s=upstream.age, serial="P")
                        ).encode(),
                        "application/json",
                    ),
                }.get(path, (None, None))
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stack():
    up = Upstream()
    args = dash.build_parser().parse_args(
        [
            "--port",
            "0",
            "--real-url",
            up.url + "/rgb/top",
            "--sim-url",
            up.url,
            "--cloud",
            f"top=overhead={up.url}/points/top",
            "--cloud",
            f"plate=side={up.url}/points/plate",
            "--real-fps",
            "20",
            "--sim-fps",
            "10",
            "--cloud-hz",
            "10",
            "--overlay-stale",
            "2",
        ]
    )
    app = dash.Dashboard(args)
    app.start()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dash.make_handler(app))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    deadline = time.time() + 5
    while time.time() < deadline and not app.status()["ok"]:
        time.sleep(0.05)
    yield up, app, base
    app.stop.set()
    srv.shutdown()
    srv.server_close()
    up.close()


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, dict(r.headers), r.read()


def post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ----------------------------------------------------------------------------- pure functions


def test_to_world_uses_columns_as_camera_axes():
    cam = dict(pos=[0.25, -0.13, 0.8], rotation_cv=DOWN)
    # 0.8 m straight ahead of a down-looking camera is the table point below it
    assert np.allclose(dash.to_world(np.array([[0.0, 0.0, 0.8]]), cam), [[0.25, -0.13, 0.0]])
    # camera +x is world +x; camera +y (image down) is world -y
    assert np.allclose(dash.to_world(np.array([[0.1, 0.1, 0.8]]), cam), [[0.35, -0.23, 0.0]])


def test_pack_round_trip_and_rgb555():
    xyz = np.array([[0.25, -0.13, 0.01], [-0.1, 0.4, 0.3]])
    data = dash.pack(xyz, np.array([[255, 0, 8], [16, 255, 255]]), np.array([0, 1]))
    rec = dash.unpack(data)
    assert rec[:, :3].tolist() == [[250, -130, 10], [-100, 400, 300]]
    assert rec[0, 3] == (31 << 10) | (0 << 5) | 1 and rec[1, 4] == 1


def test_crop_keeps_workspace_only():
    xyz = np.array([[0.2, 0.0, 0.0], [5.0, 0.0, 0.0], [0.2, 0.0, 0.5]])
    assert dash.crop(xyz).tolist() == [True, False, False]


def test_parse_camera_rejects_bad_pose():
    good = dict(
        name="c",
        world_pos=[0, 0, 1],
        world_rotation_cv=DOWN,
        fx=1,
        fy=1,
        cx=0,
        cy=0,
        width=2,
        height=2,
    )
    assert dash.parse_camera(good)["width"] == 2
    with pytest.raises(ValueError):
        dash.parse_camera({**good, "world_rotation_cv": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]})
    with pytest.raises(ValueError):
        dash.parse_camera({**good, "fx": math.nan})


def test_overlay_validation_and_expiry():
    o = dash.Overlay("anomalies", stale_s=1)
    assert o.status(0)["connected"] is False and o.status(0)["stub"] is True
    o.update(
        dict(source="detector", items=[dict(box=[10, 10, 20, 20], score=0.8, label="mark")]),
        now=100,
    )
    assert o.status(100.5)["connected"] is True
    assert o.status(102)["connected"] is False
    for bad in (
        dict(items=[]),
        dict(source="d", items=[dict(box=[0, 0, 700, 10])]),
        dict(source="d", items=[dict(box=[5, 5, 1, 1])]),
        dict(source="d", items="x"),
        dict(source="d", items=[dict(box=[0, 0, 1, 1], score=math.inf)]),
    ):
        with pytest.raises(ValueError):
            o.update(bad)
    f = dash.Overlay("focus", stale_s=1)
    assert f.update(dict(source="planner", object_id=None))["box"] is None
    with pytest.raises(ValueError):
        f.update(dict(source="planner", object_id="part-001"))  # a focus object needs a box


# ----------------------------------------------------------------------------- HTTP end to end


def test_status_config_and_static(stack):
    _, _, base = stack
    status, _, body = get(base + "/api/status")
    s = json.loads(body)
    assert status == 200 and s["ok"], s
    assert s["focus"]["connected"] is False and s["anomalies"]["connected"] is False
    config = json.loads(get(base + "/api/config")[2])
    assert config["cameras"]["top"]["fx"] == 600 and config["image"] == dict(width=640, height=480)
    assert b"Tarot Card" in get(base + "/")[2]
    assert get(base + "/static/app.js")[0] == 200
    with pytest.raises(urllib.error.HTTPError):
        get(base + "/static/../server.py")


def test_cloud_long_poll_returns_world_points(stack):
    _, _, base = stack
    status, headers, body = get(base + "/api/cloud?after=0")
    assert status == 200
    rec = dash.unpack(body)
    # top: 2 points kept (third is outside the workspace); plate: 1 point on the table
    assert len(rec) == int(headers["X-Cloud-Count"]) == 3
    assert sorted(rec[:, 4].tolist()) == [0, 0, 1]
    assert [250, -130, 0] in rec[:, :3].tolist() and [300, -500, 0] in rec[:, :3].tolist()
    seq = int(headers["X-Cloud-Seq"])
    status, headers, _ = get(base + f"/api/cloud?after={seq}")
    assert status == 200 and int(headers["X-Cloud-Seq"]) > seq  # a newer cloud arrives
    with pytest.raises(urllib.error.HTTPError):
        get(base + "/api/cloud?after=x")


def test_stale_depth_is_reported_not_drawn(stack):
    up, app, base = stack
    up.age = 10.0
    time.sleep(0.6)
    s = app.status()["sources"]["cloud"]
    assert s["cameras"]["top"]["ok"] is False and "Stale" in s["cameras"]["top"]["error"]
    time.sleep(3.2)
    assert app.status()["sources"]["cloud"]["ok"] is False


def test_dead_source_marks_degraded(stack):
    up, app, _ = stack
    up.fail.add("/rgb/top")
    time.sleep(3.5)
    s = app.status()
    assert (
        s["ok"] is False
        and s["sources"]["real"]["ok"] is False
        and "HTTP" in s["sources"]["real"]["error"]
    )
    assert s["sources"]["sim"]["ok"] is True


def test_frames_and_mjpeg_stream(stack):
    _, _, base = stack
    status, headers, body = get(base + "/frame/sim.jpg")
    assert status == 200 and body.startswith(b"\xff\xd8")  # sim PNG re-encoded as JPEG
    with urllib.request.urlopen(base + "/stream/real.mjpg", timeout=5) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        chunk = r.read(4096)
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg") and b"\xff\xd8" in chunk


def test_overlay_push_endpoints(stack):
    _, _, base = stack
    code, reply = post(
        base + "/api/anomalies", dict(source="test", items=[dict(box=[1, 2, 30, 40], score=0.7)])
    )
    assert code == 200 and reply["ok"]
    code, _ = post(
        base + "/api/focus",
        dict(source="test", object_id="part-001", label="2x2 red", box=[5, 5, 50, 50]),
    )
    assert code == 200
    s = json.loads(get(base + "/api/status")[2])
    assert s["anomalies"]["connected"] and s["anomalies"]["items"][0]["box"] == [1, 2, 30, 40]
    assert s["focus"]["object_id"] == "part-001"
    code, reply = post(
        base + "/api/anomalies", dict(source="test", items=[dict(box=[0, 0, 999, 1])])
    )
    assert code == 400 and "inside" in reply["error"]
    assert post(base + "/api/nothing", {})[0] == 404
