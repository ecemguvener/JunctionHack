"""Dashboard server tests against fake upstream services (no hardware, no sim).

Run: uv run --no-project --with numpy --with opencv-python-headless --with pytest pytest dashboard/tests
"""

import gzip
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
                    "/rgb/plate": (jpeg((40, 120, 200)), "image/jpeg"),
                    "/rgb/wrist": (jpeg((180, 150, 40)), "image/jpeg"),
                    "/camera/overhead.png": (png(), "image/png"),
                    "/camera/side.png": (png(), "image/png"),
                    "/camera/wrist.png": (png(), "image/png"),
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
    assert len(data) == 2 * 8
    xyz_mm, colour, sensor = dash.unpack(data)
    assert xyz_mm.tolist() == [[250, -130, 10], [-100, 400, 300]]
    assert colour.tolist() == [(31 << 10) | (0 << 5) | 1, (2 << 10) | (31 << 5) | 31]
    assert sensor.tolist() == [0, 1]
    with pytest.raises(ValueError):
        dash.pack(xyz, np.zeros((2, 3)), np.array([0, 2]))


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


def evidence_payload(captured_at=100, **changes):
    evidence = dict(
        captured_at=captured_at,
        frame_id="plate-frame-1",
        camera_id="plate",
        mode="real",
        box_space="top_projected_object",
        method="dark_line_heuristic",
        model_version="heuristic-v1",
        threshold_version=None,
    )
    evidence.update(changes)
    return dict(
        source="detector",
        evidence=evidence,
        items=[dict(box=[10, 10, 20, 20], score=16.85, object_id="brick-1")],
    )


def test_legacy_publisher_stays_connected_but_boxes_are_unverified():
    overlay = dash.Overlay("anomalies", stale_s=5)
    legacy = dict(source="detector", items=[dict(box=[10, 10, 20, 20], score=16.85)])
    overlay.update(legacy, now=100)
    overlay.update(legacy, now=200)
    status = overlay.status(201)
    assert status["connected"] and not status["drawable"]
    assert status["evidence_state"] == "unverified"


def test_heartbeat_does_not_refresh_acquisition_expiry():
    overlay = dash.Overlay("anomalies", stale_s=5)
    payload = evidence_payload()
    overlay.update(payload, now=100)
    assert overlay.status(101)["drawable"]
    overlay.update(payload, now=106)
    status = overlay.status(106.5)
    assert status["connected"] and status["age_s"] == 0.5
    assert status["capture_age_s"] == 6.5 and not status["drawable"]
    assert status["evidence_state"] == "capture_stale"


def test_same_frame_cannot_receive_new_capture_time():
    overlay = dash.Overlay("anomalies", stale_s=5)
    overlay.update(evidence_payload(), now=100)
    with pytest.raises(ValueError, match="Cannot change acquisition"):
        overlay.update(evidence_payload(104), now=104)
    assert overlay.status(106)["connected"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"captured_at": float("nan")},
        {"captured_at": True},
        {"captured_at": 102},
        {"camera_id": ""},
        {"frame_id": ""},
        {"mode": "unknown"},
        {"box_space": "plate"},
    ],
)
def test_invalid_provenance_rejected_without_replacing_current(changes):
    overlay = dash.Overlay("anomalies", stale_s=5)
    overlay.update(evidence_payload(), now=100)
    with pytest.raises(ValueError):
        overlay.update(evidence_payload(**changes), now=100)
    assert overlay.status(101)["evidence"]["captured_at"] == 100


def test_plate_pixel_box_is_not_drawn_on_top_camera():
    overlay = dash.Overlay("anomalies", stale_s=5)
    overlay.update(evidence_payload(box_space="source_image_pixels"), now=100)
    status = overlay.status(101)
    assert not status["drawable"]
    assert status["evidence_state"] == "source_frame_required"


def test_projected_box_and_method_are_explicit_and_not_called_calibrated():
    overlay = dash.Overlay("anomalies", stale_s=5)
    overlay.update(evidence_payload(), now=100)
    status = overlay.status(101)
    assert status["drawable"] and status["evidence_state"] == "projected_object"
    assert status["evidence"]["method"] == "dark_line_heuristic"
    assert status["evidence"]["threshold_version"] is None


def test_new_frame_with_same_score_can_establish_new_capture():
    overlay = dash.Overlay("anomalies", stale_s=5)
    overlay.update(evidence_payload(), now=100)
    overlay.update(evidence_payload(106, frame_id="plate-frame-2"), now=106)
    assert overlay.status(107)["drawable"]


def test_empty_detections_do_not_establish_good():
    overlay = dash.Overlay("anomalies", stale_s=5)
    payload = evidence_payload()
    payload["items"] = []
    overlay.update(payload, now=100)
    assert "decision" not in overlay.status(101)


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
    assert status == 200 and "Content-Encoding" not in headers
    xyz_mm, _, sensor = dash.unpack(body)
    # top: 2 points kept (third is outside the workspace); plate: 1 point on the table
    assert len(xyz_mm) == int(headers["X-Cloud-Count"]) == 3
    assert sorted(sensor.tolist()) == [0, 0, 1]
    assert [250, -130, 0] in xyz_mm.tolist() and [300, -500, 0] in xyz_mm.tolist()
    req = urllib.request.Request(base + "/api/cloud?after=0", headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.headers["Content-Encoding"] == "gzip"
        assert len(dash.unpack(gzip.decompress(r.read()))[0]) == 3
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
    seq = int(headers["X-Frame-Seq"])
    status, headers, _ = get(base + f"/frame/real.jpg?after={seq + 1000}")  # long poll, none newer
    assert status == 204
    status, headers, _ = get(base + "/frame/real.jpg?after=1")
    assert status == 200 and int(headers["X-Frame-Seq"]) > 1
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
    assert not s["anomalies"]["drawable"] and not s["focus"]["drawable"]


def test_evidence_round_trip_over_http(stack):
    _, _, base = stack
    payload = evidence_payload(time.time())
    code, reply = post(base + "/api/anomalies", payload)
    assert code == 200 and reply["ok"]
    state = json.loads(get(base + "/api/status")[2])["anomalies"]
    assert state["drawable"] and state["evidence"] == payload["evidence"]
    payload["evidence"]["captured_at"] = time.time() - 10
    payload["evidence"]["frame_id"] = "old-frame"
    assert post(base + "/api/anomalies", payload)[0] == 200
    state = json.loads(get(base + "/api/status")[2])["anomalies"]
    assert state["connected"] and not state["drawable"]
    assert get(base + "/static/evidence.js")[0] == 200
    code, reply = post(
        base + "/api/anomalies", dict(source="test", items=[dict(box=[0, 0, 999, 1])])
    )
    assert code == 400 and "inside" in reply["error"]
    assert post(base + "/api/nothing", {})[0] == 404


def test_fit_table_levels_tilted_low_table():
    rng = np.random.default_rng(1)
    xy = rng.uniform([-0.1, -0.5], [0.6, 0.4], (5000, 2))
    tilt = math.radians(4)
    z = -0.04 + np.tan(tilt) * xy[:, 0] + rng.normal(0, 0.001, len(xy))  # 40 mm low, 4 deg tilt
    table = np.column_stack([xy, z])
    brick = np.column_stack([rng.uniform(0.2, 0.25, (300, 2)), np.full(300, 0.0)])
    brick[:, 2] = -0.04 + np.tan(tilt) * brick[:, 0] + 0.03  # 30 mm above the table
    level = dash.fit_table(np.vstack([table, brick]), rng)
    assert level is not None and abs(level.tilt_deg - 4) < 0.3
    # offset = table height at the inlier centre
    assert abs(level.offset_mm - (-40 + np.tan(tilt) * level.centre[0] * 1000)) < 1
    fixed = level.apply(np.vstack([table, brick]))
    assert np.abs(fixed[:5000, 2]).max() < 0.006  # table now on z = 0
    assert abs(np.median(fixed[5000:, 2]) - 0.03 * math.cos(tilt)) < 0.003  # brick height kept


def test_fit_table_refuses_without_table():
    rng = np.random.default_rng(2)
    assert dash.fit_table(rng.uniform(0.2, 0.3, (2000, 3)), rng) is None  # nothing near z = 0
    steep = np.column_stack([rng.uniform(-0.05, 0.05, (2000, 2)), np.zeros(2000)])
    steep[:, 2] = steep[:, 0] * np.tan(math.radians(30))  # 30 deg: too steep to be the table
    assert dash.fit_table(steep, rng) is None


def test_multi_camera_frame_routes(stack):
    _, _, base = stack
    for route in (
        "/frame/real/top.jpg",
        "/frame/real/plate.jpg",
        "/frame/real/wrist.jpg",
        "/frame/sim/top.jpg",
        "/frame/sim/plate.jpg",
        "/frame/sim/wrist.jpg",
    ):
        status_code, _, body = get(base + route)
        assert status_code == 200 and body.startswith(b"\xff\xd8")


def test_analyze_camera_features_wrist_corridor_and_bricks():
    # Synthetic wrist camera image: wood background, aquamarine bowl, red brick, and purple jaws
    img = np.full((360, 640, 3), (60, 120, 175), dtype=np.uint8)
    cv2.circle(img, (320, 180), 115, (220, 165, 25), -1)  # aquamarine bowl (HSV H~101)
    cv2.rectangle(img, (280, 120), (335, 165), (35, 35, 225), -1)  # red DUPLO brick inside bowl
    cv2.rectangle(img, (205, 230), (265, 359), (145, 45, 95), -1)  # left purple jaw (HSV H~130)
    cv2.rectangle(img, (375, 230), (435, 359), (145, 45, 95), -1)  # right purple jaw (HSV H~130)
    feat = dash.analyze_camera_features("wrist", img)
    assert feat["corridor"]["ok"] is True
    assert 310.0 <= feat["corridor"]["u_pinch"] <= 330.0
    assert len(feat["bricks"]) == 1
    assert len(feat["bricks"][0]["box_uv"]) == 4
    assert len(feat["bricks"][0]["axis_uv"]) == 2
