"""Scene config parsing and camera geometry run offline without MuJoCo installed."""

import math

import numpy as np
import pytest

from brickcell.simulation import scene_config as sc
from brickcell.simulation.scene_config import (
    SceneConfigError,
    load_scene_config,
    parse_camera,
    parse_scene_config,
)

DOWN = {"pos": [0.3, 0, 1.1], "lookat": [0.3, 0, 0], "up": [0, 1, 0], "fovy": 50}


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else q[:, [1, 0, 2]]


def test_default_file_repeats_the_builtin_defaults():
    file, builtin = load_scene_config(sc.DEFAULT_PATH), parse_scene_config({})
    assert file.brick_colors == builtin.brick_colors == sc.DEFAULT_BRICK_COLORS
    assert file.plate_colors == builtin.plate_colors == sc.DEFAULT_PLATE_COLORS
    for name in ("inspection", "overview"):
        assert file.cameras[name] == builtin.cameras[name]
    # The real "Top" and "Plate" cameras are measured (2026-11-14) and on.
    assert file.cameras["overhead"].enabled and file.cameras["side"].enabled
    assert [c.name for c in file.world_cameras()] == ["inspection", "overview", "overhead", "side"]
    for name in ("overhead", "side"):
        cam = file.cameras[name]
        assert (cam.width, cam.height) == (640, 480)
        assert 0.0 < cam.pos[2] < 1.0  # above the table, below the tripod limit
    # The real "Gripper" camera: a standard pinhole camera on the gripper (no lens distortion).
    assert [c.name for c in file.attached_cameras()] == ["wrist"]
    wrist = file.cameras["wrist"]
    assert wrist.attach == "camera_mount" and (wrist.width, wrist.height) == (640, 360)
    assert file.offscreen_size() == (640, 480)


def test_default_cameras_keep_the_scene_xml_values():
    cameras = parse_scene_config({}).cameras
    assert cameras["inspection"].xml() == (
        '<camera name="inspection" pos="0.55 0.0 0.24" '
        'xyaxes="0.0 1.0 0.0 -0.287348 0.0 0.957826" fovy="22.0"/>'
    )
    assert cameras["overview"].xml() == (
        '<camera name="overview" pos="0.3 0.0 1.1" xyaxes="1.0 0.0 0.0 0.0 1.0 0.0" fovy="50.0"/>'
    )
    assert (cameras["inspection"].width, cameras["inspection"].height) == (256, 256)
    # The quaternion that MuJoCo 4.0.1 compiles from these xyaxes.
    assert cameras["inspection"].quat() == pytest.approx(
        [0.56730681, 0.42209357, 0.42209357, 0.56730681], abs=1e-7
    )


def test_opencv_to_mujoco_camera_frame():
    # OpenCV: +X right, +Y down, +Z forward. MuJoCo: +X right, +Y up, looks along -Z.
    # A camera that looks down with image right = world +X is the MuJoCo identity camera.
    down_cv = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    assert np.allclose(sc.mujoco_from_opencv(down_cv), np.eye(3))
    rng = np.random.default_rng(1)
    for _ in range(20):
        r_cv = random_rotation(rng)
        r_mj = sc.mujoco_from_opencv(r_cv)
        assert np.allclose(r_mj[:, 0], r_cv[:, 0])  # image right is the same axis
        assert np.allclose(r_mj[:, 1], -r_cv[:, 1])  # MuJoCo +Y is up, OpenCV +Y is down
        assert np.allclose(-r_mj[:, 2], r_cv[:, 2])  # MuJoCo looks along -Z, OpenCV along +Z
        assert np.linalg.det(r_mj) == pytest.approx(1)
        assert np.allclose(sc.opencv_from_mujoco(r_mj), r_cv)
        camera = parse_camera("c", {"pos": [0, 0, 1], "rotation_cv": r_cv.tolist(), "fovy": 40})
        assert camera.orientation[0] == "quat"
        assert np.allclose(camera.rotation(), r_mj) and np.allclose(camera.rotation_cv(), r_cv)


def test_quaternion_and_matrix_round_trip():
    rng = np.random.default_rng(2)
    for _ in range(50):
        rotation = random_rotation(rng)
        quat = sc.quat_from_rotation(rotation)
        assert quat[0] >= 0 and np.linalg.norm(quat) == pytest.approx(1)
        assert np.allclose(sc.rotation_from_quat(quat), rotation)
    assert np.allclose(sc.quat_from_rotation(np.eye(3)), [1, 0, 0, 0])
    assert np.allclose(sc.quat_from_rotation(np.diag([1.0, -1, -1])), [0, 1, 0, 0])


def test_the_four_orientation_forms_agree():
    lookat = parse_camera("c", DOWN)
    rotation = lookat.rotation()
    # Looks down (-Z world), image top = world +Y, image right = world +X.
    assert np.allclose(rotation, np.eye(3))
    base = {"pos": DOWN["pos"], "fovy": 50}
    forms = [
        {"xyaxes": [1, 0, 0, 0, 1, 0]},
        {"quat": [2, 0, 0, 0]},  # the loader normalizes it
        {"rotation_cv": [[1, 0, 0], [0, -1, 0], [0, 0, -1]]},
        {"rotation_cv": [1, 0, 0, 0, -1, 0, 0, 0, -1]},
    ]
    for form in forms:
        assert np.allclose(parse_camera("c", {**base, **form}).rotation(), rotation)
    tilted = parse_camera("c", {"pos": [0.55, -0.45, 0.3], "lookat": [0.1, 0, 0.02], "fovy": 45})
    forward = np.array([0.1, 0, 0.02]) - tilted.pos
    assert np.allclose(-tilted.rotation()[:, 2], forward / np.linalg.norm(forward))
    assert tilted.rotation()[2, 1] > 0  # image top points up in the world (default up = +Z)
    assert tilted.rotation()[2, 0] == pytest.approx(0)  # the image is level


def test_pinhole_projection_follows_the_opencv_convention():
    camera = parse_camera(
        "c", {"pos": [0.3, 0, 1.1], "lookat": [0.3, 0, 0], "up": [0, 1, 0]} | INTRINSICS
    )
    fx, fy, cx, cy = camera.pinhole()
    points = [[0.3, 0, 0], [0.4, 0, 0], [0.3, 0.1, 0]]
    centre, right, top = sc.project(points, camera.pos, camera.rotation_cv(), fx, fy, cx, cy)
    assert np.allclose(centre, [cx, cy])  # the lookat point is at the principal point
    assert np.allclose(right, [cx + fx * 0.1 / 1.1, cy])  # world +X = image right = larger u
    assert np.allclose(top, [cx, cy - fy * 0.1 / 1.1])  # image top = smaller v


def test_pose_from_opencv_extrinsics():
    rng = np.random.default_rng(3)
    r_world_to_camera, t = random_rotation(rng), rng.normal(size=3)
    pos, rotation_cv = sc.pose_from_extrinsics(r_world_to_camera, t)
    points = rng.normal(size=(5, 3)) + [0, 0, 6]
    local = points @ r_world_to_camera.T + t  # x_cam = R x_world + t
    expected = np.column_stack(
        (600 * local[:, 0] / local[:, 2] + 320, 590 * local[:, 1] / local[:, 2] + 240)
    )
    assert np.allclose(sc.project(points, pos, rotation_cv, 600, 590, 320, 240), expected)
    assert np.allclose(r_world_to_camera @ pos + t, 0)  # the camera centre maps to the origin


INTRINSICS = {"fx": 612.3, "fy": 608.7, "cx": 301.6, "cy": 255.2, "width": 640, "height": 480}


def test_intrinsics_become_fovy_and_mujoco_sensor_attributes():
    camera = parse_camera("c", {"pos": [0, 0, 1], "quat": [1, 0, 0, 0]} | INTRINSICS)
    assert camera.fovy == pytest.approx(math.degrees(2 * math.atan(480 / (2 * 608.7))))
    assert camera.pinhole() == (612.3, 608.7, 301.6, 255.2)
    sensor = camera.sensor()
    assert sensor["resolution"] == (640, 480)
    assert sensor["focalpixel"] == (612.3, 608.7)
    # MuJoCo: offset from the image centre (319.5, 239.5); + is left (x) and up (y).
    assert sensor["principalpixel"] == pytest.approx((319.5 - 301.6, 239.5 - 255.2))
    assert sensor["sensorsize"][0] / sensor["sensorsize"][1] == pytest.approx(640 / 480)
    for attribute in ("resolution=", "sensorsize=", "focalpixel=", "principalpixel="):
        assert attribute in camera.xml()
    assert "fovy" not in camera.xml()

    centred = parse_camera(
        "c",
        {"pos": [0, 0, 1], "quat": [1, 0, 0, 0], "fx": 500, "fy": 500, "width": 640}
        | {"height": 480},
    )
    assert centred.pinhole() == (500, 500, 319.5, 239.5)
    assert centred.sensor()["principalpixel"] == (0, 0)

    plain = parse_camera("c", DOWN | {"width": 320, "height": 240})
    focal = 120 / math.tan(math.radians(25))
    assert plain.sensor() is None
    assert plain.pinhole() == pytest.approx((focal, focal, 159.5, 119.5))
    assert sc.fovy_from_focal(sc.focal_from_fovy(50, 240), 240) == pytest.approx(50)


def test_camera_defaults_attach_and_enabled():
    camera = parse_camera("wrist", DOWN | {"attach": "camera_mount"})
    assert (camera.width, camera.height, camera.enabled) == (256, 256, True)
    config = parse_scene_config(
        {
            "cameras": {
                "wrist": DOWN | {"attach": "camera_mount"},
                "overhead": DOWN | {"width": 1280, "height": 720},
                "side": DOWN | {"enabled": False, "width": 1920, "height": 1080},
            }
        }
    )
    assert [c.name for c in config.world_cameras()] == ["inspection", "overview", "overhead"]
    assert [c.name for c in config.attached_cameras()] == ["wrist"]
    assert config.offscreen_size() == (1280, 720)  # a disabled camera does not count
    assert config.camera_size("overhead") == (1280, 720)
    assert config.camera_size("side") == config.camera_size("not-in-config") == (256, 256)
    described = config.as_dict()
    assert [c["name"] for c in described["cameras"]][-1] == "side"
    assert described["bricks"]["part-001"]["color"] == "#cc1414"


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (DOWN | {"quat": [1, 0, 0, 0]}, "exactly one"),
        ({"pos": [0, 0, 1], "fovy": 40}, "exactly one"),
        ({"pos": [0, 0, 1], "quat": [1, 0, 0, 0], "up": [0, 0, 1], "fovy": 40}, "up is only"),
        ({"lookat": [0, 0, 0], "fovy": 40}, "pos"),
        ({"pos": [0, 0], "lookat": [0, 0, 0], "fovy": 40}, "3 numbers"),
        ({"pos": [0, 0, "1"], "lookat": [0, 0, 0], "fovy": 40}, "3 numbers"),
        ({"pos": [0, 0, 1], "lookat": [0, 0, 1], "fovy": 40}, "different point"),
        ({"pos": [0, 0, 1], "lookat": [0, 0, 0], "fovy": 40}, "up is parallel"),
        ({"pos": [0, 0, 1], "xyaxes": [1, 0, 0, 2, 0, 0], "fovy": 40}, "parallel"),
        ({"pos": [0, 0, 1], "quat": [0, 0, 0, 0], "fovy": 40}, "zero"),
        ({"pos": [0, 0, 1], "rotation_cv": [[1, 0, 0], [0, 1, 0]], "fovy": 40}, "9 numbers"),
        (
            {"pos": [0, 0, 1], "rotation_cv": [[1, 0, 0], [0, 1, 0], [0, 0.5, 1]], "fovy": 40},
            "not a rotation",
        ),
        (
            {"pos": [0, 0, 1], "rotation_cv": [[1, 0, 0], [0, 1, 0], [0, 0, -1]], "fovy": 40},
            "mirror",
        ),
        (DOWN | {"fx": 600, "fy": 600, "width": 640, "height": 480}, "not both"),
        ({"pos": [0, 0, 1], "quat": [1, 0, 0, 0]}, "give fovy"),
        ({"pos": [0, 0, 1], "quat": [1, 0, 0, 0], "fx": 600, "fy": 600}, "width and height"),
        (
            {"pos": [0, 0, 1], "quat": [1, 0, 0, 0], "fx": 600, "width": 640, "height": 480},
            "fx and fy",
        ),
        (DOWN | {"fovy": 180}, "less than 180"),
        (DOWN | {"fovy": -5}, "more than 0"),
        (DOWN | {"width": 0}, "whole number"),
        (DOWN | {"height": 480.5}, "whole number"),
        (DOWN | {"fov": 40}, "unknown keys"),
        (DOWN | {"attach": "bad name!"}, "body name"),
        (DOWN | {"enabled": "yes"}, "true or false"),
        ("front", "mapping"),
    ],
)
def test_bad_camera_entries_give_clear_errors(entry, message):
    with pytest.raises(SceneConfigError, match=message):
        parse_camera("cam", entry)


def test_bad_camera_name_is_rejected():
    with pytest.raises(SceneConfigError, match="letters"):
        parse_camera('a"><geom', DOWN)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"bricks": {"part-009": {"color": "#ff0000"}}}, "unknown part id 'part-009'"),
        ({"bricks": {"part-001": {"color": "red"}}}, "not a colour"),
        ({"bricks": {"part-001": {"color": "#ff00"}}}, "not a colour"),
        ({"bricks": {"part-001": {"color": [255, 0, 0]}}}, "from 0 to 1"),
        ({"bricks": {"part-001": {"color": [1, 0]}}}, "three numbers"),
        ({"bricks": {"part-001": "#ff0000"}}, "mapping with one key"),
        ({"bricks": {"part-001": {"colour": "#ff0000"}}}, "mapping with one key"),
        ({"plates": {"maybe": {"color": "#ff0000"}}}, "unknown plate 'maybe'"),
        ({"cameras": {"top": DOWN | {"quat": [1, 0, 0, 0]}}}, "camera 'top'.*exactly one"),
        ({"cameras": {"inspection": DOWN | {"width": 640, "height": 480}}}, "256x256"),
        ({"cameras": {"inspection": DOWN | {"enabled": False}}}, "256x256"),
        ({"cameras": ["top"]}, "cameras: must be a mapping"),
        ({"camera": {}}, "unknown keys"),
        (["cameras"], "top level"),
    ],
)
def test_bad_scene_configs_give_clear_errors(raw, message):
    with pytest.raises(SceneConfigError, match=message):
        parse_scene_config(raw)


def test_colours():
    assert sc.parse_color("#FF8000") == (1.0, 128 / 255, 0.0)
    assert sc.parse_color([0.8, 0.08, 0.08]) == (0.8, 0.08, 0.08)
    assert sc.color_hex((1.0, 128 / 255, 0.0)) == "#ff8000"
    assert sc.color_hex(sc.parse_color("#0d54cc")) == "#0d54cc"
    config = parse_scene_config(
        {"bricks": {"part-002": {"color": "#ff8000"}}, "plates": {"good": {"color": [0, 0, 1]}}}
    )
    assert config.brick_colors["part-002"] == (1.0, 128 / 255, 0.0)
    assert config.brick_colors["part-001"] == sc.DEFAULT_BRICK_COLORS["part-001"]
    assert config.plate_colors == {
        "good": (0, 0, 1),
        "defective": (0.90, 0.72, 0.78),
        "spare": (0.06, 0.73, 0.88),
    }
    changed = config.with_brick_colors({"part-001": "#000000"})
    assert changed.brick_colors["part-001"] == (0, 0, 0)
    assert config.brick_colors["part-001"] == sc.DEFAULT_BRICK_COLORS["part-001"]
    with pytest.raises(SceneConfigError, match="unknown part id"):
        config.with_brick_colors({"part-9": "#000000"})


def test_brick_colour_command_line_values():
    assert sc.parse_brick_color_arg("part-001=#ff0000") == ("part-001", (1.0, 0.0, 0.0))
    assert sc.parse_brick_color_arg("part-004 = 0.1,0.2,0.3") == ("part-004", (0.1, 0.2, 0.3))
    for bad in ("part-001", "part-007=#ff0000", "part-001=red", "part-001=1,2,3"):
        with pytest.raises(SceneConfigError):
            sc.parse_brick_color_arg(bad)


def test_load_from_path_environment_and_objects(tmp_path, monkeypatch):
    path = tmp_path / "scene.yaml"
    path.write_text(
        "cameras:\n  overhead: {pos: [0.25, 0, 0.8], lookat: [0.25, 0, 0], up: [1, 0, 0],"
        " fovy: 42.5, width: 640, height: 360}\n"
        'bricks:\n  part-003: {color: "#112233"}\n'
    )
    config = load_scene_config(path)
    assert config.source == str(path)
    assert config.cameras["overhead"].enabled and "inspection" in config.cameras
    assert sc.color_hex(config.brick_colors["part-003"]) == "#112233"
    assert load_scene_config(str(path)) == config and load_scene_config(config) is config
    assert load_scene_config({"bricks": {}}).brick_colors == sc.DEFAULT_BRICK_COLORS

    monkeypatch.setenv(sc.ENV_VAR, str(path))
    assert load_scene_config().source == str(path)
    monkeypatch.delenv(sc.ENV_VAR)
    assert load_scene_config().source == str(sc.DEFAULT_PATH)

    with pytest.raises(SceneConfigError, match="not found"):
        load_scene_config(tmp_path / "missing.yaml")
    path.write_text("cameras: [")
    with pytest.raises(SceneConfigError, match="YAML"):
        load_scene_config(path)
    path.write_text("")  # an empty file gives the built-in defaults
    assert load_scene_config(path).cameras.keys() == {"inspection", "overview"}


def test_floor_color():
    file = load_scene_config(sc.DEFAULT_PATH)
    assert file.floor_color == parse_scene_config({}).floor_color == sc.DEFAULT_FLOOR_COLOR
    r, g, b = sc.DEFAULT_FLOOR_COLOR
    assert r > g > b  # light brown, as the real wooden table
    config = parse_scene_config({"floor": {"color": "#102030"}})
    assert sc.color_hex(config.floor_color) == "#102030"
    assert config.as_dict()["floor"]["color"] == "#102030"
    for bad in ({"floor": "#000000"}, {"floor": {"colour": "#000000"}}, {"floor": {"color": "x"}}):
        with pytest.raises(SceneConfigError, match="floor"):
            parse_scene_config(bad)
