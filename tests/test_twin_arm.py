"""Arm reader: ticks -> encoder -> CAD -> MuJoCo angles, and the FK check of the CAD -> MuJoCo map.

The FK test needs the MuJoCo env (it skips in the core env):
  MUJOCO_GL=egl PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" \
    artifacts/mujoco-venv/bin/python -m pytest -q tests/test_twin_arm.py
"""

import json
import math
import threading
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from brickcell.twin import arm
from brickcell.twin.state import ARM_JOINTS, StationState

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "models" / "so101-follower-calibrated.urdf"

# Recorded read-only snapshot (Intel, ~/cam-capture-20261114T044415Z/joints_before.json, follower)
# and the values that build_json.py wrote to joints.json for it.
SNAPSHOT = {
    "timestamp_utc": "2026-11-14T04:44:15.000000+00:00",
    "read_only": True,
    "arms": {
        "follower": {
            "joints": {
                "shoulder_pan": {"id": 1, "raw": 1970},
                "shoulder_lift": {"id": 2, "raw": 1246},
                "elbow_flex": {"id": 3, "raw": 2992},
                "wrist_flex": {"id": 4, "raw": 2263},
                "wrist_roll": {"id": 5, "raw": 3217},
                "gripper": {"id": 6, "raw": 1465},
            }
        }
    },
}
RECORDED = {
    "q_rad": {
        "shoulder_pan": 0.3083301383650139,
        "shoulder_lift": -1.7218934344016321,
        "elbow_flex": 1.688912847462091,
        "wrist_flex": 0.7516505860639642,
        "wrist_roll": 1.7939905314322573,
    },
    "cad_rad": {
        "shoulder_pan": 0.0,
        "shoulder_lift": -0.13917637246176673,
        "elbow_flex": -0.015817046346198627,
        "wrist_flex": 0.7993062558742783,
        "wrist_roll": 1.5661943844312396,
    },
    "gripper_percent": 9.066666666666666,
}


def test_recorded_snapshot_to_cad_and_mujoco():
    cal = arm.load_calibration()
    assert cal.matches_alignment
    ticks, stamp = arm.ticks_from_json(SNAPSHOT)
    state, details = arm.convert_ticks(ticks, cal, stamp, "fixture")
    for name in ARM_JOINTS:
        d = details["joints"][name]
        assert d["q_enc"] == pytest.approx(RECORDED["q_rad"][name], abs=1e-12)
        assert d["q_cad"] == pytest.approx(RECORDED["cad_rad"][name], abs=1e-12)
        expected = arm.wrap(RECORDED["cad_rad"][name] + arm.CAD_TO_MUJOCO_OFFSET[name])
        assert state.joints[name] == pytest.approx(expected, abs=1e-12)
    assert details["gripper"]["percent"] == pytest.approx(RECORDED["gripper_percent"])
    assert details["out_of_calibration"] == ["elbow_flex"]  # raw 2992 > range_max 2991
    # Folded rest pose: lift near the lower stop, elbow near the upper stop of the model.
    assert state.joints["shoulder_lift"] == pytest.approx(-1.75, abs=0.03)
    assert state.joints["elbow_flex"] == pytest.approx(1.515, abs=0.03)
    assert state.gripper == pytest.approx(0.0907 * arm.GRIPPER_OPEN_RAD, abs=1e-3)
    assert stamp == pytest.approx(1794631455.0)
    # The ArmState goes through the shared contract.
    s = StationState(timestamp=1.0, arm=state)
    assert StationState.from_json(s.to_json()).arm == state


def test_tick_layouts_and_file_reader(tmp_path):
    expected, _ = arm.ticks_from_json(SNAPSHOT)
    armd = {"arm": "follower", "timestamp": 12.5,
            "joints": {n: {"name": n, "position": int(v)} for n, v in expected.items()}}
    assert arm.ticks_from_json(armd) == (expected, 12.5)
    assert arm.ticks_from_json({"joints": expected}) == (expected, 0.0)
    path = tmp_path / "joints.json"
    path.write_text(json.dumps(SNAPSHOT))
    reader = arm.make_arm_reader("file", path)
    state, _ = reader.read()
    assert state.source == "file" and set(state.joints) == set(ARM_JOINTS)
    path.write_text("{")
    with pytest.raises(arm.ArmReadError):
        reader.read()
    assert arm.make_arm_reader("none") is None
    with pytest.raises(ValueError):
        arm.make_arm_reader("file")
    with pytest.raises(ValueError):
        arm.make_arm_reader("teleport")


def test_armd_reader_uses_state_endpoint():
    expected, _ = arm.ticks_from_json(SNAPSHOT)
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            seen.append(("GET", self.path))
            body = json.dumps({"arm": "follower", "timestamp": 5.0, "joints": {
                n: {"position": int(v)} for n, v in expected.items()}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        state, _ = arm.make_arm_reader("armd", config={"armd_url": url}).read()
    finally:
        server.shutdown()
    assert seen == [("GET", "/arms/follower/state?fresh=false")]
    assert state.source == "armd" and state.timestamp == 5.0
    with pytest.raises(arm.ArmReadError):
        arm.ArmdArmReader("http://127.0.0.1:9", timeout=0.5).read()


def test_gripper_and_wrap():
    assert arm.gripper_percent_to_mujoco(0) == 0.0
    assert arm.gripper_percent_to_mujoco(100) == pytest.approx(arm.GRIPPER_OPEN_RAD)
    assert arm.gripper_percent_to_mujoco(-50) == pytest.approx(arm.MUJOCO_RANGES["gripper"][0])
    assert arm.gripper_percent_to_mujoco(500) == pytest.approx(arm.MUJOCO_RANGES["gripper"][1])
    for a in np.linspace(-10, 10, 41):
        w = arm.wrap(a)
        assert -math.pi < w <= math.pi and math.isclose(math.cos(w), math.cos(a), abs_tol=1e-9)


# --------------------------------------------------------------------------- FK check


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_rotation(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


class UrdfChain:
    """Minimal URDF forward kinematics (numpy). Joint values by URDF joint name."""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = []
        for j in root.findall("joint"):
            o = j.find("origin")
            t = np.eye(4)
            t[:3, :3] = _rpy(*map(float, o.get("rpy").split()))
            t[:3, 3] = [float(v) for v in o.get("xyz").split()]
            axis = j.find("axis")
            axis = [float(v) for v in axis.get("xyz").split()] if axis is not None else None
            self.joints.append((j.get("name"), j.get("type"), j.find("parent").get("link"),
                                j.find("child").get("link"), t, axis))
        children = {j[3] for j in self.joints}
        self.root = next(x.get("name") for x in root.findall("link")
                         if x.get("name") not in children)

    def frames(self, q: dict) -> dict:
        out, todo = {self.root: np.eye(4)}, list(self.joints)
        while todo:
            for joint in list(todo):
                name, kind, parent, child, t, axis = joint
                if parent in out:
                    m = out[parent] @ t
                    if kind == "revolute":
                        r = np.eye(4)
                        r[:3, :3] = _axis_rotation(axis, q.get(name, 0.0))
                        m = m @ r
                    out[child] = m
                    todo.remove(joint)
        return out


# Joint order of tools/so101_kinematics.ARM_JOINTS: CAD joint -> servo joint.
CAD_JOINTS = dict(zip(("Revolute_1", "Revolute_2", "Revolute_3", "Revolute_2_5", "Revolute_4"),
                      ARM_JOINTS))
# Points fixed in the wrist-roll link. The CAD link "Body3" and the Menagerie body "gripper"
# have the same frame: the moving-jaw hinge is at (0.0202, ~0.019, -0.0234) in both.
GRIPPER_POINTS = {
    "origin": [0.0, 0.0, 0.0],
    "gripperframe (TCP site)": [0.012, -0.000218, -0.098127],
    "jaw hinge": [0.0202, 0.0188, -0.0234],
}


def _kabsch(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ d @ u.T
    return r, cd - r @ cs


def fk_agreement(model, offsets=None, samples=200, seed=0):
    """FK of the CAD URDF (input: encoder radians) and of the MuJoCo model (input: arm.py map).

    One rigid base transform (Kabsch over all samples) connects the two base frames. Returns
    (per-point errors in mm (samples, points), base rotation CAD -> MuJoCo).
    """
    import mujoco

    offsets = offsets or arm.CAD_TO_MUJOCO_OFFSET
    cal = arm.load_calibration()
    chain = UrdfChain(URDF)
    data = mujoco.MjData(model)
    qadr = {n: model.joint(n).qposadr[0] for n in ARM_JOINTS}
    body = model.body("gripper").id
    rng = np.random.default_rng(seed)
    lo = np.array([(cal.range_min[n] - cal.mid(n)) * arm.RADIANS_PER_TICK for n in ARM_JOINTS])
    hi = np.array([(cal.range_max[n] - cal.mid(n)) * arm.RADIANS_PER_TICK for n in ARM_JOINTS])
    points = np.array(list(GRIPPER_POINTS.values()))
    cad_pts, mj_pts = [], []
    for q_enc in rng.uniform(lo, hi, size=(samples, 5)):
        frames = chain.frames({cad: q_enc[i] for i, cad in enumerate(CAD_JOINTS)})
        t = frames["Body3"]
        cad_pts.append(points @ t[:3, :3].T + t[:3, 3])
        for i, name in enumerate(ARM_JOINTS):
            q_cad = cal.cad_at_zero[name] + cal.cad_sign[name] * q_enc[i]
            q = arm.CAD_TO_MUJOCO_SIGN[name] * q_cad + offsets[name]
            data.qpos[qadr[name]] = arm.wrap(q) if name in arm.WRAP_JOINTS else q
        mujoco.mj_kinematics(model, data)
        rot = data.xmat[body].reshape(3, 3)
        mj_pts.append(points @ rot.T + data.xpos[body])
    cad_pts, mj_pts = np.array(cad_pts), np.array(mj_pts)
    r, t = _kabsch(cad_pts.reshape(-1, 3), mj_pts.reshape(-1, 3))
    err = np.linalg.norm(cad_pts @ r.T + t - mj_pts, axis=2) * 1000
    return err, r


@pytest.fixture(scope="module")
def world_model(tmp_path_factory):
    pytest.importorskip("mujoco")
    try:
        from brickcell.simulation.world import World
    except ImportError as exc:
        pytest.skip(f"MuJoCo env not complete: {exc}")
    return World(tmp_path_factory.mktemp("world")).model


def test_cad_to_mujoco_fk_agreement(world_model):
    err, rot = fk_agreement(world_model)
    tcp = err[:, 1]
    print(f"FK agreement over {len(err)} configs: TCP mean {tcp.mean():.2f} mm, "
          f"max {tcp.max():.2f} mm; all points max {err.max():.2f} mm")
    # Measured: TCP mean 1.0 mm, max 2.6 mm (small link-length differences of the two models).
    assert tcp.mean() < 1.5 and tcp.max() < 3.5 and err.max() < 3.5
    # Both base frames have +Z up (the base transform is a turn about Z only).
    assert rot[2, 2] == pytest.approx(1.0, abs=2e-3)


def test_fk_check_detects_a_wrong_map(world_model):
    wrong = dict(arm.CAD_TO_MUJOCO_OFFSET)
    wrong["wrist_roll"] += math.pi  # a half turn of the gripper
    assert fk_agreement(world_model, wrong, samples=50)[0].max() > 20
    wrong = dict(arm.CAD_TO_MUJOCO_OFFSET)
    wrong["elbow_flex"] += 0.1
    assert fk_agreement(world_model, wrong, samples=50)[0][:, 1].mean() > 5
