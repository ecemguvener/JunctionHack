import copy

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from capture import check_state
from checks import physics_fixture_checks
from model import MOTOR_NAMES, ROOT, ServoCalibration, build, digest, new_priors, validate_prior
from published import merge, register
from replay import nominal_parameters, prepare, rollout, validate_episode


def episode():
    cal = ServoCalibration("follower")
    ticks = [2011, 2398, 1983, 1779, 3222]
    return {
        "schema": 1,
        "synthetic": True,
        "status": "complete",
        "partition": "fit",
        "session_id": "synthetic-test",
        "joint_order": list(MOTOR_NAMES),
        "calibration_sha256": cal.fingerprint,
        "alignment_sha256": digest(ROOT / "calibration/kinematics_alignment.json"),
        "samples": [{"t": i * 0.1, "ticks": ticks[:]} for i in range(21)],
        "commands": [{"t": i * 0.1, "ticks": ticks[:]} for i in range(21)],
    }


@pytest.mark.parametrize("mutation", ["stale", "incomplete", "time", "nan", "order", "gap"])
def test_reject_untrusted_recordings(mutation):
    e = episode()
    if mutation == "stale":
        e["calibration_sha256"] = "wrong"
    elif mutation == "incomplete":
        e["status"] = "incomplete"
    elif mutation == "time":
        e["samples"][2]["t"] = e["samples"][1]["t"]
    elif mutation == "nan":
        e["samples"][0]["ticks"][0] = float("nan")
    elif mutation == "order":
        e["joint_order"] = list(reversed(MOTOR_NAMES))
    else:
        e["samples"] = [e["samples"][0], e["samples"][10], e["samples"][20]]
    with pytest.raises(ValueError):
        validate_episode(e)


def test_reject_impossible_inertia():
    prior = next(iter(new_priors()["links"].values()))
    prior["diagonal_inertia_kg_m2"] = [1, 0.1, 0.1]
    with pytest.raises(ValueError, match="triangle"):
        validate_prior(prior)


def test_simulation_moves_without_overwriting_state(tmp_path):
    m = build(tmp_path)
    e = episode()
    e["commands"][5]["ticks"][2] -= 23
    t, q, ct, cq = prepare(e)
    prediction = rollout(m, t, q[0], ct, cq, nominal_parameters())
    assert prediction.shape == q.shape
    assert np.isfinite(prediction).all()
    assert np.max(np.abs(prediction - q)) > 1e-4
    # Recorded inputs are preserved; the phantom is independent forward dynamics.
    np.testing.assert_array_equal(e["samples"][5]["ticks"], episode()["samples"][5]["ticks"])
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_engine_gravity_and_contact_fixture():
    assert physics_fixture_checks()["contact_count"] > 0


def test_published_jaw_inertia_merge_preserves_parallel_axis_terms():
    frame = np.eye(4)
    frame[0, 3] = 2
    mass, center, tensor = merge(
        (1.0, np.zeros(3), np.eye(3)), (1.0, np.zeros(3), np.eye(3)), frame
    )
    assert mass == 2
    np.testing.assert_allclose(center, [1, 0, 0])
    np.testing.assert_allclose(tensor, np.diag([2, 4, 4]))


def test_inertia_frame_registration_preserves_incoming_axis():
    rng = np.random.default_rng(101)
    points = rng.normal(size=(250, 3)) * [0.04, 0.02, 0.01]
    rotation = Rotation.from_rotvec([0, 0, 0.7]).as_matrix()
    offset = np.array([0, 0, 0.03])
    r, t, evidence = register(points, points @ rotation.T + offset, incoming=True)
    np.testing.assert_allclose(r, rotation, atol=1e-5)
    np.testing.assert_allclose(t, offset, atol=1e-5)
    assert evidence["incoming_axis_error_deg"] == 0


def test_small_wrist_step_has_no_artificial_derivative_oscillation(tmp_path):
    model = build(tmp_path)
    t, q, ct, cq = prepare(episode())
    cq[5:, 4] += 0.01
    params = nominal_parameters()
    params.update(kp=[15] * 5, ki=[10] * 5, kd=[0.2] * 5)
    result = rollout(model, t, q[0], ct, cq, params)
    assert np.max(np.abs(result[:, 4] - q[0, 4])) < 0.03
    # Candidate evaluation must not leak gain/limit changes to another rollout.
    np.testing.assert_array_equal(model.actuator_gainprm[:, 0], 1)


def test_missing_position_feedback_blocks_capture():
    row = {
        "online": True,
        "healthy": True,
        "errors": [],
        "torque_enabled": True,
        "mode": 0,
        "temperature": 30,
        "voltage": 12.2,
        "current_ma": 20,
        "load": 5,
        "speed": 0,
        "position": 2000,
    }
    s = {
        "status": "online",
        "age_ms": 0,
        "joints": {n: copy.deepcopy(row) for n in (*MOTOR_NAMES, "gripper")},
    }
    check_state(s, [2000] * 5)
    s["joints"]["elbow_flex"]["online"] = False
    with pytest.raises(RuntimeError, match="Untrusted"):
        check_state(s, [2000] * 5)


def test_diagnostic_channels_do_not_add_experiment_limits():
    row = {
        "online": True,
        "healthy": True,
        "errors": [],
        "torque_enabled": True,
        "mode": 0,
        "temperature": 50,
        "voltage": 10.8,
        "current_ma": 500,
        "load": 50,
        "speed": 500,
        "position": 2000,
    }
    state = {
        "status": "online",
        "age_ms": 0,
        "joints": {n: copy.deepcopy(row) for n in (*MOTOR_NAMES, "gripper")},
    }
    check_state(state, [2000] * 5)
    state["joints"]["elbow_flex"]["position"] = 2100
    with pytest.raises(RuntimeError, match="envelope"):
        check_state(state, [2000] * 5)


def test_stale_hardware_positions_are_rejected():
    state = {"status": "online", "age_ms": 500, "joints": {}}
    with pytest.raises(RuntimeError, match="stale"):
        check_state(state, [2000] * 5)
