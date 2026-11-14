"""Forward-dynamics command replay and bounded effective-actuator fitting.

Uses MuJoCo 4.0 sysid's optimizer with a custom PID rollout residual.
Servo firmware gains are NOT numerically equal to these SI effective gains.
Mass/contact priors stay fixed: absolute mass is not observable here.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np
from mujoco import sysid

from model import MOTOR_NAMES, ServoCalibration, digest, dump, indices, load


def nominal_parameters(model_dir=None):
    result = {
        "kp": [1.5] * 5,
        "ki": [0.8] * 5,
        "kd": [0.06] * 5,
        "delay_s": 0.05,
        "torque_limit_nm": 1.5,
        "controller_dt_s": 0.01,
        "integral_limit_rad_s": 0.5,
        "status": "effective gains with fixed unmeasured inertial priors",
    }
    if model_dir is not None:
        physics = json.loads((Path(model_dir) / "inertial_priors.json").read_text())["physics"]
        result["torque_limit_nm"] = physics.get("motor_torque_limit_nm", 1.5)
    return result


def validate_episode(episode):
    if episode.get("schema") != 1 or episode.get("status") != "complete":
        raise ValueError("Episode is incomplete or unsupported")
    if episode.get("synthetic") not in (True, False):
        raise ValueError("Episode must explicitly label synthetic vs hardware data")
    if episode.get("joint_order") != list(MOTOR_NAMES):
        raise ValueError("Wrong joint ordering")
    samples, commands = episode["samples"], episode["commands"]
    for records, key in ((samples, "ticks"), (commands, "ticks")):
        times = np.array([r["t"] for r in records], float)
        values = np.array([r[key] for r in records], float)
        if len(times) < 3 or values.shape != (len(times), 5):
            raise ValueError("Missing trajectory data")
        if (
            not np.isfinite(times).all()
            or not np.isfinite(values).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("Non-finite or unordered trajectory data")
        if np.max(np.diff(times)) > 0.35:
            raise ValueError("Trajectory has excessive timestamp gaps")
    cal = ServoCalibration("follower")
    if episode.get("calibration_sha256") != cal.fingerprint:
        raise ValueError("Episode has stale calibration")
    if episode.get("alignment_sha256") != digest(
        Path(__file__).resolve().parents[1] / "calibration/kinematics_alignment.json"
    ):
        raise ValueError("Episode has stale geometric alignment")
    return episode


def prepare(episode, dt=0.02):
    validate_episode(episode)
    cal = ServoCalibration("follower")
    samples = episode["samples"]
    ts = np.array([r["t"] for r in samples])
    qs = np.array([cal.ticks_to_radians(r["ticks"]) for r in samples])
    times = np.arange(ts[0], ts[-1], dt)
    observed = np.column_stack([np.interp(times, ts, qs[:, j]) for j in range(5)])
    commands = episode["commands"]
    ct = np.array([r["t"] for r in commands])
    cq = np.array([cal.ticks_to_radians(r["ticks"]) for r in commands])
    return times, observed, ct, cq


def rollout(model, times, initial, ct, cq, parameters, timestep=None):
    """State is initialized once; observed q never overwrites the phantom."""
    qi, vi = indices(model)
    data = mujoco.MjData(model)
    data.qpos[qi] = initial
    mujoco.mj_forward(model, data)
    # A stationary real servo already carries load in its hidden integrator.
    # Initialize an estimated equilibrium state, rather than invent a release
    # from zero torque at the start of each recorded episode.
    integral = np.clip(
        data.qfrc_bias[vi] / np.asarray(parameters["ki"]),
        -parameters["integral_limit_rad_s"],
        parameters["integral_limit_rad_s"],
    )
    dt = parameters["controller_dt_s"]
    original_dt = model.opt.timestep
    if timestep is not None:
        model.opt.timestep = timestep
    nstep = int(round(dt / model.opt.timestep))
    if abs(nstep * model.opt.timestep - dt) > 1e-10:
        raise ValueError("Controller interval must be a multiple of the physics timestep")
    # Evaluate proportional/velocity feedback inside MuJoCo on every physics
    # step. Holding an explicit derivative torque for 10 ms made the tiny
    # wrist inertia numerically unstable and created fictitious oscillations.
    fields = (
        "actuator_biastype",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrllimited",
        "actuator_forcelimited",
        "actuator_forcerange",
    )
    original_actuators = {name: getattr(model, name).copy() for name in fields}
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[:, 0] = parameters["kp"]
    model.actuator_biasprm[:, 1] = -np.asarray(parameters["kp"])
    model.actuator_biasprm[:, 2] = -np.asarray(parameters["kd"])
    model.actuator_ctrllimited[:] = False
    model.actuator_forcelimited[:] = True
    model.actuator_forcerange[:, 0] = -parameters["torque_limit_nm"]
    model.actuator_forcerange[:, 1] = parameters["torque_limit_nm"]
    result = []
    start = times[0]
    try:
        for target_time in times - start:
            while data.time < target_time - 1e-8:
                delayed = start + data.time - parameters["delay_s"]
                index = np.clip(np.searchsorted(ct, delayed, side="right") - 1, 0, len(ct) - 1)
                error = cq[index] - data.qpos[qi]
                integral = np.clip(
                    integral + error * dt,
                    -parameters["integral_limit_rad_s"],
                    parameters["integral_limit_rad_s"],
                )
                data.ctrl[:] = (
                    cq[index]
                    + np.asarray(parameters["ki"]) / np.asarray(parameters["kp"]) * integral
                )
                mujoco.mj_step(model, data, nstep=nstep)
                if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qvel)) > 100:
                    raise ValueError("Divergent rollout")
            result.append(data.qpos[qi].copy())
    finally:
        model.opt.timestep = original_dt
        for name, value in original_actuators.items():
            getattr(model, name)[:] = value
    return np.array(result)


def replay(model, episode, params):
    times, observed, ct, cq = prepare(episode)
    return times, observed, rollout(model, times, observed[0], ct, cq, params)


def scores(observed, predicted):
    error = np.rad2deg(predicted - observed)
    return {
        "rmse_degrees": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "max_error_degrees": np.max(np.abs(error), axis=0).tolist(),
        "mean_rmse_degrees": float(np.sqrt(np.mean(error**2))),
    }


def fit(model_dir, fit_path, validation_path, output, max_iters=25):
    fit_episode = json.loads(Path(fit_path).read_text())
    val_episode = json.loads(Path(validation_path).read_text())
    if fit_episode["session_id"] == val_episode["session_id"]:
        raise ValueError("Fit and validation must use separate capture sessions")
    if fit_episode["partition"] != "fit" or val_episode["partition"] != "val":
        raise ValueError("Explicit fit/val partitions required")
    if fit_episode["synthetic"] != val_episode["synthetic"]:
        raise ValueError("Cannot silently mix synthetic and physical validation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = load(model_dir)
    nominal = nominal_parameters(model_dir)
    times, observed, ct, cq = prepare(fit_episode)
    params = sysid.ParameterDict()
    for name, low, high in (("kp", 0.05, 30), ("ki", 0.01, 25), ("kd", 0.005, 1.5)):
        params.add(sysid.Parameter(name, nominal[name], [low] * 5, [high] * 5))
    params.add(sysid.Parameter("delay_s", nominal["delay_s"], 0.001, 0.25))
    history = []

    def decode(x):
        p = copy.deepcopy(nominal)
        for name, start in (("kp", 0), ("ki", 5), ("kd", 10)):
            p[name] = np.asarray(x[start : start + 5]).tolist()
        p["delay_s"] = float(x[15])
        return p

    best = float("inf")

    def residual(x, _):
        nonlocal best
        p = decode(x)
        prediction = rollout(model, times, observed[0], ct, cq, p)
        errors = (prediction - observed).ravel()
        loss = float(np.mean(errors**2))
        if loss < best * (1 - 1e-6):
            best = loss
            history.append(
                {
                    "evaluation": len(history),
                    "fit_rmse_degrees": float(np.rad2deg(np.sqrt(loss))),
                    "parameters": p,
                }
            )
            dump(output / "progress.json", history)
            print(f"fit RMSE {history[-1]['fit_rmse_degrees']:.4f} degrees", flush=True)
        # No validation data enters the optimization or best-candidate selection.
        return [errors], [], []

    # Delay uses zero-order-held commands: evaluate a fixed coarse grid first,
    # then freeze it. Finite differences cannot identify a discontinuous delay.
    candidates = []
    for kp, ki, kd in ((1.5, 0.8, 0.06), (5, 3, 0.15), (15, 10, 0.2), (25, 15, 0.3)):
        for delay in (0.001, 0.05, 0.10, 0.15):
            p = copy.deepcopy(nominal)
            p.update(kp=[kp] * 5, ki=[ki] * 5, kd=[kd] * 5, delay_s=delay)
            pred = rollout(model, times, observed[0], ct, cq, p)
            candidates.append((float(np.mean((pred - observed) ** 2)), p))
    nominal = min(candidates, key=lambda row: row[0])[1]
    selected_seed = copy.deepcopy(nominal)
    for name in ("kp", "ki", "kd"):
        params[name].value[:] = nominal[name]
    params["delay_s"].value[:] = nominal["delay_s"]
    params["delay_s"].frozen = True

    def decode_frozen(x):
        return decode(np.r_[np.asarray(x).ravel(), nominal["delay_s"]])

    # Keep decode at the fixed 16-entry representation for callback snapshots.
    def residual_frozen(x, p):
        return residual(np.r_[np.asarray(x).ravel(), nominal["delay_s"]], p)

    fitted, result = sysid.optimize(
        params,
        residual_frozen,
        optimizer="scipy",
        verbose=False,
        max_iters=max_iters,
        x_scale="jac",
        diff_step=1e-3,
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-5,
    )
    final = decode_frozen(fitted.as_vector())
    before = nominal_parameters(model_dir)
    report = {
        "schema": 1,
        "synthetic": fit_episode["synthetic"],
        "fit_session": fit_episode["session_id"],
        "val_session": val_episode["session_id"],
        "fit_sha256": digest(fit_path),
        "validation_sha256": digest(validation_path),
        "model_sha256": digest(Path(model_dir) / "arm.xml"),
        "rollout_source_sha256": digest(__file__),
        "actuator_model": "MuJoCo-native affine PD with a sampled, clamped integral state",
        "inertial_input_status": json.loads((Path(model_dir) / "inertial_priors.json").read_text())[
            "status"
        ],
        "parameters_before": before,
        "parameters_after": final,
        "training_selected_seed": selected_seed,
        "history": history,
        "mass_identified": False,
        "contacts_calibrated": False,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "versions": {
            n: importlib.metadata.version(n) for n in ("mujoco", "frax", "jax", "numpy", "scipy")
        },
    }
    jac = np.asarray(result.jac)
    sv = np.linalg.svd(jac, compute_uv=False)
    report["jacobian_singular_values"] = sv.tolist()
    report["numerical_rank"] = int(np.sum(sv > max(sv[0] * 1e-6, 1e-10)))
    lower, upper = fitted.get_bounds()
    values = fitted.as_vector()
    report["parameter_bound_hits"] = np.flatnonzero(
        (values - lower < 1e-3 * (upper - lower)) | (upper - values < 1e-3 * (upper - lower))
    ).tolist()
    report["parameter_identification_warning"] = (
        "Boundary solutions and correlated effective gains do not establish true physical parameters"
    )
    for label, episode in (("fit", fit_episode), ("val", val_episode)):
        t, obs, pred_before = replay(model, episode, before)
        _, _, pred_after = replay(model, episode, final)
        report[label] = {"before": scores(obs, pred_before), "after": scores(obs, pred_after)}
        dump(
            output / f"{label}_replay.json",
            {
                "t": t.tolist(),
                "observed": obs.tolist(),
                "before": pred_before.tolist(),
                "after": pred_after.tolist(),
                "synthetic": episode["synthetic"],
                "partition": label,
            },
        )
    report["validation_improved"] = (
        report["val"]["after"]["mean_rmse_degrees"] < report["val"]["before"]["mean_rmse_degrees"]
    )
    report["status"] = (
        "trajectory-response fit with frozen inertial inputs; inspect model provenance"
    )
    dump(output / "fit_report.json", report)
    return report
