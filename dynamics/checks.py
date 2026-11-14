"""Numerical dynamics/physics checks; these do not calibrate physical contacts."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import jax
import mujoco
import numpy as np
from frax import Manipulator

from model import ARM_JOINTS, CalibratedSO101Kinematics, ServoCalibration, dump, indices, load
from replay import nominal_parameters, rollout


def physics_fixture_checks():
    xml = """<mujoco><option timestep=".001" gravity="0 0 -9.80665"
        integrator="implicitfast" solver="Newton" iterations="100" tolerance="1e-10"/>
        <worldbody><geom type="plane" size="2 2 .1" friction=".7 .005 .0001"
        solref=".01 1" solimp=".95 .99 .001"/>
        <body pos="0 0 .5"><freejoint/><geom type="sphere" size=".025" mass=".1"
        friction=".7 .005 .0001" solref=".01 1" solimp=".95 .99 .001"/></body>
        </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, nstep=100)
    freefall_error = abs(d.qpos[2] - (0.5 - 0.5 * 9.80665 * 0.1**2))
    assert freefall_error < 0.001
    mujoco.mj_step(m, d, nstep=1900)
    assert d.ncon > 0 and abs(d.qpos[2] - 0.025) < 0.001
    assert abs(d.qvel[2]) < 0.01
    return {
        "freefall_error_m": float(freefall_error),
        "settled_sphere_height_m": float(d.qpos[2]),
        "contact_count": int(d.ncon),
        "meaning": "engine numerical fixture only; arm/table contacts remain unmeasured",
    }


def run(model_dir, output):
    jax.config.update("jax_enable_x64", True)
    m = load(model_dir)
    qi, vi = indices(m)
    robot = Manipulator(str(Path(model_dir) / "arm.urdf"), joint_ordering=list(ARM_JOINTS))
    geometric = CalibratedSO101Kinematics("follower")
    cal = ServoCalibration("follower")
    q0 = cal.ticks_to_radians([2011, 2398, 1983, 1779, 3222])
    rng = np.random.default_rng(101)
    cases = [q0, *[q0 + rng.uniform(-0.08, 0.08, 5) for _ in range(4)]]
    mass_error = gravity_error = fk_error = 0.0
    minimum_eigenvalue = float("inf")
    for q in cases:
        d = mujoco.MjData(m)
        d.qpos[qi] = q
        mujoco.mj_forward(m, d)
        full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, full)
        rigid_mass = full[np.ix_(vi, vi)] - np.diag(m.dof_armature[vi])
        fm = np.asarray(robot.mass_matrix(q))
        fg = np.asarray(robot.gravity_vector(q)) * (9.80665 / 9.81)
        mass_error = max(mass_error, float(np.max(np.abs(fm - rigid_mass))))
        gravity_error = max(gravity_error, float(np.max(np.abs(fg - d.qfrc_bias[vi]))))
        minimum_eigenvalue = min(minimum_eigenvalue, float(np.linalg.eigvalsh(rigid_mass).min()))
        wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Body3")
        fk_error = max(fk_error, float(np.max(np.abs(d.xpos[wrist] - geometric.fk(q)[:3, 3]))))
    assert mass_error < 2e-8 and gravity_error < 2e-8 and fk_error < 2e-8
    assert minimum_eigenvalue > 0
    times = np.arange(0, 3, 0.02)
    control = np.tile(q0, (len(times), 1))
    control[:, 2] += 0.02 * np.sin(2 * np.pi * times / 3)
    p = nominal_parameters()
    coarse = rollout(m, times, q0, times, control, p, timestep=0.001)
    fine = rollout(m, times, q0, times, control, p, timestep=0.0005)
    convergence = float(np.max(np.abs(coarse - fine)))
    assert convergence < 0.003
    report = {
        "mass_matrix_max_error": mass_error,
        "gravity_vector_max_error": gravity_error,
        "fk_position_max_error_m": fk_error,
        "minimum_mass_matrix_eigenvalue": minimum_eigenvalue,
        "frax_gravity_rescale": "Frax 0.2 fixes gravity at 9.81; scale gravity vector by 9.80665/9.81",
        "timestep_halving_max_difference_rad": convergence,
        "cases": len(cases),
        "physics_fixture": physics_fixture_checks(),
        "physical_masses_validated": False,
        "contacts_calibrated": False,
        "versions": {
            n: importlib.metadata.version(n) for n in ("mujoco", "frax", "jax", "jaxlib", "numpy")
        },
    }
    dump(output, report)
    return report
