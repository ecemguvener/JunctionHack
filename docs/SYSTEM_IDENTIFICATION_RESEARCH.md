# SO-101 system identification: primary-source findings

Researched 2026-11-14. Documentation/source inspection only: no hardware connection, package installation, fitting run, or native API execution. Recommendations below are proposed implementation stages, not measured SO-101 performance. Keep this work in the optional simulation environment, separate from the anomalib core.

## Recommendation and likely remembered projects

### Published inertials retrieved during implementation

The official model contains link masses, COMs and full inertia tensors:
[Robot Studio SO101 URDF at commit a758567](https://github.com/TheRobotStudio/SO-ARM100/blob/a758567c3978dfeefe282ede0500085a48fe8f78/Simulation/SO101/so101_new_calib.urdf).
With the jaw consolidated into the gripper, its six physical rigid groups
total 0.632006 kg including the base. These are CAD nominal properties,
not new weighings of this unit. The implementation now registers/transforms
them into the calibrated local frames instead of using bounding-box guesses.

[BAM: Better Actuator Models](https://github.com/Rhoban/bam) is a strong
additional match to the remembered project. Its [documentation](https://bam.readthedocs.io/en/latest/)
shows comparative simulation and trajectory-based identification. Retrieved
[STS3215 M6 parameters](https://github.com/Rhoban/bam/blob/b9e11118e17fe8ec803d430202f6fcf31d57901e/bam/params/feetech_sts3215_7_4V/m6.json)
include kt=1.2753202369 Nm/A, R=2.7534212144 ohm and apparent rotor inertia
0.0215642901 kg m². Its [actuator implementation](https://github.com/Rhoban/bam/blob/b9e11118e17fe8ec803d430202f6fcf31d57901e/bam/feetech/actuator.py)
states 7.4 V conditions and a rate-limited internal target. Motor-variant and
feedback equivalence to this 12 V follower was not established; these are
reference values, not a measured torque calibration of this arm.

[Menagerie's pinned SO101 model](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101/so101.xml)
provides explicit simulation assumptions for armature, damping/friction and
torque limit. Existing real trajectories calibrate effective actuator response
with the CAD inertials frozen. Contact identification was deprioritized by
the user. This anchors the model to published inertial data; it does not uniquely
identify new per-unit masses from uncalibrated current feedback.

Use **MuJoCo 4.0.0's native `mujoco.sysid`** for command-replay fitting; use **StanfordASL Frax 0.2.0** for independent rigid-body calculations and later sensitivity analysis. Do not start by fitting every link's full inertia and every servo parameter simultaneously.

| Candidate | Verified primary evidence and match to the memory |
|---|---|
| **MuJoCo native sysid — strongest functional match** | Fits simulation outputs to recorded controls/sensors with bounded nonlinear least squares, finite-difference Gauss–Newton, and parallel CPU rollouts. Supports mass/CoM/inertia, gains, friction and measurement corrections. [4.0.0 toolbox README](https://github.com/google-deepmind/mujoco/blob/4.0.0/python/mujoco/sysid/README.md). |
| **MuJoCo Playground SO-101 — strongest robot-specific match** | A REAF-hosted Playground guide describes `tasks/sysid/so101`, replaying MCAP teleoperation logs to fit six joints' armature, frictionloss, stiffness and damping; report compares trajectories. YAM example also reports held-out RMSE and parameters to freeze. [REAF-hosted guide](https://reaf-sim.readthedocs.io/en/latest/playground/sysid.html), [repository](https://github.com/google-deepmind/mujoco_playground). Do not confuse its joint stiffness field with an actuator's `kp` without inspecting the model. |
| **IsaacLab-Newton — strong alternative-engine candidate** | CMA-ES evaluates candidates in parallel, replays real commands and defaults to time-aligned position MSE. Includes SO-101 support; its backend is Newton/MuJoCo Warp, rather than the native CPU sysid API. [Sysid README](https://github.com/es-rl/IsaacLab-Newton/blob/develop/scripts/sysid/README.md), [held-out replay guide](https://github.com/es-rl/IsaacLab-Newton/blob/develop/scripts/sim2real_gap/README.md). Useful workflow reference; not a drop-in Mac baseline. |
| **ToddlerBot** | Explicit Dynamixel sysid procedure, real excitation recordings and simulation optimization; plausible if the remembered demo involved a small servo mechanism. Different motors/robot. [Official procedure](https://github.com/hshi74/toddlerbot/blob/main/docs/tools/02_sysID.rst). |
| **SimOpt (Chebotar et al.)** | Iteratively adapts simulation randomization from real experience. Plausible real-versus-sim video memory, but a distribution-adaptation/RL workflow, rather than evidence of a turnkey SO-101 least-squares tool. The authors' page links paper/video; a canonical implementation repository was not verified. [Project](https://sites.google.com/view/simopt), [paper](https://arxiv.org/abs/1810.05687). |
| **ASAP** | Learns a delta-action model from real trajectories, then fine-tunes humanoid policies. Demonstrates IsaacGym→Genesis as well as real transfer. It does not identify interpretable link mass/inertia as its central mechanism. [Authors' repository](https://github.com/LeCAR-Lab/ASAP), [paper](https://arxiv.org/abs/2502.01143). |
| **robot_descriptions / other sysid** | `robot_descriptions.py` is a model acquisition/loading library; its documented purpose does not establish a fitting toolbox. [Repository](https://github.com/robot-descriptions/robot_descriptions.py). A separately named [JunningHuang/system-identification](https://github.com/JunningHuang/system-identification) advertises a robot benchmark and excitation scripts, but its sparse README does not establish the remembered overlay workflow. |
| **Genesis** | No canonical Genesis rigid-arm sysid example was verified in this search. [gym-genesis](https://github.com/huggingface/gym-genesis) establishes a Genesis environment, not evidence of the desired fitter. ASAP's Genesis transfer is verified above. Treat “Genesis system identification” as an unresolved candidate, not a confirmed package/API. |

None of the verified sources establishes “phantom” as the project's actual name.

## Native MuJoCo 4.0.0 API: concrete implementation starting point

The **release tag**, not just current main, contains the toolbox. Its initializer labels sysid an optional extra (`mujoco[sysid]`) and catches missing optional imports, so successful `from mujoco import sysid` alone is insufficient: also check required attributes and the installed version in the isolated environment. [4.0.0 exports](https://github.com/google-deepmind/mujoco/blob/4.0.0/python/mujoco/sysid/__init__.py).

Proposed offline skeleton (not executed). `t`, `u`, `q_measured`, `q0`, `v0` are prepared arrays; the XML must define a position actuator `elbow_act` and a named `jointpos` sensor `elbow_encoder`. The nominal/bounds below are illustrative, not SO-101 calibration values:

```python
import mujoco
from mujoco import sysid

assert mujoco.__version__ == "4.0.0"
spec = mujoco.MjSpec.from_file("so101.xml")
model = spec.compile()

def set_pd(spec, p):
    sysid.apply_pdgain(spec, "elbow_act", p.value)

params = sysid.ParameterDict()
params.add(sysid.Parameter(
    "elbow_pd", nominal=[10.0, 1.0],
    min_value=[0.1, 0.01], max_value=[100.0, 10.0], modifier=set_pd,
))
control = sysid.TimeSeries.from_control_names(
    t, u, model, names=["elbow_act"],
)
measured = sysid.TimeSeries.from_names(
    t, q_measured, model, names=["elbow_encoder"],
)
initial = sysid.create_initial_state(model, q0, v0)
episode = sysid.ModelSequences(
    "so101", spec, "fit_episode", initial, control, measured,
)
residual_fn = sysid.build_residual_fn(models_sequences=[episode])
fitted, result = sysid.optimize(initial_params=params, residual_fn=residual_fn)
sysid.save_results("results/", [episode], params, fitted, result, residual_fn)
```

Pipeline follows the [tagged example](https://github.com/google-deepmind/mujoco/blob/4.0.0/python/mujoco/sysid/README.md). Verify array shapes/order and explicitly name every actuator/sensor; default observation mapping is **all model sensors in sensor-address order**, not automatically encoder positions. `create_initial_state` constructs `mjSTATE_FULLPHYSICS`; omitted velocity/activation defaults to zero. Multiple sequences can share one specification. [TimeSeries source](https://github.com/google-deepmind/mujoco/blob/4.0.0/python/mujoco/sysid/_src/timeseries.py), [trajectory source](https://github.com/google-deepmind/mujoco/blob/4.0.0/python/mujoco/sysid/_src/trajectory.py).

`apply_pdgain(spec, actuator_name, value)` expects `[P_gain, D_gain]`. For inertia use `body_inertia_param(spec, model, body_name, inertia_type=...)`: `Mass`, `MassIpos`, or physically consistent `Pseudo`. Inspect `scale_rot_inertia` deliberately rather than changing mass while accidentally retaining an inappropriate inertia. [Modifier implementation](https://github.com/google-deepmind/mujoco/blob/4.0.0/python/mujoco/sysid/_src/model_modifier.py). The [tagged tutorial notebook](https://github.com/google-deepmind/mujoco/blob/4.0.0/python/mujoco/sysid/sysid.ipynb) is the next implementation reference; measurement correction APIs are explicitly described as not final in the README.

## What the available telemetry can identify

**Engineering inference:** replay the actual transmitted position targets into a simulated servo/controller; compare simulated encoders against measured encoders. Do not overwrite simulated `qpos` with measured positions at every step: that removes the forward-dynamics evidence being fitted. Include command timing, internal trajectory shaping, limits, deadband and integral behavior when significant. A MuJoCo PD actuator is an effective approximation, not a verified reconstruction of Feetech firmware. The SO-101 follower uses geared STS3215 servos; confirm the exact installed variant/settings. [LeRobot assembly reference](https://huggingface.co/docs/lerobot/main/en/assemble_so101), [MuJoCo actuator equations](https://mujoco.readthedocs.io/en/4.0.0/computation/index.html#actuation-model).

Position-only data can constrain **effective** delay, bandwidth, damping ratio, gain/inertia ratios, reversal/deadband behavior and friction effects when the trajectory excites them. It does not guarantee unique physical parameters. For example, scaling all inertial/gravity terms, friction, and unknown actuator torque gains by the same positive factor leaves the same position evolution in the unconstrained equations. Absolute scale needs an anchor such as independently measured mass/inertia or calibrated output torque/known load. This scale ambiguity is an inference from the [equations of motion](https://mujoco.readthedocs.io/en/4.0.0/computation/index.html#general-framework); even with measured torque, only certain inertial combinations are observable, as shown by [Wensing et al.'s observability analysis](https://journals.sagepub.com/doi/10.1177/02783649241258215).

Current telemetry does **not** become joint torque merely by relabeling it. The manufacturer lists current and load feedback but does not supply a calibrated joint-output torque transfer function in the cited [STS3215 datasheet](https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf). Treat current as a diagnostic channel until its units, sign, bias, filtering, torque constant, gearing and losses are established. Simultaneously fitting an unknown current→torque gain and all masses retains scale ambiguity. Joint encoders also cannot reveal tool deflection or absolute TCP error without external observations. These are identification limitations, not a reason to discard command-replay fitting.

## Proposed fitting stages and contact limits

1. **Establish the measurement model.** Verify joint order, sign, units, zero, timestamps and command hold/resampling convention. Record firmware gains, voltage, temperature and payload alongside logs; hash inputs/model and preserve exact optional-environment versions. Split complete capture sessions into fitting, validation and a frozen final test.
2. **Fit servo timing and response first.** Hold measured/CAD inertias fixed; use contact-free, bounded single-joint excitation at multiple rates/amplitudes/directions. Fit delay separately or on a coarse grid, then effective PD, damping/friction and armature in small groups. Keep unsupported parameters fixed. Compare full rollouts, not just one-step residuals.
3. **Fit coupled mechanics only with evidence.** Use varied configurations, coupled multi-joint excitation and known payloads; fit a small physically consistent subset of mass/CoM/inertia with tight priors. Inspect Jacobian singular values, correlations, bound hits and variation across restarts/sessions. Low residual or optimizer confidence intervals alone do not prove physical uniqueness. These stages follow from the actuator equations/observability sources above; Playground's held-out/freeze workflow provides a concrete precedent.
4. **Calibrate contacts separately.** Contact-free logs contain no information about object/table friction or contact compliance. Use dedicated known-object sliding, onset-of-slip, loading and impact observations, ideally external pose and calibrated force. Sticking alone generally supplies friction bounds, not a unique coefficient. Geometry, normal force, servo effort limits and friction can compensate for one another. Fit contact pairs and selected `solref`/`solimp` only after the arm response is stable; freeze timestep/integrator/solver settings. MuJoCo contact softness and friction dimensions have engine-specific meanings; fitted values are not automatically material constants. [Contact model](https://mujoco.readthedocs.io/en/4.0.0/computation/index.html#contact), [solver parameters](https://mujoco.readthedocs.io/en/4.0.0/modeling.html#solver-parameters).
5. **Validate prediction before deployment.** Report held-out per-joint tracking RMSE/max error, lag and reversal errors across payloads/speeds, plus parameter stability. Reject divergent/failed rollouts. Freeze settings before final test. Grasp/contact success needs separate physical evidence; an encoder match does not establish grip force or successful sorting.

## Frax's bounded role

Frax 0.2.0 is a JAX kinematics/dynamics library, not a verified turnkey contact sysid pipeline. Its README demonstrates `robot = frax.Robot("robot.urdf")`, `q = robot.default_configuration`, `robot.mass_matrix(q)` and recommends x64 and outer `jax.jit`. MJCF support is listed as future work; use a consistent URDF and compare frames/joint ordering with MuJoCo. [README](https://github.com/StanfordASL/frax#readme). Version metadata confirms `0.2.0`; core dependencies do not require CUDA, while IK adds its own newer JAX requirement. [Package metadata](https://github.com/StanfordASL/frax/blob/main/pyproject.toml).

Recommended first use: cross-check mass matrices/gravity and build offline sensitivity tools. Do not assume JAX differentiation through configurations also exposes a supported trainable inertial-parameter API, or that Frax reproduces MuJoCo contacts/servo dynamics. Native MuJoCo sysid already uses finite differences and does not require a differentiable Frax rollout. No cross-library numerical compatibility or native performance was tested here.
