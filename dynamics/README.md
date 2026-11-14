# SO-101 inertial model and trajectory identification

Optional Python 3.12 environment: MuJoCo **4.0.0**, Frax **0.2.0**, CPU JAX,
and exact versions in uv.lock. The anomaly-detector environment is untouched.

## Published mass and inertia baseline

The workflow imports The Robot Studio's SO101 CAD inertials from commit
a758567c3978dfeefe282ede0500085a48fe8f78, verifies the URDF SHA-256,
and caches source models/STLs under ignored artifacts/dynamics/upstream/.
Apache-2.0 license and source provenance are retained there.

| Physical rigid group | Published mass |
|---|---:|
| Base | 147 g |
| Shoulder | 100.006 g |
| Upper arm | 103 g |
| Lower arm | 104 g |
| Wrist | 79 g |
| Gripper plus jaw | 99 g |
| Total including base | 632.006 g |

These are **published CAD nominals**, not new weighings of this arm.
published.py registers each rigid group's geometry to the existing calibrated
tree, constraining the incoming joint axis and verifying outgoing axes.
It transforms COMs and complete 3x3 tensors, and combines jaw inertia with
the parallel-axis theorem. Fixed subpart mass is consolidated into its owner.
The input JSON retains transforms, registration residuals, source names and
hashes. The local kinematic calibration is preserved. Jaw is fixed at source
zero for the five-axis dynamics model.

MuJoCo Menagerie SO101 commit 370c43965323c0944b433b202aac5058b5f907a9 supplies
explicit simulation defaults: rotor armature 0.028 kg m², joint damping 0.60,
frictionloss 0.052 and torque cap 2.94 Nm. These are not individually measured
motor properties. BAM's available STS3215 model uses 7.4 V test conditions;
motor/feedback equivalence to this 12 V follower was not established.

## Reproduce on Intel

Run from dynamics/ in /home/intel/Desktop/tarot-club-junction:

    uv sync --locked
    export JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1
    export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED'
    uv run --locked python cli.py published --source ../artifacts/dynamics/upstream --output ../artifacts/dynamics/published_priors.json
    uv run --locked python cli.py build --priors ../artifacts/dynamics/published_priors.json --output ../artifacts/dynamics/model-published
    uv run --locked python cli.py check --model ../artifacts/dynamics/model-published --output ../artifacts/dynamics/physics-checks.json
    uv run --locked pytest

The CLI requires explicit inertial inputs. Internal heuristic new_priors()
exists only for offline test/development fixtures. To enter later measurements,
copy a published input JSON, update mass, COM and inertia_matrix_kg_m2 in the
stated local frame, and record method/uncertainty in source. Rebuild into a
new directory. Positive principal inertias and triangle inequalities are checked.

## Record real trajectories

Read ../hardware.md, ../ssh.md and ../arm-api.md first. Only armd owns the port.
api_for_capture.py runs the existing API in the separate armd environment,
using its addressed-read fallback. It changes neither deployed armd source
nor firmware protections. Store a fresh read-only PID/offset snapshot first.

The recorder requires explicit --hardware and separate session paths/partitions.
The final requested sweep is **40 degrees peak-to-peak** (±20 degrees), one
of five joints at a time; gripper stays fixed. Default command speed is the
API's configured nonzero value (observed 300 ticks/s). No extra speed, current,
load, voltage or temperature cutoffs are added by the recorder, per the user.
Existing API motion permission, clamping and blocking servo faults remain.
No firmware protection registers are modified.

Position bounds, available/fresh position data, actual goal readbacks, fixed
duration and completion tolerance are checked. Every exception/completion
requests hold; torque-off is never issued by capture.py. Communication loss
cannot provide an independent physical stop. No hardware watchdog or emergency
stop is implemented here.

    uv run --locked python capture.py --hardware --partition fit --output ../artifacts/dynamics/fit-session-10.json --pid-snapshot ../artifacts/dynamics/pid_snapshot-02.json --amplitude 20 --period 12
    uv run --locked python capture.py --hardware --partition val --output ../artifacts/dynamics/val-session-01.json --pid-snapshot ../artifacts/dynamics/pid_snapshot-02.json --amplitude 20 --period 10

Existing recordings are never overwritten. Commands are logged around 10 Hz
and fresh telemetry around 20 Hz using monotonic request-midpoint timestamps.
Joint reads are sequential, not simultaneous hardware timestamps. Current/load
remain diagnostic channels, not calibrated output torque.

## Fit and phantom

    uv run --locked python cli.py fit --model ../artifacts/dynamics/model-published-verified --fit ../artifacts/dynamics/fit-session-10.json --val ../artifacts/dynamics/val-session-01.json --output ../artifacts/dynamics/published-fit --max-iters 30
    uv run --locked python cli.py viewer --model ../artifacts/dynamics/model-published-verified --episode ../artifacts/dynamics/val-session-01.json --report ../artifacts/dynamics/published-fit/fit_report.json --output ../artifacts/dynamics/phantom-published.html

MuJoCo sysid.optimize uses bounded SciPy least squares and a full forward-
dynamics residual. Link masses/COMs/tensors stay fixed at sourced values while
effective actuator P/I/D and a coarse command-delay grid are fitted using
training data only. This is a **CAD-anchored inertial model with motion-calibrated
actuators**, not a claim that position logs uniquely identified this unit's
masses/tensors. SI gains differ from firmware coefficients and are never sent
to hardware. Several gains reach bounds; they are not unique motor constants.

Native affine P/D feedback runs every 1 ms physics step; the clamped integral
and reference update run at 100 Hz. Initial integral is estimated from gravity
equilibrium. A regression test prevents artificial wrist oscillations caused
by holding explicit derivative torque for 10 ms in an earlier development
model. Recorded positions initialize each rollout once and never overwrite
the simulated phantom mid-rollout.

Frax independently checks mass matrices and gravity vectors. Its fixed 9.81
gravity is rescaled to 9.80665; MuJoCo armature is removed before rigid-body
matrix comparison. Implicitfast/Newton, timestep-halving and numerical fixture
checks verify implementation behavior. **Contact identification is outside
this task**, as requested; floor/collision settings remain a baseline.

The HTML autoplays recorded encoder FK against an independently simulated
phantom. Play/pause/scrub, orbit and choose initial/intermediate/fitted stages.
Expand inertial inputs to inspect masses, COMs and full tensors. No server or
hardware controls are in the page. Three.js loads from pinned CDN URLs.

See ../docs/VALIDATION.md for executed evidence and limitations, and
../docs/SYSTEM_IDENTIFICATION_RESEARCH.md for primary sources.