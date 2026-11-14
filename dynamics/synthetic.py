"""Clearly labelled synthetic recovery experiment; never opens hardware."""

import json
from pathlib import Path

import numpy as np

from model import MOTOR_NAMES, ROOT, ServoCalibration, digest, dump, load
from replay import nominal_parameters, rollout


def generate(model_dir, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model = load(model_dir)
    cal = ServoCalibration("follower")
    initial = cal.ticks_to_radians([2011, 2398, 1983, 1779, 3222])
    truth = nominal_parameters()
    truth.update(
        kp=[5, 8, 5, 2, 1], ki=[2, 5, 3, 1, 0.8], kd=[0.1, 0.15, 0.1, 0.05, 0.04], delay_s=0.05
    )
    dump(destination / "synthetic_truth.json", truth)
    for partition, duration, phase in (("fit", 8, 0), ("val", 7, 0.7)):
        output = destination / f"synthetic-{partition}.json"
        if output.exists():
            raise FileExistsError(output)
        ct = np.arange(0, duration + 0.001, 0.1)
        cq = np.tile(initial, (len(ct), 1))
        for j in range(5):
            cq[:, j] += 0.035 * np.sin(2 * np.pi * ct / (2.5 + j * 0.7) + phase) * (1 - np.exp(-ct))
        command_ticks = np.array([cal.radians_to_ticks(q) for q in cq])
        cq = np.array([cal.ticks_to_radians(t) for t in command_ticks])
        times = np.arange(0, duration, 0.05)
        observed = rollout(model, times, initial, ct, cq, truth)
        ticks = np.array([cal.radians_to_ticks(q) for q in observed])
        episode = {
            "schema": 1,
            "synthetic": True,
            "status": "complete",
            "partition": partition,
            "session_id": output.stem,
            "joint_order": list(MOTOR_NAMES),
            "calibration_sha256": cal.fingerprint,
            "alignment_sha256": digest(ROOT / "calibration/kinematics_alignment.json"),
            "current_is_calibrated_torque": False,
            "source": "MuJoCo generated and tick-quantized; NOT physical arm measurements",
            "commands": [{"t": float(t), "ticks": v.tolist()} for t, v in zip(ct, command_ticks)],
            "samples": [{"t": float(t), "ticks": v.tolist()} for t, v in zip(times, ticks)],
        }
        dump(output, episode)
    print(json.dumps({"synthetic": True, "output": str(destination)}))
