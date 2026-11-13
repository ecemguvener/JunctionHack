"""Private stdin/stdout simulation worker; no hardware or IK during policy stepping."""

import base64
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from brickcell.simulation.record_policy import observe, prepare
from brickcell.simulation.world import INSPECTION, World


def main():
    world = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            clipped_values = 0
            if message["command"] == "reset":
                if world is not None:
                    world.close()
                world = World(Path(message["directory"]))
                prepare(world, int(message["seed"]))
            elif message["command"] == "step" and world is not None:
                targets = np.asarray(message["actions"], dtype=float)
                limits = world.model.actuator_ctrlrange[world.actuators]
                if (
                    targets.ndim != 2
                    or targets.shape[1] != 6
                    or not 1 <= len(targets) <= 16
                    or not np.isfinite(targets).all()
                ):
                    raise ValueError("Expected 1–16 finite six-joint actions")
                if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
                    clipped_values = int(
                        ((targets < limits[:, 0]) | (targets > limits[:, 1])).sum()
                    )
                    if not message.get("clip_to_sim_limits", False):
                        raise ValueError(
                            f"Predicted action exceeds simulator actuator limits ({clipped_values} values)"
                        )
                    targets = np.clip(targets, limits[:, 0], limits[:, 1])
                for target in targets:
                    world.data.ctrl[world.actuators] = target
                    # Deliberately bypass World.tick(): it invokes the scripted IK teacher.
                    mujoco.mj_step(world.model, world.data, nstep=10)
            else:
                raise ValueError("Reset before stepping")
            observation = observe(world)
            result = {
                "state": observation["state"].tolist(),
                "images": {
                    k: base64.b64encode(observation[k].tobytes()).decode()
                    for k in ("overview", "wrist")
                },
                "time": float(world.data.time),
                "clipped_values": clipped_values,
                "simulated_success": bool(
                    world.position("part-001")[2] > 0.10
                    and np.linalg.norm(world.ik.tcp(world.data) - INSPECTION) < 0.02
                ),
            }
        except Exception as exc:
            result = {"error": str(exc)}
        print(json.dumps(result), flush=True)
    if world is not None:
        world.close()


if __name__ == "__main__":
    main()
