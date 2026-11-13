"""Bounded, teacher-free simulated ACT rollouts; successful training is not task success."""

import argparse
import base64
import json
import os
import select
import subprocess
from pathlib import Path

import numpy as np
import torch
from train_act import load_policy


def evaluate(args):
    from physicalai.data import Observation

    torch.set_num_threads(4)
    policy, metadata = load_policy(args.checkpoint)
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "MENAGERIE_CACHE_DIR": str(Path(".cache/menagerie").absolute()),
    }
    process = subprocess.Popen(
        [
            str(Path.cwd() / "artifacts/mujoco-venv/bin/python"),
            "-m",
            "brickcell.simulation.policy_worker",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )

    def exchange(message):
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], 30)
        if not ready:
            raise TimeoutError("Simulator response exceeded 30s")
        result = json.loads(process.stdout.readline())
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    episodes = []
    try:
        for seed in args.seeds:
            success, stable, error = False, 0, None
            policy.reset()
            observation = exchange(
                {"command": "reset", "seed": seed, "directory": str(args.output.parent / "scene")}
            )
            steps = 0
            clipped_values = 0
            try:
                for _ in range(args.max_steps // 4):
                    batch = Observation(
                        state=torch.tensor([observation["state"]], dtype=torch.float32),
                        images={
                            k: torch.from_numpy(
                                np.frombuffer(base64.b64decode(v), np.uint8)
                                .copy()
                                .reshape(128, 128, 3)
                                .transpose(2, 0, 1)
                            ).float()[None]
                            / 255
                            for k, v in observation["images"].items()
                        },
                    )
                    with torch.inference_mode():
                        actions = policy.predict_action_chunk(batch)[0, :4].numpy()
                    observation = exchange(
                        {
                            "command": "step",
                            "actions": actions.tolist(),
                            "clip_to_sim_limits": args.clip_to_sim_limits,
                        }
                    )
                    clipped_values += observation["clipped_values"]
                    steps += 4
                    stable = stable + 4 if observation["simulated_success"] else 0
                    if stable >= 25:
                        success = True
                        break
            except Exception as exc:
                error = str(exc)
            episodes.append(
                dict(
                    seed=seed,
                    success=success,
                    steps=steps,
                    error=error,
                    clipped_values=clipped_values,
                )
            )
            print(episodes[-1], flush=True)
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    report = dict(
        synthetic=True,
        policy="physicalai_ACT",
        training_steps=metadata["training_steps"],
        scripted_ik_during_rollout=False,
        clip_to_sim_limits=args.clip_to_sim_limits,
        physical_success_verified=False,
        successful_episodes=sum(e["success"] for e in episodes),
        episodes=episodes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/policy/act-smoke/policy.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/policy/act-smoke/rollouts.json")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[5200, 5201])
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--clip-to-sim-limits",
        action="store_true",
        help="Explicit simulation-only action saturation; counts every clipped value",
    )
    print(json.dumps(evaluate(parser.parse_args()), indent=2))
