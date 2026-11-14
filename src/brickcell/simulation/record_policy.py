"""Record synthetic teacher demonstrations; this is not human teleoperation."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from brickcell.policy.data import CAMERAS, JOINTS, file_hash, validate_episode
from brickcell.simulation.world import BRICKS, OBJECTS, World


def prepare(world: World, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i, name in enumerate(OBJECTS):
        a = world.model.joint(f"{name}-joint").qposadr[0]
        z = BRICKS[name].grip_z  # body origin is the grasp point
        world.data.qpos[a : a + 3] = (
            [0.25 + rng.uniform(-0.004, 0.004), rng.uniform(-0.09, 0.09), z]
            if i == 0
            else [2 + i, 0, z]
        )
    mujoco.mj_forward(world.model, world.data)


def observe(world: World) -> dict:
    joints = [world.model.joint(name).qposadr[0] for name in JOINTS]
    return {
        "state": world.data.qpos[joints].astype(np.float32).copy(),
        **{
            name: np.asarray(
                Image.fromarray(world.camera(name)).resize((128, 128), Image.Resampling.BILINEAR)
            ).copy()
            for name in CAMERAS
        },
    }


def record(output: Path, episodes: int, seed: int) -> None:
    if episodes < 3 or output.exists():
        raise ValueError("Use >=3 episodes and a new output directory")
    output.mkdir(parents=True)
    rows = []
    for index in range(episodes):
        world = World(output / "scene")
        prepare(world, seed + index)
        records = []

        def capture():
            records.append(
                {
                    **observe(world),
                    "action": world.data.ctrl[world.actuators].astype(np.float32).copy(),
                    "timestamp": float(world.data.time),
                }
            )

        world.before_step = capture
        try:
            world.action("pick", "part-001")
            world.action("present", "part-001")
            if world.position("part-001")[2] < 0.10:
                raise RuntimeError("Teacher lost its grasp before presentation")
            arrays = {key: np.stack([r[key] for r in records]) for key in records[0]}
            count = validate_episode(arrays)
            name = f"episode-{index:04d}"
            path = output / f"{name}.npz"
            np.savez_compressed(path, **arrays)
            split = "train" if index < episodes - 2 else "val" if index == episodes - 2 else "test"
            rows.append(
                dict(
                    episode_id=name,
                    path=path.name,
                    seed=seed + index,
                    split=split,
                    frames=count,
                    success=True,
                    sha256=file_hash(path),
                )
            )
            print(f"{name}: {count} synchronized frames, split={split}", flush=True)
        finally:
            world.close()
    metadata = dict(
        schema_version=1,
        synthetic=True,
        teacher="scripted_IK_contact_physics",
        human_demonstrations=False,
        task="pick one separated brick and hold at inspection",
        shape_family=f"lego_{BRICKS['part-001'].mesh}",
        fps=50,
        joint_names=JOINTS,
        action_units="radians",
        action_semantics="absolute_joint_position_targets_before_physics_step",
        camera_names=CAMERAS,
        image_format="RGB_uint8_128x128",
        episodes=rows,
        versions={
            n: importlib.metadata.version(n)
            for n in ("mujoco", "physicalai-mujoco-plugin", "numpy", "pillow")
        },
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/policy-smoke"))
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=4200)
    args = parser.parse_args()
    record(args.output, args.episodes, args.seed)
