"""Generate explicitly synthetic camera fixtures from actual contact-grasp states."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from brickcell.simulation.world import BRICKS, OBJECTS, World, inspection_crop


def generate(root: Path) -> None:
    if root.exists():
        raise ValueError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)
    world = World(root / "scene")
    rng = np.random.default_rng(2026)
    camera_pos = world.model.camera("inspection").pos.copy()
    rows = []
    try:
        for split, episodes in [("train", 6), ("val", 3), ("test", 3)]:
            for episode in range(episodes):
                world.reset()
                for i, object_id in enumerate(OBJECTS):
                    defective = split != "train" and i % 2 == 1
                    world.model.geom(f"{object_id}-mark").rgba[3] = float(defective)
                    world.model.geom(f"{object_id}-solid").rgba[:3] = np.clip(
                        np.array(BRICKS[object_id].rgba) + rng.uniform(-0.015, 0.015, 3), 0, 1
                    )
                    world.model.camera("inspection").pos[:] = camera_pos + rng.uniform(
                        -0.0005, 0.0005, 3
                    )
                    world.action("pick", object_id)
                    world.action("present", object_id)
                    pixels = inspection_crop(world.camera())
                    name = f"render-{split}-{episode}-{i}"
                    relative = f"{split}/{name}.png"
                    path = root / relative
                    path.parent.mkdir(exist_ok=True)
                    Image.fromarray(pixels).save(path)
                    label = "defective" if defective else "good"
                    rows.append(
                        dict(
                            image_id=name,
                            path=relative,
                            split=split,
                            label=label,
                            object_id=name,
                            session_id=f"render-session-{split}-{episode}",
                            view_id="inspection-fixed-roi",
                            synthetic=True,
                            defect_type="synthetic_dark_surface_mark" if defective else None,
                        )
                    )
                    print(name, label, flush=True)
                    # Dataset generation uses known labels for fixture placement only.
                    world.action("move_defective" if defective else "move_good", object_id)
                    world.action("release", object_id)
                    world.action("verify_release", object_id)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "description": "SYNTHETIC MuJoCo camera fixtures; cloned geometry, no real-world accuracy claim. Fixed ROI [48,90,208,220].",
                    "samples": rows,
                },
                indent=2,
            )
        )
    finally:
        world.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/two-bowl"))
    generate(parser.parse_args().output)
