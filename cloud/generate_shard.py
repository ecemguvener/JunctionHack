"""Render one shard of the PatchCore dataset (a few episodes) for parallel generation on Modal.

brickcell.simulation.generate renders all 12 episodes in one process with one random
generator, so it cannot be split. This wrapper repeats its per-episode loop body with
one random generator per episode: numpy default_rng([2026, split index, episode]). The
result does not depend on the shard layout. Image names, labels, splits, session IDs,
the defect rule and the manifest format are the same as in generate.py. The jitter values
are different from a single local run (same ranges, different random draws).

If generate.py changes, this copy can become wrong. The wrapper compares the SHA-256 of
generate.py with GENERATE_SHA256 and stops if they differ. Then update this file (or use
the single-container generator: `--no-sharded`).

Usage:
  python cloud/generate_shard.py render --output DIR --units train:0 train:1 ...
  python cloud/generate_shard.py merge --output DIR SHARD_DIR [SHARD_DIR ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from modal_helpers import GENERATE_SPLITS, generate_units

# SHA-256 of src/brickcell/simulation/generate.py that this wrapper copies.
GENERATE_SHA256 = "ce6b7ac21eeb7f2dbe4580c30d3d5d4acbb7549a7abf4a5c0622d491cf5d3535"
SPLITS = GENERATE_SPLITS
DESCRIPTION = (
    "SYNTHETIC MuJoCo camera fixtures; cloned geometry, no real-world accuracy claim. "
    "Fixed ROI [48,90,208,220]."
)


def all_units() -> list[tuple[str, int]]:
    return [(unit.split(":")[0], int(unit.split(":")[1])) for unit in generate_units()]


def check_generator(allow_drift: bool = False) -> str:
    import brickcell.simulation.generate as generate

    digest = hashlib.sha256(Path(generate.__file__).read_bytes()).hexdigest()
    if digest != GENERATE_SHA256 and not allow_drift:
        raise SystemExit(
            f"generate.py changed (sha256 {digest}). Update cloud/generate_shard.py to match it, "
            "or run the single-container generator (--no-sharded)."
        )
    return digest


def render(output: Path, units: list[tuple[str, int]], scene_config: Path | None) -> dict:
    import numpy as np
    from PIL import Image

    from brickcell.simulation.world import OBJECTS, World, inspection_crop

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    world = World(output / "scene", scene_config)
    setup_seconds = time.perf_counter() - started
    camera_pos = world.model.camera("inspection").pos.copy()
    split_index = {name: index for index, (name, _) in enumerate(SPLITS)}
    rows, render_seconds = [], 0.0
    try:
        for split, episode in units:
            rng = np.random.default_rng([2026, split_index[split], episode])
            world.reset()
            for i, object_id in enumerate(OBJECTS):
                defective = split != "train" and i % 2 == 1
                world.model.geom(f"{object_id}-mark").rgba[3] = float(defective)
                world.model.geom(f"{object_id}-solid").rgba[:3] = np.clip(
                    np.array(world.bricks[object_id].rgba) + rng.uniform(-0.015, 0.015, 3), 0, 1
                )
                world.model.camera("inspection").pos[:] = camera_pos + rng.uniform(
                    -0.0005, 0.0005, 3
                )
                world.action("pick", object_id)
                world.action("present", object_id)
                tick = time.perf_counter()
                pixels = inspection_crop(world.camera())
                render_seconds += time.perf_counter() - tick
                name = f"render-{split}-{episode}-{i}"
                relative = f"{split}/{name}.png"
                path = output / relative
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
                world.action("move_defective" if defective else "move_good", object_id)
                world.action("release", object_id)
                world.action("verify_release", object_id)
    finally:
        world.close()
    timing = {
        "units": [f"{s}:{e}" for s, e in units],
        "images": len(rows),
        "world_setup_seconds": round(setup_seconds, 3),
        "render_seconds": round(render_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    (output / "rows.json").write_text(json.dumps({"samples": rows, "timing": timing}, indent=2))
    return timing


def merge(output: Path, shards: list[Path]) -> dict:
    """Join shard outputs into one dataset with the row order of generate.py."""
    order = {f"{s}:{e}": n for n, (s, e) in enumerate(all_units())}
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    for shard in shards:
        data = json.loads((shard / "rows.json").read_text())
        for row in data["samples"]:
            target = output / row["path"]
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(shard / row["path"], target)
            rows.append(row)

    def key(row):
        _, split, episode, index = row["image_id"].split("-")
        return order[f"{split}:{episode}"], int(index)

    rows.sort(key=key)
    expected = {f"{s}:{e}" for s, e in all_units()}
    found = {"{}:{}".format(*r["image_id"].split("-")[1:3]) for r in rows}
    if found != expected or len({r["image_id"] for r in rows}) != len(rows):
        raise SystemExit(f"Shards are incomplete or overlap: missing {sorted(expected - found)}")
    if shards:
        shutil.copytree(shards[0] / "scene", output / "scene")
    manifest = {"schema_version": 1, "description": DESCRIPTION, "samples": rows}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"images": len(rows), "shards": len(shards)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render_cmd = sub.add_parser("render")
    render_cmd.add_argument("--output", type=Path, required=True)
    render_cmd.add_argument("--units", nargs="+", required=True, help="split:episode")
    render_cmd.add_argument("--scene-config", type=Path)
    render_cmd.add_argument("--allow-generator-drift", action="store_true")
    merge_cmd = sub.add_parser("merge")
    merge_cmd.add_argument("--output", type=Path, required=True)
    merge_cmd.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "render":
        digest = check_generator(args.allow_generator_drift)
        units = [(u.split(":")[0], int(u.split(":")[1])) for u in args.units]
        timing = render(args.output, units, args.scene_config)
        print(json.dumps({"generate_sha256": digest, **timing}))
    else:
        print(json.dumps(merge(args.output, args.shards)))


if __name__ == "__main__":
    main()
