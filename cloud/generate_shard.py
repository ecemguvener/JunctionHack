"""Render one shard of the PatchCore dataset (a few episodes) for parallel generation on Modal.

brickcell.simulation.generate renders all episodes in one process with one random generator,
so its single run cannot be split. This wrapper calls the same episode function
(generate.render_episode) with one random generator per episode:
numpy default_rng([2026, split index, episode]). The result does not depend on the shard
layout. Image names, labels, splits, session IDs, the defect rule, the palette colours (a
seeded plan that does not use the random generator) and the manifest are the same as in
generate.py. Only the jitter values are different from a single local run (same ranges,
different random draws).

Usage:
  python cloud/generate_shard.py render --output DIR --units train:0 train:1 ... \
      [--palette all] [--episodes 36 27 27]
  python cloud/generate_shard.py merge --output DIR [--palette all] [--episodes 36 27 27] \
      SHARD_DIR [SHARD_DIR ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def render(
    output: Path,
    units: list[tuple[str, int]],
    scene_config: Path | None,
    palette: str = "all",
    color_config: Path | None = None,
) -> dict:
    import numpy as np

    from brickcell.simulation import generate
    from brickcell.simulation.world import World

    plan = generate.color_plan(palette, color_config)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    world = World(output / "scene", scene_config)
    setup_seconds = time.perf_counter() - started
    camera_pos = world.model.camera("inspection").pos.copy()
    rows, metadata = [], {}
    try:
        for split, episode in units:
            rng = np.random.default_rng([generate.SEED, generate.SPLIT_NAMES.index(split), episode])
            for row, meta in generate.render_episode(
                world, output, split, episode, rng, camera_pos, plan
            ):
                rows.append(row)
                metadata[row["image_id"]] = meta
    finally:
        world.close()
    timing = {
        "units": [f"{s}:{e}" for s, e in units],
        "images": len(rows),
        "world_setup_seconds": round(setup_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    data = {"samples": rows, "metadata": metadata, "timing": timing}
    (output / "rows.json").write_text(json.dumps(data, indent=2))
    return timing


def merge(
    output: Path,
    shards: list[Path],
    episodes: tuple[int, int, int],
    palette: str = "all",
    color_config: Path | None = None,
) -> dict:
    """Join shard outputs into one dataset with the row order of generate.py."""
    from brickcell.simulation import generate

    plan = generate.color_plan(palette, color_config)
    all_units = generate.units(episodes)
    order = {f"{s}:{e}": n for n, (s, e) in enumerate(all_units)}
    rows, metadata, copies = [], {}, []
    output.mkdir(parents=True, exist_ok=False)
    for shard in shards:
        data = json.loads((shard / "rows.json").read_text())
        metadata.update(data.get("metadata") or {})
        for row in data["samples"]:
            (output / row["path"]).parent.mkdir(exist_ok=True)
            copies.append((shard / row["path"], output / row["path"]))
            rows.append(row)
    # The shards are on a network Volume: one read has a long latency, so copy in parallel.
    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(lambda pair: shutil.copy2(*pair), copies))

    def key(row):
        _, split, episode, index = row["image_id"].split("-")
        return order[f"{split}:{episode}"], int(index)

    rows.sort(key=key)
    expected = {f"{s}:{e}" for s, e in all_units}
    found = {"{}:{}".format(*r["image_id"].split("-")[1:3]) for r in rows}
    if found != expected or len({r["image_id"] for r in rows}) != len(rows):
        raise SystemExit(f"Shards are incomplete or overlap: missing {sorted(expected - found)}")
    if shards:
        shutil.copytree(shards[0] / "scene", output / "scene")
    manifest = generate.build_manifest(rows, metadata, plan, episodes)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"images": len(rows), "shards": len(shards)}


def _episodes(value: list[int] | None, palette: str) -> tuple[int, int, int]:
    from brickcell.simulation import generate

    return tuple(value) if value else generate.default_episodes(palette)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {name: sub.add_parser(name) for name in ("render", "merge")}
    for command in commands.values():
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--palette", choices=("off", "main", "all"), default="all")
        command.add_argument("--episodes", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
        command.add_argument("--color-config", type=Path)
    commands["render"].add_argument("--units", nargs="+", required=True, help="split:episode")
    commands["render"].add_argument("--scene-config", type=Path)
    commands["merge"].add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "render":
        units = [(u.split(":")[0], int(u.split(":")[1])) for u in args.units]
        timing = render(args.output, units, args.scene_config, args.palette, args.color_config)
        print(json.dumps(timing))
    else:
        episodes = _episodes(args.episodes, args.palette)
        print(
            json.dumps(merge(args.output, args.shards, episodes, args.palette, args.color_config))
        )


if __name__ == "__main__":
    main()
