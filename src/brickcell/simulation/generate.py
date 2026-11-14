"""Generate explicitly synthetic camera fixtures from actual contact-grasp states.

Brick colours (``--palette``):

- ``off``: the configured brick colour of the scene (configs/sim-scene.yaml), with a small
  jitter. This is the old behaviour (6/3/3 episodes, one random generator).
- ``main``: the 12 main colours of configs/brick-palette.yaml.
- ``all`` (default): all palette colours.

With a palette, each brick of each episode gets one palette colour (palette albedo + the same
small jitter). The colour order is a seeded round-robin, so every colour appears about the same
number of times in each split, and the result does not depend on the shard layout.

The synthetic defect is a dark mark. On a dark brick the mark has little contrast, so a
"defective" label can be invisible. configs/sim-data-colors.yaml has the measured mark contrast
(CIE76 delta E) of each colour and a minimum. A colour below the minimum is used only for good
samples. ``--measure-contrast OUT`` measures the table again.

The manifest keeps the schema of brickcell.data.manifest. The colour of each image is in the
top-level ``sample_metadata`` field (load_manifest reads only ``samples``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from brickcell.palette import load_palette, rgb_to_lab

SEED = 2026
SPLIT_NAMES = ("train", "val", "test")
PALETTE_MODES = ("off", "main", "all")
# Episodes per split (train, val, test). An episode presents the 4 bricks one time.
LEGACY_EPISODES = (6, 3, 3)
# 36 train episodes = 144 good images (8 per colour for 18 colours). 27 val and 27 test
# episodes = 54 good images (3 per colour) and 54 defective images (about 4 per allowed colour).
PALETTE_EPISODES = (36, 27, 27)
COLOR_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "sim-data-colors.yaml"
COLOR_JITTER = 0.015
CAMERA_JITTER = 0.0005
DESCRIPTION = (
    "SYNTHETIC MuJoCo camera fixtures; cloned geometry, no real-world accuracy claim. "
    "Fixed ROI [48,90,208,220]."
)


def default_episodes(palette: str) -> tuple[int, int, int]:
    return LEGACY_EPISODES if palette == "off" else PALETTE_EPISODES


def split_plan(episodes: tuple[int, int, int]) -> list[tuple[str, int]]:
    return list(zip(SPLIT_NAMES, episodes))


def units(episodes: tuple[int, int, int]) -> list[tuple[str, int]]:
    """All (split, episode) units in generator order."""
    return [(split, e) for split, count in split_plan(episodes) for e in range(count)]


def is_defective(split: str, index: int) -> bool:
    """The defect rule: in val and test, every second brick has the mark."""
    return split != "train" and index % 2 == 1


@dataclass(frozen=True)
class ColorPlan:
    """Which palette colour each brick gets. mode "off" = the configured scene colour."""

    mode: str
    good: tuple[str, ...] = ()
    defective: tuple[str, ...] = ()
    excluded: tuple[tuple[str, float], ...] = ()  # (name, mark delta E) not used for defects
    min_mark_delta_e: float = 0.0
    albedo: tuple[tuple[str, tuple[float, float, float]], ...] = ()
    seed: int = SEED

    def rgb(self, name: str) -> np.ndarray:
        return np.array(dict(self.albedo)[name], float)

    def color(self, split: str, episode: int, index: int, bricks: int) -> str | None:
        """Palette name for brick ``index`` of ``episode`` (None in mode "off")."""
        if self.mode == "off":
            return None
        defective = is_defective(split, index)
        same_kind = [i for i in range(bricks) if is_defective(split, i) == defective]
        slot = episode * len(same_kind) + same_kind.index(index)
        names = self.defective if defective else self.good
        rnd, pos = divmod(slot, len(names))
        # A new seeded order for each round, so that one part does not always get one colour.
        order = np.random.default_rng(
            [self.seed, SPLIT_NAMES.index(split), int(defective), rnd]
        ).permutation(len(names))
        return names[order[pos]]

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "good_colors": list(self.good),
            "defective_colors": list(self.defective),
            "min_mark_delta_e": self.min_mark_delta_e,
            "excluded_from_defective": {name: value for name, value in self.excluded},
            "color_jitter": COLOR_JITTER,
        }


def load_color_config(path: Path | None = None) -> dict:
    raw = yaml.safe_load(Path(path or COLOR_CONFIG).read_text()) or {}
    if "min_mark_delta_e" not in raw or not isinstance(raw.get("mark_delta_e"), dict):
        raise ValueError(f"{path or COLOR_CONFIG}: need min_mark_delta_e and mark_delta_e")
    return raw


def color_plan(mode: str, color_config: Path | None = None, palette_path=None) -> ColorPlan:
    if mode not in PALETTE_MODES:
        raise ValueError(f"palette must be one of {PALETTE_MODES}, got {mode!r}")
    if mode == "off":
        return ColorPlan("off")
    palette = load_palette(palette_path)
    config = load_color_config(color_config)
    minimum = float(config["min_mark_delta_e"])
    measured = config["mark_delta_e"]
    names = palette.names(main_only=mode == "main")
    missing = [n for n in names if n not in measured]
    if missing:
        raise ValueError(
            f"No measured mark contrast for {missing}. Run "
            "`python -m brickcell.simulation.generate --measure-contrast OUT.json` and add "
            f"the values to {color_config or COLOR_CONFIG}."
        )
    defective = tuple(n for n in names if float(measured[n]) >= minimum)
    if not defective:
        raise ValueError("No palette colour has enough mark contrast for defective samples")
    return ColorPlan(
        mode=mode,
        good=tuple(names),
        defective=defective,
        excluded=tuple((n, float(measured[n])) for n in names if n not in defective),
        min_mark_delta_e=minimum,
        albedo=tuple((n, palette.albedo(n)) for n in names),
    )


def mark_delta_e(with_mark: np.ndarray, without_mark: np.ndarray) -> tuple[float, int]:
    """CIE76 delta E between the mark and the brick face under it, and the mark pixel count.

    The mark pixels are the pixels that change when the mark is shown.
    """
    mask = np.abs(with_mark.astype(int) - without_mark.astype(int)).max(axis=2) > 2
    if not mask.any():
        return 0.0, 0
    on = rgb_to_lab(with_mark[mask].reshape(-1, 3) / 255.0).mean(axis=0)
    off = rgb_to_lab(without_mark[mask].reshape(-1, 3) / 255.0).mean(axis=0)
    return round(float(np.linalg.norm(on - off)), 2), int(mask.sum())


def _render_crop(world, object_id: str, mark: bool) -> np.ndarray:
    from brickcell.simulation.world import inspection_crop

    world.model.geom(f"{object_id}-mark").rgba[3] = float(mark)
    return inspection_crop(world.camera())


def render_episode(
    world,
    root: Path,
    split: str,
    episode: int,
    rng: np.random.Generator,
    camera_pos: np.ndarray,
    plan: ColorPlan,
) -> list[tuple[dict, dict]]:
    """Render one episode (4 picks) into ``root``. Return (manifest row, metadata) pairs."""
    from PIL import Image

    from brickcell.simulation.world import OBJECTS

    world.reset()
    result = []
    for i, object_id in enumerate(OBJECTS):
        defective = is_defective(split, i)
        world.model.geom(f"{object_id}-mark").rgba[3] = float(defective)
        name = plan.color(split, episode, i, len(OBJECTS))
        # Jitter around the palette colour, or around the scene colour (world.bricks).
        base = plan.rgb(name) if name else np.array(world.bricks[object_id].rgba)
        rgb = np.clip(base + rng.uniform(-COLOR_JITTER, COLOR_JITTER, 3), 0, 1)
        world.model.geom(f"{object_id}-solid").rgba[:3] = rgb
        world.model.camera("inspection").pos[:] = camera_pos + rng.uniform(
            -CAMERA_JITTER, CAMERA_JITTER, 3
        )
        world.action("pick", object_id)
        world.action("present", object_id)
        pixels = _render_crop(world, object_id, defective)
        meta = {"color": name, "rgb": [round(float(v), 4) for v in rgb], "part": object_id}
        if defective:
            # The same state without the mark: measures the contrast of this exact label.
            meta["mark_delta_e"], meta["mark_pixels"] = mark_delta_e(
                pixels, _render_crop(world, object_id, False)
            )
            world.model.geom(f"{object_id}-mark").rgba[3] = 1.0
        image_id = f"render-{split}-{episode}-{i}"
        relative = f"{split}/{image_id}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
        label = "defective" if defective else "good"
        row = dict(
            image_id=image_id,
            path=relative,
            split=split,
            label=label,
            object_id=image_id,
            session_id=f"render-session-{split}-{episode}",
            view_id="inspection-fixed-roi",
            synthetic=True,
            defect_type="synthetic_dark_surface_mark" if defective else None,
        )
        print(image_id, label, name or "", meta.get("mark_delta_e", ""), flush=True)
        result.append((row, meta))
        # Dataset generation uses known labels for fixture placement only.
        world.action("move_defective" if defective else "move_good", object_id)
        world.action("release", object_id)
        world.action("verify_release", object_id)
    return result


def build_manifest(rows: list[dict], metadata: dict, plan: ColorPlan, episodes) -> dict:
    """Manifest (schema 1) plus the colour fields. ``metadata``: image_id -> metadata."""
    manifest = {"schema_version": 1, "description": DESCRIPTION, "samples": rows}
    if plan.mode != "off":
        manifest["palette"] = {**plan.summary(), "episodes": dict(split_plan(tuple(episodes)))}
    manifest["sample_metadata"] = {row["image_id"]: metadata[row["image_id"]] for row in rows}
    return manifest


def generate(
    root: Path,
    scene_config: object = None,
    palette: str = "all",
    episodes: tuple[int, int, int] | None = None,
    color_config: Path | None = None,
) -> None:
    """Render all episodes in one process with one random generator (seed 2026)."""
    from brickcell.simulation.world import World

    if root.exists():
        raise ValueError(f"Refusing to overwrite {root}")
    episodes = tuple(episodes or default_episodes(palette))
    plan = color_plan(palette, color_config)
    root.mkdir(parents=True)
    world = World(root / "scene", scene_config)
    rng = np.random.default_rng(SEED)
    camera_pos = world.model.camera("inspection").pos.copy()
    rows, metadata = [], {}
    try:
        for split, episode in units(episodes):
            for row, meta in render_episode(world, root, split, episode, rng, camera_pos, plan):
                rows.append(row)
                metadata[row["image_id"]] = meta
        manifest = build_manifest(rows, metadata, plan, episodes)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        world.close()


def measure_contrast(output: Path, scene_config: object = None, palette_path=None) -> dict:
    """Mark contrast (CIE76 delta E) of each palette colour on each part, at the inspection pose.

    Each part is picked and presented one time. Then each colour is rendered with and without
    the mark (no jitter). The colour value for the config is the minimum over the parts.
    """
    import tempfile

    from brickcell.simulation.world import OBJECTS, World

    palette = load_palette(palette_path)
    scratch = tempfile.TemporaryDirectory(prefix="brickcell-contrast-")
    world = World(Path(scratch.name) / "scene", scene_config)
    table: dict[str, dict[str, float]] = {name: {} for name in palette.names()}
    pixels: dict[str, int] = {}
    try:
        world.reset()
        for object_id in OBJECTS:
            world.action("pick", object_id)
            world.action("present", object_id)
            for name in palette.names():
                world.model.geom(f"{object_id}-solid").rgba[:3] = palette.albedo(name)
                on = _render_crop(world, object_id, True)
                delta, count = mark_delta_e(on, _render_crop(world, object_id, False))
                table[name][object_id] = delta
                pixels[object_id] = count
            world.model.geom(f"{object_id}-mark").rgba[3] = 0.0
            world.action("move_good", object_id)
            world.action("release", object_id)
            world.action("verify_release", object_id)
    finally:
        world.close()
        scratch.cleanup()
    result = {
        "metric": "CIE76 delta E, mean Lab of the mark pixels with and without the mark",
        "mark_pixels": pixels,
        "per_part": table,
        "mark_delta_e": {name: min(values.values()) for name, values in table.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/two-bowl"))
    parser.add_argument(
        "--scene-config",
        type=Path,
        help="YAML with cameras and colours (default: configs/sim-scene.yaml)",
    )
    parser.add_argument(
        "--palette",
        choices=PALETTE_MODES,
        default="all",
        help="brick colours: off = scene colours (old data), main = 12 main colours, all",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        help=f"episodes per split (default {PALETTE_EPISODES}; {LEGACY_EPISODES} with off)",
    )
    parser.add_argument("--color-config", type=Path, help=f"default: {COLOR_CONFIG}")
    parser.add_argument(
        "--measure-contrast",
        type=Path,
        metavar="OUT_JSON",
        help="measure the mark contrast of each palette colour, write OUT_JSON, then stop",
    )
    args = parser.parse_args()
    if args.measure_contrast:
        print(json.dumps(measure_contrast(args.measure_contrast, args.scene_config), indent=2))
    else:
        generate(args.output, args.scene_config, args.palette, args.episodes, args.color_config)
