"""Palette colours of the synthetic detector data (no MuJoCo; the plan and manifest only)."""

import json
from collections import Counter

import numpy as np
import pytest

from brickcell.data.manifest import load_manifest, validate
from brickcell.palette import load_palette
from brickcell.simulation import generate

BRICKS = 4


def _counts(plan, split, episodes, defective):
    return Counter(
        plan.color(split, e, i, BRICKS)
        for e in range(episodes)
        for i in range(BRICKS)
        if generate.is_defective(split, i) == defective
    )


def test_off_keeps_the_old_plan():
    plan = generate.color_plan("off")
    assert plan.color("val", 0, 1, BRICKS) is None
    assert generate.default_episodes("off") == (6, 3, 3)
    assert generate.units((6, 3, 3))[6] == ("val", 0)


def test_low_contrast_colours_are_good_only():
    config = generate.load_color_config()
    plan = generate.color_plan("all")
    palette = load_palette()
    assert set(plan.good) == set(palette.names())
    for name in palette.names():
        visible = config["mark_delta_e"][name] >= config["min_mark_delta_e"]
        assert (name in plan.defective) == visible
    assert "black" in dict(plan.excluded) and "black" not in plan.defective
    main = generate.color_plan("main")
    assert set(main.good) == set(palette.names(main_only=True))


def test_every_colour_appears_in_every_split_and_balanced():
    plan = generate.color_plan("all")
    train, val, test = generate.PALETTE_EPISODES
    good = _counts(plan, "train", train, False)
    assert set(good) == set(plan.good) and max(good.values()) - min(good.values()) <= 1
    for split, episodes in (("val", val), ("test", test)):
        good = _counts(plan, split, episodes, False)
        bad = _counts(plan, split, episodes, True)
        assert set(good) == set(plan.good) and min(good.values()) >= 3
        assert set(bad) == set(plan.defective) and min(bad.values()) >= 3
        assert max(bad.values()) - min(bad.values()) <= 1


def test_plan_is_reproducible_and_does_not_depend_on_the_split_size():
    first, second = generate.color_plan("all"), generate.color_plan("all")
    colors = [first.color("val", e, i, BRICKS) for e in range(5) for i in range(BRICKS)]
    assert colors == [second.color("val", e, i, BRICKS) for e in range(5) for i in range(BRICKS)]
    # The parts do not always get the same colour round after round.
    by_part = [{first.color("train", e, i, BRICKS) for e in range(36)} for i in range(BRICKS)]
    assert all(len(colors) > 5 for colors in by_part)


def test_missing_contrast_value_stops(tmp_path):
    config = tmp_path / "colors.yaml"
    config.write_text("min_mark_delta_e: 20\nmark_delta_e: {red: 90}\n")
    with pytest.raises(ValueError, match="No measured mark contrast"):
        generate.color_plan("all", config)
    with pytest.raises(ValueError):
        generate.color_plan("rainbow")


def test_mark_delta_e():
    face = np.full((10, 10, 3), 200, np.uint8)
    marked = face.copy()
    marked[2:5, 2:6] = 5
    delta, pixels = generate.mark_delta_e(marked, face)
    assert pixels == 12 and delta > 70
    assert generate.mark_delta_e(face, face) == (0.0, 0)


def test_manifest_keeps_the_schema(tmp_path):
    from PIL import Image

    plan = generate.color_plan("all")
    rows, metadata = [], {}
    episodes = (1, 1, 1)
    for split, episode in generate.units(episodes):
        for i in range(BRICKS):
            defective = generate.is_defective(split, i)
            image_id = f"render-{split}-{episode}-{i}"
            path = tmp_path / split / f"{image_id}.png"
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (8, 8), (len(rows) * 9, 0, 0)).save(path)
            rows.append(
                dict(
                    image_id=image_id,
                    path=f"{split}/{image_id}.png",
                    split=split,
                    label="defective" if defective else "good",
                    object_id=image_id,
                    session_id=f"render-session-{split}-{episode}",
                    synthetic=True,
                )
            )
            metadata[image_id] = {"color": plan.color(split, episode, i, BRICKS)}
    manifest = generate.build_manifest(rows, metadata, plan, episodes)
    assert manifest["palette"]["excluded_from_defective"]["black"] < 20
    assert set(manifest["sample_metadata"]) == {row["image_id"] for row in rows}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = validate(load_manifest(tmp_path / "manifest.json"))
    assert report["valid"], report["errors"]
