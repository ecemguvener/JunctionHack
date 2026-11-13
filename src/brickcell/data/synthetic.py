"""Deliberately tiny synthetic brick drawings, only for software smoke testing."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from brickcell.util import save_json


def generate(root: Path, seed: int = 42) -> Path:
    if (root / "manifest.json").exists():
        raise ValueError(f"Refusing to overwrite existing dataset: {root}")
    rng = np.random.default_rng(seed)
    rows = []
    for split, label, count in [
        ("train", "good", 12),
        ("val", "good", 6),
        ("val", "defective", 6),
        ("test", "good", 6),
        ("test", "defective", 6),
    ]:
        for i in range(count):
            image_id = f"synthetic-{split}-{label}-{i:03d}"
            pixels = np.clip(rng.normal(32, 2, (128, 128, 3)), 0, 255).astype("uint8")
            im = Image.fromarray(pixels)
            draw = ImageDraw.Draw(im)
            dx, dy = rng.integers(-3, 4, 2).tolist()
            draw.rectangle((28 + dx, 34 + dy, 100 + dx, 94 + dy), fill=(180 + i, 85 + i, 42 + i))
            for x in (42, 65, 88):
                for y in (48, 78):
                    draw.ellipse(
                        (x - 7 + dx, y - 7 + dy, x + 7 + dx, y + 7 + dy), fill=(220, 120 + i, 60)
                    )
            if label == "defective":
                draw.line((32 + dx, 38 + dy, 94 + dx, 88 + dy), fill=(8, 8, 8), width=9)
                draw.rectangle((60, 52, 80, 72), fill=(32, 32, 32))
            relative = f"{split}/{label}/{image_id}.png"
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            im.save(target)
            rows.append(
                dict(
                    image_id=image_id,
                    path=relative,
                    label=label,
                    split=split,
                    object_id=image_id,
                    session_id=f"synthetic-session-{split}",
                    view_id="front",
                    defect_type="drawn_crack" if label == "defective" else None,
                    synthetic=True,
                )
            )
    path = root / "manifest.json"
    save_json(
        path,
        {"schema_version": 1, "description": "SYNTHETIC SOFTWARE FIXTURE ONLY", "samples": rows},
    )
    return path
