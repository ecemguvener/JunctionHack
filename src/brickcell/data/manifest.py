"""Explicit partitions: training never consumes validation or held-out test images."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from brickcell.util import save_json, sha256


@dataclass(frozen=True)
class Sample:
    image_id: str
    path: str
    label: str
    split: str
    object_id: str | None = None
    session_id: str | None = None
    view_id: str | None = None
    defect_type: str | None = None
    mask_path: str | None = None
    defect_regions: list[list[int]] | None = None
    synthetic: bool = False
    family: str | None = None
    color: str | None = None


@dataclass
class Manifest:
    path: Path
    samples: list[Sample]

    def image_path(self, sample: Sample) -> Path:
        return (self.path.parent / sample.path).resolve()

    def partition(self, split: str) -> list[Sample]:
        return [s for s in self.samples if s.split == split]


def load_manifest(path: Path) -> Manifest:
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("Expected manifest schema_version=1")
    samples = [Sample(**row) for row in raw["samples"]]
    return Manifest(path.resolve(), samples)


def validate(manifest: Manifest, require_partitions: bool = True) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    hashes: dict[str, list[Sample]] = defaultdict(list)
    counts = Counter(f"{s.split}/{s.label}" for s in manifest.samples)
    ids: set[str] = set()
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    paths: dict[Path, list[str]] = defaultdict(list)
    for s in manifest.samples:
        if not s.image_id or s.image_id in ids:
            errors.append(f"Duplicate or empty image_id: {s.image_id}")
        ids.add(s.image_id)
        if s.label not in {"good", "defective"}:
            errors.append(f"{s.image_id}: invalid label {s.label}")
        allowed = (
            {"train", "val", "test"} if require_partitions else {"train", "val", "test", "unsplit"}
        )
        if s.split not in allowed:
            errors.append(f"{s.image_id}: invalid split {s.split}")
        if s.split == "train" and s.label != "good":
            errors.append(f"{s.image_id}: training must contain only good images")
        for field in ("object_id", "session_id"):
            value = getattr(s, field)
            if value:
                groups[(field, value)].add(s.split)
        path = manifest.image_path(s)
        paths[path].append(s.image_id)
        try:
            with Image.open(path) as im:
                size = im.size
                im.verify()
            with Image.open(path) as im:
                im.convert("RGB").load()
            hashes[sha256(path)].append(s)
            if s.mask_path:
                with Image.open(manifest.path.parent / s.mask_path) as mask:
                    if mask.size != size:
                        errors.append(f"{s.image_id}: mask/image dimensions differ")
                    mask.verify()
            for box in s.defect_regions or []:
                if len(box) != 4 or not (
                    0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]
                ):
                    errors.append(f"{s.image_id}: invalid defect bounding box")
        except (OSError, ValueError) as exc:
            errors.append(f"{s.image_id}: missing/corrupt image or mask: {exc}")
    for path, members in paths.items():
        if len(members) > 1:
            errors.append(f"Repeated image path {path}: {members}")
    duplicates = [[s.image_id for s in group] for group in hashes.values() if len(group) > 1]
    for group in duplicates:
        errors.append(f"Duplicate file hash: {group}")
    for (field, value), splits in groups.items():
        if len(splits) > 1:
            errors.append(f"Split overlap for {field}={value}: {sorted(splits)}")
    if not manifest.samples:
        errors.append("Manifest has no images")
    if require_partitions:
        for key in ("train/good", "val/good", "val/defective", "test/good", "test/defective"):
            if counts[key] == 0:
                errors.append(f"Missing required partition/class: {key}")
    if any(not s.object_id and not s.session_id for s in manifest.samples):
        warnings.append("Some images lack object/session IDs; physical leakage cannot be ruled out")
    if any(s.synthetic for s in manifest.samples):
        warnings.append(
            "SYNTHETIC fixture data: metrics are software checks, not challenge performance"
        )
    return {
        "valid": not errors,
        "counts": dict(counts),
        "errors": errors,
        "warnings": warnings,
        "duplicates": duplicates,
        "manifest_sha256": sha256(manifest.path),
    }


def require_valid(manifest: Manifest) -> dict:
    report = validate(manifest)
    if not report["valid"]:
        raise ValueError("Invalid dataset:\n" + "\n".join(report["errors"]))
    return report


def split_manifest(source: Path, output: Path, seed: int = 42) -> dict:
    """Keep connected object AND session groups together. Never silently drop defects."""
    manifest = load_manifest(source)
    initial = validate(manifest, require_partitions=False)
    if initial["errors"]:
        raise ValueError("\n".join(initial["errors"]))
    if any(s.split != "unsplit" for s in manifest.samples):
        raise ValueError(
            "Splitting requires all rows to be unsplit; refusing to reshuffle held-out data"
        )
    parent = list(range(len(manifest.samples)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, s in enumerate(manifest.samples):
        if not s.object_id and not s.session_id:
            raise ValueError(f"{s.image_id}: supply object_id or session_id before splitting")
        for field in ("object_id", "session_id"):
            value = getattr(s, field)
            if value:
                key = (field, value)
                if key in seen:
                    parent[root(i)] = root(seen[key])
                seen[key] = i
    groups: dict[int, list[Sample]] = defaultdict(list)
    for i, s in enumerate(manifest.samples):
        groups[root(i)].append(s)
    good = [g for g in groups.values() if all(s.label == "good" for s in g)]
    bad = [g for g in groups.values() if any(s.label == "defective" for s in g)]
    if len(good) < 3 or len(bad) < 2:
        raise ValueError(
            "Need >=3 independent all-good groups and >=2 groups with defects; collect more sessions/objects"
        )
    rng = random.Random(seed)
    rng.shuffle(good)
    rng.shuffle(bad)
    n_eval = max(1, int(len(good) * 0.2))
    assignments = [
        (g, "val" if i < n_eval else "test" if i < 2 * n_eval else "train")
        for i, g in enumerate(good)
    ]
    assignments += [(g, "val" if i % 2 == 0 else "test") for i, g in enumerate(bad)]
    rows = []
    import os

    for group, split in assignments:
        for sample in group:
            row = asdict(sample)
            row.update(
                split=split,
                path=os.path.relpath(manifest.image_path(sample), output.resolve().parent),
            )
            if sample.mask_path:
                row["mask_path"] = os.path.relpath(
                    manifest.path.parent / sample.mask_path, output.resolve().parent
                )
            rows.append(row)
    metadata = json.loads(source.read_text())
    save_json(output, {**metadata, "schema_version": 1, "split_seed": seed, "samples": rows})
    return require_valid(load_manifest(output))


def import_directory(root: Path, output: Path) -> None:
    """Layout: good|defective/session/object/view/image.ext; metadata is mandatory."""
    import os

    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        parts = path.relative_to(root).parts
        if len(parts) != 5 or parts[0] not in {"good", "defective"}:
            raise ValueError(f"Expected label/session/object/view/image.ext: {path}")
        label, session, obj, view, _ = parts
        rows.append(
            asdict(
                Sample(
                    sha256(path)[:20],
                    os.path.relpath(path.resolve(), output.resolve().parent),
                    label,
                    "unsplit",
                    obj,
                    session,
                    view,
                )
            )
        )
    save_json(output, {"schema_version": 1, "samples": rows})
