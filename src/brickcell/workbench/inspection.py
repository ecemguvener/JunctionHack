"""CSV import and visual review with explicit object-face metadata."""

import csv
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from brickcell.data.manifest import Manifest, Sample, load_manifest, validate
from brickcell.util import sha256
from brickcell.workbench.common import image_uri

FACES = ("front", "right", "back", "left", "top")
COLUMNS = (
    "path",
    "object_id",
    "family",
    "color",
    "session_id",
    "face",
    "label",
    "split",
    "synthetic",
    "defect_type",
)


def import_csv(source: Path, output: Path) -> dict:
    """Paths are relative to the CSV. Never copy images, guess labels or reshuffle splits."""
    if output.exists():
        raise ValueError(f"Refusing to overwrite manifest: {output}")
    samples = []
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError(f"CSV needs columns: {', '.join(COLUMNS)}")
        for line, row in enumerate(reader, 2):
            row = {k: (v or "").strip() for k, v in row.items() if k is not None}
            for key in COLUMNS:
                if key != "defect_type" and not row[key]:
                    raise ValueError(f"Line {line}: missing {key}")
            if row["face"] not in FACES:
                raise ValueError(f"Line {line}: face must be one of {FACES}; bottom excluded")
            if row["synthetic"] not in {"true", "false"}:
                raise ValueError(f"Line {line}: synthetic must be true or false")
            if row["label"] == "defective" and not row["defect_type"]:
                raise ValueError(f"Line {line}: defective images need defect_type")
            path = (source.parent / row["path"]).resolve()
            samples.append(
                Sample(
                    image_id=sha256(path)[:20],
                    path=os.path.relpath(path, output.resolve().parent),
                    label=row["label"],
                    split=row["split"],
                    object_id=row["object_id"],
                    session_id=row["session_id"],
                    view_id=row["face"],
                    synthetic=row["synthetic"] == "true",
                    defect_type=row["defect_type"] or None,
                    family=row["family"],
                    color=row["color"],
                )
            )
    # Temporary manifest shares the final parent so relative image paths stay identical.
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"schema_version": 1, "samples": [asdict(s) for s in samples]}, indent=2)
    with tempfile.NamedTemporaryFile(mode="w+", dir=output.parent, suffix=".json") as candidate:
        candidate.write(payload)
        candidate.flush()
        report = validate(Manifest(Path(candidate.name), samples), require_partitions=False)
    if report["errors"]:
        raise ValueError("Invalid import:\n" + "\n".join(report["errors"]))
    with output.open("x") as stream:
        stream.write(payload + "\n")
    return {"manifest": str(output.resolve()), "images": len(samples)}


def review_inspection(path: Path) -> tuple[dict, list]:
    manifest = load_manifest(path)
    structural = validate(manifest, require_partitions=False)
    training = validate(manifest)
    warnings = list(structural["warnings"])
    missing_metadata, cards = [], []
    coverage = defaultdict(set)
    for s in manifest.samples:
        missing = [k for k in ("object_id", "session_id", "family", "color") if not getattr(s, k)]
        if s.view_id not in FACES:
            missing.append("named face (front/right/back/left/top)")
        if s.label == "defective" and not s.defect_type:
            missing.append("defect_type")
        if missing:
            missing_metadata.append({"image_id": s.image_id, "missing": missing})
        # Coverage is per object AND session; never pool different recording sessions.
        if s.object_id and s.session_id:
            key = (s.object_id, s.session_id)
            coverage[key]  # Include objects with unknown views as uncovered.
            if s.view_id in FACES:
                coverage[key].add(s.view_id)
        try:
            with Image.open(manifest.image_path(s)) as image:
                preview = image_uri(image)
        except (OSError, ValueError):
            preview = None
        cards.append({**asdict(s), "preview": preview})
    face_coverage = [
        {
            "object_id": obj,
            "session_id": session,
            "recorded_faces": sorted(faces),
            "missing_faces": [f for f in FACES if f not in faces],
        }
        for (obj, session), faces in sorted(coverage.items())
    ]
    complete = (
        bool(face_coverage)
        and not missing_metadata
        and all(not row["missing_faces"] for row in face_coverage)
    )
    if missing_metadata:
        warnings.append(
            "Capture metadata incomplete: review missing fields before station collection"
        )
    warnings.append(
        "Recorded face labels are not proof of focus, full visibility or freedom from gripper occlusion"
    )
    warnings.append(
        "Single-face smoke training can run without five-face coverage; it cannot certify the whole object"
    )
    return {
        "kind": "inspection_images",
        "manifest": str(path.resolve()),
        "manifest_sha256": sha256(path),
        "valid_capture_batch": structural["valid"],
        "smoke_ready": training["valid"],
        "station_ready": False,
        "counts": training["counts"],
        "errors": training["errors"],
        "warnings": warnings,
        "capture_errors": structural["errors"],
        "missing_metadata": missing_metadata,
        "required_faces": list(FACES),
        "excluded_faces": ["bottom"],
        "five_face_labels_complete": complete,
        "coverage": face_coverage,
        "synthetic_images": sum(s.synthetic is True for s in manifest.samples),
        "total_images": len(manifest.samples),
    }, cards
