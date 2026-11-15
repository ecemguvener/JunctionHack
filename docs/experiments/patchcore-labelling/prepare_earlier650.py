"""Gather the exact earlier 650 RGB captures without changing source labels."""

import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from brickcell.util import save_json, sha256

run = Path(__file__).resolve().parent
project = run.parent.parent
rows = []


def add(path, metadata, expected_hash, group, reason, record):
    digest = sha256(path)
    if digest != expected_hash:
        raise ValueError(f"Source hash mismatch: {path}")
    with Image.open(path) as image:
        image.convert("RGB").load()
        dimensions = list(image.size)
    image_id = f"earlier-{len(rows):04d}"
    target = run / "raw" / f"{image_id}.jpg"
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(path, target)
    rows.append(
        dict(
            image_id=image_id,
            original_path=str(path.relative_to(project)),
            path=str(target.relative_to(run)),
            sha256=digest,
            dimensions=dimensions,
            capture_group=group,
            disposition=reason,
            source_record=str(record.relative_to(project)),
            metadata=metadata,
        )
    )


records = [
    (p, "guided")
    for p in sorted(
        (project / "artifacts/live-marks-20261114T235451Z").rglob("capture-record.json")
    )
]
records += [
    (
        project
        / "artifacts/calibration-candidates/20261115T011804Z/yellow-clean/capture-record.json",
        "calibration",
    )
]
records += [
    (project / "artifacts" / name / "capture-record.json", "earlier")
    for name in [
        "brick-002-full-baseline-01",
        "brick-002-full-baseline-02",
        "brick-002-full-raised-01",
    ]
]
for record, group in records:
    data = json.loads(record.read_text())
    copy = run / "records" / record.relative_to(project / "artifacts")
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record, copy)
    if group == "calibration":
        reason = "excluded_reserved_calibration_identity_pending"
    elif data.get("review_status") == "rejected_empty_scene":
        reason = "excluded_rejected_empty_scene"
    elif data.get("human_label") == "defective" or data.get("visual_label") == "defective":
        reason = "excluded_marked_brick"
    elif data.get("human_label") == "good" and data.get("baseline_accepted") is True:
        reason = "accepted_clean_reference_only"
    else:
        reason = "excluded_label_or_acceptance_pending"
    for snapshot in data["snapshots"]:
        metadata = {
            key: data.get(key)
            for key in [
                "object_id",
                "session_id",
                "human_label",
                "baseline_accepted",
                "eligible_for_training",
                "review_status",
                "synthetic",
            ]
        }
        metadata.update({key: snapshot[key] for key in ["camera", "serial", "frame_id"]})
        metadata["view_id"] = data.get("view_id", data.get("pose", snapshot["camera"]))
        add(
            record.parent / snapshot["files"]["rgb"],
            metadata,
            snapshot["sha256"]["rgb"],
            group,
            reason,
            record,
        )

record = project / "artifacts/arm-presentation-capture/20261115T012318Z/session.json"
session = json.loads(record.read_text())
frames = record.parent / "frames.jsonl"
for p in [record, frames, record.parent / "progress.json"]:
    target = run / "records/presentation" / p.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, target)
for line in frames.read_text().splitlines():
    data = json.loads(line)
    metadata = {
        key: data.get(key)
        for key in ["human_label", "object_id", "synthetic", "camera", "serial", "frame_id"]
    }
    metadata["session_id"] = session["session_id"]
    add(
        record.parent / data["file"],
        metadata,
        data["sha256"],
        "recording",
        "excluded_unlabeled_recording",
        record,
    )

counts = Counter(row["capture_group"] for row in rows)
assert counts == {"guided": 96, "calibration": 6, "earlier": 72, "recording": 476}, counts
assert len(rows) == 650
digests = Counter(row["sha256"] for row in rows)
duplicates = {h: n for h, n in digests.items() if n > 1}
save_json(
    run / "inventory.json",
    {
        "scope": "Original 650 RGB captures; labels unchanged",
        "source_counts": dict(counts),
        "dispositions": dict(Counter(r["disposition"] for r in rows)),
        "accepted_by_camera": dict(
            Counter(
                r["metadata"]["camera"]
                for r in rows
                if r["disposition"] == "accepted_clean_reference_only"
            )
        ),
        "duplicate_hashes": duplicates,
        "images": rows,
    },
)
assert not duplicates, duplicates
print(
    json.dumps(
        {
            "count": len(rows),
            "groups": counts,
            "accepted": Counter(
                r["metadata"]["camera"]
                for r in rows
                if r["disposition"] == "accepted_clean_reference_only"
            ),
        },
        indent=2,
    )
)
