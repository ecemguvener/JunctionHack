"""Audit current calibration candidates and score diagnostics without choosing thresholds."""

import json
import platform
from collections import Counter
from pathlib import Path

import torch
from anomalib.models import Patchcore

from brickcell.detector.workflow import torch_predict
from brickcell.inspection.images import read_rgb
from brickcell.util import save_json, sha256, versions

out = Path(__file__).resolve().parent
source = out.parent / "patchcore-combined-earlier650-v1"
inventory = json.loads((source / "inventory.json").read_text())
trained = json.loads((source / "summary.json").read_text())
torch.set_num_threads(4)
results = []
for model_summary in trained["models"]:
    camera = model_summary["camera"]
    folder = source / camera
    checkpoint = folder / "reference.ckpt"
    assert sha256(checkpoint) == model_summary["checkpoint_sha256"]
    training = json.loads((folder / "manifest.json").read_text())["samples"]
    objects = {r["object_id"] for r in training}
    sessions = {r["session_id"] for r in training}
    training_hashes = {sha256(Path(r["path"])) for r in training}
    rows = []
    for row in inventory["images"]:
        m = row["metadata"]
        if m["camera"] != camera:
            continue
        path = source / row["path"]
        assert sha256(path) == row["sha256"]
        reasons = []
        if row["disposition"] == "accepted_clean_reference_only":
            reasons.append("used_for_training")
        if m["human_label"] not in {"good", "defective"}:
            reasons.append("human_label_missing")
        if not m["object_id"]:
            reasons.append("physical_object_identity_unconfirmed")
        if not m["session_id"]:
            reasons.append("session_identity_missing")
        if m["object_id"] in objects:
            reasons.append("training_object_overlap")
        if m["session_id"] in sessions:
            reasons.append("training_session_overlap")
        if row["sha256"] in training_hashes:
            reasons.append("training_image_hash_overlap")
        if row["disposition"] == "excluded_rejected_empty_scene":
            reasons.append("rejected_empty_scene")
        if row["disposition"] == "excluded_reserved_calibration_identity_pending":
            reasons.append("source_calibration_eligibility_false")
        rows.append(
            {
                "image_id": row["image_id"],
                "source_path": row["original_path"],
                "sha256": row["sha256"],
                "label": m["human_label"],
                "object_id": m["object_id"],
                "session_id": m["session_id"],
                "disposition": row["disposition"],
                "exclusion_reasons": reasons,
                "eligible_for_validation": not reasons,
            }
        )
    eligible = [r for r in rows if r["eligible_for_validation"]]
    diagnostic_ids = {
        r["image_id"]
        for r in rows
        if r["disposition"]
        in {"excluded_marked_brick", "excluded_reserved_calibration_identity_pending"}
    }
    # These scores are diagnostics only: all selected rows fail the independence/identity audit.
    model = Patchcore.load_from_checkpoint(str(checkpoint), map_location="cpu", pre_trained=False)
    model.cpu().eval()
    diagnostic_scores = []
    for row in inventory["images"]:
        if row["image_id"] not in diagnostic_ids:
            continue
        score, latency = torch_predict(model, read_rgb(source / row["path"]), (128, 128))
        diagnostic_scores.append(
            {
                "image_id": row["image_id"],
                "raw_patchcore_distance": score,
                "latency_ms": latency,
                "label": row["metadata"]["human_label"],
                "eligible_for_calibration": False,
                "scope": "EXCLUDED_CANDIDATE_DIAGNOSTIC_ONLY",
            }
        )
    result = {
        "camera": camera,
        "serial": model_summary["serial"],
        "checkpoint_sha256": model_summary["checkpoint_sha256"],
        "training_manifest_sha256": sha256(folder / "manifest.json"),
        "status": "NEEDS_INDEPENDENT_LABELED_VALIDATION_DATA",
        "eligible_validation_count": len(eligible),
        "eligible_labels": dict(Counter(r["label"] for r in eligible)),
        "exclusion_counts": dict(
            Counter(reason for r in rows for reason in r["exclusion_reasons"])
        ),
        "training_objects": sorted(objects),
        "training_sessions": sorted(sessions),
        "threshold": None,
        "automatic_sorting_allowed": False,
        "candidates": rows,
        "diagnostic_scores": diagnostic_scores,
    }
    assert not eligible, "New eligible data found: review before calibration"
    save_json(out / f"{camera}-audit.json", result)
    results.append(
        {k: v for k, v in result.items() if k not in {"candidates", "diagnostic_scores"}}
    )
    del model
save_json(
    out / "report.json",
    {
        "host": platform.node(),
        "models": results,
        "thresholds_saved": False,
        "accuracy_evaluated": False,
        "scope": "Calibration readiness audit; diagnostic scores are not calibration evidence",
        "inventory_sha256": sha256(source / "inventory.json"),
        "script_sha256": sha256(Path(__file__)),
    },
)
save_json(out / "versions.json", versions())
save_json(
    out / "validation-intake.json",
    {
        "schema_version": 1,
        "samples": [],
        "required_sample_fields": [
            "image_id",
            "path",
            "sha256",
            "camera",
            "serial",
            "object_id",
            "session_id",
            "label",
            "split",
        ],
        "requirements": [
            "Both good and defective labels per camera in val",
            "Confirmed physical IDs distinct from all training objects, including cross-session aliases",
            "Capture sessions distinct from training and untouched test",
            "Unique hashes; matching camera serial and full-frame RGB preprocessing",
            "Visible target with valid inspection view; no empty scenes",
            "Independent test set remains untouched until threshold is frozen",
        ],
    },
)
print(json.dumps({"host": platform.node(), "models": results}, indent=2))
