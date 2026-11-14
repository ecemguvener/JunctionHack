"""Fit real PatchCore on known-clean brick-002 crops; report uncalibrated distances."""

import argparse
import json
from pathlib import Path

import numpy as np
from appearance_pilot import fit, score
from PIL import Image

from brickcell.util import save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--changed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output to preserve previous models")
    roi = [225, 220, 272, 272]
    provenance = []
    paths = {}
    for folder, label in [(args.baseline, "clean"), (args.changed, "marked")]:
        record = json.loads((folder / "capture-record.json").read_text())
        for i, row in enumerate(r for r in record["snapshots"] if r["camera"] == "top"):
            group = (
                "references"
                if label == "clean" and i < 6
                else "clean-repeat"
                if label == "clean"
                else "marked"
            )
            dest = args.output / group / f"{i:03d}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = folder / row["files"]["rgb"]
            Image.open(source).convert("RGB").crop(roi).save(dest)
            paths.setdefault(group, []).append(dest)
            provenance.append(
                dict(
                    source=str(source),
                    source_sha256=sha256(source),
                    crop_image=str(dest),
                    crop_sha256=sha256(dest),
                    role=group,
                    object_id=record["object_id"],
                    session_id=record["session_id"],
                    frame_id=row["frame_id"],
                )
            )
    save_json(
        args.output / "inputs.json",
        dict(
            roi=roi,
            scope="All images share one physical object and session; no formal val/test split",
            images=provenance,
        ),
    )
    result = fit(
        args.output / "references",
        args.output / "model",
        "brick-002",
        "brick-002-controlled-depth-20261114-a",
        roi,
        camera="top",
    )
    comparison = score(args.output / "model", paths["clean-repeat"] + paths["marked"])
    clean = [
        r["raw_patchcore_distance"] for r in comparison["images"][: len(paths["clean-repeat"])]
    ]
    marked = [
        r["raw_patchcore_distance"] for r in comparison["images"][len(paths["clean-repeat"]) :]
    ]
    summary = dict(
        model_type="Anomalib 2.7.0 PatchCore",
        input_modality="RGB",
        camera="top",
        synthetic=False,
        reference_count=result["reference_count"],
        clean_repeat_count=len(clean),
        marked_count=len(marked),
        clean_repeat_distance_range=[min(clean), max(clean)],
        marked_distance_range=[min(marked), max(marked)],
        clean_repeat_median=float(np.median(clean)),
        marked_median=float(np.median(marked)),
        threshold=None,
        decision="UNCALIBRATED",
        accuracy_evaluated=False,
        scope="Same-object reference experiment; score separation does not establish unseen-object defect accuracy",
    )
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
