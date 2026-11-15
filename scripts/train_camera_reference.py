"""Train an uncalibrated RGB reference pilot from an accepted clean camera record."""

import argparse
import json
from pathlib import Path

from appearance_pilot import fit, score
from PIL import Image

from brickcell.util import save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--camera", choices=("plate", "wrist"), required=True)
    parser.add_argument("--crop", nargs=4, type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-count", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--surface-profile", type=Path)
    parser.add_argument("--foreground-memory", action="store_true")
    parser.add_argument(
        "--layers", nargs="+", choices=["layer1", "layer2", "layer3"], default=["layer2", "layer3"]
    )
    args = parser.parse_args()
    data = json.loads(args.record.read_text())
    if (
        data.get("human_label") != "good"
        or data.get("synthetic") is not False
        or data.get("baseline_accepted") is not True
    ):
        raise ValueError("Require a recorded accepted clean reference, not an inferred label")
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    rows = [r for r in data.get("snapshots", []) if r["camera"] == args.camera]
    if not 2 <= args.fit_count < len(rows):
        raise ValueError("Need fitting references plus later clean-repeat images")
    x0, y0, x1, y1 = args.crop
    hashes, prepared = set(), []
    for i, row in enumerate(rows):
        source = args.record.parent / row["files"]["rgb"]
        digest = sha256(source)
        if digest != row["sha256"]["rgb"] or digest in hashes:
            raise ValueError("Source capture changed or duplicates another reference")
        hashes.add(digest)
        with Image.open(source) as image:
            w, h = image.size
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or min(x1 - x0, y1 - y0) < 16:
                raise ValueError("Crop must fit source and be at least 16x16")
            role = "references" if i < args.fit_count else "clean-repeat"
            target = args.output / role / f"{i:03}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").crop(args.crop).save(target)
        prepared.append(
            dict(
                source=str(source),
                source_sha256=digest,
                crop=str(target),
                crop_sha256=sha256(target),
                role=role,
                frame_id=row["frame_id"],
                serial=row["serial"],
            )
        )
    save_json(
        args.output / "inputs.json",
        dict(
            source_record_sha256=sha256(args.record),
            crop=args.crop,
            camera=args.camera,
            object_id=data["object_id"],
            session_id=data["session_id"],
            samples=prepared,
            scope="Accepted same-object clean reference only; original formal-training eligibility unchanged",
        ),
    )
    fit(
        args.output / "references",
        args.output / "model",
        data["object_id"],
        data["session_id"],
        args.crop,
        camera=args.camera,
        image_size=args.image_size,
        layers=args.layers,
        foreground_memory=args.foreground_memory,
        surface_profile=json.loads(args.surface_profile.read_text())
        if args.surface_profile
        else None,
    )
    result = score(args.output / "model", sorted((args.output / "clean-repeat").glob("*.png")))
    save_json(
        args.output / "summary.json",
        dict(
            camera=args.camera,
            architecture="Anomalib 2.7.0 PatchCore",
            scope="SAME_OBJECT_CLEAN_REFERENCE_ONLY",
            threshold=None,
            automatic_sorting_allowed=False,
            accuracy_evaluated=False,
            comparison=result,
            limitation="No marked plate/wrist validation or independent test bricks; no defect-detection claim",
        ),
    )
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
