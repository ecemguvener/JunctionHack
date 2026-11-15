"""Prepare plate/wrist RGB datasets with joint physical leakage-group checks."""

from dataclasses import asdict
from pathlib import Path

import yaml
from PIL import Image

from brickcell.data.manifest import Manifest, Sample, require_valid, validate
from brickcell.detector.config import ModelConfig
from brickcell.util import save_json, sha256, versions

CAMERAS = ("plate", "wrist")


def prepare(index_path: Path, output: Path, check_only=False):
    import json

    index_path, output = index_path.resolve(), output.resolve()
    index = json.loads(index_path.read_text())
    if set(index["cameras"]) != set(CAMERAS):
        raise ValueError("Mark inspection requires exactly plate and wrist RGB cameras")
    rows, sources = [], []
    for i, row in enumerate(index.get("samples", [])):
        if row.get("camera") not in CAMERAS or type(row.get("synthetic")) is not bool:
            raise ValueError("Each row requires plate/wrist identity and explicit synthetic flag")
        if not all(row.get(k) for k in ("object_id", "session_id", "view_id", "brick_type")):
            raise ValueError("Each image needs object/session/face and brick type")
        source = (index_path.parent / row["path"]).resolve()
        rows.append(
            Sample(
                f"capture-{i:06}",
                str(source),
                row["label"],
                row["split"],
                object_id=row["object_id"],
                session_id=row["session_id"],
                view_id=row["camera"] + ":" + row["view_id"],
                family=row["brick_type"],
                synthetic=row["synthetic"],
            )
        )
        sources.append(source)
    combined = Manifest(index_path, rows)
    report = validate(combined, require_partitions=False)
    if not report["valid"]:
        raise ValueError("Joint plate/wrist dataset invalid: " + "; ".join(report["errors"]))
    camera_rows = {}
    for camera in CAMERAS:
        settings = index["cameras"][camera]
        contract = dict(serial=settings["serial"], crop=settings["crop"], modality="rgb")
        ModelConfig(
            index_path,
            output,
            sensor_contract=contract,
            surface_profile=settings.get("surface_profile"),
            foreground_memory=settings.get("foreground_memory", False),
        )
        selected = [
            (s, row, path)
            for s, row, path in zip(rows, index["samples"], sources)
            if row["camera"] == camera
        ]
        require_valid(Manifest(index_path, [s for s, _, _ in selected]))
        for family in {s.family for s, _, _ in selected}:
            require_valid(Manifest(index_path, [s for s, _, _ in selected if s.family == family]))
        camera_rows[camera] = selected
        x0, y0, x1, y1 = settings["crop"]
        for _, _, path in selected:
            with Image.open(path) as image:
                if x1 > image.width or y1 > image.height:
                    raise ValueError(f"{camera} crop exceeds source frame")
    result = {
        "cameras": {c: len(camera_rows[c]) for c in CAMERAS},
        "index_sha256": sha256(index_path),
        "joint_groups_valid": True,
        "test_scored": False,
        "modality": "rgb",
    }
    if check_only:
        return result
    if output.exists():
        raise ValueError("Choose a new prepared output to preserve datasets/models")
    provenance = []
    for camera in CAMERAS:
        folder = output / camera
        prepared = []
        contract = dict(
            serial=index["cameras"][camera]["serial"],
            crop=index["cameras"][camera]["crop"],
            modality="rgb",
        )
        for sample, row, path in camera_rows[camera]:
            target = folder / "images" / (sample.image_id + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            source_hash = sha256(path)
            with Image.open(path) as image:
                image.convert("RGB").crop(contract["crop"]).save(target)
            data = asdict(sample)
            data["path"] = str(target.relative_to(folder))
            prepared.append(data)
            provenance.append(
                dict(
                    camera=camera,
                    source=str(path),
                    source_sha256=source_hash,
                    crop_sha256=sha256(target),
                    object_id=sample.object_id,
                    session_id=sample.session_id,
                    split=sample.split,
                )
            )
        manifest = folder / "manifest.json"
        surface_profile = index["cameras"][camera].get("surface_profile")
        save_json(
            manifest,
            dict(
                schema_version=1,
                sensor_contract=contract,
                surface_profile=surface_profile,
                samples=prepared,
            ),
        )
        require_valid(Manifest(manifest, [Sample(**s) for s in prepared]))
        config = dict(
            manifest="manifest.json",
            output_dir="model",
            image_size=[256, 256],
            seed=42,
            batch_size=4,
            num_workers=0,
            torch_threads=4,
            backbone="resnet18.tv_in1k",
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=3,
            uncertainty_fraction=0.02,
            minimum_defect_recall=1.0,
            maximum_good_nonaccept_rate=0.1,
            sensor_contract=contract,
            surface_profile=surface_profile,
            foreground_memory=index["cameras"][camera].get("foreground_memory", False),
        )
        (folder / "patchcore.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    save_json(output / "provenance.json", dict(**result, samples=provenance, versions=versions()))
    return result
