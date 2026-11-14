"""Prepare paired RGB and fixed-encoded depth manifests without choosing splits."""

import hashlib
import io
import json
import math
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from brickcell.data.manifest import Sample, load_manifest, validate
from brickcell.inspection.rgbd import (
    DEPTH_ENCODING,
    decode_snapshot,
    encode_depth,
    validate_sensor_contract,
)
from brickcell.util import save_json, sha256, versions


def prepare(index_path: Path, output: Path):
    index_path, output = index_path.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing datasets are preserved")
    index = json.loads(index_path.read_text())
    serial, crop = index["serial"], index["crop"]
    contracts = {mode: dict(serial=serial, crop=crop, modality=mode) for mode in ["rgb", "depth"]}
    contracts["depth"]["depth_encoding"] = DEPTH_ENCODING
    for contract in contracts.values():
        validate_sensor_contract(contract)
    captures = index["captures"]
    min_fraction = index.get("min_depth_fraction", 0.6)
    if not math.isfinite(min_fraction) or not 0 < min_fraction <= 1:
        raise ValueError("Minimum depth fraction must be in (0,1]")
    if not captures:
        raise ValueError("No captures in index")
    prepared = []
    seen = {mode: set() for mode in contracts}
    groups = {}
    sources = []
    for row in captures:
        if row["label"] not in ("good", "defective") or row["split"] not in (
            "train",
            "val",
            "test",
            "unsplit",
        ):
            raise ValueError("Every capture requires explicit valid label and split")
        if row["split"] == "train" and row["label"] != "good":
            raise ValueError("Depth and RGB training accept only good objects")
        for field in ["object_id", "session_id", "view_id"]:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Missing capture identity {field}")
        for field in ["object_id", "session_id"]:
            groups.setdefault((field, row[field]), set()).add(row["split"])
        path = (index_path.parent / row["snapshot"]).resolve()
        packet = json.loads(path.read_text())
        if packet["serial"] != serial or not packet.get("frame_id"):
            raise ValueError("Capture camera/frame identity mismatch")
        rgb, depth = decode_snapshot(packet)
        x0, y0, x1, y1 = crop
        if x1 > rgb.shape[1] or y1 > rgb.shape[0]:
            raise ValueError("Crop exceeds capture dimensions")
        encoded, valid = encode_depth(depth[y0:y1, x0:x1], packet["depth_scale_m"])
        if valid.mean() < min_fraction:
            raise ValueError("Capture lacks sufficient valid depth")
        images = {"rgb": rgb[y0:y1, x0:x1], "depth": encoded}
        source = dict(
            path=str(path),
            sha256=sha256(path),
            frame_id=packet["frame_id"],
            captured_at=packet["time"],
            object_id=row["object_id"],
            session_id=row["session_id"],
            view_id=row["view_id"],
            split=row["split"],
            depth_scale_m=packet["depth_scale_m"],
            valid_depth_fraction=float(valid.mean()),
        )
        sources.append(source)
        for mode, image in images.items():
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            payload = buffer.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen[mode]:
                raise ValueError(f"Duplicate {mode} encoded image hash")
            seen[mode].add(digest)
            name = f"{mode}/{digest}.png"
            sample = Sample(
                image_id=mode + "-" + digest[:20],
                path=name,
                label=row["label"],
                split=row["split"],
                object_id=row["object_id"],
                session_id=row["session_id"],
                view_id=row["view_id"],
                synthetic=False,
            )
            prepared.append((mode, sample, payload))
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Object/session split overlap across paired modalities")
    output.mkdir(parents=True)
    manifests = {}
    for mode in contracts:
        (output / mode).mkdir()
        rows = []
        for modality, sample, payload in prepared:
            if modality == mode:
                (output / sample.path).write_bytes(payload)
                rows.append(asdict(sample))
        manifest = output / f"{mode}-manifest.json"
        save_json(
            manifest,
            dict(
                schema_version=1,
                sensor_contract=contracts[mode],
                paired_index_sha256=sha256(index_path),
                samples=rows,
            ),
        )
        report = validate(load_manifest(manifest), require_partitions=False)
        if not report["valid"]:
            raise ValueError("Prepared manifest invalid: " + str(report["errors"]))
        manifests[mode] = str(manifest)
    save_json(
        output / "provenance.json",
        dict(
            index=str(index_path),
            index_sha256=sha256(index_path),
            sensor_contracts=contracts,
            source_captures=sources,
            package_versions=versions(),
            split_policy="Caller supplied; shared physical object/session assignments preserved",
        ),
    )
    return dict(manifests=manifests, captures=len(captures), trained=False, calibrated=False)
