"""Review the existing NPZ contract without importing Torch or controlling a robot."""

from collections import Counter
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from brickcell.policy.data import CAMERAS, JOINTS, file_hash, load_episodes, validate_episode
from brickcell.workbench.common import finite_list, image_uri, read_json


def load_limits(path: Path | None, synthetic: bool) -> dict | None:
    if path is None:
        return None
    limits = read_json(path)
    if limits.get("units") != "radians" or limits.get("schema_version") != 1:
        raise ValueError("Limit profile must declare schema_version=1 and radians")
    if limits.get("scope") not in {"simulation_only", "physical_measured"}:
        raise ValueError("Limit profile needs an explicit scope")
    if not synthetic and limits["scope"] == "simulation_only":
        raise ValueError("Simulation limits cannot validate physical demonstrations")
    for name in JOINTS:
        for kind in ("state", "action"):
            pair = np.asarray(limits["joints"][name][kind], dtype=float)
            if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] >= pair[1]:
                raise ValueError(f"Invalid {name}/{kind} interval")
    return limits


def review_policy(manifest: Path, limits_path: Path | None = None) -> tuple[dict, list]:
    raw = read_json(manifest)
    if not isinstance(raw.get("episodes"), list) or any(
        not isinstance(row, dict) for row in raw["episodes"]
    ):
        raise ValueError("episodes must be a list of episode objects")
    errors, warnings, episodes = [], [], []
    synthetic = raw.get("synthetic") is True
    if not isinstance(raw.get("synthetic"), bool):
        errors.append("Declare synthetic as true or false")
    if raw.get("action_units") != "radians" or tuple(raw.get("joint_names", [])) != JOINTS:
        errors.append(
            "Unsupported units/joint order: existing ACT loader requires six joints in radians"
        )
    if raw.get("state_units", "radians") != "radians":
        errors.append("Unsupported state units: this archive contract requires radians")
    if raw.get("action_semantics") != "absolute_joint_position_targets_before_physics_step":
        errors.append("Missing/unsupported action semantics; never infer action timing or units")
    if raw.get("schema_version") != 1:
        errors.append("Expected schema_version=1")
    if tuple(raw.get("camera_names", [])) != CAMERAS:
        errors.append("Expected overview and wrist camera names")
    fps = raw.get("fps")
    if not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        errors.append("fps must be finite and positive")
        fps = 50
    limits = load_limits(limits_path, synthetic)
    if limits is None:
        errors.append(
            "Joint limits unverified: supply --limits with an explicit coordinate profile"
        )
    ids, hashes, groups = set(), set(), {}
    counts = Counter()
    for index, row in enumerate(raw.get("episodes", [])):
        eid = str(row.get("episode_id", f"missing-id-{index}"))
        issues, violations = [], []
        item = {
            "episode_id": eid,
            "split": row.get("split"),
            "issues": issues,
            "violations": violations,
            "frames": 0,
            "preview": None,
        }
        if not row.get("episode_id") or eid in ids:
            issues.append("Missing or duplicate episode ID")
        ids.add(eid)
        counts[str(row.get("split"))] += 1
        if row.get("split") not in {"train", "val", "test"}:
            issues.append("Missing/invalid split")
        if row.get("success") is not True:
            issues.append("Failed or unreviewed episode: excluded from successful-teacher training")
        for field in ("seed", "object_id", "session_id"):
            value = row.get(field)
            if value is not None:
                key = (field, str(value))
                previous = groups.setdefault(key, row.get("split"))
                if previous != row.get("split"):
                    issues.append(f"Split leakage for {field}={value}")
        if not synthetic and (not row.get("object_id") or not row.get("session_id")):
            issues.append("Physical episodes need object_id and session_id for leakage checks")
        try:
            path = (manifest.parent / row["path"]).resolve()
            if not path.is_relative_to(manifest.parent.resolve()):
                raise ValueError("Archive path escapes dataset directory")
            digest = file_hash(path)
            if digest != row.get("sha256"):
                issues.append("Archive checksum mismatch")
            if digest in hashes:
                issues.append("Duplicate archive content")
            hashes.add(digest)
            with np.load(path, allow_pickle=False) as archive:
                arrays = dict(archive)
            try:
                validate_episode(arrays, fps)
            except (ValueError, KeyError, TypeError) as exc:
                issues.append(str(exc))
            timestamp = arrays.get("timestamp", np.array([]))
            if timestamp.ndim != 1:
                issues.append("timestamp must be a one-dimensional array")
            n = len(timestamp) if timestamp.ndim == 1 else 0
            item["frames"] = n
            if row.get("frames") != n:
                issues.append("Manifest frame count differs from timestamps")
            if n and limits:
                for kind in ("state", "action"):
                    values = arrays.get(kind)
                    if values is None or values.shape != (n, 6):
                        continue
                    for j, name in enumerate(JOINTS):
                        low, high = limits["joints"][name][kind]
                        # Small numerical tolerance only, never physical slack or clipping.
                        bad = np.flatnonzero(
                            np.isfinite(values[:, j])
                            & ((values[:, j] < low - 1e-6) | (values[:, j] > high + 1e-6))
                        )
                        if len(bad):
                            violations.append(
                                {
                                    "kind": kind,
                                    "joint": name,
                                    "count": len(bad),
                                    "frames": bad.tolist(),
                                    "range": [low, high],
                                    "max_excess_radians": float(
                                        np.maximum(
                                            low - values[bad, j], values[bad, j] - high
                                        ).max()
                                    ),
                                }
                            )
                if violations:
                    issues.append("Joint-limit violations; values were not clipped")
            # Broken cameras remain inspectable as missing frames, never padded silently.
            if n and all(arrays.get(k, np.array([])).shape == (n, 6) for k in ("state", "action")):
                cameras = {}
                for camera in CAMERAS:
                    images = arrays.get(camera, [])
                    cameras[camera] = [
                        image_uri(Image.fromarray(images[i]))
                        if i < len(images)
                        and images[i].shape == (128, 128, 3)
                        and images[i].dtype == np.uint8
                        else None
                        for i in range(n)
                    ]
                item["preview"] = {
                    "timestamp": finite_list(timestamp),
                    "state": finite_list(arrays["state"]),
                    "action": finite_list(arrays["action"]),
                    "cameras": cameras,
                }
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
            issues.append(f"Cannot fully read episode: {exc}")
        errors.extend(f"{eid}: {issue}" for issue in issues)
        episodes.append(item)
    for split in ("train", "val", "test"):
        if not counts[split]:
            errors.append(f"Missing {split} episodes")
    # Check the actual existing trainer contract, not just a parallel approximation.
    if not errors:
        try:
            for split in ("train", "val", "test"):
                load_episodes(manifest, split)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(f"ACT loader rejects dataset: {exc}")
    warnings.append(
        "Frame alignment follows the declared archive contract; sensor clock accuracy needs station validation"
    )
    warnings.append(
        "Smoke readiness is not sufficient demonstration diversity or learned-policy success"
    )
    if synthetic:
        warnings.append("Synthetic recordings do not replace real leader/follower demonstrations")
    return {
        "kind": "movement_policy",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_hash(manifest),
        "synthetic": synthetic,
        "movement_source": raw.get("teacher", "unspecified"),
        "units": raw.get("action_units"),
        "limit_tolerance_radians": 1e-6,
        "joint_names": list(JOINTS),
        "fps": fps,
        "limits": limits,
        "smoke_ready": not errors,
        "station_ready": False,
        "counts": dict(counts),
        "errors": errors,
        "warnings": warnings,
        "episodes": [{k: v for k, v in e.items() if k != "preview"} for e in episodes],
    }, episodes
