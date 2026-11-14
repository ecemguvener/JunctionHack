"""Portable, pickle-free demonstration archive and validation."""

import hashlib
import json
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMERAS = ("overview", "wrist")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_episode(arrays: dict, fps: int = 50) -> int:
    count = len(arrays["timestamp"])
    if count < 2 or arrays["state"].shape != (count, 6) or arrays["action"].shape != (count, 6):
        raise ValueError("Episode needs synchronized timestamps, six joint states and six actions")
    for key in ("state", "action", "timestamp"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Non-finite {key}")
    if not np.allclose(np.diff(arrays["timestamp"]), 1 / fps, atol=1e-6):
        raise ValueError("Dropped, duplicated or non-monotonic frame timestamps")
    for key in CAMERAS:
        if arrays[key].shape != (count, 128, 128, 3) or arrays[key].dtype != np.uint8:
            raise ValueError(f"{key} must be synchronized 128x128 RGB uint8")
    return count


def load_episodes(manifest: Path, split: str) -> list[dict]:
    description = json.loads(manifest.read_text())
    if description["action_units"] != "radians" or tuple(description["joint_names"]) != JOINTS:
        raise ValueError("Unsupported action units or joint order")
    ids, hashes, seeds, episodes = set(), set(), set(), []
    for row in description["episodes"]:
        if row["episode_id"] in ids or row["sha256"] in hashes or row["seed"] in seeds:
            raise ValueError("Duplicate episode, seed or archive across dataset partitions")
        ids.add(row["episode_id"])
        hashes.add(row["sha256"])
        seeds.add(row["seed"])
        if row["split"] not in ("train", "val", "test") or not row["success"]:
            raise ValueError("Only explicitly split, successful teacher episodes may train")
        path = (manifest.parent / row["path"]).resolve()
        if not path.is_relative_to(manifest.parent.resolve()) or file_hash(path) != row["sha256"]:
            raise ValueError("Invalid archive path or changed demonstration")
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        validate_episode(arrays, description["fps"])
        if row["split"] == split:
            episodes.append(arrays)
    if not episodes:
        raise ValueError(f"No {split} demonstrations")
    return episodes
