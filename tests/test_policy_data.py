import json

import numpy as np
import pytest

from brickcell.policy.data import JOINTS, file_hash, load_episodes, validate_episode


def episode():
    return dict(
        timestamp=np.array([0.0, 0.02, 0.04]),
        state=np.zeros((3, 6), np.float32),
        action=np.ones((3, 6), np.float32),
        overview=np.zeros((3, 128, 128, 3), np.uint8),
        wrist=np.zeros((3, 128, 128, 3), np.uint8),
    )


def test_synchronized_episode_and_dropped_frame():
    data = episode()
    assert validate_episode(data) == 3
    data["timestamp"][2] = 0.06
    with pytest.raises(ValueError, match="timestamps"):
        validate_episode(data)


@pytest.mark.parametrize("key", ["state", "action"])
def test_invalid_actions_or_states_rejected(key):
    data = episode()
    data[key][0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        validate_episode(data)


def test_wrong_image_format_rejected():
    data = episode()
    data["wrist"] = data["wrist"].astype(np.float32)
    with pytest.raises(ValueError, match="RGB uint8"):
        validate_episode(data)


def test_archive_provenance_and_split_overlap(tmp_path):
    archive = tmp_path / "episode.npz"
    np.savez_compressed(archive, **episode())
    row = dict(
        episode_id="one",
        path=archive.name,
        split="train",
        seed=1,
        success=True,
        sha256=file_hash(archive),
    )
    metadata = dict(action_units="radians", joint_names=JOINTS, fps=50, episodes=[row])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(metadata))
    assert len(load_episodes(manifest, "train")) == 1
    metadata["episodes"].append({**row, "split": "test"})
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Duplicate"):
        load_episodes(manifest, "train")
    metadata["episodes"].pop()
    manifest.write_text(json.dumps(metadata))
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_episodes(manifest, "train")
