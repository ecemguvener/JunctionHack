import pytest
import yaml

from brickcell.integrations.training_job import plan


def test_remote_training_keeps_grouped_splits_and_output_in_volume(manifest, tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump({"manifest": str(manifest.path), "output_dir": "model-v1"}))
    _, result = plan("job.yaml", tmp_path)
    assert result["training_device"] == "CPU"
    assert result["test_evaluation_requested"] is False
    assert result["synthetic"] is True
    (tmp_path / "model-v1").mkdir()
    with pytest.raises(ValueError, match="new output"):
        plan("job.yaml", tmp_path)
    with pytest.raises(ValueError, match="volume"):
        plan("../escaped.yaml", tmp_path)
