from pathlib import Path

import pytest
import torch

from brickcell.detector.config import ModelConfig
from brickcell.detector.surface_patchcore import foreground_embeddings


def test_background_descriptors_cannot_enter_normal_bank():
    # Two images; each spatial descriptor has a unique value to verify B/Y/X order.
    features = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    masks = torch.zeros(2, 4, 4, dtype=torch.bool)
    masks[0, :2, :2] = True
    masks[0, 2:, 2:] = True
    masks[1, :2, 2:] = True
    selected = foreground_embeddings(features, masks)
    assert selected.flatten().tolist() == [0, 3, 5]


def test_foreground_sampling_preserves_all_feature_channels():
    features = torch.tensor([[[[1.0, 2.0]], [[10.0, 20.0]]]])
    selected = foreground_embeddings(features, torch.tensor([[[False, True]]]))
    assert selected.tolist() == [[2, 20]]


def test_patch_cell_requires_at_least_half_surface_occupancy():
    features = torch.tensor([[[[1.0, 2.0]]]])
    masks = torch.zeros(1, 2, 4, dtype=torch.bool)
    masks[0, 0, 0] = True  # 25% coverage: excluded
    masks[0, 0, 2:] = True  # 50% coverage: included
    assert foreground_embeddings(features, masks).flatten().tolist() == [2]


def test_empty_object_cannot_silently_train_on_another_batch_image():
    features = torch.zeros(2, 3, 4, 4)
    masks = torch.zeros(2, 16, 16, dtype=torch.bool)
    masks[0] = True
    with pytest.raises(ValueError, match="Each training image"):
        foreground_embeddings(features, masks)


def test_foreground_memory_requires_surface_contract():
    with pytest.raises(ValueError, match="surface profile"):
        ModelConfig(Path("manifest.json"), Path("out"), foreground_memory=True)
    config = ModelConfig(
        Path("manifest.json"),
        Path("out"),
        foreground_memory=True,
        surface_profile={"hue_ranges": [[95, 115]]},
    )
    assert config.foreground_memory
