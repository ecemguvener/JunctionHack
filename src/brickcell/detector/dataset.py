"""Explicit Anomalib ImageBatch loaders: no Folder automatic partitioning."""

from __future__ import annotations

import torch
from anomalib.data import ImageBatch, ImageItem
from torch.utils.data import DataLoader, Dataset

from brickcell.data.manifest import Manifest
from brickcell.inspection.block_surface import surface_input
from brickcell.inspection.images import read_rgb


class ManifestDataset(Dataset):
    def __init__(
        self, manifest: Manifest, split: str, size: tuple[int, int], surface_profile=None
    ) -> None:
        self.manifest, self.size = manifest, size
        self.surface_profile = surface_profile
        self.samples = manifest.partition(split)
        if split == "train" and any(s.label != "good" for s in self.samples):
            raise ValueError("Training data must be normal only")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ImageItem:
        sample = self.samples[index]
        path = self.manifest.image_path(sample)
        tensor, _ = surface_input(read_rgb(path), self.size, self.surface_profile)
        image = torch.from_numpy(tensor[0])
        return ImageItem(
            image=image,
            gt_label=torch.tensor(int(sample.label == "defective")),
            image_path=str(path),
        )


def loader(
    manifest: Manifest,
    split: str,
    size: tuple[int, int],
    batch_size: int,
    workers: int,
    surface_profile=None,
) -> DataLoader:
    return DataLoader(
        ManifestDataset(manifest, split, size, surface_profile),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=ImageBatch.collate,
    )
