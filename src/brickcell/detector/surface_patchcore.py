"""Train ordinary Anomalib PatchCore with only foreground patch descriptors.

The feature extractor, coreset selection, inference graph and export are still
Anomalib 2.7.0 PatchCore. Only descriptors eligible for the normal bank change.
"""

import numpy as np
import torch
from anomalib.models import Patchcore
from torch.nn import functional as F

from brickcell.inspection.block_surface import surface_input
from brickcell.inspection.images import read_rgb


def foreground_embeddings(embedding: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Keep patch cells whose input area is predominantly inside the block.

    Flatten order matches PatchcoreModel.reshape_embedding (batch, y, x, channel).
    This is cell occupancy, not a claim about the full CNN receptive field.
    """
    if embedding.ndim != 4 or masks.ndim != 3 or len(embedding) != len(masks):
        raise ValueError("Require BCHW features and matching BHW surface masks")
    coverage = F.adaptive_avg_pool2d(masks[:, None].float(), embedding.shape[-2:])[:, 0]
    selected = coverage >= 0.5
    if not selected.flatten(1).any(dim=1).all():
        raise ValueError("Each training image must contribute foreground patch descriptors")
    flat = embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1])
    return flat[selected.reshape(-1)]


class SurfacePatchcore(Patchcore):
    """Training-only selection; saved weights load into standard Patchcore for inference."""

    surface_profile: dict | None = None

    def training_step(self, batch, *args, **kwargs):
        del args, kwargs
        if not self.surface_profile or not batch.image_path:
            raise ValueError("Foreground training requires the source images and surface profile")
        if batch.gt_label is None or torch.any(batch.gt_label != 0):
            raise ValueError("Foreground PatchCore fits only explicitly good images")
        # Dataset and preprocessing use fixed resize only; reconstruct the exact
        # input mask from its source path without misusing anomaly gt_mask labels.
        masks = []
        for path in batch.image_path:
            _, mask = surface_input(
                read_rgb(path), tuple(batch.image.shape[-2:]), self.surface_profile
            )
            masks.append(mask)
        masks = torch.from_numpy(np.stack(masks)).to(batch.image.device)
        with torch.no_grad():
            features = self.model.feature_extractor(batch.image)
            pooled = {name: self.model.feature_pooler(value) for name, value in features.items()}
            embedding = self.model.generate_embedding(pooled)
            selected = foreground_embeddings(embedding, masks)
        self.model.embedding_store.append(selected)
        self.foreground_training_counts = getattr(self, "foreground_training_counts", []) + [
            {
                "candidate_patches": int(
                    embedding.shape[0] * embedding.shape[2] * embedding.shape[3]
                ),
                "kept_patches": len(selected),
            }
        ]
        return torch.tensor(0.0, requires_grad=True, device=self.device)
