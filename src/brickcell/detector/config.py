from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    manifest: Path
    output_dir: Path
    seed: int = 42
    image_size: tuple[int, int] = (128, 128)
    batch_size: int = 4
    num_workers: int = 0
    torch_threads: int = 4
    backbone: str = "resnet18.tv_in1k"
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    pre_trained: bool = True
    coreset_sampling_ratio: float = 0.05
    num_neighbors: int = 3
    uncertainty_fraction: float = 0.02
    sensor_contract: dict | None = None

    def __post_init__(self) -> None:
        from brickcell.inspection.rgbd import validate_sensor_contract

        validate_sensor_contract(self.sensor_contract)
        if len(self.image_size) != 2 or min(self.image_size) < 64:
            raise ValueError("image_size must contain two dimensions >=64")
        if self.batch_size < 1 or self.num_workers < 0 or self.torch_threads < 1:
            raise ValueError("Invalid batch size, worker or thread count")
        if not 0 < self.coreset_sampling_ratio <= 1 or self.num_neighbors < 1:
            raise ValueError("Invalid coreset sampling ratio or neighbor count")


def load_config(path: Path) -> ModelConfig:
    data = yaml.safe_load(path.read_text())
    for key in ("manifest", "output_dir"):
        data[key] = (path.resolve().parent / data[key]).resolve()
    if "image_size" in data:
        data["image_size"] = tuple(data["image_size"])
    return ModelConfig(**data)
