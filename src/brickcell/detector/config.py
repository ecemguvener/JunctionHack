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
    minimum_defect_recall: float | None = None
    maximum_good_nonaccept_rate: float = 0.1
    sensor_contract: dict | None = None
    surface_profile: dict | None = None
    foreground_memory: bool = False

    def __post_init__(self) -> None:
        if type(self.foreground_memory) is not bool or (
            self.foreground_memory and self.surface_profile is None
        ):
            raise ValueError("Foreground memory requires an explicit block surface profile")
        if self.surface_profile is not None:
            from brickcell.inspection.block_surface import SurfaceProfile

            SurfaceProfile(**self.surface_profile)
        if self.minimum_defect_recall is not None and not 0 < self.minimum_defect_recall <= 1:
            raise ValueError("minimum_defect_recall must be in (0,1]")
        if not 0 <= self.maximum_good_nonaccept_rate <= 1:
            raise ValueError("maximum_good_nonaccept_rate must be in [0,1]")
        if self.sensor_contract is not None:
            contract = self.sensor_contract
            if (
                set(contract) != {"serial", "crop", "modality"}
                or contract["modality"] != "rgb"
                or not contract["serial"]
            ):
                raise ValueError("Mark training requires an explicit RGB camera contract")
            crop = contract["crop"]
            if (
                not isinstance(crop, list)
                or len(crop) != 4
                or any(type(v) is not int for v in crop)
                or not (0 <= crop[0] < crop[2] and 0 <= crop[1] < crop[3])
                or min(crop[2] - crop[0], crop[3] - crop[1]) < 16
            ):
                raise ValueError("Camera contract needs a valid crop of at least 16x16")
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
