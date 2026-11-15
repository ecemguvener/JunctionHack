"""CPU OpenVINO inference with a versioned preprocessing/threshold contract."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from brickcell.inspection.images import input_tensor
from brickcell.inspection.threshold import Threshold
from brickcell.util import sha256


@dataclass(frozen=True)
class Prediction:
    score: float
    latency_ms: float
    anomaly_map: np.ndarray | None


class OpenVINODetector:
    def __init__(self, model: Path) -> None:
        import openvino as ov

        self.path = model.resolve()
        self.metadata = json.loads(self.path.with_suffix(".json").read_text())
        if self.metadata["model_sha256"] != sha256(self.path) or self.metadata[
            "weights_sha256"
        ] != sha256(self.path.with_suffix(".bin")):
            raise ValueError(
                "Model artifact hash mismatch; refusing to apply threshold to a different model"
            )
        self.model_version = self.metadata["model_version"]
        threshold = self.metadata.get("threshold")
        self.threshold = Threshold(**threshold) if threshold else None
        self.size = tuple(self.metadata["input_size"])
        self.core = ov.Core()
        self.compiled = self.core.compile_model(
            str(self.path),
            device_name="CPU",
            config={"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY"},
        )
        self.request = self.compiled.create_infer_request()
        shape = list(self.compiled.input(0).shape)
        if shape != [1, 3, *self.size]:
            raise ValueError(f"Unexpected model input shape: {shape}")
        self.outputs = {port.get_any_name(): port for port in self.compiled.outputs}
        if "pred_score" not in self.outputs:
            raise ValueError(f"Model lacks pred_score output: {list(self.outputs)}")

    def predict(self, rgb: np.ndarray) -> Prediction:
        tensor = input_tensor(rgb, self.size)
        start = time.perf_counter()
        values = self.request.infer({self.compiled.input(0): tensor})
        elapsed = (time.perf_counter() - start) * 1000
        score = float(values[self.outputs["pred_score"]].reshape(-1)[0])
        if not np.isfinite(score):
            raise ValueError("Model returned a nonfinite score")
        anomaly_map = (
            values[self.outputs["anomaly_map"]].squeeze().copy()
            if "anomaly_map" in self.outputs
            else None
        )
        return Prediction(score, elapsed, anomaly_map)


def save_map(prediction: Prediction, rgb: np.ndarray, output: Path) -> str | None:
    if prediction.anomaly_map is None:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output.with_suffix(".npy"), prediction.anomaly_map)
    amap = prediction.anomaly_map
    scaled = (amap - amap.min()) / max(float(np.ptp(amap)), 1e-8)
    heat = np.zeros((*amap.shape, 3), dtype=np.uint8)
    heat[..., 0] = (scaled * 255).astype(np.uint8)
    heat_image = Image.fromarray(heat).resize((rgb.shape[1], rgb.shape[0]))
    Image.blend(Image.fromarray(rgb), heat_image, 0.4).save(output.with_suffix(".png"))
    return str(output.with_suffix(".npy"))


def benchmark(model: Path, image: Path, iterations: int = 30, warmup: int = 5) -> dict:
    import platform

    from brickcell.inspection.images import read_rgb
    from brickcell.metrics.report import latency

    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    detector = OpenVINODetector(model)
    rgb = read_rgb(image)
    for _ in range(warmup):
        detector.predict(rgb)
    values = [detector.predict(rgb).latency_ms for _ in range(iterations)]
    return {
        "device": "CPU",
        "host": platform.platform(),
        "model_version": detector.model_version,
        "image_sha256": sha256(image),
        "warmup_iterations": warmup,
        "scope": "Repeated single-image OpenVINO infer() only; excludes preprocessing/compilation; not accuracy or robot throughput",
        "latency": latency(values),
        "samples_ms": values,
    }
