from __future__ import annotations

from pathlib import Path

from brickcell.detector.runtime import OpenVINODetector, save_map
from brickcell.inspection.images import QualityConfig, check_view, read_rgb
from brickcell.inspection.types import Decision, InspectionResult
from brickcell.sources.directory import Frame


class InspectionService:
    mode = "anomalib_openvino_cpu"

    def __init__(
        self,
        detector: OpenVINODetector,
        quality: QualityConfig | None = None,
        map_dir: Path | None = None,
    ) -> None:
        contract = getattr(detector, "metadata", {}).get("sensor_contract") or {}
        if contract.get("modality") == "depth":
            raise ValueError("Depth models require aligned depth inputs via RGBDInspector")
        self.detector, self.quality, self.map_dir = detector, quality or QualityConfig(), map_dir

    def inspect(self, frame: Frame) -> InspectionResult:
        kwargs = dict(
            object_id=frame.object_id,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            view_id=frame.view_id,
            model_version=self.detector.model_version,
            threshold_version=self.detector.threshold.version if self.detector.threshold else None,
            inspector_mode=self.mode,
        )
        try:
            rgb = read_rgb(frame.path)
            quality = check_view(rgb, self.quality)
            if not quality.valid:
                return InspectionResult(
                    **kwargs,
                    is_valid_view=False,
                    anomaly_score=None,
                    decision=Decision.REINSPECT,
                    inference_latency_ms=0,
                    reason=",".join(quality.reasons),
                )
            prediction = self.detector.predict(rgb)
            threshold = self.detector.threshold
            decision = threshold.decide(prediction.score) if threshold else Decision.ERROR
            map_path = None
            if self.map_dir:
                # File names are derived from a hash, not user-supplied object paths.
                import hashlib

                name = hashlib.sha256(frame.frame_id.encode()).hexdigest()[:20]
                map_path = save_map(prediction, rgb, self.map_dir / name)
            return InspectionResult(
                **kwargs,
                is_valid_view=True,
                anomaly_score=prediction.score,
                decision=decision,
                inference_latency_ms=prediction.latency_ms,
                anomaly_map_path=map_path,
                reason="" if threshold else "No calibrated threshold available",
            )
        except Exception as exc:
            return InspectionResult(
                **kwargs,
                is_valid_view=False,
                anomaly_score=None,
                decision=Decision.ERROR,
                inference_latency_ms=0,
                reason=str(exc),
            )


class MockInspector:
    mode = "MOCK_SCRIPTED_NOT_ML"

    def __init__(self, decisions: dict[str, list[Decision]]) -> None:
        self.decisions = {k: list(v) for k, v in decisions.items()}

    def inspect(self, frame: Frame) -> InspectionResult:
        choices = self.decisions.get(frame.object_id, [])
        decision = choices.pop(0) if choices else Decision.ERROR
        return InspectionResult(
            frame.object_id,
            frame.frame_id,
            frame.timestamp,
            frame.view_id,
            decision in (Decision.GOOD, Decision.DEFECTIVE),
            None,
            decision,
            "mock-no-model",
            None,
            0,
            score_type="mock_no_score",
            reason="Scripted software fixture; no image classification",
            inspector_mode=self.mode,
        )
