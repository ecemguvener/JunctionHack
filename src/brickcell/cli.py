from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from brickcell.util import json_default, save_json


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Brickcell CPU inspection and MOCK robot supervision"
    )
    sub = root.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--model", type=Path, default=Path("artifacts/models/patchcore/model.xml"))
    doctor.add_argument("--output", type=Path, default=Path("artifacts/doctor.json"))
    synthetic = sub.add_parser("synthetic", help="Generate explicitly synthetic software fixtures")
    synthetic.add_argument("--output", type=Path, default=Path("data/synthetic"))
    validate = sub.add_parser("validate-data")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--output", type=Path, default=Path("artifacts/data_validation.json"))
    importing = sub.add_parser("import-data")
    importing.add_argument("--directory", type=Path, required=True)
    importing.add_argument("--output", type=Path, required=True)
    splitting = sub.add_parser("split-data")
    splitting.add_argument("--manifest", type=Path, required=True)
    splitting.add_argument("--output", type=Path, required=True)
    splitting.add_argument("--seed", type=int, default=42)
    for command in ("train", "export", "parity", "evaluate"):
        action = sub.add_parser(command)
        action.add_argument("--config", type=Path, default=Path("configs/patchcore.yaml"))
        if command == "export":
            action.add_argument("--checkpoint", type=Path)
    predict = sub.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--object-id", default="manual-object")
    predict.add_argument("--quality-config", type=Path, default=Path("configs/inspection.yaml"))
    predict.add_argument("--output", type=Path, default=Path("artifacts/predictions"))
    benchmark = sub.add_parser(
        "benchmark", help="Repeated CPU inference latency; not an accuracy test"
    )
    benchmark.add_argument("--model", type=Path, required=True)
    benchmark.add_argument("--image", type=Path, required=True)
    benchmark.add_argument("--iterations", type=int, default=30)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    demo = sub.add_parser("demo", help="Always uses MockRobot; no hardware driver is shipped")
    mode = demo.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--mock", action="store_true", help="Scripted inspector: no model or image inference"
    )
    mode.add_argument("--model", type=Path, help="Actual OpenVINO inference driving MockRobot")
    demo.add_argument("--images", type=Path)
    demo.add_argument("--controller-config", type=Path, default=Path("configs/controller.yaml"))
    demo.add_argument("--quality-config", type=Path, default=Path("configs/inspection.yaml"))
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    return root


def demo(args) -> dict:
    import yaml

    from brickcell.controller.robot import MockRobot
    from brickcell.controller.supervisor import ControllerConfig, Supervisor
    from brickcell.inspection.service import InspectionService, MockInspector
    from brickcell.inspection.types import Decision
    from brickcell.metrics.report import EventLog
    from brickcell.sources.directory import DirectorySource

    if args.mock:
        decisions = {
            "mock-good": [Decision.GOOD],
            "mock-defective": [Decision.DEFECTIVE],
            "mock-reinspect": [Decision.REINSPECT, Decision.GOOD],
        }
        source = DirectorySource(
            {key: [Path("MOCK_NO_IMAGE")] * len(value) for key, value in decisions.items()}
        )
        inspector = MockInspector(decisions)
    else:
        from brickcell.detector.runtime import OpenVINODetector
        from brickcell.inspection.images import QualityConfig

        if not args.images:
            raise ValueError("--images is required with --model")
        source = DirectorySource.from_directory(args.images)
        inspector = InspectionService(
            OpenVINODetector(args.model),
            QualityConfig(**yaml.safe_load(args.quality_config.read_text())),
            args.output / "maps",
        )
    config = ControllerConfig(**yaml.safe_load(args.controller_config.read_text()))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = EventLog(args.output / f"events-{run_id}.jsonl")
    report = Supervisor(MockRobot(), source, inspector, config, log).run(list(source.frames))
    report["event_log"] = str(log.path)
    save_json(args.output / "summary.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    # Keep library caches in the project; do not alter HOME or require global installs.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    os.environ.setdefault("TORCH_HOME", str(Path(".cache/torch").resolve()))
    args = parser().parse_args(argv)
    exit_code = 0
    try:
        if args.command == "doctor":
            from brickcell.doctor import doctor

            result = doctor(args.model)
            save_json(args.output, result)
            exit_code = 0 if result["healthy"] else 1
        elif args.command == "synthetic":
            from brickcell.data.synthetic import generate

            result = {"manifest": str(generate(args.output)), "synthetic": True}
        elif args.command == "validate-data":
            from brickcell.data.manifest import load_manifest, validate

            result = validate(load_manifest(args.manifest))
            save_json(args.output, result)
            exit_code = 0 if result["valid"] else 1
        elif args.command == "import-data":
            from brickcell.data.manifest import import_directory

            import_directory(args.directory, args.output)
            result = {"manifest": str(args.output), "next": "split-data"}
        elif args.command == "split-data":
            from brickcell.data.manifest import split_manifest

            result = split_manifest(args.manifest, args.output, args.seed)
        elif args.command in ("train", "export", "parity", "evaluate"):
            from brickcell.detector import workflow
            from brickcell.detector.config import load_config

            config = load_config(args.config)
            if args.command == "export":
                result = workflow.export(config, args.checkpoint)
            else:
                result = getattr(workflow, args.command)(config)
            if args.command == "parity":
                exit_code = 0 if result["passed"] else 1
        elif args.command == "predict":
            import hashlib

            import yaml

            from brickcell.detector.runtime import OpenVINODetector
            from brickcell.inspection.images import QualityConfig
            from brickcell.inspection.service import InspectionService
            from brickcell.sources.directory import Frame
            from brickcell.util import sha256

            frame_id = sha256(args.image)[:20]
            frame = Frame(
                args.object_id,
                frame_id,
                datetime.now(timezone.utc).isoformat(),
                "manual",
                args.image,
            )
            inspector = InspectionService(
                OpenVINODetector(args.model),
                QualityConfig(**yaml.safe_load(args.quality_config.read_text())),
                args.output,
            )
            result = inspector.inspect(frame)
            save_json(
                args.output / f"{hashlib.sha256(frame_id.encode()).hexdigest()[:20]}.json", result
            )
            exit_code = 1 if result.decision == "ERROR" else 0
        elif args.command == "benchmark":
            from brickcell.detector.runtime import benchmark

            result = benchmark(args.model, args.image, args.iterations, args.warmup)
            save_json(args.output, result)
        elif args.command == "demo":
            result = demo(args)
            exit_code = 1 if result["errors"] else 0
        else:
            raise ValueError("Unsupported command")
        print(json.dumps(result, default=json_default, indent=2, allow_nan=False))
        return exit_code
    except Exception as exc:
        print(f"brickcell: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.command in ("train", "export"):
            print(
                "See output_dir training.log/export.log. Pretrained weights require initial internet access; no version downgrade or random-weight fallback is performed.",
                file=sys.stderr,
            )
        return 1
