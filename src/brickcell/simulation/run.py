"""Run the existing supervisor with a rendered camera and actual OpenVINO detector."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from brickcell.controller.supervisor import ControllerConfig, Supervisor
from brickcell.detector.runtime import OpenVINODetector
from brickcell.inspection.images import QualityConfig
from brickcell.inspection.service import InspectionService
from brickcell.metrics.report import EventLog
from brickcell.simulation.bridge import LocalSimulation, PublishedInspector, SimCamera, SimRobot
from brickcell.util import json_default, save_json


def run(args) -> dict:
    connection = LocalSimulation(args.url)
    if args.reset:
        connection.call("/reset", {})
    health = connection.call("/health")
    output = args.output / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    detector = OpenVINODetector(args.model)
    quality = QualityConfig(**yaml.safe_load(args.quality_config.read_text()))
    inspector = PublishedInspector(
        InspectionService(detector, quality, output / "maps"), connection
    )
    supervisor = Supervisor(
        SimRobot(connection),
        SimCamera(connection, output / "frames"),
        inspector,
        ControllerConfig(action_timeout_s=20, inspection_timeout_s=30, poll_interval_s=0.05),
        EventLog(output / "events.jsonl"),
    )
    result = supervisor.run(health["objects"])
    result.update(
        model=str(args.model.resolve()),
        synthetic_camera=True,
        learned_robot_policy=False,
        run_id=health["run_id"],
        report_directory=str(output.resolve()),
    )
    save_json(output / "summary.json", result)
    save_json(args.output / "latest.json", result)
    connection.call("/summary", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/models/patchcore-sim/model.xml")
    )
    parser.add_argument("--quality-config", type=Path, default=Path("configs/inspection-sim.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/two-bowl/runs"))
    result = run(parser.parse_args())
    print(json.dumps(result, default=json_default, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
