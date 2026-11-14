"""Build local, read-only dataset review pages. No hardware or paid jobs."""

import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml

from brickcell.workbench.common import fresh_output, read_json, write_json
from brickcell.workbench.demo import create_demo
from brickcell.workbench.inspection import import_csv, review_inspection
from brickcell.workbench.policy import review_policy
from brickcell.workbench.view import inspection_page, policy_page


def build_policy(args) -> dict:
    report, episodes = review_policy(args.manifest, args.limits)
    output = args.output.resolve()
    report["commands"] = [
        "# Run from the repository root; commands do not launch automatically.",
        "uv sync --project policy --locked --python 3.12",
        'PYTHONPATH=src TORCH_HOME="$PWD/.cache/torch" policy/.venv/bin/python policy/train_act.py '
        f"--manifest {shlex.quote(str(args.manifest.resolve()))} "
        f"--output {shlex.quote(str(output / 'training-run'))} --steps 50",
        "# Review a held-out closed-loop rollout; training loss is not pick success.",
        "# The format fixture is for loader tests only; use reviewed robot episodes for learning.",
    ]
    if not report["smoke_ready"]:
        report["commands"].insert(
            0, "# BLOCKED: resolve report errors before using these commands."
        )
    # Optional supplied evidence is presented verbatim, never inferred from dataset readiness.
    report["training_evidence"] = read_json(args.training_report) if args.training_report else None
    report["rollout_evidence"] = read_json(args.rollout_report) if args.rollout_report else None
    fresh_output(output)
    write_json(output / "report.json", report)
    policy_page(output / "index.html", report, episodes)
    return report


def build_inspection(args) -> dict:
    report, cards = review_inspection(args.manifest)
    output = args.output.resolve()
    config = output / "patchcore-review.yaml"
    report["commands"] = [
        "# Run from the repository root. Models/reports use a fresh output directory.",
        f"uv run python -m brickcell validate-data --manifest {shlex.quote(str(args.manifest.resolve()))}",
        f"uv run python -m brickcell train --config {shlex.quote(str(config))}",
        f"uv run python -m brickcell export --config {shlex.quote(str(config))}",
        f"uv run python -m brickcell parity --config {shlex.quote(str(config))}",
        "# Freeze model and threshold before independent final test evaluation:",
        f"uv run python -m brickcell evaluate --config {shlex.quote(str(config))}",
    ]
    if not report["smoke_ready"]:
        report["commands"].insert(
            0, "# BLOCKED: resolve errors / assign grouped splits before training."
        )
    fresh_output(output)
    config.write_text(
        yaml.safe_dump(
            {
                "manifest": str(args.manifest.resolve()),
                "output_dir": str(output / "model"),
                "seed": 42,
                "image_size": [128, 128],
                "batch_size": 4,
                "num_workers": 0,
                "torch_threads": 4,
                "backbone": "resnet18.tv_in1k",
                "layers": ["layer2", "layer3"],
                "pre_trained": True,
                "coreset_sampling_ratio": 0.1,
                "num_neighbors": 3,
                "uncertainty_fraction": 0.02,
            }
        )
    )
    write_json(output / "report.json", report)
    inspection_page(output / "index.html", report, cards)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate tiny offline format fixtures")
    demo.add_argument("--output", type=Path, required=True)
    importer = commands.add_parser("import-inspection", help="Validate and import a labeled CSV")
    importer.add_argument("--csv", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    for name in ("policy", "inspection"):
        command = commands.add_parser(name, help=f"Review {name} and report smoke readiness")
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--strict", action="store_true", help="Exit 1 when smoke readiness fails"
        )
        if name == "policy":
            command.add_argument("--limits", type=Path)
            command.add_argument("--training-report", type=Path)
            command.add_argument("--rollout-report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            create_demo(args.output)
            print(f"Created SYNTHETIC FORMAT FIXTURES in {args.output}")
            return 0
        if args.command == "import-inspection":
            print(json.dumps(import_csv(args.csv, args.output), indent=2))
            return 0
        report = build_policy(args) if args.command == "policy" else build_inspection(args)
        print(
            json.dumps(
                {k: report[k] for k in ("kind", "smoke_ready", "errors", "warnings")}, indent=2
            )
        )
        print(f"Review: {args.output.resolve() / 'index.html'}")
        return int(args.strict and not report["smoke_ready"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Workbench error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
