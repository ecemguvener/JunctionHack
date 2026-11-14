"""Offline SO-101 dynamics pipeline. Hardware capture is a separate explicit command."""

import argparse
import json
from pathlib import Path

from model import build
from replay import fit
from viewer import build_viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    model = sub.add_parser("build")
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--priors", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    optimize = sub.add_parser("fit")
    optimize.add_argument("--model", type=Path, required=True)
    optimize.add_argument("--fit", type=Path, required=True)
    optimize.add_argument("--val", type=Path, required=True)
    optimize.add_argument("--output", type=Path, required=True)
    optimize.add_argument("--max-iters", type=int, default=25)
    view = sub.add_parser("viewer")
    view.add_argument("--model", type=Path, required=True)
    view.add_argument("--episode", type=Path, required=True)
    view.add_argument("--report", type=Path, required=True)
    view.add_argument("--output", type=Path, required=True)
    synthetic = sub.add_parser("synthetic")
    synthetic.add_argument("--model", type=Path, required=True)
    synthetic.add_argument("--output", type=Path, required=True)
    published = sub.add_parser("published")
    published.add_argument("--source", type=Path, required=True)
    published.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.output, args.priors)
    elif args.command == "check":
        from checks import run

        print(json.dumps(run(args.model, args.output), indent=2))
    elif args.command == "fit":
        report = fit(args.model, args.fit, args.val, args.output, args.max_iters)
        print(
            json.dumps(
                {k: report[k] for k in ("fit", "val", "optimizer_success", "validation_improved")},
                indent=2,
            )
        )
    elif args.command == "synthetic":
        from synthetic import generate

        generate(args.model, args.output)
    elif args.command == "published":
        from published import import_priors

        import_priors(args.source, args.output)
    else:
        build_viewer(args.model, args.episode, args.report, args.output)


if __name__ == "__main__":
    main()
