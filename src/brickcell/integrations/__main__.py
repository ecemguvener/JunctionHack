"""Explicit sidecar CLI; default is a local plan with no network calls."""

import argparse
import importlib
import json
from pathlib import Path

from brickcell.integrations.core import Context, HTTP, Outbox, case_id, validate

PROVIDERS = {"cloudinary", "gemini", "elevenlabs", "tigerdata", "aiven", "mongodb", "gemma", "verda"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--providers", nargs="+", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--evidence-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/integrations"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    case = validate(json.loads(args.case.read_text()))
    if not args.execute:
        print(json.dumps({"case_id": case_id(case), "providers": args.providers, "network": False}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    outbox = Outbox(args.output / "outbox.sqlite3")
    context = Context(args.evidence_root, args.output, HTTP())
    try:
        failed = False
        for provider in args.providers:
            run = importlib.import_module("brickcell.integrations." + provider).run
            receipt = outbox.deliver(case, provider, context, run)
            failed |= "error_type" in receipt
            (args.output / f"{case_id(case)}-{provider}.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps(receipt))
        if failed:
            raise SystemExit(1)
    finally:
        outbox.close()


if __name__ == "__main__":
    main()
