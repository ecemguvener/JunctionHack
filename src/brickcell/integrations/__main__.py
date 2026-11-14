"""Explicit sidecar CLI; default is a local plan with no network calls."""

import argparse
import importlib
import json
from pathlib import Path

from brickcell.integrations.core import HTTP, Context, Outbox, case_id, validate

PROVIDERS = {
    "cloudinary",
    "gemini",
    "elevenlabs",
    "tigerdata",
    "aiven",
    "mongodb",
    "gemma",
    "verda",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", type=Path)
    source.add_argument("--drain", action="store_true")
    parser.add_argument("--providers", nargs="+", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--evidence-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/integrations"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    case = validate(json.loads(args.case.read_text())) if args.case else None
    if not args.execute:
        print(
            json.dumps(
                {
                    "case_id": case_id(case) if case else None,
                    "providers": args.providers,
                    "network": False,
                }
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    outbox = Outbox(args.output / "outbox.sqlite3")
    context = Context(args.evidence_root, args.output, HTTP())
    try:
        failed = False
        jobs = (
            [(case, p) for p in args.providers]
            if case
            else [
                (json.loads(row[0]), row[1])
                for row in outbox.db.execute(
                    "SELECT case_json,provider FROM deliveries WHERE status != 'done' ORDER BY updated"
                )
                if row[1] in args.providers
            ]
        )
        for case, provider in jobs:
            run = importlib.import_module("brickcell.integrations." + provider).run
            receipt = outbox.deliver(case, provider, context, run)
            failed |= "error_type" in receipt
            (args.output / f"{case_id(case)}-{provider}.json").write_text(
                json.dumps(receipt, indent=2)
            )
            print(json.dumps(receipt))
        if failed:
            raise SystemExit(1)
    finally:
        outbox.close()


if __name__ == "__main__":
    main()
