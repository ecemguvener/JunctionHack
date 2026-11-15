"""Validate/prepare paired plate and wrist RGB mark-inspection datasets."""

import argparse
import json
from pathlib import Path

from brickcell.data.marks import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.index, args.output, args.check_only), indent=2))


if __name__ == "__main__":
    main()
