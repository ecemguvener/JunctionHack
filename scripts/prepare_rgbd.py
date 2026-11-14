"""Prepare paired real RGB/depth inputs with explicit leakage-safe partitions."""

import argparse
import json
from pathlib import Path

from brickcell.data.rgbd import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.index, args.output), indent=2))


if __name__ == "__main__":
    main()
