"""Restore versioned HTML reviews into the ignored local artifacts directory."""

import argparse
import shutil
from pathlib import Path


def restore(output: Path):
    templates = Path(__file__).with_name("review_pages")
    for source in sorted(templates.rglob("*.html")):
        relative = source.relative_to(templates)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    restore(args.output)


if __name__ == "__main__":
    main()
