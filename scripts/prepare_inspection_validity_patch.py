"""Prepare a patched LOCAL controller copy; never deploy or modify the input file."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "patches" / "inspection-validity"


def prepare(source: Path, output: Path) -> None:
    """Refuse an unknown revision, apply in a temporary directory, write a new file."""
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("Output must be a new file, different from the source")
    baseline = json.loads((BUNDLE / "baseline.json").read_text())
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != baseline["source_sha256"]:
        raise ValueError(
            "Controller differs from the reviewed deployed revision. Preserve the owner's "
            "changes and rebase/retest the inspection patch; do not force-apply it."
        )
    with tempfile.TemporaryDirectory(prefix="brickcell-inspection-") as folder:
        target = Path(folder) / "src/brickcell/block_inspector/hardware_cell.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(raw)
        for options in (["--check"], []):
            subprocess.run(
                ["git", "apply", *options, str(BUNDLE / "hardware_cell.patch")],
                cwd=folder,
                check=True,
                capture_output=True,
                text=True,
            )
        patched = target.read_bytes()
    if hashlib.sha256(patched).hexdigest() != baseline["patched_sha256"]:
        raise ValueError("Patched output hash differs from the tested revision")
    compile(patched, str(output), "exec")  # Syntax check only; never execute hardware code.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(patched)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local copied hardware_cell.py")
    parser.add_argument("--output", type=Path, required=True, help="New local review file")
    args = parser.parse_args()
    try:
        prepare(args.source, args.output)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Cannot prepare inspection patch: {exc}\n")
    print(f"Prepared {args.output}; not deployed. Requires brickcell.inspection.validity.")


if __name__ == "__main__":
    main()
