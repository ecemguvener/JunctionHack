"""Create evidence of engineering work without inventing authorship or test results."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def report(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    inventory = []
    for name in git("ls-files", "src", "scripts", "tests", "configs", "uv.lock").splitlines():
        path = root / name
        if path.is_file():
            inventory.append(
                {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
    commits = []
    for line in git("log", "-30", "--format=%H%x09%an%x09%s").splitlines():
        commit, author, subject = line.split("\t", 2)
        commits.append({"commit": commit, "git_author": author, "subject": subject})
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_source_dirty": bool(git("diff", "--name-only")),
        "source_inventory": inventory,
        "recent_commits": commits,
        "attribution": "Git authorship only; AI involvement is documented separately, not inferred.",
        "test_results": None,
        "limitations": "Hashes and history do not prove correctness, deployment or physical success.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output file to preserve earlier evidence")
    result = report(Path(__file__).resolve().parents[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
