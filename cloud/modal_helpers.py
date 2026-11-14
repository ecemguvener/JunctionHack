"""Local helpers for cloud/modal_train.py. This module does not import modal.

The helpers find the repository, record the state of the local tree (provenance),
plan parallel shards, merge shard manifests and move old local results aside.
The tests in tests/test_cloud_modal.py use them without network access.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# These directories go into each Modal container at start time, from the local tree.
# A change in them does not cause an image build.
SOURCE_DIRS = ("src", "configs", "assets", "policy", "cloud")
SOURCE_IGNORE = ["**/__pycache__", "**/*.pyc", "**/.venv", "**/.pytest_cache", "**/.ruff_cache"]


# Episodes per split (train, val, test) of brickcell.simulation.generate. This module does not
# import generate.py (it needs numpy); tests/test_cloud_modal.py checks that the values agree.
SPLIT_NAMES = ("train", "val", "test")
LEGACY_EPISODES = (6, 3, 3)  # --palette off
PALETTE_EPISODES = (36, 27, 27)  # --palette main / all
PALETTE_MODES = ("off", "main", "all")


def generate_episodes(palette: str = "all", episodes: str = "") -> tuple[int, int, int]:
    """Episodes per split. ``episodes``: "TRAIN,VAL,TEST"; empty = the default of the palette."""
    if palette not in PALETTE_MODES:
        raise ValueError(f"palette must be one of {PALETTE_MODES}, got {palette!r}")
    if not episodes:
        return LEGACY_EPISODES if palette == "off" else PALETTE_EPISODES
    values = tuple(int(v) for v in episodes.split(","))
    if len(values) != 3 or min(values) < 1:
        raise ValueError(f"episodes must be TRAIN,VAL,TEST with values >= 1, got {episodes!r}")
    return values


def generate_units(episodes: tuple[int, int, int] = LEGACY_EPISODES) -> list[str]:
    """All "split:episode" units of the PatchCore generator, in its order."""
    return [f"{split}:{e}" for split, count in zip(SPLIT_NAMES, episodes) for e in range(count)]


def per_color_report(manifest: dict, test_report: dict, threshold: float | None = None) -> dict:
    """Test results grouped by brick colour (manifest ``sample_metadata``).

    A sample is positive (defective) if its decision is DEFECTIVE. REINSPECT counts apart.
    """
    metadata = manifest.get("sample_metadata") or {}
    groups: dict[str, dict] = {}
    for row in test_report.get("samples", []):
        color = (metadata.get(row["image_id"]) or {}).get("color") or "scene"
        group = groups.setdefault(
            color, {"TP": 0, "FN": 0, "TN": 0, "FP": 0, "REINSPECT": 0, "scores": {}}
        )
        decision = row.get("decision")
        if decision is None and threshold is not None:
            decision = "DEFECTIVE" if row["score"] >= threshold else "GOOD"
        if decision == "REINSPECT":
            group["REINSPECT"] += 1
        elif row["label"] == 1:
            group["TP" if decision == "DEFECTIVE" else "FN"] += 1
        else:
            group["FP" if decision == "DEFECTIVE" else "TN"] += 1
        kind = "defective" if row["label"] == 1 else "good"
        group["scores"].setdefault(kind, []).append(round(float(row["score"]), 4))
    for group in groups.values():
        scores = group.pop("scores")
        for kind, values in scores.items():
            group[f"{kind}_score_range"] = [min(values), max(values)]
        total = group["TP"] + group["FN"] + group["TN"] + group["FP"]
        group["accuracy"] = round((group["TP"] + group["TN"]) / total, 4) if total else None
    return dict(sorted(groups.items()))


def new_run_id(now: datetime | None = None) -> str:
    """Return a sortable run ID, for example 20261114T120000Z-1a2b."""
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"


def utc_stamp(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_state(repo: Path = REPO) -> dict:
    """Commit, branch and dirty flag of the local tree. The values are None without git."""
    commit = _git(["rev-parse", "HEAD"], repo)
    status = _git(["status", "--porcelain", "--untracked-files=normal"], repo)
    changed = [line[3:] for line in (status or "").splitlines() if line.strip()]
    return {
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
        "dirty": bool(changed) if status is not None else None,
        "changed_files": changed[:200],
        "changed_file_count": len(changed),
    }


def _ignored(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & {"__pycache__", ".venv", ".pytest_cache", ".ruff_cache"}) or (
        path.suffix == ".pyc"
    )


def source_digest(repo: Path = REPO, dirs: tuple[str, ...] = SOURCE_DIRS) -> dict:
    """SHA-256 over the files that go to Modal. It identifies a dirty tree exactly."""
    digest = hashlib.sha256()
    count = 0
    for name in dirs:
        root = repo / name
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(repo)
            if _ignored(relative):
                continue
            digest.update(str(relative).encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            count += 1
    return {"sha256": digest.hexdigest(), "files": count, "dirs": list(dirs)}


def local_provenance(repo: Path = REPO) -> dict:
    """Provenance of the local tree that the Modal run uses."""
    return {"git": git_state(repo), "source": source_digest(repo)}


def move_aside(target: Path, stamp: str | None = None) -> Path | None:
    """Rename an existing local result to <name>.previous-<UTC stamp>. Never delete it."""
    if not target.exists():
        return None
    stamp = stamp or utc_stamp()
    backup = target.with_name(f"{target.name}.previous-{stamp}")
    index = 1
    while backup.exists():
        backup = target.with_name(f"{target.name}.previous-{stamp}-{index}")
        index += 1
    target.rename(backup)
    return backup


def plan_shards(episodes: int, shards: int, seed: int, minimum: int = 3) -> list[dict]:
    """Split policy episodes into shards. Each shard has at least `minimum` episodes.

    record_policy needs at least 3 episodes per call. Seeds stay the same as in one
    local run: episode i always uses seed + i.
    """
    if episodes < minimum:
        raise ValueError(f"Use at least {minimum} episodes")
    shards = max(1, min(shards, episodes // minimum))
    base, extra = divmod(episodes, shards)
    plan, start = [], 0
    for index in range(shards):
        count = base + (1 if index < extra else 0)
        plan.append(
            {"shard": index, "first_episode": start, "episodes": count, "seed": seed + start}
        )
        start += count
    return plan


def split_for(index: int, total: int) -> str:
    """The split rule of brickcell.simulation.record_policy: last two are val and test."""
    return "train" if index < total - 2 else "val" if index == total - 2 else "test"


def merge_policy_manifests(shards: list[tuple[dict, dict]]) -> tuple[dict, list[tuple[str, str]]]:
    """Merge shard manifests into one manifest with global episode names and splits.

    `shards` is a list of (plan entry, shard manifest), in shard order. The result is
    (merged manifest, [(shard-relative source path, new file name)]). File contents and
    hashes do not change; only names and splits change.
    """
    if not shards:
        raise ValueError("No shards to merge")
    rows, moves = [], []
    for plan, manifest in shards:
        for local_index, row in enumerate(manifest["episodes"]):
            index = plan["first_episode"] + local_index
            if row["seed"] != plan["seed"] + local_index:
                raise ValueError(f"Shard {plan['shard']} seed mismatch at episode {local_index}")
            name = f"episode-{index:04d}"
            moves.append((f"shard-{plan['shard']:02d}/{row['path']}", f"{name}.npz"))
            rows.append({**row, "episode_id": name, "path": f"{name}.npz"})
    for index, row in enumerate(rows):
        row["split"] = split_for(index, len(rows))
    merged = {key: value for key, value in shards[0][1].items() if key != "episodes"}
    merged["episodes"] = rows
    merged["parallel_shards"] = len(shards)
    return merged, moves


def patchcore_config(base: dict, manifest: str | Path, output_dir: str | Path) -> dict:
    """Copy of a PatchCore config with absolute manifest and output paths."""
    return {**base, "manifest": str(manifest), "output_dir": str(output_dir)}


def write_json_yaml(path: Path, data: dict) -> Path:
    """Write a config as JSON. YAML loaders read JSON, so no YAML package is necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def local_targets(kind: str, repo: Path = REPO, **names: str) -> dict[str, Path]:
    """Local paths for downloaded results. They are the paths the repo already uses."""
    if kind == "patchcore":
        return {
            "model": repo / "artifacts" / "models" / names["model_name"],
            "data": repo / "data" / names["dataset"],
        }
    if kind == "act":
        return {
            "policy": repo / "artifacts" / "policy" / names["policy_name"],
            "data": repo / "data" / names["demos"],
        }
    if kind == "demos":
        return {"data": repo / "data" / names["demos"]}
    raise ValueError(f"Unknown result kind: {kind}")


def run_record_dir(run_id: str, repo: Path = REPO) -> Path:
    return repo / "artifacts" / "modal" / "runs" / run_id
