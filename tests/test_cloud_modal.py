"""Offline checks for the Modal runner in cloud/. No network access and no Modal account."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "cloud"
sys.path.insert(0, str(CLOUD))

import modal_helpers as helpers  # noqa: E402


def test_run_id_is_sortable_and_unique():
    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)
    first, second = helpers.new_run_id(now), helpers.new_run_id(now)
    assert first.startswith("20261114T120000Z-")
    assert first != second or len(first) == len("20261114T120000Z-0000")


def test_git_state_and_source_digest(tmp_path):
    state = helpers.git_state()
    assert set(state) == {"commit", "branch", "dirty", "changed_files", "changed_file_count"}
    assert state["commit"] is None or len(state["commit"]) == 40
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "a.pyc").write_bytes(b"cache")
    first = helpers.source_digest(tmp_path, ("src",))
    assert first["files"] == 1
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert helpers.source_digest(tmp_path, ("src",))["sha256"] != first["sha256"]
    outside = helpers.git_state(tmp_path)
    assert outside["commit"] is None


def test_move_aside_keeps_old_results(tmp_path):
    target = tmp_path / "patchcore-sim"
    assert helpers.move_aside(target) is None
    target.mkdir()
    (target / "model.xml").write_text("old")
    moved = helpers.move_aside(target, stamp="20261114T000000Z")
    assert moved.name == "patchcore-sim.previous-20261114T000000Z"
    assert (moved / "model.xml").read_text() == "old" and not target.exists()
    target.mkdir()
    assert helpers.move_aside(target, stamp="20261114T000000Z").name.endswith("-1")


def test_local_targets_use_repo_paths(tmp_path):
    patchcore = helpers.local_targets("patchcore", tmp_path, model_name="m", dataset="d")
    assert patchcore == {"model": tmp_path / "artifacts/models/m", "data": tmp_path / "data/d"}
    act = helpers.local_targets("act", tmp_path, policy_name="act-modal", demos="demos")
    assert act["policy"] == tmp_path / "artifacts/policy/act-modal"
    with pytest.raises(ValueError):
        helpers.local_targets("unknown", tmp_path)


def test_plan_shards_keeps_seeds_and_minimum_size():
    plan = helpers.plan_shards(10, 3, 4200)
    assert [p["episodes"] for p in plan] == [4, 3, 3]
    assert [p["seed"] for p in plan] == [4200, 4204, 4207]
    assert helpers.plan_shards(6, 8, 0) == [
        {"shard": 0, "first_episode": 0, "episodes": 3, "seed": 0},
        {"shard": 1, "first_episode": 3, "episodes": 3, "seed": 3},
    ]
    with pytest.raises(ValueError):
        helpers.plan_shards(2, 1, 0)


def _shard_manifest(plan):
    rows = [
        {
            "episode_id": f"episode-{i:04d}",
            "path": f"episode-{i:04d}.npz",
            "seed": plan["seed"] + i,
            "split": helpers.split_for(i, plan["episodes"]),
            "frames": 10,
            "success": True,
            "sha256": f"hash-{plan['seed'] + i}",
        }
        for i in range(plan["episodes"])
    ]
    return {"schema_version": 1, "fps": 50, "episodes": rows}


def test_merge_policy_manifests_renames_and_splits_like_one_run():
    plans = helpers.plan_shards(7, 2, 4200)
    merged, moves = helpers.merge_policy_manifests([(p, _shard_manifest(p)) for p in plans])
    rows = merged["episodes"]
    assert [r["episode_id"] for r in rows] == [f"episode-{i:04d}" for i in range(7)]
    assert [r["seed"] for r in rows] == list(range(4200, 4207))
    assert [r["split"] for r in rows] == ["train"] * 5 + ["val", "test"]
    assert moves[4] == ("shard-01/episode-0000.npz", "episode-0004.npz")
    assert merged["fps"] == 50 and merged["parallel_shards"] == 2
    broken = _shard_manifest(plans[1])
    broken["episodes"][0]["seed"] = 1
    with pytest.raises(ValueError):
        helpers.merge_policy_manifests([(plans[0], _shard_manifest(plans[0])), (plans[1], broken)])


def test_generate_units_follow_generator_order():
    units = helpers.generate_units()
    assert len(units) == 12 and units[0] == "train:0" and units[6] == "val:0"
    assert units[-1] == "test:2"
    units = helpers.generate_units(helpers.generate_episodes("all"))
    assert len(units) == 90 and units[36] == "val:0" and units[-1] == "test:26"
    assert helpers.generate_episodes("off") == (6, 3, 3)
    assert helpers.generate_episodes("main", "2,1,1") == (2, 1, 1)
    for bad in [("rainbow", ""), ("all", "1,2"), ("all", "0,1,1")]:
        with pytest.raises(ValueError):
            helpers.generate_episodes(*bad)


def test_episode_plans_match_generator():
    generate = pytest.importorskip("brickcell.simulation.generate")
    assert helpers.LEGACY_EPISODES == generate.LEGACY_EPISODES
    assert helpers.PALETTE_EPISODES == generate.PALETTE_EPISODES
    assert helpers.PALETTE_MODES == generate.PALETTE_MODES
    assert helpers.SPLIT_NAMES == generate.SPLIT_NAMES
    episodes = (3, 2, 2)
    assert helpers.generate_units(episodes) == [f"{s}:{e}" for s, e in generate.units(episodes)]


def test_generate_shard_uses_the_generator_episode_function():
    import inspect

    import generate_shard

    source = inspect.getsource(generate_shard)
    assert "generate.render_episode(" in source and "generate.build_manifest(" in source
    assert "GENERATE_SHA256" not in source


def test_per_color_report_groups_test_samples():
    manifest = {
        "sample_metadata": {
            "a": {"color": "red"},
            "b": {"color": "red"},
            "c": {"color": "black"},
            "d": {"color": None},
        }
    }
    report = {
        "samples": [
            {"image_id": "a", "score": 0.5, "label": 0, "decision": "GOOD"},
            {"image_id": "b", "score": 3.0, "label": 1, "decision": "DEFECTIVE"},
            {"image_id": "c", "score": 2.0, "label": 0, "decision": "DEFECTIVE"},
            {"image_id": "d", "score": 1.0, "label": 1, "decision": "REINSPECT"},
        ]
    }
    result = helpers.per_color_report(manifest, report)
    assert list(result) == ["black", "red", "scene"]
    assert result["red"]["TP"] == result["red"]["TN"] == 1 and result["red"]["accuracy"] == 1.0
    assert result["black"]["FP"] == 1 and result["black"]["accuracy"] == 0.0
    assert result["scene"]["REINSPECT"] == 1 and result["scene"]["accuracy"] is None
    assert result["red"]["defective_score_range"] == [3.0, 3.0]


def test_patchcore_config_is_readable_as_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    base = {"manifest": "../data/two-bowl/manifest.json", "output_dir": "../x", "seed": 42}
    config = helpers.patchcore_config(base, tmp_path / "m.json", tmp_path / "out")
    path = helpers.write_json_yaml(tmp_path / "cfg.yaml", config)
    loaded = yaml.safe_load(path.read_text())
    assert loaded == {
        **base,
        "manifest": str(tmp_path / "m.json"),
        "output_dir": str(tmp_path / "out"),
    }
    assert json.loads(path.read_text()) == loaded


def test_modal_app_imports_without_network():
    pytest.importorskip("modal")
    import modal_train

    assert modal_train.app.name == "brickcell-train"
    assert modal_train.PATCHCORE_GPU[0] == "L4"
    for name in (
        "generate_shard_job",
        "merge_dataset_job",
        "train_patchcore_job",
        "train_act_job",
        "evaluate_act_job",
    ):
        assert callable(getattr(modal_train, name).remote)
