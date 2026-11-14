"""Exercise readiness gates with bad data and the real import/training contracts."""

import csv
from pathlib import Path

import numpy as np
import pytest

from brickcell.data.manifest import load_manifest, split_manifest
from brickcell.policy.data import file_hash, load_episodes
from brickcell.workbench.__main__ import main
from brickcell.workbench.common import read_json, write_json
from brickcell.workbench.demo import create_demo
from brickcell.workbench.inspection import COLUMNS, import_csv, review_inspection
from brickcell.workbench.policy import review_policy
from brickcell.workbench.status import SIMULATION_STATUS, coverage_status
from brickcell.workbench.view import page

LIMITS = Path(__file__).resolve().parents[1] / "configs/workbench-sim-limits.json"


@pytest.fixture
def fixture_data(tmp_path):
    root = tmp_path / "dataset"
    create_demo(root)
    return root


def change_episode(root, mutate):
    manifest = root / "policy/manifest.json"
    raw = read_json(manifest)
    row = raw["episodes"][0]
    path = manifest.parent / row["path"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    mutate(arrays)
    np.savez_compressed(path, **arrays)
    row["sha256"] = file_hash(path)
    write_json(manifest, raw)
    return manifest


def test_complete_offline_workflow(fixture_data, tmp_path):
    source = fixture_data / "inspection/capture.csv"
    imported = fixture_data / "inspection/imported.json"
    assert import_csv(source, imported)["images"] == 36
    report, cards = review_inspection(imported)
    assert report["smoke_ready"] and not report["five_face_labels_complete"]
    assert report["missing_metadata"] == []
    assert {s.family for s in load_manifest(imported).samples} == {"synthetic_square"}
    assert len(cards) == 36
    policy = fixture_data / "policy/manifest.json"
    report, episodes = review_policy(policy, LIMITS)
    assert report["smoke_ready"] and len(episodes) == 3
    assert load_episodes(policy, "train")[0]["state"].shape == (30, 6)
    for kind, manifest in [("policy", policy), ("inspection", imported)]:
        output = tmp_path / kind
        extra = ["--limits", str(LIMITS)] if kind == "policy" else []
        assert (
            main([kind, "--manifest", str(manifest), "--output", str(output), "--strict", *extra])
            == 0
        )
        assert (output / "index.html").exists()
        assert read_json(output / "report.json")["smoke_ready"]
    assert (tmp_path / "inspection/patchcore-review.yaml").exists()
    assert main(["demo", "--output", str(fixture_data)]) == 2


@pytest.mark.parametrize(
    "damage, expected",
    [
        (lambda a: a.pop("wrist"), "wrist"),
        (lambda a: a.update(wrist=a["wrist"][:-1]), "synchronized"),
        (lambda a: a["timestamp"].__setitem__(4, a["timestamp"][3]), "timestamps"),
        (lambda a: a["timestamp"].__setitem__(4, 90), "timestamps"),
        (lambda a: a["state"].__setitem__((3, 1), np.nan), "Non-finite"),
        (lambda a: a["action"].__setitem__((4, 2), 6), "Joint-limit"),
        (lambda a: a["state"].__setitem__((5, 2), -6), "Joint-limit"),
        (lambda a: a.update(overview=a["overview"].astype(np.float32)), "uint8"),
    ],
)
def test_episode_faults(fixture_data, damage, expected):
    report, episodes = review_policy(change_episode(fixture_data, damage), LIMITS)
    assert not report["smoke_ready"]
    assert expected in str(report["errors"])
    assert episodes  # Invalid datasets still yield reviewable findings.


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda r: r.update(action_units="degrees"), "units"),
        (lambda r: r.update(fps=0), "fps"),
        (lambda r: r["episodes"][0].update(frames=999), "frame count"),
        (lambda r: r["episodes"][0].update(success=False), "Failed"),
        (lambda r: r["episodes"][1].update(seed=0), "leakage"),
        (lambda r: r["episodes"][0].update(path="../escape.npz"), "escapes"),
        (lambda r: r["episodes"][0].update(sha256="wrong"), "checksum"),
    ],
)
def test_policy_metadata_gates(fixture_data, mutation, expected):
    path = fixture_data / "policy/manifest.json"
    raw = read_json(path)
    mutation(raw)
    write_json(path, raw)
    report, _ = review_policy(path, LIMITS)
    assert not report["smoke_ready"] and expected in str(report["errors"])


def test_missing_limits_and_scope(fixture_data, tmp_path):
    manifest = fixture_data / "policy/manifest.json"
    report, _ = review_policy(manifest)
    assert not report["smoke_ready"] and "limits unverified" in str(report["errors"])
    assert (
        main(
            [
                "policy",
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "blocked"),
                "--strict",
            ]
        )
        == 1
    )
    raw = read_json(manifest)
    raw["synthetic"] = False
    write_json(manifest, raw)
    with pytest.raises(ValueError, match="Simulation limits"):
        review_policy(manifest, LIMITS)


def edit_csv(root, change):
    path = root / "inspection/capture.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    change(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda r: r[0].update(face="bottom"), "bottom excluded"),
        (lambda r: r[0].update(family=""), "family"),
        (lambda r: r[0].update(synthetic="maybe"), "true or false"),
        (lambda r: r[0].update(label="defective", defect_type=""), "defect_type"),
        (lambda r: r.append(dict(r[0])), "Duplicate"),
        (lambda r: r[-1].update(object_id=r[0]["object_id"]), "Split overlap"),
        (lambda r: r[-1].update(session_id=r[0]["session_id"]), "Split overlap"),
    ],
)
def test_import_rejects_bad_batches(fixture_data, mutation, expected):
    output = fixture_data / "inspection/bad.json"
    with pytest.raises(ValueError, match=expected):
        import_csv(edit_csv(fixture_data, mutation), output)
    assert not output.exists()


def test_unsplit_import_and_metadata_survive_split(fixture_data):
    def unsplit(rows):
        for i, row in enumerate(rows):
            row.update(split="unsplit", session_id=f"session-{i}")

    source = edit_csv(fixture_data, unsplit)
    output = fixture_data / "inspection/unsplit.json"
    import_csv(source, output)
    report, _ = review_inspection(output)
    assert report["valid_capture_batch"] and not report["smoke_ready"]
    target = output.with_name("split.json")
    assert split_manifest(output, target)["valid"]
    assert all(s.family and s.color for s in load_manifest(target).samples)


def test_coverage_never_pools_sessions(fixture_data):
    path = fixture_data / "inspection/manifest.json"
    raw = read_json(path)
    raw["samples"] = raw["samples"][:5]
    for i, row in enumerate(raw["samples"]):
        row.update(
            view_id=["front", "right", "back", "left", "top"][i],
            object_id="one",
            session_id="session-a" if i < 3 else "session-b",
        )
    write_json(path, raw)
    report, _ = review_inspection(path)
    assert not report["five_face_labels_complete"] and len(report["coverage"]) == 2
    for row in raw["samples"]:
        row["session_id"] = "session-a"
    write_json(path, raw)
    report, _ = review_inspection(path)
    assert report["five_face_labels_complete"]
    assert not report["station_ready"]


def test_missing_corrupt_image_and_duplicate_hash(fixture_data):
    path = fixture_data / "inspection/manifest.json"
    raw = read_json(path)
    image = path.parent / raw["samples"][0]["path"]
    image.write_bytes(b"broken image")
    report, cards = review_inspection(path)
    assert not report["smoke_ready"] and cards[0]["preview"] is None
    image.unlink()
    report, _ = review_inspection(path)
    assert "missing/corrupt" in str(report["errors"])


def test_report_escapes_untrusted_text(tmp_path):
    target = tmp_path / "review.html"
    attack = '</script><script>alert("oops")</script>'
    page(target, attack, "", {"value": attack})
    assert attack not in target.read_text()
    assert "\\u003c/script>" in target.read_text()


def test_truthful_simulation_status():
    assert "scripted IK" in SIMULATION_STATUS and "not a physical arm" in SIMULATION_STATUS
    assert "0 fixed" in coverage_status(False)
    assert "1 fixed" in coverage_status(True)
    assert "0/5" in coverage_status(True)


def test_malformed_cli_input(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json")
    assert main(["policy", "--manifest", str(path), "--output", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


def test_timestamp_shape_is_not_synchronization(fixture_data):
    path = change_episode(fixture_data, lambda a: a.update(timestamp=a["timestamp"][:, None]))
    report, _ = review_policy(path, LIMITS)
    assert not report["smoke_ready"]
    assert "one-dimensional" in str(report["errors"])


def test_invalid_episode_container_is_clear(fixture_data, tmp_path):
    path = fixture_data / "policy/manifest.json"
    raw = read_json(path)
    raw["episodes"] = ["wrong"]
    write_json(path, raw)
    assert main(["policy", "--manifest", str(path), "--output", str(tmp_path / "out")]) == 2


def test_nonfinite_action_still_produces_review(fixture_data, tmp_path):
    path = change_episode(fixture_data, lambda a: a["action"].__setitem__((2, 1), np.inf))
    output = tmp_path / "nonfinite"
    assert (
        main(
            [
                "policy",
                "--manifest",
                str(path),
                "--limits",
                str(LIMITS),
                "--output",
                str(output),
                "--strict",
            ]
        )
        == 1
    )
    assert "Non-finite" in str(read_json(output / "report.json")["errors"])
    assert (output / "index.html").exists()


def test_state_units_rejected(fixture_data):
    path = fixture_data / "policy/manifest.json"
    raw = read_json(path)
    raw["state_units"] = "degrees"
    write_json(path, raw)
    report, _ = review_policy(path, LIMITS)
    assert not report["smoke_ready"] and "state units" in str(report["errors"])
