import json

from brickcell.cli import main
from brickcell.metrics.report import classification, latency


def test_complete_mock_cli(tmp_path, capsys):
    assert main(["demo", "--mock", "--output", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["completed_mock_cycles"] == 3
    assert report["reinspections"] == 1
    assert report["inspector_mode"] == "MOCK_SCRIPTED_NOT_ML"
    events = [
        json.loads(line) for line in next(tmp_path.glob("events-*.jsonl")).read_text().splitlines()
    ]
    assert any(e["event"] == "inspection" for e in events)
    assert "COMPLETE" in capsys.readouterr().out


def test_malformed_input_exit(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("not json")
    assert main(["validate-data", "--manifest", str(path)]) == 1
    assert "brickcell:" in capsys.readouterr().err


def test_metrics():
    report = classification([0, 0, 1, 1], [0.1, 0.8, 0.3, 0.9], 0.5)
    assert report["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert report["defect_miss_rate"] == 0.5
    assert latency([1])["p50_ms"] is None
    assert latency([1, 2])["p95_ms"] is None
    assert latency(list(range(20)))["p95_ms"] is not None
