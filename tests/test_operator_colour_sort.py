import importlib.util
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("operator_colour", "scripts/operator_colour_sort.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.mark.parametrize(
    ("rgb", "name"),
    [
        ((230, 20, 20), "red"),
        ((180, 230, 50), "lime"),
        ((225, 225, 225), "white"),
        ((240, 200, 20), "yellow"),
        ((140, 80, 200), "purple"),
        ((160, 175, 200), "purple"),
        ((145, 75, 30), "brown"),
    ],
)
def test_operator_mapping(rgb, name):
    assert policy.identify_colour(rgb) == name
    assert policy.LABELS[name] == ("good" if name in {"purple", "brown"} else "defective")


@pytest.mark.parametrize("rgb", [(0, 0, 0), (0, 0, 255), (float("nan"), 0, 0), (300, 0, 0)])
def test_ambiguous_colour_refused(rgb):
    with pytest.raises(ValueError):
        policy.identify_colour(rgb)


@pytest.mark.parametrize("scan_success", [True, False])
def test_failed_scan_never_overridden_and_labelled_sort_gates(scan_success):
    calls = []

    class Cell:
        def _skill_localize_next_cluster_block(self):
            self._active_brick = SimpleNamespace(mean_rgb=(145, 75, 30))
            return SimpleNamespace(succeeded=True)

        def _skill_present_and_spin_at_plate_camera(self):
            self.plate_scan_completed = scan_success
            return SimpleNamespace(succeeded=scan_success)

        def _skill_sort_block_to_pile(self, pile):
            calls.append(pile)
            self.batch_sorted_records.append({})
            return SimpleNamespace(succeeded=True)

    runtime = SimpleNamespace(
        PhysicalSO101CellExecutor=Cell,
        RealCameraAnomalyInspector=type("Inspector", (), {}),
        SkillOutcome=lambda succeeded, **kw: SimpleNamespace(succeeded=succeeded, **kw),
    )
    policy.install_policy(runtime)
    cell = Cell()
    cell.anomaly_flagged = True
    cell.batch_sorted_records = []
    cell._skill_localize_next_cluster_block()
    result = cell._skill_present_and_spin_at_plate_camera()
    assert result.succeeded == scan_success
    assert cell.anomaly_flagged == (not scan_success)
    assert not cell._skill_sort_block_to_pile("defective").succeeded
    assert not calls
    if scan_success:
        assert cell._skill_sort_block_to_pile("good").succeeded
        assert calls == ["good"]
        assert cell.batch_sorted_records[0]["decision_source"] == policy.POLICY
