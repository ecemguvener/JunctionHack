"""Physical colour-label demonstration, not learned anomaly detection.

Uses operator labels and retains the existing robot/grasp/view safety checks.
Run only with the demo_1-sorter runtime; dashboard and HUD are disabled.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LABELS = {
    "red": "defective",
    "lime": "defective",
    "white": "defective",
    "yellow": "defective",
    "purple": "good",
    "brown": "good",
}
POLICY = "operator_colour_labels_not_anomaly_detection"


def identify_colour(mean_rgb):
    """Conservative colour rule on the existing localized brick's RGB sample."""
    values = np.asarray(mean_rgb, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Invalid brick colour sample")
    if np.any(values < 0) or np.any(values > 255):
        raise ValueError("Invalid RGB range")
    rgb = np.round(values).astype(np.uint8).reshape(1, 1, 3)
    h, s, v = map(int, cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0])
    if v < 60:
        raise ValueError("Brick colour too dark to identify")
    if s <= 35 and v >= 155:
        return "white"
    if s < 30:
        raise ValueError("Uncertain pale brick colour")
    ratio = values[1] / max(values[0], 1)
    if (h <= 5 or h >= 165) and ratio < 0.34:
        return "red"
    if 8 <= h <= 17:
        return "brown"
    if 20 <= h <= 35:
        return "yellow"
    if 37 <= h <= 80:
        return "lime"
    if 100 <= h <= 158 and values[0] / max(values[2], 1) >= 0.28:
        return "purple"
    raise ValueError(f"Unsupported or boundary colour: HSV={h},{s},{v}")


def install_policy(runtime):
    """Apply labels after a successful scan; never replace a failed safety check."""
    cls = runtime.PhysicalSO101CellExecutor
    localize = cls._skill_localize_next_cluster_block
    scan = cls._skill_present_and_spin_at_plate_camera
    sort = cls._skill_sort_block_to_pile

    def labelled_localize(self):
        self.operator_colour = None
        result = localize(self)
        if not result.succeeded or self._active_brick is None:
            return result
        try:
            self.operator_colour = identify_colour(self._active_brick.mean_rgb)
        except ValueError as exc:
            raise ValueError(f"operator_colour_reinspect:{exc}; no pickup authorized") from exc
        print(
            f"{POLICY}: {self.operator_colour} -> {LABELS[self.operator_colour]} "
            f"RGB={self._active_brick.mean_rgb}",
            flush=True,
        )
        return result

    def labelled_scan(self):
        if getattr(self, "operator_colour", None) not in LABELS:
            return runtime.SkillOutcome(False, reason="operator_colour_missing")
        result = scan(self)
        if result.succeeded and self.plate_scan_completed:
            self.anomaly_flagged = LABELS[self.operator_colour] == "defective"
            self.anomaly_messages = [f"{POLICY}:{self.operator_colour}"]
        return result

    def labelled_sort(self, pile):
        colour = getattr(self, "operator_colour", None)
        if colour not in LABELS or pile != LABELS[colour]:
            return runtime.SkillOutcome(False, reason="operator_colour_destination_mismatch")
        result = sort(self, pile)
        if result.succeeded and self.batch_sorted_records:
            self.batch_sorted_records[-1].update(
                operator_colour=colour,
                decision_source=POLICY,
                inspector_mode=POLICY,
                max_plate_anomaly_distance=None,
            )
        return result

    cls._skill_localize_next_cluster_block = labelled_localize
    cls._skill_present_and_spin_at_plate_camera = labelled_scan
    cls._skill_sort_block_to_pile = labelled_sort
    runtime.RealCameraAnomalyInspector.inspector_mode = property(lambda self: POLICY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cycles", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_cycles <= 10:
        parser.error("max-cycles must be between 1 and 10")
    from brickcell.block_inspector import hardware_cell as runtime

    install_policy(runtime)
    report = runtime.run_physical_cell_deployment(
        inspection_mode="lab-dark-mark",
        dry_run=args.dry_run,
        max_cycles=args.max_cycles,
        hud_port=0,
        dashboard_url="",
        camui_url="",
        output_dir=args.output_dir,
    )
    report.update(
        inspection_policy=POLICY,
        inspector_mode=POLICY,
        threshold_kind="operator_provided_colour_labels",
        lab_contrast_threshold=None,
        operator_colour_labels=LABELS,
        anomaly_accuracy_evaluated=False,
    )
    Path(report["report_json_path"]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
