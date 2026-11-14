"""SymSkill-native Python 3.12 CogMan bilevel planner and ExpectedAtomsMonitor.

Supports two deterministic, deadlock-free STRIPS modes:
1. `plate_spin_sort` (default):
   LocalizeBlock -> PickBlock -> PresentAndSpinAtPlateCamera (CCW Joint-5 sweep)
   -> SortBlockToPile (good vs defective KALAS plate based on anomaly signal) -> ParkAtRest.
2. `eight_corner` (legacy tabletop reorientation):
   LocalizeBlock -> InspectExposedCorner -> ReorientBlockYaw -> GraspPresentAndFlipBlock -> ParkAtRest.
"""

from __future__ import annotations

import base64
import io
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from PIL import Image

from brickcell.block_inspector.ledger import EightCornerCoverageLedger

InspectionMode = Literal["clear_source_plate", "plate_spin_sort", "eight_corner"]


@dataclass(frozen=True)
class SymbolicGoal:
    instruction: str
    block_id: str
    mode: InspectionMode = "plate_spin_sort"
    require_all_8_corners: bool = False
    require_park_at_rest: bool = True


def compile_instruction(
    instruction: str,
    block_id: str = "block_0",
    mode: InspectionMode | None = None,
) -> SymbolicGoal:
    """Compile natural-language objective into grounded symbolic goal atoms."""
    text = instruction.strip().lower()
    if not text:
        raise ValueError("Instruction cannot be empty")
    resolved_mode: InspectionMode
    if mode is not None:
        resolved_mode = mode
    elif "eight_corner" in text or "8-corner" in text or "flip" in text:
        resolved_mode = "eight_corner"
    elif (
        "cluster" in text
        or "source plate" in text
        or "8 blocks" in text
        or "all blocks" in text
        or "each block" in text
    ):
        resolved_mode = "clear_source_plate"
    else:
        resolved_mode = "plate_spin_sort"
    return SymbolicGoal(
        instruction=instruction,
        block_id=block_id,
        mode=resolved_mode,
        require_all_8_corners=(resolved_mode == "eight_corner"),
        require_park_at_rest=True,
    )


@dataclass(frozen=True)
class GroundedOperator:
    """Grounded STRIPS / NSRT skill invocation."""

    name: str
    block_id: str
    params: dict = field(default_factory=dict)

    def expected_atoms(self) -> set[str]:
        b = self.block_id
        if self.name == "LocalizeNextClusterBlock":
            return {f"BlockLocalized({b})"}
        if self.name == "LocalizeBlock":
            return {f"BlockLocalized({b})", f"BlockOnTable({b})"}
        if self.name == "PickBlock":
            return {f"BlockHeld({b})"}
        if self.name == "PresentAndSpinAtPlateCamera":
            return {f"BlockHeld({b})", f"PlateScanCompleted({b})"}
        if self.name == "SortBlockToPile":
            pile = self.params.get("pile", "good")
            pile_atom = (
                f"BlockSortedDefective({b})" if pile == "defective" else f"BlockSortedGood({b})"
            )
            return {"HandEmpty()", f"BlockSorted({b})", pile_atom}
        if self.name == "InspectExposedCorner":
            c = self.params["corner_index"]
            return {f"CornerInspected({b},c_{c})"}
        if self.name == "ReorientBlockYaw":
            return {f"BlockOnTable({b})", "HandEmpty()"}
        if self.name == "GraspPresentAndFlipBlock":
            return {f"BlockOnTable({b})", "HandEmpty()"}
        if self.name == "ParkAtRest":
            return {f"BlockAtRest({b})"}
        return set()


@dataclass(frozen=True)
class SymbolicState:
    """Online predicate evaluation snapshot over perception, anomaly channel, and robot state."""

    block_id: str
    hand_empty: bool
    block_localized: bool
    block_on_table: bool
    block_held: bool
    block_at_rest: bool
    corner_upper: tuple[bool, ...] = (False,) * 8
    corner_wrist_reachable: tuple[bool, ...] = (False,) * 8
    corner_inspected: tuple[bool, ...] = (False,) * 8
    plate_scan_completed: bool = False
    anomaly_flagged: bool = False
    sorted_pile: str | None = None
    source_plate_cleared: bool = False
    remaining_source_blocks: int = 0
    sorted_good_count: int = 0
    sorted_defective_count: int = 0

    def atoms(self) -> set[str]:
        b = self.block_id
        out: set[str] = set()
        if self.hand_empty:
            out.add("HandEmpty()")
        if self.source_plate_cleared:
            out.add("SourcePlateCleared()")
        if self.block_localized:
            out.add(f"BlockLocalized({b})")
        if self.block_on_table:
            out.add(f"BlockOnTable({b})")
        if self.block_held:
            out.add(f"BlockHeld({b})")
        if self.block_at_rest:
            out.add(f"BlockAtRest({b})")
        if self.plate_scan_completed:
            out.add(f"PlateScanCompleted({b})")
        if self.anomaly_flagged:
            out.add(f"AnomalyFlagged({b})")
        if self.sorted_pile is not None:
            out.add(f"BlockSorted({b})")
            if self.sorted_pile == "defective":
                out.add(f"BlockSortedDefective({b})")
            elif self.sorted_pile == "good":
                out.add(f"BlockSortedGood({b})")
        for i in range(8):
            if self.corner_upper[i]:
                out.add(f"CornerUpperHemisphere({b},c_{i})")
            if self.corner_wrist_reachable[i]:
                out.add(f"CornerWristReachable({b},c_{i})")
            if self.corner_inspected[i]:
                out.add(f"CornerInspected({b},c_{i})")
        return out

    def goal_satisfied(self, goal: SymbolicGoal) -> bool:
        if goal.mode == "clear_source_plate":
            if not self.source_plate_cleared or not self.hand_empty:
                return False
            if goal.require_park_at_rest and not self.block_at_rest:
                return False
            return True

        if goal.mode == "plate_spin_sort":
            if not self.block_localized or not self.plate_scan_completed or not self.hand_empty:
                return False
            expected_pile = "defective" if self.anomaly_flagged else "good"
            if self.sorted_pile != expected_pile:
                return False
            if goal.require_park_at_rest and not self.block_at_rest:
                return False
            return True

        if not self.block_localized or not self.block_on_table or not self.hand_empty:
            return False
        if goal.require_all_8_corners and not all(self.corner_inspected):
            return False
        if goal.require_park_at_rest and not self.block_at_rest:
            return False
        return True


@dataclass(frozen=True)
class SkillOutcome:
    succeeded: bool
    reason: str = "ok"
    metadata: dict = field(default_factory=dict)


class SkillExecutor(Protocol):
    """Low-level execution interface implemented by MuJoCo 4.0.1 Twin or babyros/armd runtime."""

    def perceive_symbolic_state(self, block_id: str) -> SymbolicState: ...
    def current_arm_q(self) -> np.ndarray: ...
    def corner_seed_q(self, corner_index: int) -> np.ndarray | None: ...
    def execute_operator(
        self, op: GroundedOperator, cone_perturbation: np.ndarray | None = None
    ) -> SkillOutcome: ...


def plan_strips_sequence(
    state: SymbolicState,
    goal: SymbolicGoal,
    current_q: np.ndarray | None = None,
    corner_q_map: dict[int, np.ndarray] | None = None,
) -> list[GroundedOperator]:
    """Deadlock-free STRIPS planner for `clear_source_plate`, `plate_spin_sort`, and `eight_corner`."""
    b = goal.block_id
    if state.goal_satisfied(goal):
        return []

    if goal.mode == "clear_source_plate":
        if not state.source_plate_cleared:
            if not state.block_localized or state.sorted_pile is not None:
                return [GroundedOperator("LocalizeNextClusterBlock", b)]
            if not state.block_held:
                return [GroundedOperator("PickBlock", b)]
            if not state.plate_scan_completed:
                return [GroundedOperator("PresentAndSpinAtPlateCamera", b)]
            target_pile = "defective" if state.anomaly_flagged else "good"
            return [GroundedOperator("SortBlockToPile", b, {"pile": target_pile})]
        if goal.require_park_at_rest and not state.block_at_rest:
            return [GroundedOperator("ParkAtRest", b)]
        return []

    if not state.block_localized:
        return [GroundedOperator("LocalizeBlock", b)]

    if goal.mode == "plate_spin_sort":
        if state.sorted_pile is None:
            if not state.block_held:
                return [GroundedOperator("PickBlock", b)]
            if not state.plate_scan_completed:
                return [GroundedOperator("PresentAndSpinAtPlateCamera", b)]
            target_pile = "defective" if state.anomaly_flagged else "good"
            return [GroundedOperator("SortBlockToPile", b, {"pile": target_pile})]
        if goal.require_park_at_rest and not state.block_at_rest:
            return [GroundedOperator("ParkAtRest", b)]
        return []

    uninspected_upper_reachable = [
        i
        for i in range(8)
        if state.corner_upper[i]
        and state.corner_wrist_reachable[i]
        and not state.corner_inspected[i]
    ]
    uninspected_upper_unreachable = [
        i
        for i in range(8)
        if state.corner_upper[i]
        and (not state.corner_wrist_reachable[i])
        and not state.corner_inspected[i]
    ]
    uninspected_bottom = [
        i for i in range(8) if (not state.corner_upper[i]) and not state.corner_inspected[i]
    ]

    plan: list[GroundedOperator] = []
    if uninspected_upper_reachable:
        remaining = list(uninspected_upper_reachable)
        q_cursor = current_q.copy() if current_q is not None else np.zeros(5, dtype=np.float64)
        while remaining:
            if corner_q_map:
                nxt = min(
                    remaining,
                    key=lambda idx: float(
                        np.linalg.norm(corner_q_map.get(idx, q_cursor) - q_cursor)
                    ),
                )
                q_cursor = corner_q_map.get(nxt, q_cursor)
            else:
                nxt = remaining[0]
            remaining.remove(nxt)
            plan.append(GroundedOperator("InspectExposedCorner", b, {"corner_index": nxt}))
        return plan

    if uninspected_upper_unreachable:
        plan.append(
            GroundedOperator(
                "ReorientBlockYaw",
                b,
                {
                    "delta_yaw_rad": math.pi,
                    "target_corner": int(uninspected_upper_unreachable[0]),
                },
            )
        )
        return plan

    if uninspected_bottom:
        plan.append(GroundedOperator("GraspPresentAndFlipBlock", b, {}))
        return plan

    if goal.require_park_at_rest and not state.block_at_rest:
        plan.append(GroundedOperator("ParkAtRest", b, {}))
    return plan


class ExpectedAtomsMonitor:
    """Verifies post-skill symbolic atoms and resamples 3-facet cone viewpoints on failure."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        self.retry_counts: dict[int, int] = {}

    def verify(self, op: GroundedOperator, post_state: SymbolicState) -> bool:
        achieved = post_state.atoms()
        if op.name == "LocalizeNextClusterBlock":
            return "SourcePlateCleared()" in achieved or (
                f"BlockLocalized({op.block_id})" in achieved
                and f"BlockOnTable({op.block_id})" in achieved
            )
        return op.expected_atoms().issubset(achieved)

    def resample_cone_perturbation(self, corner_index: int) -> np.ndarray:
        """Perturb 3-facet visibility cone reference (alpha_1, alpha_2, alpha_3) inside alpha_f >= 0.25."""
        count = self.retry_counts.get(corner_index, 0) + 1
        self.retry_counts[corner_index] = count
        return self.rng.uniform(-0.14, 0.14, size=3).astype(np.float64)


@dataclass
class InspectionRunReport:
    instruction: str
    block_id: str
    succeeded: bool
    inspected_corners: int
    replans: int
    executed_trace: list[dict]
    elapsed_s: float
    ledger_summary: dict
    mode: str = "plate_spin_sort"
    anomaly_flagged: bool = False
    sorted_pile: str | None = None
    sorted_counts: dict[str, int] = field(default_factory=dict)
    batch_sorted_records: list[dict] = field(default_factory=list)
    report_json_path: str | None = None
    report_html_path: str | None = None


class CogManInspectionAgent:
    """Closed-loop neuro-symbolic inspection controller (Perceiver -> Planner -> Skill -> Monitor)."""

    def __init__(
        self,
        executor: SkillExecutor,
        ledger: EightCornerCoverageLedger,
        max_replans: int = 12,
        max_steps: int = 32,
        seed: int = 7,
    ) -> None:
        self.executor = executor
        self.ledger = ledger
        self.max_replans = max_replans
        self.max_steps = max_steps
        self.monitor = ExpectedAtomsMonitor(seed=seed)

    def run(
        self,
        instruction: str = "investigate this block",
        block_id: str = "block_0",
        output_dir: str | Path | None = None,
        mode: InspectionMode | None = None,
    ) -> InspectionRunReport:
        t0 = time.monotonic()
        goal = compile_instruction(instruction, block_id=block_id, mode=mode)
        effective_max_steps = (
            max(self.max_steps, 96) if goal.mode == "clear_source_plate" else self.max_steps
        )
        effective_max_replans = (
            max(self.max_replans, 16) if goal.mode == "clear_source_plate" else self.max_replans
        )
        trace: list[dict] = []
        replans = 0
        pending_perturbation: dict[int, np.ndarray] = {}

        state = self.executor.perceive_symbolic_state(block_id)
        steps = 0
        while (
            not state.goal_satisfied(goal)
            and steps < effective_max_steps
            and replans <= effective_max_replans
        ):
            corner_q_map = (
                {i: q for i in range(8) if (q := self.executor.corner_seed_q(i)) is not None}
                if goal.mode == "eight_corner"
                else None
            )
            plan = plan_strips_sequence(
                state,
                goal,
                current_q=self.executor.current_arm_q(),
                corner_q_map=corner_q_map,
            )
            if not plan:
                break

            op = plan[0]
            c_idx = op.params.get("corner_index")
            perturb = pending_perturbation.pop(c_idx, None) if c_idx is not None else None

            step_t0 = time.monotonic()
            outcome = self.executor.execute_operator(op, cone_perturbation=perturb)
            state = self.executor.perceive_symbolic_state(block_id)
            atoms_ok = outcome.succeeded and self.monitor.verify(op, state)

            trace_entry = {
                "step": steps,
                "operator": op.name,
                "params": op.params,
                "skill_succeeded": outcome.succeeded,
                "expected_atoms_achieved": atoms_ok,
                "reason": outcome.reason,
                "inspected_count": int(sum(state.corner_inspected)),
                "anomaly_flagged": state.anomaly_flagged,
                "sorted_pile": state.sorted_pile,
                "duration_s": round(time.monotonic() - step_t0, 3),
                **outcome.metadata,
            }
            trace.append(trace_entry)
            steps += 1

            if not atoms_ok:
                replans += 1
                if op.name == "InspectExposedCorner" and c_idx is not None:
                    if self.monitor.retry_counts.get(c_idx, 0) >= 2:
                        yaw_op = GroundedOperator(
                            "ReorientBlockYaw",
                            block_id,
                            {"delta_yaw_rad": 0.5 * math.pi, "target_corner": int(c_idx)},
                        )
                        yaw_t0 = time.monotonic()
                        yaw_out = self.executor.execute_operator(yaw_op)
                        state = self.executor.perceive_symbolic_state(block_id)
                        trace.append(
                            {
                                "step": steps,
                                "operator": yaw_op.name,
                                "params": yaw_op.params,
                                "skill_succeeded": yaw_out.succeeded,
                                "expected_atoms_achieved": self.monitor.verify(yaw_op, state),
                                "reason": yaw_out.reason,
                                "inspected_count": int(sum(state.corner_inspected)),
                                "duration_s": round(time.monotonic() - yaw_t0, 3),
                                **yaw_out.metadata,
                            }
                        )
                        steps += 1
                        self.monitor.retry_counts[c_idx] = 0
                    else:
                        pending_perturbation[c_idx] = self.monitor.resample_cone_perturbation(c_idx)
                elif goal.mode in {"plate_spin_sort", "clear_source_plate"} and op.name in {
                    "PresentAndSpinAtPlateCamera",
                    "SortBlockToPile",
                }:
                    # Untrusted scan view or slip must halt safely rather than re-sorting blindly
                    break

        elapsed = time.monotonic() - t0
        succeeded = state.goal_satisfied(goal)
        summary = self.ledger.summary_dict()
        plate_keyframes = getattr(self.executor, "plate_scan_keyframes", [])
        sorted_counts = dict(getattr(self.executor, "sorted_counts", {}))
        raw_batch_records = list(getattr(self.executor, "batch_sorted_records", []))
        serializable_batch = [
            {k: v for k, v in rec.items() if k != "keyframe_rgb"} for rec in raw_batch_records
        ]

        json_path_str = None
        html_path_str = None
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            json_path = out / f"inspection_{block_id}_{stamp}.json"
            html_path = out / f"inspection_{block_id}_{stamp}.html"
            payload = {
                "schema": 3,
                "synthetic_simulation": bool(getattr(self.executor, "synthetic_simulation", True)),
                "physical_success_verified": False,
                "mode": goal.mode,
                "instruction": instruction,
                "succeeded": succeeded,
                "anomaly_flagged": state.anomaly_flagged,
                "sorted_pile": state.sorted_pile,
                "sorted_counts": sorted_counts,
                "batch_sorted_records": serializable_batch,
                "elapsed_s": round(elapsed, 3),
                "replans": replans,
                "trace": trace,
                "ledger": summary,
            }
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            html_path.write_text(
                _render_dossier_html(payload, self.ledger, plate_keyframes, raw_batch_records),
                encoding="utf-8",
            )
            json_path_str = str(json_path)
            html_path_str = str(html_path)

        return InspectionRunReport(
            instruction=instruction,
            block_id=block_id,
            succeeded=succeeded,
            inspected_corners=int(sum(state.corner_inspected)),
            replans=replans,
            executed_trace=trace,
            elapsed_s=elapsed,
            ledger_summary=summary,
            mode=goal.mode,
            anomaly_flagged=state.anomaly_flagged,
            sorted_pile=state.sorted_pile,
            sorted_counts=sorted_counts,
            batch_sorted_records=serializable_batch,
            report_json_path=json_path_str,
            report_html_path=html_path_str,
        )


def _render_dossier_html(
    payload: dict,
    ledger: EightCornerCoverageLedger,
    plate_keyframes: list[np.ndarray] | None = None,
    batch_records: list[dict] | None = None,
) -> str:
    cards = []
    if batch_records:
        for rec in batch_records:
            img_tag = "<div class='empty'>No ROI captured</div>"
            kf = rec.get("keyframe_rgb")
            if isinstance(kf, np.ndarray) and kf.size > 0:
                buf = io.BytesIO()
                Image.fromarray(kf).save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                img_tag = (
                    f"<img style='width:180px;height:180px;' src='data:image/png;base64,{b64}' "
                    f"alt='batch_{rec.get('order')}'/>"
                )
            pile = str(rec.get("sorted_pile", "good"))
            cls = "miss" if pile == "defective" else "pass"
            dist_val = rec.get("max_plate_anomaly_distance")
            dist_txt = f"{dist_val:.4f} (raw distance)" if dist_val is not None else "external/none"
            cards.append(
                f"<div class='card {cls}'>"
                f"<h3>#{rec.get('order')} {rec.get('inferred_part_name')} "
                f"<span class='badge'>{pile.upper()}</span></h3>"
                f"{img_tag}"
                f"<p><b>Slot:</b> {rec.get('slot_idx')} | <b>Plate err:</b> {rec.get('plate_xy_err_mm')} mm</p>"
                f"<p><b>Max OpenVINO dist:</b> {dist_txt}</p>"
                f"</div>"
            )
    elif plate_keyframes:
        for idx, frame in enumerate(plate_keyframes):
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            cards.append(
                f"<div class='card pass'>"
                f"<h3>Plate Cam CCW Frame {idx + 1}</h3>"
                f"<img style='width:220px;height:165px;' src='data:image/png;base64,{b64}' alt='plate_{idx}'/>"
                f"</div>"
            )
    else:
        for rec in ledger.records:
            img_tag = "<div class='empty'>No ROI captured</div>"
            if rec.best_roi_rgb is not None and rec.best_roi_rgb.size > 0:
                buf = io.BytesIO()
                Image.fromarray(rec.best_roi_rgb).save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                img_tag = (
                    f"<img src='data:image/png;base64,{b64}' alt='corner_{rec.corner_index}'/>"
                )
            badge = "PASS" if rec.inspected else "MISS"
            cls = "pass" if rec.inspected else "miss"
            dist_txt = (
                f"{rec.raw_anomaly_distance:.4f} (raw distance)"
                if rec.raw_anomaly_distance is not None
                else "N/A (detector optional)"
            )
            cards.append(
                f"<div class='card {cls}'>"
                f"<h3>Corner c_{rec.corner_index} <span class='badge'>{badge}</span></h3>"
                f"{img_tag}"
                f"<p><b>Camera:</b> {rec.best_camera or '-'} | <b>Dwell:</b> {rec.dwell_count}</p>"
                f"<p><b>Sharpness:</b> {rec.best_sharpness:.1f} | <b>Standoff:</b> {rec.best_standoff_m * 100:.1f} cm</p>"
                f"<p><b>Anomaly score:</b> {dist_txt}</p>"
                f"</div>"
            )
    pile_txt = (payload.get("sorted_pile") or "NONE").upper()
    counts = payload.get("sorted_counts") or {}
    if payload["succeeded"] and payload.get("mode") == "clear_source_plate":
        status = (
            f"SOURCE PLATE CLEARED — GOOD: {counts.get('good', 0)} | "
            f"DEFECTIVE: {counts.get('defective', 0)} (SIMULATION)"
        )
    elif payload["succeeded"] and payload.get("mode") == "plate_spin_sort":
        status = f"SORTED TO {pile_txt} PILE (SIMULATION)"
    elif payload["succeeded"]:
        status = "ALL 8 CORNERS VERIFIED (SIMULATION)"
    else:
        status = "INCOMPLETE"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>SO-101 Inspection & Sort Dossier — {ledger.object_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 16px; margin-top: 18px; }}
.card {{ background: #1e293b; border-radius: 10px; padding: 14px; border: 2px solid #334155; }}
.card.pass {{ border-color: #22c55e; }}
.card.miss {{ border-color: #ef4444; }}
.card img {{ width: 128px; height: 128px; border-radius: 6px; border: 1px solid #475569; }}
.badge {{ float: right; padding: 2px 8px; border-radius: 6px; font-size: 12px; background: #334155; }}
pre {{ background: #020617; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; }}
</style></head><body>
<h1>SO-101 Autonomous Pick, Plate-Camera CCW Spin & Sort Dossier</h1>
<p><b>Objective:</b> <code>{payload["instruction"]}</code> &nbsp;|&nbsp;
<b>Mode:</b> <code>{payload.get("mode")}</code> &nbsp;|&nbsp;
<b>Part:</b> <code>{ledger.block_spec.name}</code> &nbsp;|&nbsp;
<b>Anomaly Flagged:</b> <code>{payload.get("anomaly_flagged")}</code> &nbsp;|&nbsp;
<b>Status:</b> {status} &nbsp;|&nbsp; <b>Elapsed:</b> {payload["elapsed_s"]} s</p>
<p><i>Explicit provenance: Rendered MuJoCo 4.0.1 simulation + Frax 0.2.0 IK + verified CAD priors. Anomaly scores are raw distances, never probabilities. Physical station success is unverified until hardware run.</i></p>
<div class="grid">{"".join(cards)}</div>
<h2>CogMan Symbolic Execution Trace</h2>
<pre>{json.dumps(payload["trace"], indent=2)}</pre>
</body></html>
"""
