"""Adapters bridging `SO101InspectionTwin` to core `RobotInterface` and `FrameSource` boundaries.

Enforces `AGENTS.md` architectural separation:
- `TwinRobotAdapter` implements non-blocking `RobotInterface` (`mode = "mujoco_simulation_not_physical"`).
  Driver methods return `Action` tokens immediately (`ActionStatus.PENDING`); `action_status(action)`
  advances bounded physics slices per poll and never runs any ML detector.
- `TwinPlateFrameSource` implements `FrameSource`, saving depth-isolated Plate-camera (`side`)
  CCW spin views per `(object_id, attempt)` for consumption by `InspectionService(OpenVINODetector)`
  inside `brickcell.controller.supervisor.Supervisor`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from brickcell.block_inspector.geometry import PLATE_DROP_SLOT_OFFSETS_XY
from brickcell.block_inspector.twin import (
    DEFECTIVE_PILE_XY,
    GOOD_PILE_XY,
    JOINT5_CCW_END_RAD,
    JOINT5_CCW_START_RAD,
    OBSERVATION_HOME_Q,
    PLATE_CAMERA_SCAN_POS,
    SO101InspectionTwin,
)
from brickcell.controller.robot import Action, ActionStatus, RobotInterface
from brickcell.sources.directory import Frame, FrameSource


class TwinRobotAdapter(RobotInterface):
    """Non-blocking `RobotInterface` backed by `SO101InspectionTwin` pure-contact MuJoCo physics."""

    mode = "mujoco_simulation_not_physical"

    def __init__(self, twin: SO101InspectionTwin) -> None:
        self.twin = twin
        self.connected: bool = False
        self.stopped: bool = False
        self._seq: int = 0
        self._actions: dict[str, Action] = {}
        self._statuses: dict[str, ActionStatus] = {}
        self._generators: dict[str, Iterator[bool | None]] = {}
        self._pending_pile: dict[str, str] = {}
        self._pending_above_q: dict[str, np.ndarray] = {}
        self._pending_release_q: dict[str, np.ndarray] = {}
        self.captured_spin_crops: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] = {}
        self._attempt_counts: dict[str, int] = {}

    def connect(self) -> None:
        self.connected = True
        self.stopped = False

    def disconnect(self) -> None:
        self.connected = False

    def stop_safely(self) -> None:
        self.stopped = True
        self._generators.clear()
        self.twin.status_banner = "SAFE STOP: Supervisor halted cell"
        self.twin._sync_viewer()

    def _enqueue(self, operation: str, object_id: str, gen: Iterator[bool | None]) -> Action:
        if not self.connected or self.stopped:
            raise RuntimeError("Robot adapter is disconnected or safely stopped")
        self._seq += 1
        action = Action(
            action_id=f"sim-act-{self._seq}",
            object_id=object_id,
            operation=operation,
        )
        self._actions[action.action_id] = action
        self._statuses[action.action_id] = ActionStatus.PENDING
        self._generators[action.action_id] = gen
        return action

    def action_status(self, action: Action) -> ActionStatus:
        known = self._actions.get(action.action_id)
        if known != action:
            raise ValueError("Unknown or mismatched action token")
        if self.stopped:
            self._statuses[action.action_id] = ActionStatus.FAILED
            return ActionStatus.FAILED
        cur = self._statuses[action.action_id]
        if cur != ActionStatus.PENDING:
            return cur
        gen = self._generators.get(action.action_id)
        if gen is None:
            self._statuses[action.action_id] = ActionStatus.FAILED
            return ActionStatus.FAILED
        try:
            outcome = next(gen)
            if outcome is None:
                return ActionStatus.PENDING
            final = ActionStatus.SUCCEEDED if outcome else ActionStatus.FAILED
            self._statuses[action.action_id] = final
            self._generators.pop(action.action_id, None)
            return final
        except StopIteration:
            self._statuses[action.action_id] = ActionStatus.SUCCEEDED
            self._generators.pop(action.action_id, None)
            return ActionStatus.SUCCEEDED
        except Exception:
            self._statuses[action.action_id] = ActionStatus.FAILED
            self._generators.pop(action.action_id, None)
            return ActionStatus.FAILED

    def start_pick(self, object_id: str) -> Action:
        def _gen() -> Iterator[bool | None]:
            yield None
            if len(self.twin.spawned_blocks) > 1:
                loc_out = self.twin._skill_localize_next_cluster_block()
                if not loc_out.succeeded or self.twin.source_plate_cleared:
                    yield False
                    return
            else:
                loc_out = self.twin._skill_localize()
                if not loc_out.succeeded:
                    yield False
                    return
            yield None
            pick_out = self.twin._skill_pick_block()
            yield bool(pick_out.succeeded and not self.twin.hand_empty)

        return self._enqueue("pick", object_id, _gen())

    def _run_plate_ccw_capture(self, object_id: str, attempt: int) -> Iterator[bool | None]:
        if self.twin.hand_empty:
            yield False
            return
        yield None
        radial_az = math.atan2(float(PLATE_CAMERA_SCAN_POS[1]), float(PLATE_CAMERA_SCAN_POS[0]))
        pres_sol = self.twin.ik.solve_grasp_pose(
            PLATE_CAMERA_SCAN_POS,
            jaw_yaw_world=radial_az + 0.5 * math.pi,
            pitch_down_deg=55.0,
        )
        q_start = pres_sol.q_grasp.copy()
        q_start[4] = max(
            float(self.twin.ik.cal_kin.calibration.lower[4]) + 0.05,
            JOINT5_CCW_START_RAD,
        )
        q_end = pres_sol.q_grasp.copy()
        q_end[4] = min(
            float(self.twin.ik.cal_kin.calibration.upper[4]) - 0.05,
            JOINT5_CCW_END_RAD,
        )
        q_carry = 0.5 * (self.twin.current_arm_q() + q_start)
        q_carry[1] = min(float(self.twin.current_arm_q()[1]), float(q_start[1]), -0.45)
        q_carry[4] = 0.5 * float(self.twin.current_arm_q()[4] + q_start[4])
        self.twin.move_joint_trajectory(q_carry, duration_s=0.45, gripper_pct=0.0, settle_steps=2)
        yield None
        self.twin.move_joint_trajectory(q_start, duration_s=0.55, gripper_pct=0.0, settle_steps=6)

        self.twin.plate_scan_active = True
        crops: list[tuple[np.ndarray, np.ndarray]] = []
        n_spin_steps = 48
        plate_cam = self.twin.rig.realsense_2
        sample_indices = {6, 18, 30, 42}
        for k in range(n_spin_steps + 1):
            tau = k / n_spin_steps
            q_step = q_start.copy()
            q_step[4] = float(q_start[4] + tau * (q_end[4] - q_start[4]))
            self.twin.step_control_30hz(q_step, gripper_pct=0.0)
            if k in sample_indices:
                rgb_p, depth_p = self.twin.render_rgbd("realsense_2")
                T_held = self.twin.body3_world_pose() @ self.twin._T_body3_block_held
                uv_c, z_c = plate_cam.project_world_points(T_held[:3, 3][None, :])
                u_px, v_px, z_exp = float(uv_c[0, 0]), float(uv_c[0, 1]), float(z_c[0])
                if (
                    24.0 <= u_px <= (plate_cam.width - 24.0)
                    and 24.0 <= v_px <= (plate_cam.height - 24.0)
                    and depth_p is not None
                ):
                    u0 = max(0, int(round(u_px)) - 36)
                    u1 = min(plate_cam.width, int(round(u_px)) + 36)
                    v0 = max(0, int(round(v_px)) - 36)
                    v1 = min(plate_cam.height, int(round(v_px)) + 36)
                    roi_rgb = rgb_p[v0:v1, u0:u1].copy()
                    roi_fg = (depth_p[v0:v1, u0:u1] > max(0.08, z_exp - 0.055)) & (
                        depth_p[v0:v1, u0:u1] < (z_exp + 0.055)
                    )
                    if roi_rgb.size > 0 and bool(np.any(roi_fg)):
                        crops.append((roi_rgb, roi_fg))
            if k % 16 == 0:
                yield None

        self.twin.plate_scan_active = False
        still_held = float(self.twin.ground_truth_block_pose()[2, 3]) > 0.030
        if not still_held or len(crops) == 0:
            yield False
            return
        self.twin.plate_scan_completed = True
        self.captured_spin_crops[(object_id, attempt)] = crops
        yield True

    def move_to_inspection(self, object_id: str) -> Action:
        self._attempt_counts[object_id] = 0
        return self._enqueue("present", object_id, self._run_plate_ccw_capture(object_id, 0))

    def adjust_view(self, object_id: str) -> Action:
        nxt = self._attempt_counts.get(object_id, 0) + 1
        self._attempt_counts[object_id] = nxt
        return self._enqueue("adjust_view", object_id, self._run_plate_ccw_capture(object_id, nxt))

    def _move_to_pile_gen(self, object_id: str, pile: str) -> Iterator[bool | None]:
        if self.twin.hand_empty:
            yield False
            return
        yield None
        base_xy = DEFECTIVE_PILE_XY if pile == "defective" else GOOD_PILE_XY
        slot_idx = int(self.twin.sorted_counts[pile]) % len(PLATE_DROP_SLOT_OFFSETS_XY)
        if len(self.twin.spawned_blocks) > 1:
            dx, dy = PLATE_DROP_SLOT_OFFSETS_XY[slot_idx]
            target_xy = np.array([base_xy[0] + dx, base_xy[1] + dy], dtype=np.float64)
            z_release = 0.052
        else:
            target_xy = base_xy.copy()
            z_release = 0.038

        drop_yaw = math.atan2(float(target_xy[1]), float(target_xy[0])) + 0.5 * math.pi
        above_sol = self.twin.ik.solve_grasp_pose(
            np.array([target_xy[0], target_xy[1], 0.120]),
            jaw_yaw_world=drop_yaw,
            pitch_down_deg=90.0,
        )
        release_sol = self.twin.ik.solve_grasp_pose(
            np.array([target_xy[0], target_xy[1], z_release]),
            jaw_yaw_world=drop_yaw,
            pitch_down_deg=90.0,
            q_posture=above_sol.q_grasp,
        )
        q_unwind = self.twin.current_arm_q()
        q_unwind[4] = float(above_sol.q_grasp[4])
        self.twin.move_joint_trajectory(q_unwind, duration_s=0.40, gripper_pct=0.0, settle_steps=3)
        yield None
        if pile == "defective":
            mid_sol = self.twin.ik.solve_grasp_pose(
                np.array([0.16, 0.00, 0.165], dtype=np.float64),
                jaw_yaw_world=drop_yaw,
                pitch_down_deg=90.0,
                q_posture=above_sol.q_grasp,
            )
            self.twin.move_joint_trajectory(
                mid_sol.q_grasp, duration_s=0.60, gripper_pct=0.0, settle_steps=4
            )
            yield None
        self.twin.move_joint_trajectory(
            above_sol.q_grasp, duration_s=0.55, gripper_pct=0.0, settle_steps=4
        )
        yield None
        self.twin.move_joint_trajectory(
            release_sol.q_grasp, duration_s=0.45, gripper_pct=0.0, settle_steps=4
        )
        self._pending_pile[object_id] = pile
        self._pending_above_q[object_id] = above_sol.q_grasp.copy()
        self._pending_release_q[object_id] = release_sol.q_grasp.copy()
        yield True

    def move_to_good(self, object_id: str) -> Action:
        return self._enqueue("move_good", object_id, self._move_to_pile_gen(object_id, "good"))

    def move_to_defective(self, object_id: str) -> Action:
        return self._enqueue(
            "move_defective", object_id, self._move_to_pile_gen(object_id, "defective")
        )

    def release(self, object_id: str) -> Action:
        def _gen() -> Iterator[bool | None]:
            yield None
            q_rel = self._pending_release_q.get(object_id, self.twin.current_arm_q())
            q_abv = self._pending_above_q.get(object_id, OBSERVATION_HOME_Q)
            open_pct = 0.55 if len(self.twin.spawned_blocks) > 1 else 0.65
            self.twin._release_gripper_smooth(q_rel, open_pct=open_pct)
            self.twin.hand_empty = True
            yield None
            self.twin.move_joint_trajectory(
                q_abv, duration_s=0.45, gripper_pct=0.72, settle_steps=8
            )
            yield True

        return self._enqueue("release", object_id, _gen())

    def verify_release(self, object_id: str) -> Action:
        def _gen() -> Iterator[bool | None]:
            yield None
            pile = self._pending_pile.get(object_id)
            if pile not in {"good", "defective"} or not self.twin.hand_empty:
                yield False
                return
            base_xy = DEFECTIVE_PILE_XY if pile == "defective" else GOOD_PILE_XY
            p_final = self.twin.ground_truth_block_pose()[:3, 3]
            xy_err = float(np.linalg.norm(p_final[:2] - base_xy))
            ok = xy_err <= 0.088 and float(p_final[2]) < 0.075
            if ok:
                self.twin.sorted_pile = pile
                self.twin.sorted_counts[pile] += 1
                self.twin.remaining_source_blocks = max(0, self.twin.remaining_source_blocks - 1)
                self.twin._sync_viewer()
            yield ok

        return self._enqueue("verify_release", object_id, _gen())


def compose_isolated_spin_mosaic(
    crops: list[tuple[np.ndarray, np.ndarray]],
    canvas_hw: tuple[int, int] = (96, 96),
) -> np.ndarray:
    """Compose depth-segmented 4-angle CCW spin views inside an isolated 96x96 RGB canvas."""
    h, w = canvas_hw
    bg = np.array([28, 32, 38], dtype=np.uint8)
    canvas = np.broadcast_to(bg, (h, w, 3)).copy()
    # Leave a 10 px clean uniform border around a 2x2 grid of 38x38 depth-masked brick views
    tile = 38
    offsets = [(10, 10), (10, 48), (48, 10), (48, 48)]
    for idx in range(min(4, len(crops))):
        rgb, fg = crops[idx]
        masked = np.where(fg[:, :, None], rgb, bg)
        tile_rgb = np.asarray(
            Image.fromarray(masked).resize((tile, tile), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        r0, c0 = offsets[idx]
        canvas[r0 : r0 + tile, c0 : c0 + tile] = tile_rgb
    return canvas


class TwinPlateFrameSource(FrameSource):
    """`FrameSource` writing unique depth-isolated Plate-camera CCW views per `(object_id, attempt)`."""

    def __init__(self, robot_adapter: TwinRobotAdapter, output_dir: Path) -> None:
        self.robot_adapter = robot_adapter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequence: int = 0

    def capture(self, object_id: str, attempt: int) -> Frame | None:
        crops = self.robot_adapter.captured_spin_crops.get((object_id, attempt))
        if not crops:
            return None
        self.sequence += 1
        mosaic = compose_isolated_spin_mosaic(crops)
        frame_id = f"{object_id}-plate-spin-{self.sequence}"
        img_path = self.output_dir / f"{frame_id}.png"
        Image.fromarray(mosaic).save(img_path, format="PNG")
        return Frame(
            object_id=object_id,
            frame_id=frame_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            view_id=f"plate-ccw-attempt-{attempt}",
            path=img_path,
        )
