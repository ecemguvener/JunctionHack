"""Command-line interface for the SO-101 Pick -> Plate-Camera CCW Spin -> Sort stack."""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from pathlib import Path

from brickcell.block_inspector.camera_adapters import CamUiReporter, LiveInspectionHudServer
from brickcell.block_inspector.cogman import CogManInspectionAgent
from brickcell.block_inspector.frax_ik import export_calibrated_frax_assets
from brickcell.block_inspector.geometry import DUPLO_CATALOG, REPO_ROOT
from brickcell.block_inspector.openvino_anomaly import ensure_default_plate_openvino_detector
from brickcell.block_inspector.policy_skills import collect_train_and_eval_visible_skill
from brickcell.block_inspector.real2sim import (
    Real2SimPipeline,
    load_point_cloud_file,
    synthesize_cluttered_scene_point_cloud,
)
from brickcell.block_inspector.twin import SO101InspectionTwin


def _cmd_inspect(args: argparse.Namespace) -> int:
    show_viewer = not args.headless
    resolved_mode = args.mode or (
        "clear_source_plate" if args.num_blocks > 1 else "plate_spin_sort"
    )

    def _sim_anom_fn(_tw: SO101InspectionTwin, step_k: int, q4_rad: float, _rgb) -> str | None:
        if args.simulate_anomaly and q4_rad >= 0.0 and not _tw.anomaly_flagged:
            return f"simulated_anomaly_message_at_step_{step_k}_q5_{q4_rad:.2f}rad"
        return None

    scorer = None
    thr_val = None
    if not args.no_openvino_detector and (args.num_blocks > 1 or args.use_openvino_detector):
        detector = ensure_default_plate_openvino_detector()
        assert detector.threshold is not None
        scorer = lambda roi: detector.predict(roi).score  # noqa: E731
        thr_val = float(detector.threshold.value)

    twin = SO101InspectionTwin(
        block_spec=args.part,
        initial_block_xy_yaw=(args.block_x, args.block_y, args.block_yaw),
        num_blocks=args.num_blocks,
        cluster_seed=args.cluster_seed,
        show_viewer=show_viewer,
        show_camera_windows=not getattr(args, "no_camera_windows", False),
        realtime_pacing=not args.fast,
        anomaly_scorer=scorer,
        anomaly_threshold=thr_val,
        anomaly_message_fn=_sim_anom_fn if args.simulate_anomaly else None,
    )
    hud: LiveInspectionHudServer | None = None
    step_box = [0]

    try:
        if args.hud:
            hud = LiveInspectionHudServer(port=args.hud_port)
            url = hud.start()
            print(f"[HUD] Live 3-Camera Dashboard: {url}", flush=True)
            if args.open_browser:
                webbrowser.open(url)

            def _hud_listener(tw: SO101InspectionTwin) -> None:
                if step_box[0] % 8 == 0:
                    hud.update_from_twin(tw, step_counter=step_box[0])
                step_box[0] += 1

            twin.frame_listeners.append(_hud_listener)
            hud.update_from_twin(twin, step_counter=0)

        agent = CogManInspectionAgent(executor=twin, ledger=twin.ledger)
        report = agent.run(
            instruction=args.instruction,
            block_id="block_0",
            output_dir=args.output_dir,
            mode=resolved_mode,
        )
        if hud is not None:
            hud.update_from_twin(twin, step_counter=step_box[0])

        if args.report_camui:
            CamUiReporter(args.camui_url).report(
                malformed=None if not report.succeeded else bool(report.anomaly_flagged),
                detail=(
                    f"sim {report.mode}: sorted={report.sorted_pile}, counts={report.sorted_counts}, "
                    f"anomaly={report.anomaly_flagged} ({report.elapsed_s:.1f}s)"
                ),
            )

        print(
            json.dumps(
                {
                    "instruction": report.instruction,
                    "mode": report.mode,
                    "num_blocks": args.num_blocks,
                    "part": args.part,
                    "succeeded": report.succeeded,
                    "sorted_counts": report.sorted_counts,
                    "batch_sorted_records": report.batch_sorted_records,
                    "anomaly_flagged": report.anomaly_flagged,
                    "sorted_pile": report.sorted_pile,
                    "plate_scan_valid_frames": twin.plate_scan_valid_frames,
                    "inspected_corners": report.inspected_corners,
                    "replans": report.replans,
                    "elapsed_s": round(report.elapsed_s, 2),
                    "dossier_html": report.report_html_path,
                    "dossier_json": report.report_json_path,
                },
                indent=2,
            ),
            flush=True,
        )

        if show_viewer and twin.is_viewer_running():
            hold_s = float(args.hold_viewer_seconds)
            deadline = time.monotonic() + hold_s
            while twin.is_viewer_running() and time.monotonic() < deadline:
                twin._sync_viewer()
                time.sleep(0.05)

        return 0 if report.succeeded else 1
    finally:
        if hud is not None:
            hud.stop()
        twin.close()


def _cmd_train(args: argparse.Namespace) -> int:
    summary = collect_train_and_eval_visible_skill(
        part_name=args.part,
        output_dir=args.output_dir,
        num_symmetry_episodes=args.episodes,
        chunk_horizon=args.chunk_horizon,
        show_viewer=not args.headless,
        realtime_pacing=not args.fast,
        hold_viewer_seconds=args.hold_viewer_seconds,
    )
    print(json.dumps(summary.__dict__, indent=2), flush=True)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    urdf_path, spheres_path = export_calibrated_frax_assets(out_dir)
    wendy_path = REPO_ROOT / "wendy.json"
    qualia_path = REPO_ROOT / "configs" / "qualia" / "inspection_task.json"

    wendy_ok = wendy_path.exists() and "shared-ipc" in wendy_path.read_text(encoding="utf-8")
    qualia_ok = (
        qualia_path.exists()
        and json.loads(qualia_path.read_text(encoding="utf-8")).get("displayName")
        == "Investigate Block"
    )
    result = {
        "frax_urdf": str(urdf_path),
        "frax_spheres": str(spheres_path),
        "wendy_manifest_valid": wendy_ok,
        "qualia_manifest_valid": qualia_ok,
        "duplo_assets_verified": all(
            spec.stl_path is not None and spec.stl_path.exists() for spec in DUPLO_CATALOG.values()
        ),
    }
    print(json.dumps(result, indent=2))
    return 0 if all(result.values()) else 1


def _cmd_real2sim(args: argparse.Namespace) -> int:
    import numpy as np

    pipeline = Real2SimPipeline()
    gt_blocks = None

    if args.capture_dir:
        scene_state = pipeline.perceive_capture_directory(args.capture_dir)
    elif args.cloud_file:
        pts_w, cols_w = load_point_cloud_file(args.cloud_file, rig=pipeline.rig)
        scene_state = pipeline.perceive_point_cloud(
            pts_w, cols_w, provenance=f"file:{Path(args.cloud_file).name}"
        )
    elif args.top_json or args.plate_json:
        top_pkt = (
            json.loads(Path(args.top_json).read_text(encoding="utf-8")) if args.top_json else None
        )
        plate_pkt = (
            json.loads(Path(args.plate_json).read_text(encoding="utf-8"))
            if args.plate_json
            else None
        )
        prov_tf = (
            json.loads(Path(args.provisional_transform).read_text(encoding="utf-8"))
            if args.provisional_transform
            else None
        )
        scene_state = pipeline.perceive_teammate_packets(
            top_packet=top_pkt,
            plate_packet=plate_pkt,
            provisional_transform=prov_tf,
        )
    elif args.live_depth_url:
        scene_state = pipeline.perceive_from_live_depth_http(base_url=args.live_depth_url)
    else:
        pts_w, cols_w, gt_blocks, meta = synthesize_cluttered_scene_point_cloud(
            num_blocks=args.num_blocks,
            seed=args.seed,
            include_human_arm=True,
            include_wires=True,
            include_trash=True,
            include_robot_arm_points=True,
        )
        scene_state = pipeline.perceive_point_cloud(
            pts_w,
            cols_w,
            q_arm_rad=np.asarray(meta["q_arm_rad"], dtype=np.float64),
            provenance="synthetic_cluttered_dual_d415_point_cloud",
        )

    if args.perceive_only:
        print(json.dumps(scene_state.to_summary_dict(), indent=2), flush=True)
        return 0 if len(scene_state.bricks) > 0 else 1

    report = pipeline.close_the_loop(
        scene_state=scene_state,
        controller=args.controller,
        show_viewer=not args.headless,
        realtime_pacing=not args.fast,
        output_dir=args.output_dir,
        ground_truth_blocks=gt_blocks,
        hold_viewer_seconds=args.hold_viewer_seconds,
    )
    print(json.dumps(report.__dict__, indent=2), flush=True)
    return 0 if report.execution_succeeded else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brickcell.block_inspector",
        description="SO-101 Pick -> Low Plate-Camera Joint-5 CCW Spin -> Sort (MuJoCo 4.0.1 + Frax 0.2.0)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ins = sub.add_parser(
        "inspect",
        help="Pick clustered DUPLO blocks from aquamarine plate, spin Joint 5 CCW at Plate camera, and sort into good/defective plates",
    )
    p_ins.add_argument("--part", choices=sorted(DUPLO_CATALOG), default="duplo_2x2_3437")
    p_ins.add_argument("--num-blocks", type=int, default=8)
    p_ins.add_argument("--cluster-seed", type=int, default=42)
    p_ins.add_argument(
        "--instruction",
        default="pick and classify all 8 blocks from source plate",
    )
    p_ins.add_argument(
        "--mode",
        choices=["clear_source_plate", "plate_spin_sort", "eight_corner"],
        default=None,
    )
    p_ins.add_argument(
        "--use-openvino-detector",
        action="store_true",
        help="Force OpenVINO CPU Plate-camera anomaly detector even in single-block mode",
    )
    p_ins.add_argument(
        "--no-openvino-detector",
        action="store_true",
        help="Disable OpenVINO CPU Plate-camera anomaly detector",
    )
    p_ins.add_argument(
        "--simulate-anomaly",
        action="store_true",
        help="Simulate receiving an anomaly message during the Joint-5 CCW spin (routes to defective pile)",
    )
    p_ins.add_argument("--block-x", type=float, default=0.205)
    p_ins.add_argument("--block-y", type=float, default=0.00)
    p_ins.add_argument("--block-yaw", type=float, default=0.22)
    p_ins.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "block_inspector" / "reports"),
    )
    p_ins.add_argument("--headless", action="store_true", help="Disable live desktop 3D viewer")
    p_ins.add_argument(
        "--no-camera-windows",
        action="store_true",
        help="Disable the 3 separate desktop Virtual Camera feed windows",
    )
    p_ins.add_argument("--fast", action="store_true", help="Disable 30 Hz wall-clock sleep")
    p_ins.add_argument("--hud", action="store_true", help="Start live HTTP 3-camera HUD server")
    p_ins.add_argument(
        "--open-browser", action="store_true", help="Open live 3-camera HUD in browser"
    )
    p_ins.add_argument("--hud-port", type=int, default=8095)
    p_ins.add_argument("--hold-viewer-seconds", type=float, default=3.0)
    p_ins.add_argument("--report-camui", action="store_true", help="POST result to camui (:8093)")
    p_ins.add_argument("--camui-url", default="http://127.0.0.1:8093")
    p_ins.set_defaults(func=_cmd_inspect)

    p_tr = sub.add_parser(
        "collect-and-train",
        help="Collect Pick -> Plate-Cam CCW Spin -> Sort demos, train BC policy, and evaluate live",
    )
    p_tr.add_argument("--part", choices=sorted(DUPLO_CATALOG), default="duplo_2x2_3437")
    p_tr.add_argument("--episodes", type=int, default=4)
    p_tr.add_argument("--chunk-horizon", type=int, default=8)
    p_tr.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "block_inspector" / "skill_lab"),
    )
    p_tr.add_argument("--headless", action="store_true", help="Disable live desktop 3D viewer")
    p_tr.add_argument("--fast", action="store_true", help="Disable 30 Hz wall-clock sleep")
    p_tr.add_argument("--hold-viewer-seconds", type=float, default=3.0)
    p_tr.set_defaults(func=_cmd_train)

    p_val = sub.add_parser("validate", help="Export Frax URDF/spheres and validate manifests")
    p_val.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "block_inspector" / "frax"),
    )
    p_val.set_defaults(func=_cmd_validate)

    p_r2s = sub.add_parser(
        "real2sim",
        help="Ingest 3D point clouds, filter human arms/wires/trash, replicate DUPLO & plates in MuJoCo, and close the loop",
    )
    p_r2s.add_argument(
        "--capture-dir",
        help="Path to teammate depth capture directory (realsense_depth_capture.py / board_alignment_pilot.py)",
    )
    p_r2s.add_argument("--cloud-file", help="Path to .ply, .npz, or .json point cloud file")
    p_r2s.add_argument("--top-json", help="Path to teammate top-points.json packet")
    p_r2s.add_argument("--plate-json", help="Path to teammate plate-points.json packet")
    p_r2s.add_argument(
        "--provisional-transform", help="Optional path to provisional-transform.json"
    )
    p_r2s.add_argument(
        "--live-depth-url", help="Fetch live point clouds from live_depth.py (:8094)"
    )
    p_r2s.add_argument(
        "--num-blocks",
        type=int,
        default=4,
        help="Number of DUPLO bricks when synthesizing a cluttered test cloud",
    )
    p_r2s.add_argument("--seed", type=int, default=42)
    p_r2s.add_argument(
        "--controller",
        choices=["cogman_planner", "chunked_mlp_policy"],
        default="cogman_planner",
    )
    p_r2s.add_argument(
        "--perceive-only",
        action="store_true",
        help="Run Real2Sim perception and distractor filtering without launching MuJoCo rollout",
    )
    p_r2s.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "block_inspector" / "real2sim"),
    )
    p_r2s.add_argument("--headless", action="store_true", help="Disable live desktop 3D viewer")
    p_r2s.add_argument("--fast", action="store_true", help="Disable 30 Hz wall-clock sleep")
    p_r2s.add_argument("--hold-viewer-seconds", type=float, default=2.0)
    p_r2s.set_defaults(func=_cmd_real2sim)

    p_studio = sub.add_parser(
        "studio",
        help="Launch interactive MuJoCo Studio 3D scene + 3 live Virtual Camera feed windows",
    )
    p_studio.add_argument("--num-blocks", type=int, default=8)
    p_studio.add_argument("--cluster-seed", type=int, default=42)
    p_studio.add_argument("--hold-seconds", type=float, default=180.0)
    p_studio.add_argument("--single-pass", action="store_true", help="Exit after one pass")
    p_studio.set_defaults(
        func=lambda a: __import__(
            "brickcell.block_inspector.studio_vision", fromlist=["run_live_studio_session"]
        ).run_live_studio_session(
            num_blocks=a.num_blocks,
            cluster_seed=a.cluster_seed,
            hold_seconds=a.hold_seconds,
            loop_episodes=not a.single_pass,
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
