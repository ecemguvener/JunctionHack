"""OpenVINO 2026.4.1 CPU patch-memory-bank anomaly detector & explicit split calibration.

Honors AGENTS.md invariants:
- Trains prototype memory bank strictly on `train` (`good` images only) on CPU.
- Compiles and runs baseline inference strictly on OpenVINO CPU (`OpenVINODetector`).
- Calibrates frozen raw-distance `Threshold` on `val`; evaluates once on `test` after freezing.
- Enforces `brickcell.data.manifest.require_valid` (disjoint object/session groups, no duplicate SHA-256).
- Outputs raw nearest-prototype PatchCore distance (`raw_patchcore_distance_not_probability`), NEVER a probability.
- Zero robot imports: detector modules never issue robot commands.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from brickcell.data.manifest import Manifest, Sample, load_manifest, require_valid
from brickcell.detector.runtime import OpenVINODetector
from brickcell.inspection.images import input_tensor, read_rgb
from brickcell.inspection.threshold import calibrate, threshold_dict
from brickcell.util import save_json, sha256

DEFAULT_INPUT_HW = (96, 96)


def _extract_patch_features_np(tensors_nchw: np.ndarray) -> np.ndarray:
    """Extract 6D local color + 5x5 contrast patch descriptors inside central [16:80, 16:80] ROI."""
    # tensors_nchw: (N, 3, 96, 96) float32 in [0, 1]
    # 5x5 box blur via cumulative/sliding mean
    pad = np.pad(tensors_nchw, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="edge")
    acc = np.zeros_like(tensors_nchw)
    for dy in range(5):
        for dx in range(5):
            acc += pad[:, :, dy : dy + 96, dx : dx + 96]
    mean5 = acc / 25.0
    highpass = np.abs(tensors_nchw - mean5)
    roi_mean = mean5[:, :, 16:80, 16:80]
    roi_hp = highpass[:, :, 16:80, 16:80]
    feat = np.concatenate([roi_mean, 1.5 * roi_hp], axis=1)  # (N, 6, 64, 64)
    return feat.transpose(0, 2, 3, 1).reshape(-1, 6)


def export_patch_memory_openvino_ir(
    prototypes_k6: np.ndarray,
    xml_path: Path,
    input_hw: tuple[int, int] = DEFAULT_INPUT_HW,
) -> tuple[Path, Path]:
    """Export a genuine OpenVINO opset13 IR (.xml + .bin) nearest-prototype distance network."""
    import openvino as ov
    from openvino import opset13 as ops

    h, w = input_hw
    k_proto = int(prototypes_k6.shape[0])
    inp = ops.parameter([1, 3, h, w], np.float32, name="input")

    # 5x5 depthwise/grouped box filter over RGB channels
    kernel_5x5 = np.zeros((3, 3, 5, 5), dtype=np.float32)
    for c in range(3):
        kernel_5x5[c, c, :, :] = 1.0 / 25.0
    w_blur = ops.constant(kernel_5x5)
    mean5 = ops.convolution(
        inp,
        w_blur,
        strides=[1, 1],
        pads_begin=[2, 2],
        pads_end=[2, 2],
        dilations=[1, 1],
    )
    diff = ops.subtract(inp, mean5)
    abs_diff = ops.abs(diff)
    scaled_hp = ops.multiply(abs_diff, ops.constant(np.array(1.5, dtype=np.float32)))
    feat6 = ops.concat([mean5, scaled_hp], axis=1)  # [1, 6, 96, 96]

    # Central [16:80, 16:80] crop isolating the held brick from image borders
    roi6 = ops.slice(
        feat6,
        start=ops.constant(np.array([16, 16], dtype=np.int64)),
        stop=ops.constant(np.array([h - 16, w - 16], dtype=np.int64)),
        step=ops.constant(np.array([1, 1], dtype=np.int64)),
        axes=ops.constant(np.array([2, 3], dtype=np.int64)),
    )  # [1, 6, 64, 64]

    # Compute exact squared L2 distance ||f - c_k||^2 = ||f||^2 - 2 c_k.f + ||c_k||^2 via 1x1 conv
    protos = np.asarray(prototypes_k6, dtype=np.float32)  # (K, 6)
    f_sq = ops.reduce_sum(
        ops.multiply(roi6, roi6),
        reduction_axes=ops.constant(np.array([1], dtype=np.int64)),
        keep_dims=True,
    )  # [1, 1, 64, 64]
    w_dot = ops.constant((-2.0 * protos).reshape(k_proto, 6, 1, 1))
    neg2_dot = ops.convolution(
        roi6,
        w_dot,
        strides=[1, 1],
        pads_begin=[0, 0],
        pads_end=[0, 0],
        dilations=[1, 1],
    )  # [1, K, 64, 64]
    c_sq = ops.constant(np.sum(protos * protos, axis=1).reshape(1, k_proto, 1, 1))
    dist_sq_all = ops.maximum(
        ops.add(ops.add(f_sq, neg2_dot), c_sq),
        ops.constant(np.array(1e-12, dtype=np.float32)),
    )
    min_dist_sq = ops.reduce_min(
        dist_sq_all,
        reduction_axes=ops.constant(np.array([1], dtype=np.int64)),
        keep_dims=False,
    )  # [1, 64, 64]
    anomaly_map = ops.sqrt(min_dist_sq)
    anomaly_map.set_friendly_name("anomaly_map_op")

    # Smooth 3x3 local pooling before max so single-pixel specular glint noise is suppressed
    amap_4d = ops.unsqueeze(anomaly_map, axes=ops.constant(np.array([1], dtype=np.int64)))
    w_pool3 = ops.constant(np.full((1, 1, 3, 3), 1.0 / 9.0, dtype=np.float32))
    pooled_map = ops.convolution(
        amap_4d,
        w_pool3,
        strides=[1, 1],
        pads_begin=[1, 1],
        pads_end=[1, 1],
        dilations=[1, 1],
    )
    pred_score = ops.reduce_max(
        pooled_map,
        reduction_axes=ops.constant(np.array([1, 2, 3], dtype=np.int64)),
        keep_dims=False,
    )  # [1]

    res_score = ops.result(pred_score)
    res_score.get_output_tensor(0).set_names({"pred_score"})
    res_map = ops.result(pooled_map)
    res_map.get_output_tensor(0).set_names({"anomaly_map"})

    model = ov.Model([res_score, res_map], [inp], "plate_patch_memory_anomaly")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(xml_path), compress_to_fp16=False)
    return xml_path, xml_path.with_suffix(".bin")


def generate_synthetic_plate_anomaly_dataset(
    dataset_dir: Path,
    seed: int = 42,
) -> Path:
    """Render an explicit train/val/test MuJoCo 3D Plate-camera crop dataset with disjoint groups."""
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from brickcell.block_inspector.geometry import (
        DUPLO_CATALOG,
        NOMINAL_TCP_OFFSET_BODY3,
        station_camera_rig,
    )
    from brickcell.block_inspector.twin import (
        JAW_CLOSED_RAD,
        PLATE_CAMERA_SCAN_POS,
        SpawnedBlockInfo,
        _rotmat_to_mj_quat,
        build_inspection_cell_mjcf,
    )

    rng = np.random.default_rng(seed)
    dataset_dir = Path(dataset_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    img_root = dataset_dir / "images"
    img_root.mkdir(parents=True, exist_ok=True)

    rig = station_camera_rig()
    cat_specs = [DUPLO_CATALOG[k] for k in sorted(DUPLO_CATALOG.keys())]
    spawn_list = [
        SpawnedBlockInfo(f"block_{i}", spec, (0.23, 0.0, 0.0), has_visual_defect=False)
        for i, spec in enumerate(cat_specs)
    ] + [
        SpawnedBlockInfo(f"block_{i + 4}", spec, (0.23, 0.0, 0.0), has_visual_defect=True)
        for i, spec in enumerate(cat_specs)
    ]
    xml_path = build_inspection_cell_mjcf(
        dataset_dir / "_render_cache",
        cat_specs[0],
        rig=rig,
        spawned_blocks=spawn_list,
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    from brickcell.block_inspector.frax_ik import SO101_ARM_JOINTS

    arm_qpos = [int(model.jnt_qposadr[model.joint(jn).id]) for jn in SO101_ARM_JOINTS]
    jaw_qpos = int(model.jnt_qposadr[model.joint("Revolute_5").id])
    blk_qpos = [int(model.jnt_qposadr[model.joint(f"block_{i}_joint").id]) for i in range(8)]
    body3_id = model.body("Body3").id

    # Fast exact 4-DOF MuJoCo FK solve placing TCP at PLATE_CAMERA_SCAN_POS pitched ~55 deg down
    def _tcp_res(q4: np.ndarray) -> np.ndarray:
        for j in range(4):
            data.qpos[arm_qpos[j]] = float(q4[j])
        data.qpos[arm_qpos[4]] = 0.0
        mujoco.mj_forward(model, data)
        R_b3 = data.xmat[body3_id].reshape(3, 3)
        p_tcp = data.xpos[body3_id] + R_b3 @ NOMINAL_TCP_OFFSET_BODY3
        # -Z_Body3 pitched ~55 deg downward (-sin(55 deg) = -0.819)
        pitch_err = float((-R_b3[2, 2]) - (-0.819))
        return np.append((p_tcp - PLATE_CAMERA_SCAN_POS) * 20.0, pitch_err)

    sol = least_squares(
        _tcp_res,
        x0=np.array([-0.65, -0.45, 0.95, 0.45], dtype=np.float64),
        bounds=(
            np.array([-1.9, -1.7, -1.6, -1.6]),
            np.array([1.9, 1.7, 1.6, 1.6]),
        ),
    )
    q_base4 = sol.x.copy()

    plate_cam = rig.realsense_2
    rgb_r = mujoco.Renderer(model, height=plate_cam.height, width=plate_cam.width)
    dep_r = mujoco.Renderer(model, height=plate_cam.height, width=plate_cam.width)
    dep_r.enable_depth_rendering()
    bg_iso = np.array([28, 32, 38], dtype=np.uint8)

    samples: list[dict] = []
    # Disjoint object_id and session_id per split; train contains ONLY good images (AGENTS.md)
    split_specs = [
        ("train", "good", 32),
        ("val", "good", 8),
        ("val", "defective", 8),
        ("test", "good", 8),
        ("test", "defective", 8),
    ]

    try:
        for split, label, count in split_specs:
            for k in range(count):
                obj_id = f"{split}_{label}_obj_{k:03d}"
                sess_id = f"{split}_session_{k // 2:02d}_{label}"
                cat_idx = k % 4
                active_blk = cat_idx if label == "good" else (cat_idx + 4)

                # Park inactive bricks well outside camera frustum
                for bi, adr in enumerate(blk_qpos):
                    if bi != active_blk:
                        data.qpos[adr : adr + 3] = [-0.60, 0.10 * bi, -0.30]
                        data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

                q4_spin = float(np.linspace(-2.75, 2.75, count, endpoint=True)[k]) + float(
                    rng.uniform(-0.10, 0.10)
                )
                for j in range(4):
                    data.qpos[arm_qpos[j]] = float(q_base4[j] + rng.uniform(-0.015, 0.015))
                data.qpos[arm_qpos[4]] = float(np.clip(q4_spin, -2.90, 2.90))
                data.qpos[jaw_qpos] = JAW_CLOSED_RAD
                mujoco.mj_forward(model, data)

                R_b3 = data.xmat[body3_id].reshape(3, 3).copy()
                # Top-down pick aligns brick +Z_B with +Z_Body3 (`Revolute_4` spin axis)
                R_local = Rotation.from_euler("z", 0.25 * np.pi * (k % 4)).as_matrix()
                R_blk = R_b3 @ R_local
                p_blk = data.xpos[body3_id] + R_b3 @ NOMINAL_TCP_OFFSET_BODY3
                adr_a = blk_qpos[active_blk]
                data.qpos[adr_a : adr_a + 3] = p_blk
                data.qpos[adr_a + 3 : adr_a + 7] = _rotmat_to_mj_quat(R_blk)
                mujoco.mj_forward(model, data)

                rgb_r.update_scene(data, camera=plate_cam.name)
                rgb_full = rgb_r.render().copy()
                dep_r.update_scene(data, camera=plate_cam.name)
                dep_full = dep_r.render().copy()

                uv_c, z_c = plate_cam.project_world_points(p_blk[None, :])
                u_c = int(np.clip(round(float(uv_c[0, 0])), 48, plate_cam.width - 48))
                v_c = int(np.clip(round(float(uv_c[0, 1])), 48, plate_cam.height - 48))
                ze = float(z_c[0])
                crop = rgb_full[v_c - 48 : v_c + 48, u_c - 48 : u_c + 48].copy()
                dep_crop = dep_full[v_c - 48 : v_c + 48, u_c - 48 : u_c + 48]

                # Depth-isolate foreground held brick + jaws so background bricks on plates never pollute patches
                fg = (dep_crop > max(0.08, ze - 0.065)) & (dep_crop < (ze + 0.065))
                crop = np.where(fg[:, :, None], crop, bg_iso)

                noise = rng.normal(0.0, 1.2, size=crop.shape)
                rgb_u8 = np.clip(crop.astype(np.float64) + noise, 0, 255).astype(np.uint8)
                rel_path = f"images/{split}_{label}_{k:03d}.png"
                Image.fromarray(rgb_u8).save(dataset_dir / rel_path)
                samples.append(
                    asdict(
                        Sample(
                            image_id=f"{split}-{label}-{k:03d}",
                            path=rel_path,
                            label=label,
                            split=split,
                            object_id=obj_id,
                            session_id=sess_id,
                            view_id=f"plate_ccw_{k % 4}",
                            defect_type=(
                                "synthetic_surface_scorch_blemish" if label == "defective" else None
                            ),
                            defect_regions=[[28, 28, 68, 68]] if label == "defective" else None,
                            synthetic=True,
                        )
                    )
                )
    finally:
        rgb_r.close()
        dep_r.close()

    manifest_path = dataset_dir / "manifest.json"
    save_json(manifest_path, {"schema_version": 1, "samples": samples})
    require_valid(load_manifest(manifest_path))
    return manifest_path


def train_calibrate_and_evaluate_plate_detector(
    manifest_path: Path,
    output_dir: Path,
    num_prototypes: int = 32,
    seed: int = 42,
) -> tuple[OpenVINODetector, dict]:
    """Fit memory bank on `train/good`, calibrate frozen threshold on `val`, evaluate once on `test`."""
    import openvino as ov

    manifest: Manifest = load_manifest(manifest_path)
    validation_report = require_valid(manifest)

    train_samples = manifest.partition("train")
    val_samples = manifest.partition("val")
    test_samples = manifest.partition("test")
    if any(s.label != "good" for s in train_samples):
        raise ValueError("Training partition must contain only good samples")

    # 1. Fit KMeans patch memory bank strictly on train/good crops (CPU)
    train_tensors = np.concatenate(
        [input_tensor(read_rgb(manifest.image_path(s)), DEFAULT_INPUT_HW) for s in train_samples],
        axis=0,
    )
    feats = _extract_patch_features_np(train_tensors)
    if len(feats) > 12000:
        rng = np.random.default_rng(seed)
        feats = feats[rng.choice(len(feats), size=12000, replace=False)]
    km = KMeans(n_clusters=num_prototypes, n_init=6, random_state=seed)
    km.fit(feats)
    prototypes = km.cluster_centers_.astype(np.float32)

    # 2. Export OpenVINO IR (.xml + .bin) and compile on CPU to score `val`
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path, bin_path = export_patch_memory_openvino_ir(
        prototypes, output_dir / "plate_anomaly_detector.xml", input_hw=DEFAULT_INPUT_HW
    )
    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU")
    req = compiled.create_infer_request()
    score_port = {p.get_any_name(): p for p in compiled.outputs}["pred_score"]

    val_scores: list[float] = []
    val_labels: list[int] = []
    for s in val_samples:
        t = input_tensor(read_rgb(manifest.image_path(s)), DEFAULT_INPUT_HW)
        out = req.infer({compiled.input(0): t})
        val_scores.append(float(out[score_port].reshape(-1)[0]))
        val_labels.append(1 if s.label == "defective" else 0)

    frozen_threshold = calibrate(
        scores=val_scores,
        labels=val_labels,
        manifest_hash=validation_report["manifest_sha256"],
        uncertainty_fraction=0.0,
    )

    model_sha = sha256(xml_path)
    weights_sha = sha256(bin_path)
    model_version = f"plate-patch-ov-{model_sha[:12]}"
    meta = {
        "schema_version": 1,
        "synthetic_training_data": True,
        "model_version": model_version,
        "openvino_version": ov.__version__,
        "input_size": list(DEFAULT_INPUT_HW),
        "score_type": "raw_patchcore_distance_not_probability",
        "model_sha256": model_sha,
        "weights_sha256": weights_sha,
        "dataset_manifest_sha256": validation_report["manifest_sha256"],
        "threshold": threshold_dict(frozen_threshold),
    }
    save_json(xml_path.with_suffix(".json"), meta)

    # 3. Load frozen OpenVINODetector and evaluate ONCE on held-out `test` partition
    detector = OpenVINODetector(xml_path)
    assert detector.threshold is not None
    tp = tn = fp = fn = 0
    test_scores: list[float] = []
    for s in test_samples:
        pred = detector.predict(read_rgb(manifest.image_path(s)))
        test_scores.append(pred.score)
        pred_bad = pred.score >= detector.threshold.value
        actual_bad = s.label == "defective"
        if pred_bad and actual_bad:
            tp += 1
        elif (not pred_bad) and (not actual_bad):
            tn += 1
        elif pred_bad and (not actual_bad):
            fp += 1
        else:
            fn += 1

    eval_summary = {
        "synthetic_fixture": True,
        "score_type": "raw_patchcore_distance_not_probability",
        "model_version": detector.model_version,
        "openvino_version": ov.__version__,
        "threshold_value": detector.threshold.value,
        "threshold_version": detector.threshold.version,
        "partition_counts": validation_report["counts"],
        "test_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "test_min_raw_distance": float(min(test_scores)),
        "test_max_raw_distance": float(max(test_scores)),
    }
    save_json(output_dir / "evaluation_test_once.json", eval_summary)
    return detector, eval_summary


def ensure_default_plate_openvino_detector(
    model_dir: Path | None = None,
) -> OpenVINODetector:
    """Load or build the default simulation OpenVINO CPU Plate-camera anomaly detector."""
    target_dir = (
        Path(model_dir)
        if model_dir is not None
        else Path(__file__).resolve().parents[3]
        / "artifacts"
        / "block_inspector"
        / "openvino_plate_detector"
    )
    xml_path = target_dir / "model" / "plate_anomaly_detector.xml"
    json_path = xml_path.with_suffix(".json")
    if xml_path.is_file() and json_path.is_file():
        return OpenVINODetector(xml_path)
    manifest_path = generate_synthetic_plate_anomaly_dataset(target_dir / "dataset")
    detector, _ = train_calibrate_and_evaluate_plate_detector(manifest_path, target_dir / "model")
    return detector
