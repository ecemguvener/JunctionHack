"""Export an uncalibrated pilot and check CPU numerical parity, never defect accuracy."""

import argparse
import json
from pathlib import Path

import openvino as ov
import torch
from anomalib.engine import Engine
from anomalib.models import Patchcore

from brickcell.inspection.block_surface import surface_input, surface_score
from brickcell.inspection.images import read_rgb
from brickcell.util import save_json, sha256, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    metadata = json.loads((args.pilot / "pilot.json").read_text())
    checkpoint = args.pilot / "reference.ckpt"
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Pilot checkpoint hash mismatch")
    torch.set_num_threads(4)
    model = (
        Patchcore.load_from_checkpoint(str(checkpoint), map_location="cpu", pre_trained=False)
        .cpu()
        .eval()
    )
    engine = Engine(
        accelerator="cpu", devices=1, logger=False, default_root_dir=args.output / "engine"
    )
    xml = Path(
        engine.export(
            model=model,
            export_type="openvino",
            export_root=args.output,
            model_file_name="reference-only",
            input_size=tuple(metadata["image_size"]),
            compression_type=None,
            onnx_kwargs={"dynamic_axes": {}, "dynamic_shapes": None, "opset_version": 18},
        )
    )
    compiled = ov.Core().compile_model(
        str(xml), "CPU", {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY"}
    )
    ports = {port.get_any_name(): port for port in compiled.outputs}
    rows = []
    for path in args.images:
        tensor, mask = surface_input(
            read_rgb(path), tuple(metadata["image_size"]), metadata.get("surface_profile")
        )
        with torch.inference_mode():
            prediction = model(torch.from_numpy(tensor))
        expected, _ = surface_score(
            prediction.pred_score.item(), prediction.anomaly_map.squeeze().numpy(), mask
        )
        result = compiled({compiled.input(0): tensor})
        actual, _ = surface_score(
            float(result[ports["pred_score"]].reshape(-1)[0]),
            result[ports["anomaly_map"]].squeeze(),
            mask,
        )
        rows.append(
            dict(
                image=str(path),
                sha256=sha256(path),
                torch_distance=expected,
                openvino_distance=actual,
                absolute_difference=abs(expected - actual),
            )
        )
    report = dict(
        scope="UNCALIBRATED_REFERENCE_EXPORT_PARITY_NOT_ACCURACY",
        threshold=None,
        automatic_sorting_allowed=False,
        device="CPU",
        absolute_tolerance=0.001,
        checkpoint_sha256=metadata["checkpoint_sha256"],
        pilot_metadata_sha256=sha256(args.pilot / "pilot.json"),
        model_sha256=sha256(xml),
        weights_sha256=sha256(xml.with_suffix(".bin")),
        versions=versions(),
        rows=rows,
        passed=all(row["absolute_difference"] <= 0.001 for row in rows),
        max_absolute_difference=max(row["absolute_difference"] for row in rows),
    )
    save_json(args.output / "parity.json", report)
    print(json.dumps({key: report[key] for key in ("scope", "passed", "max_absolute_difference")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
