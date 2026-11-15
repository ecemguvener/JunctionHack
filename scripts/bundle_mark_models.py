"""Package existing calibrated plate/wrist exports; never train or command hardware."""

import argparse
import json
import shutil
from pathlib import Path

from brickcell.detector.config import ModelConfig
from brickcell.inspection.threshold import Threshold
from brickcell.util import save_json, sha256

SERIALS = {"plate": "243622060450", "wrist": "sonix-wrist"}
FILES = (
    "model.xml",
    "model.bin",
    "model.json",
    "metadata.json",
    "package_versions.json",
    "dataset_validation.json",
    "parity.json",
    "test_report.json",
)


def bundle(plate: Path, wrist: Path, output: Path, template: Path, check_runtime=False):
    output = output.resolve()
    if output.exists():
        raise ValueError("Choose a new bundle output directory")
    config = json.loads(template.read_text())
    sources = {"plate": plate.resolve(), "wrist": wrist.resolve()}
    digests = {}
    # Validate both before writing a bundle. No synthetic or uncalibrated fallback.
    for camera, folder in sources.items():
        for name in FILES:
            if not (folder / name).is_file():
                raise ValueError(f"{camera}: missing {name}")
        meta = json.loads((folder / "model.json").read_text())
        contract = meta.get("sensor_contract") or {}
        ModelConfig(folder / "manifest.json", folder, sensor_contract=contract)
        if meta.get("synthetic") is not False or contract.get("serial") != SERIALS[camera]:
            raise ValueError(f"{camera}: requires a real matching RGB sensor contract")
        threshold = Threshold(**(meta.get("threshold") or {}))
        if (
            threshold.sample_count < 2
            or threshold.score_type != "raw_patchcore_distance_not_probability"
            or threshold.score_type != meta.get("score_type")
        ):
            raise ValueError(f"{camera}: invalid calibration evidence")
        for name, key in (("model.xml", "model_sha256"), ("model.bin", "weights_sha256")):
            if sha256(folder / name) != meta.get(key):
                raise ValueError(f"{camera}: {name} hash mismatch")
        validation = json.loads((folder / "dataset_validation.json").read_text())
        if (
            validation.get("valid") is not True
            or validation.get("manifest_sha256") != threshold.calibration_manifest_sha256
        ):
            raise ValueError(f"{camera}: dataset evidence differs from calibration")
        training = json.loads((folder / "metadata.json").read_text())
        if any(
            training.get(key) != meta.get(key)
            for key in ("checkpoint_sha256", "model_version", "threshold", "sensor_contract")
        ):
            raise ValueError(f"{camera}: training metadata differs from export")
        parity = json.loads((folder / "parity.json").read_text())
        if parity.get("passed") is not True or parity.get("partition") != "val":
            raise ValueError(f"{camera}: requires passing validation parity")
        report = json.loads((folder / "test_report.json").read_text())
        if (
            report.get("partition") != "held_out_test"
            or report.get("synthetic") is not False
            or report.get("model_version") != meta.get("model_version")
            or report.get("threshold_version") != threshold.version
        ):
            raise ValueError(f"{camera}: test evidence differs from frozen model/threshold")
        if check_runtime:
            from brickcell.detector.runtime import OpenVINODetector

            OpenVINODetector(folder / "model.xml")
        config["cameras"][camera]["crop"] = contract["crop"]
        config["cameras"][camera]["models"] = {"rgb": f"../models/{camera}/model.xml"}
        digests[camera] = {name: sha256(folder / name) for name in FILES}
    for camera, folder in sources.items():
        target = output / "models" / camera
        target.mkdir(parents=True)
        for name in FILES:
            shutil.copyfile(folder / name, target / name)
            if sha256(target / name) != digests[camera][name]:
                raise ValueError("Source changed while copying; discard this incomplete bundle")
    save_json(output / "configs/mark-viewer.json", config)
    result = dict(
        schema_version=1,
        files={
            f"models/{camera}/{name}": digest
            for camera, files in digests.items()
            for name, digest in files.items()
        },
        runtime_compiled_on_packaging_host=bool(check_runtime),
        scope="Export handoff only; no physical run or accuracy acceptance implied",
    )
    result["files"]["configs/mark-viewer.json"] = sha256(output / "configs/mark-viewer.json")
    save_json(output / "bundle.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plate", type=Path, required=True, help="Plate export directory")
    parser.add_argument("--wrist", type=Path, required=True, help="Wrist export directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("configs/mark-viewer.json"))
    parser.add_argument("--check-runtime", action="store_true", help="Compile both on OpenVINO CPU")
    args = parser.parse_args()
    print(
        json.dumps(bundle(args.plate, args.wrist, args.output, args.template, args.check_runtime))
    )


if __name__ == "__main__":
    main()
