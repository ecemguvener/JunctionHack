"""Bounded remote training job; explicit grouped data and no test evaluation."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from brickcell.data.manifest import load_manifest, require_valid
from brickcell.detector.config import load_config
from brickcell.integrations.core import canonical


def plan(config_path, root):
    root = Path(root).resolve()
    path = (root / config_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Training config must stay in the experiment volume")
    config = load_config(path)
    if not config.manifest.is_relative_to(root) or not config.output_dir.is_relative_to(root):
        raise ValueError("Dataset and output must stay in the experiment volume")
    if config.output_dir.exists():
        raise ValueError("Use a new output to preserve existing models")
    manifest = load_manifest(config.manifest)
    require_valid(manifest)
    if any(not manifest.image_path(sample).is_relative_to(root) for sample in manifest.samples):
        raise ValueError("Training image escaped the experiment volume")
    record = {
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(config.manifest.read_bytes()).hexdigest(),
        "image_hashes": [
            hashlib.sha256(manifest.image_path(s).read_bytes()).hexdigest()
            for s in manifest.samples
        ],
        "synthetic": any(s.synthetic for s in manifest.samples),
        "output": str(config.output_dir),
        "training_device": "CPU",
        "test_evaluation_requested": False,
    }
    record["job_id"] = hashlib.sha256(canonical(record).encode()).hexdigest()
    return path, record


def execute(config_path, root):
    path, record = plan(config_path, root)
    log_dir = Path(root) / "job-logs"
    log_dir.mkdir(exist_ok=True)
    with (log_dir / (record["job_id"] + ".log")).open("x") as log:
        subprocess.run(
            [sys.executable, "-m", "brickcell", "train", "--config", str(path)],
            check=True,
            timeout=3500,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    output = Path(record["output"])
    record["checkpoint_sha256"] = hashlib.sha256((output / "model.ckpt").read_bytes()).hexdigest()
    record["status"] = "trained_and_validation_calibrated_not_test_evaluated"
    (output / "remote-job.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
