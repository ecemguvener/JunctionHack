"""Run the full PatchCore workflow in ONE Python process: validate-data, train, export,
parity and evaluate. The heavy imports (torch, anomalib, openvino) load one time.

Each step calls the normal Brickcell CLI (brickcell.cli.main), so the step code is the
same as on a laptop. Only the fit uses the GPU: brickcell.detector.workflow.train sets
accelerator="cpu", and this wrapper replaces anomalib.engine.Engine with a subclass
that uses the GPU when the engine has max_epochs (only the training engine has it).
Lightning moves the model back to the CPU at the end of fit. Calibration, export,
parity and evaluation use the CPU, as on the Intel laptop.

Usage:
  python cloud/patchcore_steps.py --config CFG --manifest M --validation-output V \
      --timings T [--cpu-train]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Deterministic cuBLAS kernels. Lightning sets deterministic=True for the fit.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def install_gpu_engine() -> str:
    """Make the training Engine use CUDA. Return the device name."""
    import anomalib.engine
    import torch

    if not torch.cuda.is_available():
        print("patchcore_steps: CUDA is not available, training uses the CPU", flush=True)
        return "cpu"
    base = anomalib.engine.Engine

    class GpuEngine(base):
        def __init__(self, *args, **kwargs):
            if kwargs.get("accelerator") == "cpu" and kwargs.get("max_epochs") is not None:
                kwargs["accelerator"] = "gpu"
            super().__init__(*args, **kwargs)

    anomalib.engine.Engine = GpuEngine
    name = torch.cuda.get_device_name(0)
    print(f"patchcore_steps: training engine uses CUDA ({name})", flush=True)
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--cpu-train", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    import anomalib.engine  # noqa: F401  (the import time is measured on its own)
    import openvino  # noqa: F401
    import torch  # noqa: F401

    from brickcell.cli import main as cli

    timings = {"imports": round(time.perf_counter() - started, 3)}
    device = "cpu" if args.cpu_train else install_gpu_engine()
    steps = [
        (
            "validate-data",
            ["validate-data", "--manifest", args.manifest, "--output", args.validation_output],
        ),
        ("train", ["train", "--config", args.config]),
        ("export", ["export", "--config", args.config]),
        ("parity", ["parity", "--config", args.config]),
        ("evaluate", ["evaluate", "--config", args.config]),
    ]
    results = []
    for name, argv in steps:
        tick = time.perf_counter()
        code = cli(argv)
        seconds = round(time.perf_counter() - tick, 3)
        results.append({"step": name, "seconds": seconds, "returncode": code})
        print(f"[{name}] exit {code} after {seconds} s", flush=True)
        if code != 0:
            break
    args.timings.write_text(
        json.dumps({"train_device": device, "timings": timings, "steps": results}, indent=2)
    )
    return 0 if all(r["returncode"] == 0 for r in results) and len(results) == len(steps) else 1


if __name__ == "__main__":
    sys.exit(main())
