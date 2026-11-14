"""Run policy/train_act.py on a CUDA GPU without a change to that script.

train_act.py makes all tensors on the default device. This wrapper sets the
default torch device to CUDA, runs the script as __main__ with the same
arguments, and then writes the true device into report.json (the script
writes "CPU"). The checkpoint loads on the CPU, as before.

Usage: python cloud/act_gpu.py --manifest <manifest.json> --output <dir> --steps N
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1] / "policy"


def main(argv: list[str]) -> int:
    import torch

    device = "CPU"
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        torch.backends.cudnn.benchmark = True
        device = f"CUDA ({torch.cuda.get_device_name(0)})"
        # ACT makes its positional table with torch.from_numpy, which is always on the CPU.
        # Put that table on the default device (CUDA).
        import physicalai.policies.act.model as act_model

        table = act_model._create_sinusoidal_pos_embedding
        act_model._create_sinusoidal_pos_embedding = lambda *a, **k: table(*a, **k).to("cuda")
    print(f"act_gpu: default torch device is {device}", flush=True)
    sys.path.insert(0, str(POLICY))
    sys.argv = [str(POLICY / "train_act.py"), *argv]
    runpy.run_path(str(POLICY / "train_act.py"), run_name="__main__")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/policy/act-smoke"))
    known, _ = parser.parse_known_args(argv)
    report_path = known.output / "report.json"
    report = json.loads(report_path.read_text())
    report["device"] = device
    report["device_note"] = (
        "Set by cloud/act_gpu.py (torch default device); train_act.py is unchanged"
    )
    report_path.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
