"""Probe native libraries in subprocesses so a native crash becomes a clear diagnostic."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def probe(code: str) -> dict:
    try:
        process = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        if process.returncode:
            return {"ok": False, "exit_code": process.returncode, "error": process.stderr[-3000:]}
        return {"ok": True, "result": json.loads(process.stdout.strip().splitlines()[-1])}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def doctor(model: Path) -> dict:
    packages = {}
    for name in (
        "anomalib",
        "torch",
        "torchvision",
        "openvino",
        "opencv-python-headless",
        "pillow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch = probe(
        "import json,torch,torchvision; print(json.dumps(dict(torch=torch.__version__,torchvision=torchvision.__version__,mps_available=torch.backends.mps.is_available(),cuda_available=torch.cuda.is_available())))"
    )
    ov = probe(
        "import json,openvino as ov; c=ov.Core(); print(json.dumps(dict(version=ov.__version__,devices=c.available_devices)))"
    )
    images = probe(
        'import json,cv2,numpy as np; from PIL import Image; im=Image.new("RGB",(16,16)); cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2GRAY); print(json.dumps(dict(pillow=True,opencv=True)))'
    )
    warnings = []
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        warnings.append("Native Apple Silicon execution is NOT validated by this host")
    if not ov["ok"]:
        warnings.append(
            "OpenVINO probe failed. On macOS a native Xbyak/JIT crash can be a sandbox restriction; retry from Terminal. No device availability is assumed."
        )
    warnings.append(
        "Training uses CPU even when MPS is available; Intel GPU/NPU and robot hardware are untested"
    )
    assets = {
        str(model.with_suffix(ext)): model.with_suffix(ext).exists()
        for ext in (".xml", ".bin", ".json")
    }
    if not all(assets.values()):
        warnings.append(
            "Trained/exported model assets missing; run train and export before real inference"
        )
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "torch": torch,
        "openvino": ov,
        "images": images,
        "model_assets": assets,
        "warnings": warnings,
        "healthy": all(p is not None for p in packages.values())
        and all(p["ok"] for p in (torch, ov, images)),
    }
