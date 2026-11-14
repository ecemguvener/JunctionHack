"""Modal-only CPU training experiment. Core environment never imports this module."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
app = modal.App("brickcell-patchcore-experiments")
volume = modal.Volume.from_name("brickcell-inspection-experiments", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv==0.13.8")
    .add_local_file(str(ROOT / "pyproject.toml"), "/opt/brickcell/pyproject.toml", copy=True)
    .add_local_file(str(ROOT / "uv.lock"), "/opt/brickcell/uv.lock", copy=True)
    .add_local_dir(str(ROOT / "src"), "/opt/brickcell/src", copy=True)
    .run_commands("cd /opt/brickcell && uv sync --locked --no-dev")
)


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={"/experiments": volume})
def train(config: str):
    import sys

    sys.path.insert(0, "/opt/brickcell/src")
    # Use the locked image environment for the actual trainer subprocess.
    import json
    import subprocess

    code = (
        "import json; from brickcell.integrations.training_job import execute; "
        f'print(json.dumps(execute({config!r}, "/experiments")))'
    )
    result = subprocess.run(
        ["/opt/brickcell/.venv/bin/python", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=3550,
    )
    volume.commit()
    return json.loads(result.stdout)


@app.local_entrypoint()
def main(config: str):
    print(train.remote(config))
