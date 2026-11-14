import subprocess
import sys


def test_existing_inspector_api_import_does_not_load_optional_mujoco():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from brickcell.block_inspector import Real2SimPipeline; "
            'assert "mujoco" not in sys.modules; assert callable(Real2SimPipeline)',
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
