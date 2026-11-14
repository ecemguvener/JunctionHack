"""Brickcell training jobs on Modal (modal.com).

The jobs use the CURRENT local working tree: src/, configs/, assets/, policy/ and
cloud/ go into each container when it starts. A source change does not cause an
image build. Only a change to a lock file or to configs/mujoco-requirements.txt does.

Commands (run from the repository root, with the `modal` client installed):

  modal run cloud/modal_train.py::warm                         # build all images
  modal run cloud/modal_train.py::pipeline --dataset two-bowl  # generate -> PatchCore
  modal run cloud/modal_train.py::pipeline --dataset two-bowl --palette off  # old scene colours
  modal run cloud/modal_train.py::generate --dataset two-bowl
  modal run cloud/modal_train.py::train_patchcore --dataset two-bowl
  modal run cloud/modal_train.py::record_policy --demos policy-smoke --episodes 6
  modal run cloud/modal_train.py::train_act --demos policy-smoke --steps 2000
  modal run cloud/modal_train.py::evaluate_act --run-id <train_act run id>

See docs/MODAL_TRAINING.md.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modal_helpers as helpers  # noqa: E402

APP_NAME = "brickcell-train"
VOLUME_NAME = "brickcell-train"
# Mid-range GPUs. The list is a fallback order: Modal takes the first one with capacity.
# (On 2026-11-14 an L4-only request waited more than 10 minutes for capacity.)
PATCHCORE_GPU = ["L4", "A10G", "L40S"]
ACT_GPU = ["L4", "A10G", "L40S"]
VOL = Path("/vol")
REPO = Path("/repo")  # the local tree in the container
CACHE = Path("/opt/cache")  # downloads made at image build time
WORK = Path("/tmp/work")
CORE_PY = "/opt/core/.venv/bin/python"
POLICY_PY = "/opt/policy/.venv/bin/python"
SIM_PY = "/opt/sim/bin/python"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
# Same versions as uv.lock and policy/uv.lock, but the CUDA build (the locks pin the CPU build).
TORCH_CUDA = ("torch==2.10.0+cu128", "torchvision==0.25.0+cu128")
MINUTE = 60
DOWNLOAD_THREADS = 32  # parallel file reads from the Volume
# Modal image builders set a PyPI mirror. The lock files name pypi.org and
# download.pytorch.org, so `uv sync --locked` fails with the mirror set.
UNSET_INDEX = "env " + " ".join(
    f"-u {name}"
    for name in (
        "UV_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_EXTRA_INDEX_URL",
        "UV_INDEX",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
    )
)

LOCAL_REPO = helpers.REPO

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUMES = {str(VOL): volume}

CACHE_ENV = {
    "MENAGERIE_CACHE_DIR": str(CACHE / "menagerie"),
    "TORCH_HOME": str(CACHE / "torch"),
    "HF_HOME": str(CACHE / "huggingface"),
}
RUNTIME_ENV = {
    **CACHE_ENV,
    "PYTHONPATH": str(REPO / "src"),
    "PYTHONUNBUFFERED": "1",
    "HF_HUB_OFFLINE": "1",  # the weights are in the image; no hub round trip
    "XDG_CACHE_HOME": "/tmp/xdg-cache",
    "MPLCONFIGDIR": "/tmp/matplotlib",
    # Headless MuJoCo rendering on a CPU container: OSMesa (Mesa software renderer).
    "MUJOCO_GL": "osmesa",
    "PYOPENGL_PLATFORM": "osmesa",
}


def _base_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("libosmesa6", "libgl1", "libegl1", "libglib2.0-0")
        .pip_install("uv==0.13.4")
        .env(CACHE_ENV)
    )


def _with_sim_env(image: modal.Image) -> modal.Image:
    """MuJoCo env from configs/mujoco-requirements.txt, with the SO-101 model in the image."""
    requirements = LOCAL_REPO / "configs" / "mujoco-requirements.txt"
    prefetch = (
        "from physicalai_mujoco_plugin.compose import fetch_profile; "
        "from physicalai_mujoco_plugin.profiles import get_profile; "
        "print(fetch_profile(get_profile('so101')))"
    )
    return image.add_local_file(
        requirements, "/opt/req/mujoco-requirements.txt", copy=True
    ).run_commands(
        "uv venv /opt/sim --python 3.12",
        f"uv pip install --python {SIM_PY} -r /opt/req/mujoco-requirements.txt",
        f'{SIM_PY} -c "{prefetch}"',
    )


def _with_locked_env(image: modal.Image, project: str, target: str, prefetch: str) -> modal.Image:
    """`uv sync --locked`, the CUDA build of the same torch versions, then the weights."""
    local = LOCAL_REPO / project if project else LOCAL_REPO
    python = f"{target}/.venv/bin/python"
    return (
        image.add_local_file(local / "pyproject.toml", f"{target}/pyproject.toml", copy=True)
        .add_local_file(local / "uv.lock", f"{target}/uv.lock", copy=True)
        .run_commands(
            f"cd {target} && {UNSET_INDEX} uv sync --locked --no-install-project --no-dev "
            "--python 3.12",
            f"uv pip install --python {python} --index-url {TORCH_CUDA_INDEX} "
            + " ".join(TORCH_CUDA),
            f'{python} -c "{prefetch}"',
        )
    )


def _with_source(image: modal.Image) -> modal.Image:
    """Add the local tree at container start (copy=False): no image build for source edits."""
    image = image.env(RUNTIME_ENV)
    for name in helpers.SOURCE_DIRS:
        image = image.add_local_dir(
            LOCAL_REPO / name, str(REPO / name), ignore=helpers.SOURCE_IGNORE
        )
    return image.add_local_file(Path(helpers.__file__), "/root/modal_helpers.py")


TIMM_PREFETCH = "import timm; timm.create_model('resnet18.tv_in1k', pretrained=True)"
TORCHVISION_PREFETCH = "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"

sim_image = _with_source(_with_sim_env(_base_image()))
core_image = _with_source(_with_locked_env(_base_image(), "", "/opt/core", TIMM_PREFETCH))
policy_image = _with_source(
    _with_locked_env(_with_sim_env(_base_image()), "policy", "/opt/policy", TORCHVISION_PREFETCH)
)


# ----------------------------------------------------------------------------------------------
# Container-side helpers
# ----------------------------------------------------------------------------------------------


def _container_info(submitted_at: float | None = None, started_at: float | None = None) -> dict:
    gpu = None
    try:
        query = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        gpu = subprocess.run(query, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    info = {
        "function_call_id": modal.current_function_call_id(),
        "task_id": os.environ.get("MODAL_TASK_ID"),
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
        "region": os.environ.get("MODAL_REGION"),
        "cloud": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "gpu": gpu or None,
        "cpu_count": os.cpu_count(),
    }
    if submitted_at and started_at:
        # Time from the local .remote() call to the start of the function body: scheduling,
        # image load and container start. The two clocks can differ by a fraction of a second.
        info["cold_start_seconds"] = round(started_at - submitted_at, 3)
    return info


def _run(name: str, command: list[str], log: Path, cwd: Path = WORK) -> dict:
    """Run one step, print its output, append it to `log` and measure the wall time."""
    cwd.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] $ {' '.join(command)}", flush=True)
    started = time.perf_counter()
    with log.open("a") as stream:
        stream.write(f"\n### {name}: {' '.join(command)}\n")
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            stream.write(line)
        code = process.wait()
    seconds = round(time.perf_counter() - started, 3)
    print(f"[{name}] exit {code} after {seconds} s", flush=True)
    return {"step": name, "seconds": seconds, "returncode": code}


def _check(steps: list[dict]) -> None:
    failed = [s["step"] for s in steps if s["returncode"] != 0]
    if failed:
        raise RuntimeError(f"Steps failed: {failed}. See the run log on the volume.")


def _new_dataset_dir(name: str, overwrite: bool) -> Path:
    target = VOL / "data" / name
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Volume already has data/{name}. Use --overwrite (the old one moves aside) "
                "or a new name."
            )
        print(f"Moved old volume data to {helpers.move_aside(target)}", flush=True)
    return target


def _prefetch(root: Path) -> dict:
    """Read all files under ``root`` in parallel, so that later reads come from the cache.

    One Volume read has a long latency; validate-data reads the images one at a time.
    """
    started = time.perf_counter()
    files = [p for p in root.rglob("*") if p.is_file()]
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        size = sum(pool.map(lambda p: len(p.read_bytes()), files))
    return {"files": len(files), "bytes": size, "seconds": round(time.perf_counter() - started, 3)}


def _copy_tree(source: Path, target: Path) -> None:
    """shutil.copytree with parallel file copies (for a Volume source or target)."""
    files = [p for p in source.rglob("*") if p.is_file()]
    for path in files:
        (target / path.relative_to(source)).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        list(pool.map(lambda p: shutil.copy2(p, target / p.relative_to(source)), files))


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# ----------------------------------------------------------------------------------------------
# Remote functions: detector data and PatchCore
# ----------------------------------------------------------------------------------------------


@app.function(image=sim_image, cpu=4, memory=4096, timeout=20 * MINUTE, volumes=VOLUMES)
def generate_shard_job(
    run_id: str,
    shard: int,
    units: list[str],
    scene_config: str,
    submitted_at: float,
    palette: str = "all",
    color_config: str = "",
) -> dict:
    """Render a few episodes (cloud/generate_shard.py) into the volume staging area."""
    started_at = time.time()
    scratch = Path("/tmp/shard")
    shutil.rmtree(scratch, ignore_errors=True)
    command = [SIM_PY, str(REPO / "cloud" / "generate_shard.py"), "render", "--output"]
    command += [str(scratch), "--palette", palette, "--units", *units]
    if scene_config:
        command += ["--scene-config", str(REPO / scene_config)]
    if color_config:
        command += ["--color-config", str(REPO / color_config)]
    log = VOL / "runs" / "generate" / run_id / f"shard-{shard:02d}.log"
    steps = [_run("render", command, log, cwd=REPO)]
    _check(steps)
    staging = VOL / "staging" / run_id / f"shard-{shard:02d}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(scratch, staging)
    volume.commit()
    timing = json.loads((scratch / "rows.json").read_text())["timing"]
    return {
        "shard": shard,
        "steps": steps,
        "timing": timing,
        **_container_info(submitted_at, started_at),
    }


@app.function(image=sim_image, cpu=2, memory=2048, timeout=10 * MINUTE, volumes=VOLUMES)
def merge_dataset_job(
    dataset: str,
    run_id: str,
    shards: int,
    overwrite: bool,
    palette: str = "all",
    episodes: tuple[int, int, int] = helpers.PALETTE_EPISODES,
    color_config: str = "",
) -> dict:
    """Join the shard outputs into data/<dataset> with the generate.py row order."""
    volume.reload()
    staging = VOL / "staging" / run_id
    scratch = Path("/tmp/merged") / dataset
    shutil.rmtree(scratch.parent, ignore_errors=True)
    command = [SIM_PY, str(REPO / "cloud" / "generate_shard.py"), "merge", "--output"]
    command += [str(scratch), "--palette", palette, "--episodes", *map(str, episodes)]
    if color_config:
        command += ["--color-config", str(REPO / color_config)]
    command += [str(staging / f"shard-{i:02d}") for i in range(shards)]
    steps = [_run("merge", command, VOL / "runs" / "generate" / run_id / "merge.log", cwd=REPO)]
    _check(steps)
    target = _new_dataset_dir(dataset, overwrite)
    _copy_tree(scratch, target)
    shutil.rmtree(staging, ignore_errors=True)
    volume.commit()
    manifest = json.loads((target / "manifest.json").read_text())
    return {"dataset": dataset, "images": len(manifest["samples"]), "steps": steps}


@app.function(image=sim_image, cpu=4, memory=4096, timeout=60 * MINUTE, volumes=VOLUMES)
def generate_job(
    dataset: str,
    run_id: str,
    overwrite: bool,
    scene_config: str,
    palette: str = "all",
    episodes: tuple[int, int, int] = helpers.PALETTE_EPISODES,
    color_config: str = "",
) -> dict:
    """brickcell.simulation.generate in one container: one random generator (slow)."""
    target = _new_dataset_dir(dataset, overwrite)
    scratch = Path("/tmp/generate") / dataset
    shutil.rmtree(scratch.parent, ignore_errors=True)
    command = [SIM_PY, "-m", "brickcell.simulation.generate", "--output", str(scratch)]
    command += ["--palette", palette, "--episodes", *map(str, episodes)]
    if scene_config:
        command += ["--scene-config", str(REPO / scene_config)]
    if color_config:
        command += ["--color-config", str(REPO / color_config)]
    steps = [_run("generate", command, VOL / "runs" / "generate" / run_id / "generate.log", REPO)]
    _check(steps)
    shutil.copytree(scratch, target)
    volume.commit()
    manifest = json.loads((target / "manifest.json").read_text())
    return {"dataset": dataset, "images": len(manifest["samples"]), "steps": steps}


@app.function(
    image=core_image, gpu=PATCHCORE_GPU, cpu=4, memory=16384, timeout=45 * MINUTE, volumes=VOLUMES
)
def train_patchcore_job(
    dataset: str,
    run_id: str,
    submitted_at: float = 0.0,
    base_config: str = "configs/patchcore-sim.yaml",
    gpu_train: bool = True,
) -> dict:
    """validate-data, train (GPU), export, parity, evaluate (CPU) in ONE warm process."""
    started_at = time.time()
    volume.reload()
    manifest = VOL / "data" / dataset / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"No dataset on the volume: data/{dataset}. Run generate first.")
    prefetch = _prefetch(manifest.parent)
    print(f"Prefetched the dataset: {prefetch}", flush=True)
    run = VOL / "runs" / "patchcore" / run_id
    model = run / "model"
    # The function runs in the image Python; PyYAML is only in the core env.
    read = "import json, sys, yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))"
    base = json.loads(
        subprocess.run(
            [CORE_PY, "-c", read, str(REPO / base_config)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    config = helpers.write_json_yaml(
        run / "config.yaml", helpers.patchcore_config(base, manifest, model)
    )
    command = [
        CORE_PY, str(REPO / "cloud" / "patchcore_steps.py"), "--config", str(config),
        "--manifest", str(manifest), "--validation-output", str(model / "data-validation.json"),
        "--timings", str(run / "timings.json"),
    ]  # fmt: skip
    if not gpu_train:
        command.append("--cpu-train")
    process = _run("patchcore", command, run / "pipeline.log")
    timings = _read_json(run / "timings.json") or {}
    metadata = _read_json(model / "metadata.json") or {}
    parity = _read_json(model / "parity.json") or {}
    test = _read_json(model / "test_report.json") or {}
    result = {
        "dataset": dataset,
        "run_id": run_id,
        "volume_path": str(run),
        "process_seconds": process["seconds"],
        "prefetch": prefetch,
        "imports_seconds": (timings.get("timings") or {}).get("imports"),
        "steps": timings.get("steps", []),
        "container": _container_info(submitted_at, started_at),
        "train_device": timings.get("train_device"),
        "export_device": "CPU",
        "torch_build": TORCH_CUDA,
        "training_seconds_reported": metadata.get("training_seconds"),
        "model_version": metadata.get("model_version"),
        "threshold": (metadata.get("threshold") or {}).get("value"),
        "parity_passed": parity.get("passed"),
        "parity_max_absolute_difference": parity.get("max_absolute_difference"),
        "test_metrics": test.get("binary_metrics_at_frozen_threshold"),
    }
    helpers.write_json_yaml(model / "modal_result.json", result)
    volume.commit()
    _check([process])
    return result


# ----------------------------------------------------------------------------------------------
# Remote functions: ACT policy
# ----------------------------------------------------------------------------------------------


@app.function(image=policy_image, cpu=4, memory=8192, timeout=60 * MINUTE, volumes=VOLUMES)
def record_policy_shard(run_id: str, plan: dict) -> dict:
    """Record a slice of scripted-teacher episodes (seed + i, as in one local run)."""
    scratch = Path("/tmp/record") / f"shard-{plan['shard']:02d}"
    shutil.rmtree(scratch, ignore_errors=True)
    command = [
        SIM_PY, "-m", "brickcell.simulation.record_policy",
        "--output", str(scratch), "--episodes", str(plan["episodes"]), "--seed", str(plan["seed"]),
    ]  # fmt: skip
    log = VOL / "runs" / "record_policy" / run_id / f"shard-{plan['shard']:02d}.log"
    steps = [_run("record_policy", command, log, cwd=REPO)]
    _check(steps)
    staging = VOL / "staging" / run_id / f"shard-{plan['shard']:02d}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(scratch, staging, ignore=shutil.ignore_patterns("scene"))
    volume.commit()
    return {"plan": plan, "steps": steps, "container": _container_info()}


@app.function(image=policy_image, cpu=2, memory=4096, timeout=20 * MINUTE, volumes=VOLUMES)
def merge_policy_job(demos: str, run_id: str, plans: list[dict], overwrite: bool = False) -> dict:
    """Merge shard outputs into one dataset and validate it with brickcell.policy.data."""
    volume.reload()
    staging = VOL / "staging" / run_id
    shards = [
        (plan, json.loads((staging / f"shard-{plan['shard']:02d}" / "manifest.json").read_text()))
        for plan in plans
    ]
    merged, moves = helpers.merge_policy_manifests(shards)
    scratch = Path("/tmp/merge") / demos
    shutil.rmtree(scratch.parent, ignore_errors=True)
    scratch.mkdir(parents=True)
    for source, name in moves:
        shutil.copy2(staging / source, scratch / name)
    (scratch / "manifest.json").write_text(json.dumps(merged, indent=2))
    check = (
        "import sys; from pathlib import Path; from brickcell.policy.data import load_episodes; "
        "[print(s, len(load_episodes(Path(sys.argv[1]), s))) for s in ('train', 'val', 'test')]"
    )
    log = VOL / "runs" / "record_policy" / run_id / "merge.log"
    steps = [_run("validate", [SIM_PY, "-c", check, str(scratch / "manifest.json")], log)]
    _check(steps)
    target = _new_dataset_dir(demos, overwrite)
    shutil.copytree(scratch, target)
    shutil.rmtree(staging, ignore_errors=True)
    volume.commit()
    return {
        "demos": demos,
        "episodes": len(merged["episodes"]),
        "frames": sum(row["frames"] for row in merged["episodes"]),
        "steps": steps,
    }


@app.function(
    image=policy_image, gpu=ACT_GPU, cpu=8, memory=16384, timeout=120 * MINUTE, volumes=VOLUMES
)
def train_act_job(demos: str, run_id: str, steps: int = 2000, batch_size: int = 8) -> dict:
    """Train ACT with policy/train_act.py on the GPU (wrapper: cloud/act_gpu.py)."""
    volume.reload()
    manifest = VOL / "data" / demos / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"No demonstrations on the volume: data/{demos}.")
    run = VOL / "runs" / "act" / run_id
    output = run / "policy"
    command = [
        POLICY_PY, str(REPO / "cloud" / "act_gpu.py"), "--manifest", str(manifest),
        "--output", str(output), "--steps", str(steps), "--batch-size", str(batch_size),
    ]  # fmt: skip
    process = [_run("train_act", command, run / "train.log")]
    report = _read_json(output / "report.json") or {}
    result = {
        "demos": demos,
        "run_id": run_id,
        "steps": process,
        "container": _container_info(),
        "training_steps": steps,
        "batch_size": batch_size,
        "training_seconds_reported": report.get("training_seconds"),
        "seconds_per_step": (report["training_seconds"] / steps) if report else None,
        "device": report.get("device"),
        "first_loss": report.get("first_loss"),
        "last_loss": report.get("last_loss"),
        "torch_build": TORCH_CUDA,
    }
    helpers.write_json_yaml(output / "modal_result.json", result)
    volume.commit()
    _check(process)
    return result


@app.function(image=policy_image, cpu=4, memory=8192, timeout=60 * MINUTE, volumes=VOLUMES)
def evaluate_act_job(
    run_id: str, seeds: list[int], max_steps: int = 600, clip_to_sim_limits: bool = False
) -> dict:
    """Teacher-free rollouts with policy/evaluate_act.py on the CPU (the simulator is the cost)."""
    volume.reload()
    run = VOL / "runs" / "act" / run_id
    checkpoint = run / "policy" / "policy.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint on the volume: {checkpoint}")
    # evaluate_act.py expects ./artifacts/mujoco-venv, ./src and ./.cache/menagerie.
    work = Path("/tmp/eval")
    (work / "artifacts").mkdir(parents=True, exist_ok=True)
    (work / ".cache").mkdir(parents=True, exist_ok=True)
    for link, target in [
        (work / "artifacts" / "mujoco-venv", Path("/opt/sim")),
        (work / "src", REPO / "src"),
        (work / ".cache" / "menagerie", CACHE / "menagerie"),
    ]:
        if not link.exists():
            link.symlink_to(target)
    name = "rollouts-bounded.json" if clip_to_sim_limits else "rollouts.json"
    output = run / "policy" / name
    command = [
        POLICY_PY, str(REPO / "policy" / "evaluate_act.py"), "--checkpoint", str(checkpoint),
        "--output", str(output), "--max-steps", str(max_steps), "--seeds", *map(str, seeds),
    ]  # fmt: skip
    if clip_to_sim_limits:
        command.append("--clip-to-sim-limits")
    steps = [_run("evaluate_act", command, run / "evaluate.log", cwd=work)]
    report = _read_json(output) or {}
    volume.commit()
    _check(steps)
    return {
        "run_id": run_id,
        "steps": steps,
        "container": _container_info(),
        "successful_episodes": report.get("successful_episodes"),
    }


# ----------------------------------------------------------------------------------------------
# Local side: download and provenance
# ----------------------------------------------------------------------------------------------


def _download(remote: str, local: Path, run_id: str) -> int:
    """Copy a volume directory to `local`. An existing local copy moves aside first."""
    from modal.volume import FileEntryType

    remote = remote.strip("/")
    entries = [e for e in volume.listdir(remote, recursive=True) if e.type == FileEntryType.FILE]
    if not entries:
        print(f"Nothing to download at {remote}")
        return 0
    partial = local.with_name(f"{local.name}.partial-{run_id}")
    shutil.rmtree(partial, ignore_errors=True)

    def fetch(entry) -> None:
        path = partial / Path(entry.path.strip("/")).relative_to(remote)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            for chunk in volume.read_file(entry.path):
                stream.write(chunk)

    # One request per file has a long round trip, so read many files at the same time.
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        list(pool.map(fetch, entries))
    local.parent.mkdir(parents=True, exist_ok=True)
    moved = helpers.move_aside(local)
    if moved:
        print(f"Moved old local copy to {moved}")
    partial.rename(local)
    print(f"Downloaded {len(entries)} files: volume:/{remote} -> {local}")
    return len(entries)


def _provenance(run_id: str, kind: str, local: dict, phases: dict, **parts) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "run_id": run_id,
        "modal_app_name": APP_NAME,
        "modal_app_id": app.app_id,
        "modal_volume": VOLUME_NAME,
        "modal_client_version": modal.__version__,
        "gpu_fallback_order": {"patchcore": PATCHCORE_GPU, "act": ACT_GPU},
        "local_tree": local,
        "phase_seconds": phases,
        **parts,
    }


def _save_record(run_id: str, provenance: dict, copy_to: Path | None = None) -> None:
    path = helpers.write_json_yaml(helpers.run_record_dir(run_id) / "provenance.json", provenance)
    if copy_to is not None and copy_to.exists():
        helpers.write_json_yaml(copy_to / "modal_provenance.json", provenance)
    print(f"Provenance: {path}")


def _timed(phases: dict, name: str, call, *args, **kwargs):
    tick = time.perf_counter()
    result = call(*args, **kwargs)
    phases[name] = round(time.perf_counter() - tick, 3)
    return result


def _generate(dataset, run_id, overwrite, scene_config, sharded, episodes_per_shard, phases, data):
    """``data``: palette, episodes ("TRAIN,VAL,TEST" or ""), color_config."""
    palette, color_config = data["palette"], data["color_config"]
    episodes = helpers.generate_episodes(palette, data["episodes"])
    plan = {"palette": palette, "episodes": list(episodes), "color_config": color_config}
    if not sharded:
        job = generate_job.remote
        args = (dataset, run_id, overwrite, scene_config, palette, episodes, color_config)
        return {**plan, **_timed(phases, "generate", job, *args)}
    units = helpers.generate_units(episodes)
    shards = [units[i : i + episodes_per_shard] for i in range(0, len(units), episodes_per_shard)]
    now = time.time()
    print(f"Rendering {len(units)} episodes ({palette} palette) in {len(shards)} containers")
    calls = [
        (run_id, i, shard, scene_config, now, palette, color_config)
        for i, shard in enumerate(shards)
    ]
    results = _timed(phases, "generate_render", lambda: list(generate_shard_job.starmap(calls)))
    merged = _timed(
        phases,
        "generate_merge",
        merge_dataset_job.remote,
        dataset,
        run_id,
        len(shards),
        overwrite,
        palette,
        episodes,
        color_config,
    )
    shard_seconds = sorted(r["steps"][0]["seconds"] for r in results)
    return {
        **plan,
        "shards": len(shards),
        "episodes_per_shard": episodes_per_shard,
        "cold_start_seconds_max": max(r.get("cold_start_seconds", 0) for r in results),
        "shard_seconds_max": shard_seconds[-1],
        "shard_seconds_median": shard_seconds[len(shard_seconds) // 2],
        "shard_seconds_sum": round(sum(shard_seconds), 3),
        "world_setup_seconds_max": max(r["timing"]["world_setup_seconds"] for r in results),
        "merge": merged,
        "per_shard": results,
    }


def _patchcore(dataset, model_name, run_id, download, gpu_train, phases) -> dict:
    result = _timed(
        phases,
        "patchcore",
        train_patchcore_job.remote,
        dataset,
        run_id,
        time.time(),
        gpu_train=gpu_train,
    )
    if download:
        target = helpers.local_targets("patchcore", model_name=model_name, dataset=dataset)["model"]
        _timed(phases, "download", _download, f"runs/patchcore/{run_id}/model", target, run_id)
        result["local_model"] = str(target)
    return result


# ----------------------------------------------------------------------------------------------
# Local entrypoints (the CLI)
# ----------------------------------------------------------------------------------------------


@app.local_entrypoint()
def warm():
    """Build (or confirm) all images. `modal run` builds every image of the app first."""
    print("All images are built and cached.")


def _data_options(palette: str, episodes: str, color_config: str) -> dict:
    helpers.generate_episodes(palette, episodes)  # stop early on a bad value
    return {"palette": palette, "episodes": episodes, "color_config": color_config}


def _download_data(dataset: str, run_id: str, phases: dict) -> Path:
    target = helpers.local_targets("patchcore", model_name="-", dataset=dataset)["data"]
    _timed(phases, "download_data", _download, f"data/{dataset}", target, run_id)
    return target


def _write_per_color(model_dir: Path, data_dir: Path) -> dict | None:
    """Group the test results of the downloaded model by brick colour."""
    manifest, report = data_dir / "manifest.json", model_dir / "test_report.json"
    if not (manifest.exists() and report.exists()):
        return None
    result = helpers.per_color_report(
        json.loads(manifest.read_text()), json.loads(report.read_text())
    )
    helpers.write_json_yaml(model_dir / "per_color_test.json", result)
    return result


@app.local_entrypoint()
def generate(
    dataset: str = "two-bowl",
    overwrite: bool = False,
    scene_config: str = "",
    sharded: bool = True,
    episodes_per_shard: int = 1,
    download_data: bool = False,
    palette: str = "all",
    episodes: str = "",
    color_config: str = "",
):
    """Render the PatchCore dataset into the volume (data/<dataset>).

    --palette off|main|all (brick colours), --episodes TRAIN,VAL,TEST (default by palette).
    """
    run_id, local, phases = helpers.new_run_id(), helpers.local_provenance(), {}
    data = _data_options(palette, episodes, color_config)
    result = _generate(
        dataset, run_id, overwrite, scene_config, sharded, episodes_per_shard, phases, data
    )
    if download_data:
        _download_data(dataset, run_id, phases)
    _save_record(run_id, _provenance(run_id, "generate", local, phases, generate=result))
    print(
        json.dumps(
            {"phases": phases, "generate": {k: v for k, v in result.items() if k != "per_shard"}},
            indent=2,
        )
    )


@app.local_entrypoint()
def download_data(dataset: str = "two-bowl"):
    """Download data/<dataset> from the Volume to data/<dataset> (the old copy moves aside)."""
    phases = {}
    target = _download_data(dataset, helpers.new_run_id(), phases)
    print(json.dumps({"phases": phases, "local": str(target)}))


@app.local_entrypoint()
def train_patchcore(
    dataset: str = "two-bowl",
    model_name: str = "patchcore-sim",
    download: bool = True,
    gpu_train: bool = True,
):
    """Train, export, parity-check and evaluate PatchCore on a volume dataset."""
    run_id, local, phases = helpers.new_run_id(), helpers.local_provenance(), {}
    result = _patchcore(dataset, model_name, run_id, download, gpu_train, phases)
    target = Path(result["local_model"]) if download else None
    _save_record(run_id, _provenance(run_id, "patchcore", local, phases, patchcore=result), target)
    print(json.dumps({"phases": phases, "patchcore": result}, indent=2))


@app.local_entrypoint()
def pipeline(
    dataset: str = "two-bowl",
    model_name: str = "patchcore-sim",
    overwrite: bool = False,
    scene_config: str = "",
    sharded: bool = True,
    episodes_per_shard: int = 1,
    download: bool = True,
    gpu_train: bool = True,
    palette: str = "all",
    episodes: str = "",
    color_config: str = "",
    download_data: bool = True,
):
    """generate (parallel) -> validate-data -> train -> export -> parity -> evaluate.

    --palette off|main|all (brick colours), --episodes TRAIN,VAL,TEST (default by palette).
    With --download-data (default) the dataset also goes to data/<dataset> (the old copy moves
    aside), so that `brickcell parity` runs on the laptop, and per_color_test.json is written.
    """
    started = time.perf_counter()
    run_id, local, phases = helpers.new_run_id(), helpers.local_provenance(), {}
    data = _data_options(palette, episodes, color_config)
    git = local["git"]
    print(
        f"Run {run_id}: dataset={dataset} model={model_name} git={git['commit']} dirty={git['dirty']}"
    )
    generated = _generate(
        dataset, run_id, overwrite, scene_config, sharded, episodes_per_shard, phases, data
    )
    trained = _patchcore(dataset, model_name, run_id, download, gpu_train, phases)
    target = Path(trained["local_model"]) if download else None
    if download_data:
        data_dir = _download_data(dataset, run_id, phases)
        if target is not None:
            trained["per_color_test"] = _write_per_color(target, data_dir)
    phases["total"] = round(time.perf_counter() - started, 3)
    provenance = _provenance(
        run_id, "pipeline", local, phases, generate=generated, patchcore=trained
    )
    _save_record(run_id, provenance, target)
    summary = {k: v for k, v in generated.items() if k != "per_shard"}
    print(json.dumps({"phases": phases, "generate": summary, "patchcore": trained}, indent=2))


def _record(demos, episodes, shards, seed, overwrite, run_id, phases) -> dict:
    plans = helpers.plan_shards(episodes, shards, seed)
    print(f"Recording {episodes} episodes in {len(plans)} parallel containers")
    recorded = _timed(
        phases, "record", lambda: list(record_policy_shard.starmap([(run_id, p) for p in plans]))
    )
    merged = _timed(
        phases, "record_merge", merge_policy_job.remote, demos, run_id, plans, overwrite
    )
    return {"shards": recorded, "merge": merged}


@app.local_entrypoint()
def record_policy(
    demos: str = "policy-smoke",
    episodes: int = 6,
    shards: int = 2,
    seed: int = 4200,
    overwrite: bool = False,
):
    """Record scripted-teacher demonstrations in parallel shards into the volume data/<demos>."""
    run_id, local, phases = helpers.new_run_id(), helpers.local_provenance(), {}
    result = _record(demos, episodes, shards, seed, overwrite, run_id, phases)
    _save_record(run_id, _provenance(run_id, "record_policy", local, phases, record_policy=result))
    print(json.dumps({"phases": phases, **result}, indent=2))


@app.local_entrypoint()
def train_act(
    demos: str = "policy-smoke",
    policy_name: str = "act-modal",
    steps: int = 2000,
    batch_size: int = 8,
    gpu: str = "",
    download: bool = True,
):
    """Train ACT on a GPU from volume demonstrations. Download to artifacts/policy/<name>."""
    run_id, local, phases = helpers.new_run_id(), helpers.local_provenance(), {}
    job = train_act_job.with_options(gpu=gpu) if gpu else train_act_job
    result = _timed(phases, "train_act", job.remote, demos, run_id, steps, batch_size)
    target = helpers.local_targets("act", policy_name=policy_name, demos=demos)["policy"]
    if download:
        _timed(phases, "download", _download, f"runs/act/{run_id}/policy", target, run_id)
    provenance = _provenance(run_id, "train_act", local, phases, train_act=result)
    _save_record(run_id, provenance, target if download else None)
    print(json.dumps({"phases": phases, "train_act": result}, indent=2))
    print(f"Evaluate: modal run cloud/modal_train.py::evaluate_act --run-id {run_id}")


@app.local_entrypoint()
def evaluate_act(
    run_id: str,
    policy_name: str = "act-modal",
    seeds: str = "5200,5201",
    max_steps: int = 600,
    clip_to_sim_limits: bool = False,
):
    """Teacher-free rollouts of a train_act run; the report goes to artifacts/policy/<name>/."""
    local, phases = helpers.local_provenance(), {}
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    result = _timed(
        phases,
        "evaluate_act",
        evaluate_act_job.remote,
        run_id,
        seed_list,
        max_steps,
        clip_to_sim_limits,
    )
    name = "rollouts-bounded.json" if clip_to_sim_limits else "rollouts.json"
    target = helpers.local_targets("act", policy_name=policy_name, demos="-")["policy"] / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        for chunk in volume.read_file(f"runs/act/{run_id}/policy/{name}"):
            stream.write(chunk)
    print(f"Downloaded {target}")
    provenance = _provenance(run_id, "evaluate_act", local, phases, evaluate_act=result)
    _save_record(f"{run_id}-evaluate", provenance)
    print(json.dumps({"phases": phases, "evaluate_act": result}, indent=2))
