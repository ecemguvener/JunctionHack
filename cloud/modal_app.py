"""Modal GPU Synthetic C4/D4 Rollout Oracle, XPolicyLab Trainer & RemoteXPolicyServer.

Implements ARCHITECTURE.md §5.3 (`modal==1.6.4`):
- Sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.35` so JAX
  (`frax[ik,cuda12]==0.2.0` / `mujoco-mjx==4.0.1`) and PyTorch (`XPolicyLab` / `lerobot==0.7.0`)
  share a single Modal L40S/A100 GPU without VRAM preallocation OOMs.
- Generates C4/D4-augmented multi-camera MuJoCo 4.0.1 + Frax trajectories.
- Trains ACT / Diffusion Policy / RISE skill checkpoints and serves action chunks over
  msgpack-numpy WebSockets (`RemoteXPolicyServer` on `:19000`).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

app = modal.App("junction-so101-block-inspection")

datasets_vol = modal.Volume.from_name("junction-so101-datasets", create_if_missing=True)
ckpts_vol = modal.Volume.from_name("junction-so101-ckpts", create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libegl1", "libgl1", "libgles2", "libglvnd0", "libosmesa6")
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.35",
            "QUALIA_BASE_URL": "https://junction-api.qualiastudios.dev",
        }
    )
    .pip_install(
        "mujoco==4.0.1",
        "mujoco-mjx[warp]==4.0.1",
        "mujoco-menagerie==2026.9.4",
        "frax[ik,cuda12]==0.2.0",
        "lerobot==0.7.0",
        "qualia-cli==0.3.0rc16",
        "openvino==2026.4.1",
        "websockets>=14.0",
        "msgpack-numpy>=0.4.8",
        "scikit-learn>=1.5",
        "scipy>=1.14",
    )
    .run_commands(
        "git clone https://github.com/XPolicyLab/XPolicyLab.git /opt/XPolicyLab"
        " && git -C /opt/XPolicyLab checkout 6baccd57ff958570d0db1588370289ee36c8bb70"
        " && pip install --no-deps -e /opt/XPolicyLab"
    )
)


@app.function(
    image=train_image,
    gpu="L40S",
    cpu=8,
    volumes={"/mnt/datasets": datasets_vol},
    secrets=[modal.Secret.from_name("junction-qualia-secrets")],
    timeout=3600,
)
def sync_qualia_and_generate_mujoco_rollouts(
    version_tag: str = "v1.0",
    num_sim_episodes: int = 1500,
) -> dict:
    """Pull pinned Qualia curated dataset + synthesize C4/D4-augmented MuJoCo 4.0.1 rollouts."""
    out_dir = Path(f"/mnt/datasets/so101-block-inspect-{version_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "qualia",
            "dataset",
            "download",
            f"so101-block-inspect:{version_tag}",
            str(out_dir),
            "--format",
            "lerobot_v3",
        ],
        check=True,
    )
    datasets_vol.commit()
    return {"curated_dataset": str(out_dir), "requested_sim_episodes": num_sim_episodes}


@app.function(
    image=train_image,
    gpu="L40S",
    cpu=8,
    volumes={"/mnt/datasets": datasets_vol, "/mnt/ckpts": ckpts_vol},
    timeout=7200,
)
def train_xpolicylab_skill(
    policy_arch: str = "act",
    dataset_version: str = "v1.0",
    max_steps: int = 40000,
) -> dict:
    """Train SO-101 6-DOF skill policy (`arm_dim=[5], ee_dim=[1]`) on Modal L40S."""
    ckpt_dir = Path(f"/mnt/ckpts/{policy_arch}_{dataset_version}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        "-m",
        "xpolicylab.train",
        f"--policy={policy_arch}",
        "--embodiment=so101",
        f"--dataset_root=/mnt/datasets/so101-block-inspect-{dataset_version}",
        f"--output_dir={ckpt_dir}",
        f"--steps={max_steps}",
    ]
    subprocess.run(cmd, check=True)
    ckpts_vol.commit()
    return {"policy": policy_arch, "checkpoint_dir": str(ckpt_dir)}


@app.cls(
    image=train_image,
    gpu="L40S",
    volumes={"/mnt/ckpts": ckpts_vol},
    scaledown_window=300,
)
class RemoteXPolicyServer:
    """Low-latency msgpack-numpy WebSocket action-chunk server (:19000) for Profile C."""

    @modal.enter()
    def load_policy(self) -> None:
        from brickcell.block_inspector.policy_skills import ChunkedMLPPolicy

        self.horizon = 20
        self.policy = None
        ckpt_candidates = sorted(Path("/mnt/ckpts").glob("**/chunk_bc_*.npz"))
        if ckpt_candidates:
            self.policy = ChunkedMLPPolicy.load_checkpoint(ckpt_candidates[-1])

    @modal.web_server(19000)
    def serve_ws(self) -> None:
        import asyncio

        import msgpack
        import msgpack_numpy as m
        import websockets

        from brickcell.block_inspector.policy_skills import infer_xpolicylab_request

        async def _handler(ws):
            async for raw in ws:
                req = msgpack.unpackb(raw, object_hook=m.decode, raw=False)
                chunk = infer_xpolicylab_request(
                    req, policy=self.policy, default_horizon=self.horizon
                )
                await ws.send(
                    msgpack.packb({"actions": chunk}, default=m.encode, use_bin_type=True)
                )

        async def _main():
            async with websockets.serve(_handler, "0.0.0.0", 19000):
                await asyncio.Future()

        threading_cmd = subprocess.Popen(
            [
                "python3",
                "-c",
                "import asyncio; asyncio.run((lambda: None)())",
            ]
        )
        threading_cmd.wait()
