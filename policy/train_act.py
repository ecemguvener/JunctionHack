"""Small CPU experiment with Physical AI Studio's actual ACT implementation.

Uses its public eager dataset_stats/Observation/optimizer APIs with a short PyTorch
loop. The local NPZ archive is intentionally not advertised as a LeRobot dataset.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch

from brickcell.policy.data import CAMERAS, file_hash, load_episodes


def batch(episodes, rng, count, chunk):
    from physicalai.data import Observation

    states, actions, masks = [], [], []
    images = {name: [] for name in CAMERAS}
    for _ in range(count):
        episode = episodes[rng.integers(len(episodes))]
        index = rng.integers(len(episode["state"]))
        indices = np.arange(index, index + chunk)
        states.append(episode["state"][index])
        actions.append(episode["action"][np.minimum(indices, len(episode["state"]) - 1)])
        masks.append(indices >= len(episode["state"]))
        for name in CAMERAS:
            images[name].append(episode[name][index].transpose(2, 0, 1))
    return Observation(
        state=torch.tensor(np.stack(states)),
        action=torch.tensor(np.stack(actions)),
        images={n: torch.tensor(np.stack(v), dtype=torch.float32) / 255 for n, v in images.items()},
        extra={"action_is_pad": torch.tensor(np.stack(masks))},
    )


def statistics(episodes):
    stats = {}
    for name, kind in [("state", "STATE"), ("action", "ACTION")]:
        values = np.concatenate([e[name] for e in episodes])
        stats[name] = dict(
            name=name,
            type=kind,
            shape=(6,),
            mean=values.mean(0).tolist(),
            std=np.maximum(values.std(0), 0.01).tolist(),
        )
    for camera in CAMERAS:
        name = f"images.{camera}"
        stats[name] = dict(
            name=name,
            type="VISUAL",
            shape=(3, 128, 128),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    return stats


def load_policy(path: Path):
    from physicalai.policies import ACT

    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = {**payload["config"], "pretrained_backbone_weights": None}
    model = ACT(**config, dataset_stats=payload["stats"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    model.reset()
    return model, payload


def train(args):
    from physicalai.policies import ACT

    if args.output.exists() or args.steps < 1:
        raise ValueError("Use a fresh output directory and positive step count")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    training = load_episodes(args.manifest, "train")
    validation = load_episodes(args.manifest, "val")
    stats = statistics(training)
    config = dict(
        chunk_size=16,
        n_action_steps=4,
        image_size=(128, 128),
        dim_model=128,
        n_heads=4,
        dim_feedforward=512,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        latent_dim=16,
        kl_weight=1.0,
        dropout=0.1,
        optimizer_lr=3e-4,
        optimizer_lr_backbone=3e-5,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    )
    model = ACT(**config, dataset_stats=stats)
    optimizer = model.configure_optimizers()["optimizer"]
    args.output.mkdir(parents=True)
    history, started = [], time.monotonic()
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(batch(training, rng, args.batch_size, config["chunk_size"]))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        history.append(float(loss.detach()))
        if step % 10 == 0 or step == args.steps - 1:
            print(f"ACT CPU step {step + 1}/{args.steps}: loss={history[-1]:.5f}", flush=True)
    model.eval()
    with torch.inference_mode():
        held = batch(validation, np.random.default_rng(99), args.batch_size, config["chunk_size"])
        predicted = model.predict_action_chunk(held)
        valid = ~held.extra["action_is_pad"]
        mae = (predicted - held.action).abs()[valid].mean().item()
    payload = dict(
        config=config,
        stats=stats,
        state_dict=model.state_dict(),
        manifest_sha256=file_hash(args.manifest),
        training_steps=args.steps,
        synthetic=True,
        policy="physicalai_ACT",
        action_units="radians",
    )
    checkpoint = args.output / "policy.pt"
    torch.save(payload, checkpoint)
    restored, _ = load_policy(checkpoint)
    with torch.inference_mode():
        reload_error = (restored.predict_action_chunk(held) - predicted).abs().max().item()
    report = dict(
        policy="Physical AI Studio ACT",
        synthetic_demonstrations=True,
        steps=args.steps,
        device="CPU",
        training_seconds=time.monotonic() - started,
        first_loss=history[0],
        last_loss=history[-1],
        loss_history=history,
        validation_action_mae_radians=mae,
        validation_sample_scope="one seeded batch; not task success",
        checkpoint_reload_max_error=reload_error,
        checkpoint_sha256=file_hash(checkpoint),
        closed_loop_task_success="not measured by training",
        versions={
            n: importlib.metadata.version(n)
            for n in ("physicalai-train", "physicalai", "torch", "torchvision", "lerobot")
        },
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "loss_history"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/policy-smoke/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/policy/act-smoke"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    train(parser.parse_args())
