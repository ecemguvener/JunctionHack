"""Apply a StationState to a fresh MuJoCo World and render the sim cameras (MuJoCo environment).

  python -m brickcell.twin.apply --state state.json --out DIR [--real-fixture DIR]
                                 [--real-image overhead=frame.png ...]

Writes into DIR:
  report.json                   the apply_state report
  sim-<camera>.png              sim render of each camera (default: overhead and side)
  blend-<camera>.png            50/50 blend of sim and real (only with a real image)
  side-by-side-<camera>.png     sim left, real right (only with a real image)

This is the visual check of the whole twin: the sim bricks must lie on the real bricks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from brickcell.simulation.world import World
from brickcell.twin.sim_sync import apply_state, load_sim_config
from brickcell.twin.state import StationState, load_fixture


def compare_images(sim: np.ndarray, real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(50/50 blend, side by side). The real image is resized to the sim size if they differ."""
    if real.shape[:2] != sim.shape[:2]:
        real = np.asarray(Image.fromarray(real).resize((sim.shape[1], sim.shape[0])))
    blend = ((sim.astype(np.uint16) + real.astype(np.uint16)) // 2).astype(np.uint8)
    return blend, np.concatenate([sim, real], axis=1)


def real_images(fixture: Path | None, images: list[str]) -> dict[str, np.ndarray]:
    result = {}
    if fixture is not None:
        result.update({frame.camera: frame.rgb for frame in load_fixture(fixture)})
    for entry in images:
        camera, _, path = entry.partition("=")
        if not path:
            raise ValueError(f"--real-image needs CAMERA=PATH, not {entry!r}")
        result[camera.strip()] = np.asarray(Image.open(path.strip()).convert("RGB"))
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--state", type=Path, required=True, help="StationState JSON file")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--real-fixture", type=Path, help="station fixture (meta.json + <camera>.npz)"
    )
    parser.add_argument(
        "--real-image", action="append", default=[], metavar="CAMERA=PATH", help="a real RGB image"
    )
    parser.add_argument(
        "--camera", action="append", help="sim camera to render (default: overhead and side)"
    )
    parser.add_argument("--twin-config", type=Path, help="default: configs/twin.yaml")
    parser.add_argument("--scene-config", type=Path, help="default: configs/sim-scene.yaml")
    parser.add_argument("--settle-s", type=float, help="override sim.settle_s (0 = no settle)")
    parser.add_argument("--relax-s", type=float, help="override sim.relax_s (0 = no relax)")
    args = parser.parse_args(argv)

    overrides = {
        k: v for k, v in (("settle_s", args.settle_s), ("relax_s", args.relax_s)) if v is not None
    }
    config = load_sim_config(args.twin_config, overrides)
    state = StationState.from_json(args.state.read_text())
    real = real_images(args.real_fixture, args.real_image)
    args.out.mkdir(parents=True, exist_ok=True)

    world = World(args.out / "world", args.scene_config)
    try:
        report = apply_state(world, state, config)
        cameras = args.camera or ["overhead", "side"]
        written = []
        for camera in cameras:
            sim = world.camera(camera)
            Image.fromarray(sim).save(args.out / f"sim-{camera}.png")
            written.append(f"sim-{camera}.png")
            if camera in real:
                blend, pair = compare_images(sim, real[camera])
                Image.fromarray(blend).save(args.out / f"blend-{camera}.png")
                Image.fromarray(pair).save(args.out / f"side-by-side-{camera}.png")
                written += [f"blend-{camera}.png", f"side-by-side-{camera}.png"]
    finally:
        world.close()
    report["images"] = written
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": str(args.out), "images": written, "flags": report["flags"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
