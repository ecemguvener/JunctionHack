"""Small offline format fixtures. No simulator, policy or real measurements implied."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from brickcell.data.synthetic import generate
from brickcell.policy.data import CAMERAS, JOINTS, file_hash
from brickcell.workbench.common import fresh_output, read_json, write_json
from brickcell.workbench.inspection import COLUMNS


def create_demo(output: Path) -> None:
    fresh_output(output)
    inspection = generate(output / "inspection")
    raw = read_json(inspection)
    for row in raw["samples"]:
        row.update(family="synthetic_square", color="orange")
        if row["label"] == "defective":
            row["defect_type"] = "drawn_marker_fixture"
    write_json(inspection, raw)
    with (inspection.parent / "capture.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in raw["samples"]:
            writer.writerow(
                {
                    **{k: row.get(k, "") for k in COLUMNS if k != "face"},
                    "face": row["view_id"],
                    "synthetic": "true",
                }
            )
    policy = output / "policy"
    policy.mkdir()
    rows = []
    for seed, split in enumerate(("train", "val", "test")):
        count = 30
        timestamp = np.arange(count) / 50
        state = np.zeros((count, 6), dtype=np.float32)
        state[:, 0] = np.sin(timestamp + seed) * 0.1
        action = state.copy()
        action[:, 0] += 0.01
        cameras = {c: [] for c in CAMERAS}
        for i in range(count):
            for camera in CAMERAS:
                image = Image.new("RGB", (128, 128), (20, 35 + seed * 10, 45))
                draw = ImageDraw.Draw(image)
                x = 20 + i * 2
                draw.rectangle((x, 45, x + 20, 65), fill="orange")
                draw.text((5, 5), f"FORMAT FIXTURE\n{camera}\nframe {i}", fill="white")
                cameras[camera].append(np.asarray(image))
        path = policy / f"episode-{seed}.npz"
        np.savez_compressed(
            path,
            timestamp=timestamp,
            state=state,
            action=action,
            **{c: np.stack(images) for c, images in cameras.items()},
        )
        rows.append(
            {
                "episode_id": f"format-fixture-{seed}",
                "path": path.name,
                "split": split,
                "seed": seed,
                "frames": count,
                "success": True,
                "sha256": file_hash(path),
            }
        )
    write_json(
        policy / "manifest.json",
        {
            "schema_version": 1,
            "synthetic": True,
            "human_demonstrations": False,
            "teacher": "synthetic_format_fixture_NOT_robot_motion",
            "fps": 50,
            "action_units": "radians",
            "joint_names": list(JOINTS),
            "action_semantics": "absolute_joint_position_targets_before_physics_step",
            "camera_names": list(CAMERAS),
            "episodes": rows,
            "description": "Offline schema/playback fixture. Success is fixture completion only, not a grasp.",
        },
    )
