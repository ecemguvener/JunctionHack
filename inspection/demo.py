"""Dependency-free anomaly inspection lesson. Does not connect to hardware."""
import argparse
import json
import math
import random
from pathlib import Path


def decide(score, threshold):
    if not all(math.isfinite(v) for v in (score, threshold)):
        raise ValueError("Score and threshold must be finite")
    if score < 0 or threshold <= 0:
        raise ValueError("Score must be nonnegative and threshold positive")
    return "reject" if score >= threshold else "accept"


def sample(rng, marked=False):
    # Fixed camera, fixed part position, 32 x 32 grayscale pixels (0..255).
    pixels = [180 + rng.randint(-3, 3) for _ in range(32 * 32)]
    if marked:
        for y in range(12, 20):
            for x in range(14, 18):
                pixels[y * 32 + x] = 35 + rng.randint(-3, 3)
    return pixels


def score_image(pixels, normal):
    differences = [abs(a - b) for a, b in zip(pixels, normal)]
    # Average the 1% most different pixels: a small mark is not diluted away.
    count = max(1, len(differences) // 100)
    return sum(sorted(differences, reverse=True)[:count]) / count, differences


def svg_image(pixels, heatmap=False):
    cells = []
    for i, value in enumerate(pixels):
        value = max(0, min(255, round(value)))
        color = "rgb(%d,0,0)" % value if heatmap else "rgb(%d,%d,%d)" % (value, value, value)
        cells.append('<rect x="%d" y="%d" width="1" height="1" fill="%s"/>' % (i % 32, i // 32, color))
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="256" height="256">' + "".join(cells) + '</svg>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("inspection/output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    train = [sample(rng) for _ in range(20)]
    normal = [sum(values) / len(values) for values in zip(*train)]
    # Calibration uses separate clean samples, never the final test images.
    calibration = [score_image(sample(rng), normal)[0] for _ in range(10)]
    threshold = max(calibration) + 5
    model = {"method": "aligned grayscale mean baseline", "width": 32,
             "normal_pixels": normal, "threshold": threshold,
             "training_images": 20, "calibration_images": 10,
             "synthetic": True}
    (args.output / "model.json").write_text(json.dumps(model, indent=2))
    records = []
    for name, marked in (("clean", False), ("marked", True)):
        pixels = sample(rng, marked)
        score, differences = score_image(pixels, normal)
        decision = decide(score, threshold)
        record = {"image": name, "score": round(score, 3),
                  "threshold": round(threshold, 3), "decision": decision,
                  "planned_destination": "defective bowl" if decision == "reject" else "good bowl",
                  "motion_executed": False, "synthetic": True}
        records.append(record)
        (args.output / (name + ".svg")).write_text(svg_image(pixels))
        (args.output / (name + "_heatmap.svg")).write_text(svg_image(differences, True))
        print(json.dumps(record))
    (args.output / "results.json").write_text(json.dumps(records, indent=2))
    sections = []
    for r in records:
        name = r["image"]
        sections.append('<h2>%s: %s</h2><p>Score %.3f / threshold %.3f</p><img alt="Input" src="%s.svg"><img alt="Difference heatmap" src="%s_heatmap.svg">' % (name, r["decision"], r["score"], threshold, name, name))
    (args.output / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Inspection lesson</title><style>body{font:18px system-ui;max-width:850px;margin:40px auto}img{margin:12px;image-rendering:pixelated}</style><h1>Learn normal → detect a mark → decide</h1><p>Synthetic images; grayscale baseline; no arm movement. Red pixels show deviation from learned normal appearance. Scores are pixel differences, not probabilities.</p>' + ''.join(sections))
    print("Report:", args.output / "report.html")


if __name__ == "__main__":
    main()
