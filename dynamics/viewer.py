"""Make a self-contained recorded-vs-simulated CAD phantom page; no server."""

import json
from pathlib import Path

from model import ARM_JOINTS, CalibratedSO101Kinematics, display_geometry, load
from replay import prepare, rollout, scores


def build_viewer(model_dir, episode_path, report_path, output):
    episode = json.loads(Path(episode_path).read_text())
    report = json.loads(Path(report_path).read_text())
    model = load(model_dir)
    geometric = CalibratedSO101Kinematics("follower")
    times, observed, ct, cq = prepare(episode, dt=0.04)
    parameters = [("Before fitting", report["parameters_before"])]
    history = report["history"]
    if history:
        for fraction in (0.25, 0.5, 0.75):
            row = history[min(int(fraction * len(history)), len(history) - 1)]
            parameters.append(
                (f"Fit progress · {row['fit_rmse_degrees']:.2f}° training RMSE", row["parameters"])
            )
    parameters.append(("Fitted candidate", report["parameters_after"]))
    candidates = []
    for label, params in parameters:
        prediction = rollout(model, times, observed[0], ct, cq, params)
        candidates.append(
            {
                "label": label,
                "q": prediction.tolist(),
                "parameters": params,
                "rmse": scores(observed, prediction)["mean_rmse_degrees"],
            }
        )
    inertial_inputs = json.loads((Path(model_dir) / "inertial_priors.json").read_text())
    payload = {
        "geometry": display_geometry(),
        "jointNames": list(ARM_JOINTS),
        "offsets": geometric.cad_at_zero.tolist(),
        "signs": geometric.cad_sign.tolist(),
        "t": times.tolist(),
        "observed": observed.tolist(),
        "candidates": candidates,
        "synthetic": episode["synthetic"],
        "partition": episode["partition"],
        "masses": inertial_inputs["links"],
        "massStatus": inertial_inputs["status"],
    }
    html = (Path(__file__).parent / "phantom.html").read_text()
    html = html.replace(
        "/*PAYLOAD*/", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    )
    Path(output).write_text(html)
    return output
