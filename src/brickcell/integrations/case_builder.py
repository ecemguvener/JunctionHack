"""Bridge existing InspectionResult JSON into a case without rerunning inspection."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from brickcell.inspection.types import Decision, InspectionResult
from brickcell.integrations.core import evidence, validate


def build(result, session_id, camera_id, captured_at, synthetic, assets, root):
    data = dict(result)
    data["decision"] = Decision(data["decision"])
    inspected = InspectionResult(**data)
    timestamp = datetime.fromisoformat(inspected.timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or abs(timestamp.timestamp() - captured_at) > 0.001:
        raise ValueError("Source result timestamp must match explicit camera acquisition time")
    case = {
        "object_id": inspected.object_id,
        "session_id": session_id,
        "frame_id": inspected.frame_id,
        "camera_id": camera_id,
        "view_id": inspected.view_id,
        "captured_at": captured_at,
        "synthetic": synthetic,
        "decision": str(inspected.decision),
        "is_valid_view": inspected.is_valid_view,
        "model_version": inspected.model_version,
        "threshold_version": inspected.threshold_version,
        "inference_latency_ms": inspected.inference_latency_ms,
        "score": inspected.anomaly_score,
        "score_type": inspected.score_type,
        "reason": inspected.reason,
        "inspector_mode": inspected.inspector_mode,
        "assets": [],
    }
    root = Path(root).resolve()
    for role, filename in assets.items():
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Evidence file must stay inside the evidence root")
        asset = {
            "role": role,
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        evidence(asset, root)
        case["assets"].append(asset)
    return validate(case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--captured-at", type=float, required=True)
    parser.add_argument("--data-kind", choices=("real", "synthetic"), required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        json.loads(args.result.read_text()),
        args.session_id,
        args.camera_id,
        args.captured_at,
        args.data_kind == "synthetic",
        {k: getattr(args, k) for k in ("reference", "observed", "overlay")},
        args.evidence_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(args.output)


if __name__ == "__main__":
    main()
