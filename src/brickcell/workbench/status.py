"""Truthful labels shared by the simulation panel and software tests."""

SIMULATION_STATUS = (
    "**SIMULATION · SYNTHETIC CAMERA IMAGES**\n\n"
    "Movement: **scripted IK**, not a learned policy. "
    "Detector: **Anomalib PatchCore / OpenVINO CPU**.\n\n"
    "The viewer's Connected indicator means browser-to-local-server connection, "
    "not a physical arm connection."
)


def coverage_status(captured: bool) -> str:
    return (
        f"**Coverage: {int(captured)} fixed camera view captured for this object.**\n\n"
        "Named object faces confirmed: **0/5**. Front, right, back, left and studded top "
        "are not individually verified by this demo. Bottom is excluded. "
        "A single-view GOOD decision does not certify all five faces."
    )
