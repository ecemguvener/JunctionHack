"""Verda Inference: hosted image-capable explanation, separate from local inference."""

from brickcell.integrations.core import required
from brickcell.integrations.vision_endpoint import explain


def run(case, context):
    return explain(
        case,
        context,
        "Verda Inference",
        "https://inference.verda.com/v1",
        required("VERDA_MODEL"),
        required("VERDA_INFERENCE_API_KEY"),
    )
