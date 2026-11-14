"""Credential presence diagnostics only; values and live account status stay private."""

import os

SETTINGS = {
    "cloudinary": (
        "Cloudinary",
        ["CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"],
    ),
    "gemini": ("Google Gemini API", ["GEMINI_API_KEY", "GEMINI_MODEL"]),
    "elevenlabs": ("ElevenLabs", ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]),
    "tigerdata": ("Tiger Data", ["TIGERDATA_URL"]),
    "aiven": (
        "Aiven Kafka",
        ["AIVEN_KAFKA_BOOTSTRAP", "AIVEN_KAFKA_USER", "AIVEN_KAFKA_PASSWORD", "AIVEN_KAFKA_TOPIC"],
    ),
    "mongodb": ("MongoDB Atlas", ["MONGODB_URI"]),
    "gemma": ("Google Gemma (self-hosted)", ["GEMMA_BASE_URL", "GEMMA_MODEL"]),
    "verda": ("Verda Inference", ["VERDA_INFERENCE_API_KEY", "VERDA_MODEL"]),
    "oxylabs": ("Oxylabs Web API", ["OXYLABS_WEB_API_KEY", "REFERENCE_ALLOWED_HOSTS"]),
    "contextdev": ("Context.dev", ["CONTEXT_DEV_API_KEY", "REFERENCE_ALLOWED_HOSTS"]),
}


def status(providers):
    result = []
    for key in providers:
        product, names = SETTINGS[key]
        missing = [name for name in names if not os.environ.get(name, "").strip()]
        result.append(
            {
                "provider": key,
                "product": product,
                "configuration_present": not missing,
                "missing_variables": missing,
                "live_connection_tested": False,
            }
        )
    return result
