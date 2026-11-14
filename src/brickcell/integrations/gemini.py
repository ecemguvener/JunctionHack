"""Google Gemini API: structured visual explanations that cannot classify or sort."""

import base64
import json
from urllib.parse import quote

from brickcell.integrations.core import required
from brickcell.integrations.explanation import SCHEMA, checked, images, prompt


def run(case, context):
    model = required("GEMINI_MODEL")
    parts = [{"text": prompt(case)}] + [
        {"inlineData": {"mimeType": mime, "data": base64.b64encode(content).decode()}}
        for content, mime in images(case, context.root)
    ]
    response = context.http.json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent",
        {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": SCHEMA,
                "temperature": 0.1,
                "maxOutputTokens": 2048,
            },
        },
        {"x-goog-api-key": required("GEMINI_API_KEY")},
    )
    candidates = response.get("candidates", [])
    if not candidates or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Gemini response blocked or incomplete")
    text = "".join(
        p.get("text", "") for p in candidates[0]["content"]["parts"] if not p.get("thought")
    )
    return checked(json.loads(text), case, "Google Gemini API", model)
