"""Image-capable, self-hosted/vendor chat endpoints; no OpenAI service calls."""

import base64
import json
from urllib.parse import urlsplit

from brickcell.integrations.explanation import SCHEMA, checked, images, prompt


def explain(case, context, provider, base_url, model, token):
    if urlsplit(base_url).hostname in ("api.openai.com", "openai.com"):
        raise ValueError("This adapter is for the explicitly selected non-OpenAI provider")
    content = [{"type": "text", "text": prompt(case)}]
    for data, mime in images(case, context.root):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode()},
            }
        )
    headers = {"Authorization": "Bearer " + token} if token else {}
    response = context.http.json(
        base_url.rstrip("/") + "/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "brickcell_visual_evidence",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        headers,
    )
    choices = response.get("choices", [])
    if not choices or choices[0].get("finish_reason") != "stop":
        raise ValueError("Vision endpoint response was incomplete")
    return checked(json.loads(choices[0]["message"]["content"]), case, provider, model)
