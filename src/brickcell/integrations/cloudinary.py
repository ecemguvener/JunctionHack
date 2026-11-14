"""Cloudinary: immutable, hash-addressed evidence uploads, separate from training."""

import base64
import os
from urllib.parse import quote, urlencode, urlsplit

from brickcell.integrations.core import case_id, data_url, required


def run(case, context):
    cloud = quote(required("CLOUDINARY_CLOUD_NAME"), safe="")
    token = base64.b64encode(
        f"{required('CLOUDINARY_API_KEY')}:{required('CLOUDINARY_API_SECRET')}".encode()
    ).decode()
    delivery = os.environ.get("CLOUDINARY_DELIVERY_TYPE", "authenticated")
    if delivery not in ("authenticated", "upload"):
        raise ValueError("Unsupported Cloudinary delivery type")
    uploaded = []
    if not case.get("assets"):
        raise ValueError("Cloudinary needs actual evidence assets")
    for asset in case["assets"]:
        public_id = f"brickcell/{case_id(case)}/{asset['role']}-{asset['sha256']}"
        payload = urlencode(
            {
                "file": data_url(asset, context.root),
                "public_id": public_id,
                "overwrite": "false",
                "type": delivery,
            }
        ).encode()
        import json

        result = json.loads(
            context.http.request(
                f"https://api.cloudinary.com/v1_1/{cloud}/image/upload",
                headers={
                    "Authorization": "Basic " + token,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                raw=payload,
            )
        )
        if (
            result.get("public_id") != public_id
            or urlsplit(result.get("secure_url", "")).scheme != "https"
        ):
            raise ValueError("Cloudinary returned mismatched evidence identity")
        uploaded.append(
            {
                "role": asset["role"],
                "sha256": asset["sha256"],
                "public_id": public_id,
                "url": result["secure_url"],
                "version": result.get("version"),
                "delivery_type": delivery,
                "requires_signed_delivery": delivery == "authenticated",
            }
        )
    return {"evidence": uploaded, "training_files_modified": False}
