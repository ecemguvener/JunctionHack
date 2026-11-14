# Cloudinary — inspection evidence delivery

Cloudinary receives the original reference/observed/overlay PNG or JPEG bytes,
verified against the case's SHA-256 before upload. Training files are read-only.
Assets use case-hash/role/content-hash public IDs and `overwrite=false`, preserving
old evidence. Receipts store vendor identity, version, URL and original hash so
the UI can show before/after evidence without pretending a transformed preview
is the original training input.

Set `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` in an
ignored environment file and export them into the sidecar process. Use:

```bash
uv run python -m brickcell.integrations --case artifacts/case.json \
  --providers cloudinary --evidence-root artifacts --execute
```

Delivery defaults to `authenticated`. Its receipt URL requires signed delivery
before a browser can load it; do not expose the API secret to create browser
signatures. For deliberately public demo evidence, explicitly export
`CLOUDINARY_DELIVERY_TYPE=upload`. That publishes evidence to the asset URL.
No account is created, no credentials are logged and no UI is silently changed.
A duplicate remote ID can return a conflict after a lost receipt; reconcile the
remote asset/hash before retrying rather than overwriting it.

Offline tests inspect the actual outbound body, byte preservation, immutable
IDs and authenticated default. Account upload and signed delivery are pending.
[Official upload API](https://cloudinary.com/documentation/image_upload_api_reference).
