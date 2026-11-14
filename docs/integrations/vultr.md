# Vultr — shared evidence/report hosting

Vultr can host the authenticated report backend without putting a cloud service
between cameras and local inspection. `brickcell.integrations.report_server`
serves recorded cases, per-provider delivery receipts, public Cloudinary evidence,
advisory explanations and explicit-play ElevenLabs MP3s. The UI uses DOM text
nodes for untrusted result text and never offers robot controls. Synthetic cases
are visibly marked. Authenticated Cloudinary assets require signed delivery and
are not incorrectly displayed as public images.

For local smoke/review, export `BRICKCELL_REPORT_USER` and
`BRICKCELL_REPORT_PASSWORD` in your process and run:

```bash
uv run python -m brickcell.integrations.report_server --output artifacts/integrations
```

Open http://127.0.0.1:18110; browser Basic authentication protects the page and
case/audio APIs. `/health` only reports service identity. `POST /api/cases` accepts
`{"case": <validated case>, "providers": ["cloudinary", "gemini"]}` and durably queues
work, returning 202. It makes no provider calls. Start the single worker explicitly:

```bash
uv run python -m brickcell.integrations --drain --providers cloudinary gemini \
  --evidence-root artifacts --output artifacts/integrations --execute
```

A hosted worker needs approved evidence files at the same relative paths under
its evidence root. Images are not uploaded to Vultr automatically. The gallery
shows the 100 latest case IDs, original decisions and pending/failed/done status.
Repeated successful deliveries are skipped; failed work is retried explicitly.
Use one worker per outbox and make it a private operator system.

For a Vultr VM, install the repository/locked core environment under `/opt/brickcell`,
create a dedicated `brickcell` service user and writable
`/var/lib/brickcell/integrations`, and store credentials in root-managed
`/etc/brickcell/integrations.env`. Use the example systemd unit and a Caddy HTTPS
reverse proxy from `deploy/vultr/`; replace its example hostname and configure DNS.
Only ports 80/443 should be public; port 18110 remains loopback. TLS is mandatory
for remote Basic authentication. Provisioning, OS setup, firewall changes, DNS,
HTTPS and service startup require separate explicit deployment work. This commit
does not rent, SSH into or deploy a server, and never exposes the camera/arm APIs.

Offline handler tests verify authentication, queued-only POST behavior, persisted
state and traversal rejection. No Vultr VM, public URL or rendered browser demo
has been verified. [Vultr documentation](https://docs.vultr.com/).
