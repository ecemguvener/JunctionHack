import base64
import io
import json

from brickcell.integrations.report_server import handler, records


def test_hosted_report_auth_queues_but_never_executes(image_case, tmp_path):
    server = handler(tmp_path, "operator", "test-password")
    request = object.__new__(server)
    responses = []
    request.send = lambda code, body, kind="application/json": responses.append((code, body))
    request.path = "/api/cases"
    request.headers = {}
    request.do_GET()
    assert responses[-1][0] == 401
    body = json.dumps({"case": image_case, "providers": ["gemini"]}).encode()
    request.headers = {
        "Authorization": "Basic " + base64.b64encode(b"operator:test-password").decode(),
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    request.rfile = io.BytesIO(body)
    request.do_POST()
    assert responses[-1][0] == 202
    assert json.loads(responses[-1][1])["provider_calls"] is False
    assert records(tmp_path / "outbox.sqlite3")[0]["deliveries"][0]["status"] == "pending"
    request.path = "/audio/../../secret.mp3"
    request.do_GET()
    assert responses[-1][0] == 404
