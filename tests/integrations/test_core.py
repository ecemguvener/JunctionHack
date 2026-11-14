import json
from pathlib import Path

import pytest

from brickcell.integrations.core import Context, Outbox, case_id, validate


@pytest.fixture
def case():
    return json.loads(Path("configs/integrations/demo-case.json").read_text())


def test_case_refuses_untrusted_acceptance(case):
    case["decision"] = "GOOD"
    with pytest.raises(ValueError, match="valid view"):
        validate(case)


def test_provider_failure_is_redacted_and_retry_does_not_change_decision(case, tmp_path):
    outbox = Outbox(tmp_path / "outbox.db")
    context = Context(tmp_path, tmp_path, None)

    def bad(c, ctx):
        c["decision"] = "GOOD"
        raise RuntimeError("credential-should-not-appear")

    result = outbox.deliver(case, "example", context, bad)
    assert "credential" not in json.dumps(result)
    assert case["decision"] == "REINSPECT"
    calls = []
    for _ in range(2):
        result = outbox.deliver(
            case, "example", context, lambda *a: calls.append(1) or {"ok": True}
        )
    assert calls == [1]
    assert result["case_id"] == case_id(case)
    outbox.close()
