import json
import sys
from types import SimpleNamespace

import pytest

from brickcell.integrations.aiven import run
from brickcell.integrations.core import Context


@pytest.mark.parametrize("ack", [True, False])
def test_kafka_requires_delivery_ack_and_preserves_identity(image_case, tmp_path, monkeypatch, ack):
    for key in ("BOOTSTRAP", "USER", "PASSWORD", "TOPIC"):
        monkeypatch.setenv("AIVEN_KAFKA_" + key, "test")

    class Producer:
        def __init__(self, config):
            assert config["security.protocol"] == "SASL_SSL"
            assert config["enable.idempotence"] is True

        def produce(self, topic, key, value, on_delivery):
            data = json.loads(value)
            assert data["type"] == "inspection.needs_review"
            assert data["inspection"]["decision"] == "REINSPECT"
            if ack:
                on_delivery(None, SimpleNamespace(partition=lambda: 0, offset=lambda: 42))

        def flush(self, timeout):
            return 0 if ack else 1

    monkeypatch.setitem(sys.modules, "confluent_kafka", SimpleNamespace(Producer=Producer))
    if ack:
        assert run(image_case, Context(tmp_path, tmp_path, None))["offset"] == 42
    else:
        with pytest.raises(RuntimeError, match="acknowledged"):
            run(image_case, Context(tmp_path, tmp_path, None))
