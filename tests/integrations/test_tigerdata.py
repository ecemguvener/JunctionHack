import sys
from types import SimpleNamespace

from brickcell.integrations.core import Context
from brickcell.integrations.tigerdata import run


def test_tigerdata_uses_tls_parameters_and_idempotent_acquisition_insert(
    image_case, tmp_path, monkeypatch
):
    monkeypatch.setenv("TIGERDATA_URL", "postgresql://test.invalid/test")
    calls = []

    class DB:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def cursor(self):
            return self

        def execute(self, sql, values):
            calls.append((sql, values))

    def connect(url, **kwargs):
        assert kwargs["sslmode"] == "verify-full"
        assert kwargs["connect_timeout"] == 5
        return DB()

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))
    run(image_case, Context(tmp_path, tmp_path, None))
    sql, values = calls[0]
    assert "ON CONFLICT" in sql
    assert "%s" in sql and image_case["object_id"] not in sql
    assert values[1].timestamp() == image_case["captured_at"]
    assert values[8] is None
