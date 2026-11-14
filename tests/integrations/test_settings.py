import sys

from brickcell.integrations.__main__ import main


def test_setup_diagnostics_never_print_credentials(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "NeverPrintThisValue")
    monkeypatch.setenv("GEMINI_MODEL", "")
    monkeypatch.setattr(sys, "argv", ["integrations", "--status", "--providers", "gemini"])
    main()
    output = capsys.readouterr().out
    assert "NeverPrintThisValue" not in output
    assert "GEMINI_MODEL" in output
    assert "Google Gemini API" in output
    assert '"network": false' in output
