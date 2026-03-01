from __future__ import annotations

from pathlib import Path

from scpn_fusion.ui import dashboard_launcher


def test_main_invokes_streamlit_with_app_path(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_call(cmd):  # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        return 7

    monkeypatch.setattr(dashboard_launcher.subprocess, "call", _fake_call)
    monkeypatch.setattr(dashboard_launcher.sys, "executable", "python-test")

    rc = dashboard_launcher.main()
    assert rc == 7
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:4] == ["python-test", "-m", "streamlit", "run"]
    assert Path(cmd[4]).name == "app.py"
