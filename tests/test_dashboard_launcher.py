# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

from pathlib import Path

from scpn_fusion.ui import dashboard_launcher


def test_main_invokes_streamlit_with_app_path(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_call(cmd: list[str], env: dict[str, str] | None = None) -> int:
        calls.append(list(cmd))
        assert env is not None
        assert env["STREAMLIT_CONFIG_DIR"]
        return 7

    monkeypatch.setattr(dashboard_launcher.subprocess, "call", _fake_call)
    monkeypatch.setattr(dashboard_launcher.sys, "executable", "python-test")

    rc = dashboard_launcher.main()
    assert rc == 7
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:4] == ["python-test", "-m", "streamlit", "run"]
    assert Path(cmd[4]).name == "app.py"


def test_main_applies_security_config_directory(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class _TempDir:
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix

        def __enter__(self) -> str:
            return str(tmp_path)

        def __exit__(self, *exc: object) -> None:
            return None

    def _fake_call(cmd: list[str], env: dict[str, str] | None = None) -> int:
        calls.append(list(cmd))
        assert env is not None
        assert env["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] == "false"
        assert env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
        config_dir = Path(env["STREAMLIT_CONFIG_DIR"])
        config_text = (config_dir / "config.toml").read_text(encoding="utf-8")
        assert "enableCORS = true" in config_text
        assert "enableXsrfProtection = true" in config_text
        return 0

    monkeypatch.setattr(dashboard_launcher.subprocess, "call", _fake_call)
    monkeypatch.setattr(dashboard_launcher.sys, "executable", "python-test")
    monkeypatch.setattr(dashboard_launcher.tempfile, "TemporaryDirectory", _TempDir)

    assert dashboard_launcher.main() == 0
    assert len(calls) == 1
