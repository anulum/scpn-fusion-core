# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Streamlit App Tests
"""Tests for the Streamlit dashboard app entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1",
    reason="Streamlit UI tests require the optional ui dependency group.",
)
AppTest = cast(Any, streamlit_testing).AppTest

from scpn_fusion.ui import app as dashboard_app  # noqa: E402


def test_resolve_config_path_uses_packaged_iter_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default config path resolves through the package-data adapter."""
    expected = Path("/tmp/scpn-ui/iter_config.json")
    monkeypatch.setattr(dashboard_app, "default_iter_config_path", lambda: expected)

    assert dashboard_app._resolve_config_path() == str(expected)


def test_resolve_config_path_prefers_validation_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom config names resolve from the active validation data root first."""
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    config_path = validation_dir / "custom.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(dashboard_app, "data_root", lambda: tmp_path)

    assert dashboard_app._resolve_config_path("custom.json") == str(config_path)


def test_resolve_config_path_falls_back_to_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing custom configs fall back to the current working directory."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setattr(dashboard_app, "data_root", lambda: data_root)
    monkeypatch.chdir(cwd)

    assert dashboard_app._resolve_config_path("local.json") == str(cwd / "local.json")


def test_streamlit_app_renders_default_dashboard_shell() -> None:
    """The real Streamlit script renders its default five-tab dashboard shell."""
    app_path = Path(dashboard_app.__file__).resolve()

    rendered = AppTest.from_file(str(app_path)).run(timeout=15)

    assert not rendered.exception
    assert [title.value for title in rendered.title] == [dashboard_app.APP_TITLE]
    assert dashboard_app.APP_SUBTITLE in rendered.markdown[0].value
    assert [tab.label for tab in rendered.tabs] == [
        "Plasma Physics",
        "Ignition & Q",
        "Nuclear Engineering",
        "Power Plant",
        "Shot Replay",
    ]
    assert "Reactor Parameters" in [header.value for header in rendered.header]
