# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Streamlit Security Headers Tests
"""Tests for dashboard browser security header installation."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any, cast

import pytest

from scpn_fusion.ui import security_headers


class _HeaderRecorder:
    """Minimal Tornado-like handler that records header writes."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}

    def set_header(self, name: str, value: str) -> None:
        """Record a header assignment."""
        self.headers[name] = value


class _FakeRequestHandler(_HeaderRecorder):
    """Tornado RequestHandler stand-in for installer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def set_default_headers(self) -> None:
        """Record that the original Tornado hook still ran."""
        self.calls.append("original")
        self.set_header("Existing", "kept")


def test_header_wrapper_preserves_existing_headers_and_adds_csp() -> None:
    """Wrap the original hook and append the default security policy."""
    calls: list[str] = []

    def _original(handler: _HeaderRecorder) -> None:
        calls.append("original")
        handler.set_header("Existing", "kept")

    wrapped = security_headers._header_install_wrapper(
        _original,
        security_headers.SECURITY_HEADERS,
    )
    handler = _HeaderRecorder()

    wrapped(handler)

    assert calls == ["original"]
    assert handler.headers["Existing"] == "kept"
    assert "default-src 'self'" in handler.headers["Content-Security-Policy"]
    assert handler.headers["X-Frame-Options"] == "DENY"
    assert handler.headers["X-Content-Type-Options"] == "nosniff"


def test_header_wrapper_snapshots_custom_headers() -> None:
    """Freeze custom header mappings at install time."""
    mutable_headers = {"X-Frame-Options": "DENY"}

    wrapped = security_headers._header_install_wrapper(
        lambda _handler: None,
        mutable_headers,
    )
    mutable_headers["X-Frame-Options"] = "SAMEORIGIN"
    mutable_headers["X-Late"] = "mutated"
    handler = _HeaderRecorder()

    wrapped(handler)

    assert handler.headers == {"X-Frame-Options": "DENY"}


def test_header_wrapper_rejects_empty_headers() -> None:
    """Reject no-op security configurations."""
    with pytest.raises(ValueError, match="security headers must not be empty"):
        security_headers._header_install_wrapper(lambda _handler: None, {})


def test_header_wrapper_rejects_blank_header_name() -> None:
    """Reject invalid header names before patching Tornado."""
    with pytest.raises(ValueError, match="security header names must be non-empty"):
        security_headers._header_install_wrapper(lambda _handler: None, {" ": "DENY"})


def test_header_wrapper_rejects_blank_header_value() -> None:
    """Reject no-op header values before patching Tornado."""
    with pytest.raises(ValueError, match="security header 'X-Test' must have a non-empty value"):
        security_headers._header_install_wrapper(lambda _handler: None, {"X-Test": " "})


def test_install_tornado_security_headers_returns_false_without_tornado(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return ``False`` when Tornado is not importable."""
    monkeypatch.setitem(sys.modules, "tornado.web", None)

    assert security_headers.install_tornado_security_headers() is False


def test_install_tornado_security_headers_patches_fake_tornado_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch a Tornado-like RequestHandler and leave the wrapper idempotent."""
    fake_tornado = ModuleType("tornado")
    fake_web = ModuleType("tornado.web")
    cast(Any, fake_web).RequestHandler = _FakeRequestHandler
    cast(Any, fake_tornado).web = fake_web
    monkeypatch.setitem(sys.modules, "tornado", fake_tornado)
    monkeypatch.setitem(sys.modules, "tornado.web", fake_web)

    assert security_headers.install_tornado_security_headers({"X-Test": "enabled"}) is True
    first_wrapper = cast(
        Callable[[_FakeRequestHandler], None],
        _FakeRequestHandler.set_default_headers,
    )
    assert getattr(first_wrapper, "_scpn_security_headers", False) is True

    handler = _FakeRequestHandler()
    first_wrapper(handler)

    assert handler.calls == ["original"]
    assert handler.headers["Existing"] == "kept"
    assert handler.headers["X-Test"] == "enabled"

    assert security_headers.install_tornado_security_headers({"X-Test": "new"}) is True
    assert _FakeRequestHandler.set_default_headers is first_wrapper
