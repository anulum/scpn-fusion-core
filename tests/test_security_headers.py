# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Streamlit Security Headers Tests
"""Tests for dashboard browser security header installation."""

from __future__ import annotations

from scpn_fusion.ui import security_headers


def test_header_wrapper_preserves_existing_headers_and_adds_csp() -> None:
    calls: list[str] = []

    class _Handler:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def set_header(self, name: str, value: str) -> None:
            self.headers[name] = value

    def _original(handler: _Handler) -> None:
        calls.append("original")
        handler.set_header("Existing", "kept")

    wrapped = security_headers._header_install_wrapper(
        _original,
        security_headers.SECURITY_HEADERS,
    )
    handler = _Handler()

    wrapped(handler)

    assert calls == ["original"]
    assert handler.headers["Existing"] == "kept"
    assert "default-src 'self'" in handler.headers["Content-Security-Policy"]
    assert handler.headers["X-Frame-Options"] == "DENY"
    assert handler.headers["X-Content-Type-Options"] == "nosniff"
