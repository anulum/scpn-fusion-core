# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Streamlit Security Headers
"""Install browser security headers for the Streamlit dashboard server."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

SECURITY_HEADERS: Mapping[str, str] = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "base-uri 'self'; "
        "object-src 'none'; "
        "frame-ancestors 'none'; "
        "img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "connect-src 'self' ws: wss:; "
        "font-src 'self' data:"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


def _header_install_wrapper(
    original: Callable[[Any], None],
    headers: Mapping[str, str],
) -> Callable[[Any], None]:
    def _set_default_headers(self: Any) -> None:
        original(self)
        for name, value in headers.items():
            self.set_header(name, value)

    _set_default_headers._scpn_security_headers = True
    return _set_default_headers


def install_tornado_security_headers(headers: Mapping[str, str] = SECURITY_HEADERS) -> bool:
    """Patch Tornado request handlers so Streamlit responses carry security headers."""
    try:
        import tornado.web
    except ImportError:
        return False

    current = tornado.web.RequestHandler.set_default_headers
    if getattr(current, "_scpn_security_headers", False):
        return True
    tornado.web.RequestHandler.set_default_headers = _header_install_wrapper(current, headers)
    return True


__all__ = ["SECURITY_HEADERS", "install_tornado_security_headers"]
