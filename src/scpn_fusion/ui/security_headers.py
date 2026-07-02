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
from typing import Any, cast

HeaderItems = tuple[tuple[str, str], ...]

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


def _normalise_header_items(headers: Mapping[str, str]) -> HeaderItems:
    """Validate and freeze a security-header mapping.

    Parameters
    ----------
    headers : Mapping[str, str]
        Header names and values that should be added to each Tornado response.

    Returns
    -------
    tuple of tuple of str
        Immutable header name/value pairs captured at install time.

    Raises
    ------
    ValueError
        If the mapping is empty or contains blank names or values.
    """
    items = tuple(headers.items())
    if not items:
        raise ValueError("security headers must not be empty.")
    for name, value in items:
        if not name.strip():
            raise ValueError("security header names must be non-empty.")
        if not value.strip():
            raise ValueError(f"security header {name!r} must have a non-empty value.")
    return items


def _header_install_wrapper(
    original: Callable[[Any], None],
    headers: Mapping[str, str],
) -> Callable[[Any], None]:
    """Wrap Tornado's default-header hook with a frozen security policy."""
    header_items = _normalise_header_items(headers)

    def _set_default_headers(self: Any) -> None:
        original(self)
        for name, value in header_items:
            self.set_header(name, value)

    wrapped = cast(Any, _set_default_headers)
    wrapped._scpn_security_headers = True
    return _set_default_headers


def install_tornado_security_headers(headers: Mapping[str, str] = SECURITY_HEADERS) -> bool:
    """Patch Tornado request handlers so Streamlit responses carry security headers.

    Parameters
    ----------
    headers : Mapping[str, str]
        Non-empty security header policy installed on Tornado responses.

    Returns
    -------
    bool
        ``True`` when Tornado is present and the hook is installed or already
        installed. ``False`` when Tornado is not importable.

    Raises
    ------
    ValueError
        If ``headers`` is empty or contains blank names or values.
    """
    try:
        import tornado.web
    except ImportError:
        return False

    current = tornado.web.RequestHandler.set_default_headers
    if getattr(current, "_scpn_security_headers", False):
        return True
    request_handler = cast(Any, tornado.web.RequestHandler)
    request_handler.set_default_headers = _header_install_wrapper(current, headers)
    return True


__all__ = ["SECURITY_HEADERS", "install_tornado_security_headers"]
