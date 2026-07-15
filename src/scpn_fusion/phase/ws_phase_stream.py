# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — WebSocket Phase Sync Stream
"""
Async WebSocket server streaming RealtimeMonitor tick snapshots.

Start standalone::

    python -m scpn_fusion.phase.ws_phase_stream --port 8765

Or embed in an existing asyncio loop.  Non-loopback bindings require a
``SCPN_PHASE_STREAM_TOKEN`` value and may be served with ``--tls-cert`` and
``--tls-key`` for WSS::

    server = PhaseStreamServer(monitor)
    await server.serve(host="127.0.0.1", port=8765)

Clients receive JSON frames every tick::

    {"tick": 1, "R_global": 0.42, "V_global": 0.83, "lambda_exp": -0.12, ...}
"""

from __future__ import annotations

import asyncio
import hmac
import importlib
import json
import logging
import math
import os
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scpn_fusion.phase.realtime_monitor import RealtimeMonitor

logger = logging.getLogger(__name__)

#: Forward-secret AEAD cipher allowlist for the TLS 1.2 handshake (TLS 1.3 suites
#: are negotiated separately by OpenSSL and are always available).  Restricting
#: the 1.2 suites to ECDHE + GCM/ChaCha20 removes static-RSA and CBC options.
_TLS_CIPHER_SUITES = ":".join(
    (
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-ECDSA-CHACHA20-POLY1305",
        "ECDHE-RSA-CHACHA20-POLY1305",
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        "ECDHE-RSA-AES128-GCM-SHA256",
    )
)


def _constant_time_eq(candidate: str, expected: str) -> bool:
    """Compare two tokens in constant time to deny a timing side-channel."""
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


def _bearer_token_from_headers(websocket: Any) -> str | None:
    headers = getattr(websocket, "request_headers", None)
    if headers is None:
        return None
    auth = headers.get("Authorization") if hasattr(headers, "get") else None
    if not isinstance(auth, str):
        return None
    prefix = "Bearer "
    if not auth.startswith(prefix):
        return None
    token = auth[len(prefix) :].strip()
    return token or None


@dataclass
class PhaseStreamServer:
    """Async WebSocket server wrapping a RealtimeMonitor."""

    monitor: RealtimeMonitor
    tick_interval_s: float = 0.001
    auth_token: str | None = None
    max_command_messages_per_second: int = 20
    command_value_bound: float = 1.0e3
    _clients: set[Any] = field(default_factory=set, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the auth token from the environment when not provided."""
        if self.auth_token is None:
            token = os.environ.get("SCPN_PHASE_STREAM_TOKEN")
            self.auth_token = token if token else None
        if self.max_command_messages_per_second < 1:
            raise ValueError("max_command_messages_per_second must be >= 1.")
        if self.command_value_bound <= 0.0 or not math.isfinite(self.command_value_bound):
            raise ValueError("command_value_bound must be a positive, finite magnitude.")

    def _header_authorized(self, websocket: Any) -> bool:
        if self.auth_token is None:
            return True
        token = _bearer_token_from_headers(websocket)
        if token is None:
            return False
        return _constant_time_eq(token, self.auth_token)

    def _message_authorized(self, payload: dict[str, Any]) -> bool:
        if self.auth_token is None:
            return True
        if payload.get("action") != "auth":
            return False
        token = payload.get("token")
        if not isinstance(token, str):
            return False
        return _constant_time_eq(token, self.auth_token)

    def _coerce_command_value(self, cmd: dict[str, Any]) -> float | None:
        """Return the finite, in-range ``value`` of a numeric command, or ``None``.

        Rejects a missing key, a non-numeric (including ``bool``) value, a
        non-finite (``NaN``/``inf``) value, or a magnitude beyond
        ``command_value_bound``.  A rejected command is logged and ignored,
        matching the malformed-payload contract, so a hostile client cannot drive
        the monitor into unbounded or non-finite state.
        """
        action = cmd.get("action")
        if "value" not in cmd:
            logger.warning("Ignoring %s command with no value", action)
            return None
        raw = cmd["value"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            logger.warning("Ignoring %s command with non-numeric value", action)
            return None
        value = float(raw)
        if not math.isfinite(value) or abs(value) > self.command_value_bound:
            logger.warning("Ignoring %s command with out-of-range value", action)
            return None
        return value

    async def _close_unauthorized(self, websocket: Any) -> None:
        close = getattr(websocket, "close", None)
        if close is not None:
            await close(code=1008, reason="unauthorized")

    async def _handler(self, websocket: Any) -> None:
        authorized = self._header_authorized(websocket)
        command_times: list[float] = []
        self._clients.add(websocket)
        logger.info("Client connected (%d total)", len(self._clients))
        try:
            async for msg in websocket:
                now = time.monotonic()
                command_times = [stamp for stamp in command_times if now - stamp < 1.0]
                if len(command_times) >= self.max_command_messages_per_second:
                    logger.warning("Closing phase-stream client after command-rate limit breach")
                    await self._close_unauthorized(websocket)
                    break
                command_times.append(now)
                try:
                    cmd = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if not isinstance(cmd, dict):
                    continue
                if not authorized:
                    authorized = self._message_authorized(cmd)
                    if not authorized:
                        logger.warning("Closing unauthorized phase-stream client")
                        await self._close_unauthorized(websocket)
                        break
                    continue
                if cmd.get("action") == "set_psi":
                    value = self._coerce_command_value(cmd)
                    if value is not None:
                        self.monitor.psi_driver = value
                elif cmd.get("action") == "set_pac_gamma":
                    value = self._coerce_command_value(cmd)
                    if value is not None:
                        self.monitor.pac_gamma = value
                elif cmd.get("action") == "reset":
                    self.monitor.reset(seed=cmd.get("seed", 42))
                elif cmd.get("action") == "stop":
                    self._running = False
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected (%d remain)", len(self._clients))

    async def _tick_loop(self) -> None:
        self._running = True
        while self._running:
            if not self._clients:
                await asyncio.sleep(0.05)
                continue
            snap = self.monitor.tick()
            frame = json.dumps(snap)
            dead = set()
            for ws in self._clients:
                try:
                    await ws.send(frame)
                except (ConnectionError, OSError):
                    dead.add(ws)
            self._clients -= dead
            await asyncio.sleep(self.tick_interval_s)

    async def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        """Start WebSocket server and tick loop."""
        if not _is_loopback_host(host) and self.auth_token is None:
            raise ValueError(
                "Exposed phase-stream WebSocket bindings require SCPN_PHASE_STREAM_TOKEN."
            )
        try:
            websockets: Any = importlib.import_module("websockets")
        except ImportError as exc:
            raise ImportError("pip install websockets") from exc

        tick_task = asyncio.create_task(self._tick_loop())
        async with websockets.serve(self._handler, host, port, ssl=ssl_context):
            scheme = "wss" if ssl_context is not None else "ws"
            logger.info("Phase stream listening on %s://%s:%d", scheme, host, port)
            await tick_task

    def serve_sync(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        """Blocking entry point."""
        asyncio.run(self.serve(host, port, ssl_context=ssl_context))


def _server_tls_context(certfile: str | None, keyfile: str | None) -> ssl.SSLContext | None:
    if certfile is None and keyfile is None:
        return None
    if not certfile or not keyfile:
        raise ValueError("Both --tls-cert and --tls-key are required for WSS.")
    cert_path = Path(certfile)
    key_path = Path(keyfile)
    if not cert_path.is_file() or not key_path.is_file():
        raise ValueError("TLS certificate and key files must exist.")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.set_ciphers(_TLS_CIPHER_SUITES)
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    return context


def main() -> None:
    """Run the async WebSocket phase-stream server from CLI.

    The server publishes one JSON snapshot per active WebSocket client per tick.
    Use :mod:`argparse` options to control topology, discretisation, and
    coupling intensity for reproducible replay.
    """
    import argparse

    parser = argparse.ArgumentParser(description="SCPN Phase Sync WebSocket Stream")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--n-per", type=int, default=50)
    parser.add_argument("--zeta", type=float, default=0.5)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument("--tick-interval", type=float, default=0.001)
    parser.add_argument("--auth-token-env", default="SCPN_PHASE_STREAM_TOKEN")
    parser.add_argument("--max-command-rate", type=int, default=20)
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    auth_token = os.environ.get(args.auth_token_env)
    mon = RealtimeMonitor.from_paper27(
        L=args.layers,
        N_per=args.n_per,
        zeta_uniform=args.zeta,
        psi_driver=args.psi,
    )
    server = PhaseStreamServer(
        monitor=mon,
        tick_interval_s=args.tick_interval,
        auth_token=auth_token,
        max_command_messages_per_second=args.max_command_rate,
    )
    server.serve_sync(
        host=args.host,
        port=args.port,
        ssl_context=_server_tls_context(args.tls_cert, args.tls_key),
    )


if __name__ == "__main__":
    main()
