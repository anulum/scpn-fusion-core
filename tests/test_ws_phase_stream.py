# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — WebSocket Phase Stream Tests
"""Strict tests for the WebSocket phase-stream server."""

from __future__ import annotations

import asyncio
import json
import ssl
import sys
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, cast

import pytest
from typing_extensions import Self

from scpn_fusion.phase import ws_phase_stream as stream_mod
from scpn_fusion.phase.realtime_monitor import RealtimeMonitor
from scpn_fusion.phase.ws_phase_stream import (
    _TLS_CIPHER_SUITES,
    PhaseStreamServer,
    _bearer_token_from_headers,
    _constant_time_eq,
    _is_loopback_host,
    _server_tls_context,
)


def _make_monitor() -> RealtimeMonitor:
    """Return a small deterministic phase monitor for stream tests."""
    return RealtimeMonitor.from_paper27(L=4, N_per=10, zeta_uniform=0.5, psi_driver=0.0)


class _FakeWS:
    """Minimal async WebSocket test double."""

    def __init__(
        self,
        messages: list[str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._messages = list(messages or [])
        self._sent: list[str] = []
        self._idx = 0
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.request_headers = dict(headers or {})

    def __aiter__(self) -> Self:
        """Return this object as its own async iterator."""
        return self

    async def __anext__(self) -> str:
        """Yield queued client messages in order."""
        if self._idx >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def send(self, data: str) -> None:
        """Capture a server-sent frame."""
        self._sent.append(data)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        """Capture a close request."""
        self.closed = True
        self.close_code = code
        self.close_reason = reason


class _HeaderlessWS:
    """WebSocket double with no request-header mapping."""

    request_headers: None = None


class _DeadWS(_FakeWS):
    """WebSocket double that fails on send."""

    async def send(self, data: str) -> None:
        """Raise a connection failure for dead-client cleanup."""
        del data
        raise ConnectionError("gone")


class _FakeServeContext:
    """Async context manager returned by fake websockets.serve."""

    entered = False

    async def __aenter__(self) -> Self:
        """Record context entry."""
        _FakeServeContext.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Accept normal context exit."""
        del exc_type, exc, tb


class _FakeWebsocketsModule:
    """Fake websockets module exposing the serve factory."""

    calls: ClassVar[list[tuple[Any, str, int, ssl.SSLContext | None]]] = []

    @staticmethod
    def serve(
        handler: Any,
        host: str,
        port: int,
        *,
        ssl: ssl.SSLContext | None = None,
    ) -> _FakeServeContext:
        """Capture the websocket serve call and return an async context."""
        _FakeWebsocketsModule.calls.append((handler, host, port, ssl))
        return _FakeServeContext()


class _FakeSSLContext:
    """SSLContext stand-in that records certificate loading."""

    def __init__(self, protocol: object) -> None:
        self.protocol = protocol
        self.minimum_version: object | None = None
        self.loaded: tuple[str, str] | None = None
        self.ciphers: str | None = None

    def set_ciphers(self, ciphers: str) -> None:
        """Record the negotiated cipher allowlist."""
        self.ciphers = ciphers

    def load_cert_chain(self, certfile: str, keyfile: str) -> None:
        """Record certificate and key paths."""
        self.loaded = (certfile, keyfile)


class _FakeRealtimeMonitor:
    """RealtimeMonitor class stand-in used by the CLI test."""

    calls: ClassVar[list[tuple[int, int, float, float]]] = []

    @staticmethod
    def from_paper27(
        *,
        L: int,
        N_per: int,
        zeta_uniform: float,
        psi_driver: float,
    ) -> RealtimeMonitor:
        """Capture monitor construction arguments and return a real monitor."""
        _FakeRealtimeMonitor.calls.append((L, N_per, zeta_uniform, psi_driver))
        return _make_monitor()


class _FakePhaseStreamServer:
    """PhaseStreamServer stand-in used by the CLI test."""

    instances: ClassVar[list[_FakePhaseStreamServer]] = []

    def __init__(
        self,
        *,
        monitor: RealtimeMonitor,
        tick_interval_s: float,
        auth_token: str | None,
        max_command_messages_per_second: int,
    ) -> None:
        self.monitor = monitor
        self.tick_interval_s = tick_interval_s
        self.auth_token = auth_token
        self.max_command_messages_per_second = max_command_messages_per_second
        self.serve_args: tuple[str, int, ssl.SSLContext | None] | None = None
        _FakePhaseStreamServer.instances.append(self)

    def serve_sync(
        self,
        host: str,
        port: int,
        *,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        """Capture sync serving arguments."""
        self.serve_args = (host, port, ssl_context)


def test_loopback_and_bearer_token_helpers() -> None:
    """Loopback and bearer-token helpers handle accepted and rejected inputs."""
    assert _is_loopback_host("127.0.0.1")
    assert _is_loopback_host("::1")
    assert _is_loopback_host("localhost")
    assert not _is_loopback_host("0.0.0.0")
    assert _bearer_token_from_headers(_HeaderlessWS()) is None
    assert _bearer_token_from_headers(_FakeWS(headers={"Authorization": "Token secret"})) is None
    assert _bearer_token_from_headers(_FakeWS(headers={"Authorization": "Bearer   "})) is None
    assert (
        _bearer_token_from_headers(_FakeWS(headers={"Authorization": "Bearer secret"})) == "secret"
    )


def test_init_and_message_auth_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialization validates rate limits and resolves environment tokens."""
    monkeypatch.setenv("SCPN_PHASE_STREAM_TOKEN", "from-env")
    server = PhaseStreamServer(monitor=_make_monitor(), tick_interval_s=0.01)
    assert server.tick_interval_s == 0.01
    assert server.auth_token == "from-env"
    assert server._message_authorized({"action": "set_psi"}) is False
    monkeypatch.delenv("SCPN_PHASE_STREAM_TOKEN")
    assert (
        PhaseStreamServer(monitor=_make_monitor(), auth_token=None)._message_authorized({}) is True
    )
    with pytest.raises(ValueError, match="max_command_messages_per_second"):
        PhaseStreamServer(monitor=_make_monitor(), max_command_messages_per_second=0)


def test_handler_commands_and_ignored_payloads() -> None:
    """Handler applies valid commands while ignoring malformed payloads."""

    async def _run() -> None:
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon)
        ws = _FakeWS(
            [
                "not-json",
                json.dumps(["not", "a", "dict"]),
                json.dumps({"action": "set_psi", "value": 0.5}),
                json.dumps({"action": "set_pac_gamma", "value": 0.3}),
                json.dumps({"action": "reset", "seed": 99}),
                json.dumps({"action": "stop"}),
            ]
        )
        await server._handler(ws)
        assert mon.psi_driver == pytest.approx(0.5)
        assert mon.pac_gamma == pytest.approx(0.3)
        assert server._running is False
        assert ws not in server._clients

    asyncio.run(_run())


def test_handler_authorization_paths() -> None:
    """Handler rejects unauthorized clients and accepts header or message auth."""

    async def _run() -> None:
        mon = _make_monitor()
        rejected = _FakeWS([json.dumps({"action": "set_psi", "value": 0.5})])
        await PhaseStreamServer(monitor=mon, auth_token="secret")._handler(rejected)
        assert mon.psi_driver == pytest.approx(0.0)
        assert rejected.closed is True
        assert rejected.close_code == 1008

        header = _FakeWS(
            [json.dumps({"action": "set_psi", "value": 0.5})],
            headers={"Authorization": "Bearer secret"},
        )
        await PhaseStreamServer(monitor=mon, auth_token="secret")._handler(header)
        assert mon.psi_driver == pytest.approx(0.5)
        assert header.closed is False

        message = _FakeWS(
            [
                json.dumps({"action": "auth", "token": "secret"}),
                json.dumps({"action": "set_pac_gamma", "value": 0.4}),
            ]
        )
        await PhaseStreamServer(monitor=mon, auth_token="secret")._handler(message)
        assert mon.pac_gamma == pytest.approx(0.4)
        assert message.closed is False

    asyncio.run(_run())


def test_constant_time_token_comparison() -> None:
    """Constant-time comparison accepts equal tokens and rejects mismatches."""
    assert _constant_time_eq("secret", "secret") is True
    assert _constant_time_eq("secret", "secre") is False
    assert _constant_time_eq("secret", "Secret") is False
    # Non-ASCII tokens are UTF-8 encoded before comparison, never raising.
    assert _constant_time_eq("héslo-λ", "héslo-λ") is True
    assert _constant_time_eq("héslo-λ", "heslo-l") is False


def test_header_and_message_auth_use_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auth paths delegate to the constant-time comparison for both channels."""
    calls: list[tuple[str, str]] = []

    def _spy(candidate: str, expected: str) -> bool:
        calls.append((candidate, expected))
        return candidate == expected

    monkeypatch.setattr(stream_mod, "_constant_time_eq", _spy)
    server = PhaseStreamServer(monitor=_make_monitor(), auth_token="secret")
    assert server._header_authorized(_FakeWS(headers={"Authorization": "Bearer secret"})) is True
    assert server._header_authorized(_FakeWS(headers={})) is False  # no token, no compare
    assert server._message_authorized({"action": "auth", "token": "secret"}) is True
    assert server._message_authorized({"action": "auth", "token": 42}) is False  # non-str token
    assert server._message_authorized({"action": "set_psi"}) is False  # wrong action
    assert ("secret", "secret") in calls


def test_command_value_bound_is_validated() -> None:
    """A non-positive or non-finite command-value bound is rejected at init."""
    with pytest.raises(ValueError, match="command_value_bound"):
        PhaseStreamServer(monitor=_make_monitor(), command_value_bound=0.0)
    with pytest.raises(ValueError, match="command_value_bound"):
        PhaseStreamServer(monitor=_make_monitor(), command_value_bound=float("inf"))


def test_coerce_command_value_validation() -> None:
    """Command-value coercion accepts finite in-range numbers and rejects the rest."""
    server = PhaseStreamServer(monitor=_make_monitor(), command_value_bound=10.0)
    assert server._coerce_command_value({"action": "set_psi", "value": 0.5}) == pytest.approx(0.5)
    assert server._coerce_command_value({"action": "set_psi", "value": 3}) == pytest.approx(3.0)
    assert server._coerce_command_value({"action": "set_psi"}) is None  # missing key
    assert server._coerce_command_value({"action": "set_psi", "value": "x"}) is None  # non-numeric
    assert server._coerce_command_value({"action": "set_psi", "value": True}) is None  # bool
    assert server._coerce_command_value({"action": "set_psi", "value": float("nan")}) is None
    assert server._coerce_command_value({"action": "set_psi", "value": float("inf")}) is None
    assert (
        server._coerce_command_value({"action": "set_psi", "value": 25.0}) is None
    )  # out of range
    assert server._coerce_command_value({"action": "set_psi", "value": -25.0}) is None


def test_handler_rejects_hostile_command_values() -> None:
    """Malicious NaN/inf/out-of-range/missing values never mutate monitor state."""

    async def _run() -> None:
        mon = _make_monitor()
        mon.psi_driver = 0.25
        mon.pac_gamma = 0.75
        server = PhaseStreamServer(monitor=mon, command_value_bound=10.0)
        ws = _FakeWS(
            [
                json.dumps({"action": "set_psi", "value": float("nan")}),
                json.dumps({"action": "set_psi"}),
                json.dumps({"action": "set_pac_gamma", "value": 1e9}),
                json.dumps({"action": "set_pac_gamma", "value": "boom"}),
            ]
        )
        await server._handler(ws)
        # Every hostile command was ignored; prior state is preserved unchanged.
        assert mon.psi_driver == pytest.approx(0.25)
        assert mon.pac_gamma == pytest.approx(0.75)
        # A subsequent valid command still applies.
        ok = _FakeWS([json.dumps({"action": "set_psi", "value": 1.5})])
        await server._handler(ok)
        assert mon.psi_driver == pytest.approx(1.5)

    asyncio.run(_run())


def test_handler_closes_rate_limited_client() -> None:
    """Handler closes clients that exceed the command rate limit."""

    async def _run() -> None:
        server = PhaseStreamServer(monitor=_make_monitor(), max_command_messages_per_second=1)
        ws = _FakeWS(
            [
                json.dumps({"action": "set_psi", "value": 0.1}),
                json.dumps({"action": "set_psi", "value": 0.2}),
            ]
        )
        await server._handler(ws)
        assert ws.closed is True
        assert ws.close_code == 1008

    asyncio.run(_run())


def test_tick_loop_sends_idles_and_removes_dead_clients() -> None:
    """Tick loop sends frames, idles without clients, and removes dead sockets."""

    async def _run() -> None:
        server = PhaseStreamServer(monitor=_make_monitor(), tick_interval_s=0.001)
        live = _FakeWS()
        dead = _DeadWS()
        server._clients.update({live, dead})

        async def _stop_after_ticks() -> None:
            await asyncio.sleep(0.03)
            server._running = False

        await asyncio.gather(server._tick_loop(), _stop_after_ticks())
        assert len(live._sent) > 0
        assert "R_global" in json.loads(live._sent[0])
        assert dead not in server._clients

        idle_server = PhaseStreamServer(monitor=_make_monitor(), tick_interval_s=0.001)

        async def _stop_idle() -> None:
            await asyncio.sleep(0.06)
            idle_server._running = False

        await asyncio.gather(idle_server._tick_loop(), _stop_idle())

    asyncio.run(_run())


def test_serve_import_security_and_success_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve rejects unsafe/import-failed cases and runs the websocket context."""

    async def _run() -> None:
        exposed = PhaseStreamServer(monitor=_make_monitor(), auth_token=None)
        with pytest.raises(ValueError, match="SCPN_PHASE_STREAM_TOKEN"):
            await exposed.serve(host="0.0.0.0")

        monkeypatch.setitem(sys.modules, "websockets", None)
        with pytest.raises(ImportError, match="websockets"):
            await PhaseStreamServer(monitor=_make_monitor()).serve()
        monkeypatch.delitem(sys.modules, "websockets", raising=False)

        async def fake_tick_loop() -> None:
            server._running = False

        server = PhaseStreamServer(monitor=_make_monitor(), auth_token="secret")
        monkeypatch.setattr(server, "_tick_loop", fake_tick_loop)
        _FakeWebsocketsModule.calls.clear()
        _FakeServeContext.entered = False
        stream_mod_any = cast(Any, stream_mod)
        monkeypatch.setattr(
            stream_mod_any.importlib, "import_module", lambda name: _FakeWebsocketsModule
        )
        await server.serve(host="0.0.0.0", port=9001, ssl_context=None)
        assert _FakeServeContext.entered is True
        assert _FakeWebsocketsModule.calls[-1][1:] == ("0.0.0.0", 9001, None)

    asyncio.run(_run())


def test_serve_sync_runs_async_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synchronous serving delegates to the async serve coroutine."""
    server = PhaseStreamServer(monitor=_make_monitor())
    calls: list[tuple[str, int, ssl.SSLContext | None]] = []

    async def fake_serve(
        host: str,
        port: int,
        *,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        calls.append((host, port, ssl_context))

    monkeypatch.setattr(server, "serve", fake_serve)
    server.serve_sync(host="localhost", port=9002, ssl_context=None)
    assert calls == [("localhost", 9002, None)]


def test_tls_context_validation_and_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """TLS context helper validates pairs and loads existing cert/key paths."""
    assert _server_tls_context(None, None) is None
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("fake-cert\n", encoding="utf-8")
    key.write_text("fake-key\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Both --tls-cert and --tls-key"):
        _server_tls_context(str(cert), None)
    with pytest.raises(ValueError, match="TLS certificate and key files must exist"):
        _server_tls_context(str(cert), str(tmp_path / "missing-key.pem"))

    stream_mod_any = cast(Any, stream_mod)
    monkeypatch.setattr(stream_mod_any.ssl, "SSLContext", _FakeSSLContext)
    context = _server_tls_context(str(cert), str(key))
    context_any = cast(Any, context)
    assert context_any.minimum_version == ssl.TLSVersion.TLSv1_2
    assert context_any.ciphers == _TLS_CIPHER_SUITES
    assert "ECDHE" in _TLS_CIPHER_SUITES and "RSA-AES256-GCM" in _TLS_CIPHER_SUITES
    assert context_any.loaded == (str(cert), str(key))


def test_main_wires_cli_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI entry point wires monitor, server, auth token, and TLS context."""
    _FakeRealtimeMonitor.calls.clear()
    _FakePhaseStreamServer.instances.clear()
    argv = [
        "ws_phase_stream",
        "--host",
        "0.0.0.0",
        "--port",
        "9010",
        "--layers",
        "3",
        "--n-per",
        "4",
        "--zeta",
        "0.7",
        "--psi",
        "0.2",
        "--tick-interval",
        "0.02",
        "--auth-token-env",
        "TOKEN_ENV",
        "--max-command-rate",
        "5",
        "--tls-cert",
        "cert.pem",
        "--tls-key",
        "key.pem",
    ]
    fake_tls = cast(ssl.SSLContext, object())
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("TOKEN_ENV", "cli-token")
    monkeypatch.setattr(stream_mod, "RealtimeMonitor", _FakeRealtimeMonitor)
    monkeypatch.setattr(stream_mod, "PhaseStreamServer", _FakePhaseStreamServer)
    monkeypatch.setattr(stream_mod, "_server_tls_context", lambda cert, key: fake_tls)

    stream_mod.main()

    assert _FakeRealtimeMonitor.calls == [(3, 4, 0.7, 0.2)]
    server = _FakePhaseStreamServer.instances[-1]
    assert server.tick_interval_s == 0.02
    assert server.auth_token == "cli-token"
    assert server.max_command_messages_per_second == 5
    assert server.serve_args == ("0.0.0.0", 9010, fake_tls)
