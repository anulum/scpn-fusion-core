# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — WebSocket Phase Sync Stream
"""
Async WebSocket server streaming RealtimeMonitor tick snapshots.

Start standalone::

    python -m scpn_fusion.phase.ws_phase_stream --port 8765

Or embed in an existing asyncio loop::

    server = PhaseStreamServer(monitor)
    await server.serve(host="0.0.0.0", port=8765)

Clients receive JSON frames every tick::

    {"tick": 1, "R_global": 0.42, "V_global": 0.83, "lambda_exp": -0.12, ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from scpn_fusion.phase.realtime_monitor import RealtimeMonitor

logger = logging.getLogger(__name__)


@dataclass
class PhaseStreamServer:
    """Async WebSocket server wrapping a RealtimeMonitor."""

    monitor: RealtimeMonitor
    tick_interval_s: float = 0.001
    _clients: set = field(default_factory=set, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    async def _handler(self, websocket: Any) -> None:
        self._clients.add(websocket)
        logger.info("Client connected (%d total)", len(self._clients))
        try:
            async for msg in websocket:
                try:
                    cmd = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if cmd.get("action") == "set_psi":
                    self.monitor.psi_driver = float(cmd["value"])
                elif cmd.get("action") == "set_pac_gamma":
                    self.monitor.pac_gamma = float(cmd["value"])
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

    async def serve(self, host: str = "0.0.0.0", port: int = 8765) -> None:  # nosec B104
        """Start WebSocket server and tick loop."""
        try:
            import websockets  # noqa: F811
        except ImportError as exc:
            raise ImportError("pip install websockets") from exc

        tick_task = asyncio.create_task(self._tick_loop())
        async with websockets.serve(self._handler, host, port):
            logger.info("Phase stream listening on ws://%s:%d", host, port)
            await tick_task

    def serve_sync(self, host: str = "0.0.0.0", port: int = 8765) -> None:  # nosec B104
        """Blocking entry point."""
        asyncio.run(self.serve(host, port))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SCPN Phase Sync WebSocket Stream")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")  # nosec B104
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--n-per", type=int, default=50)
    parser.add_argument("--zeta", type=float, default=0.5)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument("--tick-interval", type=float, default=0.001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    mon = RealtimeMonitor.from_paper27(
        L=args.layers,
        N_per=args.n_per,
        zeta_uniform=args.zeta,
        psi_driver=args.psi,
    )
    server = PhaseStreamServer(monitor=mon, tick_interval_s=args.tick_interval)
    server.serve_sync(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
