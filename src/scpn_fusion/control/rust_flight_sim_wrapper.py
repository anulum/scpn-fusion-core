# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Rust Flight Sim Wrapper
"""Command-line wrapper for the Rust-native tokamak flight simulator.

The wrapper is intentionally minimal: it validates that the optional Rust
extension is importable, parses latency-oriented CLI flags, executes one shot,
and prints a deterministic summary of timing and disruption outcomes.
"""

import argparse
import logging
import sys
from typing import Protocol, cast

from scpn_fusion.core import _multi_compat

logger = logging.getLogger(__name__)


class _FlightSimReport(Protocol):
    """Structured report returned by the Rust flight simulator."""

    steps: int
    wall_time_ms: float
    mean_abs_r_error: float
    disrupted: bool


class _FlightSim(Protocol):
    """Runtime protocol implemented by the Rust flight simulator instance."""

    def run_shot(self, duration: float, *, deterministic: bool = False) -> _FlightSimReport:
        """Run a single simulated tokamak shot and return latency metrics."""


class _FlightSimFactory(Protocol):
    """Constructor protocol for the Rust flight simulator class."""

    def __call__(self, r_target: float, z_target: float, hz: float) -> _FlightSim:
        """Create a Rust flight simulator bound to one target and control rate."""


def run() -> None:
    """Execute a Rust flight-simulation shot and log latency metrics.

    The helper exits with code 1 when the optional native extension is missing.
    It writes a machine-readable execution envelope as structured log entries
    so downstream wrappers can parse reproducible latency fields.

    Returns
    -------
        None.

    Raises
    ------
        SystemExit: If ``scpn_fusion_rs`` cannot be imported.
    """
    try:
        flight_sim_cls = cast(
            _FlightSimFactory,
            _multi_compat.dispatch_rust_symbol("PyRustFlightSim"),
        )
    except (AttributeError, ImportError):
        logger.error("scpn_fusion_rs extension not found. Build with cargo first.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Rust-Native High-Speed Flight Sim")
    parser.add_argument("--hz", type=float, default=10000.0, help="Control frequency (Hz)")
    parser.add_argument("--duration", type=float, default=30.0, help="Shot duration (s)")
    parser.add_argument(
        "--deterministic", action="store_true", help="Enable high-precision busy-wait loop"
    )
    args = parser.parse_args()

    logger.info("--- Initiating Rust-Native Flight Sim (%.0f Hz) ---", args.hz)
    if args.deterministic:
        logger.info("  [HARDENED] Deterministic timing enabled (Busy-wait mode)")
    sim = flight_sim_cls(6.2, 0.0, args.hz)
    report = sim.run_shot(args.duration, deterministic=args.deterministic)

    logger.info("Shot Complete:")
    logger.info("  Steps: %d", report.steps)
    logger.info("  Wall Time: %.2f ms", report.wall_time_ms)
    logger.info("  Avg Latency: %.2f us", report.wall_time_ms * 1000 / report.steps)
    logger.info("  Mean R Error: %.6f", report.mean_abs_r_error)
    logger.info("  Disrupted: %s", report.disrupted)


if __name__ == "__main__":
    run()
