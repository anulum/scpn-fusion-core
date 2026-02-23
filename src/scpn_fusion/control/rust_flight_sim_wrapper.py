# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Rust Flight Sim Wrapper
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path
import argparse

logger = logging.getLogger(__name__)

def run():
    try:
        import scpn_fusion_rs
    except ImportError:
        logger.error("scpn_fusion_rs extension not found. Build with cargo first.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Rust-Native High-Speed Flight Sim")
    parser.add_argument("--hz", type=float, default=10000.0, help="Control frequency (Hz)")
    parser.add_argument("--duration", type=float, default=30.0, help="Shot duration (s)")
    parser.add_argument("--deterministic", action="store_true", help="Enable high-precision busy-wait loop")
    args = parser.parse_args()

    logger.info("--- Initiating Rust-Native Flight Sim (%.0f Hz) ---", args.hz)
    if args.deterministic:
        logger.info("  [HARDENED] Deterministic timing enabled (Busy-wait mode)")
    sim = scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, args.hz)
    report = sim.run_shot(args.duration, deterministic=args.deterministic)

    logger.info("Shot Complete:")
    logger.info("  Steps: %d", report.steps)
    logger.info("  Wall Time: %.2f ms", report.wall_time_ms)
    logger.info("  Avg Latency: %.2f us", report.wall_time_ms * 1000 / report.steps)
    logger.info("  Mean R Error: %.6f", report.mean_abs_r_error)
    logger.info("  Disrupted: %s", report.disrupted)

if __name__ == "__main__":
    run()
