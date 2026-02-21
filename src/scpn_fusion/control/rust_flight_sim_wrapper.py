# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Rust Flight Sim Wrapper
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
import argparse

def run():
    try:
        import scpn_fusion_rs
    except ImportError:
        print("Error: scpn_fusion_rs extension not found. Build with cargo first.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Rust-Native High-Speed Flight Sim")
    parser.add_argument("--hz", type=float, default=10000.0, help="Control frequency (Hz)")
    parser.add_argument("--duration", type=float, default=30.0, help="Shot duration (s)")
    parser.add_argument("--deterministic", action="store_true", help="Enable high-precision busy-wait loop")
    args = parser.parse_args()

    print(f"--- Initiating Rust-Native Flight Sim ({args.hz} Hz) ---")
    if args.deterministic:
        print("  [HARDENED] Deterministic timing enabled (Busy-wait mode)")
    sim = scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, args.hz)
    report = sim.run_shot(args.duration, deterministic=args.deterministic)
    
    print(f"Shot Complete:")
    print(f"  Steps: {report.steps}")
    print(f"  Wall Time: {report.wall_time_ms:.2f} ms")
    print(f"  Avg Latency: {(report.wall_time_ms * 1000 / report.steps):.2f} μs")
    print(f"  Mean R Error: {report.mean_abs_r_error:.6f}")
    print(f"  Disrupted: {report.disrupted}")

if __name__ == "__main__":
    run()
