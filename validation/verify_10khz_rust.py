# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — 10kHz Rust Verification
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Verifies the migrated Rust flight simulator at 10kHz and 30kHz.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Literal

try:
    import scpn_fusion_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("Warning: scpn_fusion_rs not found. Using mock for verification script.")


@dataclass(frozen=True)
class FlightReport:
    steps: int
    duration_s: float
    wall_time_ms: float
    mean_abs_r_error: float
    mean_abs_z_error: float
    disrupted: bool
    r_history: tuple[float, ...]
    z_history: tuple[float, ...]
    max_step_time_us: float | None
    timing_measured: bool


class MockRustFlightSim:
    def __init__(self, target_r: float = 6.2, target_z: float = 0.0, control_hz: float = 10000.0):
        self.control_hz = control_hz
        self.target_r = target_r
        self.target_z = target_z

    def run_shot(self, shot_duration_s: float) -> FlightReport:
        steps = int(shot_duration_s * self.control_hz)
        return FlightReport(
            steps=steps,
            duration_s=float(shot_duration_s),
            wall_time_ms=float("nan"),
            mean_abs_r_error=0.001,
            mean_abs_z_error=0.002,
            disrupted=False,
            r_history=tuple([self.target_r] * steps),
            z_history=tuple([self.target_z] * steps),
            max_step_time_us=None,
            timing_measured=False,
        )


Mode = Literal["auto", "mock", "measure"]


def _build_sim(control_hz: float, mode: Mode):
    if mode == "mock":
        return MockRustFlightSim(control_hz=control_hz)
    if mode == "measure":
        if not HAS_RUST:
            raise RuntimeError(
                "Rust backend unavailable. Install/build scpn_fusion_rs or use --mode mock."
            )
        return scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, control_hz)
    if HAS_RUST:
        return scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, control_hz)
    return MockRustFlightSim(control_hz=control_hz)


def _format_latency(report: FlightReport) -> tuple[str, str]:
    if not report.timing_measured:
        return "N/A (mock mode)", "N/A (mock mode)"
    if report.steps <= 0 or not math.isfinite(report.wall_time_ms):
        return "N/A", "N/A"
    avg_us = report.wall_time_ms * 1000.0 / report.steps
    max_us = report.max_step_time_us if report.max_step_time_us is not None else float("nan")
    avg_text = f"{avg_us:.2f} us" if math.isfinite(avg_us) else "N/A"
    max_text = f"{max_us:.2f} us" if math.isfinite(max_us) else "N/A"
    return avg_text, max_text


def _coerce_report(raw: object, *, timing_measured: bool) -> FlightReport:
    def _as_float(name: str, default: float = float("nan")) -> float:
        value = getattr(raw, name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _as_bool(name: str, default: bool = False) -> bool:
        return bool(getattr(raw, name, default))

    def _as_tuple(name: str) -> tuple[float, ...]:
        value = getattr(raw, name, ())
        try:
            return tuple(float(v) for v in value)
        except TypeError:
            return ()

    max_step = getattr(raw, "max_step_time_us", None)
    max_step_us: float | None
    if max_step is None:
        max_step_us = None
    else:
        try:
            max_step_us = float(max_step)
        except (TypeError, ValueError):
            max_step_us = None

    return FlightReport(
        steps=int(getattr(raw, "steps", 0)),
        duration_s=_as_float("duration_s", 0.0),
        wall_time_ms=_as_float("wall_time_ms"),
        mean_abs_r_error=_as_float("mean_abs_r_error", 0.0),
        mean_abs_z_error=_as_float("mean_abs_z_error", 0.0),
        disrupted=_as_bool("disrupted", False),
        r_history=_as_tuple("r_history"),
        z_history=_as_tuple("z_history"),
        max_step_time_us=max_step_us,
        timing_measured=timing_measured,
    )


def _print_report(
    label: str,
    report: FlightReport,
    *,
    print_fn: Callable[[str], None],
) -> None:
    avg_text, max_text = _format_latency(report)
    print_fn(f"\n{label}")
    print_fn(f"  Steps executed: {report.steps}")
    if math.isfinite(report.wall_time_ms):
        print_fn(f"  Total wall time: {report.wall_time_ms:.2f} ms")
    else:
        print_fn("  Total wall time: N/A (mock mode)")
    print_fn(f"  Avg latency per step: {avg_text}")
    print_fn(f"  Max jitter (step time): {max_text}")
    print_fn(f"  Mean R error: {report.mean_abs_r_error:.6f}")
    print_fn(f"  Disrupted: {report.disrupted}")


def run_verification(
    *,
    mode: Mode = "auto",
    shot_seconds: float = 30.0,
    print_fn: Callable[[str], None] = print,
) -> dict[str, object]:
    print_fn("=== SCPN Fusion Core: 10kHz Rust Migration Verification ===")

    hz_10k = 10000.0
    sim_10k = _build_sim(hz_10k, mode)
    raw_10k = sim_10k.run_shot(float(shot_seconds))
    report_10k = _coerce_report(raw_10k, timing_measured=not isinstance(sim_10k, MockRustFlightSim))
    _print_report("10kHz Test (Target: 100 us per loop)", report_10k, print_fn=print_fn)

    hz_30k = 30000.0
    sim_30k = _build_sim(hz_30k, mode)
    raw_30k = sim_30k.run_shot(float(shot_seconds))
    report_30k = _coerce_report(raw_30k, timing_measured=not isinstance(sim_30k, MockRustFlightSim))
    _print_report("30kHz Test (Target: 33 us per loop)", report_30k, print_fn=print_fn)

    if report_10k.timing_measured and report_30k.timing_measured:
        print_fn("\n[SUCCESS] Rust-native verification completed with measured timings.")
    else:
        print_fn(
            "\n[MOCK] Structural verification completed; timing metrics disabled. "
            "Use --mode measure with scpn_fusion_rs for hardware timing evidence."
        )

    return {
        "mode": mode,
        "has_rust_backend": HAS_RUST,
        "timing_measured": bool(report_10k.timing_measured and report_30k.timing_measured),
        "reports": {
            "10khz": report_10k,
            "30khz": report_30k,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "mock", "measure"),
        default="auto",
        help="auto=use Rust when available, mock=structural path only, measure=fail without Rust.",
    )
    parser.add_argument(
        "--shot-seconds",
        type=float,
        default=30.0,
        help="Shot duration passed to run_shot (default: 30.0).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_verification(mode=args.mode, shot_seconds=args.shot_seconds)
