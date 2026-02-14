# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Latency benchmark: PID vs SNN (NumPy) vs SNN (SC-NeuroCore).

Measures per-control-cycle compute latency over 100,000 iterations.
Reports: mean, median (p50), p95, p99, min, max in microseconds.

Usage
-----
    python -m validation.benchmark_snn_latency          # from repo root
    python validation/benchmark_snn_latency.py          # direct invocation

Output
------
Formatted table to stdout and JSON to ``validation/results/latency_benchmark.json``.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running as a script.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from scpn_fusion.control.pid_baseline import PIDController, PIDConfig
from scpn_fusion.scpn.vertical_control_net import VerticalControlNet
from scpn_fusion.scpn.vertical_snn_controller import VerticalSNNController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_ITERATIONS = 1_000
BENCHMARK_ITERATIONS = 100_000

# Fixed test inputs (representative of a small upward displacement).
Z_MEAS = 0.003        # 3 mm vertical displacement
DZ_MEAS = 0.1         # 0.1 m/s vertical velocity


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Per-controller latency statistics in microseconds."""
    controller: str
    backend: str
    iterations: int
    mean_us: float
    median_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    total_s: float


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    generated_at_utc: str
    warmup_iterations: int
    benchmark_iterations: int
    z_meas: float
    dz_meas: float
    results: List[Dict]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _compute_stats(
    name: str,
    backend: str,
    latencies_s: np.ndarray,
) -> LatencyStats:
    """Compute latency statistics from raw per-iteration timings.

    Parameters
    ----------
    name : str
        Controller name for display.
    backend : str
        Backend identifier (e.g. "pid", "numpy", "sc_neurocore").
    latencies_s : ndarray
        Per-iteration elapsed times in seconds.

    Returns
    -------
    LatencyStats
        Aggregated statistics with all values in microseconds.
    """
    us = latencies_s * 1e6  # convert to microseconds
    return LatencyStats(
        controller=name,
        backend=backend,
        iterations=len(us),
        mean_us=float(np.mean(us)),
        median_us=float(np.median(us)),
        p95_us=float(np.percentile(us, 95)),
        p99_us=float(np.percentile(us, 99)),
        min_us=float(np.min(us)),
        max_us=float(np.max(us)),
        total_s=float(np.sum(latencies_s)),
    )


def _run_benchmark(
    controller,
    name: str,
    backend: str,
    warmup: int = WARMUP_ITERATIONS,
    iterations: int = BENCHMARK_ITERATIONS,
) -> LatencyStats:
    """Benchmark a single controller.

    Runs *warmup* calls (not measured) then *iterations* calls,
    recording ``time.perf_counter()`` around each ``controller.compute()``.

    Parameters
    ----------
    controller
        Object with a ``.compute(z, dz)`` method and a ``.reset()`` method.
    name : str
        Human-readable controller name.
    backend : str
        Backend identifier.
    warmup : int
        Number of unmeasured warm-up iterations.
    iterations : int
        Number of measured iterations.

    Returns
    -------
    LatencyStats
        Aggregated latency statistics.
    """
    # ---- Warm-up (not measured) ------------------------------------------
    for _ in range(warmup):
        controller.compute(Z_MEAS, DZ_MEAS)

    controller.reset()

    # ---- Benchmark (measured) --------------------------------------------
    latencies = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        t0 = time.perf_counter()
        controller.compute(Z_MEAS, DZ_MEAS)
        t1 = time.perf_counter()
        latencies[i] = t1 - t0

    return _compute_stats(name, backend, latencies)


# ---------------------------------------------------------------------------
# Controller factories
# ---------------------------------------------------------------------------

def _make_pid() -> PIDController:
    """Create a default PID controller."""
    return PIDController(PIDConfig())


def _make_snn_numpy() -> VerticalSNNController:
    """Create a VerticalSNNController with force_numpy=True."""
    vcn = VerticalControlNet()
    vcn.create_net()
    return VerticalSNNController(vcn, force_numpy=True, seed=42)


def _make_snn_sc_neurocore() -> Optional[VerticalSNNController]:
    """Create a VerticalSNNController using SC-NeuroCore backend.

    Returns None if sc_neurocore is not available.
    """
    try:
        from sc_neurocore import SCDenseLayer  # noqa: F401
    except ImportError:
        return None

    vcn = VerticalControlNet()
    vcn.create_net()
    return VerticalSNNController(vcn, force_numpy=False, seed=42)


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------

_HEADER_FMT = (
    "{:<25s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}"
)
_ROW_FMT = (
    "{:<25s}  {:>10.2f}  {:>10.2f}  {:>10.2f}  {:>10.2f}  {:>10.2f}  {:>10.2f}"
)


def _print_table(stats_list: List[LatencyStats]) -> None:
    """Print a formatted results table to stdout."""
    sep = "-" * 97
    print()
    print(sep)
    print("  SCPN Fusion Core -- SNN Latency Benchmark")
    print(f"  {WARMUP_ITERATIONS:,} warmup + {BENCHMARK_ITERATIONS:,} measured iterations")
    print(f"  Test input: z = {Z_MEAS} m, dz = {DZ_MEAS} m/s")
    print(sep)
    print(
        _HEADER_FMT.format(
            "Controller",
            "Mean (us)",
            "p50 (us)",
            "p95 (us)",
            "p99 (us)",
            "Min (us)",
            "Max (us)",
        )
    )
    print(sep)
    for s in stats_list:
        print(
            _ROW_FMT.format(
                f"{s.controller} [{s.backend}]",
                s.mean_us,
                s.median_us,
                s.p95_us,
                s.p99_us,
                s.min_us,
                s.max_us,
            )
        )
    print(sep)
    print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _save_json(
    stats_list: List[LatencyStats],
    output_path: Path,
) -> None:
    """Save benchmark results to a JSON file.

    Parameters
    ----------
    stats_list : list[LatencyStats]
        Results for all benchmarked controllers.
    output_path : Path
        Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = BenchmarkResult(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        warmup_iterations=WARMUP_ITERATIONS,
        benchmark_iterations=BENCHMARK_ITERATIONS,
        z_meas=Z_MEAS,
        dz_meas=DZ_MEAS,
        results=[asdict(s) for s in stats_list],
    )
    with open(output_path, "w") as fh:
        json.dump(asdict(result), fh, indent=2)

    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> List[LatencyStats]:
    """Execute the full latency benchmark and return statistics.

    Benchmarks three controller paths in order:
    1. PID baseline (expected < 10 us/cycle)
    2. SNN with NumPy float-path (expected < 1 ms/cycle)
    3. SNN with SC-NeuroCore (expected < 100 us/cycle; skipped if unavailable)

    Returns
    -------
    list[LatencyStats]
        One entry per benchmarked controller.
    """
    all_stats: List[LatencyStats] = []

    # ---- 1. PID Controller -----------------------------------------------
    print("Benchmarking PID controller ...")
    pid = _make_pid()
    stats_pid = _run_benchmark(pid, "PIDController", "pid")
    all_stats.append(stats_pid)
    print(f"  PID  mean={stats_pid.mean_us:.2f} us  p99={stats_pid.p99_us:.2f} us")

    # ---- 2. SNN (NumPy) --------------------------------------------------
    print("Benchmarking SNN (NumPy float-path) ...")
    snn_np = _make_snn_numpy()
    stats_snn_np = _run_benchmark(snn_np, "VerticalSNNController", "numpy")
    all_stats.append(stats_snn_np)
    print(f"  SNN-NumPy  mean={stats_snn_np.mean_us:.2f} us  p99={stats_snn_np.p99_us:.2f} us")

    # ---- 3. SNN (SC-NeuroCore) -------------------------------------------
    print("Benchmarking SNN (SC-NeuroCore) ...")
    snn_sc = _make_snn_sc_neurocore()
    if snn_sc is not None:
        stats_snn_sc = _run_benchmark(snn_sc, "VerticalSNNController", "sc_neurocore")
        all_stats.append(stats_snn_sc)
        print(
            f"  SNN-SC  mean={stats_snn_sc.mean_us:.2f} us  "
            f"p99={stats_snn_sc.p99_us:.2f} us"
        )
    else:
        print("  SC-NeuroCore not available -- skipping.")

    return all_stats


def main() -> None:
    """Entry point: run benchmark, print table, save JSON."""
    logging.basicConfig(level=logging.WARNING)
    stats = run_benchmark()
    _print_table(stats)

    output_path = Path(__file__).resolve().parent / "results" / "latency_benchmark.json"
    _save_json(stats, output_path)


if __name__ == "__main__":
    main()
