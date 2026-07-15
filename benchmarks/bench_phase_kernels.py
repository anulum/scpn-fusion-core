#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Phase Kernel Benchmark
"""Benchmark the ``kuramoto_step`` and ``upde_tick`` dispatcher tiers (M-3).

Times the Rust ``fusion-phase`` tier against the NumPy floor on identical
seeded workloads (single-population Kuramoto-Sakaguchi stepping and the
multi-layer UPDE tick) and records the cross-tier trajectory agreement.
The kernels are deterministic, so agreement is bounded by floating-point
summation order. Timings are local non-isolated regression evidence, not
production throughput claims.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT = ROOT / "validation" / "reports" / "phase_kernels_benchmark.json"
SCHEMA = "scpn-fusion-core.phase-kernels-benchmark.v1"
BENCH_SEED = 2026
KURAMOTO_N = 10_000
KURAMOTO_STEPS = 500
UPDE_LAYERS = 8
UPDE_N_PER_LAYER = 1024
UPDE_TICKS = 200
DEFAULT_REPEATS = 3

sys.path.insert(0, str(SRC))

import numpy.typing as npt  # noqa: E402

from scpn_fusion.core import _multi_compat as multi  # noqa: E402
from scpn_fusion.core import _multi_compat_providers as providers  # noqa: E402

FloatArray: TypeAlias = npt.NDArray[np.float64]
MetricRow: TypeAlias = dict[str, Any]


def _time_loop(fn: Callable[[], FloatArray], repeats: int) -> tuple[float, FloatArray]:
    """Return the median wall time and final state of a benchmark loop."""
    elapsed: list[float] = []
    state: FloatArray | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        state = fn()
        elapsed.append(time.perf_counter() - start)
    assert state is not None
    return float(np.median(elapsed)), state


def _kuramoto_workload(impl: Callable[..., Any]) -> Callable[[], FloatArray]:
    """Build the seeded fixed-step Kuramoto workload for one tier."""
    rng = np.random.default_rng(BENCH_SEED)
    theta0 = rng.uniform(-np.pi, np.pi, size=KURAMOTO_N)
    omega = rng.normal(0.0, 0.2, size=KURAMOTO_N)

    def run() -> FloatArray:
        theta = theta0.copy()
        for _ in range(KURAMOTO_STEPS):
            out = impl(theta, omega, dt=1e-3, K=1.5, alpha=0.05, zeta=0.4, psi=0.0, wrap=True)
            theta = out["theta1"]
        return np.asarray(theta, dtype=np.float64)

    return run


def _kuramoto_run_workload(impl: Callable[..., Any]) -> Callable[[], FloatArray]:
    """Build the seeded batched-run Kuramoto workload for one tier.

    Uses the same seeded initial state as :func:`_kuramoto_workload` so the
    batched trajectory is identical to the per-step one; the batched tier runs
    the whole loop in one dispatch instead of ``KURAMOTO_STEPS`` calls.
    """
    rng = np.random.default_rng(BENCH_SEED)
    theta0 = rng.uniform(-np.pi, np.pi, size=KURAMOTO_N)
    omega = rng.normal(0.0, 0.2, size=KURAMOTO_N)

    def run() -> FloatArray:
        out = impl(
            theta0,
            omega,
            n_steps=KURAMOTO_STEPS,
            dt=1e-3,
            K=1.5,
            alpha=0.05,
            zeta=0.4,
            psi=0.0,
            wrap=True,
        )
        return np.asarray(out["theta_final"], dtype=np.float64)

    return run


def _upde_workload(impl: Callable[..., Any]) -> Callable[[], FloatArray]:
    """Build the seeded fixed-tick UPDE workload for one tier."""
    rng = np.random.default_rng(BENCH_SEED)
    total = UPDE_LAYERS * UPDE_N_PER_LAYER
    theta0 = rng.uniform(-np.pi, np.pi, size=total)
    omega = rng.normal(0.0, 0.1, size=total)
    offsets = np.arange(0, total + 1, UPDE_N_PER_LAYER, dtype=np.intp)
    K = 0.5 * np.eye(UPDE_LAYERS) + 0.05 * rng.uniform(size=(UPDE_LAYERS, UPDE_LAYERS))
    alpha = np.zeros((UPDE_LAYERS, UPDE_LAYERS))
    zeta = np.full(UPDE_LAYERS, 0.3)

    def run() -> FloatArray:
        theta = theta0.copy()
        for _ in range(UPDE_TICKS):
            out = impl(
                theta,
                omega,
                offsets,
                K,
                alpha,
                zeta,
                dt=1e-3,
                psi_global=0.2,
                actuation_gain=1.0,
                pac_gamma=0.2,
                wrap=True,
            )
            theta = np.asarray(out["theta1"], dtype=np.float64)
        return theta

    return run


def _bench_kernel(
    label: str,
    numpy_impl: Callable[..., Any],
    rust_impl: Callable[..., Any],
    workload: Callable[[Callable[..., Any]], Callable[[], FloatArray]],
    repeats: int,
    rust_available: bool,
) -> MetricRow:
    """Benchmark one kernel across its tiers and record agreement."""
    numpy_s, numpy_state = _time_loop(workload(numpy_impl), repeats)
    row: MetricRow = {"kernel": label, "numpy_median_s": numpy_s}
    if rust_available:
        rust_s, rust_state = _time_loop(workload(rust_impl), repeats)
        delta = float(np.linalg.norm(rust_state - numpy_state))
        row["rust_median_s"] = rust_s
        row["rust_over_numpy_speedup"] = numpy_s / max(rust_s, 1e-12)
        row["rust_vs_numpy_final_state_rel_l2"] = delta / max(
            float(np.linalg.norm(numpy_state)), 1e-30
        )
    return row


def _upde_run_workload(impl: Callable[..., Any]) -> Callable[[], FloatArray]:
    """Build the seeded batched-run UPDE workload for one tier."""
    rng = np.random.default_rng(BENCH_SEED)
    total = UPDE_LAYERS * UPDE_N_PER_LAYER
    theta0 = rng.uniform(-np.pi, np.pi, size=total)
    omega = rng.normal(0.0, 0.1, size=total)
    offsets = np.arange(0, total + 1, UPDE_N_PER_LAYER, dtype=np.intp)
    K = 0.5 * np.eye(UPDE_LAYERS) + 0.05 * rng.uniform(size=(UPDE_LAYERS, UPDE_LAYERS))
    alpha = np.zeros((UPDE_LAYERS, UPDE_LAYERS))
    zeta = np.full(UPDE_LAYERS, 0.3)

    def run() -> FloatArray:
        out = impl(
            theta0,
            omega,
            offsets,
            K,
            alpha,
            zeta,
            n_steps=UPDE_TICKS,
            dt=1e-3,
            psi_global=0.2,
            actuation_gain=1.0,
            pac_gamma=0.2,
            wrap=True,
        )
        return np.asarray(out["theta_final"], dtype=np.float64)

    return run


def build_report(*, repeats: int = DEFAULT_REPEATS) -> MetricRow:
    """Build the phase-kernel benchmark report payload."""
    rust_available = multi.is_available(multi.BackendTier.RUST)
    kernels = [
        _bench_kernel(
            f"kuramoto_step (N={KURAMOTO_N}, steps={KURAMOTO_STEPS})",
            providers._numpy_kuramoto_step,
            providers._rust_kuramoto_step,
            _kuramoto_workload,
            repeats,
            rust_available,
        ),
        _bench_kernel(
            f"kuramoto_run batched (N={KURAMOTO_N}, steps={KURAMOTO_STEPS})",
            providers._numpy_kuramoto_run,
            providers._rust_kuramoto_run,
            _kuramoto_run_workload,
            repeats,
            rust_available,
        ),
        _bench_kernel(
            f"upde_tick (L={UPDE_LAYERS}, N/layer={UPDE_N_PER_LAYER}, ticks={UPDE_TICKS})",
            providers._numpy_upde_tick,
            providers._rust_upde_tick,
            _upde_workload,
            repeats,
            rust_available,
        ),
        _bench_kernel(
            f"upde_run batched (L={UPDE_LAYERS}, N/layer={UPDE_N_PER_LAYER}, ticks={UPDE_TICKS})",
            providers._numpy_upde_run,
            providers._rust_upde_run,
            _upde_run_workload,
            repeats,
            rust_available,
        ),
    ]
    return {
        "schema": SCHEMA,
        "seed": BENCH_SEED,
        "repeats": repeats,
        "rust_available": rust_available,
        "kernels": kernels,
        "problem_contract": (
            "identical seeded deterministic workloads per tier; the kernels "
            "carry no RNG, so final-state agreement is bounded by "
            "floating-point summation order"
        ),
        "benchmark_evidence": {
            "classification": "local_non_isolated_regression",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the phase-kernel benchmark and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    report = build_report(repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
