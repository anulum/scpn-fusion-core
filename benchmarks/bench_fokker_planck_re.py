#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fokker-Planck RE Tier Benchmark
"""Benchmark the dispatched runaway-electron Fokker-Planck backends.

Measures the MUSCL-Hancock step-loop wall-clock per backend tier (RUST and
NUMPY) on the shared 1D-in-momentum runaway-electron model, checks the shared
physics invariants (finite ``(n_re, current_re)`` diagnostics and a
non-negative distribution), and records which tier the class-kernel dispatcher
selects. Timings are local non-isolated regression evidence, not production
throughput claims.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT = ROOT / "validation" / "reports" / "fokker_planck_re_benchmark.json"
SCHEMA = "scpn-fusion-core.fokker-planck-re-benchmark.v1"
KERNEL_NAME = "fokker_planck_re"
DEFAULT_GRID = 200
DEFAULT_P_MAX = 100.0
DEFAULT_STEPS = 200
DEFAULT_REPEATS = 3

sys.path.insert(0, str(SRC))

from scpn_fusion.core import _multi_compat as multi  # noqa: E402
from scpn_fusion.core import _multi_compat_providers as providers  # noqa: E402

MetricRow: TypeAlias = dict[str, Any]

# Bounded unforced regime: no accelerating field, so the explicit scheme stays
# well conditioned and the tiers are directly comparable.
_STEP_PARAMS = (1.0e-6, 0.0, 1.0e19, 5000.0, 1.0)


def _backend_loaders() -> dict[str, Any]:
    """Return the tier-name -> class-loader mapping for the Fokker-Planck kernel."""
    return {
        "RUST": providers._load_rust_fokker_planck,
        "NUMPY": providers._load_numpy_fokker_planck,
    }


def _seed_distribution(grid: int, p_max: float) -> list[float]:
    """Return a Gaussian momentum seed on the shared log-spaced grid."""
    numpy_kernel = providers._load_numpy_fokker_planck()(grid, p_max)
    p_grid = numpy_kernel.get_p()
    return (1.0e10 * np.exp(-((p_grid - 5.0) ** 2) / 2.0)).tolist()


def _run_backend(
    kernel_cls: type, grid: int, p_max: float, steps: int, repeats: int, seed: list[float]
) -> MetricRow:
    """Time the seeded step loop for one backend and check its invariants."""
    elapsed: list[float] = []
    final_n_re = float("nan")
    invariants_passed = False
    for _ in range(repeats):
        kernel = kernel_cls(grid, p_max)
        kernel.set_f(seed)
        start = time.perf_counter()
        history = [kernel.step(*_STEP_PARAMS) for _ in range(steps)]
        elapsed.append(time.perf_counter() - start)
        densities = np.asarray([n_re for n_re, _ in history], dtype=np.float64)
        currents = np.asarray([current for _, current in history], dtype=np.float64)
        distribution = np.asarray(kernel.get_f(), dtype=np.float64)
        final_n_re = float(densities[-1])
        invariants_passed = bool(
            np.all(np.isfinite(densities))
            and np.all(np.isfinite(currents))
            and np.all(distribution >= 0.0)
        )
    return {
        "median_loop_s": float(np.median(elapsed)),
        "median_step_us": float(np.median(elapsed) / steps * 1e6),
        "final_n_re": final_n_re,
        "invariants_passed": invariants_passed,
    }


def build_report(
    *,
    grid: int = DEFAULT_GRID,
    p_max: float = DEFAULT_P_MAX,
    steps: int = DEFAULT_STEPS,
    repeats: int = DEFAULT_REPEATS,
) -> MetricRow:
    """Build the Fokker-Planck runaway-electron backend benchmark report payload."""
    registered = multi.registered_kernel_classes().get(KERNEL_NAME, [])
    selected_cls = multi.dispatch_kernel_class(KERNEL_NAME)
    seed = _seed_distribution(grid, p_max)
    backends: dict[str, MetricRow] = {}
    for tier_name, loader in _backend_loaders().items():
        try:
            kernel_cls = loader()
        except (ImportError, AttributeError, TypeError) as exc:
            backends[tier_name] = {"status": "unavailable", "reason": str(exc)}
            continue
        row = _run_backend(kernel_cls, grid, p_max, steps, repeats, seed)
        row["status"] = "available"
        row["class_name"] = kernel_cls.__name__
        backends[tier_name] = row
    speedup: float | None = None
    rust_row = backends.get("RUST", {})
    numpy_row = backends.get("NUMPY", {})
    if rust_row.get("status") == "available" and numpy_row.get("status") == "available":
        speedup = float(numpy_row["median_loop_s"] / max(rust_row["median_loop_s"], 1e-12))
    all_invariants = all(
        row.get("invariants_passed", False)
        for row in backends.values()
        if row.get("status") == "available"
    )
    return {
        "schema": SCHEMA,
        "kernel": KERNEL_NAME,
        "grid": grid,
        "p_max": p_max,
        "steps": steps,
        "repeats": repeats,
        "registered_tiers": registered,
        "selected_class": selected_cls.__name__,
        "backends": backends,
        "rust_over_numpy_speedup": speedup,
        "physics_invariants_passed": all_invariants,
        "model_contract": (
            "1D-in-momentum runaway-electron Fokker-Planck: MUSCL-Hancock "
            "advection, central-difference diffusion half-step, operator-split "
            "avalanche/Dreicer/knock-on sources; backends implement the "
            "identical scheme and agree deterministically to floating-point "
            "summation order in this bounded unforced regime"
        ),
        "benchmark_evidence": {
            "classification": "local_non_isolated_regression",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Fokker-Planck runaway-electron backend benchmark and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID)
    parser.add_argument("--p-max", type=float, default=DEFAULT_P_MAX)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    report = build_report(grid=args.grid, p_max=args.p_max, steps=args.steps, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["physics_invariants_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
