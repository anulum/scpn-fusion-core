#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Hall-MHD Discovery Tier Benchmark
"""Benchmark the dispatched Hall-MHD discovery simulator backends.

Measures the sim-loop wall-clock per backend tier (RUST and NUMPY) on the
reconciled reduced Hall-MHD model, checks the shared physics invariants
(finite non-negative energies, unforced late-time decay), and records which
tier the class-kernel dispatcher selects. Timings are local non-isolated
regression evidence, not production throughput claims.
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
REPORT = ROOT / "validation" / "reports" / "hall_mhd_discovery_benchmark.json"
SCHEMA = "scpn-fusion-core.hall-mhd-discovery-benchmark.v1"
KERNEL_NAME = "hall_mhd_discovery"
DEFAULT_GRID = 64
DEFAULT_STEPS = 200
DEFAULT_REPEATS = 3
BENCH_SEED = 2026

sys.path.insert(0, str(SRC))

from scpn_fusion.core import _multi_compat as multi  # noqa: E402

MetricRow: TypeAlias = dict[str, Any]


def _backend_loaders() -> dict[str, Any]:
    """Return the tier-name → class-loader mapping for the Hall-MHD kernel."""
    return {
        "RUST": multi._load_rust_hall_mhd,
        "NUMPY": multi._load_numpy_hall_mhd,
    }


def _run_backend(simulator_cls: type, grid: int, steps: int, repeats: int) -> MetricRow:
    """Time the seeded sim loop for one backend and check its invariants."""
    elapsed: list[float] = []
    final_energy = float("nan")
    invariants_passed = False
    for _ in range(repeats):
        sim = simulator_cls(grid, None, None, seed=BENCH_SEED, background_amplitude=0.0)
        start = time.perf_counter()
        results = [sim.step() for _ in range(steps)]
        elapsed.append(time.perf_counter() - start)
        totals = np.asarray([total for total, _ in results], dtype=np.float64)
        zonals = np.asarray([zonal for _, zonal in results], dtype=np.float64)
        final_energy = float(totals[-1])
        invariants_passed = bool(
            np.all(np.isfinite(totals))
            and np.all(totals >= 0.0)
            and np.all(np.isfinite(zonals))
            and totals[-1] <= totals[0]
        )
    return {
        "median_loop_s": float(np.median(elapsed)),
        "median_step_us": float(np.median(elapsed) / steps * 1e6),
        "final_energy": final_energy,
        "invariants_passed": invariants_passed,
    }


def build_report(
    *, grid: int = DEFAULT_GRID, steps: int = DEFAULT_STEPS, repeats: int = DEFAULT_REPEATS
) -> MetricRow:
    """Build the Hall-MHD discovery backend benchmark report payload."""
    registered = multi.registered_kernel_classes().get(KERNEL_NAME, [])
    selected_cls = multi.dispatch_kernel_class(KERNEL_NAME)
    backends: dict[str, MetricRow] = {}
    for tier_name, loader in _backend_loaders().items():
        try:
            simulator_cls = loader()
        except (ImportError, AttributeError, TypeError) as exc:
            backends[tier_name] = {"status": "unavailable", "reason": str(exc)}
            continue
        row = _run_backend(simulator_cls, grid, steps, repeats)
        row["status"] = "available"
        row["class_name"] = simulator_cls.__name__
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
        "steps": steps,
        "repeats": repeats,
        "seed": BENCH_SEED,
        "registered_tiers": registered,
        "selected_class": selected_cls.__name__,
        "backends": backends,
        "rust_over_numpy_speedup": speedup,
        "physics_invariants_passed": all_invariants,
        "model_contract": (
            "reconciled reduced Hall-MHD: hyper-viscous -nu k^4 U, resistive "
            "-eta k^2 psi, optional static current-sheet drive; backends are "
            "statistically equivalent (language-native seeded RNG), not bit-exact"
        ),
        "benchmark_evidence": {
            "classification": "local_non_isolated_regression",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Hall-MHD discovery backend benchmark and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    report = build_report(grid=args.grid, steps=args.steps, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["physics_invariants_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
