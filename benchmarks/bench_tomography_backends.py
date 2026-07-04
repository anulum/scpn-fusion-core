#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tomography Backend Benchmark
"""Benchmark the tomography reconstruction backends on a synthetic phantom.

Times the Rust Tikhonov-NNLS path against the SciPy ``lsq_linear`` and SART
paths on the same seeded phantom and geometry, and records the cross-backend
relative L2 agreement. Timings are local non-isolated regression evidence,
not production throughput claims.
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
REPORT = ROOT / "validation" / "reports" / "tomography_backends_benchmark.json"
SCHEMA = "scpn-fusion-core.tomography-backends-benchmark.v1"
DEFAULT_GRID_RES = 20
DEFAULT_REPEATS = 3
BENCH_SEED = 2026

sys.path.insert(0, str(SRC))

import numpy.typing as npt  # noqa: E402

from scpn_fusion.diagnostics.tomography import PlasmaTomography  # noqa: E402

FloatArray: TypeAlias = npt.NDArray[np.float64]
MetricRow: TypeAlias = dict[str, Any]


class _BenchKernel:
    """Minimal kernel stub carrying the reconstruction-domain grids."""

    def __init__(self) -> None:
        self.R = np.linspace(4.0, 8.0, 65)
        self.Z = np.linspace(-4.0, 4.0, 65)


class _BenchSensors:
    """Deterministic 16-chord fan geometry for the benchmark phantom."""

    def __init__(self) -> None:
        self.kernel = _BenchKernel()
        origin = np.array([6.0, 5.0])
        targets_r = np.linspace(3.0, 9.0, 16)
        self.bolo_chords = [(origin, np.array([float(r), -4.0])) for r in targets_r]


def _time_method(
    tomo: PlasmaTomography, signals: FloatArray, method: str, repeats: int
) -> MetricRow:
    """Time one reconstruction method and capture its solution."""
    elapsed: list[float] = []
    solution: FloatArray | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        solution = tomo.reconstruct(signals, method=method)
        elapsed.append(time.perf_counter() - start)
    assert solution is not None
    return {
        "median_solve_s": float(np.median(elapsed)),
        "non_negative": bool(np.min(solution) >= 0.0),
        "solution": solution,
    }


def build_report(*, grid_res: int = DEFAULT_GRID_RES, repeats: int = DEFAULT_REPEATS) -> MetricRow:
    """Build the tomography backend benchmark report payload."""
    tomo = PlasmaTomography(_BenchSensors(), grid_res=grid_res, verbose=False)
    rng = np.random.default_rng(BENCH_SEED)
    phantom = rng.uniform(0.0, 1.0, size=tomo.n_pixels)
    signals = np.asarray(tomo.A @ phantom, dtype=np.float64)

    backends: dict[str, MetricRow] = {}
    for method in ("rust", "lsq_linear", "sart"):
        try:
            row = _time_method(tomo, signals, method, repeats)
        except (ImportError, AttributeError, TypeError) as exc:
            backends[method] = {"status": "unavailable", "reason": str(exc)}
            continue
        backends[method] = {
            "status": "available",
            "median_solve_s": row["median_solve_s"],
            "non_negative": row["non_negative"],
        }
        backends[method]["_solution"] = row["solution"]

    reference = backends.get("lsq_linear", {})
    agreement: dict[str, float] = {}
    if reference.get("status") == "available":
        ref_solution = reference["_solution"]
        ref_norm = float(np.linalg.norm(ref_solution))
        for method, row in backends.items():
            if method == "lsq_linear" or row.get("status") != "available":
                continue
            delta = float(np.linalg.norm(row["_solution"] - ref_solution))
            agreement[f"{method}_vs_lsq_linear_rel_l2"] = delta / max(ref_norm, 1e-30)
    for row in backends.values():
        row.pop("_solution", None)

    speedup: float | None = None
    rust_row = backends.get("rust", {})
    if rust_row.get("status") == "available" and reference.get("status") == "available":
        speedup = float(reference["median_solve_s"] / max(rust_row["median_solve_s"], 1e-12))

    return {
        "schema": SCHEMA,
        "grid_res": grid_res,
        "repeats": repeats,
        "seed": BENCH_SEED,
        "chords": 16,
        "backends": backends,
        "cross_backend_agreement": agreement,
        "rust_over_lsq_linear_speedup": speedup,
        "problem_contract": (
            "identical endpoint-inclusive geometry matrix and Tikhonov-NNLS "
            "objective min ||Ax-b||^2 + lambda ||x||^2, x >= 0, per backend"
        ),
        "benchmark_evidence": {
            "classification": "local_non_isolated_regression",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tomography backend benchmark and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-res", type=int, default=DEFAULT_GRID_RES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    report = build_report(grid_res=args.grid_res, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
