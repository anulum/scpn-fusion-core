#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GPU GS Smoother Benchmark
"""Benchmark the ``gs_rb_sor_smooth`` kernel tiers across grid sizes.

Times the wgpu compute-shader tier against the float64 NumPy reference on
identical fixed-sweep Red-Black SOR workloads of the toroidal GS* operator
(129x129 to 513x513), and records the cross-tier relative L2 agreement,
which is bounded by f32 round-off rather than by the algorithm. Timings are
local non-isolated regression evidence, not production throughput claims.
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
REPORT = ROOT / "validation" / "reports" / "gpu_gs_solver_benchmark.json"
SCHEMA = "scpn-fusion-core.gpu-gs-solver-benchmark.v1"
DEFAULT_GRIDS = (129, 257, 513)
DEFAULT_SWEEPS = 200
DEFAULT_REPEATS = 3
BENCH_SEED = 2026
R_LEFT, R_RIGHT = 4.0, 8.0
Z_BOTTOM, Z_TOP = -4.0, 4.0
OMEGA = 1.3

sys.path.insert(0, str(SRC))

import numpy.typing as npt  # noqa: E402

from scpn_fusion.core import _multi_compat as multi  # noqa: E402

FloatArray: TypeAlias = npt.NDArray[np.float64]
MetricRow: TypeAlias = dict[str, Any]


def _problem(n: int) -> tuple[FloatArray, FloatArray]:
    """Build a seeded Gaussian-source GS* smoothing problem on an n x n grid."""
    rng = np.random.default_rng(BENCH_SEED)
    r_axis = np.linspace(R_LEFT, R_RIGHT, n)
    z_axis = np.linspace(Z_BOTTOM, Z_TOP, n)
    r_grid, z_grid = np.meshgrid(r_axis, z_axis)
    source = -np.exp(-((r_grid - 6.0) ** 2 + z_grid**2) / 0.5)
    psi0 = rng.normal(0.0, 1e-3, size=(n, n))
    psi0[0, :] = psi0[-1, :] = psi0[:, 0] = psi0[:, -1] = 0.0
    return psi0, source


def _time_tier(
    impl: Any, psi0: FloatArray, source: FloatArray, sweeps: int, repeats: int
) -> MetricRow:
    """Time one smoother tier and capture its solution."""
    elapsed: list[float] = []
    solution: FloatArray | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        solution = impl(
            psi0, source, R_LEFT, R_RIGHT, Z_BOTTOM, Z_TOP, omega=OMEGA, n_sweeps=sweeps
        )
        elapsed.append(time.perf_counter() - start)
    assert solution is not None
    return {"median_solve_s": float(np.median(elapsed)), "solution": solution}


def build_report(
    *,
    grids: Sequence[int] = DEFAULT_GRIDS,
    sweeps: int = DEFAULT_SWEEPS,
    repeats: int = DEFAULT_REPEATS,
) -> MetricRow:
    """Build the GPU GS smoother benchmark report payload."""
    gpu_available = multi.is_available(multi.BackendTier.GPU)
    gpu_info: str | None = None
    if gpu_available:
        import scpn_fusion_rs

        gpu_info = scpn_fusion_rs.py_gpu_info()

    rows: list[MetricRow] = []
    for n in grids:
        psi0, source = _problem(n)
        numpy_row = _time_tier(multi._numpy_gs_rb_sor_smooth, psi0, source, sweeps, repeats)
        row: MetricRow = {
            "grid": f"{n}x{n}",
            "n": n,
            "sweeps": sweeps,
            "numpy_median_s": numpy_row["median_solve_s"],
        }
        if gpu_available:
            gpu_row = _time_tier(multi._gpu_gs_rb_sor_smooth, psi0, source, sweeps, repeats)
            ref = numpy_row["solution"]
            delta = float(np.linalg.norm(gpu_row["solution"] - ref))
            row["gpu_median_s"] = gpu_row["median_solve_s"]
            row["gpu_over_numpy_speedup"] = float(
                numpy_row["median_solve_s"] / max(gpu_row["median_solve_s"], 1e-12)
            )
            row["gpu_vs_numpy_rel_l2"] = delta / max(float(np.linalg.norm(ref)), 1e-30)
        rows.append(row)

    return {
        "schema": SCHEMA,
        "kernel": "gs_rb_sor_smooth",
        "seed": BENCH_SEED,
        "repeats": repeats,
        "omega": OMEGA,
        "gpu_available": gpu_available,
        "gpu_adapter": gpu_info,
        "grids": rows,
        "problem_contract": (
            "identical fixed-sweep Red-Black SOR of the toroidal GS* operator "
            "d2/dR2 - (1/R) d/dR + d2/dZ2 on a seeded Gaussian-source problem; "
            "GPU tier computes in f32, so cross-tier agreement is f32-bounded"
        ),
        "benchmark_evidence": {
            "classification": "local_non_isolated_regression",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GPU GS smoother benchmark and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, nargs="+", default=list(DEFAULT_GRIDS))
    parser.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    report = build_report(grids=args.grids, sweeps=args.sweeps, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
