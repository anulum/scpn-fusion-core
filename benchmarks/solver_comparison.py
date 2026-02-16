#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Python Solver Comparison Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""
Time FusionKernel with SOR, Newton-Kantorovich, and (optionally)
Rust multigrid on 33x33 and 65x65 grids. Outputs markdown table.

Usage:
    python benchmarks/solver_comparison.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from scpn_fusion.core.fusion_kernel import FusionKernel


def _benchmark_sor(nr: int, nz: int, max_iter: int = 200) -> dict[str, object]:
    config = {
        "reactor_name": f"bench_sor_{nr}x{nz}",
        "grid_resolution": [nr, nz],
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.5, "Z_max": 1.5},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"R": 3.5, "Z": 1.0, "current": 5e5},
            {"R": 3.5, "Z": -1.0, "current": 5e5},
        ],
        "solver": {
            "max_iterations": max_iter,
            "convergence_threshold": 1e-6,
            "relaxation_factor": 1.8,
        },
    }
    kernel = FusionKernel(config)
    t0 = time.perf_counter()
    kernel.solve()
    dt = time.perf_counter() - t0
    return {
        "solver": "SOR",
        "grid": f"{nr}x{nz}",
        "iterations": kernel.convergence_history[-1] if kernel.convergence_history else max_iter,
        "wall_time_ms": dt * 1000,
    }


def _benchmark_newton(nr: int, nz: int) -> dict[str, object]:
    config = {
        "reactor_name": f"bench_newton_{nr}x{nz}",
        "grid_resolution": [nr, nz],
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.5, "Z_max": 1.5},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"R": 3.5, "Z": 1.0, "current": 5e5},
            {"R": 3.5, "Z": -1.0, "current": 5e5},
        ],
        "solver": {
            "max_iterations": 200,
            "convergence_threshold": 1e-6,
            "relaxation_factor": 1.8,
        },
    }
    kernel = FusionKernel(config)

    # Check if Newton-Kantorovich solver is available
    if not hasattr(kernel, "solve_newton"):
        return {
            "solver": "Newton-K",
            "grid": f"{nr}x{nz}",
            "iterations": "N/A",
            "wall_time_ms": "N/A",
        }

    t0 = time.perf_counter()
    kernel.solve_newton()
    dt = time.perf_counter() - t0
    return {
        "solver": "Newton-K",
        "grid": f"{nr}x{nz}",
        "iterations": kernel.newton_iters if hasattr(kernel, "newton_iters") else "?",
        "wall_time_ms": dt * 1000,
    }


def main() -> None:
    results: list[dict[str, object]] = []

    for nr, nz in [(33, 33), (65, 65)]:
        print(f"\n--- Grid {nr}x{nz} ---")

        r = _benchmark_sor(nr, nz)
        results.append(r)
        print(f"  SOR:     {r['wall_time_ms']:.1f} ms  ({r['iterations']} iters)")

        r = _benchmark_newton(nr, nz)
        results.append(r)
        print(f"  Newton:  {r['wall_time_ms']} ms  ({r['iterations']} iters)")

    # Print markdown table
    print("\n### Python Solver Comparison\n")
    print("| Solver | Grid | Iterations | Wall Time (ms) |")
    print("|--------|------|-----------|---------------|")
    for r in results:
        wt = r["wall_time_ms"]
        wt_str = f"{wt:.1f}" if isinstance(wt, float) else str(wt)
        print(f"| {r['solver']} | {r['grid']} | {r['iterations']} | {wt_str} |")

    print("\n*Run on local machine. For CI numbers, see Criterion benchmarks.*")


if __name__ == "__main__":
    main()
