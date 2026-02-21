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
import json
import tempfile
import os

from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.neural_equilibrium import NeuralEquilibriumKernel


def _benchmark_sor(nr: int, nz: int, max_iter: int = 200) -> dict[str, object]:
    config = {
        "reactor_name": f"bench_sor_{nr}x{nz}",
        "grid_resolution": [nr, nz],
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.5, "Z_max": 1.5},
        "physics": {"plasma_current_target": 1.0e6},
        "coils": [
            {"name": "PF1", "r": 3.5, "z": 1.0, "current": 5e5},
            {"name": "PF2", "r": 3.5, "z": -1.0, "current": 5e5},
            {"name": "PF3", "r": 1.0, "z": 1.0, "current": 5e5},
            {"name": "PF4", "r": 1.0, "z": -1.0, "current": 5e5},
            {"name": "PF5", "r": 2.2, "z": 2.0, "current": 5e5},
        ],
        "solver": {
            "max_iterations": max_iter,
            "convergence_threshold": 1e-6,
            "relaxation_factor": 0.5,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=".") as f:
        json.dump(config, f)
        tmp_path = f.name
    
    try:
        kernel = FusionKernel(tmp_path)
        t0 = time.perf_counter()
        kernel.solve_equilibrium()
        dt = time.perf_counter() - t0
        return {
            "solver": "SOR",
            "grid": f"{nr}x{nz}",
            "iterations": kernel.cfg["solver"]["max_iterations"],
            "wall_time_ms": dt * 1000,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _benchmark_newton(nr: int, nz: int) -> dict[str, object]:
    config = {
        "reactor_name": f"bench_newton_{nr}x{nz}",
        "grid_resolution": [nr, nz],
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.5, "Z_max": 1.5},
        "physics": {"plasma_current_target": 1.0e6},
        "coils": [
            {"name": "PF1", "r": 3.5, "z": 1.0, "current": 5e5},
            {"name": "PF2", "r": 3.5, "z": -1.0, "current": 5e5},
            {"name": "PF3", "r": 1.0, "z": 1.0, "current": 5e5},
            {"name": "PF4", "r": 1.0, "z": -1.0, "current": 5e5},
            {"name": "PF5", "r": 2.2, "z": 2.0, "current": 5e5},
        ],
        "solver": {
            "solver_method": "newton",
            "max_iterations": 20, # Newton is expensive per iter
            "convergence_threshold": 1e-6,
            "relaxation_factor": 0.5,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=".") as f:
        json.dump(config, f)
        tmp_path = f.name

    try:
        kernel = FusionKernel(tmp_path)
        t0 = time.perf_counter()
        res = kernel.solve_equilibrium()
        dt = time.perf_counter() - t0
        return {
            "solver": "Newton-K",
            "grid": f"{nr}x{nz}",
            "iterations": res.get("iterations", "?"),
            "wall_time_ms": dt * 1000,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _benchmark_neural(nr: int, nz: int) -> dict[str, object]:
    # Neural kernel uses fixed grid (129x129 usually in weights)
    # but the API allows it to be used as a drop-in.
    config_path = Path(__file__).resolve().parents[1] / "iter_config.json"
    if not config_path.exists():
        # Create a temp config if needed, or use existing one
        config_path = Path("iter_config.json")
    
    try:
        kernel = NeuralEquilibriumKernel(config_path)
        t0 = time.perf_counter()
        kernel.solve_equilibrium()
        dt = time.perf_counter() - t0
        return {
            "solver": "Neural (MLP)",
            "grid": f"{kernel.accel.cfg.grid_shape[1]}x{kernel.accel.cfg.grid_shape[0]}",
            "iterations": 1,
            "wall_time_ms": dt * 1000,
        }
    except Exception as exc:
        print(f"  Neural Error: {exc}")
        return {
            "solver": "Neural (MLP)",
            "grid": "N/A",
            "iterations": "N/A",
            "wall_time_ms": "N/A",
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
        print(f"  Newton:  {r['wall_time_ms']:.1f} ms  ({r['iterations']} iters)")

    print("\n--- Neural Surrogate ---")
    r = _benchmark_neural(129, 129)
    results.append(r)
    wt = r['wall_time_ms']
    wt_str = f"{wt:.3f}" if isinstance(wt, float) else str(wt)
    print(f"  Neural:  {wt_str} ms  ({r['iterations']} iters)")


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
