# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Solver Benchmark: SOR vs Multigrid vs Python
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
3-way benchmark comparing inner linear solver performance:

1. Rust Picard+SOR (production default)
2. Rust Picard+Multigrid (newly wired)
3. Python Picard+Jacobi (pure NumPy baseline)

Runs 50 synthetic equilibrium solves with each method and reports
per-shot timing, RMSE consistency, and speedup factors.

Requires: scpn_fusion_rs (maturin develop --release)
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent


def find_config() -> str:
    """Locate a suitable config file for benchmarking."""
    candidates = [
        BASE / "validation" / "iter_validated_config.json",
        BASE / "iter_config.json",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    print("ERROR: No config file found. Tried:", [str(c) for c in candidates])
    sys.exit(1)


def benchmark_rust_solver(config_path: str, method: str, n_shots: int) -> dict:
    """Run n_shots solves with the specified Rust solver method."""
    from scpn_fusion_rs import PyFusionKernel

    times_ms = []
    residuals = []
    converged_count = 0

    for _ in range(n_shots):
        k = PyFusionKernel(config_path)
        k.set_solver_method(method)

        t0 = time.perf_counter()
        result = k.solve_equilibrium()
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)
        residuals.append(result.residual)
        if result.converged:
            converged_count += 1

    return {
        "method": f"Rust {method}",
        "shots": n_shots,
        "converged": converged_count,
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "p95_ms": sorted(times_ms)[int(0.95 * len(times_ms))],
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "mean_residual": statistics.mean(residuals),
        "times_ms": times_ms,
    }


def benchmark_python_solver(config_path: str, n_shots: int) -> dict:
    """Run n_shots solves with the pure-Python FusionKernel."""
    try:
        from scpn_fusion.core.fusion_kernel import FusionKernel
    except ImportError:
        return {
            "method": "Python Jacobi",
            "shots": 0,
            "converged": 0,
            "mean_ms": float("nan"),
            "median_ms": float("nan"),
            "p95_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "mean_residual": float("nan"),
            "times_ms": [],
            "note": "scpn_fusion.core.fusion_kernel not importable",
        }

    times_ms = []
    residuals = []
    converged_count = 0

    for _ in range(n_shots):
        k = FusionKernel(config_path)

        t0 = time.perf_counter()
        result = k.solve_equilibrium()
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)
        if hasattr(result, "residual"):
            residuals.append(result.residual)
        if hasattr(result, "converged") and result.converged:
            converged_count += 1

    return {
        "method": "Python Jacobi",
        "shots": n_shots,
        "converged": converged_count,
        "mean_ms": statistics.mean(times_ms) if times_ms else float("nan"),
        "median_ms": statistics.median(times_ms) if times_ms else float("nan"),
        "p95_ms": sorted(times_ms)[int(0.95 * len(times_ms))] if times_ms else float("nan"),
        "min_ms": min(times_ms) if times_ms else float("nan"),
        "max_ms": max(times_ms) if times_ms else float("nan"),
        "mean_residual": statistics.mean(residuals) if residuals else float("nan"),
        "times_ms": times_ms,
    }


def print_results(results: list[dict]):
    """Pretty-print benchmark comparison table."""
    print()
    print("=" * 80)
    print("  SCPN Fusion Core — Solver Benchmark Results")
    print("=" * 80)
    print()

    header = f"  {'Method':20s} {'Shots':>6s} {'Conv':>5s} {'Mean(ms)':>10s} {'Median(ms)':>11s} {'P95(ms)':>9s} {'Min(ms)':>9s} {'Max(ms)':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        print(
            f"  {r['method']:20s} "
            f"{r['shots']:6d} "
            f"{r['converged']:5d} "
            f"{r['mean_ms']:10.1f} "
            f"{r['median_ms']:11.1f} "
            f"{r['p95_ms']:9.1f} "
            f"{r['min_ms']:9.1f} "
            f"{r['max_ms']:9.1f}"
        )

    # Speedup comparison
    if len(results) >= 2 and results[0]["mean_ms"] > 0:
        baseline = results[0]["mean_ms"]
        print()
        print("  Speedup vs Rust SOR:")
        for r in results[1:]:
            if r["mean_ms"] > 0 and not np.isnan(r["mean_ms"]):
                speedup = r["mean_ms"] / baseline
                faster = baseline / r["mean_ms"]
                if faster >= 1.0:
                    print(f"    {r['method']:20s} → {faster:.1f}x faster")
                else:
                    print(f"    {r['method']:20s} → {1.0/faster:.1f}x slower")

    print()
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark SOR vs Multigrid solvers")
    parser.add_argument("--shots", type=int, default=50, help="Number of shots per method")
    parser.add_argument("--config", type=str, default=None, help="Config JSON path")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python baseline")
    parser.add_argument("--json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    config_path = args.config or find_config()
    n_shots = args.shots

    print(f"Config: {config_path}")
    print(f"Shots per method: {n_shots}")
    print()

    results = []

    # 1. Rust SOR
    print("Running Rust Picard+SOR...")
    results.append(benchmark_rust_solver(config_path, "sor", n_shots))

    # 2. Rust Multigrid
    print("Running Rust Picard+Multigrid...")
    results.append(benchmark_rust_solver(config_path, "multigrid", n_shots))

    # 3. Python baseline
    if not args.skip_python:
        # Use fewer shots for Python (it's much slower)
        py_shots = min(n_shots, 5)
        print(f"Running Python Picard+Jacobi ({py_shots} shots)...")
        results.append(benchmark_python_solver(config_path, py_shots))

    print_results(results)

    # Save JSON summary
    if args.json:
        summary = []
        for r in results:
            s = {k: v for k, v in r.items() if k != "times_ms"}
            summary.append(s)
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
