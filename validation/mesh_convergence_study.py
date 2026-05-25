# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GS Solver Mesh Convergence Study
"""
Mesh convergence study for the Grad-Shafranov elliptic solver.

Runs the Solov'ev analytic benchmark at multiple grid resolutions to
determine the spatial order of accuracy.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

REPORT_DIR = Path("validation/reports")
REPORT_JSON = REPORT_DIR / "mesh_convergence.json"
REPORT_MD = REPORT_DIR / "mesh_convergence.md"
DEFAULT_RESOLUTIONS = (17, 33, 65, 129)
DEFAULT_MIN_CONVERGENCE_RATE = 1.8


def run_solovev_benchmark(nr: int, nz: int, max_iter: int = 25000, tol: float = 1e-10) -> dict:
    """Run Solov'ev benchmark on a nr x nz grid."""
    R_min, R_max = 1.0, 3.0
    Z_min, Z_max = -1.5, 1.5

    R = np.linspace(R_min, R_max, nr)
    Z = np.linspace(Z_min, Z_max, nz)
    dR = (R_max - R_min) / (nr - 1)
    dZ = (Z_max - Z_min) / (nz - 1)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")

    # Solov'ev solution: psi = c1*R^4/8 + c2*Z^2
    c1, c2 = 1.0, 0.5
    psi_exact = c1 * RR**4 / 8.0 + c2 * ZZ**2

    # Source: Delta* psi = c1*R^2 + 2*c2
    source = c1 * RR**2 + 2.0 * c2

    # Initialise psi with zeros, apply Dirichlet BCs from exact solution
    psi = np.zeros((nr, nz))
    psi[0, :] = psi_exact[0, :]
    psi[-1, :] = psi_exact[-1, :]
    psi[:, 0] = psi_exact[:, 0]
    psi[:, -1] = psi_exact[:, -1]

    dR2 = dR**2
    dZ2 = dZ**2

    # Conservative omega for stability.
    # For large grids in Python, Gauss-Seidel (1.0) or low SOR (1.2) is safer.
    omega = 1.2

    t0 = time.perf_counter()
    final_iter = 0
    for k in range(max_iter):
        final_iter = k
        # SOR iteration (vectorized by row to speed up Python)
        for i in range(1, nr - 1):
            R_i = R[i]
            a_E = 1.0 / dR2 - 1.0 / (2.0 * R_i * dR)
            a_W = 1.0 / dR2 + 1.0 / (2.0 * R_i * dR)
            a_NS = 1.0 / dZ2
            a_C = 2.0 / dR2 + 2.0 / dZ2

            gs = (
                a_E * psi[i + 1, 1:-1]
                + a_W * psi[i - 1, 1:-1]
                + a_NS * (psi[i, 2:] + psi[i, :-2])
                - source[i, 1:-1]
            ) / a_C
            psi[i, 1:-1] += omega * (gs - psi[i, 1:-1])

        if k % 200 == 0:
            res = (
                (1.0 / dR2 - 1.0 / (2.0 * RR[1:-1, 1:-1] * dR)) * psi[2:, 1:-1]
                + (1.0 / dR2 + 1.0 / (2.0 * RR[1:-1, 1:-1] * dR)) * psi[:-2, 1:-1]
                + (1.0 / dZ2) * (psi[1:-1, 2:] + psi[1:-1, :-2])
                - (2.0 / dR2 + 2.0 / dZ2) * psi[1:-1, 1:-1]
                - source[1:-1, 1:-1]
            )
            if np.max(np.abs(res)) < tol:
                break
            if np.any(np.isnan(psi)):
                break

    wall_time = time.perf_counter() - t0

    # Compute error (NRMSE on interior)
    interior = psi[1:-1, 1:-1]
    exact_int = psi_exact[1:-1, 1:-1]
    abs_err = np.abs(interior - exact_int)
    rmse = np.sqrt(np.mean(abs_err**2))
    nrmse = rmse / (np.max(psi_exact) - np.min(psi_exact))

    return {
        "nr": nr,
        "nz": nz,
        "h": dR,
        "nrmse": float(nrmse),
        "rmse": float(rmse),
        "max_err": float(np.max(abs_err)),
        "wall_time_s": wall_time,
        "iterations": final_iter + 1,
    }


def add_convergence_rates(results: list[dict]) -> list[dict]:
    """Return benchmark rows annotated with adjacent-grid convergence rates."""
    rated = [dict(row) for row in results]
    for row in rated:
        h = float(row["h"])
        nrmse = float(row["nrmse"])
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("convergence rows require strictly decreasing positive h")
        if not np.isfinite(nrmse) or nrmse <= 0.0:
            raise ValueError("convergence rows require positive finite nrmse")
    for i in range(1, len(rated)):
        h_ratio = rated[i - 1]["h"] / rated[i]["h"]
        if not np.isfinite(h_ratio) or h_ratio <= 1.0:
            raise ValueError("convergence rows require strictly decreasing positive h")
        err_ratio = rated[i - 1]["nrmse"] / rated[i]["nrmse"]
        rated[i]["convergence_rate"] = float(np.log(err_ratio) / np.log(h_ratio))
    return rated


def summarise_convergence_contract(
    results: list[dict], min_rate: float = DEFAULT_MIN_CONVERGENCE_RATE
) -> dict:
    """Summarise whether the manufactured GS solve is at least second-order."""
    rates = [
        float(row["convergence_rate"])
        for row in results
        if "convergence_rate" in row and np.isfinite(row["convergence_rate"])
    ]
    min_observed = min(rates) if rates else float("nan")
    passed = bool(rates) and min_observed >= min_rate
    return {
        "contract": "fixed-boundary manufactured Solov'ev GS solve shows second-order mesh convergence",
        "passed": passed,
        "required_min_rate": float(min_rate),
        "min_convergence_rate": float(min_observed),
        "rated_grid_count": len(rates),
    }


def _machine_context() -> dict:
    return {
        "platform": platform.platform(),
        "cpu_count": int(os.cpu_count() or 0),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def main() -> int:
    resolutions = list(DEFAULT_RESOLUTIONS)
    results = []

    print(f"{'Grid':<10} {'h':<12} {'NRMSE':<12} {'Time (s)':<10} {'Iters':<10}")
    print("-" * 60)

    for res in resolutions:
        res_data = run_solovev_benchmark(res, res)
        results.append(res_data)
        print(
            f"{res}x{res:<5} {res_data['h']:<12.4e} {res_data['nrmse']:<12.4e} "
            f"{res_data['wall_time_s']:<10.4f} {res_data['iterations']:<10}"
        )

    results = add_convergence_rates(results)
    summary = summarise_convergence_contract(results)
    machine = _machine_context()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_JSON, "w") as f:
        json.dump({"summary": summary, "machine": machine, "results": results}, f, indent=2)
        f.write("\n")

    with open(REPORT_MD, "w") as f:
        f.write("# GS Solver Mesh Convergence Study\n\n")
        f.write(
            "Determines the spatial order of accuracy for the Grad-Shafranov elliptic solver.\n\n"
        )
        f.write("## Contract\n\n")
        f.write(f"- Contract: {summary['contract']}\n")
        f.write(f"- Status: {'PASS' if summary['passed'] else 'FAIL'}\n")
        f.write(f"- Required minimum adjacent-grid rate: {summary['required_min_rate']:.2f}\n")
        f.write(f"- Minimum observed adjacent-grid rate: {summary['min_convergence_rate']:.6f}\n")
        f.write(f"- Rated grid transitions: {summary['rated_grid_count']}\n\n")
        f.write("## Machine\n\n")
        f.write(f"- Platform: {machine['platform']}\n")
        f.write(f"- CPU count: {machine['cpu_count']}\n")
        f.write(f"- Python: {machine['python']}\n")
        f.write(f"- NumPy: {machine['numpy']}\n\n")
        f.write("## Results\n\n")
        f.write("| Grid | h | NRMSE | Rate | Time (s) | Iters |\n")
        f.write("|------|---|-------|------|----------|-------|\n")
        for r in results:
            rate_str = f"{r['convergence_rate']:.2f}" if "convergence_rate" in r else "N/A"
            f.write(
                f"| {r['nr']}x{r['nz']} | {r['h']:.4e} | {r['nrmse']:.4e} | "
                f"{rate_str} | {r['wall_time_s']:.4f} | {r['iterations']} |\n"
            )

    print(f"\nContract status: {'PASS' if summary['passed'] else 'FAIL'}")
    print(f"Results saved to {REPORT_JSON} and {REPORT_MD}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
