# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GS Solver Mesh Convergence Study
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""
Mesh convergence study for the Grad-Shafranov elliptic solver.

Runs the Solov'ev analytic benchmark at multiple grid resolutions to
determine the spatial order of accuracy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


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


def main():
    resolutions = [17, 33, 65, 129]
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

    # Compute convergence rates
    for i in range(1, len(results)):
        h_ratio = results[i - 1]["h"] / results[i]["h"]
        err_ratio = results[i - 1]["nrmse"] / results[i]["nrmse"]
        rate = np.log(err_ratio) / np.log(h_ratio)
        results[i]["convergence_rate"] = float(rate)

    report_dir = Path("validation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "mesh_convergence.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    md_path = report_dir / "mesh_convergence.md"
    with open(md_path, "w") as f:
        f.write("# GS Solver Mesh Convergence Study\n\n")
        f.write(
            "Determines the spatial order of accuracy for the Grad-Shafranov elliptic solver.\n\n"
        )
        f.write("| Grid | h | NRMSE | Rate | Time (s) | Iters |\n")
        f.write("|------|---|-------|------|----------|-------|\n")
        for r in results:
            rate_str = f"{r['convergence_rate']:.2f}" if "convergence_rate" in r else "N/A"
            f.write(
                f"| {r['nr']}x{r['nz']} | {r['h']:.4e} | {r['nrmse']:.4e} | "
                f"{rate_str} | {r['wall_time_s']:.4f} | {r['iterations']} |\n"
            )

    print(f"\nResults saved to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
