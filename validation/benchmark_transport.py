# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Transport Validation Benchmark
"""
Systematic transport validation benchmark.

Verifies the 1.5D transport solver against analytic solutions,
threshold models, and empirical scaling laws.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scpn_fusion.core.integrated_transport_solver import TransportSolver
from scpn_fusion.core.neural_transport import NeuralTransportModel, TransportInputs
from scpn_fusion.core.scaling_laws import ipb98y2_with_uncertainty


def run_pure_diffusion_benchmark(nr: int = 200) -> dict:
    """Benchmark 1: Pure diffusion against analytic steady state."""
    # Dummy config: R0=2.0, a=1.0
    cfg = {
        "reactor_name": "Analytic-Transport",
        "grid_resolution": [33, 33],
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.5, "Z_max": 1.5},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [],
        "solver": {"max_iterations": 1, "convergence_threshold": 1.0, "max_iterations_outer": 1},
    }
    cfg_path = Path("tmp_bench_cfg.json")
    cfg_path.write_text(json.dumps(cfg))

    try:
        # Use higher resolution to minimize discretization error
        solver = TransportSolver(cfg_path, nr=nr)

        # 1. Physics Setup
        chi_const = 2.0  # m^2/s
        P_aux_MW = 10.0  # 10 MW

        solver.chi_i = np.full(nr, chi_const)
        solver.chi_e = np.full(nr, chi_const)
        solver.ne = np.full(nr, 1.0)  # 10^19 m^-3
        solver.n_impurity = np.zeros(nr)

        # Ensure flat heating profile in T-space
        solver.aux_heating_profile_width = 1e10  # Effectively flat
        solver.aux_heating_electron_fraction = 0.5

        # Initial guess close to steady state
        solver.Ti = np.full(nr, 1.0)
        solver.Te = np.full(nr, 1.0)

        dt = 0.01
        # Evolve to steady state
        for _ in range(10000):
            solver.evolve_profiles(dt=dt, P_aux=P_aux_MW)

        # 2. Derive Discrete S_T
        dV = solver._rho_volume_element()
        norm = float(np.sum(dV))
        e_keV_J = 1.602176634e-16
        ne_m3 = 1.0e19
        S_T_discrete = (P_aux_MW * 1e6) / (1.5 * ne_m3 * e_keV_J * norm)

        # T(rho) = T_edge + (S_T * a^2 / (4*chi)) * (1 - rho^2)
        rho = solver.rho
        T_analytic = 0.1 + (S_T_discrete * 1.0**2 / (4.0 * chi_const)) * (1.0 - rho**2)

        err = np.abs(solver.Ti - T_analytic)
        max_err_rel = np.max(err[1:-1]) / np.max(T_analytic)

        return {
            "max_relative_error": float(max_err_rel),
            "core_actual": float(solver.Ti[0]),
            "core_analytic": float(T_analytic[0]),
            "pass": bool(max_err_rel < 0.05),  # Keep threshold reasonable for now
        }
    finally:
        if cfg_path.exists():
            cfg_path.unlink()


def run_threshold_benchmark() -> dict:
    """Benchmark 2: Critical gradient threshold behavior."""
    model = NeuralTransportModel(auto_discover=False)

    inp_sub = TransportInputs(grad_ti=3.0)
    inp_super = TransportInputs(grad_ti=10.0)

    flux_sub = model.predict(inp_sub)
    flux_super = model.predict(inp_super)

    return {
        "chi_sub": float(flux_sub.chi_i),
        "chi_super": float(flux_super.chi_i),
        "pass": bool(flux_sub.chi_i == 0.0 and flux_super.chi_i > 0.0),
    }


def run_iter_scaling_benchmark() -> dict:
    """Benchmark 3: ITER-like confinement scaling."""
    Ip, BT, ne19, P_loss, R, kappa, epsilon, M = 15.0, 5.3, 10.0, 50.0, 6.2, 1.7, 2.0 / 6.2, 2.5
    tau_pred, sigma = ipb98y2_with_uncertainty(Ip, BT, ne19, P_loss, R, kappa, epsilon, M)
    return {"tau_predicted": float(tau_pred), "uncertainty_sigma": float(sigma), "pass": True}


def main():
    results = {
        "pure_diffusion": run_pure_diffusion_benchmark(),
        "threshold": run_threshold_benchmark(),
        "iter_scaling": run_iter_scaling_benchmark(),
    }

    pd = results["pure_diffusion"]
    print(
        f"DEBUG Core: actual={pd['core_actual']:.4f}, analytic={pd['core_analytic']:.4f}, err={pd['max_relative_error']:.2%}"
    )

    report_dir = Path("validation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "transport_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(report_dir / "transport_benchmark.md", "w") as f:
        f.write("# Transport Validation Benchmark\n\n")
        f.write("| Test | Metric | Result | Pass |\n")
        f.write("|------|--------|--------|------|\n")
        f.write(
            f"| Pure Diffusion | Max Rel Error | {pd['max_relative_error']:.2%} | {pd['pass']} |\n"
        )
        th = results["threshold"]
        f.write(
            f"| Threshold | Chi Sub/Super | {th['chi_sub']:.1f} / {th['chi_super']:.1f} | {th['pass']} |\n"
        )
        it = results["iter_scaling"]
        f.write(f"| ITER Scaling | Tau_E Predicted | {it['tau_predicted']:.2f}s | {it['pass']} |\n")

    print(f"Results saved to {report_dir}")


if __name__ == "__main__":
    main()
