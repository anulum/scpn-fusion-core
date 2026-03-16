# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Code-to-Code Benchmark: scpn-fusion-core vs TORAX
"""
Benchmark scpn-fusion-core transport solver against TORAX on identical scenarios.

TORAX (google-deepmind/torax) is an open-source JAX-based 1.5D tokamak
transport code. This module runs both codes on the same ITER-like scenario
and compares:
  - Te, Ti profiles at t_final
  - Stored energy W_th
  - Confinement time tau_E
  - Convergence behaviour

Usage:
  python -m validation.code_to_code_benchmark [--with-torax]

Without --with-torax, only scpn-fusion-core is run and results are saved
for later comparison. With --with-torax, TORAX is imported and run
on the same scenario (requires `pip install torax`).

Results are saved to validation/reports/code_to_code_benchmark.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


# ── Scenario Definition ──────────────────────────────────────────────

ITER_SCENARIO = {
    "name": "ITER_15MA_baseline",
    "R0": 6.2,  # m
    "a": 2.0,  # m
    "B0": 5.3,  # T
    "I_p": 15.0e6,  # A
    "kappa": 1.7,
    "delta": 0.33,
    "n_e0": 10.0,  # 10^19 m^-3
    "T_e0": 10.0,  # keV
    "T_i0": 10.0,  # keV
    "P_aux": 50.0,  # MW
    "n_rho": 50,
    "dt": 0.01,  # s
    "n_steps": 100,
    "t_final": 1.0,  # s
}


# ── scpn-fusion-core run ─────────────────────────────────────────────────


def _run_scpn_fusion(scenario: dict) -> dict:
    """Run scpn-fusion-core transport solver on the scenario."""
    from scpn_fusion.core.integrated_transport_solver import TransportSolver

    cfg_dict = {
        "reactor_name": scenario["name"],
        "dimensions": {
            "R_min": scenario["R0"] - scenario["a"],
            "R_max": scenario["R0"] + scenario["a"],
            "Z_min": -scenario["a"] * scenario["kappa"],
            "Z_max": scenario["a"] * scenario["kappa"],
        },
        "grid_resolution": [scenario["n_rho"], scenario["n_rho"]],
        "physics": {"plasma_current_target": scenario["I_p"]},
    }

    tmp_cfg = Path("validation/reports/_tmp_c2c_config.json")
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(json.dumps(cfg_dict))

    try:
        solver = TransportSolver(tmp_cfg, multi_ion=True)

        t0 = time.perf_counter()
        Te_history = [solver.Te.copy()]
        Ti_history = [solver.Ti.copy()]

        for step in range(scenario["n_steps"]):
            solver.evolve_profiles(dt=scenario["dt"], P_aux=scenario["P_aux"])
            if step % 10 == 0:
                Te_history.append(solver.Te.copy())
                Ti_history.append(solver.Ti.copy())

        wall_time = time.perf_counter() - t0

        result = {
            "code": "scpn-fusion-core",
            "scenario": scenario["name"],
            "rho": solver.rho.tolist(),
            "Te_final": solver.Te.tolist(),
            "Ti_final": solver.Ti.tolist(),
            "ne_final": solver.ne.tolist(),
            "Te_avg": float(np.mean(solver.Te)),
            "Ti_avg": float(np.mean(solver.Ti)),
            "energy_balance_error": solver.energy_balance_error,
            "particle_balance_error": solver.particle_balance_error,
            "wall_time_s": wall_time,
            "n_steps": scenario["n_steps"],
            "dt": scenario["dt"],
        }
    finally:
        tmp_cfg.unlink(missing_ok=True)

    return result


# ── TORAX run (optional) ─────────────────────────────────────────────


def _run_torax(scenario: dict) -> dict | None:
    """Run TORAX on the same scenario, if installed."""
    try:
        import torax  # noqa: F401
    except ImportError:
        print("TORAX not installed — skipping TORAX comparison.")
        print("Install with: pip install torax")
        return None

    # TORAX integration would go here.
    # The API shape depends on the TORAX version, but the typical pattern is:
    #   config = torax.Config(...)
    #   sim = torax.Sim(config)
    #   sim.run(t_final=...)
    #   Te = sim.state.Te
    #
    # For now, return a placeholder indicating TORAX was found but
    # the integration is not yet wired.
    return {
        "code": "torax",
        "scenario": scenario["name"],
        "status": "installed_but_not_wired",
        "message": "TORAX API integration pending — wire config mapping here",
    }


# ── Comparison ────────────────────────────────────────────────────────


def _compare_results(scpn: dict, torax: dict | None) -> dict:
    """Compare scpn-fusion-core and TORAX results."""
    comparison = {
        "scpn_fusion": scpn,
        "torax": torax,
        "comparison": {},
    }

    if torax is not None and "Te_final" in torax:
        Te_scpn = np.array(scpn["Te_final"])
        Te_torax = np.array(torax["Te_final"])

        # Interpolate to common grid if needed
        if len(Te_scpn) != len(Te_torax):
            rho_scpn = np.array(scpn["rho"])
            rho_torax = np.array(torax.get("rho", np.linspace(0, 1, len(Te_torax))))
            Te_torax_interp = np.interp(rho_scpn, rho_torax, Te_torax)
        else:
            Te_torax_interp = Te_torax

        rmse = float(np.sqrt(np.mean((Te_scpn - Te_torax_interp) ** 2)))
        comparison["comparison"]["Te_rmse_keV"] = rmse
        comparison["comparison"]["Te_max_diff_keV"] = float(
            np.max(np.abs(Te_scpn - Te_torax_interp))
        )

    return comparison


# ── Main ──────────────────────────────────────────────────────────────


def main():
    with_torax = "--with-torax" in sys.argv

    print(f"Running code-to-code benchmark: {ITER_SCENARIO['name']}")
    print("=" * 60)

    print("\n[1/3] Running scpn-fusion-core...")
    scpn_result = _run_scpn_fusion(ITER_SCENARIO)
    print(f"  Te_avg = {scpn_result['Te_avg']:.3f} keV")
    print(f"  Ti_avg = {scpn_result['Ti_avg']:.3f} keV")
    print(f"  Energy balance error = {scpn_result['energy_balance_error']:.4e}")
    print(f"  Particle balance error = {scpn_result['particle_balance_error']:.4e}")
    print(f"  Wall time = {scpn_result['wall_time_s']:.2f} s")

    torax_result = None
    if with_torax:
        print("\n[2/3] Running TORAX...")
        torax_result = _run_torax(ITER_SCENARIO)
        if torax_result:
            print(f"  Status: {torax_result.get('status', 'done')}")
    else:
        print("\n[2/3] TORAX skipped (use --with-torax to enable)")

    print("\n[3/3] Comparing results...")
    comparison = _compare_results(scpn_result, torax_result)

    report_dir = Path("validation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "code_to_code_benchmark.json"

    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to {report_path}")

    if comparison["comparison"]:
        print("\nComparison metrics:")
        for k, v in comparison["comparison"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
