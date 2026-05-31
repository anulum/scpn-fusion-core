#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed electromagnetic nonlinear GK fidelity gate.

The native nonlinear GK path currently implements compact algebraic
``A_parallel`` and ``B_parallel`` closures. This benchmark verifies those
closures and their field-energy histories separately from electrostatic GK
runs, then keeps full Vlasov-Maxwell parity blocked until same-deck
electromagnetic GENE/CGYRO/GS2 outputs and Maxwell evolution evidence exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver

REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "gk_electromagnetic_fidelity.json"
MD_REPORT = REPORT_DIR / "gk_electromagnetic_fidelity.md"
REQUIRED_EXTERNAL_SOLVERS = ["GENE", "CGYRO", "GS2"]
REQUIRED_EXTERNAL_OBSERVABLES = [
    "electromagnetic_phi_energy",
    "electromagnetic_apar_energy",
    "electromagnetic_bpar_energy",
    "ion_heat_flux_spectrum",
    "electron_heat_flux_spectrum",
    "nonlinear_distribution_function",
    "nonlinear_distribution_function_imag",
]
OMITTED_PHYSICS = [
    "Faraday induction equation for evolving B",
    "displacement-current Ampere-Maxwell evolution",
    "self-consistent inductive parallel electric field evolution",
    "external same-deck electromagnetic GENE/CGYRO/GS2 output parity",
]
GRID_CONVERGENCE_CASES = [
    {"case_id": "compact_em_4x4x8", "n_kx": 4, "n_ky": 4, "n_theta": 8},
    {"case_id": "compact_em_6x6x10", "n_kx": 6, "n_ky": 6, "n_theta": 10},
    {"case_id": "compact_em_8x8x12", "n_kx": 8, "n_ky": 8, "n_theta": 12},
]
GRID_CONVERGENCE_RELATIVE_ENERGY_TOLERANCE = 5.0e-1


def _maxwell_evolution_contract(*, compact_closure_ready: bool) -> dict[str, Any]:
    """Return the fail-closed full-Maxwell equation implementation contract."""
    equations = [
        {
            "compact_closure_ready": False,
            "equation_id": "faraday_induction",
            "equation": "dB/dt = -curl(E)",
            "implemented_by_native_solver": False,
            "native_status": "missing_time_evolved_magnetic_field",
            "required_for_full_vlasov_maxwell": True,
        },
        {
            "compact_closure_ready": False,
            "equation_id": "ampere_maxwell_displacement_current",
            "equation": "curl(B) = mu0 J + mu0 eps0 dE/dt",
            "implemented_by_native_solver": False,
            "native_status": "missing_displacement_current_evolution",
            "required_for_full_vlasov_maxwell": True,
        },
        {
            "compact_closure_ready": False,
            "equation_id": "inductive_parallel_electric_field",
            "equation": "E_parallel = -grad_parallel(phi) - dA_parallel/dt",
            "implemented_by_native_solver": False,
            "native_status": "missing_self_consistent_inductive_parallel_e_field",
            "required_for_full_vlasov_maxwell": True,
        },
        {
            "compact_closure_ready": compact_closure_ready,
            "equation_id": "compact_parallel_ampere_closure",
            "equation": "A_parallel from compact parallel Ampere closure",
            "implemented_by_native_solver": compact_closure_ready,
            "native_status": "implemented_as_algebraic_closure_not_maxwell_evolution",
            "required_for_full_vlasov_maxwell": False,
        },
        {
            "compact_closure_ready": compact_closure_ready,
            "equation_id": "compact_perpendicular_pressure_balance_closure",
            "equation": "B_parallel from perpendicular pressure-balance closure",
            "implemented_by_native_solver": compact_closure_ready,
            "native_status": "implemented_as_algebraic_closure_not_maxwell_evolution",
            "required_for_full_vlasov_maxwell": False,
        },
    ]
    blocking_equation_ids = [
        str(equation["equation_id"])
        for equation in equations
        if equation["required_for_full_vlasov_maxwell"]
        and not equation["implemented_by_native_solver"]
    ]
    return {
        "blocking_equation_ids": blocking_equation_ids,
        "equations": equations,
        "full_vlasov_maxwell_parity_ready": not blocking_equation_ids,
        "native_field_evolution_mode": "compact_algebraic_Apar_Bpar_closure",
    }


def _config(*, electromagnetic: bool) -> NonlinearGKConfig:
    return NonlinearGKConfig(
        n_kx=4,
        n_ky=4,
        n_theta=8,
        n_vpar=5,
        n_mu=4,
        n_species=2,
        kinetic_electrons=True,
        electromagnetic=electromagnetic,
        nonlinear=True,
        collisions=False,
        hyper_coeff=0.0,
        dt=0.005,
        n_steps=4,
        save_interval=1,
        cfl_adapt=False,
    )


def _run_gate(*, electromagnetic: bool, seed: int) -> dict[str, Any]:
    cfg = _config(electromagnetic=electromagnetic)
    solver = NonlinearGKSolver(cfg)
    result = solver.run(solver.init_state(amplitude=1e-5, seed=seed))
    total_closes = bool(
        np.allclose(
            result.total_energy_t,
            result.particle_free_energy_t
            + result.phi_energy_t
            + result.A_parallel_energy_t
            + result.B_parallel_energy_t,
        )
    )
    compact_closure_ready = bool(
        electromagnetic
        and result.compact_maxwell_closure_pass_t.size == result.time.size
        and bool(np.all(result.compact_maxwell_closure_pass_t))
        and bool(np.all(np.isfinite(result.maxwell_ampere_parallel_linf_residual_t)))
        and bool(np.all(np.isfinite(result.maxwell_pressure_balance_linf_residual_t)))
    )
    return {
        "A_parallel_energy_max": float(np.max(result.A_parallel_energy_t))
        if result.A_parallel_energy_t.size
        else 0.0,
        "B_parallel_energy_max": float(np.max(result.B_parallel_energy_t))
        if result.B_parallel_energy_t.size
        else 0.0,
        "compact_closure_ready": compact_closure_ready,
        "electromagnetic_enabled": electromagnetic,
        "field_energy_total_closes": total_closes,
        "full_faraday_displacement_current_supported": bool(
            result.full_faraday_displacement_current_supported
        ),
        "full_vlasov_maxwell_parity_ready": bool(result.full_vlasov_maxwell_parity_ready),
        "max_ampere_parallel_linf_residual": float(
            np.max(result.maxwell_ampere_parallel_linf_residual_t)
        )
        if result.maxwell_ampere_parallel_linf_residual_t.size
        else 0.0,
        "max_pressure_balance_linf_residual": float(
            np.max(result.maxwell_pressure_balance_linf_residual_t)
        )
        if result.maxwell_pressure_balance_linf_residual_t.size
        else 0.0,
        "saved_steps": int(result.time.size),
        "time_history_ready": bool(result.time.size > 0),
    }


def _grid_convergence_config(*, n_kx: int, n_ky: int, n_theta: int) -> NonlinearGKConfig:
    """Return a compact electromagnetic grid-refinement case."""
    return NonlinearGKConfig(
        n_kx=n_kx,
        n_ky=n_ky,
        n_theta=n_theta,
        n_vpar=5,
        n_mu=4,
        n_species=2,
        kinetic_electrons=True,
        electromagnetic=True,
        nonlinear=True,
        collisions=False,
        hyper_coeff=0.0,
        dt=0.0025,
        n_steps=5,
        save_interval=1,
        cfl_adapt=False,
        beta_e=0.02,
    )


def _relative_drift(values: np.ndarray) -> float:
    """Return max relative drift from the first finite history value."""
    if values.size == 0:
        return float("inf")
    finite_values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(finite_values)):
        return float("inf")
    baseline = max(abs(float(finite_values[0])), 1.0e-30)
    return float(np.max(np.abs(finite_values - finite_values[0])) / baseline)


def _run_grid_convergence_evidence() -> dict[str, Any]:
    """Run local compact-EM grid refinement without promoting Maxwell parity."""
    grid_rows: list[dict[str, Any]] = []
    for case in GRID_CONVERGENCE_CASES:
        cfg = _grid_convergence_config(
            n_kx=int(case["n_kx"]),
            n_ky=int(case["n_ky"]),
            n_theta=int(case["n_theta"]),
        )
        solver = NonlinearGKSolver(cfg)
        result = solver.run(solver.init_single_mode(kx_idx=0, ky_idx=1, amplitude=1.0e-5))
        total_field_energy = (
            result.phi_energy_t + result.A_parallel_energy_t + result.B_parallel_energy_t
        )
        total_energy = result.total_energy_t
        grid_rows.append(
            {
                "A_parallel_energy_final": float(result.A_parallel_energy_t[-1])
                if result.A_parallel_energy_t.size
                else 0.0,
                "B_parallel_energy_final": float(result.B_parallel_energy_t[-1])
                if result.B_parallel_energy_t.size
                else 0.0,
                "case_id": str(case["case_id"]),
                "compact_closure_ready": bool(
                    result.compact_maxwell_closure_pass_t.size == result.time.size
                    and np.all(result.compact_maxwell_closure_pass_t)
                ),
                "field_energy_total_closes": bool(
                    np.allclose(
                        total_energy,
                        result.particle_free_energy_t + total_field_energy,
                    )
                ),
                "grid_shape": {
                    "n_kx": cfg.n_kx,
                    "n_ky": cfg.n_ky,
                    "n_theta": cfg.n_theta,
                    "n_vpar": cfg.n_vpar,
                    "n_mu": cfg.n_mu,
                    "n_species": cfg.n_species,
                },
                "max_ampere_parallel_linf_residual": float(
                    np.max(result.maxwell_ampere_parallel_linf_residual_t)
                )
                if result.maxwell_ampere_parallel_linf_residual_t.size
                else float("inf"),
                "max_pressure_balance_linf_residual": float(
                    np.max(result.maxwell_pressure_balance_linf_residual_t)
                )
                if result.maxwell_pressure_balance_linf_residual_t.size
                else float("inf"),
                "phi_energy_final": float(result.phi_energy_t[-1])
                if result.phi_energy_t.size
                else 0.0,
                "relative_total_energy_drift": _relative_drift(result.total_energy_t),
                "saved_steps": int(result.time.size),
                "total_field_energy_final": float(total_field_energy[-1])
                if total_field_energy.size
                else 0.0,
            }
        )

    row_drifts = [float(row["relative_total_energy_drift"]) for row in grid_rows]
    field_energy_histories_finite = bool(
        grid_rows
        and all(
            np.isfinite(row["total_field_energy_final"])
            and np.isfinite(row["relative_total_energy_drift"])
            for row in grid_rows
        )
    )
    compact_closure_residuals_converged = bool(
        grid_rows
        and all(
            row["compact_closure_ready"]
            and row["max_ampere_parallel_linf_residual"] <= 1.0e-12
            and row["max_pressure_balance_linf_residual"] <= 1.0e-12
            for row in grid_rows
        )
    )
    field_energy_refinement_comparison_ready = bool(
        len(grid_rows) >= 3
        and all(row["field_energy_total_closes"] for row in grid_rows)
        and field_energy_histories_finite
    )
    max_relative_total_energy_drift = max(row_drifts) if row_drifts else float("inf")
    grid_convergence_ready = bool(
        compact_closure_residuals_converged
        and field_energy_refinement_comparison_ready
        and max_relative_total_energy_drift <= GRID_CONVERGENCE_RELATIVE_ENERGY_TOLERANCE
    )
    return {
        "compact_closure_residuals_converged": compact_closure_residuals_converged,
        "field_energy_histories_finite": field_energy_histories_finite,
        "field_energy_refinement_comparison_ready": field_energy_refinement_comparison_ready,
        "grid_case_count": len(grid_rows),
        "grid_convergence_ready": grid_convergence_ready,
        "grid_rows": grid_rows,
        "max_relative_total_energy_drift": max_relative_total_energy_drift,
        "relative_energy_tolerance": GRID_CONVERGENCE_RELATIVE_ENERGY_TOLERANCE,
        "schema": "gk-electromagnetic-grid-convergence.v1",
        "scope": "native_compact_Apar_Bpar_field_energy_histories",
        "status": "accepted_local_compact_em_grid_convergence"
        if grid_convergence_ready
        else "blocked_local_compact_em_grid_convergence_failed",
    }


def run_benchmark() -> dict[str, Any]:
    """Run the local compact-EM gate and return fail-closed parity status."""
    electrostatic_gate = _run_gate(electromagnetic=False, seed=211)
    electromagnetic_gate = _run_gate(electromagnetic=True, seed=223)
    compact_ready = bool(electromagnetic_gate["compact_closure_ready"])
    grid_convergence_evidence = _run_grid_convergence_evidence()
    grid_convergence_ready = bool(grid_convergence_evidence["grid_convergence_ready"])
    maxwell_contract = _maxwell_evolution_contract(compact_closure_ready=compact_ready)
    status = (
        "blocked_missing_full_vlasov_maxwell_field_solve"
        if compact_ready and grid_convergence_ready
        else "blocked_compact_electromagnetic_contract_failed"
    )
    return {
        "benchmark": "gk_electromagnetic_fidelity",
        "compact_em_contract_ready": compact_ready,
        "description": (
            "Separate electrostatic and electromagnetic nonlinear GK gate. "
            "Compact A_parallel/B_parallel diagnostics are local readiness "
            "evidence only, not full Vlasov-Maxwell parity."
        ),
        "electromagnetic_gate": electromagnetic_gate,
        "electromagnetic_grid_convergence_evidence": grid_convergence_evidence,
        "electromagnetic_grid_convergence_ready": grid_convergence_ready,
        "electrostatic_gate": electrostatic_gate,
        "external_em_parity_comparison_ready": False,
        "locally_actionable_contract_ready": compact_ready and grid_convergence_ready,
        "maxwell_evolution_contract": maxwell_contract,
        "missing_full_fidelity_requirements": [
            "full Faraday/displacement-current Maxwell field evolution",
            "same-deck electromagnetic GENE/CGYRO/GS2 output artifacts",
            "native electromagnetic phi/A_parallel/B_parallel same-case parity thresholds",
            "same-deck external electromagnetic grid-convergence evidence",
        ],
        "omitted_physics": OMITTED_PHYSICS,
        "required_external_observables": REQUIRED_EXTERNAL_OBSERVABLES,
        "required_external_solver_families": REQUIRED_EXTERNAL_SOLVERS,
        "schema": "gk-electromagnetic-fidelity.v1",
        "status": status,
    }


def write_reports(report: dict[str, Any], *, report_dir: Path = REPORT_DIR) -> None:
    """Write JSON and Markdown electromagnetic GK fidelity reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / JSON_REPORT.name).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# GK Electromagnetic Fidelity Gate",
        "",
        str(report["description"]),
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Compact EM contract ready: `{report['compact_em_contract_ready']}`",
        (
            "- External EM parity comparison ready: "
            f"`{report['external_em_parity_comparison_ready']}`"
        ),
        "",
        "## Gate rows",
        "",
        "| Gate | EM enabled | Time history | Compact closure | Full Vlasov-Maxwell parity |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for name in ("electrostatic_gate", "electromagnetic_gate"):
        gate = report[name]
        lines.append(
            "| {name} | `{em}` | `{history}` | `{compact}` | `{parity}` |".format(
                compact=gate["compact_closure_ready"],
                em=gate["electromagnetic_enabled"],
                history=gate["time_history_ready"],
                name=name,
                parity=gate["full_vlasov_maxwell_parity_ready"],
            )
        )
    grid_evidence = report["electromagnetic_grid_convergence_evidence"]
    lines.extend(
        [
            "",
            "## Compact-EM grid convergence evidence",
            "",
            f"- Schema: `{grid_evidence['schema']}`",
            f"- Status: `{grid_evidence['status']}`",
            (f"- Grid convergence ready: `{grid_evidence['grid_convergence_ready']}`"),
            (
                "- Max relative total-energy drift: "
                f"`{grid_evidence['max_relative_total_energy_drift']:.6e}`"
            ),
            (f"- Relative energy tolerance: `{grid_evidence['relative_energy_tolerance']:.6e}`"),
            "",
            "| Case | Grid | Field-energy closure | Compact closure | Relative total-energy drift |",
            "|---|---|:---:|:---:|---:|",
        ]
    )
    for row in grid_evidence["grid_rows"]:
        shape = row["grid_shape"]
        lines.append(
            "| {case_id} | `{n_kx}x{n_ky}x{n_theta}x{n_vpar}x{n_mu}` | "
            "`{field}` | `{compact}` | {drift:.6e} |".format(
                case_id=row["case_id"],
                compact=row["compact_closure_ready"],
                drift=row["relative_total_energy_drift"],
                field=row["field_energy_total_closes"],
                n_kx=shape["n_kx"],
                n_ky=shape["n_ky"],
                n_theta=shape["n_theta"],
                n_vpar=shape["n_vpar"],
                n_mu=shape["n_mu"],
            )
        )
    lines.extend(["", "## Omitted physics", ""])
    for item in report["omitted_physics"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Maxwell evolution contract", ""])
    contract = report["maxwell_evolution_contract"]
    lines.append(f"- Native field-evolution mode: `{contract['native_field_evolution_mode']}`")
    lines.append(
        f"- Full Vlasov-Maxwell parity ready: `{contract['full_vlasov_maxwell_parity_ready']}`"
    )
    lines.append(
        "- Blocking equation ids: "
        + ", ".join(f"`{item}`" for item in contract["blocking_equation_ids"])
    )
    lines.extend(
        [
            "",
            "| Equation id | Implemented | Compact closure | Native status |",
            "|---|:---:|:---:|---|",
        ]
    )
    for equation in contract["equations"]:
        lines.append(
            "| {equation_id} | `{implemented}` | `{compact}` | {status} |".format(
                compact=equation["compact_closure_ready"],
                equation_id=equation["equation_id"],
                implemented=equation["implemented_by_native_solver"],
                status=equation["native_status"],
            )
        )
    lines.extend(["", "## Missing full-fidelity requirements", ""])
    for item in report["missing_full_fidelity_requirements"]:
        lines.append(f"- {item}")
    lines.append("")
    (report_dir / MD_REPORT.name).write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero if full EM parity is blocked"
    )
    args = parser.parse_args(argv)
    report = run_benchmark()
    write_reports(report)
    if args.check and report["status"] != "accepted_full_vlasov_maxwell_parity":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
