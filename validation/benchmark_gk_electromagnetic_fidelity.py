#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed electromagnetic nonlinear GK fidelity gate.

The native nonlinear GK path currently implements compact algebraic
``A_parallel`` and ``B_parallel`` closures plus a local source-free spectral
Maxwell field-evolution contract. This benchmark verifies those contracts
separately from electrostatic GK runs, then keeps full Vlasov-Maxwell parity
blocked until the Maxwell fields are coupled to 5D kinetic current moments and
same-deck electromagnetic GENE/CGYRO/GS2 outputs are compared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.gk_maxwell_evolution import (
    MaxwellEvolutionConfig,
    run_local_maxwell_evolution,
)
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
    "self-consistent kinetic current coupling in the nonlinear 5D Vlasov-Maxwell loop",
    "external same-deck electromagnetic GENE/CGYRO/GS2 output parity",
]
GRID_CONVERGENCE_CASES: list[dict[str, int | str]] = [
    {"case_id": "compact_em_4x4x8", "n_kx": 4, "n_ky": 4, "n_theta": 8},
    {"case_id": "compact_em_6x6x10", "n_kx": 6, "n_ky": 6, "n_theta": 10},
    {"case_id": "compact_em_8x8x12", "n_kx": 8, "n_ky": 8, "n_theta": 12},
]
GRID_CONVERGENCE_RELATIVE_ENERGY_TOLERANCE = 5.0e-1
NATIVE_EM_SAME_CASE_ABSOLUTE_TOLERANCE = 1.0e-18
NATIVE_EM_SAME_CASE_RELATIVE_TOLERANCE = 1.0e-15


def _maxwell_evolution_contract(
    *, compact_closure_ready: bool, maxwell_evolution_evidence: dict[str, Any]
) -> dict[str, Any]:
    """Return the fail-closed full-Maxwell equation implementation contract."""
    faraday_ready = bool(maxwell_evolution_evidence["faraday_induction_supported"])
    ampere_ready = bool(maxwell_evolution_evidence["ampere_maxwell_displacement_current_supported"])
    inductive_ready = bool(
        maxwell_evolution_evidence["inductive_parallel_electric_field_supported"]
    )
    magnetic_divergence_ready = bool(
        maxwell_evolution_evidence["magnetic_divergence_constraint_supported"]
    )
    equations = [
        {
            "compact_closure_ready": False,
            "equation_id": "faraday_induction",
            "equation": "dB/dt = -curl(E)",
            "implemented_by_native_solver": faraday_ready,
            "native_status": "implemented_as_local_source_free_spectral_field_evolution",
            "required_for_full_vlasov_maxwell": True,
        },
        {
            "compact_closure_ready": False,
            "equation_id": "ampere_maxwell_displacement_current",
            "equation": "curl(B) = mu0 J + mu0 eps0 dE/dt",
            "implemented_by_native_solver": ampere_ready,
            "native_status": "implemented_as_local_source_free_spectral_field_evolution",
            "required_for_full_vlasov_maxwell": True,
        },
        {
            "compact_closure_ready": False,
            "equation_id": "inductive_parallel_electric_field",
            "equation": "E_parallel = -grad_parallel(phi) - dA_parallel/dt",
            "implemented_by_native_solver": inductive_ready,
            "native_status": "implemented_as_local_source_free_spectral_field_evolution",
            "required_for_full_vlasov_maxwell": True,
        },
        {
            "compact_closure_ready": False,
            "equation_id": "magnetic_divergence_constraint",
            "equation": "div(B) = 0",
            "implemented_by_native_solver": magnetic_divergence_ready,
            "native_status": "implemented_as_local_source_free_spectral_field_evolution",
            "required_for_full_vlasov_maxwell": True,
        },
        {
            "compact_closure_ready": False,
            "equation_id": "self_consistent_kinetic_current_coupling",
            "equation": "J_parallel and perpendicular current moments from evolved 5D distribution",
            "implemented_by_native_solver": False,
            "native_status": "missing_self_consistent_5d_kinetic_current_coupling",
            "required_for_full_vlasov_maxwell": True,
        },
        {
            "compact_closure_ready": False,
            "equation_id": "same_deck_external_em_parity",
            "equation": "same-deck phi/A_parallel/B_parallel comparison to GENE/CGYRO/GS2",
            "implemented_by_native_solver": False,
            "native_status": "missing_external_same_deck_em_outputs_and_thresholds",
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
        "native_field_evolution_mode": "local_spectral_maxwell_evolution",
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


def _relative_drift(values: NDArray[np.float64]) -> float:
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


def _run_maxwell_evolution_evidence() -> dict[str, Any]:
    """Run native source-free Maxwell evolution evidence for the EM gate."""
    result = run_local_maxwell_evolution(
        MaxwellEvolutionConfig(n_kx=6, n_ky=6, n_steps=16, dt=1.0e-12, seed=311)
    )
    return cast(dict[str, Any], result.to_evidence())


def _run_native_em_same_case_threshold_evidence() -> dict[str, Any]:
    """Gate deterministic native EM same-case field-energy thresholds."""
    cfg = _config(electromagnetic=True)
    solver_a = NonlinearGKSolver(cfg)
    solver_b = NonlinearGKSolver(cfg)
    result_a = solver_a.run(solver_a.init_state(amplitude=1.0e-5, seed=419))
    result_b = solver_b.run(solver_b.init_state(amplitude=1.0e-5, seed=419))
    total_field_a = (
        result_a.phi_energy_t + result_a.A_parallel_energy_t + result_a.B_parallel_energy_t
    )
    total_field_b = (
        result_b.phi_energy_t + result_b.A_parallel_energy_t + result_b.B_parallel_energy_t
    )
    observables: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
        "electromagnetic_apar_energy": (
            result_a.A_parallel_energy_t,
            result_b.A_parallel_energy_t,
        ),
        "electromagnetic_bpar_energy": (
            result_a.B_parallel_energy_t,
            result_b.B_parallel_energy_t,
        ),
        "electromagnetic_phi_energy": (result_a.phi_energy_t, result_b.phi_energy_t),
        "electromagnetic_total_field_energy": (total_field_a, total_field_b),
    }
    rows: list[dict[str, Any]] = []
    for observable, (native, replay) in sorted(observables.items()):
        shape_matches = native.shape == replay.shape
        if shape_matches and native.size:
            max_absolute_error = float(np.max(np.abs(native - replay)))
            denominator = max(float(np.max(np.abs(native))), 1.0e-30)
            max_relative_error = float(max_absolute_error / denominator)
        else:
            max_absolute_error = float("inf")
            max_relative_error = float("inf")
        rows.append(
            {
                "absolute_tolerance": NATIVE_EM_SAME_CASE_ABSOLUTE_TOLERANCE,
                "max_absolute_error": max_absolute_error,
                "max_relative_error": max_relative_error,
                "observable": observable,
                "relative_tolerance": NATIVE_EM_SAME_CASE_RELATIVE_TOLERANCE,
                "shape": list(native.shape),
                "shape_matches": shape_matches,
                "threshold_pass": bool(
                    shape_matches
                    and max_absolute_error <= NATIVE_EM_SAME_CASE_ABSOLUTE_TOLERANCE
                    and max_relative_error <= NATIVE_EM_SAME_CASE_RELATIVE_TOLERANCE
                ),
            }
        )
    same_case_thresholds_ready = bool(rows and all(row["threshold_pass"] for row in rows))
    return {
        "benchmark_case_id": "native_em_replay_4x4x8_seed419",
        "observable_count": len(rows),
        "observable_rows": rows,
        "reference_kind": "native_deterministic_replay_not_external_parity",
        "same_case_thresholds_ready": same_case_thresholds_ready,
        "schema": "gk-native-em-same-case-thresholds.v1",
        "status": "accepted_native_em_same_case_thresholds"
        if same_case_thresholds_ready
        else "blocked_native_em_same_case_thresholds_failed",
    }


def run_benchmark() -> dict[str, Any]:
    """Run the local compact-EM gate and return fail-closed parity status."""
    electrostatic_gate = _run_gate(electromagnetic=False, seed=211)
    electromagnetic_gate = _run_gate(electromagnetic=True, seed=223)
    compact_ready = bool(electromagnetic_gate["compact_closure_ready"])
    grid_convergence_evidence = _run_grid_convergence_evidence()
    grid_convergence_ready = bool(grid_convergence_evidence["grid_convergence_ready"])
    maxwell_evolution_evidence = _run_maxwell_evolution_evidence()
    native_same_case_threshold_evidence = _run_native_em_same_case_threshold_evidence()
    maxwell_evolution_ready = bool(
        maxwell_evolution_evidence["status"] == "accepted_local_source_free_maxwell_evolution"
    )
    native_same_case_thresholds_ready = bool(
        native_same_case_threshold_evidence["same_case_thresholds_ready"]
    )
    maxwell_contract = _maxwell_evolution_contract(
        compact_closure_ready=compact_ready,
        maxwell_evolution_evidence=maxwell_evolution_evidence,
    )
    status = (
        "blocked_missing_external_em_parity_outputs"
        if (
            compact_ready
            and grid_convergence_ready
            and maxwell_evolution_ready
            and native_same_case_thresholds_ready
        )
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
        "locally_actionable_contract_ready": (
            compact_ready
            and grid_convergence_ready
            and maxwell_evolution_ready
            and native_same_case_thresholds_ready
        ),
        "maxwell_evolution_contract": maxwell_contract,
        "maxwell_evolution_evidence": maxwell_evolution_evidence,
        "native_em_same_case_threshold_evidence": native_same_case_threshold_evidence,
        "missing_full_fidelity_requirements": [
            "self-consistent kinetic current coupling in the nonlinear 5D Vlasov-Maxwell loop",
            "same-deck electromagnetic GENE/CGYRO/GS2 output artifacts",
            "external electromagnetic phi/A_parallel/B_parallel same-case parity thresholds",
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
    maxwell_evidence = report["maxwell_evolution_evidence"]
    lines.extend(
        [
            "",
            "## Local Maxwell evolution evidence",
            "",
            f"- Schema: `{maxwell_evidence['schema']}`",
            f"- Status: `{maxwell_evidence['status']}`",
            (f"- Faraday induction supported: `{maxwell_evidence['faraday_induction_supported']}`"),
            (
                "- Ampere-Maxwell displacement current supported: "
                f"`{maxwell_evidence['ampere_maxwell_displacement_current_supported']}`"
            ),
            (
                "- Inductive parallel electric field supported: "
                f"`{maxwell_evidence['inductive_parallel_electric_field_supported']}`"
            ),
            (
                "- Self-consistent kinetic current supported: "
                f"`{maxwell_evidence['self_consistent_kinetic_current_supported']}`"
            ),
            (
                "- Magnetic divergence constraint supported: "
                f"`{maxwell_evidence['magnetic_divergence_constraint_supported']}`"
            ),
            (
                "- Max relative total-field-energy drift: "
                f"`{maxwell_evidence['max_relative_total_field_energy_drift']:.6e}`"
            ),
            (f"- Max Faraday residual: `{maxwell_evidence['max_faraday_linf_residual']:.6e}`"),
            (
                "- Max Ampere-Maxwell residual: "
                f"`{maxwell_evidence['max_ampere_maxwell_linf_residual']:.6e}`"
            ),
            (
                "- Max inductive parallel electric-field residual: "
                f"`{maxwell_evidence['max_inductive_e_parallel_linf_residual']:.6e}`"
            ),
            (
                "- Max magnetic divergence residual: "
                f"`{maxwell_evidence['max_magnetic_divergence_linf_residual']:.6e}`"
            ),
        ]
    )
    native_thresholds = report["native_em_same_case_threshold_evidence"]
    lines.extend(
        [
            "",
            "## Native EM same-case threshold evidence",
            "",
            f"- Schema: `{native_thresholds['schema']}`",
            f"- Status: `{native_thresholds['status']}`",
            f"- Benchmark case id: `{native_thresholds['benchmark_case_id']}`",
            f"- Reference kind: `{native_thresholds['reference_kind']}`",
            (f"- Same-case thresholds ready: `{native_thresholds['same_case_thresholds_ready']}`"),
            "",
            "| Observable | Shape | Max absolute error | Absolute tolerance | "
            "Max relative error | Relative tolerance | Pass |",
            "|---|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in native_thresholds["observable_rows"]:
        lines.append(
            "| {observable} | `{shape}` | {abs_err:.6e} | {abs_tol:.6e} | "
            "{rel_err:.6e} | {rel_tol:.6e} | `{passed}` |".format(
                abs_err=row["max_absolute_error"],
                abs_tol=row["absolute_tolerance"],
                observable=row["observable"],
                passed=row["threshold_pass"],
                rel_err=row["max_relative_error"],
                rel_tol=row["relative_tolerance"],
                shape="x".join(str(item) for item in row["shape"]),
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
