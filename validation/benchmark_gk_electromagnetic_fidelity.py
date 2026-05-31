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


def run_benchmark() -> dict[str, Any]:
    """Run the local compact-EM gate and return fail-closed parity status."""
    electrostatic_gate = _run_gate(electromagnetic=False, seed=211)
    electromagnetic_gate = _run_gate(electromagnetic=True, seed=223)
    compact_ready = bool(electromagnetic_gate["compact_closure_ready"])
    status = (
        "blocked_missing_full_vlasov_maxwell_field_solve"
        if compact_ready
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
        "electrostatic_gate": electrostatic_gate,
        "external_em_parity_comparison_ready": False,
        "locally_actionable_contract_ready": compact_ready,
        "missing_full_fidelity_requirements": [
            "full Faraday/displacement-current Maxwell field evolution",
            "same-deck electromagnetic GENE/CGYRO/GS2 output artifacts",
            "native electromagnetic phi/A_parallel/B_parallel same-case parity thresholds",
            "grid-convergence evidence for electromagnetic field-energy histories",
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
    lines.extend(["", "## Omitted physics", ""])
    for item in report["omitted_physics"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Missing full-fidelity requirements", ""])
    for item in report["missing_full_fidelity_requirements"]:
        lines.append(f"- {item}")
    lines.append("")
    (report_dir / MD_REPORT.name).write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit non-zero if full EM parity is blocked")
    args = parser.parse_args(argv)
    report = run_benchmark()
    write_reports(report)
    if args.check and report["status"] != "accepted_full_vlasov_maxwell_parity":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
