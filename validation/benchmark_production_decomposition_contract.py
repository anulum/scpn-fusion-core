#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed production-scale decomposition contract benchmark."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.gk_domain_decomposition import (  # noqa: E402
    GKDomainDecompositionPlan,
    build_radial_toroidal_decomposition,
    decomposition_invariant_metrics,
    rank_tile_communication_contract,
)

REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "production_decomposition_contract.json"
MD_REPORT = REPORT_DIR / "production_decomposition_contract.md"


def _plan_row(case_id: str, plan: GKDomainDecompositionPlan) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "halo": plan.halo,
        "halo_overhead_ratio": plan.halo_overhead_ratio,
        "max_owned_cells": plan.max_owned_cells,
        "min_owned_cells": plan.min_owned_cells,
        "n_mu": plan.n_mu,
        "n_radial": plan.n_radial,
        "n_theta": plan.n_theta,
        "n_toroidal": plan.n_toroidal,
        "n_vpar": plan.n_vpar,
        "owned_cell_imbalance": plan.owned_cell_imbalance,
        "radial_parts": plan.radial_parts,
        "toroidal_parts": plan.toroidal_parts,
        "total_halo_phase_cells": plan.total_halo_phase_cells,
        "total_owned_phase_cells": plan.total_owned_phase_cells,
        "total_ranks": plan.total_ranks,
    }


def _cpu_benchmark_row(case_id: str, plan: GKDomainDecompositionPlan) -> dict[str, Any]:
    cell_count = plan.total_owned_phase_cells
    state = np.linspace(0.0, 1.0, num=cell_count, dtype=np.float64).reshape(
        plan.n_radial,
        plan.n_toroidal,
        plan.n_theta,
        plan.n_vpar,
        plan.n_mu,
    )
    started = time.perf_counter()
    metrics = decomposition_invariant_metrics(plan, state)
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    return {
        "case_id": case_id,
        "cells_per_second": cell_count / elapsed,
        "elapsed_s": elapsed,
        "free_energy_relative_error": metrics.free_energy_relative_error,
        "halo_exchange_pass": metrics.halo_exchange_pass,
        "inventory_relative_error": metrics.inventory_relative_error,
        "owned_phase_cells": cell_count,
        "reconstruction_linf_error": metrics.reconstruction_linf_error,
    }


def _hardware_metadata() -> dict[str, Any]:
    return {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }


def run_benchmark() -> dict[str, Any]:
    """Return decomposition-contract evidence while keeping scaling fail-closed."""
    medium_plan = build_radial_toroidal_decomposition(
        n_radial=64,
        n_toroidal=32,
        n_theta=16,
        n_vpar=16,
        n_mu=8,
        radial_parts=4,
        toroidal_parts=2,
        halo=1,
    )
    production_plan = build_radial_toroidal_decomposition(
        n_radial=256,
        n_toroidal=128,
        n_theta=32,
        n_vpar=32,
        n_mu=16,
        radial_parts=8,
        toroidal_parts=4,
        halo=1,
    )
    local_cpu_plan = build_radial_toroidal_decomposition(
        n_radial=64,
        n_toroidal=32,
        n_theta=8,
        n_vpar=8,
        n_mu=4,
        radial_parts=4,
        toroidal_parts=2,
        halo=1,
    )
    cases = [
        _plan_row("medium_64x32_4x2", medium_plan),
        _plan_row("production_256x128_8x4", production_plan),
    ]
    communication_contract_rows = rank_tile_communication_contract(production_plan)
    cpu_benchmark_rows = [_cpu_benchmark_row("local_cpu_64x32_4x2", local_cpu_plan)]
    coverage_pass = all(case["min_owned_cells"] > 0 for case in cases)
    imbalance_pass = all(case["owned_cell_imbalance"] <= 1.05 for case in cases)
    communication_contract_ready = all(
        bool(row["communication_contract_ready"]) for row in communication_contract_rows
    )
    halo_exchange_pass = all(bool(row["halo_exchange_pass"]) for row in cpu_benchmark_rows)
    decomposition_invariant_pass = all(
        row["reconstruction_linf_error"] == 0.0
        and row["inventory_relative_error"] == 0.0
        and row["free_energy_relative_error"] == 0.0
        for row in cpu_benchmark_rows
    )
    contract_pass = bool(
        coverage_pass
        and imbalance_pass
        and communication_contract_ready
        and halo_exchange_pass
        and decomposition_invariant_pass
    )
    return {
        "benchmark": "production_decomposition_contract",
        "schema": "production-decomposition-contract.v1",
        "description": (
            "Deterministic radial/toroidal decomposition contract for production-scale "
            "5D nonlinear GK scheduling. This is not distributed runtime scaling evidence."
        ),
        "cases": cases,
        "communication_contract_ready": communication_contract_ready,
        "communication_contract_rows": communication_contract_rows,
        "contract_pass": contract_pass,
        "coverage_pass": coverage_pass,
        "cpu_benchmark_rows": cpu_benchmark_rows,
        "decomposition_invariant_pass": decomposition_invariant_pass,
        "halo_exchange_pass": halo_exchange_pass,
        "hardware_metadata": _hardware_metadata(),
        "imbalance_pass": imbalance_pass,
        "production_scale_ready": False,
        "reproducible_commands": [
            "python validation/benchmark_production_decomposition_contract.py",
            "python -m pytest tests/test_gk_domain_decomposition.py -q",
        ],
        "status": "blocked_contract_ready_missing_distributed_runtime_scaling",
        "missing_requirements": [
            "MPI or multi-GPU execution path over the declared rank tiles",
            "large-grid cluster/GPU wall-time scaling report",
            "same-physics convergence evidence across distributed decomposition shapes",
            "hardware-specific multi-rank throughput and efficiency thresholds",
        ],
    }


def write_reports(report: dict[str, Any]) -> None:
    """Write JSON and Markdown decomposition reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Production Decomposition Contract",
        "",
        report["description"],
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Contract pass: `{report['contract_pass']}`",
        f"- Communication contract ready: `{report['communication_contract_ready']}`",
        f"- Halo exchange pass: `{report['halo_exchange_pass']}`",
        f"- Decomposition invariant pass: `{report['decomposition_invariant_pass']}`",
        f"- Production-scale ready: `{report['production_scale_ready']}`",
        f"- Python: `{report['hardware_metadata']['python_version']}`",
        f"- CPU count: `{report['hardware_metadata']['cpu_count']}`",
        "",
        "| Case | R x T | Parts | Ranks | Imbalance | Halo overhead |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        lines.append(
            "| {case_id} | {nr} x {nt} | {rp} x {tp} | {ranks} | {imb:.6f} | {halo:.6f} |".format(
                case_id=case["case_id"],
                nr=case["n_radial"],
                nt=case["n_toroidal"],
                rp=case["radial_parts"],
                tp=case["toroidal_parts"],
                ranks=case["total_ranks"],
                imb=case["owned_cell_imbalance"],
                halo=case["halo_overhead_ratio"],
            )
        )
    lines.extend(
        [
            "",
            "## Rank communication contract",
            "",
            "| Rank | Neighbours | Halo face shapes ready |",
            "|---:|---|:---:|",
        ]
    )
    for row in report["communication_contract_rows"]:
        neighbours = ", ".join(f"{face}={rank}" for face, rank in row["neighbour_ranks"].items())
        lines.append(
            "| {rank} | {neighbours} | `{ready}` |".format(
                neighbours=neighbours,
                rank=row["rank"],
                ready=row["communication_contract_ready"],
            )
        )
    lines.extend(
        [
            "",
            "## Local CPU halo/invariant benchmark",
            "",
            "| Case | Owned phase cells | Elapsed s | Cells/s | Halo | Reconstruction L_inf | Inventory rel | Free-energy rel |",
            "|---|---:|---:|---:|:---:|---:|---:|---:|",
        ]
    )
    for row in report["cpu_benchmark_rows"]:
        lines.append(
            "| {case_id} | {cells} | {elapsed:.6e} | {rate:.6e} | `{halo}` | {recon:.6e} | {inventory:.6e} | {energy:.6e} |".format(
                case_id=row["case_id"],
                cells=row["owned_phase_cells"],
                elapsed=row["elapsed_s"],
                energy=row["free_energy_relative_error"],
                halo=row["halo_exchange_pass"],
                inventory=row["inventory_relative_error"],
                rate=row["cells_per_second"],
                recon=row["reconstruction_linf_error"],
            )
        )
    lines.extend(["", "## Reproducible commands", ""])
    lines.extend(f"- `{item}`" for item in report["reproducible_commands"])
    lines.extend(["", "## Missing requirements", ""])
    lines.extend(f"- {item}" for item in report["missing_requirements"])
    lines.append("")
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = run_benchmark()
    write_reports(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
