"""Fail-closed production-scale decomposition contract benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.gk_domain_decomposition import (  # noqa: E402
    GKDomainDecompositionPlan,
    build_radial_toroidal_decomposition,
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


def run_benchmark() -> dict[str, Any]:
    """Return decomposition-contract evidence while keeping scaling fail-closed."""
    cases = [
        _plan_row(
            "medium_64x32_4x2",
            build_radial_toroidal_decomposition(
                n_radial=64,
                n_toroidal=32,
                n_theta=16,
                n_vpar=16,
                n_mu=8,
                radial_parts=4,
                toroidal_parts=2,
                halo=1,
            ),
        ),
        _plan_row(
            "production_256x128_8x4",
            build_radial_toroidal_decomposition(
                n_radial=256,
                n_toroidal=128,
                n_theta=32,
                n_vpar=32,
                n_mu=16,
                radial_parts=8,
                toroidal_parts=4,
                halo=1,
            ),
        ),
    ]
    coverage_pass = all(case["min_owned_cells"] > 0 for case in cases)
    imbalance_pass = all(case["owned_cell_imbalance"] <= 1.05 for case in cases)
    contract_pass = bool(coverage_pass and imbalance_pass)
    return {
        "benchmark": "production_decomposition_contract",
        "schema": "production-decomposition-contract.v1",
        "description": (
            "Deterministic radial/toroidal decomposition contract for production-scale "
            "5D nonlinear GK scheduling. This is not distributed runtime scaling evidence."
        ),
        "cases": cases,
        "contract_pass": contract_pass,
        "coverage_pass": coverage_pass,
        "imbalance_pass": imbalance_pass,
        "production_scale_ready": False,
        "status": "blocked_contract_ready_missing_distributed_runtime_scaling",
        "missing_requirements": [
            "MPI or multi-GPU execution path over the declared rank tiles",
            "halo exchange implementation and correctness tests",
            "large-grid cluster/GPU wall-time scaling report",
            "same-physics convergence evidence across decomposition shapes",
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
        f"- Production-scale ready: `{report['production_scale_ready']}`",
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
