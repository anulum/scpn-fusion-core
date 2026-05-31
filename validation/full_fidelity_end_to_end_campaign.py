"""Integrated fail-closed full-fidelity campaign across all native blockers.

This module intentionally keeps all six declared full-fidelity blockers in one
machine-readable report. It does not downgrade missing public reference data to
passing status; production parity requires local artifacts, provenance, license
metadata, checksums, thresholds, and external solver comparison evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "validation" / "reports"
PUBLIC_SOURCES = ROOT / "validation" / "reference_data" / "full_fidelity_public_sources.json"
PUBLIC_SOURCE_DOWNLOADS = REPORT_DIR / "full_fidelity_public_source_downloads.json"
JSON_REPORT = REPORT_DIR / "full_fidelity_end_to_end_campaign.json"
MD_REPORT = REPORT_DIR / "full_fidelity_end_to_end_campaign.md"

from validation.benchmark_full_fidelity_acceptance import run_benchmark as run_acceptance
from validation.benchmark_impurity_transport_contract import (
    run_benchmark as run_impurity_contract,
)
from validation.benchmark_runaway_dream_contract import run_benchmark as run_runaway_contract


def _load_sources() -> dict[str, Any]:
    registry = json.loads(PUBLIC_SOURCES.read_text(encoding="utf-8"))
    if registry.get("schema") != "full-fidelity-public-source-registry.v1":
        raise ValueError("full-fidelity public source registry schema mismatch")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("full-fidelity public source registry must define sources")
    return registry


def _load_downloads() -> dict[str, Any]:
    if not PUBLIC_SOURCE_DOWNLOADS.exists():
        return {
            "all_reachable_downloads_completed": False,
            "cache_root": "data/external/full_fidelity_public_sources",
            "items": [],
            "schema": "full-fidelity-public-source-downloads.v1",
        }
    downloads = json.loads(PUBLIC_SOURCE_DOWNLOADS.read_text(encoding="utf-8"))
    if downloads.get("schema") != "full-fidelity-public-source-downloads.v1":
        raise ValueError("full-fidelity public source download report schema mismatch")
    return downloads


def _sources_for(registry: dict[str, Any], surface: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in registry["sources"]:
        if source.get("surface") == surface:
            out.append(dict(source))
    return out


def _acceptance_surface(report: dict[str, Any], surface: str) -> dict[str, Any]:
    for row in report["surfaces"]:
        if row["surface"] == surface:
            return row
    raise KeyError(surface)


def _report_exists(path: str) -> bool:
    return (ROOT / path).exists()


def run_campaign() -> dict[str, Any]:
    """Return integrated full-fidelity status for all six blocker classes."""
    registry = _load_sources()
    downloads = _load_downloads()
    acceptance = run_acceptance()
    runaway = run_runaway_contract(repeats=3)
    impurity = run_impurity_contract()
    gk = _acceptance_surface(acceptance, "native_nonlinear_gyrokinetics")
    runaway_surface = _acceptance_surface(acceptance, "runaway_electrons")
    impurity_surface = _acceptance_surface(acceptance, "impurity_transport")

    lanes = [
        {
            "lane": "gene_cgyro_gs2_nonlinear_gk_parity",
            "surface": "native_nonlinear_gyrokinetics",
            "status": "blocked_missing_public_reference_artifacts",
            "locally_actionable_contract_ready": bool(gk["implemented_dimensions"]),
            "reference_cases_ready": bool(gk["reference_cases"]["ready"]),
            "sources": _sources_for(registry, "native_nonlinear_gyrokinetics"),
            "next_required_evidence": gk["missing_requirements"],
        },
        {
            "lane": "full_maxwell_electromagnetic_fidelity",
            "surface": "native_nonlinear_gyrokinetics",
            "status": "blocked_missing_full_vlasov_maxwell_field_solve",
            "locally_actionable_contract_ready": bool(
                gk["implemented_dimensions"].get("electromagnetic_a_parallel_surface")
                and gk["implemented_dimensions"].get("electromagnetic_b_parallel_surface")
                and gk["implemented_dimensions"].get("electromagnetic_energy_history_export")
            ),
            "reference_cases_ready": bool(gk["reference_cases"]["ready"]),
            "sources": _sources_for(registry, "native_nonlinear_gyrokinetics"),
            "next_required_evidence": [
                "native nonlinear Ampere/Faraday closure beyond compact A_parallel/B_parallel contracts",
                "GENE/CGYRO/GS2 electromagnetic field-energy and transport parity artifacts",
            ],
        },
        {
            "lane": "production_scale_decomposition",
            "surface": "solver_runtime",
            "status": "blocked_missing_cluster_scaling_evidence",
            "locally_actionable_contract_ready": _report_exists(
                "validation/reports/upcloud_l4_native_solver_benchmarks.json"
            ),
            "reference_cases_ready": False,
            "sources": [],
            "next_required_evidence": [
                "radial/toroidal domain decomposition implementation",
                "multi-GPU or cluster scaling reports on production-size grids",
                "large-grid warm GPU throughput and convergence evidence",
            ],
        },
        {
            "lane": "dream_grade_runaway_electrons",
            "surface": "runaway_electrons",
            "status": "blocked_missing_public_dream_artifacts",
            "locally_actionable_contract_ready": bool(
                runaway["passed"]
                and runaway_surface["implemented_dimensions"].get(
                    "dream_style_multidimensional_artifact_export"
                )
            ),
            "reference_cases_ready": bool(runaway_surface["reference_cases"]["ready"]),
            "sources": _sources_for(registry, "runaway_electrons"),
            "next_required_evidence": runaway_surface["missing_requirements"],
        },
        {
            "lane": "aurora_strahl_grade_impurities",
            "surface": "impurity_transport",
            "status": "blocked_missing_public_aurora_strahl_artifacts",
            "locally_actionable_contract_ready": bool(
                impurity["passed"]
                and impurity_surface["implemented_dimensions"].get(
                    "charge_state_resolved_density_artifact_export"
                )
            ),
            "reference_cases_ready": bool(impurity_surface["reference_cases"]["ready"]),
            "sources": _sources_for(registry, "impurity_transport"),
            "next_required_evidence": impurity_surface["missing_requirements"],
        },
        {
            "lane": "free_boundary_equilibrium_strict_parity",
            "surface": "free_boundary_equilibrium",
            "status": "blocked_missing_external_coil_current_reference_artifacts",
            "locally_actionable_contract_ready": _report_exists(
                "validation/reports/free_boundary_benchmark.json"
            ),
            "reference_cases_ready": False,
            "sources": _sources_for(registry, "free_boundary_equilibrium"),
            "next_required_evidence": [
                "public coil-current sidecars or machine coil metadata for GEQDSK rows",
                "strict FreeGS backend convergence on public free-boundary cases",
                "profile-source/free-boundary reconstruction parity artifacts",
            ],
        },
    ]

    return {
        "benchmark": "full_fidelity_end_to_end_campaign",
        "schema": "full-fidelity-end-to-end-campaign.v1",
        "description": "Integrated fail-closed campaign covering GK, EM, production-scale runtime, DREAM, Aurora/STRAHL, and free-boundary blockers.",
        "public_source_cache_root": downloads["cache_root"],
        "public_source_download_report": str(PUBLIC_SOURCE_DOWNLOADS.relative_to(ROOT)),
        "public_sources_cached": bool(downloads["all_reachable_downloads_completed"]),
        "public_source_registry": str(PUBLIC_SOURCES.relative_to(ROOT)),
        "acceptance_report": "validation/reports/full_fidelity_acceptance_benchmark.json",
        "lanes": lanes,
        "all_locally_actionable_contracts_ready": all(
            bool(lane["locally_actionable_contract_ready"]) for lane in lanes
        ),
        "reference_parity_ready": all(bool(lane["reference_cases_ready"]) for lane in lanes),
        "acceptance_passed": False,
        "status": "not_full_fidelity",
    }


def write_reports(report: dict[str, Any]) -> None:
    """Write JSON and Markdown integrated campaign reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Full-Fidelity End-to-End Campaign",
        "",
        "This report keeps all declared full-fidelity blockers in one fail-closed gate.",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Acceptance passed: `{report['acceptance_passed']}`",
        f"- Public source registry: `{report['public_source_registry']}`",
        f"- Public source download report: `{report['public_source_download_report']}`",
        f"- Public sources cached: `{report['public_sources_cached']}`",
        f"- Public source cache root: `{report['public_source_cache_root']}`",
        f"- Local contracts ready: `{report['all_locally_actionable_contracts_ready']}`",
        f"- Reference parity ready: `{report['reference_parity_ready']}`",
        "",
        "| Lane | Status | Local contract ready | Reference parity ready | Sources | Next evidence |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for lane in report["lanes"]:
        sources = ", ".join(source["solver_family"] for source in lane["sources"]) or "none"
        evidence = "<br>".join(lane["next_required_evidence"])
        lines.append(
            "| {lane} | {status} | {local} | {reference} | {sources} | {evidence} |".format(
                lane=lane["lane"],
                status=lane["status"],
                local=lane["locally_actionable_contract_ready"],
                reference=lane["reference_cases_ready"],
                sources=sources,
                evidence=evidence,
            )
        )
    lines.append("")
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run the integrated campaign and persist reports."""
    report = run_campaign()
    write_reports(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
