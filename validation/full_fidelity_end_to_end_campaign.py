"""Integrated fail-closed full-fidelity campaign across all native blockers.

This module intentionally keeps all six declared full-fidelity blockers in one
machine-readable report. It does not downgrade missing public reference data to
passing status; production parity requires local artifacts, provenance, license
metadata, checksums, thresholds, and external solver comparison evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "validation" / "reports"
PUBLIC_SOURCES = ROOT / "validation" / "reference_data" / "full_fidelity_public_sources.json"
PUBLIC_SOURCE_DOWNLOADS = REPORT_DIR / "full_fidelity_public_source_downloads.json"
REFERENCE_ARTIFACT_CONVERSION = REPORT_DIR / "full_fidelity_reference_artifact_conversion.json"
DREAM_EXECUTION_REQUEST = REPORT_DIR / "dream_reference_execution_request.json"
AURORA_EXECUTION_ARTIFACT = REPORT_DIR / "aurora_reference_execution_artifact.json"
GK_DECK_INVENTORY = REPORT_DIR / "gk_public_reference_deck_inventory.json"
GK_EXTERNAL_PARITY = REPORT_DIR / "gk_external_nonlinear_parity.json"
GK_ELECTROMAGNETIC_FIDELITY = REPORT_DIR / "gk_electromagnetic_fidelity.json"
PRODUCTION_DECOMPOSITION = REPORT_DIR / "production_decomposition_contract.json"
FREE_BOUNDARY_MACHINE_METADATA = REPORT_DIR / "free_boundary_public_machine_metadata_inventory.json"
FREEGS_PUBLIC_RECONSTRUCTION = REPORT_DIR / "freegs_public_example_reconstruction.json"
FREE_BOUNDARY_STRICT_PARITY = REPORT_DIR / "free_boundary_strict_parity_benchmark.json"
JSON_REPORT = REPORT_DIR / "full_fidelity_end_to_end_campaign.json"
MD_REPORT = REPORT_DIR / "full_fidelity_end_to_end_campaign.md"

from validation.benchmark_full_fidelity_acceptance import run_benchmark as run_acceptance
from validation.benchmark_gk_electromagnetic_fidelity import (
    run_benchmark as run_gk_electromagnetic_fidelity,
)
from validation.benchmark_impurity_transport_contract import (
    run_benchmark as run_impurity_contract,
)
from validation.benchmark_runaway_dream_contract import run_benchmark as run_runaway_contract
from validation.benchmark_sas_dataset_manifest import run_benchmark as run_sas_dataset_readiness


def _load_sources() -> dict[str, Any]:
    registry = cast(dict[str, Any], json.loads(PUBLIC_SOURCES.read_text(encoding="utf-8")))
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
    downloads = cast(
        dict[str, Any], json.loads(PUBLIC_SOURCE_DOWNLOADS.read_text(encoding="utf-8"))
    )
    if downloads.get("schema") != "full-fidelity-public-source-downloads.v1":
        raise ValueError("full-fidelity public source download report schema mismatch")
    return downloads


def _load_conversion() -> dict[str, Any]:
    if not REFERENCE_ARTIFACT_CONVERSION.exists():
        return {
            "accepted_full_fidelity_artifacts": 0,
            "partial_output_artifacts": 0,
            "schema": "full-fidelity-reference-artifact-conversion.v1",
        }
    conversion = cast(
        dict[str, Any], json.loads(REFERENCE_ARTIFACT_CONVERSION.read_text(encoding="utf-8"))
    )
    if conversion.get("schema") != "full-fidelity-reference-artifact-conversion.v1":
        raise ValueError("full-fidelity reference artifact conversion report schema mismatch")
    return conversion


def _load_dream_execution() -> dict[str, Any]:
    if not DREAM_EXECUTION_REQUEST.exists():
        return {
            "reference_output_ready": False,
            "schema": "dream-reference-execution-request.v1",
            "settings_deck_generated": False,
            "status": "not_run",
        }
    report = cast(dict[str, Any], json.loads(DREAM_EXECUTION_REQUEST.read_text(encoding="utf-8")))
    if report.get("schema") != "dream-reference-execution-request.v1":
        raise ValueError("DREAM reference execution request schema mismatch")
    return report


def _load_aurora_execution() -> dict[str, Any]:
    if not AURORA_EXECUTION_ARTIFACT.exists():
        return {
            "artifact_generated": False,
            "missing_full_fidelity_requirements": [],
            "reference_output_ready": False,
            "schema": "aurora-reference-execution-artifact.v1",
            "status": "not_run",
        }
    report = cast(dict[str, Any], json.loads(AURORA_EXECUTION_ARTIFACT.read_text(encoding="utf-8")))
    if report.get("schema") != "aurora-reference-execution-artifact.v1":
        raise ValueError("Aurora reference execution artifact schema mismatch")
    return report


def _load_gk_deck_inventory() -> dict[str, Any]:
    if not GK_DECK_INVENTORY.exists():
        return {
            "deck_count": 0,
            "missing_full_fidelity_requirements": [],
            "output_summary_count": 0,
            "reference_output_ready": False,
            "schema": "gk-public-reference-deck-inventory-report.v1",
            "status": "not_run",
        }
    report = cast(dict[str, Any], json.loads(GK_DECK_INVENTORY.read_text(encoding="utf-8")))
    if report.get("schema") != "gk-public-reference-deck-inventory-report.v1":
        raise ValueError("GK public reference deck inventory schema mismatch")
    return report


def _load_gk_external_parity() -> dict[str, Any]:
    if not GK_EXTERNAL_PARITY.exists():
        return {
            "converted_reference_artifacts": 0,
            "grid_convergence_ready": False,
            "missing_full_fidelity_requirements": [],
            "native_same_case_comparison_ready": False,
            "production_scale_scaling_ready": False,
            "reference_output_ready": False,
            "schema": "gk-external-nonlinear-output-parity-report.v1",
            "status": "not_run",
        }
    report = cast(dict[str, Any], json.loads(GK_EXTERNAL_PARITY.read_text(encoding="utf-8")))
    if report.get("schema") != "gk-external-nonlinear-output-parity-report.v1":
        raise ValueError("GK external nonlinear parity report schema mismatch")
    return report


def _load_production_decomposition() -> dict[str, Any]:
    if not PRODUCTION_DECOMPOSITION.exists():
        return {
            "contract_pass": False,
            "missing_requirements": [],
            "production_scale_ready": False,
            "schema": "production-decomposition-contract.v1",
            "status": "not_run",
        }
    report = cast(dict[str, Any], json.loads(PRODUCTION_DECOMPOSITION.read_text(encoding="utf-8")))
    if report.get("schema") != "production-decomposition-contract.v1":
        raise ValueError("production decomposition contract schema mismatch")
    return report


def _load_free_boundary_machine_metadata() -> dict[str, Any]:
    if not FREE_BOUNDARY_MACHINE_METADATA.exists():
        return {
            "machine_config_count": 0,
            "machine_metadata_ready": False,
            "missing_full_fidelity_requirements": [],
            "schema": "free-boundary-public-machine-metadata-inventory-report.v1",
            "status": "not_run",
        }
    report = cast(
        dict[str, Any], json.loads(FREE_BOUNDARY_MACHINE_METADATA.read_text(encoding="utf-8"))
    )
    if report.get("schema") != "free-boundary-public-machine-metadata-inventory-report.v1":
        raise ValueError("free-boundary public machine metadata inventory schema mismatch")
    return report


def _load_freegs_public_reconstruction() -> dict[str, Any]:
    if not FREEGS_PUBLIC_RECONSTRUCTION.exists():
        return {
            "case_count": 0,
            "missing_full_fidelity_requirements": [],
            "schema": "freegs-public-example-reconstruction-report.v1",
            "status": "not_run",
            "strict_free_boundary_parity_evidence": {
                "blocking_requirements": [],
                "coil_vacuum_sidecar_ready": False,
                "failed_threshold_check_count": 0,
                "geometry_containment_evidence": {
                    "boundary_containment_metric_ready": False,
                    "strict_geometry_containment_ready": False,
                },
                "grid_convergence_ready": False,
                "native_same_case_profile_source_ready": False,
                "status": "not_run",
                "strict_threshold_acceptance_ready": False,
            },
            "vacuum_comparison_pass": False,
        }
    report = cast(
        dict[str, Any], json.loads(FREEGS_PUBLIC_RECONSTRUCTION.read_text(encoding="utf-8"))
    )
    if report.get("schema") != "freegs-public-example-reconstruction-report.v1":
        raise ValueError("FreeGS public example reconstruction schema mismatch")
    report.setdefault(
        "strict_free_boundary_parity_evidence",
        {
            "blocking_requirements": [],
            "coil_vacuum_sidecar_ready": False,
            "failed_threshold_check_count": 0,
            "geometry_containment_evidence": {
                "boundary_containment_metric_ready": False,
                "strict_geometry_containment_ready": False,
            },
            "grid_convergence_ready": False,
            "native_same_case_profile_source_ready": False,
            "status": "not_run",
            "strict_threshold_acceptance_ready": False,
        },
    )
    return report


def _load_free_boundary_strict_parity() -> dict[str, Any]:
    if not FREE_BOUNDARY_STRICT_PARITY.exists():
        return {
            "accepted_full_fidelity": False,
            "blockers": [],
            "checks": {},
            "failed_threshold_check_count": 0,
            "schema": "free-boundary-strict-parity-benchmark.v1",
            "status": "not_run",
        }
    report = cast(
        dict[str, Any], json.loads(FREE_BOUNDARY_STRICT_PARITY.read_text(encoding="utf-8"))
    )
    if report.get("schema") != "free-boundary-strict-parity-benchmark.v1":
        raise ValueError("free-boundary strict parity benchmark schema mismatch")
    return report


def _sources_for(registry: dict[str, Any], surface: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in registry["sources"]:
        if source.get("surface") == surface:
            out.append(dict(source))
    return out


def _acceptance_surface(report: dict[str, Any], surface: str) -> dict[str, Any]:
    for row in report["surfaces"]:
        if row["surface"] == surface:
            return cast(dict[str, Any], row)
    raise KeyError(surface)


def _report_exists(path: str) -> bool:
    return (ROOT / path).exists()


def run_campaign() -> dict[str, Any]:
    """Return integrated full-fidelity status for all six blocker classes."""
    registry = _load_sources()
    downloads = _load_downloads()
    conversion = _load_conversion()
    dream_execution = _load_dream_execution()
    aurora_execution = _load_aurora_execution()
    gk_deck_inventory = _load_gk_deck_inventory()
    gk_external_parity = _load_gk_external_parity()
    gk_electromagnetic_fidelity = run_gk_electromagnetic_fidelity()
    production_decomposition = _load_production_decomposition()
    production_halo_face_integrity = production_decomposition.get(
        "halo_face_integrity_evidence",
        {
            "distributed_runtime_halo_exchange_ready": False,
            "halo_face_integrity_pass": False,
        },
    )
    free_boundary_machine_metadata = _load_free_boundary_machine_metadata()
    freegs_public_reconstruction = _load_freegs_public_reconstruction()
    free_boundary_strict_gate = _load_free_boundary_strict_parity()
    free_boundary_strict = cast(
        dict[str, Any],
        freegs_public_reconstruction["strict_free_boundary_parity_evidence"],
    )
    free_boundary_geometry = cast(
        dict[str, Any],
        free_boundary_strict.get(
            "geometry_containment_evidence",
            {
                "boundary_containment_metric_ready": False,
                "strict_geometry_containment_ready": False,
            },
        ),
    )
    acceptance = run_acceptance()
    runaway = run_runaway_contract(repeats=3)
    runaway_operator = runaway["native_kinetic_operator_evidence"]
    runaway_source_budget = runaway_operator["source_term_budget_evidence"]
    impurity = run_impurity_contract()
    impurity_operator = impurity["native_impurity_transport_evidence"]
    impurity_source_sink_budget = impurity_operator["source_sink_budget_evidence"]
    impurity_same_case = impurity_operator["same_case_aurora_strahl_comparison"]
    gk = _acceptance_surface(acceptance, "native_nonlinear_gyrokinetics")
    runaway_surface = _acceptance_surface(acceptance, "runaway_electrons")
    impurity_surface = _acceptance_surface(acceptance, "impurity_transport")
    sas_dataset_readiness = run_sas_dataset_readiness()

    lanes = [
        {
            "lane": "sas_dataset_readiness",
            "surface": "external_reference_data",
            "status": str(sas_dataset_readiness["status"]),
            "locally_actionable_contract_ready": bool(
                sas_dataset_readiness["manifest_present"]
                and sas_dataset_readiness["available_entries"]
                and sas_dataset_readiness["checksum_rows"]
            ),
            "reference_cases_ready": bool(
                sas_dataset_readiness["accepted_full_fidelity_dataset_ready"]
            ),
            "sources": [],
            "next_required_evidence": sas_dataset_readiness["next_required_evidence"],
        },
        {
            "lane": "gene_cgyro_gs2_nonlinear_gk_parity",
            "surface": "native_nonlinear_gyrokinetics",
            "status": (
                str(gk_external_parity["status"])
                if gk_external_parity["status"] != "not_run"
                else "blocked_public_gk_decks_indexed_missing_solver_output_parity"
                if gk_deck_inventory["deck_count"]
                else "blocked_missing_public_reference_artifacts"
            ),
            "locally_actionable_contract_ready": bool(gk["implemented_dimensions"]),
            "reference_cases_ready": bool(gk["reference_cases"]["ready"]),
            "sources": _sources_for(registry, "native_nonlinear_gyrokinetics"),
            "next_required_evidence": (
                gk_external_parity["missing_full_fidelity_requirements"]
                or gk_deck_inventory["missing_full_fidelity_requirements"]
                or gk["missing_requirements"]
            ),
        },
        {
            "lane": "full_maxwell_electromagnetic_fidelity",
            "surface": "native_nonlinear_gyrokinetics",
            "status": str(gk_electromagnetic_fidelity["status"]),
            "locally_actionable_contract_ready": bool(
                gk_electromagnetic_fidelity["locally_actionable_contract_ready"]
            ),
            "reference_cases_ready": bool(gk["reference_cases"]["ready"]),
            "sources": _sources_for(registry, "native_nonlinear_gyrokinetics"),
            "next_required_evidence": gk_electromagnetic_fidelity[
                "missing_full_fidelity_requirements"
            ],
        },
        {
            "lane": "production_scale_decomposition",
            "surface": "solver_runtime",
            "status": str(production_decomposition["status"]),
            "locally_actionable_contract_ready": bool(production_decomposition["contract_pass"]),
            "reference_cases_ready": False,
            "sources": [],
            "next_required_evidence": production_decomposition["missing_requirements"]
            or [
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
                and runaway_operator["native_artifact_ready"]
                and runaway_source_budget["all_observable_budgets_finite"]
                and runaway_source_budget["all_nonnegative_power_channels"]
                and runaway_surface["implemented_dimensions"].get(
                    "dream_style_multidimensional_artifact_export"
                )
            ),
            "reference_cases_ready": bool(runaway_surface["reference_cases"]["ready"]),
            "sources": _sources_for(registry, "runaway_electrons"),
            "next_required_evidence": (
                runaway_operator["blocking_requirements"] or runaway_surface["missing_requirements"]
            ),
        },
        {
            "lane": "aurora_strahl_grade_impurities",
            "surface": "impurity_transport",
            "status": (
                str(impurity_same_case["status"])
                if impurity_same_case["comparison_ready"]
                else "blocked_partial_public_atomic_artifact_not_transport_parity"
                if aurora_execution["reference_output_ready"]
                else "blocked_missing_public_aurora_strahl_artifacts"
            ),
            "locally_actionable_contract_ready": bool(
                impurity["passed"]
                and impurity_operator["native_artifact_ready"]
                and impurity_source_sink_budget["all_budget_terms_finite"]
                and impurity_source_sink_budget["source_sink_transfer_conservative"]
                and impurity_surface["implemented_dimensions"].get(
                    "charge_state_resolved_density_artifact_export"
                )
            ),
            "reference_cases_ready": bool(impurity_surface["reference_cases"]["ready"]),
            "sources": _sources_for(registry, "impurity_transport"),
            "next_required_evidence": (
                impurity_operator["blocking_requirements"]
                or aurora_execution["missing_full_fidelity_requirements"]
                or impurity_surface["missing_requirements"]
            ),
        },
        {
            "lane": "free_boundary_equilibrium_strict_parity",
            "surface": "free_boundary_equilibrium",
            "status": (
                free_boundary_strict_gate["status"]
                if freegs_public_reconstruction["case_count"]
                else "blocked_machine_metadata_indexed_missing_same_case_free_boundary_reconstruction"
                if free_boundary_machine_metadata["machine_metadata_ready"]
                else "blocked_missing_external_coil_current_reference_artifacts"
            ),
            "locally_actionable_contract_ready": _report_exists(
                "validation/reports/free_boundary_benchmark.json"
            )
            and bool(free_boundary_machine_metadata["machine_metadata_ready"])
            and bool(freegs_public_reconstruction["vacuum_comparison_pass"]),
            "reference_cases_ready": bool(
                free_boundary_strict_gate.get("accepted_full_fidelity") is True
            ),
            "sources": _sources_for(registry, "free_boundary_equilibrium"),
            "next_required_evidence": (
                []
                if free_boundary_strict_gate.get("accepted_full_fidelity") is True
                else free_boundary_strict_gate["blockers"]
                or free_boundary_strict["blocking_requirements"]
                or freegs_public_reconstruction["missing_full_fidelity_requirements"]
                or free_boundary_machine_metadata["missing_full_fidelity_requirements"]
                or [
                    "public coil-current sidecars or machine coil metadata for GEQDSK rows",
                    "strict FreeGS backend convergence on public free-boundary cases",
                    "profile-source/free-boundary reconstruction parity artifacts",
                ]
            ),
        },
    ]

    return {
        "benchmark": "full_fidelity_end_to_end_campaign",
        "schema": "full-fidelity-end-to-end-campaign.v1",
        "description": "Integrated fail-closed campaign covering GK, EM, production-scale runtime, DREAM, Aurora/STRAHL, and free-boundary blockers.",
        "sas_dataset_readiness_report": "validation/reports/sas_dataset_readiness.json",
        "sas_dataset_readiness_status": str(sas_dataset_readiness["status"]),
        "sas_dataset_available_entries": int(sas_dataset_readiness["available_entries"]),
        "sas_dataset_blocked_entries": int(sas_dataset_readiness["blocked_entries"]),
        "sas_dataset_checksum_rows": int(sas_dataset_readiness["checksum_rows"]),
        "sas_dataset_external_parity_outputs_ready": bool(
            sas_dataset_readiness["external_parity_outputs_ready"]
        ),
        "sas_dataset_accepted_full_fidelity_ready": bool(
            sas_dataset_readiness["accepted_full_fidelity_dataset_ready"]
        ),
        "sas_dataset_next_required_evidence": list(sas_dataset_readiness["next_required_evidence"]),
        "public_source_cache_root": downloads["cache_root"],
        "public_source_download_report": str(PUBLIC_SOURCE_DOWNLOADS.relative_to(ROOT)),
        "public_sources_cached": bool(downloads["all_reachable_downloads_completed"]),
        "public_reference_artifact_conversion_report": str(
            REFERENCE_ARTIFACT_CONVERSION.relative_to(ROOT)
        ),
        "partial_public_output_artifacts": int(conversion["partial_output_artifacts"]),
        "accepted_public_reference_artifacts": int(conversion["accepted_full_fidelity_artifacts"]),
        "dream_reference_execution_report": str(DREAM_EXECUTION_REQUEST.relative_to(ROOT)),
        "dream_settings_deck_generated": bool(dream_execution["settings_deck_generated"]),
        "dream_reference_output_ready": bool(dream_execution["reference_output_ready"]),
        "dream_reference_execution_status": str(dream_execution["status"]),
        "runaway_native_kinetic_operator_evidence_ready": bool(
            runaway_operator["native_artifact_ready"]
        ),
        "runaway_full_momentum_pitch_radius_operator_ready": bool(
            runaway_operator["full_momentum_pitch_radius_operator_ready"]
        ),
        "runaway_dream_same_case_threshold_ready": bool(
            runaway_operator["dream_same_case_threshold_ready"]
        ),
        "runaway_source_term_budget_evidence_ready": bool(
            runaway_source_budget["all_observable_budgets_finite"]
            and runaway_source_budget["all_nonnegative_power_channels"]
        ),
        "runaway_source_term_budget_dream_same_case_ready": bool(
            runaway_source_budget["dream_same_case_budget_ready"]
        ),
        "runaway_kinetic_operator_evidence_status": str(
            runaway_operator["operator_evidence_status"]
        ),
        "aurora_reference_execution_report": str(AURORA_EXECUTION_ARTIFACT.relative_to(ROOT)),
        "aurora_reference_artifact_generated": bool(aurora_execution["artifact_generated"]),
        "aurora_reference_output_ready": bool(aurora_execution["reference_output_ready"]),
        "aurora_reference_execution_status": str(aurora_execution["status"]),
        "impurity_native_transport_evidence_ready": bool(
            impurity_operator["native_artifact_ready"]
        ),
        "impurity_charge_state_radial_transport_operator_ready": bool(
            impurity_operator["charge_state_radial_transport_operator_ready"]
        ),
        "impurity_aurora_strahl_same_case_threshold_ready": bool(
            impurity_operator["aurora_strahl_same_case_threshold_ready"]
        ),
        "impurity_aurora_strahl_same_case_comparison_ready": bool(
            impurity_operator["aurora_strahl_same_case_comparison_ready"]
        ),
        "impurity_aurora_strahl_same_case_threshold_passed": bool(
            impurity_operator["aurora_strahl_same_case_threshold_passed"]
        ),
        "impurity_aurora_strahl_same_case_comparison_status": str(impurity_same_case["status"]),
        "impurity_source_sink_budget_evidence_ready": bool(
            impurity_source_sink_budget["all_budget_terms_finite"]
            and impurity_source_sink_budget["source_sink_transfer_conservative"]
            and impurity_source_sink_budget["line_radiation_nonnegative"]
        ),
        "impurity_source_sink_budget_aurora_strahl_same_case_ready": bool(
            impurity_source_sink_budget["aurora_strahl_same_case_budget_ready"]
        ),
        "impurity_transport_operator_evidence_status": str(
            impurity_operator["operator_evidence_status"]
        ),
        "gk_public_deck_inventory_report": str(GK_DECK_INVENTORY.relative_to(ROOT)),
        "gk_public_decks_indexed": int(gk_deck_inventory["deck_count"]),
        "gk_public_outputs_indexed": int(gk_deck_inventory["output_summary_count"]),
        "gk_public_deck_inventory_status": str(gk_deck_inventory["status"]),
        "gk_external_nonlinear_parity_report": str(GK_EXTERNAL_PARITY.relative_to(ROOT)),
        "gk_electromagnetic_fidelity_report": str(GK_ELECTROMAGNETIC_FIDELITY.relative_to(ROOT)),
        "gk_electromagnetic_fidelity_status": str(gk_electromagnetic_fidelity["status"]),
        "gk_electromagnetic_compact_closure_ready": bool(
            gk_electromagnetic_fidelity["compact_em_contract_ready"]
        ),
        "gk_electromagnetic_grid_convergence_ready": bool(
            gk_electromagnetic_fidelity["electromagnetic_grid_convergence_ready"]
        ),
        "gk_electromagnetic_maxwell_evolution_ready": bool(
            gk_electromagnetic_fidelity["maxwell_evolution_evidence"]["status"]
            == "accepted_local_source_free_maxwell_evolution"
        ),
        "gk_electromagnetic_native_same_case_thresholds_ready": bool(
            gk_electromagnetic_fidelity["native_em_same_case_threshold_evidence"][
                "same_case_thresholds_ready"
            ]
        ),
        "gk_electromagnetic_self_consistent_kinetic_current_ready": bool(
            gk_electromagnetic_fidelity["maxwell_evolution_evidence"][
                "self_consistent_kinetic_current_supported"
            ]
        ),
        "gk_electromagnetic_external_parity_ready": bool(
            gk_electromagnetic_fidelity["external_em_parity_comparison_ready"]
        ),
        "gk_external_reference_artifacts_converted": int(
            gk_external_parity["converted_reference_artifacts"]
        ),
        "gk_external_reference_output_ready": bool(gk_external_parity["reference_output_ready"]),
        "gk_native_same_case_comparison_ready": bool(
            gk_external_parity["native_same_case_comparison_ready"]
        ),
        "gk_grid_convergence_ready": bool(gk_external_parity["grid_convergence_ready"]),
        "gk_production_scale_scaling_ready": bool(
            gk_external_parity["production_scale_scaling_ready"]
        ),
        "gk_evidence_package_ready": bool(gk_external_parity["evidence_package_ready"]),
        "gk_external_nonlinear_parity_status": str(gk_external_parity["status"]),
        "production_decomposition_report": str(PRODUCTION_DECOMPOSITION.relative_to(ROOT)),
        "production_decomposition_contract_pass": bool(production_decomposition["contract_pass"]),
        "production_decomposition_halo_face_integrity_ready": bool(
            production_halo_face_integrity["halo_face_integrity_pass"]
        ),
        "production_decomposition_distributed_halo_exchange_ready": bool(
            production_halo_face_integrity["distributed_runtime_halo_exchange_ready"]
        ),
        "production_scale_ready": bool(production_decomposition["production_scale_ready"]),
        "production_decomposition_status": str(production_decomposition["status"]),
        "free_boundary_machine_metadata_report": str(
            FREE_BOUNDARY_MACHINE_METADATA.relative_to(ROOT)
        ),
        "free_boundary_machine_metadata_indexed": int(
            free_boundary_machine_metadata["machine_config_count"]
        ),
        "free_boundary_machine_metadata_ready": bool(
            free_boundary_machine_metadata["machine_metadata_ready"]
        ),
        "free_boundary_machine_metadata_status": str(free_boundary_machine_metadata["status"]),
        "freegs_public_example_reconstruction_report": str(
            FREEGS_PUBLIC_RECONSTRUCTION.relative_to(ROOT)
        ),
        "free_boundary_strict_parity_report": str(FREE_BOUNDARY_STRICT_PARITY.relative_to(ROOT)),
        "freegs_public_example_cases": int(freegs_public_reconstruction["case_count"]),
        "freegs_public_example_vacuum_comparison_pass": bool(
            freegs_public_reconstruction["vacuum_comparison_pass"]
        ),
        "freegs_public_example_external_output_ready": bool(
            freegs_public_reconstruction.get("external_nonlinear_output_ready", False)
        ),
        "freegs_public_example_reconstruction_status": str(freegs_public_reconstruction["status"]),
        "free_boundary_strict_threshold_acceptance_ready": bool(
            free_boundary_strict_gate.get("checks", {}).get(
                "strict_threshold_acceptance_ready",
                free_boundary_strict["strict_threshold_acceptance_ready"],
            )
        ),
        "free_boundary_geometry_containment_ready": bool(
            free_boundary_strict_gate.get("checks", {}).get(
                "geometry_containment_ready",
                free_boundary_geometry["strict_geometry_containment_ready"],
            )
        ),
        "free_boundary_boundary_containment_metric_ready": bool(
            free_boundary_strict_gate.get("checks", {}).get(
                "boundary_containment_metric_ready",
                free_boundary_geometry["boundary_containment_metric_ready"],
            )
        ),
        "free_boundary_grid_convergence_ready": bool(
            free_boundary_strict_gate.get("checks", {}).get(
                "grid_convergence_ready",
                free_boundary_strict["grid_convergence_ready"],
            )
        ),
        "free_boundary_coil_vacuum_sidecar_ready": bool(
            free_boundary_strict_gate.get("checks", {}).get(
                "coil_vacuum_sidecar_ready",
                free_boundary_strict["coil_vacuum_sidecar_ready"],
            )
        ),
        "free_boundary_same_case_public_reference_output_ready": bool(
            free_boundary_strict_gate.get("checks", {}).get(
                "same_case_public_reference_output_ready", False
            )
        ),
        "free_boundary_failed_threshold_check_count": int(
            free_boundary_strict_gate.get(
                "failed_threshold_check_count",
                free_boundary_strict["failed_threshold_check_count"],
            )
        ),
        "free_boundary_strict_parity_status": str(
            free_boundary_strict_gate.get("status", free_boundary_strict["status"])
        ),
        "free_boundary_strict_parity_blockers": list(
            free_boundary_strict_gate.get("blockers", free_boundary_strict["blocking_requirements"])
        ),
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
        f"- SAS dataset readiness report: `{report['sas_dataset_readiness_report']}`",
        f"- SAS dataset readiness status: `{report['sas_dataset_readiness_status']}`",
        f"- SAS dataset available entries: `{report['sas_dataset_available_entries']}`",
        f"- SAS dataset blocked entries: `{report['sas_dataset_blocked_entries']}`",
        f"- SAS dataset checksum rows: `{report['sas_dataset_checksum_rows']}`",
        (
            "- SAS dataset external parity outputs ready: "
            f"`{report['sas_dataset_external_parity_outputs_ready']}`"
        ),
        (
            "- SAS dataset accepted full-fidelity ready: "
            f"`{report['sas_dataset_accepted_full_fidelity_ready']}`"
        ),
        f"- Public source registry: `{report['public_source_registry']}`",
        f"- Public source download report: `{report['public_source_download_report']}`",
        f"- Public sources cached: `{report['public_sources_cached']}`",
        f"- Public source cache root: `{report['public_source_cache_root']}`",
        (
            "- Public reference artifact conversion report: "
            f"`{report['public_reference_artifact_conversion_report']}`"
        ),
        f"- Partial public output artifacts: `{report['partial_public_output_artifacts']}`",
        (
            "- Accepted public reference artifacts: "
            f"`{report['accepted_public_reference_artifacts']}`"
        ),
        f"- DREAM execution report: `{report['dream_reference_execution_report']}`",
        f"- DREAM settings deck generated: `{report['dream_settings_deck_generated']}`",
        f"- DREAM reference output ready: `{report['dream_reference_output_ready']}`",
        f"- DREAM execution status: `{report['dream_reference_execution_status']}`",
        (
            "- Runaway native kinetic operator evidence ready: "
            f"`{report['runaway_native_kinetic_operator_evidence_ready']}`"
        ),
        (
            "- Runaway full momentum-pitch-radius operator ready: "
            f"`{report['runaway_full_momentum_pitch_radius_operator_ready']}`"
        ),
        (
            "- Runaway DREAM same-case thresholds ready: "
            f"`{report['runaway_dream_same_case_threshold_ready']}`"
        ),
        (
            "- Runaway source-term budget evidence ready: "
            f"`{report['runaway_source_term_budget_evidence_ready']}`"
        ),
        (
            "- Runaway source-term DREAM same-case budget ready: "
            f"`{report['runaway_source_term_budget_dream_same_case_ready']}`"
        ),
        (
            "- Runaway kinetic operator evidence status: "
            f"`{report['runaway_kinetic_operator_evidence_status']}`"
        ),
        f"- Aurora execution report: `{report['aurora_reference_execution_report']}`",
        f"- Aurora artifact generated: `{report['aurora_reference_artifact_generated']}`",
        f"- Aurora reference output ready: `{report['aurora_reference_output_ready']}`",
        f"- Aurora execution status: `{report['aurora_reference_execution_status']}`",
        (
            "- Impurity native transport evidence ready: "
            f"`{report['impurity_native_transport_evidence_ready']}`"
        ),
        (
            "- Impurity charge-state radial transport operator ready: "
            f"`{report['impurity_charge_state_radial_transport_operator_ready']}`"
        ),
        (
            "- Impurity Aurora/STRAHL same-case comparison ready: "
            f"`{report['impurity_aurora_strahl_same_case_comparison_ready']}`"
        ),
        (
            "- Impurity Aurora/STRAHL same-case threshold checks ready: "
            f"`{report['impurity_aurora_strahl_same_case_threshold_ready']}`"
        ),
        (
            "- Impurity Aurora/STRAHL same-case thresholds passed: "
            f"`{report['impurity_aurora_strahl_same_case_threshold_passed']}`"
        ),
        (
            "- Impurity Aurora/STRAHL same-case comparison status: "
            f"`{report['impurity_aurora_strahl_same_case_comparison_status']}`"
        ),
        (
            "- Impurity source/sink budget evidence ready: "
            f"`{report['impurity_source_sink_budget_evidence_ready']}`"
        ),
        (
            "- Impurity source/sink Aurora/STRAHL same-case budget ready: "
            f"`{report['impurity_source_sink_budget_aurora_strahl_same_case_ready']}`"
        ),
        (
            "- Impurity transport operator evidence status: "
            f"`{report['impurity_transport_operator_evidence_status']}`"
        ),
        f"- GK deck inventory report: `{report['gk_public_deck_inventory_report']}`",
        f"- GK public decks indexed: `{report['gk_public_decks_indexed']}`",
        f"- GK public outputs indexed: `{report['gk_public_outputs_indexed']}`",
        f"- GK deck inventory status: `{report['gk_public_deck_inventory_status']}`",
        f"- GK electromagnetic fidelity report: `{report['gk_electromagnetic_fidelity_report']}`",
        f"- GK electromagnetic fidelity status: `{report['gk_electromagnetic_fidelity_status']}`",
        (
            "- GK electromagnetic compact closure ready: "
            f"`{report['gk_electromagnetic_compact_closure_ready']}`"
        ),
        (
            "- GK electromagnetic grid convergence ready: "
            f"`{report['gk_electromagnetic_grid_convergence_ready']}`"
        ),
        (
            "- GK electromagnetic Maxwell evolution ready: "
            f"`{report['gk_electromagnetic_maxwell_evolution_ready']}`"
        ),
        (
            "- GK electromagnetic native same-case thresholds ready: "
            f"`{report['gk_electromagnetic_native_same_case_thresholds_ready']}`"
        ),
        (
            "- GK electromagnetic self-consistent kinetic current ready: "
            f"`{report['gk_electromagnetic_self_consistent_kinetic_current_ready']}`"
        ),
        (
            "- GK electromagnetic external parity ready: "
            f"`{report['gk_electromagnetic_external_parity_ready']}`"
        ),
        f"- Production decomposition report: `{report['production_decomposition_report']}`",
        (
            "- Production decomposition contract pass: "
            f"`{report['production_decomposition_contract_pass']}`"
        ),
        (
            "- Production decomposition halo-face integrity ready: "
            f"`{report['production_decomposition_halo_face_integrity_ready']}`"
        ),
        (
            "- Production decomposition distributed halo exchange ready: "
            f"`{report['production_decomposition_distributed_halo_exchange_ready']}`"
        ),
        f"- Production-scale ready: `{report['production_scale_ready']}`",
        f"- Production decomposition status: `{report['production_decomposition_status']}`",
        (
            "- Free-boundary machine metadata report: "
            f"`{report['free_boundary_machine_metadata_report']}`"
        ),
        (
            "- Free-boundary machine metadata indexed: "
            f"`{report['free_boundary_machine_metadata_indexed']}`"
        ),
        (
            "- Free-boundary machine metadata ready: "
            f"`{report['free_boundary_machine_metadata_ready']}`"
        ),
        (
            "- Free-boundary machine metadata status: "
            f"`{report['free_boundary_machine_metadata_status']}`"
        ),
        (
            "- FreeGS public example reconstruction report: "
            f"`{report['freegs_public_example_reconstruction_report']}`"
        ),
        f"- FreeGS public example cases: `{report['freegs_public_example_cases']}`",
        (
            "- FreeGS public example vacuum comparison pass: "
            f"`{report['freegs_public_example_vacuum_comparison_pass']}`"
        ),
        (
            "- FreeGS public example external output ready: "
            f"`{report['freegs_public_example_external_output_ready']}`"
        ),
        (
            "- FreeGS public example reconstruction status: "
            f"`{report['freegs_public_example_reconstruction_status']}`"
        ),
        (
            "- Free-boundary strict threshold acceptance ready: "
            f"`{report['free_boundary_strict_threshold_acceptance_ready']}`"
        ),
        (
            "- Free-boundary geometry containment ready: "
            f"`{report['free_boundary_geometry_containment_ready']}`"
        ),
        (
            "- Free-boundary boundary-containment metric ready: "
            f"`{report['free_boundary_boundary_containment_metric_ready']}`"
        ),
        (
            "- Free-boundary grid convergence ready: "
            f"`{report['free_boundary_grid_convergence_ready']}`"
        ),
        (
            "- Free-boundary coil/vacuum sidecar ready: "
            f"`{report['free_boundary_coil_vacuum_sidecar_ready']}`"
        ),
        (
            "- Free-boundary same-case public reference output ready: "
            f"`{report['free_boundary_same_case_public_reference_output_ready']}`"
        ),
        (
            "- Free-boundary failed threshold checks: "
            f"`{report['free_boundary_failed_threshold_check_count']}`"
        ),
        (f"- Free-boundary strict parity status: `{report['free_boundary_strict_parity_status']}`"),
        (
            "- Free-boundary strict parity blockers: "
            f"`{', '.join(report['free_boundary_strict_parity_blockers'])}`"
        ),
        (f"- Free-boundary strict parity report: `{report['free_boundary_strict_parity_report']}`"),
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
