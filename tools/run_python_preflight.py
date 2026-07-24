# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Compose and execute release/research preflight command sequences."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECK_TIMEOUT_SECONDS = 1800.0
_PREFLIGHT_LATENCY_OUTPUT_JSON = "artifacts/_tmp_scpn_end_to_end_latency.json"
_PREFLIGHT_LATENCY_OUTPUT_MD = "artifacts/_tmp_scpn_end_to_end_latency.md"
_PREFLIGHT_VERTICAL_CONTROL_OUTPUT_JSON = "artifacts/_tmp_vertical_control_replay_profiles.json"
_PREFLIGHT_VERTICAL_CONTROL_OUTPUT_MD = "artifacts/_tmp_vertical_control_replay_profiles.md"
_CLAIMS_EVIDENCE_MAP = REPO_ROOT / "docs" / "internal" / "CLAIMS_EVIDENCE_MAP.md"
_INTERNAL_READINESS_REGISTER = REPO_ROOT / "docs" / "internal" / "INTERNAL_READINESS_REGISTER.md"
_INTERNAL_READINESS_SOURCE = REPO_ROOT / "docs" / "internal" / "INTERNAL_READINESS_SOURCE.md"
_INTERNAL_READINESS_DOCS_CLAIMS = (
    REPO_ROOT / "docs" / "internal" / "INTERNAL_READINESS_DOCS_CLAIMS.md"
)
_INTERNAL_READINESS_SUMMARY = REPO_ROOT / "docs" / "internal" / "INTERNAL_READINESS_SUMMARY.json"
_SOURCE_P0P1_READINESS_MD = REPO_ROOT / "docs" / "internal" / "SOURCE_P0P1_READINESS.md"
_SOURCE_P0P1_READINESS_JSON = REPO_ROOT / "docs" / "internal" / "SOURCE_P0P1_READINESS.json"
_RELEASE_READINESS = REPO_ROOT / "docs" / "RELEASE_READINESS.md"


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _internal_files_available(*paths: Path) -> bool:
    """Return true when all private gitignored governance inputs exist locally."""
    return all(path.exists() for path in paths)


def _normalize_check_timeout_seconds(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("check_timeout_seconds must be finite and > 0.")
    return timeout


def _build_release_checks(
    *,
    skip_version_metadata: bool,
    skip_packaging_contract: bool,
    skip_lfs_hygiene: bool,
    skip_claims_audit: bool,
    skip_claim_range_guard: bool,
    skip_claims_map: bool,
    skip_safety_traceability: bool,
    skip_psi_gate_attribution: bool,
    skip_surrogate_uq_cards: bool,
    skip_torax_imas_interchange: bool,
    skip_torax_same_physics_config: bool,
    skip_fair_validation_packs: bool,
    skip_capability_manifest: bool,
    skip_readiness_register: bool,
    skip_readiness_scope_reports: bool,
    skip_release_delta_guard: bool,
    skip_source_readiness: bool,
    skip_untested_module_guard: bool,
    skip_deprecated_default_lane_guard: bool,
    skip_release_readiness: bool,
    skip_shot_manifest: bool,
    skip_reference_data_provenance: bool,
    skip_shot_splits: bool,
    skip_disruption_calibration: bool,
    skip_disruption_replay_pipeline: bool,
    skip_disruption_transfer_generalization: bool,
    skip_eped_domain_contract: bool,
    skip_transport_uncertainty: bool,
    skip_vertical_control_replay: bool,
    skip_torax_strict_backend: bool,
    skip_sparc_strict_backend: bool,
    skip_freegs_strict_backend: bool,
    skip_multi_ion_conservation: bool,
    skip_end_to_end_latency: bool,
    skip_notebook_quality: bool,
    skip_threshold_smoke: bool,
    skip_mypy: bool,
    enable_strict_backend_checks: bool,
    enable_freegs_strict_backend_check: bool,
) -> list[tuple[str, list[str]]]:
    checks: list[tuple[str, list[str]]] = []
    if not skip_version_metadata:
        # Hardening: Run metadata sync first to ensure consistency
        checks.append(
            (
                "Metadata drift check",
                [
                    sys.executable,
                    "tools/sync_metadata.py",
                    "--check",
                ],
            )
        )
        checks.append(
            (
                "Version metadata consistency",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_version_metadata.py",
                    "-q",
                ],
            )
        )
    if not skip_packaging_contract:
        checks.append(
            (
                "Packaging contract guard",
                [
                    sys.executable,
                    "tools/check_packaging_contract.py",
                    "--summary-json",
                    "artifacts/packaging_contract_summary.json",
                ],
            )
        )
    if not skip_lfs_hygiene:
        checks.append(
            (
                "Git LFS hygiene guard",
                [
                    sys.executable,
                    "tools/check_lfs_hygiene.py",
                ],
            )
        )
    if not skip_claims_audit:
        checks.append(
            (
                "Claims evidence audit",
                [
                    sys.executable,
                    "tools/claims_audit.py",
                ],
            )
        )
    if not skip_claim_range_guard:
        checks.append(
            (
                "Claims range guard",
                [
                    sys.executable,
                    "tools/claim_range_guard.py",
                    "--summary-json",
                    "artifacts/claim_range_guard_summary.json",
                ],
            )
        )
    if not skip_claims_map and _internal_files_available(_CLAIMS_EVIDENCE_MAP):
        checks.append(
            (
                "Claims evidence map drift check",
                [
                    sys.executable,
                    "tools/generate_claims_evidence_map.py",
                    "--check",
                ],
            )
        )
    if not skip_safety_traceability:
        checks.append(
            (
                "Safety traceability matrix drift check",
                [
                    sys.executable,
                    "tools/generate_safety_traceability.py",
                    "--check",
                ],
            )
        )
    if not skip_psi_gate_attribution:
        checks.append(
            (
                "Psi gate attribution drift check",
                [
                    sys.executable,
                    "tools/generate_psi_gate_attribution.py",
                    "--check",
                ],
            )
        )
    if not skip_surrogate_uq_cards:
        checks.append(
            (
                "Surrogate UQ cards drift check",
                [
                    sys.executable,
                    "tools/generate_surrogate_uq_cards.py",
                    "--check",
                ],
            )
        )
    if not skip_torax_imas_interchange:
        checks.append(
            (
                "TORAX IMAS interchange drift check",
                [
                    sys.executable,
                    "validation/torax_imas_interchange.py",
                    "--check",
                ],
            )
        )
    if not skip_torax_same_physics_config:
        checks.append(
            (
                "TORAX same-physics config drift check",
                [
                    sys.executable,
                    "validation/torax_same_physics_config_study.py",
                    "--check",
                ],
            )
        )
    if not skip_fair_validation_packs:
        checks.append(
            (
                "FAIR validation packs drift check",
                [
                    sys.executable,
                    "tools/export_zenodo_dataset.py",
                    "--check",
                    "--no-export",
                ],
            )
        )
    if not skip_capability_manifest:
        checks.append(
            (
                "Capability manifest drift check",
                [
                    sys.executable,
                    "tools/capability_manifest.py",
                    "--check",
                ],
            )
        )
    if not skip_readiness_register and _internal_files_available(_INTERNAL_READINESS_REGISTER):
        checks.append(
            (
                "Internal readiness register drift check",
                [
                    sys.executable,
                    "tools/generate_readiness_register.py",
                    "--check",
                ],
            )
        )
    if not skip_readiness_scope_reports and _internal_files_available(
        _INTERNAL_READINESS_SOURCE,
        _INTERNAL_READINESS_DOCS_CLAIMS,
        _INTERNAL_READINESS_SUMMARY,
    ):
        checks.append(
            (
                "Internal readiness scope-report drift check",
                [
                    sys.executable,
                    "tools/generate_readiness_scope_reports.py",
                    "--check",
                ],
            )
        )
    if not skip_release_delta_guard and _internal_files_available(_INTERNAL_READINESS_SUMMARY):
        checks.append(
            (
                "Release delta non-regression guard",
                [
                    sys.executable,
                    "tools/release_delta_guard.py",
                    "--summary-json",
                    "artifacts/release_delta_guard_summary.json",
                ],
            )
        )
    if not skip_source_readiness and _internal_files_available(
        _SOURCE_P0P1_READINESS_MD,
        _SOURCE_P0P1_READINESS_JSON,
    ):
        checks.append(
            (
                "Source P0/P1 source readiness drift check",
                [
                    sys.executable,
                    "tools/generate_source_p0p1_readiness.py",
                    "--check",
                ],
            )
        )
    if not skip_untested_module_guard:
        checks.append(
            (
                "Untested module linkage guard",
                [
                    sys.executable,
                    "tools/check_test_module_linkage.py",
                    "--summary-json",
                    "artifacts/untested_module_guard_summary.json",
                ],
            )
        )
    if not skip_deprecated_default_lane_guard:
        checks.append(
            (
                "Deprecated default lane guard",
                [
                    sys.executable,
                    "tools/deprecated_default_lane_guard.py",
                ],
            )
        )
    if not skip_release_readiness and _internal_files_available(_RELEASE_READINESS):
        checks.append(
            (
                "Release readiness gate",
                [
                    sys.executable,
                    "tools/check_release_readiness.py",
                ],
            )
        )
    if not skip_shot_manifest:
        checks.append(
            (
                "Disruption shot provenance manifest check",
                [
                    sys.executable,
                    "tools/generate_disruption_shot_manifest.py",
                    "--check",
                ],
            )
        )
    if not skip_reference_data_provenance:
        checks.append(
            (
                "Reference-data provenance manifest check",
                [
                    sys.executable,
                    "tools/generate_reference_data_provenance_manifest.py",
                    "--check",
                ],
            )
        )
    if not skip_shot_splits:
        checks.append(
            (
                "Disruption shot split leakage check",
                [
                    sys.executable,
                    "tools/check_disruption_shot_splits.py",
                ],
            )
        )
    if not skip_disruption_calibration:
        checks.append(
            (
                "Disruption risk calibration holdout check",
                [
                    sys.executable,
                    "tools/generate_disruption_risk_calibration.py",
                    "--check",
                ],
            )
        )
    if not skip_disruption_replay_pipeline:
        checks.append(
            (
                "Disruption replay pipeline contract benchmark",
                [
                    sys.executable,
                    "validation/benchmark_disruption_replay_pipeline.py",
                    "--strict",
                ],
            )
        )
    if not skip_disruption_transfer_generalization:
        checks.append(
            (
                "Disruption transfer-generalization benchmark",
                [
                    sys.executable,
                    "validation/benchmark_disruption_transfer_generalization.py",
                    "--strict",
                ],
            )
        )
    if not skip_eped_domain_contract:
        checks.append(
            (
                "EPED domain contract benchmark",
                [
                    sys.executable,
                    "validation/benchmark_eped_domain_contract.py",
                    "--strict",
                ],
            )
        )
    if not skip_transport_uncertainty:
        checks.append(
            (
                "Transport uncertainty envelope benchmark",
                [
                    sys.executable,
                    "validation/benchmark_transport_uncertainty_envelope.py",
                    "--strict",
                ],
            )
        )
    if not skip_vertical_control_replay:
        checks.append(
            (
                "Vertical-control replay profile-suite benchmark",
                [
                    sys.executable,
                    "validation/vertical_control_replay_benchmark.py",
                    "--all-profiles",
                    "--strict",
                    "--output-json",
                    _PREFLIGHT_VERTICAL_CONTROL_OUTPUT_JSON,
                    "--output-md",
                    _PREFLIGHT_VERTICAL_CONTROL_OUTPUT_MD,
                ],
            )
        )
    if enable_strict_backend_checks and not skip_torax_strict_backend:
        checks.append(
            (
                "TORAX real-reference parity drift check",
                [
                    sys.executable,
                    "validation/benchmark_torax_real_parity.py",
                    "--check",
                ],
            )
        )
    if enable_strict_backend_checks and not skip_sparc_strict_backend:
        checks.append(
            (
                "SPARC GEQDSK strict-backend benchmark",
                [
                    sys.executable,
                    "validation/benchmark_sparc_geqdsk_rmse.py",
                    "--strict-backend",
                ],
            )
        )
    if (
        enable_strict_backend_checks
        and enable_freegs_strict_backend_check
        and not skip_freegs_strict_backend
    ):
        checks.append(
            (
                "FreeGS/FreeGSNKE strict parity check",
                [
                    sys.executable,
                    "validation/benchmark_free_boundary_strict_parity.py",
                    "--check",
                    "--strict",
                ],
            )
        )
    if not skip_multi_ion_conservation:
        checks.append(
            (
                "Multi-ion transport conservation benchmark",
                [
                    sys.executable,
                    "validation/benchmark_multi_ion_transport_conservation.py",
                    "--strict",
                ],
            )
        )
    if not skip_end_to_end_latency:
        checks.append(
            (
                "SCPN end-to-end latency benchmark",
                [
                    sys.executable,
                    "validation/scpn_end_to_end_latency.py",
                    "--strict",
                    "--output-json",
                    _PREFLIGHT_LATENCY_OUTPUT_JSON,
                    "--output-md",
                    _PREFLIGHT_LATENCY_OUTPUT_MD,
                ],
            )
        )
    if not skip_notebook_quality:
        checks.append(
            (
                "Golden notebook quality gate",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_neuro_symbolic_control_demo_notebook.py",
                    "-q",
                ],
            )
        )
    if not skip_threshold_smoke:
        checks.append(
            (
                "Task 5/6 threshold smoke",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_task5_disruption_mitigation_integration.py::test_task5_campaign_passes_thresholds_smoke",
                    "tests/test_task6_heating_neutronics_realism.py::test_task6_campaign_passes_thresholds_smoke",
                    "-q",
                ],
            )
        )
    if not skip_mypy:
        checks.append(("mypy expansion guard", [sys.executable, "tools/mypy_expansion_guard.py"]))
        checks.append(("mypy strict", [sys.executable, "tools/run_mypy_strict.py"]))
    checks.append(("docstring numpy convention", [sys.executable, "tools/run_ruff_docstrings.py"]))
    return checks


def _build_research_checks(*, skip_research_suite: bool) -> list[tuple[str, list[str]]]:
    checks: list[tuple[str, list[str]]] = []
    if not skip_research_suite:
        checks.append(
            (
                "Hermetic experimental pytest suite",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/",
                    "-q",
                    "-m",
                    "experimental and not (external_reference or dedicated_hardware)",
                ],
            )
        )
    return checks


def _build_checks(
    *,
    gate: str,
    skip_version_metadata: bool,
    skip_packaging_contract: bool,
    skip_lfs_hygiene: bool,
    skip_claims_audit: bool,
    skip_claim_range_guard: bool,
    skip_claims_map: bool,
    skip_safety_traceability: bool,
    skip_psi_gate_attribution: bool,
    skip_surrogate_uq_cards: bool,
    skip_torax_imas_interchange: bool,
    skip_torax_same_physics_config: bool,
    skip_fair_validation_packs: bool,
    skip_capability_manifest: bool,
    skip_readiness_register: bool,
    skip_readiness_scope_reports: bool,
    skip_release_delta_guard: bool,
    skip_source_readiness: bool,
    skip_untested_module_guard: bool,
    skip_deprecated_default_lane_guard: bool,
    skip_release_readiness: bool,
    skip_shot_manifest: bool,
    skip_reference_data_provenance: bool,
    skip_shot_splits: bool,
    skip_disruption_calibration: bool,
    skip_disruption_replay_pipeline: bool,
    skip_disruption_transfer_generalization: bool,
    skip_eped_domain_contract: bool,
    skip_transport_uncertainty: bool,
    skip_vertical_control_replay: bool,
    skip_torax_strict_backend: bool,
    skip_sparc_strict_backend: bool,
    skip_freegs_strict_backend: bool,
    skip_multi_ion_conservation: bool,
    skip_end_to_end_latency: bool,
    skip_notebook_quality: bool,
    skip_threshold_smoke: bool,
    skip_mypy: bool,
    skip_research_suite: bool,
    enable_strict_backend_checks: bool,
    enable_freegs_strict_backend_check: bool,
) -> list[tuple[str, list[str]]]:
    checks: list[tuple[str, list[str]]] = []
    if gate in {"release", "all"}:
        checks.extend(
            _build_release_checks(
                skip_version_metadata=skip_version_metadata,
                skip_packaging_contract=skip_packaging_contract,
                skip_lfs_hygiene=skip_lfs_hygiene,
                skip_claims_audit=skip_claims_audit,
                skip_claim_range_guard=skip_claim_range_guard,
                skip_claims_map=skip_claims_map,
                skip_safety_traceability=skip_safety_traceability,
                skip_psi_gate_attribution=skip_psi_gate_attribution,
                skip_surrogate_uq_cards=skip_surrogate_uq_cards,
                skip_torax_imas_interchange=skip_torax_imas_interchange,
                skip_torax_same_physics_config=skip_torax_same_physics_config,
                skip_fair_validation_packs=skip_fair_validation_packs,
                skip_capability_manifest=skip_capability_manifest,
                skip_readiness_register=skip_readiness_register,
                skip_readiness_scope_reports=skip_readiness_scope_reports,
                skip_release_delta_guard=skip_release_delta_guard,
                skip_source_readiness=skip_source_readiness,
                skip_untested_module_guard=skip_untested_module_guard,
                skip_deprecated_default_lane_guard=skip_deprecated_default_lane_guard,
                skip_release_readiness=skip_release_readiness,
                skip_shot_manifest=skip_shot_manifest,
                skip_reference_data_provenance=skip_reference_data_provenance,
                skip_shot_splits=skip_shot_splits,
                skip_disruption_calibration=skip_disruption_calibration,
                skip_disruption_replay_pipeline=skip_disruption_replay_pipeline,
                skip_disruption_transfer_generalization=skip_disruption_transfer_generalization,
                skip_eped_domain_contract=skip_eped_domain_contract,
                skip_transport_uncertainty=skip_transport_uncertainty,
                skip_vertical_control_replay=skip_vertical_control_replay,
                skip_torax_strict_backend=skip_torax_strict_backend,
                skip_sparc_strict_backend=skip_sparc_strict_backend,
                skip_freegs_strict_backend=skip_freegs_strict_backend,
                skip_multi_ion_conservation=skip_multi_ion_conservation,
                skip_end_to_end_latency=skip_end_to_end_latency,
                skip_notebook_quality=skip_notebook_quality,
                skip_threshold_smoke=skip_threshold_smoke,
                skip_mypy=skip_mypy,
                enable_strict_backend_checks=enable_strict_backend_checks,
                enable_freegs_strict_backend_check=enable_freegs_strict_backend_check,
            )
        )
    if gate in {"research", "all"}:
        checks.extend(_build_research_checks(skip_research_suite=skip_research_suite))
    return checks


def _run_check(name: str, cmd: list[str], *, timeout_seconds: float) -> int:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"[preflight] {name}: {rendered}")
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(
            (f"[preflight] TIMEOUT at '{name}' after {timeout_seconds:.1f}s."),
            file=sys.stderr,
        )
        return 124
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    """Run selected preflight gate checks and stop on first critical failure.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when all selected checks pass, ``1`` when any check fails or a
        timeout occurs.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run local/CI Python preflight checks with gate profiles (release, research, or both)."
        )
    )
    parser.add_argument(
        "--gate",
        choices=("release", "research", "all"),
        default="release",
        help=(
            "Gate profile to run. "
            "'release' excludes experimental-only lanes, "
            "'research' runs experimental tests that need neither external references "
            "nor dedicated hardware, "
            "'all' runs both."
        ),
    )
    parser.add_argument(
        "--skip-version-metadata",
        action="store_true",
        help="Skip tests/test_version_metadata.py",
    )
    parser.add_argument(
        "--skip-packaging-contract",
        action="store_true",
        help="Skip tools/check_packaging_contract.py",
    )
    parser.add_argument(
        "--skip-lfs-hygiene",
        action="store_true",
        help="Skip tools/check_lfs_hygiene.py",
    )
    parser.add_argument(
        "--skip-notebook-quality",
        action="store_true",
        help="Skip tests/test_neuro_symbolic_control_demo_notebook.py",
    )
    parser.add_argument(
        "--skip-claims-audit",
        action="store_true",
        help="Skip tools/claims_audit.py",
    )
    parser.add_argument(
        "--skip-claim-range-guard",
        action="store_true",
        help="Skip tools/claim_range_guard.py",
    )
    parser.add_argument(
        "--skip-claims-map",
        action="store_true",
        help="Skip tools/generate_claims_evidence_map.py --check",
    )
    parser.add_argument(
        "--skip-safety-traceability",
        action="store_true",
        help="Skip tools/generate_safety_traceability.py --check",
    )
    parser.add_argument(
        "--skip-psi-gate-attribution",
        action="store_true",
        help="Skip tools/generate_psi_gate_attribution.py --check",
    )
    parser.add_argument(
        "--skip-surrogate-uq-cards",
        action="store_true",
        help="Skip tools/generate_surrogate_uq_cards.py --check",
    )
    parser.add_argument(
        "--skip-torax-imas-interchange",
        action="store_true",
        help="Skip validation/torax_imas_interchange.py --check",
    )
    parser.add_argument(
        "--skip-torax-same-physics-config",
        action="store_true",
        help="Skip validation/torax_same_physics_config_study.py --check",
    )
    parser.add_argument(
        "--skip-fair-validation-packs",
        action="store_true",
        help="Skip tools/export_zenodo_dataset.py --check --no-export",
    )
    parser.add_argument(
        "--skip-capability-manifest",
        action="store_true",
        help="Skip tools/capability_manifest.py --check",
    )
    parser.add_argument(
        "--skip-readiness-register",
        action="store_true",
        help="Skip tools/generate_readiness_register.py --check",
    )
    parser.add_argument(
        "--skip-readiness-scope-reports",
        action="store_true",
        help="Skip tools/generate_readiness_scope_reports.py --check",
    )
    parser.add_argument(
        "--skip-release-delta-guard",
        action="store_true",
        help="Skip tools/release_delta_guard.py non-regression check.",
    )
    parser.add_argument(
        "--skip-source-readiness",
        action="store_true",
        help="Skip tools/generate_source_p0p1_readiness.py --check",
    )
    parser.add_argument(
        "--skip-untested-module-guard",
        action="store_true",
        help="Skip tools/check_test_module_linkage.py",
    )
    parser.add_argument(
        "--skip-deprecated-default-lane-guard",
        action="store_true",
        help="Skip tools/deprecated_default_lane_guard.py",
    )
    parser.add_argument(
        "--skip-release-readiness",
        action="store_true",
        help="Skip tools/check_release_readiness.py",
    )
    parser.add_argument(
        "--skip-shot-manifest",
        action="store_true",
        help="Skip tools/generate_disruption_shot_manifest.py --check",
    )
    parser.add_argument(
        "--skip-reference-data-provenance",
        action="store_true",
        help="Skip tools/generate_reference_data_provenance_manifest.py --check",
    )
    parser.add_argument(
        "--skip-shot-splits",
        action="store_true",
        help="Skip tools/check_disruption_shot_splits.py",
    )
    parser.add_argument(
        "--skip-disruption-calibration",
        action="store_true",
        help="Skip tools/generate_disruption_risk_calibration.py --check",
    )
    parser.add_argument(
        "--skip-disruption-replay-pipeline",
        action="store_true",
        help="Skip validation/benchmark_disruption_replay_pipeline.py --strict",
    )
    parser.add_argument(
        "--skip-disruption-transfer-generalization",
        action="store_true",
        help="Skip validation/benchmark_disruption_transfer_generalization.py --strict",
    )
    parser.add_argument(
        "--skip-eped-domain-contract",
        action="store_true",
        help="Skip validation/benchmark_eped_domain_contract.py --strict",
    )
    parser.add_argument(
        "--skip-transport-uncertainty",
        action="store_true",
        help="Skip validation/benchmark_transport_uncertainty_envelope.py --strict",
    )
    parser.add_argument(
        "--skip-vertical-control-replay",
        action="store_true",
        help="Skip validation/vertical_control_replay_benchmark.py --all-profiles --strict",
    )
    parser.add_argument(
        "--skip-torax-strict-backend",
        action="store_true",
        help="Skip validation/benchmark_torax_real_parity.py --check",
    )
    parser.add_argument(
        "--skip-sparc-strict-backend",
        action="store_true",
        help="Skip validation/benchmark_sparc_geqdsk_rmse.py --strict-backend",
    )
    parser.add_argument(
        "--skip-freegs-strict-backend",
        action="store_true",
        help="Skip validation/benchmark_free_boundary_strict_parity.py --check --strict",
    )
    parser.add_argument(
        "--enable-strict-backend-checks",
        action="store_true",
        default=bool(
            os.environ.get("SCPN_ENABLE_STRICT_BACKEND_CHECKS", "").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        help=(
            "Enable canonical TORAX/SPARC strict evidence checks "
            "(also enabled by SCPN_ENABLE_STRICT_BACKEND_CHECKS=1)."
        ),
    )
    parser.add_argument(
        "--enable-freegs-strict-backend-check",
        action="store_true",
        default=bool(
            os.environ.get("SCPN_ENABLE_FREEGS_STRICT_BACKEND_CHECKS", "").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        help=(
            "Force FreeGS/FreeGSNKE strict parity check when strict evidence checks are enabled "
            "(also enabled by SCPN_ENABLE_FREEGS_STRICT_BACKEND_CHECKS=1)."
        ),
    )
    parser.add_argument(
        "--skip-multi-ion-conservation",
        action="store_true",
        help="Skip validation/benchmark_multi_ion_transport_conservation.py --strict",
    )
    parser.add_argument(
        "--skip-end-to-end-latency",
        action="store_true",
        help="Skip validation/scpn_end_to_end_latency.py --strict",
    )
    parser.add_argument(
        "--skip-threshold-smoke",
        action="store_true",
        help=(
            "Skip Task 5/6 threshold smoke tests "
            "(tests/test_task5_disruption_mitigation_integration.py and "
            "tests/test_task6_heating_neutronics_realism.py)."
        ),
    )
    parser.add_argument(
        "--skip-mypy",
        action="store_true",
        help="Skip tools/run_mypy_strict.py",
    )
    parser.add_argument(
        "--skip-research-suite",
        action="store_true",
        help="Skip the hermetic experimental pytest lane.",
    )
    parser.add_argument(
        "--check-timeout-seconds",
        type=float,
        default=DEFAULT_CHECK_TIMEOUT_SECONDS,
        help="Per-check subprocess timeout in seconds.",
    )
    args = parser.parse_args(argv)
    enable_freegs_strict_backend_check = bool(args.enable_freegs_strict_backend_check)
    if (
        args.enable_strict_backend_checks
        and not args.skip_freegs_strict_backend
        and not enable_freegs_strict_backend_check
    ):
        print(
            "[preflight] Skipping FreeGS/FreeGSNKE strict parity check: explicit opt-in "
            "disabled (use --enable-freegs-strict-backend-check or "
            "SCPN_ENABLE_FREEGS_STRICT_BACKEND_CHECKS=1)."
        )
    try:
        check_timeout_seconds = _normalize_check_timeout_seconds(args.check_timeout_seconds)
    except ValueError as exc:
        parser.error(str(exc))

    checks = _build_checks(
        gate=args.gate,
        skip_version_metadata=args.skip_version_metadata,
        skip_packaging_contract=args.skip_packaging_contract,
        skip_lfs_hygiene=args.skip_lfs_hygiene,
        skip_claims_audit=args.skip_claims_audit,
        skip_claim_range_guard=args.skip_claim_range_guard,
        skip_claims_map=args.skip_claims_map,
        skip_safety_traceability=args.skip_safety_traceability,
        skip_psi_gate_attribution=args.skip_psi_gate_attribution,
        skip_surrogate_uq_cards=args.skip_surrogate_uq_cards,
        skip_torax_imas_interchange=args.skip_torax_imas_interchange,
        skip_torax_same_physics_config=args.skip_torax_same_physics_config,
        skip_fair_validation_packs=args.skip_fair_validation_packs,
        skip_capability_manifest=args.skip_capability_manifest,
        skip_readiness_register=args.skip_readiness_register,
        skip_readiness_scope_reports=args.skip_readiness_scope_reports,
        skip_release_delta_guard=args.skip_release_delta_guard,
        skip_source_readiness=args.skip_source_readiness,
        skip_untested_module_guard=args.skip_untested_module_guard,
        skip_deprecated_default_lane_guard=args.skip_deprecated_default_lane_guard,
        skip_release_readiness=args.skip_release_readiness,
        skip_shot_manifest=args.skip_shot_manifest,
        skip_reference_data_provenance=args.skip_reference_data_provenance,
        skip_shot_splits=args.skip_shot_splits,
        skip_disruption_calibration=args.skip_disruption_calibration,
        skip_disruption_replay_pipeline=args.skip_disruption_replay_pipeline,
        skip_disruption_transfer_generalization=args.skip_disruption_transfer_generalization,
        skip_eped_domain_contract=args.skip_eped_domain_contract,
        skip_transport_uncertainty=args.skip_transport_uncertainty,
        skip_vertical_control_replay=args.skip_vertical_control_replay,
        skip_torax_strict_backend=args.skip_torax_strict_backend,
        skip_sparc_strict_backend=args.skip_sparc_strict_backend,
        skip_freegs_strict_backend=args.skip_freegs_strict_backend,
        skip_multi_ion_conservation=args.skip_multi_ion_conservation,
        skip_end_to_end_latency=args.skip_end_to_end_latency,
        skip_notebook_quality=args.skip_notebook_quality,
        skip_threshold_smoke=args.skip_threshold_smoke,
        skip_mypy=args.skip_mypy,
        skip_research_suite=args.skip_research_suite,
        enable_strict_backend_checks=args.enable_strict_backend_checks,
        enable_freegs_strict_backend_check=enable_freegs_strict_backend_check,
    )
    if not checks:
        print("[preflight] No checks selected.")
        return 0

    for name, cmd in checks:
        rc = _run_check(name, cmd, timeout_seconds=check_timeout_seconds)
        if rc != 0:
            print(
                f"[preflight] FAILED at '{name}' with exit code {rc}.",
                file=sys.stderr,
            )
            return rc

    print("[preflight] All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
