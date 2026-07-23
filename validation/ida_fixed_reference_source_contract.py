# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed evidence contract for the IDA fixed-reference source ablation."""

from __future__ import annotations

import importlib
import math
from typing import Any, Callable, cast

_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_payload_sha256: Callable[[dict[str, Any]], str] = _same_case._payload_sha256
_walk_finite: Callable[[object], None] = _same_case._walk_finite

SCHEMA_VERSION = "scpn-fusion.ida-fixed-reference-source-ablation.v2"
BENCHMARK_ID = "DIII-D-IDA-FB-JAX-B-SOURCE-ABLATION"
EVALUATION_CASE_ID = "freegs_16_diiid_public_example"
PROFILE_COEFFICIENT_COUNT = 12
PROFILE_DEGREE = 3
PROFILE_SAMPLE_COUNT = 129
CLAIM_FIELDS = (
    "control_admission",
    "facility_validation",
    "pcs_deployment",
    "scientific_validation",
    "safety_admission",
)
ROUTING_THRESHOLDS = {
    "axis_anchor_tv_reduction_max": 5.0e-3,
    "boundary_anchor_tv_reduction_min": 2.0e-2,
    "compact_exact_tv_delta_max": 1.0e-10,
    "fixed_reference_tv_max": 2.0e-2,
    "profile_fit_relative_l2_max": 1.0e-10,
    "reference_anchors_residual_tv_min": 1.0e-1,
    "self_consistent_to_fixed_tv_ratio_min": 5.0,
}
SOURCE_PATHS = {
    "ablation": "validation/diagnose_ida_fixed_reference_source.py",
    "contract": "validation/ida_fixed_reference_source_contract.py",
    "profile_basis": "src/scpn_fusion/core/jax_profile_basis.py",
    "same_case_benchmark": "validation/benchmark_ida_same_case.py",
    "solver": "src/scpn_fusion/core/jax_free_boundary_predictive.py",
}
FREEGS_PUBLIC_EXAMPLE_PATH = "data/external/full_fidelity_public_sources/repos/freegs/16-DIIID.py"
SAME_CASE_REPORT_RELATIVE_PATH = "validation/reports/ida_same_case_evidence.json"
SOURCE_EQUATION = "Jphi=R*pprime+FFprime/(mu0*R), smooth LCFS cutoff, Ip-normalised"
_TOP_LEVEL_FIELDS = {
    "benchmark_id",
    "blockers",
    "candidate_anchor_ablation",
    "claim_boundary",
    "environment",
    "fixed_reference_sources",
    "generated_at",
    "input_contract",
    "payload_sha256",
    "profile_fit",
    "routing",
    "routing_thresholds",
    "schema_version",
    "self_consistent_candidate",
    "source_artifacts",
    "source_same_case",
    "status",
}
_ENVIRONMENT_FIELDS = {
    "affinity_cpu_count",
    "backend",
    "devices",
    "freegs_version",
    "host_load_1m_5m_15m",
    "isolated_host",
    "jax_version",
    "jaxlib_version",
    "machine",
    "platform",
    "python_version",
    "x64_enabled",
}
_DISTRIBUTION_FIELDS = {
    "candidate_centroid_m",
    "centroid_delta_m",
    "cosine_similarity",
    "l1_distance",
    "reference_centroid_m",
    "total_variation_distance",
}
_SUPPORT_FIELDS = {
    "absolute_current_inside_reference_fraction",
    "absolute_current_outside_reference_fraction",
    "candidate_mask_sha256",
    "candidate_point_count",
    "false_negative_fraction_of_reference",
    "false_negative_point_count",
    "false_positive_fraction_of_candidate",
    "false_positive_point_count",
    "intersection_point_count",
    "iou",
    "reference_mask_sha256",
    "reference_point_count",
    "relative_floor",
    "union_point_count",
}
_SHA256_LENGTH = 64
_GIT_OID_LENGTH = 40


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_git_oid(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _GIT_OID_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a full lowercase Git object ID")
    return value


def _require_number(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _validate_current_source(row: object, *, field: str) -> None:
    if not isinstance(row, dict) or set(row) != {
        "current_density_sha256",
        "distribution",
        "support",
    }:
        raise ValueError(f"{field} fields are invalid")
    _require_sha256(row["current_density_sha256"], field=f"{field}.current_density_sha256")
    distribution = row["distribution"]
    support = row["support"]
    if not isinstance(distribution, dict) or set(distribution) != _DISTRIBUTION_FIELDS:
        raise ValueError(f"{field}.distribution fields are invalid")
    if not isinstance(support, dict) or set(support) != _SUPPORT_FIELDS:
        raise ValueError(f"{field}.support fields are invalid")
    for name in ("candidate_centroid_m", "centroid_delta_m", "reference_centroid_m"):
        centroid = distribution[name]
        if not isinstance(centroid, dict) or set(centroid) != {"r", "z"}:
            raise ValueError(f"{field}.distribution.{name} fields are invalid")
        for axis in ("r", "z"):
            _require_number(centroid[axis], field=f"{field}.distribution.{name}.{axis}")
    for name in ("cosine_similarity", "l1_distance", "total_variation_distance"):
        _require_number(
            distribution[name],
            field=f"{field}.distribution.{name}",
            minimum=0.0,
        )
    if distribution["l1_distance"] != 2.0 * distribution["total_variation_distance"]:
        raise ValueError(f"{field}.distribution TV/L1 relation is inconsistent")
    for name in ("candidate_mask_sha256", "reference_mask_sha256"):
        _require_sha256(support[name], field=f"{field}.support.{name}")


def _routing(
    *,
    profile_fit: dict[str, dict[str, Any]],
    compact: dict[str, Any],
    exact: dict[str, Any],
    self_consistent: dict[str, Any],
) -> dict[str, Any]:
    compact_tv = _require_number(
        compact["distribution"]["total_variation_distance"],
        field="compact TV",
        minimum=0.0,
    )
    exact_tv = _require_number(
        exact["distribution"]["total_variation_distance"],
        field="exact TV",
        minimum=0.0,
    )
    self_consistent_tv = _require_number(
        self_consistent["distribution"]["total_variation_distance"],
        field="self-consistent TV",
        minimum=0.0,
    )
    fit_max = max(
        _require_number(row["relative_l2_error"], field=f"{name} fit", minimum=0.0)
        for name, row in profile_fit.items()
    )
    compact_exact_delta = abs(compact_tv - exact_tv)
    ratio = self_consistent_tv / max(exact_tv, 1.0e-30)
    profile_representation_excluded = bool(
        fit_max <= ROUTING_THRESHOLDS["profile_fit_relative_l2_max"]
        and compact_exact_delta <= ROUTING_THRESHOLDS["compact_exact_tv_delta_max"]
    )
    fixed_reference_source_matches = bool(exact_tv <= ROUTING_THRESHOLDS["fixed_reference_tv_max"])
    geometry_separation = bool(ratio >= ROUTING_THRESHOLDS["self_consistent_to_fixed_tv_ratio_min"])
    if profile_representation_excluded and fixed_reference_source_matches and geometry_separation:
        next_target = "self_consistent_equilibrium_geometry_and_flux_normalisation"
    else:
        next_target = "source_convention_or_profile_representation_unresolved"
    return {
        "compact_exact_tv_delta": compact_exact_delta,
        "fixed_reference_source_matches": fixed_reference_source_matches,
        "geometry_separation": geometry_separation,
        "next_ratcheting_target": next_target,
        "profile_fit_relative_l2_max": fit_max,
        "profile_representation_excluded": profile_representation_excluded,
        "self_consistent_minus_exact_fixed_tv": self_consistent_tv - exact_tv,
        "self_consistent_to_exact_fixed_tv_ratio": ratio,
    }


def _anchor_routing(anchor_ablation: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def tv(name: str) -> float:
        return _require_number(
            anchor_ablation[name]["distribution"]["total_variation_distance"],
            field=f"candidate_anchor_ablation.{name}.TV",
            minimum=0.0,
        )

    production = tv("production_smooth_anchors")
    reference_both = tv("reference_axis_reference_boundary")
    candidate_axis_reference_boundary = tv("candidate_axis_reference_boundary")
    reference_axis_candidate_boundary = tv("reference_axis_candidate_boundary")
    boundary_reduction = production - candidate_axis_reference_boundary
    axis_reduction = production - reference_axis_candidate_boundary
    boundary_secondary = bool(
        boundary_reduction >= ROUTING_THRESHOLDS["boundary_anchor_tv_reduction_min"]
    )
    axis_excluded = bool(abs(axis_reduction) <= ROUTING_THRESHOLDS["axis_anchor_tv_reduction_max"])
    geometry_primary = bool(
        reference_both >= ROUTING_THRESHOLDS["reference_anchors_residual_tv_min"]
    )
    if boundary_secondary and axis_excluded and geometry_primary:
        target = "candidate_flux_geometry_primary_boundary_anchor_secondary_axis_anchor_excluded"
    else:
        target = "geometry_anchor_contributions_unresolved"
    return {
        "axis_anchor_excluded": axis_excluded,
        "axis_anchor_tv_reduction": axis_reduction,
        "boundary_anchor_secondary": boundary_secondary,
        "boundary_anchor_tv_reduction": boundary_reduction,
        "geometry_primary": geometry_primary,
        "next_ratcheting_target": target,
        "reference_anchors_residual_fraction": reference_both / max(production, 1.0e-30),
        "reference_anchors_residual_tv": reference_both,
    }


def build_report(
    *,
    generated_at: str,
    environment: dict[str, Any],
    source_artifacts: dict[str, dict[str, str]],
    source_same_case: dict[str, Any],
    profile_fit: dict[str, dict[str, Any]],
    fixed_reference_sources: dict[str, dict[str, Any]],
    candidate_anchor_ablation: dict[str, dict[str, Any]],
    self_consistent_candidate: dict[str, Any],
    source_commit: str,
    source_worktree_clean: bool,
    cutoff_width: float,
) -> dict[str, Any]:
    """Build and validate a self-digested fail-closed ablation report."""
    if not generated_at.strip():
        raise ValueError("generated_at must not be empty")
    route = _routing(
        profile_fit=profile_fit,
        compact=fixed_reference_sources["compact_bspline"],
        exact=fixed_reference_sources["exact_sampled"],
        self_consistent=self_consistent_candidate,
    )
    route["candidate_anchor_ablation"] = _anchor_routing(candidate_anchor_ablation)
    blockers = [
        "facility_validation_not_bound",
        "isolated_latency_evidence_missing",
        "pcs_and_safety_programmes_not_bound",
        "same_case_accuracy_threshold_failed",
        "statistically_held_out_case_missing",
    ]
    if not source_worktree_clean:
        blockers.append("source_worktree_not_clean")
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "blockers": blockers,
        "candidate_anchor_ablation": candidate_anchor_ablation,
        "claim_boundary": {field: False for field in CLAIM_FIELDS},
        "environment": environment,
        "fixed_reference_sources": fixed_reference_sources,
        "generated_at": generated_at,
        "input_contract": {
            "cutoff_width": cutoff_width,
            "evaluation_case_id": EVALUATION_CASE_ID,
            "fixed_geometry": "FreeGS reference psi, axis, and boundary",
            "profile_coefficient_count": PROFILE_COEFFICIENT_COUNT,
            "profile_degree": PROFILE_DEGREE,
            "profile_sample_count": PROFILE_SAMPLE_COUNT,
            "source_equation": SOURCE_EQUATION,
        },
        "payload_sha256": "",
        "profile_fit": profile_fit,
        "routing": route,
        "routing_thresholds": dict(ROUTING_THRESHOLDS),
        "schema_version": SCHEMA_VERSION,
        "self_consistent_candidate": self_consistent_candidate,
        "source_artifacts": {
            **source_artifacts,
            "repository": {"git_commit": source_commit, "path": "."},
        },
        "source_same_case": source_same_case,
        "status": "diagnostic_complete_claims_blocked",
    }
    report["payload_sha256"] = _payload_sha256(report)
    validate_report(report, cutoff_width=cutoff_width)
    return report


def validate_report(report: dict[str, Any], *, cutoff_width: float) -> None:
    """Reject schema drift, tamper, overclaim, and inconsistent routing."""
    if set(report) != _TOP_LEVEL_FIELDS:
        raise ValueError("ablation report top-level fields do not match the v2 schema")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("unsupported fixed-reference source ablation contract")
    _walk_finite(report)
    if report.get("payload_sha256") != _payload_sha256(report):
        raise ValueError("payload_sha256 does not match report content")
    _require_sha256(report.get("payload_sha256"), field="payload_sha256")
    if report.get("claim_boundary") != {field: False for field in CLAIM_FIELDS}:
        raise ValueError("claim_boundary must keep every promotion claim false")
    if report.get("routing_thresholds") != ROUTING_THRESHOLDS:
        raise ValueError("routing thresholds do not match the frozen v2 contract")
    if report.get("status") != "diagnostic_complete_claims_blocked":
        raise ValueError("source ablation cannot promote an admitted status")
    environment = report.get("environment")
    if not isinstance(environment, dict) or set(environment) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment fields do not match the frozen runtime contract")
    if environment["x64_enabled"] is not True:
        raise ValueError("environment must bind a JAX FP64 execution")
    if report.get("input_contract") != {
        "cutoff_width": cutoff_width,
        "evaluation_case_id": EVALUATION_CASE_ID,
        "fixed_geometry": "FreeGS reference psi, axis, and boundary",
        "profile_coefficient_count": PROFILE_COEFFICIENT_COUNT,
        "profile_degree": PROFILE_DEGREE,
        "profile_sample_count": PROFILE_SAMPLE_COUNT,
        "source_equation": SOURCE_EQUATION,
    }:
        raise ValueError("input_contract does not match the frozen v2 diagnostic")
    _validate_sources(report)
    profile_fit = _validate_profiles(report)
    fixed, self_consistent = _validate_currents(report)
    anchor_ablation = _validate_anchor_ablation(report)
    expected_route = _routing(
        profile_fit=profile_fit,
        compact=fixed["compact_bspline"],
        exact=fixed["exact_sampled"],
        self_consistent=self_consistent,
    )
    expected_route["candidate_anchor_ablation"] = _anchor_routing(anchor_ablation)
    if report.get("routing") != expected_route:
        raise ValueError("routing is inconsistent with the measured diagnostics")
    _validate_same_case_binding(report, self_consistent)


def _validate_sources(report: dict[str, Any]) -> None:
    source_artifacts = report.get("source_artifacts")
    expected_sources = {*SOURCE_PATHS, "freegs_public_example", "repository"}
    if not isinstance(source_artifacts, dict) or set(source_artifacts) != expected_sources:
        raise ValueError("source_artifacts do not match the v2 contract")
    for name in {*SOURCE_PATHS, "freegs_public_example"}:
        artifact = source_artifacts[name]
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise ValueError(f"source_artifacts.{name} fields are invalid")
        _require_sha256(artifact["sha256"], field=f"source_artifacts.{name}.sha256")
        expected_path = SOURCE_PATHS.get(name, FREEGS_PUBLIC_EXAMPLE_PATH)
        if artifact["path"] != expected_path:
            raise ValueError(f"source_artifacts.{name}.path is invalid")
    repository = source_artifacts["repository"]
    if not isinstance(repository, dict) or repository.get("path") != ".":
        raise ValueError("source_artifacts.repository fields are invalid")
    _require_git_oid(repository.get("git_commit"), field="source_artifacts.repository.git_commit")


def _validate_profiles(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profile_fit = report.get("profile_fit")
    if not isinstance(profile_fit, dict) or set(profile_fit) != {"ffprime", "pprime"}:
        raise ValueError("profile_fit fields are invalid")
    for name, row in profile_fit.items():
        if not isinstance(row, dict) or set(row) != {
            "exact_sha256",
            "reconstructed_sha256",
            "relative_l2_error",
            "relative_max_error",
            "sample_count",
        }:
            raise ValueError(f"profile_fit.{name} fields are invalid")
        _require_sha256(row["exact_sha256"], field=f"profile_fit.{name}.exact_sha256")
        _require_sha256(
            row["reconstructed_sha256"],
            field=f"profile_fit.{name}.reconstructed_sha256",
        )
        _require_number(
            row["relative_l2_error"],
            field=f"profile_fit.{name}.relative_l2_error",
            minimum=0.0,
        )
        _require_number(
            row["relative_max_error"],
            field=f"profile_fit.{name}.relative_max_error",
            minimum=0.0,
        )
        if row["sample_count"] != PROFILE_SAMPLE_COUNT:
            raise ValueError(f"profile_fit.{name}.sample_count is invalid")
    return cast(dict[str, dict[str, Any]], profile_fit)


def _validate_currents(
    report: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    fixed = report.get("fixed_reference_sources")
    self_consistent = report.get("self_consistent_candidate")
    if not isinstance(fixed, dict) or set(fixed) != {"compact_bspline", "exact_sampled"}:
        raise ValueError("fixed_reference_sources fields are invalid")
    if not isinstance(self_consistent, dict) or set(self_consistent) != {
        "candidate_psi_sha256",
        "distribution",
        "source_report_payload_sha256",
        "support",
    }:
        raise ValueError("self_consistent_candidate fields are invalid")
    for name, row in fixed.items():
        _validate_current_source(row, field=f"fixed_reference_sources.{name}")
    for name in ("candidate_psi_sha256", "source_report_payload_sha256"):
        _require_sha256(self_consistent[name], field=f"self_consistent_candidate.{name}")
    _validate_current_source(
        {
            "current_density_sha256": self_consistent["candidate_psi_sha256"],
            "distribution": self_consistent["distribution"],
            "support": self_consistent["support"],
        },
        field="self_consistent_candidate",
    )
    return cast(dict[str, dict[str, Any]], fixed), cast(dict[str, Any], self_consistent)


def _validate_anchor_ablation(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    anchor_ablation = report.get("candidate_anchor_ablation")
    expected = {
        "candidate_axis_reference_boundary",
        "production_smooth_anchors",
        "reference_axis_candidate_boundary",
        "reference_axis_reference_boundary",
    }
    if not isinstance(anchor_ablation, dict) or set(anchor_ablation) != expected:
        raise ValueError("candidate_anchor_ablation fields are invalid")
    for name, row in anchor_ablation.items():
        _validate_current_source(row, field=f"candidate_anchor_ablation.{name}")
    return cast(dict[str, dict[str, Any]], anchor_ablation)


def _validate_same_case_binding(
    report: dict[str, Any],
    self_consistent: dict[str, Any],
) -> None:
    source = report.get("source_same_case")
    if not isinstance(source, dict) or set(source) != {
        "case_id",
        "grid_shape",
        "path",
        "payload_sha256",
        "source_commit",
    }:
        raise ValueError("source_same_case fields are invalid")
    if source["case_id"] != EVALUATION_CASE_ID or source["grid_shape"] != [129, 129]:
        raise ValueError("source_same_case does not bind the frozen DIII-D evaluation")
    if source["path"] != SAME_CASE_REPORT_RELATIVE_PATH:
        raise ValueError("source_same_case.path is invalid")
    _require_sha256(source["payload_sha256"], field="source_same_case.payload_sha256")
    _require_git_oid(source["source_commit"], field="source_same_case.source_commit")
    if source["payload_sha256"] != self_consistent["source_report_payload_sha256"]:
        raise ValueError("self-consistent source payload does not match source_same_case")


def render_markdown(report: dict[str, Any], *, cutoff_width: float) -> str:
    """Render a concise diagnostic summary."""
    validate_report(report, cutoff_width=cutoff_width)
    compact = report["fixed_reference_sources"]["compact_bspline"]["distribution"]
    exact = report["fixed_reference_sources"]["exact_sampled"]["distribution"]
    self_consistent = report["self_consistent_candidate"]["distribution"]
    anchor = report["candidate_anchor_ablation"]
    route = report["routing"]
    anchor_route = route["candidate_anchor_ablation"]
    return "\n".join(
        [
            "# IDA fixed-reference current-source ablation",
            "",
            f"- Status: `{report['status']}`",
            f"- Payload SHA-256: `{report['payload_sha256']}`",
            "- Facility/control/PCS/safety/scientific admission: `false`",
            "",
            "## Measured isolation",
            "",
            "| Source path | current TV distance | centroid ΔR (m) | centroid ΔZ (m) |",
            "|---|---:|---:|---:|",
            (
                f"| Fixed reference + exact samples | {exact['total_variation_distance']:.9g} | "
                f"{exact['centroid_delta_m']['r']:.9g} | "
                f"{exact['centroid_delta_m']['z']:.9g} |"
            ),
            (
                f"| Fixed reference + compact B-spline | "
                f"{compact['total_variation_distance']:.9g} | "
                f"{compact['centroid_delta_m']['r']:.9g} | "
                f"{compact['centroid_delta_m']['z']:.9g} |"
            ),
            (
                f"| Self-consistent candidate | {self_consistent['total_variation_distance']:.9g} | "
                f"{self_consistent['centroid_delta_m']['r']:.9g} | "
                f"{self_consistent['centroid_delta_m']['z']:.9g} |"
            ),
            (
                f"| Candidate ψ + reference boundary | "
                f"{anchor['candidate_axis_reference_boundary']['distribution']['total_variation_distance']:.9g} | "
                f"{anchor['candidate_axis_reference_boundary']['distribution']['centroid_delta_m']['r']:.9g} | "
                f"{anchor['candidate_axis_reference_boundary']['distribution']['centroid_delta_m']['z']:.9g} |"
            ),
            (
                f"| Candidate ψ + reference axis and boundary | "
                f"{anchor['reference_axis_reference_boundary']['distribution']['total_variation_distance']:.9g} | "
                f"{anchor['reference_axis_reference_boundary']['distribution']['centroid_delta_m']['r']:.9g} | "
                f"{anchor['reference_axis_reference_boundary']['distribution']['centroid_delta_m']['z']:.9g} |"
            ),
            "",
            f"- Maximum profile-fit relative L2 error: "
            f"`{route['profile_fit_relative_l2_max']:.9g}`",
            f"- Self-consistent / exact-fixed TV ratio: "
            f"`{route['self_consistent_to_exact_fixed_tv_ratio']:.9g}`",
            f"- Anchor routing: `{anchor_route['next_ratcheting_target']}`",
            f"- Next ratcheting target: `{route['next_ratcheting_target']}`",
            "",
            "This routes engineering work only; it is not a physical-validation or admission result.",
            "",
        ]
    )
