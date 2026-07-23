# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed contract for the fixed-reference GS operator decomposition."""

from __future__ import annotations

import importlib
import math
from typing import Any, Callable, cast

_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_payload_sha256: Callable[[dict[str, Any]], str] = _same_case._payload_sha256
_walk_finite: Callable[[object], None] = _same_case._walk_finite

SCHEMA_VERSION = "scpn-fusion.ida-fixed-reference-operator-residual.v1"
BENCHMARK_ID = "DIII-D-IDA-FB-JAX-B-OPERATOR-RESIDUAL"
EVALUATION_CASE_ID = "freegs_16_diiid_public_example"
GRID_SHAPE = [129, 129]
CLAIM_FIELDS = (
    "control_admission",
    "facility_validation",
    "pcs_deployment",
    "scientific_validation",
    "safety_admission",
)
CLOSURE_MAX_ABS = 1.0e-10
INTERIOR_COMPONENTS = (
    "exact_source_convention",
    "freegs_fourth_order_baseline",
    "second_order_operator",
    "vacuum_discretisation",
)
WALL_COMPONENTS = (
    "coil_vacuum_convention",
    "exact_source_convention",
    "plasma_response_quadrature",
)
OPERATOR_RESIDUALS = (
    "freegs_fourth_order_reference_current_plasma_flux",
    "native_second_order_compact_source_total_flux",
    "native_second_order_exact_source_total_flux",
    "native_second_order_reference_current_plasma_flux",
    "native_second_order_reference_current_total_flux",
)
WALL_RESIDUALS = (
    "compact_source_total_flux",
    "exact_source_total_flux",
    "reference_current_total_flux",
)
SOURCE_PATHS = {
    "contract": "validation/ida_fixed_reference_operator_contract.py",
    "diagnostic": "validation/diagnose_ida_fixed_reference_operator.py",
    "freegs_boundary": (
        "data/external/full_fidelity_public_sources/repos/freegs/freegs/boundary.py"
    ),
    "freegs_operator": (
        "data/external/full_fidelity_public_sources/repos/freegs/freegs/gradshafranov.py"
    ),
    "source_ablation": "validation/diagnose_ida_fixed_reference_source.py",
    "solver": "src/scpn_fusion/core/jax_free_boundary_predictive.py",
}
SOURCE_ABLATION_PATH = "validation/reports/ida_fixed_reference_source_ablation.json"
SAME_CASE_PATH = "validation/reports/ida_same_case_evidence.json"
FREEGS_EXAMPLE_PATH = "data/external/full_fidelity_public_sources/repos/freegs/16-DIIID.py"
_SHA256_LENGTH = 64
_GIT_OID_LENGTH = 40
_METRIC_FIELDS = {
    "field_sha256",
    "linf",
    "relative_l2_to_reference_scale",
    "rms",
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
_TOP_LEVEL_FIELDS = {
    "benchmark_id",
    "blockers",
    "claim_boundary",
    "closure",
    "coil_region_diagnostic",
    "environment",
    "generated_at",
    "input_contract",
    "interior_components",
    "operator_residuals",
    "payload_sha256",
    "routing",
    "schema_version",
    "source_ablation",
    "source_artifacts",
    "source_same_case",
    "status",
    "wall_components",
    "wall_residuals",
}


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


def _require_number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _validate_metric(row: object, *, field: str) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != _METRIC_FIELDS:
        raise ValueError(f"{field} fields are invalid")
    _require_sha256(row["field_sha256"], field=f"{field}.field_sha256")
    for name in ("linf", "relative_l2_to_reference_scale", "rms"):
        _require_number(row[name], field=f"{field}.{name}")
    return cast(dict[str, Any], row)


def _routing(
    *,
    interior_components: dict[str, dict[str, Any]],
    wall_components: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    interior_candidates = {
        name: float(interior_components[name]["relative_l2_to_reference_scale"])
        for name in (
            "exact_source_convention",
            "second_order_operator",
            "vacuum_discretisation",
        )
    }
    wall_candidates = {
        name: float(wall_components[name]["relative_l2_to_reference_scale"])
        for name in WALL_COMPONENTS
    }
    interior_dominant = max(interior_candidates, key=interior_candidates.__getitem__)
    wall_dominant = max(wall_candidates, key=wall_candidates.__getitem__)
    next_target_by_component = {
        "exact_source_convention": "current_support_and_source_convention",
        "second_order_operator": "discrete_operator_order_and_stencil",
        "vacuum_discretisation": "vacuum_field_discrete_harmonicity",
    }
    wall_target_by_component = {
        "coil_vacuum_convention": "coil_green_function_convention",
        "exact_source_convention": "wall_source_support_and_quadrature",
        "plasma_response_quadrature": "plasma_wall_response_quadrature",
    }
    return {
        "interior_component_relative_l2": interior_candidates,
        "interior_dominant_component": interior_dominant,
        "next_interior_target": next_target_by_component[interior_dominant],
        "next_wall_target": wall_target_by_component[wall_dominant],
        "wall_component_relative_l2": wall_candidates,
        "wall_dominant_component": wall_dominant,
    }


def build_report(
    *,
    generated_at: str,
    environment: dict[str, Any],
    source_artifacts: dict[str, dict[str, str]],
    source_same_case: dict[str, Any],
    source_ablation: dict[str, Any],
    operator_residuals: dict[str, dict[str, Any]],
    interior_components: dict[str, dict[str, Any]],
    wall_residuals: dict[str, dict[str, Any]],
    wall_components: dict[str, dict[str, Any]],
    closure: dict[str, float],
    coil_region_diagnostic: dict[str, Any],
    source_commit: str,
    source_worktree_clean: bool,
) -> dict[str, Any]:
    """Build a self-digested diagnostic report with all admission claims false."""
    if not generated_at.strip():
        raise ValueError("generated_at must not be empty")
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
        "claim_boundary": {field: False for field in CLAIM_FIELDS},
        "closure": closure,
        "coil_region_diagnostic": coil_region_diagnostic,
        "environment": environment,
        "generated_at": generated_at,
        "input_contract": {
            "evaluation_case_id": EVALUATION_CASE_ID,
            "freegs_operator_order": 4,
            "grid_shape": GRID_SHAPE,
            "interior_measurement_region": "non-zero FreeGS reference Jtor support",
            "native_operator": ("second-order centered Delta-star with identity wall rows"),
            "reference_geometry": "fixed converged FreeGS total and plasma psi",
        },
        "interior_components": interior_components,
        "operator_residuals": operator_residuals,
        "payload_sha256": "",
        "routing": _routing(
            interior_components=interior_components,
            wall_components=wall_components,
        ),
        "schema_version": SCHEMA_VERSION,
        "source_ablation": source_ablation,
        "source_artifacts": {
            **source_artifacts,
            "repository": {"git_commit": source_commit, "path": "."},
        },
        "source_same_case": source_same_case,
        "status": "diagnostic_complete_claims_blocked",
        "wall_components": wall_components,
        "wall_residuals": wall_residuals,
    }
    report["payload_sha256"] = _payload_sha256(report)
    validate_report(report)
    return report


def validate_report(report: dict[str, Any]) -> None:
    """Reject schema drift, tamper, overclaim, and derived-routing drift."""
    if set(report) != _TOP_LEVEL_FIELDS:
        raise ValueError("operator report top-level fields do not match the v1 schema")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported fixed-reference operator contract")
    if report.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("operator report benchmark_id is invalid")
    _walk_finite(report)
    if report.get("payload_sha256") != _payload_sha256(report):
        raise ValueError("payload_sha256 does not match report content")
    _require_sha256(report.get("payload_sha256"), field="payload_sha256")
    if report.get("claim_boundary") != {field: False for field in CLAIM_FIELDS}:
        raise ValueError("claim_boundary must keep every promotion claim false")
    if report.get("status") != "diagnostic_complete_claims_blocked":
        raise ValueError("operator diagnostic cannot promote an admitted status")
    if report.get("input_contract") != {
        "evaluation_case_id": EVALUATION_CASE_ID,
        "freegs_operator_order": 4,
        "grid_shape": GRID_SHAPE,
        "interior_measurement_region": "non-zero FreeGS reference Jtor support",
        "native_operator": "second-order centered Delta-star with identity wall rows",
        "reference_geometry": "fixed converged FreeGS total and plasma psi",
    }:
        raise ValueError("input_contract does not match the frozen v1 diagnostic")
    environment = report.get("environment")
    if not isinstance(environment, dict) or set(environment) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment fields do not match the runtime contract")
    if environment["x64_enabled"] is not True:
        raise ValueError("environment must bind a JAX FP64 execution")
    _validate_sources(report)
    operator_residuals = _validate_metric_map(
        report.get("operator_residuals"),
        expected=OPERATOR_RESIDUALS,
        field="operator_residuals",
    )
    interior = _validate_metric_map(
        report.get("interior_components"),
        expected=INTERIOR_COMPONENTS,
        field="interior_components",
    )
    _validate_metric_map(
        report.get("wall_residuals"),
        expected=WALL_RESIDUALS,
        field="wall_residuals",
    )
    wall = _validate_metric_map(
        report.get("wall_components"),
        expected=WALL_COMPONENTS,
        field="wall_components",
    )
    del operator_residuals
    coil_region = report.get("coil_region_diagnostic")
    if not isinstance(coil_region, dict) or set(coil_region) != {
        "all_interior_vacuum_field",
        "coil_filament_count",
        "coil_filaments_inside_domain",
        "outside_reference_support_l2_fraction",
        "reference_plasma_support_fraction",
        "reference_plasma_support_point_count",
        "reference_plasma_support_vacuum_field",
    }:
        raise ValueError("coil_region_diagnostic fields are invalid")
    _validate_metric(
        coil_region["all_interior_vacuum_field"],
        field="coil_region_diagnostic.all_interior_vacuum_field",
    )
    _validate_metric(
        coil_region["reference_plasma_support_vacuum_field"],
        field="coil_region_diagnostic.reference_plasma_support_vacuum_field",
    )
    for name in (
        "outside_reference_support_l2_fraction",
        "reference_plasma_support_fraction",
    ):
        fraction = _require_number(
            coil_region[name],
            field=f"coil_region_diagnostic.{name}",
        )
        if fraction > 1.0:
            raise ValueError(f"coil_region_diagnostic.{name} must be <= 1")
    filament_count = coil_region["coil_filament_count"]
    inside_count = coil_region["coil_filaments_inside_domain"]
    if (
        isinstance(filament_count, bool)
        or not isinstance(filament_count, int)
        or filament_count <= 0
        or isinstance(inside_count, bool)
        or not isinstance(inside_count, int)
        or inside_count < 0
        or inside_count > filament_count
    ):
        raise ValueError("coil_region_diagnostic filament counts are invalid")
    point_count = coil_region["reference_plasma_support_point_count"]
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count <= 0:
        raise ValueError("coil_region_diagnostic.reference_plasma_support_point_count is invalid")
    closure = report.get("closure")
    if not isinstance(closure, dict) or set(closure) != {
        "interior_compact_max_abs",
        "interior_exact_max_abs",
        "wall_compact_max_abs",
        "wall_exact_max_abs",
    }:
        raise ValueError("closure fields are invalid")
    for name, value in closure.items():
        measured = _require_number(value, field=f"closure.{name}")
        if measured > CLOSURE_MAX_ABS:
            raise ValueError(f"closure.{name} exceeds the algebraic closure threshold")
    if report.get("routing") != _routing(
        interior_components=interior,
        wall_components=wall,
    ):
        raise ValueError("routing is inconsistent with measured components")
    _validate_bindings(report)


def _validate_metric_map(
    value: object,
    *,
    expected: tuple[str, ...],
    field: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{field} fields are invalid")
    for name, row in value.items():
        _validate_metric(row, field=f"{field}.{name}")
    return cast(dict[str, dict[str, Any]], value)


def _validate_sources(report: dict[str, Any]) -> None:
    artifacts = report.get("source_artifacts")
    expected = {*SOURCE_PATHS, "freegs_public_example", "repository"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ValueError("source_artifacts fields are invalid")
    for name in {*SOURCE_PATHS, "freegs_public_example"}:
        row = artifacts[name]
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError(f"source_artifacts.{name} fields are invalid")
        expected_path = SOURCE_PATHS.get(name, FREEGS_EXAMPLE_PATH)
        if row["path"] != expected_path:
            raise ValueError(f"source_artifacts.{name}.path is invalid")
        _require_sha256(row["sha256"], field=f"source_artifacts.{name}.sha256")
    repository = artifacts["repository"]
    if not isinstance(repository, dict) or repository.get("path") != ".":
        raise ValueError("source_artifacts.repository fields are invalid")
    _require_git_oid(repository.get("git_commit"), field="source_artifacts.repository.git_commit")


def _validate_bindings(report: dict[str, Any]) -> None:
    same_case = report.get("source_same_case")
    if not isinstance(same_case, dict) or set(same_case) != {
        "case_id",
        "grid_shape",
        "path",
        "payload_sha256",
        "source_commit",
    }:
        raise ValueError("source_same_case fields are invalid")
    if (
        same_case["case_id"] != EVALUATION_CASE_ID
        or same_case["grid_shape"] != GRID_SHAPE
        or same_case["path"] != SAME_CASE_PATH
    ):
        raise ValueError("source_same_case does not bind the frozen evaluation")
    _require_sha256(same_case["payload_sha256"], field="source_same_case.payload_sha256")
    _require_git_oid(same_case["source_commit"], field="source_same_case.source_commit")

    ablation = report.get("source_ablation")
    if not isinstance(ablation, dict) or set(ablation) != {
        "path",
        "payload_sha256",
        "source_commit",
        "source_same_case_payload_sha256",
    }:
        raise ValueError("source_ablation fields are invalid")
    if ablation["path"] != SOURCE_ABLATION_PATH:
        raise ValueError("source_ablation.path is invalid")
    _require_sha256(ablation["payload_sha256"], field="source_ablation.payload_sha256")
    _require_git_oid(ablation["source_commit"], field="source_ablation.source_commit")
    _require_sha256(
        ablation["source_same_case_payload_sha256"],
        field="source_ablation.source_same_case_payload_sha256",
    )
    if ablation["source_same_case_payload_sha256"] != same_case["payload_sha256"]:
        raise ValueError("source ablation and same-case payload bindings disagree")


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise decomposition summary."""
    validate_report(report)
    route = report["routing"]
    lines = [
        "# IDA fixed-reference operator residual",
        "",
        f"- Status: `{report['status']}`",
        f"- Payload SHA-256: `{report['payload_sha256']}`",
        "- Facility/control/PCS/safety/scientific admission: `false`",
        "",
        "## Interior decomposition",
        "",
        "| Component | relative L2 to reference scale | RMS | Linf |",
        "|---|---:|---:|---:|",
    ]
    for name in INTERIOR_COMPONENTS:
        row = report["interior_components"][name]
        lines.append(
            f"| {name} | {row['relative_l2_to_reference_scale']:.9g} | "
            f"{row['rms']:.9g} | {row['linf']:.9g} |"
        )
    lines.extend(
        [
            "",
            "## Wall decomposition",
            "",
            "| Component | relative L2 to reference scale | RMS | Linf |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in WALL_COMPONENTS:
        row = report["wall_components"][name]
        lines.append(
            f"| {name} | {row['relative_l2_to_reference_scale']:.9g} | "
            f"{row['rms']:.9g} | {row['linf']:.9g} |"
        )
    lines.extend(
        [
            "",
            "## Region guard",
            "",
            (
                f"- Reference plasma support: "
                f"`{report['coil_region_diagnostic']['reference_plasma_support_point_count']}` "
                "interior points"
            ),
            (
                f"- Coil filaments inside the rectangular domain: "
                f"`{report['coil_region_diagnostic']['coil_filaments_inside_domain']}` / "
                f"`{report['coil_region_diagnostic']['coil_filament_count']}`"
            ),
            (
                "- Vacuum-operator L2 outside reference plasma support: "
                f"`{report['coil_region_diagnostic']['outside_reference_support_l2_fraction']:.9g}`"
            ),
            "",
            f"- Dominant interior component: `{route['interior_dominant_component']}`",
            f"- Next interior target: `{route['next_interior_target']}`",
            f"- Dominant wall component: `{route['wall_dominant_component']}`",
            f"- Next wall target: `{route['next_wall_target']}`",
            "",
            "This is a fixed-reference engineering decomposition, not a validation or admission result.",
            "",
        ]
    )
    return "\n".join(lines)
