# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed contract for the IDA fixed-reference source-mechanism decomposition."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, cast

SCHEMA_VERSION = "scpn-fusion.ida-fixed-reference-source-mechanism.v1"
BENCHMARK_ID = "DIII-D-IDA-FB-JAX-B-SOURCE-MECHANISM"
EVALUATION_CASE_ID = "freegs_16_diiid_public_example"
GRID_SHAPE = [129, 129]
SAME_CASE_PATH = "validation/reports/ida_same_case_evidence.json"
SOURCE_ABLATION_PATH = "validation/reports/ida_fixed_reference_source_ablation.json"
OPERATOR_DECOMPOSITION_PATH = "validation/reports/ida_fixed_reference_operator_residual.json"
CONTROL_REPOSITORY_PATH = "../SCPN-CONTROL"
CLOSURE_MAX_ABS = 1.0e-8

CLAIM_FIELDS = (
    "control_admission",
    "facility_validation",
    "pcs_deployment",
    "safety_admission",
    "scientific_validation",
)
CURRENT_FIELDS = (
    "freegs_hard_romberg",
    "freegs_hard_rectangular_normalised",
    "fusion_smooth_unscaled",
    "fusion_smooth_rectangular_normalised",
)
MECHANISM_COMPONENTS = (
    "hard_rectangular_normalisation",
    "smooth_cutoff",
    "smooth_ip_normalisation",
)
SOURCE_PATHS = {
    "contract": "validation/ida_fixed_reference_source_mechanism_contract.py",
    "diagnostic": "validation/diagnose_ida_fixed_reference_source_mechanism.py",
    "plasma_support": "src/scpn_fusion/core/jax_plasma_support.py",
    "predictive_solver": "src/scpn_fusion/core/jax_free_boundary_predictive.py",
    "control_profile_source": ("../SCPN-CONTROL/src/scpn_control/core/gs_profile_source.py"),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
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
_CURRENT_METRIC_FIELDS = {
    "absolute_current_outside_reference_fraction",
    "candidate_support_point_count",
    "current_density_sha256",
    "rectangular_current_a",
    "reference_support_point_count",
    "relative_ip_error",
    "relative_l2_to_reference",
    "total_variation_distance",
}
_VECTOR_METRIC_FIELDS = {
    "field_sha256",
    "linf",
    "relative_l2_to_reference_scale",
    "rms",
}
_TOP_LEVEL_FIELDS = {
    "benchmark_id",
    "bindings",
    "claim_boundary",
    "closure",
    "control_parity",
    "current_fields",
    "current_vectors",
    "environment",
    "generated_at",
    "input_contract",
    "interior_source_vectors",
    "payload_sha256",
    "routing",
    "schema_version",
    "source_artifacts",
    "status",
    "wall_response_vectors",
}


def _payload_sha256(report: dict[str, Any]) -> str:
    payload = {name: value for name, value in report.items() if name != "payload_sha256"}
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _walk_finite(value: object, *, field: str = "report") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_finite(item, field=f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for name, item in value.items():
            _walk_finite(item, field=f"{field}.{name}")
        return
    raise ValueError(f"{field} contains unsupported value type {type(value).__name__}")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_git_oid(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_OID_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full lowercase Git object id")
    return value


def _require_number(
    value: object,
    *,
    field: str,
    minimum: float | None = 0.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    measured = float(value)
    if not math.isfinite(measured):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and measured < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return measured


def _dominant_component(metrics: dict[str, dict[str, Any]]) -> str:
    return max(
        MECHANISM_COMPONENTS,
        key=lambda name: float(metrics[name]["relative_l2_to_reference_scale"]),
    )


def _routing(
    *,
    current_vectors: dict[str, dict[str, Any]],
    interior_source_vectors: dict[str, dict[str, Any]],
    wall_response_vectors: dict[str, dict[str, Any]],
) -> dict[str, str]:
    current = _dominant_component(current_vectors)
    interior = _dominant_component(interior_source_vectors)
    wall = _dominant_component(wall_response_vectors)
    if "smooth_cutoff" in {current, interior, wall}:
        next_target = "self_consistent_equilibrium_geometry_and_boundary_response"
    elif "smooth_ip_normalisation" in {current, interior, wall}:
        next_target = "smooth_ip_normalisation_quadrature"
    else:
        next_target = "hard_mask_current_normalisation_quadrature"
    return {
        "current_dominant_component": current,
        "interior_source_dominant_component": interior,
        "next_ratcheting_target": next_target,
        "wall_response_dominant_component": wall,
    }


def build_report(
    *,
    generated_at: str,
    environment: dict[str, Any],
    source_artifacts: dict[str, dict[str, Any]],
    bindings: dict[str, dict[str, Any]],
    current_fields: dict[str, dict[str, Any]],
    current_vectors: dict[str, dict[str, Any]],
    interior_source_vectors: dict[str, dict[str, Any]],
    wall_response_vectors: dict[str, dict[str, Any]],
    control_parity: dict[str, Any],
    closure: dict[str, float],
    cutoff_width: float,
) -> dict[str, Any]:
    """Build and validate one immutable diagnostic payload."""
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "bindings": bindings,
        "claim_boundary": {field: False for field in CLAIM_FIELDS},
        "closure": closure,
        "control_parity": control_parity,
        "current_fields": current_fields,
        "current_vectors": current_vectors,
        "environment": environment,
        "generated_at": generated_at,
        "input_contract": {
            "control_semantics": (
                "external profiles; hard 0<=psi_N<1 support; rectangular Ip normalisation"
            ),
            "cutoff_width": cutoff_width,
            "evaluation_case_id": EVALUATION_CASE_ID,
            "grid_shape": GRID_SHAPE,
            "mechanism_sequence": list(MECHANISM_COMPONENTS),
            "reference_geometry": "fixed converged FreeGS total psi",
        },
        "interior_source_vectors": interior_source_vectors,
        "routing": _routing(
            current_vectors=current_vectors,
            interior_source_vectors=interior_source_vectors,
            wall_response_vectors=wall_response_vectors,
        ),
        "schema_version": SCHEMA_VERSION,
        "source_artifacts": source_artifacts,
        "status": "diagnostic_complete_claims_blocked",
        "wall_response_vectors": wall_response_vectors,
    }
    report["payload_sha256"] = _payload_sha256(report)
    validate_report(report, cutoff_width=cutoff_width)
    return report


def _validate_current_fields(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != set(CURRENT_FIELDS):
        raise ValueError("current_fields fields are invalid")
    rows = cast(dict[str, dict[str, Any]], value)
    for name, row in rows.items():
        if not isinstance(row, dict) or set(row) != _CURRENT_METRIC_FIELDS:
            raise ValueError(f"current_fields.{name} fields are invalid")
        _require_sha256(
            row["current_density_sha256"],
            field=f"current_fields.{name}.current_density_sha256",
        )
        _require_number(
            row["rectangular_current_a"],
            field=f"current_fields.{name}.rectangular_current_a",
            minimum=None,
        )
        for metric in (
            "absolute_current_outside_reference_fraction",
            "relative_ip_error",
            "relative_l2_to_reference",
            "total_variation_distance",
        ):
            measured = _require_number(
                row[metric],
                field=f"current_fields.{name}.{metric}",
            )
            if (
                metric
                in {
                    "absolute_current_outside_reference_fraction",
                    "total_variation_distance",
                }
                and measured > 1.0
            ):
                raise ValueError(f"current_fields.{name}.{metric} must be <= 1")
        for count in ("candidate_support_point_count", "reference_support_point_count"):
            measured_count = row[count]
            if (
                isinstance(measured_count, bool)
                or not isinstance(measured_count, int)
                or measured_count <= 0
            ):
                raise ValueError(f"current_fields.{name}.{count} must be a positive integer")
    return rows


def _validate_vector_map(value: object, *, field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != set(MECHANISM_COMPONENTS):
        raise ValueError(f"{field} fields are invalid")
    rows = cast(dict[str, dict[str, Any]], value)
    for name, row in rows.items():
        if not isinstance(row, dict) or set(row) != _VECTOR_METRIC_FIELDS:
            raise ValueError(f"{field}.{name} fields are invalid")
        _require_sha256(row["field_sha256"], field=f"{field}.{name}.field_sha256")
        for metric in ("linf", "relative_l2_to_reference_scale", "rms"):
            _require_number(row[metric], field=f"{field}.{name}.{metric}")
    return rows


def _validate_source_artifacts(value: object) -> None:
    expected = {*SOURCE_PATHS, "control_repository", "fusion_repository", "freegs_public_example"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("source_artifacts fields are invalid")
    for name in {*SOURCE_PATHS, "freegs_public_example"}:
        row = value[name]
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError(f"source_artifacts.{name} fields are invalid")
        if name in SOURCE_PATHS and row["path"] != SOURCE_PATHS[name]:
            raise ValueError(f"source_artifacts.{name}.path is invalid")
        _require_sha256(row["sha256"], field=f"source_artifacts.{name}.sha256")
    for name, expected_path in (
        ("control_repository", CONTROL_REPOSITORY_PATH),
        ("fusion_repository", "."),
    ):
        row = value[name]
        if not isinstance(row, dict) or set(row) != {"git_commit", "path", "worktree_clean"}:
            raise ValueError(f"source_artifacts.{name} fields are invalid")
        if row["path"] != expected_path or not isinstance(row["worktree_clean"], bool):
            raise ValueError(f"source_artifacts.{name} fields are invalid")
        _require_git_oid(row["git_commit"], field=f"source_artifacts.{name}.git_commit")


def _validate_binding_row(
    row: object,
    *,
    field: str,
    expected_path: str,
    extra_fields: set[str],
) -> dict[str, Any]:
    expected = {"path", "payload_sha256", "source_commit", *extra_fields}
    if not isinstance(row, dict) or set(row) != expected or row.get("path") != expected_path:
        raise ValueError(f"{field} fields are invalid")
    _require_sha256(row["payload_sha256"], field=f"{field}.payload_sha256")
    _require_git_oid(row["source_commit"], field=f"{field}.source_commit")
    return cast(dict[str, Any], row)


def _validate_bindings(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "operator_decomposition",
        "same_case",
        "source_ablation",
    }:
        raise ValueError("bindings fields are invalid")
    same = _validate_binding_row(
        value["same_case"],
        field="bindings.same_case",
        expected_path=SAME_CASE_PATH,
        extra_fields={"case_id", "grid_shape"},
    )
    if same["case_id"] != EVALUATION_CASE_ID or same["grid_shape"] != GRID_SHAPE:
        raise ValueError("bindings.same_case does not bind the frozen evaluation")
    source = _validate_binding_row(
        value["source_ablation"],
        field="bindings.source_ablation",
        expected_path=SOURCE_ABLATION_PATH,
        extra_fields={"source_same_case_payload_sha256"},
    )
    operator = _validate_binding_row(
        value["operator_decomposition"],
        field="bindings.operator_decomposition",
        expected_path=OPERATOR_DECOMPOSITION_PATH,
        extra_fields={
            "source_ablation_payload_sha256",
            "source_same_case_payload_sha256",
        },
    )
    for field, digest in (
        (
            "bindings.source_ablation.source_same_case_payload_sha256",
            source["source_same_case_payload_sha256"],
        ),
        (
            "bindings.operator_decomposition.source_same_case_payload_sha256",
            operator["source_same_case_payload_sha256"],
        ),
    ):
        _require_sha256(digest, field=field)
        if digest != same["payload_sha256"]:
            raise ValueError(f"{field} disagrees with bindings.same_case")
    _require_sha256(
        operator["source_ablation_payload_sha256"],
        field="bindings.operator_decomposition.source_ablation_payload_sha256",
    )
    if operator["source_ablation_payload_sha256"] != source["payload_sha256"]:
        raise ValueError("operator and source-ablation payload bindings disagree")


def validate_report(report: dict[str, Any], *, cutoff_width: float) -> None:
    """Reject schema drift, payload tamper, overclaim, and algebraic-routing drift."""
    if set(report) != _TOP_LEVEL_FIELDS:
        raise ValueError("source-mechanism report top-level fields do not match v1")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported source-mechanism contract")
    if report.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("source-mechanism benchmark_id is invalid")
    _walk_finite(report)
    if report.get("payload_sha256") != _payload_sha256(report):
        raise ValueError("payload_sha256 does not match report content")
    _require_sha256(report.get("payload_sha256"), field="payload_sha256")
    if report.get("claim_boundary") != {field: False for field in CLAIM_FIELDS}:
        raise ValueError("claim_boundary must keep every promotion claim false")
    if report.get("status") != "diagnostic_complete_claims_blocked":
        raise ValueError("source-mechanism diagnostic cannot promote an admitted status")
    if report.get("input_contract") != {
        "control_semantics": (
            "external profiles; hard 0<=psi_N<1 support; rectangular Ip normalisation"
        ),
        "cutoff_width": cutoff_width,
        "evaluation_case_id": EVALUATION_CASE_ID,
        "grid_shape": GRID_SHAPE,
        "mechanism_sequence": list(MECHANISM_COMPONENTS),
        "reference_geometry": "fixed converged FreeGS total psi",
    }:
        raise ValueError("input_contract does not match the frozen v1 diagnostic")
    environment = report.get("environment")
    if not isinstance(environment, dict) or set(environment) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment fields do not match the runtime contract")
    if environment["x64_enabled"] is not True:
        raise ValueError("environment must bind JAX FP64 execution")
    _validate_source_artifacts(report.get("source_artifacts"))
    _validate_bindings(report.get("bindings"))
    _validate_current_fields(report.get("current_fields"))
    current = _validate_vector_map(report.get("current_vectors"), field="current_vectors")
    interior = _validate_vector_map(
        report.get("interior_source_vectors"),
        field="interior_source_vectors",
    )
    wall = _validate_vector_map(
        report.get("wall_response_vectors"),
        field="wall_response_vectors",
    )
    parity = report.get("control_parity")
    if not isinstance(parity, dict) or set(parity) != {
        "absolute_current_outside_freegs_support_fraction",
        "actual_current_density_sha256",
        "formula_current_density_sha256",
        "hard_mask_sha256",
        "max_abs_a_per_m2",
        "rectangular_current_a",
        "relative_l2",
        "relative_l2_to_freegs_reference",
        "total_variation_to_freegs_reference",
    }:
        raise ValueError("control_parity fields are invalid")
    for name in (
        "actual_current_density_sha256",
        "formula_current_density_sha256",
        "hard_mask_sha256",
    ):
        _require_sha256(parity[name], field=f"control_parity.{name}")
    if (
        _require_number(parity["max_abs_a_per_m2"], field="control_parity.max_abs_a_per_m2")
        > 1.0e-8
    ):
        raise ValueError("CONTROL implementation and explicit formula exceed max-abs parity")
    if _require_number(parity["relative_l2"], field="control_parity.relative_l2") > 1.0e-14:
        raise ValueError("CONTROL implementation and explicit formula exceed relative-L2 parity")
    _require_number(
        parity["rectangular_current_a"],
        field="control_parity.rectangular_current_a",
        minimum=None,
    )
    for name in (
        "absolute_current_outside_freegs_support_fraction",
        "relative_l2_to_freegs_reference",
        "total_variation_to_freegs_reference",
    ):
        measured = _require_number(parity[name], field=f"control_parity.{name}")
        if name != "relative_l2_to_freegs_reference" and measured > 1.0:
            raise ValueError(f"control_parity.{name} must be <= 1")
    closure = report.get("closure")
    if not isinstance(closure, dict) or set(closure) != {
        "current_max_abs_a_per_m2",
        "interior_source_max_abs",
        "wall_response_max_abs_wb",
    }:
        raise ValueError("closure fields are invalid")
    for name, value in closure.items():
        if _require_number(value, field=f"closure.{name}") > CLOSURE_MAX_ABS:
            raise ValueError(f"closure.{name} exceeds the algebraic closure threshold")
    if report.get("routing") != _routing(
        current_vectors=current,
        interior_source_vectors=interior,
        wall_response_vectors=wall,
    ):
        raise ValueError("routing is inconsistent with measured mechanism vectors")


def render_markdown(report: dict[str, Any], *, cutoff_width: float) -> str:
    """Render a concise fixed-reference mechanism summary."""
    validate_report(report, cutoff_width=cutoff_width)
    lines = [
        "# IDA fixed-reference source-mechanism decomposition",
        "",
        f"- Status: `{report['status']}`",
        f"- Payload SHA-256: `{report['payload_sha256']}`",
        "- Facility/control/PCS/safety/scientific admission: `false`",
        "",
        "## Fixed-reference current fields",
        "",
        "| Construction | rectangular current (A) | relative L2 | TV | outside support |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CURRENT_FIELDS:
        row = report["current_fields"][name]
        lines.append(
            f"| {name} | {row['rectangular_current_a']:.12g} | "
            f"{row['relative_l2_to_reference']:.9g} | "
            f"{row['total_variation_distance']:.9g} | "
            f"{row['absolute_current_outside_reference_fraction']:.9g} |"
        )
    lines.extend(
        [
            "",
            "## Sequential mechanism vectors",
            "",
            "| Component | current relative L2 | interior-source relative L2 | wall relative L2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in MECHANISM_COMPONENTS:
        lines.append(
            f"| {name} | "
            f"{report['current_vectors'][name]['relative_l2_to_reference_scale']:.9g} | "
            f"{report['interior_source_vectors'][name]['relative_l2_to_reference_scale']:.9g} | "
            f"{report['wall_response_vectors'][name]['relative_l2_to_reference_scale']:.9g} |"
        )
    route = report["routing"]
    lines.extend(
        [
            "",
            f"- Dominant current component: `{route['current_dominant_component']}`",
            f"- Dominant interior-source component: `{route['interior_source_dominant_component']}`",
            f"- Dominant wall-response component: `{route['wall_response_dominant_component']}`",
            f"- Next ratcheting target: `{route['next_ratcheting_target']}`",
            "",
            "This is a fixed-reference engineering decomposition, not a validation or admission result.",
            "",
        ]
    )
    return "\n".join(lines)
