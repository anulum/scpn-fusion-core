# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed contract for response-weighted IDA operator decomposition."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, cast

SCHEMA_VERSION = "scpn-fusion.ida-operator-response.v1"
BENCHMARK_ID = "DIII-D-IDA-FB-JAX-B-OPERATOR-RESPONSE"
EVALUATION_CASE_ID = "freegs_16_diiid_public_example"
GRID_SHAPE = [129, 129]
SAME_CASE_PATH = "validation/reports/ida_same_case_evidence.json"
OPERATOR_DECOMPOSITION_PATH = "validation/reports/ida_fixed_reference_operator_residual.json"
SOURCE_MECHANISM_PATH = "validation/reports/ida_fixed_reference_source_mechanism.json"
FIXED_POINT_PATH = "validation/reports/ida_fixed_point_stability.json"
COMPONENTS = (
    "freegs_fourth_order_baseline",
    "native_second_order_stencil",
    "coil_vacuum_discretisation",
    "exact_source_convention",
)
NATIVE_OPERATOR_COMPONENTS = COMPONENTS[:3]
CLAIM_FIELDS = (
    "control_admission",
    "facility_validation",
    "held_out_validation",
    "isolated_latency_admission",
    "pcs_deployment",
    "safety_admission",
    "scientific_validation",
)
SOURCE_PATHS = {
    "contract": "validation/ida_operator_response_contract.py",
    "diagnostic": "validation/diagnose_ida_operator_response.py",
    "fixed_point_contract": "validation/ida_fixed_point_stability_contract.py",
    "fixed_point_diagnostic": "validation/diagnose_ida_fixed_point_stability.py",
    "field_operations": "validation/ida_operator_response_fields.py",
    "multigrid": "src/scpn_fusion/core/jax_multigrid_precond.py",
    "operator_contract": "validation/ida_fixed_reference_operator_contract.py",
    "operator_diagnostic": "validation/diagnose_ida_fixed_reference_operator.py",
    "predictive_solver": "src/scpn_fusion/core/jax_free_boundary_predictive.py",
    "same_case_benchmark": "validation/benchmark_ida_same_case.py",
    "source_ablation": "validation/diagnose_ida_fixed_reference_source.py",
}
BASE_BLOCKERS = {
    "facility_validation_not_bound",
    "isolated_latency_evidence_missing",
    "pcs_and_safety_programmes_not_bound",
    "real_shot_predictive_dataset_missing",
    "same_case_accuracy_threshold_failed",
    "statistically_held_out_case_missing",
}
FORCING_CLOSURE_MAX_ABS = 1.0e-8
RESPONSE_CLOSURE_MAX_ABS_WB = 1.0e-9
FIXED_POINT_CONSISTENCY_MAX_ABS_WB = 1.0e-9

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
_FORCING_FIELDS = {
    "field_sha256",
    "l2",
    "linf",
    "relative_l2_to_exact_source_residual",
}
_RESPONSE_FIELDS = {
    "cosine_to_terminal_error",
    "field_sha256",
    "l2_wb",
    "linf_wb",
    "projection_on_terminal_error",
    "relative_l2_to_terminal_error",
}
_CLOSURE_FIELDS = {
    "exact_source_forcing_max_abs",
    "exact_source_response_max_abs_wb",
    "fixed_point_native_operator_max_abs_wb",
    "native_operator_forcing_max_abs",
    "native_operator_response_max_abs_wb",
}
_TOP_LEVEL_FIELDS = {
    "benchmark_id",
    "bindings",
    "blockers",
    "claim_boundary",
    "closure",
    "environment",
    "forcing_decomposition",
    "generated_at",
    "input_contract",
    "payload_sha256",
    "response_decomposition",
    "routing",
    "schema_version",
    "source_artifacts",
    "status",
}


def _payload_sha256(report: dict[str, Any]) -> str:
    """Return the canonical SHA-256 of a report without its signature field."""
    payload = {name: value for name, value in report.items() if name != "payload_sha256"}
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
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
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    measured = float(value)
    if not math.isfinite(measured):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and measured < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and measured > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return measured


def _input_contract() -> dict[str, Any]:
    return {
        "component_order": list(COMPONENTS),
        "evaluation_case_id": EVALUATION_CASE_ID,
        "grid_shape": GRID_SHAPE,
        "inner_solver": "mg_preconditioned_bicgstab",
        "inverse_boundary_rows": "identity_with_zero_component_wall_rhs",
        "reference": "converged_FreeGS_total_psi",
        "response_sign": "minus_native_inverse_of_stationarity_residual",
        "solver_physics_changed": False,
    }


def _routing(response_decomposition: dict[str, Any]) -> dict[str, Any]:
    components = cast(dict[str, dict[str, Any]], response_decomposition["components"])
    dominant = max(COMPONENTS, key=lambda name: float(components[name]["l2_wb"]))
    if dominant == "exact_source_convention":
        target = "current_support_and_source_convention"
    elif dominant == "native_second_order_stencil":
        target = "operator_order_and_grid_convergence"
    elif dominant == "coil_vacuum_discretisation":
        target = "coil_vacuum_grid_convergence"
    else:
        target = "freegs_fourth_order_reference_residual"
    return {
        "dominant_response_component": dominant,
        "next_ratcheting_target": target,
        "solver_physics_changed": False,
    }


def build_report(
    *,
    generated_at: str,
    environment: dict[str, Any],
    source_artifacts: dict[str, dict[str, Any]],
    bindings: dict[str, dict[str, Any]],
    forcing_decomposition: dict[str, Any],
    response_decomposition: dict[str, Any],
    closure: dict[str, float],
) -> dict[str, Any]:
    """Build and validate one immutable operator-response evidence payload."""
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "bindings": bindings,
        "blockers": sorted(BASE_BLOCKERS),
        "claim_boundary": {field: False for field in CLAIM_FIELDS},
        "closure": closure,
        "environment": environment,
        "forcing_decomposition": forcing_decomposition,
        "generated_at": generated_at,
        "input_contract": _input_contract(),
        "response_decomposition": response_decomposition,
        "routing": _routing(response_decomposition),
        "schema_version": SCHEMA_VERSION,
        "source_artifacts": source_artifacts,
        "status": "diagnostic_complete_claims_blocked",
    }
    report["payload_sha256"] = _payload_sha256(report)
    validate_report(report)
    return report


def _validate_metric(
    value: object,
    *,
    field: str,
    expected_fields: set[str],
    response: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{field} fields are invalid")
    row = cast(dict[str, Any], value)
    _require_sha256(row["field_sha256"], field=f"{field}.field_sha256")
    if response:
        _require_number(
            row["cosine_to_terminal_error"],
            field=f"{field}.cosine_to_terminal_error",
            minimum=-1.0,
            maximum=1.0,
        )
        for name in ("l2_wb", "linf_wb", "relative_l2_to_terminal_error"):
            _require_number(row[name], field=f"{field}.{name}", minimum=0.0)
        _require_number(
            row["projection_on_terminal_error"],
            field=f"{field}.projection_on_terminal_error",
        )
    else:
        for name in ("l2", "linf", "relative_l2_to_exact_source_residual"):
            _require_number(row[name], field=f"{field}.{name}", minimum=0.0)
    return row


def _validate_decomposition(value: object, *, response: bool) -> None:
    total_fields = (
        {"components", "exact_source_total", "native_operator_total"}
        if response
        else {"components", "exact_source_residual", "native_operator_residual"}
    )
    if not isinstance(value, dict) or set(value) != total_fields:
        raise ValueError("decomposition fields are invalid")
    components = value["components"]
    if not isinstance(components, dict) or set(components) != set(COMPONENTS):
        raise ValueError("decomposition components are invalid")
    metric_fields = _RESPONSE_FIELDS if response else _FORCING_FIELDS
    for name in COMPONENTS:
        _validate_metric(
            components[name],
            field=f"components.{name}",
            expected_fields=metric_fields,
            response=response,
        )
    total_names = (
        ("native_operator_total", "exact_source_total")
        if response
        else ("native_operator_residual", "exact_source_residual")
    )
    for name in total_names:
        _validate_metric(
            value[name],
            field=name,
            expected_fields=metric_fields,
            response=response,
        )


def _validate_binding_row(
    value: object,
    *,
    field: str,
    path: str,
    extra_fields: set[str],
) -> dict[str, Any]:
    expected = {"path", "payload_sha256", "source_commit", *extra_fields}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"bindings.{field} fields are invalid")
    row = cast(dict[str, Any], value)
    if row["path"] != path:
        raise ValueError(f"bindings.{field}.path is invalid")
    _require_sha256(row["payload_sha256"], field=f"bindings.{field}.payload_sha256")
    _require_git_oid(row["source_commit"], field=f"bindings.{field}.source_commit")
    for name in extra_fields:
        _require_sha256(row[name], field=f"bindings.{field}.{name}")
    return row


def _validate_bindings(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "fixed_point",
        "operator_decomposition",
        "same_case",
        "source_mechanism",
    }:
        raise ValueError("bindings fields are invalid")
    same = _validate_binding_row(
        value["same_case"],
        field="same_case",
        path=SAME_CASE_PATH,
        extra_fields={"candidate_psi_sha256", "reference_psi_sha256"},
    )
    operator = _validate_binding_row(
        value["operator_decomposition"],
        field="operator_decomposition",
        path=OPERATOR_DECOMPOSITION_PATH,
        extra_fields={"same_case_payload_sha256"},
    )
    mechanism = _validate_binding_row(
        value["source_mechanism"],
        field="source_mechanism",
        path=SOURCE_MECHANISM_PATH,
        extra_fields={"operator_payload_sha256", "same_case_payload_sha256"},
    )
    fixed_point = _validate_binding_row(
        value["fixed_point"],
        field="fixed_point",
        path=FIXED_POINT_PATH,
        extra_fields={"same_case_payload_sha256", "source_mechanism_payload_sha256"},
    )
    same_digest = same["payload_sha256"]
    if operator["same_case_payload_sha256"] != same_digest:
        raise ValueError("operator decomposition does not bind the selected same-case payload")
    if (
        mechanism["same_case_payload_sha256"] != same_digest
        or mechanism["operator_payload_sha256"] != operator["payload_sha256"]
    ):
        raise ValueError("source mechanism does not bind the selected evidence chain")
    if (
        fixed_point["same_case_payload_sha256"] != same_digest
        or fixed_point["source_mechanism_payload_sha256"] != mechanism["payload_sha256"]
    ):
        raise ValueError("fixed point does not bind the selected evidence chain")


def _validate_source_artifacts(value: object) -> None:
    expected = {*SOURCE_PATHS, "freegs_public_example", "repository"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("source_artifacts fields are invalid")
    for name in {*SOURCE_PATHS, "freegs_public_example"}:
        row = value[name]
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError(f"source_artifacts.{name} fields are invalid")
        if name in SOURCE_PATHS and row["path"] != SOURCE_PATHS[name]:
            raise ValueError(f"source_artifacts.{name}.path is invalid")
        _require_sha256(row["sha256"], field=f"source_artifacts.{name}.sha256")
    repository = value["repository"]
    if not isinstance(repository, dict) or set(repository) != {
        "git_commit",
        "path",
        "worktree_clean",
    }:
        raise ValueError("source_artifacts.repository fields are invalid")
    if repository["path"] != "." or repository["worktree_clean"] is not True:
        raise ValueError("operator-response evidence requires a clean canonical repository")
    _require_git_oid(repository["git_commit"], field="source_artifacts.repository.git_commit")


def validate_report(report: dict[str, Any]) -> None:
    """Reject malformed, stale-chain, overclaiming, or numerically open reports."""
    if set(report) != _TOP_LEVEL_FIELDS:
        raise ValueError("operator-response report top-level fields are invalid")
    if report["schema_version"] != SCHEMA_VERSION or report["benchmark_id"] != BENCHMARK_ID:
        raise ValueError("operator-response report identity is invalid")
    if report["status"] != "diagnostic_complete_claims_blocked":
        raise ValueError("operator-response status is invalid")
    if report["input_contract"] != _input_contract():
        raise ValueError("operator-response input contract is invalid")
    if report["blockers"] != sorted(BASE_BLOCKERS):
        raise ValueError("operator-response blockers are invalid")
    if report["claim_boundary"] != {field: False for field in CLAIM_FIELDS}:
        raise ValueError("operator-response claim_boundary must remain false")
    if (
        not isinstance(report["environment"], dict)
        or set(report["environment"]) != _ENVIRONMENT_FIELDS
        or report["environment"]["x64_enabled"] is not True
    ):
        raise ValueError("operator-response environment is invalid")
    _validate_bindings(report["bindings"])
    _validate_source_artifacts(report["source_artifacts"])
    _validate_decomposition(report["forcing_decomposition"], response=False)
    _validate_decomposition(report["response_decomposition"], response=True)
    closure = report["closure"]
    if not isinstance(closure, dict) or set(closure) != _CLOSURE_FIELDS:
        raise ValueError("operator-response closure fields are invalid")
    thresholds = {
        "exact_source_forcing_max_abs": FORCING_CLOSURE_MAX_ABS,
        "exact_source_response_max_abs_wb": RESPONSE_CLOSURE_MAX_ABS_WB,
        "fixed_point_native_operator_max_abs_wb": FIXED_POINT_CONSISTENCY_MAX_ABS_WB,
        "native_operator_forcing_max_abs": FORCING_CLOSURE_MAX_ABS,
        "native_operator_response_max_abs_wb": RESPONSE_CLOSURE_MAX_ABS_WB,
    }
    for name, threshold in thresholds.items():
        if _require_number(closure[name], field=f"closure.{name}", minimum=0.0) > threshold:
            raise ValueError(f"operator-response {name} exceeds the frozen threshold")
    expected_routing = _routing(cast(dict[str, Any], report["response_decomposition"]))
    if report["routing"] != expected_routing:
        raise ValueError("operator-response routing is inconsistent")
    _walk_finite(report)
    if report["payload_sha256"] != _payload_sha256(report):
        raise ValueError("operator-response payload_sha256 is invalid")


def render_markdown(report: dict[str, Any]) -> str:
    """Render the validated payload as concise human-readable evidence."""
    validate_report(report)
    response = cast(dict[str, Any], report["response_decomposition"])
    components = cast(dict[str, dict[str, Any]], response["components"])
    lines = [
        "# IDA operator-response decomposition",
        "",
        f"- Payload: `{report['payload_sha256']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Status: `{report['status']}`",
        f"- Dominant response: `{report['routing']['dominant_response_component']}`",
        f"- Next ratchet: `{report['routing']['next_ratcheting_target']}`",
        "- Solver physics changed: `false`",
        "",
        "## Response-weighted components",
        "",
        "| Component | Relative L2 to terminal error | Projection | Cosine |",
        "|---|---:|---:|---:|",
    ]
    for name in COMPONENTS:
        row = components[name]
        lines.append(
            f"| `{name}` | {row['relative_l2_to_terminal_error']:.12g} "
            f"| {row['projection_on_terminal_error']:.12g} "
            f"| {row['cosine_to_terminal_error']:.12g} |"
        )
    lines.extend(
        [
            "",
            "## Closure",
            "",
            *[f"- `{name}`: `{value:.12g}`" for name, value in sorted(report["closure"].items())],
            "",
            "## Claim boundary",
            "",
            *[f"- {name.replace('_', ' ')}: `false`" for name in CLAIM_FIELDS],
            "",
        ]
    )
    return "\n".join(lines)
