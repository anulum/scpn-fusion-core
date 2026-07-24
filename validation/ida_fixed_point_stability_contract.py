# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed contract for the IDA fixed-point stability diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, cast

SCHEMA_VERSION = "scpn-fusion.ida-fixed-point-stability.v1"
BENCHMARK_ID = "DIII-D-IDA-FB-JAX-B-FIXED-POINT-STABILITY"
EVALUATION_CASE_ID = "freegs_16_diiid_public_example"
GRID_SHAPE = [129, 129]
TRAJECTORY_STEPS = 4
SAME_CASE_PATH = "validation/reports/ida_same_case_evidence.json"
SOURCE_MECHANISM_PATH = "validation/reports/ida_fixed_reference_source_mechanism.json"
CLAIM_FIELDS = (
    "control_admission",
    "facility_validation",
    "pcs_deployment",
    "safety_admission",
    "scientific_validation",
)
COMPONENTS = (
    "native_operator_residual",
    "boundary_anchor",
    "source_mechanism",
)
JVP_DIRECTIONS = ("terminal_error", "source_mechanism")
SOURCE_PATHS = {
    "contract": "validation/ida_fixed_point_stability_contract.py",
    "diagnostic": "validation/diagnose_ida_fixed_point_stability.py",
    "multigrid": "src/scpn_fusion/core/jax_multigrid_precond.py",
    "predictive_solver": "src/scpn_fusion/core/jax_free_boundary_predictive.py",
    "same_case_benchmark": "validation/benchmark_ida_same_case.py",
    "source_ablation": "validation/diagnose_ida_fixed_reference_source.py",
    "source_mechanism": "validation/diagnose_ida_fixed_reference_source_mechanism.py",
}
BASE_BLOCKERS = {
    "facility_validation_not_bound",
    "pcs_and_safety_programmes_not_bound",
    "real_shot_predictive_dataset_missing",
    "same_case_accuracy_threshold_failed",
    "statistically_held_out_case_missing",
}
MAP_PARITY_RELATIVE_L2_MAX = 1.0e-12
MAP_PARITY_MAX_ABS_WB = 1.0e-12
DECOMPOSITION_CLOSURE_MAX_ABS_WB = 1.0e-9

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
_VECTOR_FIELDS = {
    "cosine_to_terminal_error",
    "field_sha256",
    "l2_wb",
    "linf_wb",
    "projection_on_terminal_error",
    "relative_l2_to_terminal_error",
}
_JVP_FIELDS = {
    "alignment_with_input",
    "gain_l2",
    "input_sha256",
    "output_sha256",
}
_TRAJECTORY_FIELDS = {
    "distance_to_candidate_relative_to_terminal",
    "distance_to_reference_relative_to_terminal",
    "projection_on_terminal_error",
    "psi_sha256",
    "step",
}
_TOP_LEVEL_FIELDS = {
    "benchmark_id",
    "bindings",
    "blockers",
    "claim_boundary",
    "decomposition",
    "environment",
    "generated_at",
    "input_contract",
    "jvp_gains",
    "map_parity",
    "payload_sha256",
    "routing",
    "schema_version",
    "source_artifacts",
    "status",
    "trajectory",
}


def _payload_sha256(report: dict[str, Any]) -> str:
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
        "anderson_acceleration": False,
        "evaluation_case_id": EVALUATION_CASE_ID,
        "grid_shape": GRID_SHAPE,
        "inner_solver": "mg_preconditioned_bicgstab",
        "ip_fraction": 1.0,
        "jacobian_method": "jax.linearize_stationary_picard_map",
        "reference": "converged_FreeGS_total_psi",
        "separatrix_refinement": 1.0,
        "trajectory_steps": TRAJECTORY_STEPS,
    }


def _routing(
    *,
    decomposition: dict[str, Any],
    jvp_gains: dict[str, dict[str, Any]],
    map_parity: dict[str, Any],
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    component = max(
        COMPONENTS,
        key=lambda name: float(decomposition["components"][name]["l2_wb"]),
    )
    gain = float(jvp_gains["terminal_error"]["gain_l2"])
    moves_toward_candidate = float(
        trajectory[-1]["distance_to_candidate_relative_to_terminal"]
    ) < float(trajectory[0]["distance_to_candidate_relative_to_terminal"])
    parity_ok = (
        float(map_parity["relative_l2"]) <= MAP_PARITY_RELATIVE_L2_MAX
        and float(map_parity["max_abs_wb"]) <= MAP_PARITY_MAX_ABS_WB
    )
    if not parity_ok:
        target = "stationary_map_parity_failure"
    elif gain > 1.0:
        target = "amplifying_geometry_source_feedback"
    elif moves_toward_candidate:
        target = f"{component}_reference_stationarity"
    else:
        target = "stationary_map_forcing_decomposition"
    return {
        "dominant_forcing_component": component,
        "locally_amplifying_along_terminal_error": gain > 1.0,
        "next_ratcheting_target": target,
        "raw_picard_moves_toward_candidate": moves_toward_candidate,
        "stationary_map_parity_ok": parity_ok,
    }


def build_report(
    *,
    generated_at: str,
    environment: dict[str, Any],
    source_artifacts: dict[str, dict[str, Any]],
    bindings: dict[str, dict[str, Any]],
    decomposition: dict[str, Any],
    map_parity: dict[str, Any],
    jvp_gains: dict[str, dict[str, Any]],
    trajectory: list[dict[str, Any]],
    source_worktree_clean: bool,
) -> dict[str, Any]:
    """Build and validate one immutable fixed-point diagnostic report."""
    blockers = sorted(BASE_BLOCKERS)
    if not source_worktree_clean:
        blockers.append("source_worktree_not_clean")
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "bindings": bindings,
        "blockers": blockers,
        "claim_boundary": {field: False for field in CLAIM_FIELDS},
        "decomposition": decomposition,
        "environment": environment,
        "generated_at": generated_at,
        "input_contract": _input_contract(),
        "jvp_gains": jvp_gains,
        "map_parity": map_parity,
        "routing": _routing(
            decomposition=decomposition,
            jvp_gains=jvp_gains,
            map_parity=map_parity,
            trajectory=trajectory,
        ),
        "schema_version": SCHEMA_VERSION,
        "source_artifacts": source_artifacts,
        "status": "diagnostic_complete_claims_blocked",
        "trajectory": trajectory,
    }
    report["payload_sha256"] = _payload_sha256(report)
    validate_report(report)
    return report


def _validate_vector(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _VECTOR_FIELDS:
        raise ValueError(f"{field} fields are invalid")
    row = cast(dict[str, Any], value)
    _require_sha256(row["field_sha256"], field=f"{field}.field_sha256")
    _require_number(
        row["cosine_to_terminal_error"],
        field=f"{field}.cosine_to_terminal_error",
        minimum=-1.0,
        maximum=1.0,
    )
    _require_number(row["l2_wb"], field=f"{field}.l2_wb", minimum=0.0)
    _require_number(row["linf_wb"], field=f"{field}.linf_wb", minimum=0.0)
    _require_number(
        row["projection_on_terminal_error"],
        field=f"{field}.projection_on_terminal_error",
    )
    _require_number(
        row["relative_l2_to_terminal_error"],
        field=f"{field}.relative_l2_to_terminal_error",
        minimum=0.0,
    )
    return row


def _validate_decomposition(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "closure_max_abs_wb",
        "components",
        "total_forcing",
    }:
        raise ValueError("decomposition fields are invalid")
    row = cast(dict[str, Any], value)
    if not isinstance(row["components"], dict) or set(row["components"]) != set(COMPONENTS):
        raise ValueError("decomposition.components fields are invalid")
    for name in COMPONENTS:
        _validate_vector(row["components"][name], field=f"decomposition.components.{name}")
    _validate_vector(row["total_forcing"], field="decomposition.total_forcing")
    if (
        _require_number(
            row["closure_max_abs_wb"],
            field="decomposition.closure_max_abs_wb",
            minimum=0.0,
        )
        > DECOMPOSITION_CLOSURE_MAX_ABS_WB
    ):
        raise ValueError("decomposition closure exceeds the frozen threshold")
    return row


def _validate_bindings(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"same_case", "source_mechanism"}:
        raise ValueError("bindings fields are invalid")
    same = value["same_case"]
    if not isinstance(same, dict) or set(same) != {
        "candidate_psi_sha256",
        "path",
        "payload_sha256",
        "reference_psi_sha256",
        "source_commit",
    }:
        raise ValueError("bindings.same_case fields are invalid")
    if same["path"] != SAME_CASE_PATH:
        raise ValueError("bindings.same_case.path is invalid")
    for name in ("candidate_psi_sha256", "payload_sha256", "reference_psi_sha256"):
        _require_sha256(same[name], field=f"bindings.same_case.{name}")
    _require_git_oid(same["source_commit"], field="bindings.same_case.source_commit")
    mechanism = value["source_mechanism"]
    if not isinstance(mechanism, dict) or set(mechanism) != {
        "path",
        "payload_sha256",
        "same_case_payload_sha256",
        "source_commit",
    }:
        raise ValueError("bindings.source_mechanism fields are invalid")
    if mechanism["path"] != SOURCE_MECHANISM_PATH:
        raise ValueError("bindings.source_mechanism.path is invalid")
    _require_sha256(mechanism["payload_sha256"], field="bindings.source_mechanism.payload_sha256")
    _require_git_oid(mechanism["source_commit"], field="bindings.source_mechanism.source_commit")
    if mechanism["same_case_payload_sha256"] != same["payload_sha256"]:
        raise ValueError("source mechanism does not bind the selected same-case payload")


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
    if repository["path"] != "." or not isinstance(repository["worktree_clean"], bool):
        raise ValueError("source_artifacts.repository fields are invalid")
    _require_git_oid(repository["git_commit"], field="source_artifacts.repository.git_commit")


def validate_report(report: dict[str, Any]) -> None:
    """Reject schema drift, tamper, overclaim, and derived-routing forgery."""
    if set(report) != _TOP_LEVEL_FIELDS:
        raise ValueError("fixed-point report top-level fields do not match v1")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported fixed-point stability contract")
    if report.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("fixed-point benchmark_id is invalid")
    _walk_finite(report)
    if report.get("payload_sha256") != _payload_sha256(report):
        raise ValueError("payload_sha256 does not match report content")
    _require_sha256(report.get("payload_sha256"), field="payload_sha256")
    if report.get("claim_boundary") != {field: False for field in CLAIM_FIELDS}:
        raise ValueError("claim_boundary must keep every promotion claim false")
    if report.get("status") != "diagnostic_complete_claims_blocked":
        raise ValueError("fixed-point diagnostic cannot promote an admitted status")
    blockers = report.get("blockers")
    if not isinstance(blockers, list) or not BASE_BLOCKERS.issubset(set(blockers)):
        raise ValueError("blockers must preserve every base admission blocker")
    if report.get("input_contract") != _input_contract():
        raise ValueError("input_contract does not match the frozen v1 diagnostic")
    environment = report.get("environment")
    if not isinstance(environment, dict) or set(environment) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment fields do not match the runtime contract")
    if environment["x64_enabled"] is not True:
        raise ValueError("environment must bind JAX FP64 execution")
    _validate_source_artifacts(report.get("source_artifacts"))
    _validate_bindings(report.get("bindings"))
    decomposition = _validate_decomposition(report.get("decomposition"))
    map_parity = report.get("map_parity")
    if not isinstance(map_parity, dict) or set(map_parity) != {
        "manual_map_sha256",
        "max_abs_wb",
        "production_map_sha256",
        "relative_l2",
        "sha256_match",
    }:
        raise ValueError("map_parity fields are invalid")
    manual_sha = _require_sha256(
        map_parity["manual_map_sha256"],
        field="map_parity.manual_map_sha256",
    )
    production_sha = _require_sha256(
        map_parity["production_map_sha256"],
        field="map_parity.production_map_sha256",
    )
    if map_parity["sha256_match"] is not (manual_sha == production_sha):
        raise ValueError("map_parity.sha256_match is inconsistent")
    _require_number(map_parity["max_abs_wb"], field="map_parity.max_abs_wb", minimum=0.0)
    _require_number(map_parity["relative_l2"], field="map_parity.relative_l2", minimum=0.0)
    jvp_gains = report.get("jvp_gains")
    if not isinstance(jvp_gains, dict) or set(jvp_gains) != set(JVP_DIRECTIONS):
        raise ValueError("jvp_gains fields are invalid")
    jvp_rows = cast(dict[str, dict[str, Any]], jvp_gains)
    for name, row in jvp_rows.items():
        if not isinstance(row, dict) or set(row) != _JVP_FIELDS:
            raise ValueError(f"jvp_gains.{name} fields are invalid")
        _require_sha256(row["input_sha256"], field=f"jvp_gains.{name}.input_sha256")
        _require_sha256(row["output_sha256"], field=f"jvp_gains.{name}.output_sha256")
        _require_number(row["gain_l2"], field=f"jvp_gains.{name}.gain_l2", minimum=0.0)
        _require_number(
            row["alignment_with_input"],
            field=f"jvp_gains.{name}.alignment_with_input",
            minimum=-1.0,
            maximum=1.0,
        )
    trajectory = report.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) != TRAJECTORY_STEPS + 1:
        raise ValueError("trajectory must contain the frozen number of steps")
    rows = cast(list[dict[str, Any]], trajectory)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _TRAJECTORY_FIELDS:
            raise ValueError(f"trajectory[{index}] fields are invalid")
        if row["step"] != index:
            raise ValueError("trajectory steps must be contiguous and ordered")
        _require_sha256(row["psi_sha256"], field=f"trajectory[{index}].psi_sha256")
        for name in (
            "distance_to_candidate_relative_to_terminal",
            "distance_to_reference_relative_to_terminal",
        ):
            _require_number(row[name], field=f"trajectory[{index}].{name}", minimum=0.0)
        _require_number(
            row["projection_on_terminal_error"],
            field=f"trajectory[{index}].projection_on_terminal_error",
        )
    expected_routing = _routing(
        decomposition=decomposition,
        jvp_gains=jvp_rows,
        map_parity=cast(dict[str, Any], map_parity),
        trajectory=rows,
    )
    if report.get("routing") != expected_routing:
        raise ValueError("routing is inconsistent with measured fixed-point evidence")


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise fixed-point stability summary."""
    validate_report(report)
    lines = [
        "# IDA fixed-point stability diagnostic",
        "",
        f"- Status: `{report['status']}`",
        f"- Payload SHA-256: `{report['payload_sha256']}`",
        "- Facility/control/PCS/safety/scientific admission: `false`",
        "",
        "## Stationary-map forcing decomposition",
        "",
        "| Component | relative L2 to terminal error | projection on terminal error | cosine |",
        "|---|---:|---:|---:|",
    ]
    for name in COMPONENTS:
        row = report["decomposition"]["components"][name]
        lines.append(
            f"| {name} | {row['relative_l2_to_terminal_error']:.9g} | "
            f"{row['projection_on_terminal_error']:.9g} | "
            f"{row['cosine_to_terminal_error']:.9g} |"
        )
    total = report["decomposition"]["total_forcing"]
    lines.append(
        f"| **total** | {total['relative_l2_to_terminal_error']:.9g} | "
        f"{total['projection_on_terminal_error']:.9g} | "
        f"{total['cosine_to_terminal_error']:.9g} |"
    )
    lines.extend(
        [
            "",
            "## Local gains and frozen-map trajectory",
            "",
            f"- Terminal-error JVP gain: `{report['jvp_gains']['terminal_error']['gain_l2']:.9g}`",
            f"- Source-mechanism JVP gain: "
            f"`{report['jvp_gains']['source_mechanism']['gain_l2']:.9g}`",
            f"- Raw Picard map moves toward candidate: "
            f"`{str(report['routing']['raw_picard_moves_toward_candidate']).lower()}`",
            f"- Next ratcheting target: `{report['routing']['next_ratcheting_target']}`",
            "",
            "| Step | distance to reference / terminal | distance to candidate / terminal | projection |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in report["trajectory"]:
        lines.append(
            f"| {row['step']} | "
            f"{row['distance_to_reference_relative_to_terminal']:.9g} | "
            f"{row['distance_to_candidate_relative_to_terminal']:.9g} | "
            f"{row['projection_on_terminal_error']:.9g} |"
        )
    lines.extend(
        [
            "",
            "This is a single-case engineering diagnostic, not experimental validation.",
            "",
        ]
    )
    return "\n".join(lines)
