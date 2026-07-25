# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Evidence Contract
"""Fail-closed contract for IDA coil-vacuum grid-convergence evidence."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from typing import Any, cast

SCHEMA_VERSION = "scpn-fusion.ida-coil-vacuum-grid-convergence.v1"
BENCHMARK_ID = "DIII-D-IDA-FB-JAX-B-CVGC1"
EVALUATION_CASE_ID = "freegs_16_diiid_public_example"
GRID_RESOLUTIONS = (33, 65, 129, 257)
GRID_TRIPLES = ("33_65_129", "65_129_257")
SAME_CASE_PATH = "validation/reports/ida_same_case_evidence.json"
SOURCE_ABLATION_PATH = "validation/reports/ida_fixed_reference_source_ablation.json"
OPERATOR_PATH = "validation/reports/ida_fixed_reference_operator_residual.json"
SOURCE_MECHANISM_PATH = "validation/reports/ida_fixed_reference_source_mechanism.json"
FIXED_POINT_PATH = "validation/reports/ida_fixed_point_stability.json"
RESPONSE_PATH = "validation/reports/ida_operator_response.json"
REPORT_PATH = "validation/reports/ida_coil_vacuum_grid_convergence.json"
MARKDOWN_PATH = "validation/reports/ida_coil_vacuum_grid_convergence.md"

EXPECTED_PAYLOADS = {
    "fixed_point": "54cfbc43d5337fce5c1667888e21a51c9936189def728b5dc062b30649e85a41",
    "operator": "e90161760b6ff1d1d803041da98bcf63781fcecd6ee2f1741a5f3b083e305dda",
    "response": "56eddd06af69d4726433aaed1aff43bb1f92af4def1a4efecbe5a7ab81de911e",
    "same_case": "32f0b51b4c80ffc011dc02c1054ab1d11668c1ec5f38fc6ac0570f87cd8c9a3c",
    "source_ablation": "e84c92dacb0f42b812a82d35796e6a94d8c8a15f72d431579ef71fbc43d941f4",
    "source_mechanism": "7430cd866d69a8c101e7c914d78cd3c989e2e6640d517a52f354c14642a3655c",
}
EXPECTED_BINDING_PATHS = {
    "fixed_point": FIXED_POINT_PATH,
    "operator": OPERATOR_PATH,
    "response": RESPONSE_PATH,
    "same_case": SAME_CASE_PATH,
    "source_ablation": SOURCE_ABLATION_PATH,
    "source_mechanism": SOURCE_MECHANISM_PATH,
}
EXPECTED_ANCHOR_FORCING_SHA256 = "ddc8b3c0b4b0ecd041b87f2549a56cd03bfb1c0acf5eb680fca94e174e9f90e2"
EXPECTED_ANCHOR_RESPONSE_SHA256 = "b075e6fa94e40b3ccb01d0cd700a5ca6df7408cc8fbb2666c9d86cb15ecba2b9"

CLAIM_FIELDS = (
    "collaborator_validation",
    "control_admission",
    "experimental_diiid_validation",
    "facility_validation",
    "held_out_validation",
    "isolated_latency_admission",
    "pcs_deployment",
    "production_physics_admission",
    "safety_admission",
    "scientific_validation",
)
SOURCE_PATHS = {
    "contract": "validation/ida_coil_vacuum_grid_contract.py",
    "diagnostic": "validation/diagnose_ida_coil_vacuum_grid_convergence.py",
    "convergence": "validation/ida_coil_vacuum_grid_convergence.py",
    "field_operations": "validation/ida_coil_vacuum_grid_fields.py",
    "manifest_validation": "validation/ida_coil_vacuum_grid_manifest.py",
    "measurement_validation": "validation/ida_coil_vacuum_grid_measurement.py",
    "fixed_operator_diagnostic": "validation/diagnose_ida_fixed_reference_operator.py",
    "operator_response_contract": "validation/ida_operator_response_contract.py",
    "operator_response_diagnostic": "validation/diagnose_ida_operator_response.py",
    "operator_response_fields": "validation/ida_operator_response_fields.py",
    "predictive_solver": "src/scpn_fusion/core/jax_free_boundary_predictive.py",
    "runtime": "validation/ida_coil_vacuum_grid_runtime.py",
}
RUNTIME_SOURCE_NAMES = (
    "freegs_boundary",
    "freegs_machine",
    "freegs_operator",
    "freegs_public_example",
    "freegs_shaped_coil",
)
BASE_BLOCKERS = {
    "collaborator_evidence_missing",
    "facility_validation_not_bound",
    "held_out_real_shot_evidence_missing",
    "isolated_latency_evidence_missing",
    "parent_b_accuracy_admission_blocked",
    "pcs_and_safety_programmes_not_bound",
}

PARITY_NRMSE_MAX = 1.0e-12
PARITY_MAX_ABS_WB = 1.0e-12
PARTITION_CLOSURE_MAX_ABS = 1.0e-10
CURRENT_RECOVERY_WEIGHTED_MAX = 0.05
CURRENT_TREND_SLACK = 1.0e-9
SOURCE_FREE_ORDER_MIN = 1.5
FINEST_RESPONSE_RELATIVE_L2_MAX = 0.05
SOURCE_LOCALISATION_L2_FRACTION_MIN = 0.95
R_BOUNDS_M = (0.1, 2.8)
Z_BOUNDS_M = (-1.8, 1.8)
FIXED_PHYSICAL_RADIUS_M = 0.225

FIELD_METRIC_FIELDS = {
    "area_weighted_l2",
    "area_weighted_rms",
    "field_sha256",
    "l2",
    "linf",
}
COMPARISON_METRIC_FIELDS = {
    "area_weighted_l2",
    "area_weighted_rms",
    "cosine",
    "linf",
    "projection",
    "relative_l2",
}
RECOVERY_METRIC_FIELDS = {
    "absolute_error_a_turns",
    "expected_a_turns",
    "recovered_a_turns",
    "relative_error",
    "signed_error_a_turns",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_TOP_LEVEL_FIELDS = {
    "anchor",
    "benchmark_id",
    "bindings",
    "blockers",
    "claim_boundary",
    "coil_manifest",
    "convergence",
    "environment",
    "gates",
    "generated_at",
    "grids",
    "hypotheses",
    "input_contract",
    "payload_sha256",
    "routing",
    "schema_version",
    "source_artifacts",
    "status",
}


def _payload_sha256(report: dict[str, Any]) -> str:
    """Return the canonical report digest excluding its signature field."""
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


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _input_contract() -> dict[str, Any]:
    """Return the immutable numerical design embedded in every report."""
    return {
        "evaluation_case_id": EVALUATION_CASE_ID,
        "fixed_physical_radius": "two_times_coarse_grid_max_spacing",
        "grid_comparison_regions": [
            "fixed_physical_source_free",
            "full",
            "plasma_support",
            "source_footprint",
        ],
        "grid_relative_radii": [1.0, 2.0, 4.0],
        "inner_solver": "mg_preconditioned_bicgstab",
        "inverse_boundary_rows": "identity_with_zero_component_wall_rhs",
        "primary_source_radius_rho_h": 2.0,
        "r_bounds_m": list(R_BOUNDS_M),
        "required_resolutions": list(GRID_RESOLUTIONS),
        "response_sign": "minus_native_inverse_of_stationarity_residual",
        "solver_physics_changed": False,
        "z_bounds_m": list(Z_BOUNDS_M),
    }


def _hypotheses() -> dict[str, str]:
    """Return the six predeclared hypotheses without post-result editing."""
    return {
        "H1": "raw_forcing_is_localised_to_the_deterministic_source_footprint",
        "H2": "fixed_physical_source_free_operator_error_is_second_order",
        "H3": "parent_coil_effective_current_is_recovered_on_fine_grids",
        "H4": "source_and_source_free_inverse_responses_close_to_total",
        "H5": "source_free_response_is_stable_under_refinement",
        "H6": "freegs_and_native_green_function_conventions_do_not_drift",
    }


def _structural_gates(
    *,
    anchor: dict[str, Any],
    coil_manifest: dict[str, Any],
    grids: list[dict[str, Any]],
) -> dict[str, bool]:
    resolutions = [row.get("resolution") for row in grids]
    parity = all(bool(row["vacuum_fields"]["source_free_parity"]["passes"]) for row in grids)
    closures = all(
        _require_number(
            row["forcing_partition"]["closure_max_abs"],
            field="forcing_partition.closure_max_abs",
            minimum=0.0,
        )
        <= PARTITION_CLOSURE_MAX_ABS
        and _require_number(
            row["response_partition"]["closure_max_abs_wb"],
            field="response_partition.closure_max_abs_wb",
            minimum=0.0,
        )
        <= PARTITION_CLOSURE_MAX_ABS
        for row in grids
    )
    return {
        "anchor_exact": (
            anchor.get("forcing_sha256") == EXPECTED_ANCHOR_FORCING_SHA256
            and anchor.get("response_sha256") == EXPECTED_ANCHOR_RESPONSE_SHA256
        ),
        "four_grid_ladder_complete": resolutions == list(GRID_RESOLUTIONS),
        "manifest_integrity": (
            coil_manifest.get("parent_count") == 18 and coil_manifest.get("filament_count") == 216
        ),
        "partition_closure": closures,
        "runtime_and_upstream_binding": True,
        "source_free_green_parity": parity,
    }


def _numerical_gates(
    grids: list[dict[str, Any]],
    convergence: dict[str, Any],
) -> dict[str, bool]:
    by_resolution = {int(row["resolution"]): row for row in grids}
    recovery = {
        resolution: float(by_resolution[resolution]["current_recovery"]["weighted_primary_error"])
        for resolution in GRID_RESOLUTIONS
    }
    trend = all(
        recovery[right] <= recovery[left] * (1.0 + CURRENT_TREND_SLACK) + 1.0e-15
        for left, right in ((65, 129), (129, 257))
    )
    orders = cast(dict[str, dict[str, float]], convergence["source_free_forcing_order"])
    return {
        "current_recovery_fine": (
            recovery[129] <= CURRENT_RECOVERY_WEIGHTED_MAX
            and recovery[257] <= CURRENT_RECOVERY_WEIGHTED_MAX
        ),
        "current_recovery_non_increasing": trend,
        "finest_response_stability": (
            float(convergence["finest_source_free_response"]["relative_l2"])
            <= FINEST_RESPONSE_RELATIVE_L2_MAX
        ),
        "source_free_observed_order": all(
            float(orders[name]["observed_order"]) >= SOURCE_FREE_ORDER_MIN for name in GRID_TRIPLES
        ),
        "source_localisation": all(
            float(row["forcing_partition"]["primary_l2_fraction"])
            >= SOURCE_LOCALISATION_L2_FRACTION_MIN
            for row in grids
        ),
    }


def _routing(
    structural: dict[str, bool],
    numerical: dict[str, bool],
) -> dict[str, Any]:
    structural_pass = all(structural.values())
    source_pass = (
        numerical["current_recovery_fine"]
        and numerical["current_recovery_non_increasing"]
        and numerical["source_localisation"]
    )
    vacuum_pass = numerical["source_free_observed_order"] and numerical["finest_response_stability"]
    if not structural_pass:
        state = "blocked_incomplete_required_ladder"
    elif source_pass and vacuum_pass:
        state = "coil_source_footprint_resolved"
    elif source_pass:
        state = "vacuum_operator_discretisation_unresolved"
    elif vacuum_pass:
        state = "coil_source_discretisation_unresolved"
    else:
        state = "mixed_source_and_vacuum_error"
    return {
        "solver_physics_changed": False,
        "source_numerics_pass": source_pass,
        "state": state,
        "vacuum_numerics_pass": vacuum_pass,
    }


def build_report(
    *,
    generated_at: str,
    environment: dict[str, Any],
    source_artifacts: dict[str, dict[str, Any]],
    bindings: dict[str, dict[str, Any]],
    anchor: dict[str, Any],
    coil_manifest: dict[str, Any],
    grids: list[dict[str, Any]],
    convergence: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate one complete four-grid evidence payload."""
    structural = _structural_gates(anchor=anchor, coil_manifest=coil_manifest, grids=grids)
    numerical = _numerical_gates(grids, convergence)
    routing = _routing(structural, numerical)
    report: dict[str, Any] = {
        "anchor": anchor,
        "benchmark_id": BENCHMARK_ID,
        "bindings": bindings,
        "blockers": sorted(BASE_BLOCKERS),
        "claim_boundary": {field: False for field in CLAIM_FIELDS},
        "coil_manifest": coil_manifest,
        "convergence": convergence,
        "environment": environment,
        "gates": {
            "all_structural_pass": all(structural.values()),
            "numerical": numerical,
            "structural": structural,
        },
        "generated_at": generated_at,
        "grids": grids,
        "hypotheses": _hypotheses(),
        "input_contract": _input_contract(),
        "routing": routing,
        "schema_version": SCHEMA_VERSION,
        "source_artifacts": source_artifacts,
        "status": (
            "diagnostic_complete_claims_blocked"
            if all(structural.values())
            else "blocked_incomplete_required_ladder"
        ),
    }
    report["payload_sha256"] = _payload_sha256(report)
    validate_report(report)
    return report


def validate_report(report: dict[str, Any]) -> None:
    """Reject incomplete, stale, overclaiming, or internally inconsistent evidence."""
    manifest_validation = cast(
        Any,
        importlib.import_module("validation.ida_coil_vacuum_grid_manifest"),
    )
    measurement_validation = cast(
        Any,
        importlib.import_module("validation.ida_coil_vacuum_grid_measurement"),
    )
    if set(report) != _TOP_LEVEL_FIELDS:
        raise ValueError("grid-convergence report top-level fields are invalid")
    if report["schema_version"] != SCHEMA_VERSION or report["benchmark_id"] != BENCHMARK_ID:
        raise ValueError("grid-convergence report identity is invalid")
    if (
        not isinstance(report["generated_at"], str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", report["generated_at"]) is None
    ):
        raise ValueError("grid-convergence generated_at is invalid")
    if not isinstance(report["environment"], dict) or not report["environment"]:
        raise ValueError("grid-convergence environment is invalid")
    if report["input_contract"] != _input_contract() or report["hypotheses"] != _hypotheses():
        raise ValueError("grid-convergence frozen design is invalid")
    if report["blockers"] != sorted(BASE_BLOCKERS):
        raise ValueError("grid-convergence blockers are invalid")
    if report["claim_boundary"] != {field: False for field in CLAIM_FIELDS}:
        raise ValueError("grid-convergence claim boundary must remain false")
    manifest_validation.validate_bindings(report["bindings"])
    manifest_validation.validate_source_artifacts(report["source_artifacts"])
    manifest_validation.validate_anchor(report["anchor"])
    manifest = cast(dict[str, Any], report["coil_manifest"])
    manifest_validation.validate_manifest(manifest)
    grids = measurement_validation.validate_grids(report["grids"], manifest=manifest)
    convergence = measurement_validation.validate_convergence(report["convergence"])
    structural = _structural_gates(
        anchor=cast(dict[str, Any], report["anchor"]),
        coil_manifest=cast(dict[str, Any], report["coil_manifest"]),
        grids=grids,
    )
    numerical = _numerical_gates(grids, cast(dict[str, Any], convergence))
    expected_gates = {
        "all_structural_pass": all(structural.values()),
        "numerical": numerical,
        "structural": structural,
    }
    if report["gates"] != expected_gates:
        raise ValueError("grid-convergence gates are inconsistent")
    expected_routing = _routing(structural, numerical)
    if report["routing"] != expected_routing:
        raise ValueError("grid-convergence routing is inconsistent")
    expected_status = (
        "diagnostic_complete_claims_blocked"
        if all(structural.values())
        else "blocked_incomplete_required_ladder"
    )
    if report["status"] != expected_status:
        raise ValueError("grid-convergence status is inconsistent")
    _walk_finite(report)
    if report["payload_sha256"] != _payload_sha256(report):
        raise ValueError("grid-convergence payload_sha256 is invalid")


def render_markdown(report: dict[str, Any]) -> str:
    """Render validated evidence as concise human-readable Markdown."""
    validate_report(report)
    lines = [
        "# IDA coil-vacuum grid convergence",
        "",
        f"- Payload: `{report['payload_sha256']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Status: `{report['status']}`",
        f"- Routing: `{report['routing']['state']}`",
        "- Solver physics changed: `false`",
        "",
        "## Grid ladder",
        "",
        "| Grid | Primary forcing L2 fraction | Weighted current error | Response closure [Wb] |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["grids"]:
        lines.append(
            f"| {row['resolution']} | "
            f"{row['forcing_partition']['primary_l2_fraction']:.12g} | "
            f"{row['current_recovery']['weighted_primary_error']:.12g} | "
            f"{row['response_partition']['closure_max_abs_wb']:.12g} |"
        )
    lines.extend(["", "## Source-free observed order", ""])
    for name, row in report["convergence"]["source_free_forcing_order"].items():
        lines.append(f"- `{name}`: `{row['observed_order']:.12g}`")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            *[
                f"- {name.replace('_', ' ')}: `{str(value).lower()}`"
                for name, value in sorted(report["gates"]["numerical"].items())
            ],
            "",
            "## Claim boundary",
            "",
            *[f"- {name.replace('_', ' ')}: `false`" for name in CLAIM_FIELDS],
            "",
        ]
    )
    return "\n".join(lines)
