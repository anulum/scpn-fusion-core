# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Fixed-Physical Coil-Vacuum Contract
"""Fail-closed schema for fixed-physical coil-vacuum response evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, cast

from validation import ida_coil_vacuum_grid_contract as grid_contract
from validation.ida_coil_vacuum_grid_fields import canonical_sha256

SCHEMA_VERSION = "scpn-fusion.ida-coil-vacuum-fixed-physical-response.v1"
BENCHMARK_ID = "diiid_ida_coil_vacuum_fixed_physical_response"
REPORT_PATH = "validation/reports/ida_coil_vacuum_fixed_physical_response.json"
MARKDOWN_PATH = "validation/reports/ida_coil_vacuum_fixed_physical_response.md"
UPSTREAM_PATH = grid_contract.REPORT_PATH
EXPECTED_UPSTREAM_PAYLOAD = "7f04e4cb4217d920f19eaecfbd7738b86aa9db93fa37894df3b4f68c7f211193"
EXPECTED_ANCHOR_SHA256 = "9813680aacf307026fee29f9299080b2266d281e5546758680777269323c33ec"
EXPECTED_COIL_MANIFEST_SHA256 = "63c9d649c8c509340d0fc0a21588b9ec4fe873bdf00f19c4ba44fa2a85b705b1"
FIXED_PHYSICAL_RADIUS_M = 0.225
PARTITION_CLOSURE_MAX_ABS = grid_contract.PARTITION_CLOSURE_MAX_ABS
CURRENT_RECOVERY_WEIGHTED_MAX = grid_contract.CURRENT_RECOVERY_WEIGHTED_MAX
CURRENT_TREND_SLACK = grid_contract.CURRENT_TREND_SLACK
SOURCE_LOCALISATION_L2_FRACTION_MIN = grid_contract.SOURCE_LOCALISATION_L2_FRACTION_MIN
SOURCE_FREE_ORDER_MIN = grid_contract.SOURCE_FREE_ORDER_MIN
RESPONSE_FRACTION_OF_TOTAL_MAX = grid_contract.FINEST_RESPONSE_RELATIVE_L2_MAX
GRID_TRIPLES = grid_contract.GRID_TRIPLES
CLAIM_FIELDS = grid_contract.CLAIM_FIELDS
BASE_BLOCKERS = grid_contract.BASE_BLOCKERS
SOURCE_PATHS = {
    "fixed_physical_contract": "validation/ida_coil_vacuum_fixed_physical_contract.py",
    "fixed_physical_diagnostic": ("validation/diagnose_ida_coil_vacuum_fixed_physical_response.py"),
    "fixed_physical_response": "validation/ida_coil_vacuum_fixed_physical_response.py",
    "grid_convergence": "validation/ida_coil_vacuum_grid_convergence.py",
    "grid_diagnostic": "validation/diagnose_ida_coil_vacuum_grid_convergence.py",
    "grid_runtime": "validation/ida_coil_vacuum_grid_runtime.py",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OID_RE = re.compile(r"[0-9a-f]{40,64}")
_TOP_LEVEL_FIELDS = {
    "benchmark_id",
    "blockers",
    "claim_boundary",
    "convergence",
    "environment",
    "execution_binding",
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
    "upstream_binding",
}


def _canonical_json(value: object) -> bytes:
    """Serialise one value with the report's canonical JSON form."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(report: dict[str, Any]) -> str:
    """Hash all report fields except the digest itself."""
    return hashlib.sha256(
        _canonical_json({key: value for key, value in report.items() if key != "payload_sha256"})
    ).hexdigest()


def _walk_finite(value: object, *, field: str = "report") -> None:
    """Reject booleans-as-numbers and non-finite values recursively."""
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
        for key, item in value.items():
            _walk_finite(item, field=f"{field}.{key}")
        return
    raise ValueError(f"{field} contains an unsupported value")


def _number(value: object, *, field: str, minimum: float = 0.0) -> float:
    """Return one finite non-boolean number at or above a bound."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} is outside its valid range")
    return result


def _input_contract() -> dict[str, Any]:
    """Return the pre-execution numerical design."""
    return {
        "fixed_physical_radius_m": FIXED_PHYSICAL_RADIUS_M,
        "grid_resolutions": list(grid_contract.GRID_RESOLUTIONS),
        "inverse_boundary_rows": "identity_with_zero_component_wall_rhs",
        "inverse_initial_guess": "zero",
        "inverse_solver": "mg_preconditioned_bicgstab",
        "mask_application": "fixed_physical_partition_before_inverse",
        "production_solver_physics_changed": False,
        "response_amplitude_denominator": "unchanged_total_response_area_weighted_l2",
        "response_sign": "minus_native_inverse_of_stationarity_residual",
        "upstream_schema": grid_contract.SCHEMA_VERSION,
    }


def _hypotheses() -> dict[str, str]:
    """Return the frozen CVGC2 hypotheses."""
    return {
        "H1": "fixed_physical_source_current_recovery_is_monotone_on_fine_grids",
        "H2": "fixed_physical_source_free_forcing_is_second_order",
        "H3": "fixed_physical_source_free_response_is_second_order",
        "H4": "fixed_partition_inverse_responses_close_to_the_unchanged_total",
        "H5": "fine_grid_source_free_response_is_below_the_frozen_total_response_fraction",
    }


def load_upstream_report(root: Path) -> dict[str, Any]:
    """Load and validate the immutable CVGC1 report from a repository root."""
    path = root / UPSTREAM_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CVGC1 upstream report must be an object")
    report = cast(dict[str, Any], value)
    grid_contract.validate_report(report)
    if report["payload_sha256"] != EXPECTED_UPSTREAM_PAYLOAD:
        raise ValueError("CVGC1 upstream payload does not match the frozen binding")
    return report


def upstream_binding(root: Path, report: dict[str, Any]) -> dict[str, str]:
    """Return exact path, payload, and file digests for CVGC1."""
    content = (root / UPSTREAM_PATH).read_bytes()
    return {
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "path": UPSTREAM_PATH,
        "payload_sha256": str(report["payload_sha256"]),
        "schema_version": str(report["schema_version"]),
    }


def build_execution_binding(
    *,
    anchor: object,
    coil_manifest: object,
    source_artifacts: object,
) -> dict[str, str]:
    """Bind the frozen upstream objects and this execution's source manifest."""
    binding = {
        "anchor_sha256": canonical_sha256(anchor),
        "coil_manifest_sha256": canonical_sha256(coil_manifest),
        "source_artifacts_sha256": canonical_sha256(source_artifacts),
    }
    if binding["anchor_sha256"] != EXPECTED_ANCHOR_SHA256:
        raise ValueError("CVGC2 anchor does not match the frozen CVGC1 anchor")
    if binding["coil_manifest_sha256"] != EXPECTED_COIL_MANIFEST_SHA256:
        raise ValueError("CVGC2 coil manifest does not match the frozen CVGC1 manifest")
    return binding


def _validate_source_artifacts(value: object) -> None:
    """Validate executed-source hashes and clean repository provenance."""
    if not isinstance(value, dict) or set(value) != {*SOURCE_PATHS, "repository"}:
        raise ValueError("source artifact set is invalid")
    repository = value["repository"]
    if not isinstance(repository, dict) or set(repository) != {
        "git_commit",
        "path",
        "worktree_clean",
    }:
        raise ValueError("repository provenance is invalid")
    if (
        _GIT_OID_RE.fullmatch(str(repository["git_commit"])) is None
        or repository["path"] != "."
        or repository["worktree_clean"] is not True
    ):
        raise ValueError("CVGC2 execution requires a clean canonical source commit")
    for name, path in SOURCE_PATHS.items():
        artifact = value[name]
        if not isinstance(artifact, dict) or artifact.get("path") != path:
            raise ValueError(f"{name} source path is invalid")
        if _SHA256_RE.fullmatch(str(artifact.get("sha256"))) is None:
            raise ValueError(f"{name} source digest is invalid")


def _validate_field_metric(value: object, *, field: str) -> None:
    """Validate one field metric emitted by the shared runtime."""
    if not isinstance(value, dict) or set(value) != {
        "area_weighted_l2",
        "area_weighted_rms",
        "field_sha256",
        "l2",
        "linf",
    }:
        raise ValueError(f"{field} metric fields are invalid")
    for name in ("area_weighted_l2", "area_weighted_rms", "l2", "linf"):
        _number(value[name], field=f"{field}.{name}")
    if _SHA256_RE.fullmatch(str(value["field_sha256"])) is None:
        raise ValueError(f"{field}.field_sha256 is invalid")


def _validate_grids(value: object) -> list[dict[str, Any]]:
    """Validate the complete ordered four-grid measurement set."""
    if not isinstance(value, list) or len(value) != len(grid_contract.GRID_RESOLUTIONS):
        raise ValueError("CVGC2 grids must contain the complete four-grid ladder")
    rows: list[dict[str, Any]] = []
    expected = {
        "current_recovery_weighted_error",
        "fixed_source_l2_fraction",
        "forcing_closure_max_abs",
        "forcing_partition",
        "inverse_timings_ms",
        "resolution",
        "response_partition",
        "source_free_response_fraction_of_total",
        "total_response_reproduction_max_abs_wb",
    }
    for value_row in value:
        if not isinstance(value_row, dict) or set(value_row) != expected:
            raise ValueError("CVGC2 grid fields are invalid")
        row = cast(dict[str, Any], value_row)
        if isinstance(row["resolution"], bool) or not isinstance(row["resolution"], int):
            raise ValueError("CVGC2 resolution is invalid")
        for name in (
            "current_recovery_weighted_error",
            "fixed_source_l2_fraction",
            "forcing_closure_max_abs",
            "source_free_response_fraction_of_total",
            "total_response_reproduction_max_abs_wb",
        ):
            _number(row[name], field=name)
        for partition_name in ("forcing_partition", "response_partition"):
            partition = row[partition_name]
            closure_name = "closure_max_abs_wb" if partition_name == "response_partition" else None
            required = {"source", "source_free", "total"}
            if closure_name is not None:
                required.add(closure_name)
            if not isinstance(partition, dict) or set(partition) != required:
                raise ValueError(f"{partition_name} fields are invalid")
            if closure_name is not None:
                _number(partition[closure_name], field=closure_name)
            for name in ("source", "source_free", "total"):
                _validate_field_metric(partition[name], field=f"{partition_name}.{name}")
        response = row["response_partition"]
        source_free_l2 = float(response["source_free"]["area_weighted_l2"])
        total_l2 = float(response["total"]["area_weighted_l2"])
        if total_l2 <= 0.0:
            raise ValueError("response_partition.total.area_weighted_l2 must be positive")
        derived_fraction = source_free_l2 / total_l2
        if float(row["source_free_response_fraction_of_total"]) != derived_fraction:
            raise ValueError("source-free response fraction does not match response metrics")
        timings = row["inverse_timings_ms"]
        if not isinstance(timings, dict) or set(timings) != {
            "inverse_source",
            "inverse_source_free",
            "inverse_total",
        }:
            raise ValueError("inverse timing fields are invalid")
        for name, timing in timings.items():
            _number(timing, field=f"inverse_timings_ms.{name}")
        rows.append(row)
    if [row["resolution"] for row in rows] != list(grid_contract.GRID_RESOLUTIONS):
        raise ValueError("CVGC2 grid resolutions are not the frozen ladder")
    return rows


def _validate_convergence(value: object) -> dict[str, Any]:
    """Validate pairwise metrics and both observed-order surfaces."""
    if not isinstance(value, dict) or set(value) != {
        "pairwise",
        "source_free_forcing_order",
        "source_free_response_order",
    }:
        raise ValueError("CVGC2 convergence fields are invalid")
    pairwise = value["pairwise"]
    if not isinstance(pairwise, dict) or set(pairwise) != {"33_65", "65_129", "129_257"}:
        raise ValueError("CVGC2 pairwise ladder is invalid")
    for pair in pairwise.values():
        if not isinstance(pair, dict) or set(pair) != {
            "fixed_physical_source_free",
            "full_interior",
            "plasma_support",
        }:
            raise ValueError("CVGC2 pairwise regions are invalid")
        for region in pair.values():
            if not isinstance(region, dict) or set(region) != {"forcing", "response"}:
                raise ValueError("CVGC2 pairwise fields are invalid")
            for metric in region.values():
                if not isinstance(metric, dict) or set(metric) != {
                    "area_weighted_l2",
                    "area_weighted_rms",
                    "cosine",
                    "linf",
                    "projection",
                    "relative_l2",
                }:
                    raise ValueError("CVGC2 comparison metric fields are invalid")
                for name, number in metric.items():
                    _number(number, field=f"comparison.{name}", minimum=-math.inf)
    for order_name in ("source_free_forcing_order", "source_free_response_order"):
        orders = value[order_name]
        if not isinstance(orders, dict) or set(orders) != set(GRID_TRIPLES):
            raise ValueError(f"{order_name} triples are invalid")
        for order in orders.values():
            if not isinstance(order, dict) or set(order) != {
                "coarse_to_medium_rms",
                "medium_to_fine_rms",
                "observed_order",
            }:
                raise ValueError(f"{order_name} fields are invalid")
            for name, number in order.items():
                _number(number, field=f"{order_name}.{name}")
    return cast(dict[str, Any], value)


def _numerical_gates(
    grids: list[dict[str, Any]],
    convergence: dict[str, Any],
) -> dict[str, bool]:
    """Evaluate only thresholds frozen in the execution plan."""
    by_resolution = {int(row["resolution"]): row for row in grids}
    recovery = {
        resolution: float(by_resolution[resolution]["current_recovery_weighted_error"])
        for resolution in grid_contract.GRID_RESOLUTIONS
    }
    return {
        "current_recovery_fine": (
            recovery[129] <= CURRENT_RECOVERY_WEIGHTED_MAX
            and recovery[257] <= CURRENT_RECOVERY_WEIGHTED_MAX
        ),
        "current_recovery_non_increasing": all(
            recovery[right] <= recovery[left] * (1.0 + CURRENT_TREND_SLACK) + 1.0e-15
            for left, right in ((65, 129), (129, 257))
        ),
        "fixed_source_localisation": all(
            float(row["fixed_source_l2_fraction"]) >= SOURCE_LOCALISATION_L2_FRACTION_MIN
            for row in grids
        ),
        "source_free_forcing_order": all(
            float(convergence["source_free_forcing_order"][name]["observed_order"])
            >= SOURCE_FREE_ORDER_MIN
            for name in GRID_TRIPLES
        ),
        "source_free_response_fine_amplitude": all(
            float(by_resolution[resolution]["source_free_response_fraction_of_total"])
            <= RESPONSE_FRACTION_OF_TOTAL_MAX
            for resolution in (129, 257)
        ),
        "source_free_response_order": all(
            float(convergence["source_free_response_order"][name]["observed_order"])
            >= SOURCE_FREE_ORDER_MIN
            for name in GRID_TRIPLES
        ),
    }


def _routing(structural_pass: bool, numerical: dict[str, bool]) -> dict[str, Any]:
    """Route source and vacuum results without changing admission claims."""
    source_pass = all(
        numerical[name]
        for name in (
            "current_recovery_fine",
            "current_recovery_non_increasing",
            "fixed_source_localisation",
        )
    )
    vacuum_pass = all(
        numerical[name]
        for name in (
            "source_free_forcing_order",
            "source_free_response_fine_amplitude",
            "source_free_response_order",
        )
    )
    if not structural_pass:
        state = "blocked_upstream_or_partition_drift"
    elif source_pass and vacuum_pass:
        state = "fixed_physical_source_and_vacuum_numerics_resolved"
    elif source_pass:
        state = "fixed_physical_vacuum_numerics_unresolved"
    elif vacuum_pass:
        state = "fixed_physical_source_numerics_unresolved"
    else:
        state = "fixed_physical_source_and_vacuum_numerics_unresolved"
    return {
        "production_solver_physics_changed": False,
        "source_numerics_pass": source_pass,
        "state": state,
        "vacuum_numerics_pass": vacuum_pass,
    }


def build_report(
    *,
    generated_at: str,
    environment: dict[str, Any],
    execution_binding: dict[str, str],
    source_artifacts: dict[str, dict[str, Any]],
    upstream: dict[str, str],
    grids: list[dict[str, Any]],
    convergence: dict[str, Any],
) -> dict[str, Any]:
    """Build and self-validate one complete CVGC2 evidence payload."""
    checked_grids = _validate_grids(grids)
    checked_convergence = _validate_convergence(convergence)
    closures = all(
        float(row["forcing_closure_max_abs"]) <= PARTITION_CLOSURE_MAX_ABS
        and float(row["response_partition"]["closure_max_abs_wb"]) <= PARTITION_CLOSURE_MAX_ABS
        and float(row["total_response_reproduction_max_abs_wb"]) <= PARTITION_CLOSURE_MAX_ABS
        for row in checked_grids
    )
    numerical = _numerical_gates(checked_grids, checked_convergence)
    structural = {
        "four_grid_ladder_complete": True,
        "partition_and_total_response_closure": closures,
        "production_solver_unchanged": True,
        "upstream_cvgc1_exact": upstream.get("payload_sha256") == EXPECTED_UPSTREAM_PAYLOAD,
    }
    routing = _routing(all(structural.values()), numerical)
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "blockers": sorted(BASE_BLOCKERS),
        "claim_boundary": {field: False for field in CLAIM_FIELDS},
        "convergence": convergence,
        "environment": environment,
        "execution_binding": execution_binding,
        "gates": {"numerical": numerical, "structural": structural},
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
            else "blocked_incomplete_or_drifted"
        ),
        "upstream_binding": upstream,
    }
    report["payload_sha256"] = _payload_sha256(report)
    validate_report(report)
    return report


def validate_report(report: dict[str, Any]) -> None:
    """Reject drifted, incomplete, non-finite, or overclaiming CVGC2 evidence."""
    if set(report) != _TOP_LEVEL_FIELDS:
        raise ValueError("CVGC2 top-level fields are invalid")
    if report["schema_version"] != SCHEMA_VERSION or report["benchmark_id"] != BENCHMARK_ID:
        raise ValueError("CVGC2 identity is invalid")
    if (
        not isinstance(report["generated_at"], str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", report["generated_at"]) is None
    ):
        raise ValueError("CVGC2 generated_at is invalid")
    if report["input_contract"] != _input_contract() or report["hypotheses"] != _hypotheses():
        raise ValueError("CVGC2 frozen design is invalid")
    if report["blockers"] != sorted(BASE_BLOCKERS):
        raise ValueError("CVGC2 blockers are invalid")
    if report["claim_boundary"] != {field: False for field in CLAIM_FIELDS}:
        raise ValueError("CVGC2 claim boundary must remain false")
    upstream = report["upstream_binding"]
    if not isinstance(upstream, dict) or set(upstream) != {
        "file_sha256",
        "path",
        "payload_sha256",
        "schema_version",
    }:
        raise ValueError("CVGC2 upstream binding is invalid")
    if (
        upstream["path"] != UPSTREAM_PATH
        or upstream["payload_sha256"] != EXPECTED_UPSTREAM_PAYLOAD
        or upstream["schema_version"] != grid_contract.SCHEMA_VERSION
        or _SHA256_RE.fullmatch(str(upstream["file_sha256"])) is None
    ):
        raise ValueError("CVGC2 upstream binding drifted")
    execution = report["execution_binding"]
    _validate_source_artifacts(report["source_artifacts"])
    expected_execution = {
        "anchor_sha256": EXPECTED_ANCHOR_SHA256,
        "coil_manifest_sha256": EXPECTED_COIL_MANIFEST_SHA256,
        "source_artifacts_sha256": canonical_sha256(report["source_artifacts"]),
    }
    if not isinstance(execution, dict) or execution != expected_execution:
        raise ValueError("CVGC2 execution binding drifted")
    if not isinstance(report["environment"], dict) or not report["environment"]:
        raise ValueError("CVGC2 environment is invalid")
    grids = _validate_grids(report["grids"])
    convergence = _validate_convergence(report["convergence"])
    structural = cast(dict[str, bool], report["gates"]["structural"])
    numerical = cast(dict[str, bool], report["gates"]["numerical"])
    expected_numerical = _numerical_gates(grids, convergence)
    closures = all(
        float(row["forcing_closure_max_abs"]) <= PARTITION_CLOSURE_MAX_ABS
        and float(row["response_partition"]["closure_max_abs_wb"]) <= PARTITION_CLOSURE_MAX_ABS
        and float(row["total_response_reproduction_max_abs_wb"]) <= PARTITION_CLOSURE_MAX_ABS
        for row in grids
    )
    expected_structural = {
        "four_grid_ladder_complete": True,
        "partition_and_total_response_closure": closures,
        "production_solver_unchanged": True,
        "upstream_cvgc1_exact": True,
    }
    if structural != expected_structural or numerical != expected_numerical:
        raise ValueError("CVGC2 gates do not match measured evidence")
    expected_routing = _routing(all(structural.values()), numerical)
    if report["routing"] != expected_routing:
        raise ValueError("CVGC2 routing does not match measured gates")
    expected_status = (
        "diagnostic_complete_claims_blocked"
        if all(structural.values())
        else "blocked_incomplete_or_drifted"
    )
    if report["status"] != expected_status:
        raise ValueError("CVGC2 status is inconsistent")
    if report["payload_sha256"] != _payload_sha256(report):
        raise ValueError("CVGC2 payload digest is invalid")
    _walk_finite(report)


def render_markdown(report: dict[str, Any]) -> str:
    """Render the validated CVGC2 evidence summary."""
    validate_report(report)
    lines = [
        "# DIII-D / IDA fixed-physical coil-vacuum response",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Payload: `{report['payload_sha256']}`",
        f"- Upstream CVGC1: `{report['upstream_binding']['payload_sha256']}`",
        f"- Routing: `{report['routing']['state']}`",
        "- Production solver physics changed: `false`",
        "- Scientific, facility, control, safety, PCS and held-out claims: `false`",
        "",
        "| Grid | fixed-source L2 fraction | fixed current error | source-free response / total | response closure [Wb] |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["grids"]:
        lines.append(
            f"| {row['resolution']} | {row['fixed_source_l2_fraction']:.8g} | "
            f"{row['current_recovery_weighted_error']:.8g} | "
            f"{row['source_free_response_fraction_of_total']:.8g} | "
            f"{row['response_partition']['closure_max_abs_wb']:.8g} |"
        )
    lines.extend(["", "## Observed orders", ""])
    for name in GRID_TRIPLES:
        forcing = report["convergence"]["source_free_forcing_order"][name]["observed_order"]
        response = report["convergence"]["source_free_response_order"][name]["observed_order"]
        lines.append(f"- `{name}` forcing `{forcing:.8g}`, response `{response:.8g}`")
    lines.extend(["", "## Admission boundary", ""])
    for field in CLAIM_FIELDS:
        lines.append(f"- {field.replace('_', ' ')}: `false`")
    lines.extend(
        ["", "CVGC1 remains immutable; its failed relative-response gate is not rewritten.", ""]
    )
    return "\n".join(lines)
