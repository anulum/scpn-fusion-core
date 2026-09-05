# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Threshold and grid evidence diagnostics
"""Validate stored threshold/grid summaries without granting scientific admission."""

from __future__ import annotations

import math
from typing import Any, TypeGuard


# Reduced poloidal flux: Phi = 2*pi*psi; preserve the producer's psi-unit limit.
NUMERIC_THRESHOLDS = {
    "psi_n_rmse": ("<=", 0.05),
    "axis_error_m": ("<=", 0.025),
    "current_closure_relative_error": ("<=", 0.05),
    "boundary_max_abs_error_wb": ("<=", 1e-10),
    "xpoint_psi_n_error_max": ("<=", 0.05),
    "boundary_containment_fraction": (">=", 1.0),
}
GRID_RESOLUTIONS = (33, 65, 129)
GRID_METRICS = (
    "psi_n_rmse",
    "native_plasma_psi_rmse",
    "xpoint_psi_n_error_max",
    "current_closure_relative_error",
)


def _finite_number(value: object) -> TypeGuard[int | float]:
    """Reject booleans and nonfinite or unrepresentable JSON numbers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _diagnostic_thresholds(case: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Recompute summary diagnostics, without certifying underlying measurements."""
    checks = case.get("threshold_checks", [])
    if not isinstance(checks, list) or any(not isinstance(row, dict) for row in checks):
        raise ValueError("threshold_checks must be a list of objects")
    by_metric: dict[str, dict[str, Any]] = {}
    for check in checks:
        metric = check.get("metric")
        if not isinstance(metric, str) or metric in by_metric:
            raise ValueError("threshold_checks require unique metric names")
        if metric not in NUMERIC_THRESHOLDS and metric != "q_profile_sanity_status":
            raise ValueError(f"unsupported threshold metric: {metric}")
        by_metric[metric] = check
    failed: list[dict[str, Any]] = []
    contradictions: list[str] = []
    for metric, (comparator, limit) in NUMERIC_THRESHOLDS.items():
        check = by_metric.get(metric, {})
        value = check.get("value")
        valid = (
            _finite_number(value)
            and value >= 0
            and _finite_number(check.get("limit"))
            and check["limit"] == limit
            and check.get("comparator") == comparator
        )
        passed = (
            valid
            and _finite_number(value)
            and (value <= limit if comparator == "<=" else value == limit)
        )
        if check.get("valid") is not valid or check.get("passed") is not passed:
            contradictions.append(metric)
        if not passed:
            failed.append(
                {
                    "metric": metric,
                    "value": value if _finite_number(value) else None,
                    "limit": limit,
                    "comparator": comparator,
                }
            )
    q_check = by_metric.get("q_profile_sanity_status", {})
    q_label_matches = (
        q_check.get("value") == "pass_finite_signed_q_profile"
        and q_check.get("limit") == "pass_finite_signed_q_profile"
        and q_check.get("comparator") == "=="
    )
    if q_check.get("passed") is not q_label_matches:
        contradictions.append("q_profile_sanity_status")
    if not q_label_matches:
        failed.append(
            {
                "metric": "q_profile_sanity_status",
                "value": q_check.get("value"),
                "limit": "pass_finite_signed_q_profile",
                "comparator": "==",
            }
        )
    return failed, contradictions


def _threshold_case_rows(strict: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate case structure and separate diagnostics from scientific admission."""
    rows: list[dict[str, Any]] = []
    cases = strict.get("cases", [])
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("cases must be a list of objects")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("cases require unique nonempty case_id values")
        seen.add(case_id)
        failed_checks, contradictions = _diagnostic_thresholds(case)
        readiness = {
            key: case.get(key) is True
            for key in (
                "external_nonlinear_output_ready",
                "native_same_case_profile_source_ready",
                "strict_threshold_acceptance_ready",
            )
        }
        rows.append(
            {
                "case_id": case_id,
                **dict.fromkeys(readiness, False),
                "reported_readiness": readiness,
                "contradictory_threshold_metrics": contradictions,
                "unverified_threshold_metrics": ["q_profile_sanity_status"],
                "failed_threshold_check_count": len(failed_checks),
                "failed_threshold_checks": [
                    {
                        "metric": str(check.get("metric", "")),
                        "value": check.get("value"),
                        "limit": check.get("limit"),
                        "comparator": str(check.get("comparator", "")),
                    }
                    for check in failed_checks
                ],
            }
        )
    return rows


def _grid_measurements(case: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Index actual ladder rows, rejecting ambiguous or unsupported grids."""
    resolution_rows = case.get("resolution_rows", [])
    if not isinstance(resolution_rows, list) or any(
        not isinstance(row, dict) for row in resolution_rows
    ):
        raise ValueError("resolution_rows must be a list of objects")
    by_resolution: dict[int, dict[str, Any]] = {}
    for row in resolution_rows:
        dimensions = row.get("grid")
        if not isinstance(dimensions, dict):
            raise ValueError("grid must contain integer nx and ny")
        nx, ny = dimensions.get("nx"), dimensions.get("ny")
        if (
            type(nx) is not int
            or type(ny) is not int
            or nx != ny
            or nx not in GRID_RESOLUTIONS
            or nx in by_resolution
        ):
            raise ValueError("grid ladder requires unique square grids of 33/65/129")
        by_resolution[nx] = row
    return by_resolution


def _resolution_diagnostics(resolution: int, row: dict[str, Any]) -> dict[str, Any]:
    """Compare per-grid threshold verdicts with their duplicated metric values."""
    failed, contradictions = _diagnostic_thresholds(row)
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    checks = {check["metric"]: check for check in row.get("threshold_checks", [])}
    inconsistent: list[str] = []
    for metric in GRID_METRICS:
        if metric not in NUMERIC_THRESHOLDS:
            continue
        summary_value = metrics.get(metric)
        threshold_value = checks.get(metric, {}).get("value")
        if (
            not _finite_number(summary_value)
            or not _finite_number(threshold_value)
            or summary_value != threshold_value
        ):
            inconsistent.append(metric)
    return {
        "resolution": resolution,
        "failed_threshold_check_count": len(failed),
        "failed_threshold_checks": failed,
        "contradictory_threshold_metrics": contradictions,
        "inconsistent_metric_values": inconsistent,
        "reported_failed_count_matches": type(row.get("failed_threshold_check_count")) is int
        and row["failed_threshold_check_count"] == len(failed),
        "unverified_threshold_metrics": ["q_profile_sanity_status"],
    }


def _grid_diagnostics(case: dict[str, Any]) -> dict[str, Any]:
    """Recompute ladder counts, threshold consistency and relative monotonicity."""
    by_resolution = _grid_measurements(case)
    resolutions = sorted(by_resolution)
    monotone: dict[str, bool] = {}
    for metric in GRID_METRICS:
        values: list[int | float] = []
        for resolution in resolutions:
            metrics = by_resolution[resolution].get("metrics")
            value = metrics.get(metric) if isinstance(metrics, dict) else None
            if _finite_number(value) and value >= 0:
                values.append(value)
        monotone[metric] = len(values) == len(GRID_RESOLUTIONS) and all(
            after <= before * (1.0 + 1e-9) for before, after in zip(values, values[1:])
        )
    derived: dict[str, Any] = {
        "observed_resolution_count": len(resolutions),
        "required_resolution_count": len(GRID_RESOLUTIONS),
        "missing_resolution_count": len(set(GRID_RESOLUTIONS) - set(resolutions)),
        "monotone_nonincreasing_metrics": monotone,
    }
    contradictions = [
        key
        for key, value in derived.items()
        if type(case.get(key)) is not type(value) or case.get(key) != value
    ]
    reported_monotone = case.get("monotone_nonincreasing_metrics")
    if (
        isinstance(reported_monotone, dict)
        and any(type(value) is not bool for value in reported_monotone.values())
        and "monotone_nonincreasing_metrics" not in contradictions
    ):
        contradictions.append("monotone_nonincreasing_metrics")
    expected_observed = [{"nx": resolution, "ny": resolution} for resolution in resolutions]
    if case.get("observed_resolutions") != expected_observed:
        contradictions.append("observed_resolutions")
    return {
        **derived,
        "observed_resolutions": resolutions,
        "resolution_diagnostics": [
            _resolution_diagnostics(resolution, by_resolution[resolution])
            for resolution in resolutions
        ],
        "contradictory_grid_fields": contradictions,
        "grid_convergence_case_ready": False,
        "reported_grid_convergence_case_ready": case.get("grid_convergence_case_ready") is True,
        "blocking_reason": "grid_summary_has_no_verified_field_custody",
    }


def _grid_case_rows(strict: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate case membership and derive non-admitting grid diagnostics."""
    grid = strict.get("grid_convergence_evidence", {})
    if not isinstance(grid, dict):
        raise ValueError("grid_convergence_evidence must be an object")
    rows: list[dict[str, Any]] = []
    cases = grid.get("cases", [])
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("grid cases must be a list of objects")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("grid cases require unique nonempty case_id values")
        seen.add(case_id)
        rows.append(
            {
                "case_id": case_id,
                "machine_class": str(case.get("machine_class", "")),
                **_grid_diagnostics(case),
            }
        )
    return rows


def evaluate_threshold_evidence(strict: dict[str, Any]) -> dict[str, Any]:
    """Derive case and resolution diagnostics from stored summary rows.

    Parameters
    ----------
    strict : dict
        Strict evidence section of a public FreeGS reconstruction report.

    Returns
    -------
    dict
        Threshold cases, grid cases and required resolution count. Readiness
        remains false without verified field/profile custody.

    Raises
    ------
    ValueError
        When case identities, metric rows or grid dimensions are malformed.
    """
    return {
        "threshold_cases": _threshold_case_rows(strict),
        "grid_cases": _grid_case_rows(strict),
        "required_resolution_count": len(GRID_RESOLUTIONS),
    }
