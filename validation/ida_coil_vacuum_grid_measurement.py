# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Measurement Validation
"""Measured-grid and nested-convergence validators for CVGC1 evidence."""

from __future__ import annotations

import math
from typing import Any, cast

import validation.ida_coil_vacuum_grid_contract as contract


def _validate_field_metric(value: object, *, field: str) -> None:
    """Validate one deterministic field-norm record."""
    if not isinstance(value, dict) or set(value) != contract.FIELD_METRIC_FIELDS:
        raise ValueError(f"{field} fields are invalid")
    for name in contract.FIELD_METRIC_FIELDS - {"field_sha256"}:
        contract._require_number(value[name], field=f"{field}.{name}", minimum=0.0)
    contract._require_sha256(value["field_sha256"], field=f"{field}.field_sha256")


def _validate_comparison_metric(value: object, *, field: str) -> None:
    """Validate one exact-restriction comparison record."""
    if not isinstance(value, dict) or set(value) != contract.COMPARISON_METRIC_FIELDS:
        raise ValueError(f"{field} fields are invalid")
    for name in ("area_weighted_l2", "area_weighted_rms", "linf", "relative_l2"):
        contract._require_number(value[name], field=f"{field}.{name}", minimum=0.0)
    contract._require_number(
        value["cosine"],
        field=f"{field}.cosine",
        minimum=-1.0 - 1.0e-12,
        maximum=1.0 + 1.0e-12,
    )
    contract._require_number(value["projection"], field=f"{field}.projection")


def _validate_mask_summary(
    value: object,
    *,
    field: str,
    cell_area_m2: float,
) -> int:
    """Validate one point-count/physical-area mask summary."""
    if not isinstance(value, dict) or set(value) != {"area_m2", "point_count", "point_fraction"}:
        raise ValueError(f"{field} fields are invalid")
    count = value["point_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(f"{field}.point_count is invalid")
    area = contract._require_number(value["area_m2"], field=f"{field}.area_m2", minimum=0.0)
    fraction = contract._require_number(
        value["point_fraction"],
        field=f"{field}.point_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(area, count * cell_area_m2, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError(f"{field}.area_m2 is inconsistent")
    if count == 0 and fraction != 0.0:
        raise ValueError(f"{field}.point_fraction is inconsistent")
    return count


def _validate_recovery_metric(value: object, *, field: str) -> None:
    """Validate one signed current-recovery record and its derived errors."""
    if not isinstance(value, dict) or set(value) != contract.RECOVERY_METRIC_FIELDS:
        raise ValueError(f"{field} fields are invalid")
    expected = contract._require_number(
        value["expected_a_turns"],
        field=f"{field}.expected_a_turns",
    )
    recovered = contract._require_number(
        value["recovered_a_turns"],
        field=f"{field}.recovered_a_turns",
    )
    absolute = contract._require_number(
        value["absolute_error_a_turns"],
        field=f"{field}.absolute_error_a_turns",
        minimum=0.0,
    )
    relative = contract._require_number(
        value["relative_error"],
        field=f"{field}.relative_error",
        minimum=0.0,
    )
    signed = contract._require_number(
        value["signed_error_a_turns"],
        field=f"{field}.signed_error_a_turns",
    )
    if not math.isclose(absolute, abs(recovered - expected), rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise ValueError(f"{field}.absolute_error_a_turns is inconsistent")
    if not math.isclose(signed, recovered - expected, rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise ValueError(f"{field}.signed_error_a_turns is inconsistent")
    expected_relative = absolute / max(abs(expected), 1.0e-30)
    if not math.isclose(relative, expected_relative, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError(f"{field}.relative_error is inconsistent")


def _validate_grid_row(
    row: dict[str, Any],
    *,
    manifest_parents: list[dict[str, Any]],
) -> None:
    """Validate every measured surface for one required grid."""
    parent_names = [str(parent["name"]) for parent in manifest_parents]
    parent_effective = {
        str(parent["name"]): float(parent["effective_current_a_turns"])
        for parent in manifest_parents
    }
    raw_resolution = row["resolution"]
    if not isinstance(raw_resolution, int) or isinstance(raw_resolution, bool):
        raise ValueError("grid resolution must be an integer")
    resolution = raw_resolution
    grid = row["grid"]
    expected_grid_fields = {
        "cell_area_m2",
        "d_r_m",
        "d_z_m",
        "filament_phase_sha256",
        "minimum_filament_to_node_distance_m",
        "nearest_distance_max_m",
        "nearest_distance_mean_m",
        "r_bounds_m",
        "z_bounds_m",
    }
    if not isinstance(grid, dict) or set(grid) != expected_grid_fields:
        raise ValueError("grid geometry fields are invalid")
    if grid["r_bounds_m"] != list(contract.R_BOUNDS_M) or grid["z_bounds_m"] != list(
        contract.Z_BOUNDS_M
    ):
        raise ValueError("grid physical domain is invalid")
    d_r = (contract.R_BOUNDS_M[1] - contract.R_BOUNDS_M[0]) / (resolution - 1)
    d_z = (contract.Z_BOUNDS_M[1] - contract.Z_BOUNDS_M[0]) / (resolution - 1)
    measured_d_r = contract._require_number(grid["d_r_m"], field="grid.d_r_m", minimum=0.0)
    measured_d_z = contract._require_number(grid["d_z_m"], field="grid.d_z_m", minimum=0.0)
    if not math.isclose(measured_d_r, d_r, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("grid d_r_m is invalid")
    if not math.isclose(measured_d_z, d_z, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("grid d_z_m is invalid")
    cell_area = contract._require_number(
        grid["cell_area_m2"],
        field="grid.cell_area_m2",
        minimum=0.0,
    )
    if not math.isclose(cell_area, d_r * d_z, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("grid cell_area_m2 is invalid")
    contract._require_sha256(
        grid["filament_phase_sha256"],
        field="grid.filament_phase_sha256",
    )
    minimum_distance = contract._require_number(
        grid["minimum_filament_to_node_distance_m"],
        field="grid.minimum_filament_to_node_distance_m",
        minimum=0.0,
    )
    mean_distance = contract._require_number(
        grid["nearest_distance_mean_m"],
        field="grid.nearest_distance_mean_m",
        minimum=minimum_distance,
    )
    contract._require_number(
        grid["nearest_distance_max_m"],
        field="grid.nearest_distance_max_m",
        minimum=mean_distance,
    )

    masks = row["masks"]
    expected_mask_fields = {
        "fixed_physical",
        "fixed_physical_radius_m",
        "plasma_support",
        "primary_fixed_overlap_point_count",
        "rho_h_le_1",
        "rho_h_le_2",
        "rho_h_le_4",
    }
    if not isinstance(masks, dict) or set(masks) != expected_mask_fields:
        raise ValueError("grid mask fields are invalid")
    if not math.isclose(
        contract._require_number(
            masks["fixed_physical_radius_m"],
            field="masks.fixed_physical_radius_m",
            minimum=0.0,
        ),
        contract.FIXED_PHYSICAL_RADIUS_M,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("fixed physical radius is invalid")
    mask_counts = {
        name: _validate_mask_summary(
            masks[name],
            field=f"masks.{name}",
            cell_area_m2=cell_area,
        )
        for name in ("fixed_physical", "plasma_support", "rho_h_le_1", "rho_h_le_2", "rho_h_le_4")
    }
    if not (
        0 < mask_counts["rho_h_le_1"] <= mask_counts["rho_h_le_2"] <= mask_counts["rho_h_le_4"]
    ):
        raise ValueError("grid-relative mask nesting is invalid")
    for name, count in mask_counts.items():
        expected_fraction = count / (resolution * resolution)
        if not math.isclose(
            float(masks[name]["point_fraction"]),
            expected_fraction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"masks.{name}.point_fraction is inconsistent")
    overlap = masks["primary_fixed_overlap_point_count"]
    if (
        not isinstance(overlap, int)
        or isinstance(overlap, bool)
        or overlap < 0
        or overlap > min(mask_counts["rho_h_le_2"], mask_counts["fixed_physical"])
    ):
        raise ValueError("primary/fixed mask overlap is invalid")
    if mask_counts["plasma_support"] <= 0:
        raise ValueError("plasma support mask must not be empty")

    forcing = row["forcing_partition"]
    if not isinstance(forcing, dict) or set(forcing) != {
        "closure_max_abs",
        "primary_l2_fraction",
        "sensitivity_l2_fraction",
        "source",
        "source_free",
        "total",
    }:
        raise ValueError("forcing partition fields are invalid")
    contract._require_number(forcing["closure_max_abs"], field="forcing closure", minimum=0.0)
    primary_fraction = contract._require_number(
        forcing["primary_l2_fraction"],
        field="forcing primary_l2_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    sensitivity = forcing["sensitivity_l2_fraction"]
    if not isinstance(sensitivity, dict) or set(sensitivity) != {
        "rho_h_le_1",
        "rho_h_le_2",
        "rho_h_le_4",
    }:
        raise ValueError("forcing sensitivity fields are invalid")
    fractions = [
        contract._require_number(
            sensitivity[name],
            field=f"forcing sensitivity {name}",
            minimum=0.0,
            maximum=1.0,
        )
        for name in ("rho_h_le_1", "rho_h_le_2", "rho_h_le_4")
    ]
    if not fractions[0] <= fractions[1] <= fractions[2] or not math.isclose(
        primary_fraction,
        fractions[1],
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("forcing sensitivity nesting is invalid")
    for name in ("source", "source_free", "total"):
        _validate_field_metric(forcing[name], field=f"forcing_partition.{name}")

    response = row["response_partition"]
    if not isinstance(response, dict) or set(response) != {
        "closure_max_abs_wb",
        "source",
        "source_free",
        "total",
    }:
        raise ValueError("response partition fields are invalid")
    contract._require_number(
        response["closure_max_abs_wb"],
        field="response closure",
        minimum=0.0,
    )
    for name in ("source", "source_free", "total"):
        _validate_field_metric(response[name], field=f"response_partition.{name}")

    vacuum = row["vacuum_fields"]
    if not isinstance(vacuum, dict) or set(vacuum) != {
        "freegs",
        "native",
        "source_free_parity",
    }:
        raise ValueError("vacuum field fields are invalid")
    _validate_field_metric(vacuum["freegs"], field="vacuum_fields.freegs")
    _validate_field_metric(vacuum["native"], field="vacuum_fields.native")
    parity = vacuum["source_free_parity"]
    if not isinstance(parity, dict) or set(parity) != {
        "max_abs_wb",
        "nrmse",
        "passes",
        "point_count",
    }:
        raise ValueError("source-free parity fields are invalid")
    if (
        not isinstance(parity["point_count"], int)
        or isinstance(parity["point_count"], bool)
        or parity["point_count"] <= 0
    ):
        raise ValueError("source-free parity point_count is invalid")
    contract._require_number(
        parity["max_abs_wb"],
        field="source-free parity max_abs_wb",
        minimum=0.0,
    )
    contract._require_number(
        parity["nrmse"],
        field="source-free parity nrmse",
        minimum=0.0,
    )
    if parity["passes"] is not True:
        raise ValueError("source-free Green-function parity passes must be true")

    recovery = row["current_recovery"]
    if not isinstance(recovery, dict) or set(recovery) != {
        "aggregate",
        "parents",
        "weighted_fixed_physical_error",
        "weighted_primary_error",
    }:
        raise ValueError("current recovery fields are invalid")
    parents = recovery["parents"]
    if (
        not isinstance(parents, list)
        or any(not isinstance(item, dict) for item in parents)
        or [item.get("name") for item in parents] != parent_names
    ):
        raise ValueError("current recovery parent rows are invalid")
    primary_metrics: list[dict[str, float]] = []
    fixed_metrics: list[dict[str, float]] = []
    for parent in parents:
        if not isinstance(parent, dict) or set(parent) != {"name", "regions"}:
            raise ValueError("current recovery parent row fields are invalid")
        regions = parent["regions"]
        if not isinstance(regions, dict) or set(regions) != {
            "fixed_physical",
            "full_interior",
            "primary",
            "source_free",
        }:
            raise ValueError("current recovery parent regions are invalid")
        for name, metric in regions.items():
            _validate_recovery_metric(metric, field=f"current_recovery.{parent['name']}.{name}")
            if not math.isclose(
                float(metric["expected_a_turns"]),
                parent_effective[str(parent["name"])],
                rel_tol=1.0e-12,
                abs_tol=1.0e-9,
            ):
                raise ValueError("current recovery expected current disagrees with manifest")
        primary_metrics.append(cast(dict[str, float], regions["primary"]))
        fixed_metrics.append(cast(dict[str, float], regions["fixed_physical"]))
    aggregate = recovery["aggregate"]
    expected_aggregate_fields = {
        "fixed_physical_a_turns",
        "full_interior_a_turns",
        "primary_a_turns",
        "source_free_a_turns",
        "target_absolute_sum_a_turns",
        "target_net_a_turns",
    }
    if not isinstance(aggregate, dict) or set(aggregate) != expected_aggregate_fields:
        raise ValueError("current recovery aggregate fields are invalid")
    for name in expected_aggregate_fields:
        contract._require_number(
            aggregate[name],
            field=f"current_recovery.aggregate.{name}",
        )
    target_net = math.fsum(parent_effective.values())
    target_absolute = math.fsum(abs(value) for value in parent_effective.values())
    if not math.isclose(
        float(aggregate["target_net_a_turns"]),
        target_net,
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ) or not math.isclose(
        float(aggregate["target_absolute_sum_a_turns"]),
        target_absolute,
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise ValueError("current recovery aggregate target disagrees with manifest")
    expected_weighted = {
        "weighted_fixed_physical_error": (
            math.fsum(metric["absolute_error_a_turns"] for metric in fixed_metrics)
            / target_absolute
        ),
        "weighted_primary_error": (
            math.fsum(metric["absolute_error_a_turns"] for metric in primary_metrics)
            / target_absolute
        ),
    }
    for name, expected in expected_weighted.items():
        measured = contract._require_number(
            recovery[name],
            field=f"current_recovery.{name}",
            minimum=0.0,
        )
        if not math.isclose(measured, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError(f"current_recovery.{name} is inconsistent")

    timings = row["timings_ms"]
    expected_timings = {
        "freegs_field",
        "inverse_source",
        "inverse_source_free",
        "inverse_total",
        "native_field",
    }
    if not isinstance(timings, dict) or set(timings) != expected_timings:
        raise ValueError("grid timing fields are invalid")
    for name, measured in timings.items():
        contract._require_number(measured, field=f"timings_ms.{name}", minimum=0.0)


def validate_grids(
    value: object,
    *,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate the exact ordered four-grid measurement ladder."""
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("grids must contain object rows")
    rows = cast(list[dict[str, Any]], value)
    if [row.get("resolution") for row in rows] != list(contract.GRID_RESOLUTIONS):
        raise ValueError("grids must contain the exact required ladder")
    required = {
        "current_recovery",
        "forcing_partition",
        "grid",
        "masks",
        "resolution",
        "response_partition",
        "timings_ms",
        "vacuum_fields",
    }
    manifest_parents = cast(list[dict[str, Any]], manifest["parents"])
    for row in rows:
        if set(row) != required:
            raise ValueError("grid row fields are invalid")
        _validate_grid_row(row, manifest_parents=manifest_parents)
        parity = row["vacuum_fields"]["source_free_parity"]
        if (
            float(parity["nrmse"]) > contract.PARITY_NRMSE_MAX
            or float(parity["max_abs_wb"]) > contract.PARITY_MAX_ABS_WB
            or parity["passes"] is not True
        ):
            raise ValueError("source-free Green-function parity is open")
        if (
            float(row["forcing_partition"]["closure_max_abs"]) > contract.PARTITION_CLOSURE_MAX_ABS
            or float(row["response_partition"]["closure_max_abs_wb"])
            > contract.PARTITION_CLOSURE_MAX_ABS
        ):
            raise ValueError("forcing/response partition closure is open")
    return rows


def validate_convergence(value: object) -> dict[str, Any]:
    """Validate exact nested-grid regional comparisons and order metrics."""
    if not isinstance(value, dict) or set(value) != {
        "finest_source_free_response",
        "pairwise",
        "source_free_forcing_order",
    }:
        raise ValueError("convergence fields are invalid")
    pairwise = value["pairwise"]
    if not isinstance(pairwise, dict) or set(pairwise) != {"33_65", "65_129", "129_257"}:
        raise ValueError("convergence pairwise fields are invalid")
    required_regions = {
        "fixed_physical_source_free",
        "full",
        "plasma_support",
        "source_footprint",
    }
    for pair_name, pair in pairwise.items():
        if not isinstance(pair, dict) or set(pair) != required_regions:
            raise ValueError(f"convergence pairwise.{pair_name} regions are invalid")
        for region_name, region in pair.items():
            if not isinstance(region, dict) or set(region) != {"forcing", "response"}:
                raise ValueError(
                    f"convergence pairwise.{pair_name}.{region_name} fields are invalid"
                )
            for surface, metric in region.items():
                _validate_comparison_metric(
                    metric,
                    field=f"convergence.{pair_name}.{region_name}.{surface}",
                )
    orders = value["source_free_forcing_order"]
    if not isinstance(orders, dict) or set(orders) != set(contract.GRID_TRIPLES):
        raise ValueError("source-free forcing order fields are invalid")
    for name, row in orders.items():
        if not isinstance(row, dict) or set(row) != {
            "coarse_to_medium_rms",
            "medium_to_fine_rms",
            "observed_order",
        }:
            raise ValueError(f"source-free forcing order {name} fields are invalid")
        contract._require_number(
            row["coarse_to_medium_rms"],
            field=f"{name}.coarse_rms",
            minimum=0.0,
        )
        contract._require_number(
            row["medium_to_fine_rms"],
            field=f"{name}.fine_rms",
            minimum=0.0,
        )
        if row["coarse_to_medium_rms"] <= 0.0 or row["medium_to_fine_rms"] <= 0.0:
            raise ValueError(f"source-free forcing order {name} differences must be non-zero")
        contract._require_number(row["observed_order"], field=f"{name}.observed_order")
    _validate_comparison_metric(
        value["finest_source_free_response"],
        field="finest_source_free_response",
    )
    return value
