# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Fixed-Physical Coil-Vacuum Response
"""Fixed-physical forcing partitions and exact nested-grid response metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from validation import ida_coil_vacuum_grid_contract as grid_contract
from validation.ida_coil_vacuum_grid_convergence import GridResult
from validation.ida_coil_vacuum_grid_fields import (
    BoolArray,
    FloatArray,
    comparison_metric,
    field_metric,
    forcing_l2_fraction,
    observed_order_from_three,
    restrict_to_shape,
)
from validation.ida_coil_vacuum_grid_runtime import partition_inverse_response

PAIRWISE_REGIONS = (
    "fixed_physical_source_free",
    "full_interior",
    "plasma_support",
)


@dataclass(frozen=True)
class FixedPhysicalGridResult:
    """Metrics and arrays for one fixed-physical inverse partition.

    Attributes
    ----------
    resolution
        Number of nodes along each axis.
    report
        JSON-serialisable measurements for the grid.
    source_forcing_zr, source_free_forcing_zr
        Fixed-physical source and source-free forcing arrays.
    source_response_zr, source_free_response_zr
        Native inverse responses to the two forcing components.
    total_response_zr
        Native inverse response to the unchanged total forcing.
    interior_mask, fixed_source_free_mask, plasma_support_mask
        Frozen comparison regions in Z/R orientation.
    """

    resolution: int
    report: dict[str, Any]
    source_forcing_zr: FloatArray
    source_free_forcing_zr: FloatArray
    source_response_zr: FloatArray
    source_free_response_zr: FloatArray
    total_response_zr: FloatArray
    interior_mask: BoolArray
    fixed_source_free_mask: BoolArray
    plasma_support_mask: BoolArray


def _response_fraction(source_free_l2: float, total_l2: float) -> float:
    """Return source-free response amplitude relative to the total response."""
    if not np.isfinite(source_free_l2) or source_free_l2 < 0.0:
        raise ValueError("source-free response L2 must be finite and non-negative")
    if not np.isfinite(total_l2) or total_l2 <= 0.0:
        raise ValueError("total response L2 must be finite and positive")
    return source_free_l2 / total_l2


def build_fixed_physical_grid(grid: GridResult) -> FixedPhysicalGridResult:
    """Invert the CVGC1 total forcing using its fixed-physical source mask.

    Parameters
    ----------
    grid
        One bound CVGC1 grid result containing the unchanged total forcing,
        masks, plasma support, and total native response.

    Returns
    -------
    FixedPhysicalGridResult
        Fixed-partition arrays and metrics using the same native inverse.

    Raises
    ------
    ValueError
        If the grid is outside the frozen ladder, masks are inconsistent, or
        the recomputed total response drifts from CVGC1.
    """
    if grid.resolution not in grid_contract.GRID_RESOLUTIONS:
        raise ValueError("resolution must belong to the frozen four-grid ladder")
    shape = grid.total_forcing_zr.shape
    expected_shape = (grid.resolution, grid.resolution)
    if shape != expected_shape:
        raise ValueError("total forcing shape must match the declared resolution")
    for name, mask in {
        "interior": grid.interior_mask,
        "fixed source-free": grid.fixed_source_free_mask,
        "plasma support": grid.plasma_support_mask,
    }.items():
        if mask.shape != shape:
            raise ValueError(f"{name} mask shape must match total forcing")
    if np.any(grid.fixed_source_free_mask & ~grid.interior_mask):
        raise ValueError("fixed source-free mask must be inside the interior")
    if np.any(grid.plasma_support_mask & ~grid.interior_mask):
        raise ValueError("plasma support mask must be inside the interior")
    r_grid = np.linspace(*grid_contract.R_BOUNDS_M, grid.resolution, dtype=np.float64)
    z_grid = np.linspace(*grid_contract.Z_BOUNDS_M, grid.resolution, dtype=np.float64)
    source_free = np.where(grid.fixed_source_free_mask, grid.total_forcing_zr, 0.0)
    source = np.asarray(grid.total_forcing_zr - source_free, dtype=np.float64)
    forcing_closure = float(np.max(np.abs(grid.total_forcing_zr - source - source_free)))
    response_report, responses, timings = partition_inverse_response(
        grid.total_forcing_zr,
        source,
        source_free,
        r_grid=r_grid,
        z_grid=z_grid,
    )
    total_drift = float(np.max(np.abs(responses["total"] - grid.total_response_zr)))
    if total_drift > grid_contract.PARTITION_CLOSURE_MAX_ABS:
        raise ValueError("fixed-physical total response drifted from CVGC1")
    total_l2 = float(response_report["total"]["area_weighted_l2"])
    source_free_l2 = float(response_report["source_free"]["area_weighted_l2"])
    d_area = float(grid.report["grid"]["cell_area_m2"])
    report = {
        "current_recovery_weighted_error": float(
            grid.report["current_recovery"]["weighted_fixed_physical_error"]
        ),
        "forcing_closure_max_abs": forcing_closure,
        "fixed_source_l2_fraction": forcing_l2_fraction(
            grid.total_forcing_zr,
            mask=np.asarray(~grid.fixed_source_free_mask & grid.interior_mask, dtype=np.bool_),
        ),
        "forcing_partition": {
            "source": field_metric(source, d_area_m2=d_area),
            "source_free": field_metric(source_free, d_area_m2=d_area),
            "total": field_metric(grid.total_forcing_zr, d_area_m2=d_area),
        },
        "inverse_timings_ms": timings,
        "resolution": grid.resolution,
        "response_partition": response_report,
        "source_free_response_fraction_of_total": _response_fraction(
            source_free_l2,
            total_l2,
        ),
        "total_response_reproduction_max_abs_wb": total_drift,
    }
    return FixedPhysicalGridResult(
        resolution=grid.resolution,
        report=report,
        source_forcing_zr=source,
        source_free_forcing_zr=np.asarray(source_free, dtype=np.float64),
        source_response_zr=responses["source"],
        source_free_response_zr=responses["source_free"],
        total_response_zr=responses["total"],
        interior_mask=grid.interior_mask,
        fixed_source_free_mask=grid.fixed_source_free_mask,
        plasma_support_mask=grid.plasma_support_mask,
    )


def _restrict_mask(mask: BoolArray, shape: tuple[int, int]) -> BoolArray:
    """Restrict a nested mask without interpolating its physical boundary."""
    return np.asarray(restrict_to_shape(mask, shape) > 0.5, dtype=np.bool_)


def _require_shared_mask(name: str, coarse: BoolArray, fine: BoolArray) -> None:
    """Require a frozen physical mask to agree on all shared coarse nodes."""
    shape = (int(coarse.shape[0]), int(coarse.shape[1]))
    if not np.array_equal(coarse, _restrict_mask(fine, shape)):
        raise ValueError(f"{name} mask drifted on nested shared nodes")


def _pairwise_row(
    coarse: FixedPhysicalGridResult,
    fine: FixedPhysicalGridResult,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compare fixed-partition fields across one exact nested-grid pair."""
    shape = (coarse.resolution, coarse.resolution)
    _require_shared_mask(
        "fixed physical source-free",
        coarse.fixed_source_free_mask,
        fine.fixed_source_free_mask,
    )
    _require_shared_mask("interior", coarse.interior_mask, fine.interior_mask)
    _require_shared_mask(
        "plasma support",
        coarse.plasma_support_mask,
        fine.plasma_support_mask,
    )
    fine_forcing = restrict_to_shape(fine.source_free_forcing_zr, shape)
    fine_response = restrict_to_shape(fine.source_free_response_zr, shape)
    d_area = float(
        (grid_contract.R_BOUNDS_M[1] - grid_contract.R_BOUNDS_M[0])
        * (grid_contract.Z_BOUNDS_M[1] - grid_contract.Z_BOUNDS_M[0])
        / (coarse.resolution - 1) ** 2
    )
    masks = {
        "fixed_physical_source_free": coarse.fixed_source_free_mask,
        "full_interior": coarse.interior_mask,
        "plasma_support": coarse.plasma_support_mask,
    }
    return {
        name: {
            "forcing": comparison_metric(
                coarse.source_free_forcing_zr,
                fine_forcing,
                mask=mask,
                d_area_m2=d_area,
            ),
            "response": comparison_metric(
                coarse.source_free_response_zr,
                fine_response,
                mask=mask,
                d_area_m2=d_area,
            ),
        }
        for name, mask in masks.items()
    }


def build_fixed_physical_convergence(
    results: list[FixedPhysicalGridResult],
) -> dict[str, Any]:
    """Build exact-restriction forcing and response convergence evidence.

    Parameters
    ----------
    results
        Results in the exact `33, 65, 129, 257` order.

    Returns
    -------
    dict
        Pairwise regional metrics and observed orders over both triples.

    Raises
    ------
    ValueError
        If the ladder is incomplete, reordered, duplicated, or mask-drifted.
    """
    if [row.resolution for row in results] != list(grid_contract.GRID_RESOLUTIONS):
        raise ValueError("results must contain the exact ordered four-grid ladder")
    by_resolution = {row.resolution: row for row in results}
    pairwise = {
        f"{coarse}_{fine}": _pairwise_row(by_resolution[coarse], by_resolution[fine])
        for coarse, fine in ((33, 65), (65, 129), (129, 257))
    }
    forcing_order: dict[str, Any] = {}
    response_order: dict[str, Any] = {}
    for name, (coarse, medium, fine) in {
        "33_65_129": (33, 65, 129),
        "65_129_257": (65, 129, 257),
    }.items():
        forcing_order[name] = observed_order_from_three(
            by_resolution[coarse].source_free_forcing_zr,
            by_resolution[medium].source_free_forcing_zr,
            by_resolution[fine].source_free_forcing_zr,
            mask=by_resolution[coarse].interior_mask,
        )
        response_order[name] = observed_order_from_three(
            by_resolution[coarse].source_free_response_zr,
            by_resolution[medium].source_free_response_zr,
            by_resolution[fine].source_free_response_zr,
            mask=by_resolution[coarse].interior_mask,
        )
    return {
        "pairwise": pairwise,
        "source_free_forcing_order": forcing_order,
        "source_free_response_order": response_order,
    }
