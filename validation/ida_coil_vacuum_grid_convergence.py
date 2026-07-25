# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Convergence
"""Exact nested-grid comparisons for the coil-vacuum diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import validation.ida_coil_vacuum_grid_contract as contract
from validation.ida_coil_vacuum_grid_fields import (
    BoolArray,
    FloatArray,
    comparison_metric,
    observed_order_from_three,
    restrict_to_shape,
)

PAIRWISE_REGIONS = (
    "fixed_physical_source_free",
    "full",
    "plasma_support",
    "source_footprint",
)


@dataclass(frozen=True)
class GridResult:
    """Private arrays and public metrics for one required grid."""

    resolution: int
    report: dict[str, Any]
    total_forcing_zr: FloatArray
    source_forcing_zr: FloatArray
    source_free_forcing_zr: FloatArray
    total_response_zr: FloatArray
    source_response_zr: FloatArray
    source_free_response_zr: FloatArray
    interior_mask: BoolArray
    primary_source_mask: BoolArray
    fixed_source_free_mask: BoolArray
    plasma_support_mask: BoolArray


def _restrict_mask(mask: BoolArray, shape: tuple[int, int]) -> BoolArray:
    """Restrict one nested boolean mask without interpolation."""
    restricted = restrict_to_shape(mask, shape)
    return np.asarray(restricted > 0.5, dtype=np.bool_)


def _require_invariant_mask(
    *,
    name: str,
    coarse: BoolArray,
    fine: BoolArray,
) -> None:
    """Require one physical-region mask to agree on shared coarse nodes."""
    restricted = _restrict_mask(fine, (int(coarse.shape[0]), int(coarse.shape[1])))
    if not np.array_equal(restricted, coarse):
        raise ValueError(f"{name} mask drifted on nested shared nodes")


def _region_pair(
    *,
    coarse_forcing: FloatArray,
    fine_forcing: FloatArray,
    coarse_response: FloatArray,
    fine_response: FloatArray,
    mask: BoolArray,
    d_area_m2: float,
) -> dict[str, dict[str, Any]]:
    """Compare forcing and response fields on one predeclared coarse region."""
    return {
        "forcing": comparison_metric(
            coarse_forcing,
            fine_forcing,
            mask=mask,
            d_area_m2=d_area_m2,
        ),
        "response": comparison_metric(
            coarse_response,
            fine_response,
            mask=mask,
            d_area_m2=d_area_m2,
        ),
    }


def _pairwise_row(coarse: GridResult, fine: GridResult) -> dict[str, Any]:
    """Build all four required region comparisons for one nested grid pair."""
    coarse_shape = (
        int(coarse.total_forcing_zr.shape[0]),
        int(coarse.total_forcing_zr.shape[1]),
    )
    _require_invariant_mask(
        name="fixed physical source-free",
        coarse=coarse.fixed_source_free_mask,
        fine=fine.fixed_source_free_mask,
    )
    _require_invariant_mask(
        name="full interior",
        coarse=coarse.interior_mask,
        fine=fine.interior_mask,
    )
    _require_invariant_mask(
        name="plasma support",
        coarse=coarse.plasma_support_mask,
        fine=fine.plasma_support_mask,
    )
    fine_fields = {
        "source_forcing": restrict_to_shape(fine.source_forcing_zr, coarse_shape),
        "source_free_forcing": restrict_to_shape(fine.source_free_forcing_zr, coarse_shape),
        "source_response": restrict_to_shape(fine.source_response_zr, coarse_shape),
        "source_free_response": restrict_to_shape(fine.source_free_response_zr, coarse_shape),
        "total_forcing": restrict_to_shape(fine.total_forcing_zr, coarse_shape),
        "total_response": restrict_to_shape(fine.total_response_zr, coarse_shape),
    }
    d_area = float(coarse.report["grid"]["cell_area_m2"])
    return {
        "fixed_physical_source_free": _region_pair(
            coarse_forcing=coarse.source_free_forcing_zr,
            fine_forcing=fine_fields["source_free_forcing"],
            coarse_response=coarse.source_free_response_zr,
            fine_response=fine_fields["source_free_response"],
            mask=coarse.fixed_source_free_mask,
            d_area_m2=d_area,
        ),
        "full": _region_pair(
            coarse_forcing=coarse.total_forcing_zr,
            fine_forcing=fine_fields["total_forcing"],
            coarse_response=coarse.total_response_zr,
            fine_response=fine_fields["total_response"],
            mask=coarse.interior_mask,
            d_area_m2=d_area,
        ),
        "plasma_support": _region_pair(
            coarse_forcing=coarse.total_forcing_zr,
            fine_forcing=fine_fields["total_forcing"],
            coarse_response=coarse.total_response_zr,
            fine_response=fine_fields["total_response"],
            mask=coarse.plasma_support_mask,
            d_area_m2=d_area,
        ),
        "source_footprint": _region_pair(
            coarse_forcing=coarse.source_forcing_zr,
            fine_forcing=fine_fields["source_forcing"],
            coarse_response=coarse.source_response_zr,
            fine_response=fine_fields["source_response"],
            mask=coarse.primary_source_mask,
            d_area_m2=d_area,
        ),
    }


def build_convergence(results: list[GridResult]) -> dict[str, Any]:
    """Build exact-restriction regional metrics and both order triples."""
    if [row.resolution for row in results] != list(contract.GRID_RESOLUTIONS):
        raise ValueError("results must contain the exact ordered four-grid ladder")
    by_resolution = {row.resolution: row for row in results}
    pairwise = {
        f"{coarse_size}_{fine_size}": _pairwise_row(
            by_resolution[coarse_size],
            by_resolution[fine_size],
        )
        for coarse_size, fine_size in ((33, 65), (65, 129), (129, 257))
    }
    orders = {
        "33_65_129": observed_order_from_three(
            by_resolution[33].source_free_forcing_zr,
            by_resolution[65].source_free_forcing_zr,
            by_resolution[129].source_free_forcing_zr,
            mask=by_resolution[33].fixed_source_free_mask,
        ),
        "65_129_257": observed_order_from_three(
            by_resolution[65].source_free_forcing_zr,
            by_resolution[129].source_free_forcing_zr,
            by_resolution[257].source_free_forcing_zr,
            mask=by_resolution[65].fixed_source_free_mask,
        ),
    }
    shape_129 = (
        int(by_resolution[129].source_free_response_zr.shape[0]),
        int(by_resolution[129].source_free_response_zr.shape[1]),
    )
    finest_on_129 = restrict_to_shape(
        by_resolution[257].source_free_response_zr,
        shape_129,
    )
    finest = comparison_metric(
        finest_on_129,
        by_resolution[129].source_free_response_zr,
        mask=by_resolution[129].fixed_source_free_mask,
        d_area_m2=float(by_resolution[129].report["grid"]["cell_area_m2"]),
    )
    return {
        "finest_source_free_response": finest,
        "pairwise": pairwise,
        "source_free_forcing_order": orders,
    }
