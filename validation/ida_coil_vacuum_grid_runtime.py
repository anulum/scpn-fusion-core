# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Runtime
"""Per-grid execution and cross-grid metrics for the coil-vacuum diagnostic."""

from __future__ import annotations

import importlib
import math
import time
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

import validation.ida_coil_vacuum_grid_contract as contract
from validation.ida_coil_vacuum_grid_convergence import GridResult
from validation.ida_coil_vacuum_grid_fields import (
    BoolArray,
    FilamentRecord,
    FloatArray,
    ParentCoilRecord,
    SourceMasks,
    canonical_sha256,
    field_metric,
    filament_arrays,
    flatten_filaments,
    forcing_l2_fraction,
    implied_current_density,
    integrate_current,
    mask_summary,
    native_lhs_zr,
    nearest_source_geometry,
    recovery_metric,
    source_masks,
    weighted_recovery_error,
    zero_identity_wall,
)

_source = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_source"),
)
_response_fields = cast(
    Any,
    importlib.import_module("validation.ida_operator_response_fields"),
)
_predictive = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_predictive"),
)


def _interior_mask(shape: tuple[int, int]) -> BoolArray:
    mask = np.ones(shape, dtype=np.bool_)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    return mask


def _vacuum_fields(
    *,
    tokamak: Any,
    filaments: tuple[FilamentRecord, ...],
    r_grid: FloatArray,
    z_grid: FloatArray,
    reference_129_zr: FloatArray | None,
) -> tuple[FloatArray, FloatArray, dict[str, float]]:
    r_mesh, z_mesh = np.meshgrid(r_grid, z_grid)
    freegs_start = time.perf_counter()
    evaluated_freegs = np.asarray(tokamak.psi(r_mesh, z_mesh), dtype=np.float64)
    freegs_elapsed = (time.perf_counter() - freegs_start) * 1000.0
    if reference_129_zr is not None:
        if evaluated_freegs.shape != reference_129_zr.shape:
            raise ValueError("129 FreeGS analytic vacuum field shape drifted")
        if not np.array_equal(evaluated_freegs, reference_129_zr):
            maximum = float(np.max(np.abs(evaluated_freegs - reference_129_zr)))
            if maximum > 1.0e-14:
                raise ValueError("129 FreeGS analytic vacuum field disagrees with equilibrium")
        freegs_vacuum = reference_129_zr
    else:
        freegs_vacuum = evaluated_freegs
    coil_r, coil_z, coil_current = filament_arrays(filaments)
    native_start = time.perf_counter()
    native_vacuum = np.asarray(
        _predictive.vacuum_field_si(
            jnp.asarray(r_grid),
            jnp.asarray(z_grid),
            jnp.asarray(coil_r),
            jnp.asarray(coil_z),
            jnp.asarray(coil_current),
            _source.MU0_SI,
        ),
        dtype=np.float64,
    )
    native_elapsed = (time.perf_counter() - native_start) * 1000.0
    return (
        freegs_vacuum,
        native_vacuum,
        {"freegs_field": freegs_elapsed, "native_field": native_elapsed},
    )


def _source_free_parity(
    freegs_vacuum: FloatArray,
    native_vacuum: FloatArray,
    *,
    mask: BoolArray,
) -> dict[str, Any]:
    if freegs_vacuum.shape != native_vacuum.shape or mask.shape != freegs_vacuum.shape:
        raise ValueError("vacuum parity fields/mask are incompatible")
    if not np.any(mask):
        raise ValueError("vacuum parity mask must select source-free points")
    reference = np.asarray(freegs_vacuum[mask], dtype=np.float64)
    delta = np.asarray((native_vacuum - freegs_vacuum)[mask], dtype=np.float64)
    nrmse = float(
        math.sqrt(np.mean(delta * delta)) / max(math.sqrt(np.mean(reference * reference)), 1.0e-30)
    )
    maximum = float(np.max(np.abs(delta)))
    return {
        "max_abs_wb": maximum,
        "nrmse": nrmse,
        "passes": bool(
            nrmse <= contract.PARITY_NRMSE_MAX and maximum <= contract.PARITY_MAX_ABS_WB
        ),
        "point_count": int(np.count_nonzero(mask)),
    }


def _bind_reference_forcing(
    evaluated_forcing: FloatArray,
    reference_129_forcing_zr: FloatArray | None,
) -> FloatArray:
    """Bind the exact 129 anchor after a strict independent reconstruction check."""
    if reference_129_forcing_zr is None:
        return evaluated_forcing
    if evaluated_forcing.shape != reference_129_forcing_zr.shape:
        raise ValueError("129 reconstructed forcing shape drifted")
    maximum = float(np.max(np.abs(evaluated_forcing - reference_129_forcing_zr)))
    if maximum > contract.PARITY_MAX_ABS_WB:
        raise ValueError("129 reconstructed forcing disagrees with the bound anchor")
    return reference_129_forcing_zr


def _partition_response(
    total_forcing: FloatArray,
    source_forcing: FloatArray,
    source_free_forcing: FloatArray,
    *,
    r_grid: FloatArray,
    z_grid: FloatArray,
) -> tuple[dict[str, Any], dict[str, FloatArray], dict[str, float]]:
    d_r = float(r_grid[1] - r_grid[0])
    d_z = float(z_grid[1] - z_grid[0])
    preconditioner = _predictive.build_gs_mg_preconditioner(
        total_forcing.shape,
        jnp.asarray(r_grid),
        d_r,
        d_z,
    )
    zero = np.zeros_like(total_forcing)
    fields = {
        "source": source_forcing,
        "source_free": source_free_forcing,
        "total": total_forcing,
    }
    responses: dict[str, FloatArray] = {}
    timings: dict[str, float] = {}
    for name, forcing in fields.items():
        started = time.perf_counter()
        responses[name] = -_response_fields.native_inverse(
            forcing,
            r_grid=r_grid,
            d_r=d_r,
            d_z=d_z,
            preconditioner=preconditioner,
            x0_zr=zero,
        )
        timings[f"inverse_{name}"] = (time.perf_counter() - started) * 1000.0
    closure = float(
        np.max(np.abs(responses["total"] - responses["source"] - responses["source_free"]))
    )
    d_area = d_r * d_z
    report = {
        "closure_max_abs_wb": closure,
        "source": field_metric(responses["source"], d_area_m2=d_area),
        "source_free": field_metric(responses["source_free"], d_area_m2=d_area),
        "total": field_metric(responses["total"], d_area_m2=d_area),
    }
    return report, responses, timings


def _current_recovery(
    *,
    tokamak: Any,
    parents: tuple[ParentCoilRecord, ...],
    r_grid: FloatArray,
    z_grid: FloatArray,
    fixed_physical_radius_m: float,
    aggregate_density: FloatArray,
    aggregate_masks: SourceMasks,
) -> dict[str, Any]:
    d_area = float((r_grid[1] - r_grid[0]) * (z_grid[1] - z_grid[0]))
    r_mesh, z_mesh = np.meshgrid(r_grid, z_grid)
    coil_by_name = {str(name): coil for name, coil in tokamak.coils}
    parent_rows: list[dict[str, Any]] = []
    primary_metrics: list[dict[str, float]] = []
    fixed_metrics: list[dict[str, float]] = []
    for parent in parents:
        coil = coil_by_name.get(parent.name)
        if coil is None:
            raise ValueError(f"runtime tokamak is missing parent coil {parent.name}")
        parent_flux = np.asarray(coil.psi(r_mesh, z_mesh), dtype=np.float64)
        parent_forcing = zero_identity_wall(
            native_lhs_zr(parent_flux, r_grid=r_grid, z_grid=z_grid),
            field=f"{parent.name} forcing",
        )
        parent_density = implied_current_density(
            parent_forcing,
            r_grid=r_grid,
            mu0=_source.MU0_SI,
        )
        geometry = nearest_source_geometry(r_grid, z_grid, parent.filaments)
        masks = source_masks(
            geometry,
            fixed_physical_radius_m=fixed_physical_radius_m,
        )
        full = _interior_mask((int(parent_flux.shape[0]), int(parent_flux.shape[1])))
        regions = {
            "fixed_physical": integrate_current(
                parent_density,
                mask=masks.fixed_physical,
                d_area_m2=d_area,
            ),
            "full_interior": integrate_current(
                parent_density,
                mask=full,
                d_area_m2=d_area,
            ),
            "primary": integrate_current(
                parent_density,
                mask=masks.rho_h_le_2,
                d_area_m2=d_area,
            ),
            "source_free": integrate_current(
                parent_density,
                mask=np.asarray(~masks.rho_h_le_2, dtype=np.bool_),
                d_area_m2=d_area,
            ),
        }
        metrics = {
            name: recovery_metric(value, parent.effective_current_a_turns)
            for name, value in regions.items()
        }
        primary_metrics.append(metrics["primary"])
        fixed_metrics.append(metrics["fixed_physical"])
        parent_rows.append({"name": parent.name, "regions": metrics})
    aggregate_full = _interior_mask(
        (int(aggregate_density.shape[0]), int(aggregate_density.shape[1]))
    )
    aggregate = {
        "fixed_physical_a_turns": integrate_current(
            aggregate_density,
            mask=aggregate_masks.fixed_physical,
            d_area_m2=d_area,
        ),
        "full_interior_a_turns": integrate_current(
            aggregate_density,
            mask=aggregate_full,
            d_area_m2=d_area,
        ),
        "primary_a_turns": integrate_current(
            aggregate_density,
            mask=aggregate_masks.rho_h_le_2,
            d_area_m2=d_area,
        ),
        "source_free_a_turns": integrate_current(
            aggregate_density,
            mask=np.asarray(~aggregate_masks.rho_h_le_2, dtype=np.bool_),
            d_area_m2=d_area,
        ),
        "target_net_a_turns": math.fsum(parent.effective_current_a_turns for parent in parents),
        "target_absolute_sum_a_turns": math.fsum(
            abs(parent.effective_current_a_turns) for parent in parents
        ),
    }
    return {
        "aggregate": aggregate,
        "parents": parent_rows,
        "weighted_fixed_physical_error": weighted_recovery_error(tuple(fixed_metrics)),
        "weighted_primary_error": weighted_recovery_error(tuple(primary_metrics)),
    }


def run_grid(
    *,
    resolution: int,
    tokamak: Any,
    parents: tuple[ParentCoilRecord, ...],
    r_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    fixed_physical_radius_m: float,
    reference_129_zr: FloatArray | None,
    reference_129_forcing_zr: FloatArray | None,
    plasma_support_mask: BoolArray,
) -> GridResult:
    """Execute one required grid with unchanged field and inverse operators."""
    if resolution not in contract.GRID_RESOLUTIONS:
        raise ValueError("resolution must be one of the four required grids")
    r_grid = np.linspace(*r_bounds, resolution, dtype=np.float64)
    z_grid = np.linspace(*z_bounds, resolution, dtype=np.float64)
    d_r = float(r_grid[1] - r_grid[0])
    d_z = float(z_grid[1] - z_grid[0])
    d_area = d_r * d_z
    filaments = flatten_filaments(parents)
    geometry = nearest_source_geometry(r_grid, z_grid, filaments)
    masks = source_masks(
        geometry,
        fixed_physical_radius_m=fixed_physical_radius_m,
    )
    interior = _interior_mask((resolution, resolution))
    if plasma_support_mask.shape != interior.shape:
        raise ValueError("plasma support mask must match the runtime grid")
    plasma_support = np.asarray(plasma_support_mask & interior, dtype=np.bool_)
    if not np.any(plasma_support):
        raise ValueError("plasma support mask must select points on the runtime grid")
    primary_source = np.asarray(masks.rho_h_le_2 & interior, dtype=np.bool_)
    fixed_source_free = np.asarray(~masks.fixed_physical & interior, dtype=np.bool_)
    freegs_vacuum, native_vacuum, field_timings = _vacuum_fields(
        tokamak=tokamak,
        filaments=filaments,
        r_grid=r_grid,
        z_grid=z_grid,
        reference_129_zr=reference_129_zr,
    )
    total_forcing = _bind_reference_forcing(
        zero_identity_wall(
            native_lhs_zr(freegs_vacuum, r_grid=r_grid, z_grid=z_grid),
            field="total coil-vacuum forcing",
        ),
        reference_129_forcing_zr,
    )
    source_forcing = np.where(masks.rho_h_le_2, total_forcing, 0.0)
    source_free_forcing = np.asarray(total_forcing - source_forcing, dtype=np.float64)
    forcing_closure = float(np.max(np.abs(total_forcing - source_forcing - source_free_forcing)))
    current_density = implied_current_density(
        total_forcing,
        r_grid=r_grid,
        mu0=_source.MU0_SI,
    )
    response_report, responses, inverse_timings = _partition_response(
        total_forcing,
        source_forcing,
        source_free_forcing,
        r_grid=r_grid,
        z_grid=z_grid,
    )
    nearest = geometry.distance_m
    report = {
        "current_recovery": _current_recovery(
            tokamak=tokamak,
            parents=parents,
            r_grid=r_grid,
            z_grid=z_grid,
            fixed_physical_radius_m=fixed_physical_radius_m,
            aggregate_density=current_density,
            aggregate_masks=masks,
        ),
        "forcing_partition": {
            "closure_max_abs": forcing_closure,
            "primary_l2_fraction": forcing_l2_fraction(
                total_forcing,
                mask=masks.rho_h_le_2,
            ),
            "sensitivity_l2_fraction": {
                "rho_h_le_1": forcing_l2_fraction(
                    total_forcing,
                    mask=masks.rho_h_le_1,
                ),
                "rho_h_le_2": forcing_l2_fraction(
                    total_forcing,
                    mask=masks.rho_h_le_2,
                ),
                "rho_h_le_4": forcing_l2_fraction(
                    total_forcing,
                    mask=masks.rho_h_le_4,
                ),
            },
            "source": field_metric(source_forcing, d_area_m2=d_area),
            "source_free": field_metric(source_free_forcing, d_area_m2=d_area),
            "total": field_metric(total_forcing, d_area_m2=d_area),
        },
        "grid": {
            "cell_area_m2": d_area,
            "d_r_m": d_r,
            "d_z_m": d_z,
            "filament_phase_sha256": canonical_sha256(geometry.filament_phase_rz),
            "minimum_filament_to_node_distance_m": float(np.min(nearest)),
            "nearest_distance_max_m": float(np.max(nearest)),
            "nearest_distance_mean_m": float(np.mean(nearest)),
            "r_bounds_m": list(r_bounds),
            "z_bounds_m": list(z_bounds),
        },
        "masks": {
            "fixed_physical": mask_summary(masks.fixed_physical, d_area_m2=d_area),
            "fixed_physical_radius_m": fixed_physical_radius_m,
            "primary_fixed_overlap_point_count": int(
                np.count_nonzero(masks.rho_h_le_2 & masks.fixed_physical)
            ),
            "plasma_support": mask_summary(plasma_support, d_area_m2=d_area),
            "rho_h_le_1": mask_summary(masks.rho_h_le_1, d_area_m2=d_area),
            "rho_h_le_2": mask_summary(masks.rho_h_le_2, d_area_m2=d_area),
            "rho_h_le_4": mask_summary(masks.rho_h_le_4, d_area_m2=d_area),
        },
        "resolution": resolution,
        "response_partition": response_report,
        "timings_ms": {**field_timings, **inverse_timings},
        "vacuum_fields": {
            "freegs": field_metric(freegs_vacuum, d_area_m2=d_area),
            "native": field_metric(native_vacuum, d_area_m2=d_area),
            "source_free_parity": _source_free_parity(
                freegs_vacuum,
                native_vacuum,
                mask=fixed_source_free,
            ),
        },
    }
    return GridResult(
        resolution=resolution,
        report=report,
        total_forcing_zr=total_forcing,
        source_forcing_zr=source_forcing,
        source_free_forcing_zr=source_free_forcing,
        total_response_zr=responses["total"],
        source_response_zr=responses["source"],
        source_free_response_zr=responses["source_free"],
        interior_mask=interior,
        primary_source_mask=primary_source,
        fixed_source_free_mask=fixed_source_free,
        plasma_support_mask=plasma_support,
    )
