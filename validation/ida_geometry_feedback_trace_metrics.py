# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Checkpoint metrics for the IDA compiled geometry-feedback trace."""

from __future__ import annotations

import importlib
import math
from typing import Any, Callable, TypeAlias, cast

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import validation.ida_geometry_feedback_trace_contract as contract

_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_ablation = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_source"),
)
_predictive = cast(
    Any,
    importlib.import_module("scpn_fusion.core.jax_free_boundary_predictive"),
)

FloatArray: TypeAlias = NDArray[np.float64]
_array_sha256: Callable[[object], str] = _same_case._array_sha256
_shape_error_diagnostics: Callable[..., dict[str, Any]] = _same_case._shape_error_diagnostics
_plasma_current: Callable[..., Any] = _predictive._plasma_current
_laplacian_star: Callable[..., Any] = _predictive._laplacian_star
_predictive_gs_residual: Callable[..., Any] = _predictive.predictive_gs_residual


def _finite_array(value: object, *, field: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite two-dimensional array")
    return array


def _current_row(
    *,
    psi: FloatArray,
    axis: float,
    boundary: float,
    reference_psi_n: FloatArray,
    reference_current: FloatArray,
    reference_mask: NDArray[np.bool_],
    reference_axis: float,
    reference_boundary: float,
    r_grid: FloatArray,
    z_grid: FloatArray,
    knots: FloatArray,
    pprime_values: FloatArray,
    ffprime_values: FloatArray,
    ip_now: float,
    d_area: float,
) -> dict[str, Any]:
    span = boundary - axis
    if not math.isfinite(span) or abs(span) <= 1.0e-30:
        raise ValueError("checkpoint flux span must be finite and non-zero")
    current = np.asarray(
        _plasma_current(
            jnp.asarray(psi),
            jnp.asarray(r_grid),
            jnp.asarray(axis),
            jnp.asarray(boundary),
            jnp.asarray(knots),
            jnp.asarray(pprime_values),
            jnp.asarray(ffprime_values),
            jnp.asarray(ip_now),
            jnp.asarray(d_area),
            _same_case.DEFAULT_CUTOFF_WIDTH,
            _ablation.MU0_SI,
        ),
        dtype=np.float64,
    )
    diagnostics = _shape_error_diagnostics(
        candidate_psi_n=(psi - axis) / span,
        reference_psi_n=reference_psi_n,
        reference_current_mask=reference_mask,
        reference_current_density=reference_current,
        candidate_current_density=current,
        r_grid=r_grid,
        z_grid=z_grid,
        candidate_axis=axis,
        candidate_boundary=boundary,
        reference_axis=reference_axis,
        reference_boundary=reference_boundary,
    )
    distribution = diagnostics["current_distribution"]
    return {
        "centroid_delta_m": distribution["centroid_delta_m"],
        "cosine_similarity": distribution["cosine_similarity"],
        "current_density_sha256": _array_sha256(current),
        "total_variation_distance": distribution["total_variation_distance"],
    }


def _physical_residual(
    *,
    psi: FloatArray,
    ip_now: float,
    coil_current: FloatArray,
    pprime_values: FloatArray,
    ffprime_values: FloatArray,
    r_grid: FloatArray,
    z_grid: FloatArray,
    coil_r: FloatArray,
    coil_z: FloatArray,
    knots: FloatArray,
    response: jnp.ndarray,
    wall_indices: jnp.ndarray,
    source_indices: jnp.ndarray,
) -> dict[str, float]:
    residual = np.asarray(
        _predictive_gs_residual(
            jnp.asarray(psi),
            jnp.asarray(coil_current),
            jnp.asarray(pprime_values),
            jnp.asarray(ffprime_values),
            jnp.asarray(r_grid),
            jnp.asarray(z_grid),
            jnp.asarray(coil_r),
            jnp.asarray(coil_z),
            jnp.asarray(knots),
            jnp.asarray(ip_now),
            response,
            wall_indices,
            source_indices,
            _same_case.DEFAULT_CUTOFF_WIDTH,
            _ablation.MU0_SI,
        ),
        dtype=np.float64,
    )
    d_r = float(r_grid[1] - r_grid[0])
    d_z = float(z_grid[1] - z_grid[0])
    laplacian = np.asarray(
        _laplacian_star(jnp.asarray(psi), jnp.asarray(r_grid), d_r, d_z),
        dtype=np.float64,
    )
    interior = np.s_[1:-1, 1:-1]
    interior_rms = float(np.sqrt(np.mean(np.square(residual[interior]))))
    laplacian_rms = float(np.sqrt(np.mean(np.square(laplacian[interior]))))
    wall = np.asarray(wall_indices, dtype=np.int64)
    residual_flat = residual.reshape(-1)
    psi_flat = psi.reshape(-1)
    return {
        "interior_relative_rms": interior_rms / max(laplacian_rms, 1.0e-30),
        "wall_relative_l2": float(
            np.linalg.norm(residual_flat[wall])
            / max(float(np.linalg.norm(psi_flat[wall])), 1.0e-30)
        ),
    }


def _checkpoint_row(
    *,
    run_name: str,
    iteration_index: int,
    psi_before: object,
    fixed_point_residual: object,
    psi_after: object,
    ip_now: float,
    refinement: float,
    converged: bool,
    terminal: bool,
    context: dict[str, Any],
) -> dict[str, Any]:
    before = _finite_array(psi_before, field="psi_before")
    residual = _finite_array(fixed_point_residual, field="fixed_point_residual")
    after = _finite_array(psi_after, field="psi_after")
    axis = float(_ablation._smooth_axis_flux(jnp.asarray(before)))
    active_boundary = float(
        _ablation._smooth_xpoint_flux(
            jnp.asarray(before),
            context["r_jax"],
            context["z_jax"],
            refinement=refinement,
        )
    )
    physical_boundary = float(
        _ablation._smooth_xpoint_flux(
            jnp.asarray(before),
            context["r_jax"],
            context["z_jax"],
            refinement=1.0,
        )
    )
    production = _current_row(
        psi=before,
        axis=axis,
        boundary=active_boundary,
        ip_now=ip_now,
        **context["current"],
    )
    counterfactual = _current_row(
        psi=before,
        axis=axis,
        boundary=context["reference_boundary"],
        ip_now=ip_now,
        **context["current"],
    )
    before_norm = max(float(np.linalg.norm(before)), 1.0)
    return {
        "converged": converged,
        "fixed_point": {
            "accepted_update_relative_l2": float(np.linalg.norm(after - before)) / before_norm,
            "residual_linf_wb": float(np.max(np.abs(residual))),
            "residual_relative_l2": float(np.linalg.norm(residual)) / before_norm,
        },
        "geometry": {
            "active_boundary_wb": active_boundary,
            "active_span_wb": abs(active_boundary - axis),
            "axis_wb": axis,
            "physical_boundary_wb": physical_boundary,
            "physical_span_wb": abs(physical_boundary - axis),
        },
        "ip_fraction": abs(ip_now / context["ip_target"]),
        "ip_now_a": abs(ip_now),
        "iteration_index": iteration_index,
        "phase": contract.phase_for_iteration(run_name, iteration_index),
        "physical_residual": _physical_residual(
            psi=before,
            ip_now=ip_now,
            **context["residual"],
        ),
        "production_current": production,
        "psi_after_sha256": _array_sha256(after),
        "psi_before_sha256": _array_sha256(before),
        "reference_boundary_counterfactual": counterfactual,
        "separatrix_refinement": refinement,
        "terminal": terminal,
    }


def trace_run(
    *,
    run_name: str,
    trace: Any,
    requested_indices: tuple[int, ...],
    iteration_cap: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Convert one exact compiled trace into ordered checkpoint metrics."""
    rows: dict[int, dict[str, Any]] = {}
    recorded = np.asarray(trace.recorded, dtype=np.bool_)
    for position, iteration_index in enumerate(trace.checkpoint_indices):
        if recorded[position]:
            rows[iteration_index] = _checkpoint_row(
                run_name=run_name,
                iteration_index=iteration_index,
                psi_before=trace.psi_before[position],
                fixed_point_residual=trace.fixed_point_residual[position],
                psi_after=trace.psi_after[position],
                ip_now=float(trace.ip_now[position]),
                refinement=float(trace.separatrix_refinement[position]),
                converged=bool(trace.converged[position]),
                terminal=False,
                context=context,
            )
    terminal_index = trace.terminal_iteration_index
    rows[terminal_index] = _checkpoint_row(
        run_name=run_name,
        iteration_index=terminal_index,
        psi_before=trace.terminal_psi_before,
        fixed_point_residual=trace.terminal_fixed_point_residual,
        psi_after=trace.terminal_psi_after,
        ip_now=trace.terminal_ip_now,
        refinement=trace.terminal_separatrix_refinement,
        converged=trace.terminal_converged,
        terminal=True,
        context=context,
    )
    return {
        "checkpoint_indices_requested": list(requested_indices),
        "checkpoints": [rows[index] for index in sorted(rows)],
        "iteration_cap": iteration_cap,
        "iteration_count": trace.iteration_count,
        "terminal_psi_sha256": _array_sha256(trace.equilibrium),
        "terminated_early": trace.iteration_count < iteration_cap,
    }
