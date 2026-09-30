# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Outer Solver
"""Budgeted total-field optimization with complete equilibrium transactions."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import lsq_linear

from scpn_fusion.core.free_boundary_solver_state import FreeBoundaryState
from scpn_fusion.core.fusion_kernel_free_boundary import (
    _as_finite_points,
    _as_finite_vector,
    _kernel_boundary_points,
    compute_external_flux,
    reconstruct_boundary_flux_from_coils,
    resolve_shape_target_flux,
)
from scpn_fusion.core.fusion_kernel_numerics import FloatArray

if TYPE_CHECKING:
    from scpn_fusion.core.fusion_kernel import CoilSet


def sample_grid(kernel: Any, field: FloatArray, points: FloatArray) -> FloatArray:
    """Apply one bilinear sampling operator to an explicit grid.

    Parameters
    ----------
    kernel : Any
        Kernel providing the uniform radial and vertical axes.
    field : FloatArray
        Explicit flux grid, independent of mutable kernel storage.
    points : FloatArray
        Finite cylindrical sample points in metres.

    Returns
    -------
    FloatArray
        Sampled reduced flux in Wb/rad.
    """
    ir = np.clip(np.searchsorted(kernel.R, points[:, 0]) - 1, 0, kernel.NR - 2)
    iz = np.clip(np.searchsorted(kernel.Z, points[:, 1]) - 1, 0, kernel.NZ - 2)
    tr = np.clip((points[:, 0] - kernel.R[ir]) / kernel.dR, 0.0, 1.0)
    tz = np.clip((points[:, 1] - kernel.Z[iz]) / kernel.dZ, 0.0, 1.0)
    return np.asarray(
        (1 - tr) * (1 - tz) * field[iz, ir]
        + tr * (1 - tz) * field[iz, ir + 1]
        + (1 - tr) * tz * field[iz + 1, ir]
        + tr * tz * field[iz + 1, ir + 1],
        dtype=np.float64,
    )


def merit(residual: FloatArray, currents: FloatArray, alpha: float) -> float:
    """Evaluate the declared engineering objective in squared reduced flux.

    Parameters
    ----------
    residual : FloatArray
        Total-field target discrepancies in Wb/rad.
    currents : FloatArray
        Physical coil currents in amperes.
    alpha : float
        Regularization in squared reduced flux per squared ampere.

    Returns
    -------
    float
        Squared-flux objective value.
    """
    return 0.5 * float(residual @ residual) + 0.5 * alpha * float(currents @ currents)


def _flux_rms(flux: FloatArray) -> float:
    """Evaluate RMS without overflowing squares or erasing subnormal flux.

    Parameters
    ----------
    flux : FloatArray
        Nonempty sampled flux or residual values in Wb/rad.

    Returns
    -------
    float
        RMS in Wb/rad, retaining zero and nonfinite diagnostic values.
    """
    scale = float(np.max(np.abs(flux)))
    if scale == 0.0 or not np.isfinite(scale):
        return scale
    return scale * float(np.sqrt(np.mean((flux / scale) ** 2)))


def bounded_candidate(
    response: FloatArray,
    target: FloatArray,
    currents: FloatArray,
    limits: FloatArray,
    alpha: float,
) -> FloatArray | None:
    """Solve the regularized subproblem after eliminating fixed coordinates.

    Parameters
    ----------
    response : FloatArray
        Matched unit-grid sampled coil response.
    target : FloatArray
        Desired coil contribution after removing frozen plasma flux.
    currents : FloatArray
        Feasible accepted physical currents.
    limits : FloatArray
        Nonnegative symmetric current limits, possibly infinite.
    alpha : float
        Nonnegative dimensional regularization.

    Returns
    -------
    FloatArray or None
        Feasible absolute currents, or absence on an actual subproblem failure.
    """
    free = limits > 0.0
    candidate = currents.copy()
    candidate[~free] = 0.0
    if not np.any(free):
        return candidate
    columns = response[:, free]
    matrix = np.vstack([columns, np.sqrt(alpha) * np.eye(int(np.count_nonzero(free)))])
    rhs = np.concatenate(
        [target - response[:, ~free] @ candidate[~free], np.zeros(int(np.count_nonzero(free)))]
    )
    result = lsq_linear(matrix, rhs, bounds=(-limits[free], limits[free]), method="trf")
    if not result.success or not np.all(np.isfinite(result.x)):
        return None
    candidate[free] = result.x
    return candidate


def finite_fields(kernel: Any) -> bool:
    """Check every bound equilibrium field before accepting its state.

    Parameters
    ----------
    kernel : Any
        Actual equilibrium kernel after one execution.

    Returns
    -------
    bool
        Whether flux, source and both magnetic fields are finite grid arrays.
    """
    shape = (kernel.NZ, kernel.NR)
    return all(
        hasattr(kernel, name)
        and np.shape(getattr(kernel, name)) == shape
        and np.all(np.isfinite(getattr(kernel, name)))
        for name in ("Psi", "J_phi", "B_R", "B_Z")
    )


def result_diagnostics(
    kernel: Any,
    coils: CoilSet,
    state: FreeBoundaryState,
    *,
    calls: int,
    diff: float,
    accepted_steps: int,
    outcome: str,
    origin: str,
    trial_log: list[dict[str, Any]],
    points: FloatArray | None,
    target: FloatArray | None,
    response: FloatArray | None,
    limits: FloatArray,
    limiter_points: FloatArray | None,
    axis_point: FloatArray | None,
    x_points: FloatArray | None,
) -> dict[str, Any]:
    """Report actual returned fields without promoting engineering admission.

    Parameters
    ----------
    kernel : Any
        Kernel containing the restored final state.
    coils : CoilSet
        Restored physical coil request.
    state : FreeBoundaryState
        Owned final state and actual inner result.
    calls : int
        Actual equilibrium execution count.
    diff : float
        Last accepted flux-grid change.
    accepted_steps : int
        Accepted outer steps after the initial solve.
    outcome : str
        Engineering stopping reason.
    origin : str
        Provenance of the returned state.
    trial_log : list[dict[str, Any]]
        Actual executed candidates and acceptance reasons.
    points : FloatArray or None
        Shape sample coordinates.
    target : FloatArray or None
        Immutable total-field target.
    response : FloatArray or None
        Matched sampled unit-coil grids.
    limits : FloatArray
        Current limits used by the subproblem.
    limiter_points : FloatArray or None
        Optional vacuum diagnostic limiter.
    axis_point : FloatArray or None
        Optional vacuum diagnostic axis coordinate.
    x_points : FloatArray or None
        Optional vacuum diagnostic null coordinates.

    Returns
    -------
    dict[str, Any]
        Legacy diagnostics plus explicit actual state and unadmitted outcomes.
    """
    wall_points = _kernel_boundary_points(kernel)
    actual_wall = sample_grid(kernel, state.fields["Psi"], wall_points)
    boundary = reconstruct_boundary_flux_from_coils(
        kernel,
        coils,
        boundary_points=wall_points,
        target_flux=actual_wall,
        limiter_points=limiter_points,
        axis_point=axis_point,
        x_points=x_points,
    )
    shape: dict[str, Any] | None = None
    if points is not None and target is not None and response is not None:
        achieved = sample_grid(kernel, state.fields["Psi"], points)
        residual = achieved - target
        rms = _flux_rms(residual)
        target_rms = _flux_rms(target)
        shape = {
            "solver_mode": "free_boundary_solver_shape_current_optimization",
            "target_point_count": len(target),
            "coil_count": len(coils.positions),
            "response_rank": int(np.linalg.matrix_rank(response)),
            "response_condition": float(np.linalg.cond(response)),
            "target_flux": target.copy(),
            "achieved_flux": achieved,
            "residual": residual,
            "flux_rmse": rms,
            "flux_relative_rmse": rms / target_rms if target_rms > 0 else None,
            "max_abs_flux_residual": float(np.max(np.abs(residual))),
            "active_current_bounds": int(
                np.count_nonzero(np.isclose(np.abs(state.currents), limits, rtol=0.0))
            ),
        }
    inner = state.inner_result
    inner_status = (
        "failed"
        if inner is None
        else ("converged" if inner.get("converged") is True else "not_converged")
    )
    psi = state.fields["Psi"]
    iz, ir = np.unravel_index(int(np.argmax(psi)), psi.shape)
    topology = {
        "axis_point": (float(kernel.R[ir]), float(kernel.Z[iz])),
        "axis_flux": float(psi[iz, ir]),
        "x_point": kernel.find_x_point(psi),
        "role": "diagnostic_from_returned_flux",
    }
    return {
        "outer_iterations": calls,
        "final_diff": diff,
        "coil_currents": state.currents.copy(),
        "vacuum_boundary_abs_error": boundary["max_abs_error"],
        "boundary_reconstruction": boundary,
        "shape_optimization": shape,
        "inner_status": inner_status,
        "outer_outcome": outcome,
        "canonical_admission": "not_evaluated",
        "accepted_steps": accepted_steps,
        "state_origin": origin,
        "trial_log": deepcopy(trial_log),
        "inner_result": deepcopy(inner),
        "topology": topology,
        "fields": {name: field.copy() for name, field in state.fields.items()},
    }


def run_outer_equilibrium(
    kernel: Any,
    coils: CoilSet,
    max_outer_iter: int,
    tol: float,
    optimize_shape: bool,
    tikhonov_alpha: float,
    limiter_points: FloatArray | None,
    axis_point: FloatArray | None,
    x_points: FloatArray | None,
) -> dict[str, Any]:
    """Execute a bounded sequence of complete, rollback-capable equilibria.

    Parameters
    ----------
    kernel : Any
        Actual public equilibrium kernel.
    coils : CoilSet
        Physical request, updated only to a returned solved iterate.
    max_outer_iter : int
        Hard budget for all equilibrium executions, including rejected trials.
    tol : float
        Engineering accepted-grid-change threshold in Wb/rad.
    optimize_shape : bool
        Enable total-field shape optimization.
    tikhonov_alpha : float
        Regularization in squared reduced flux per squared ampere.
    limiter_points : FloatArray or None
        Vacuum diagnostic limiter coordinates.
    axis_point : FloatArray or None
        Vacuum diagnostic axis coordinate.
    x_points : FloatArray or None
        Vacuum diagnostic null coordinates.

    Returns
    -------
    dict[str, Any]
        Actual final state and engineering outcome, without canonical admission.
    """
    if (
        isinstance(max_outer_iter, bool)
        or not isinstance(max_outer_iter, int)
        or max_outer_iter < 1
    ):
        raise ValueError("max_outer_iter must be an integer >= 1.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and >= 0.")
    if not np.isfinite(tikhonov_alpha) or tikhonov_alpha < 0:
        raise ValueError("tikhonov_alpha must be finite and non-negative.")
    count = len(coils.positions)
    if count < 1:
        raise ValueError("CoilSet.positions must contain at least one coil.")
    currents = _as_finite_vector(coils.currents, name="currents", length=count)
    limits = (
        np.full(count, np.inf)
        if coils.current_limits is None
        else np.abs(_as_finite_vector(coils.current_limits, name="current_limits", length=count))
    )
    if np.any(np.abs(currents) > limits):
        raise ValueError("initial coil currents must satisfy current limits.")
    points = None
    target = None
    response = None
    if optimize_shape:
        points = _as_finite_points(coils.target_flux_points, name="target_flux_points").copy()
        if coils.target_flux_values is not None:
            target = _as_finite_vector(
                coils.target_flux_values, name="target_flux_values", length=len(points)
            ).copy()
        columns = []
        from scpn_fusion.core.fusion_kernel import CoilSet as Request

        for index in range(count):
            unit = np.zeros(count)
            unit[index] = 1.0
            unit_request = Request(positions=coils.positions, currents=unit, turns=coils.turns)
            columns.append(sample_grid(kernel, compute_external_flux(kernel, unit_request), points))
        response = np.column_stack(columns)
    external = compute_external_flux(kernel, coils)
    if not np.all(np.isfinite(external)):
        raise ValueError("external coil grid must contain only finite values.")
    reconstruct_boundary_flux_from_coils(
        kernel,
        coils,
        boundary_points=_kernel_boundary_points(kernel),
        limiter_points=limiter_points,
        axis_point=axis_point,
        x_points=x_points,
    )
    if kernel.hpc.is_available() and not kernel.hpc.supports_state_snapshot():
        raise RuntimeError("native free-boundary solve requires state checkpoint support")
    prior = FreeBoundaryState.capture(kernel, coils)
    initial_flux = prior.fields["Psi"].copy()
    initial_flux[0, :] = external[0, :]
    initial_flux[-1, :] = external[-1, :]
    initial_flux[:, 0] = external[:, 0]
    initial_flux[:, -1] = external[:, -1]
    calls = 1
    accepted_steps = 0
    trials: list[dict[str, Any]] = []
    outcome = "initial_solve_failed"
    origin = "pre_call_unsolved"
    diff = float("inf")
    state = prior
    try:
        inner = kernel.solve_equilibrium(preserve_initial_state=True, boundary_flux=external)
        if finite_fields(kernel):
            state = FreeBoundaryState.capture(kernel, coils, inner)
            origin = "initial_failed_solve"
            diff = float(np.max(np.abs(state.fields["Psi"] - initial_flux)))
        else:
            prior.restore(kernel, coils)
    except Exception:
        prior.restore(kernel, coils)
    if points is not None and target is None and state.inner_result is not None:
        target = resolve_shape_target_flux(kernel, coils).copy()
    # Fixed-current callers explicitly request diagnostic outer executions.
    # Keep unsuccessful diagnostic states distinct from accepted iterates.
    if not optimize_shape and state.inner_result is not None:
        while (
            state.inner_result.get("converged") is not True
            and calls < max_outer_iter
            and diff >= tol
        ):
            state.restore(kernel, coils)
            calls += 1
            reason = "inner_failed"
            try:
                inner = kernel.solve_equilibrium(
                    preserve_initial_state=True, boundary_flux=external
                )
                if finite_fields(kernel):
                    diagnostic = FreeBoundaryState.capture(kernel, coils, inner)
                    diff = float(np.max(np.abs(diagnostic.fields["Psi"] - state.fields["Psi"])))
                    state = diagnostic
                    origin = "failed_diagnostic_solve"
                    reason = (
                        "inner_converged"
                        if inner.get("converged") is True
                        else "inner_not_converged"
                    )
                else:
                    reason = "nonfinite_fields"
            except Exception:
                reason = "inner_failed"
            trials.append(
                {
                    "role": "fixed_current_diagnostic",
                    "reason": reason,
                    "accepted": False,
                    "solve_count": calls,
                }
            )
            if reason in {"inner_failed", "nonfinite_fields"}:
                state.restore(kernel, coils)
                break
    if state.inner_result is not None and state.inner_result.get("converged") is True:
        origin = "accepted_solve"
        outcome = "budget_exhausted"
        while True:
            if not optimize_shape and diff < tol:
                outcome = "outer_stationary"
                break
            candidate = state.currents.copy()
            plasma = None
            model_merit = 0.0
            if response is not None and target is not None and points is not None:
                total = sample_grid(kernel, state.fields["Psi"], points)
                plasma = total - response @ state.currents
                try:
                    fitted = bounded_candidate(
                        response, target - plasma, state.currents, limits, tikhonov_alpha
                    )
                except Exception:
                    fitted = None
                if fitted is None:
                    outcome = "subproblem_failed"
                    break
                candidate = fitted
                model_merit = merit(total - target, state.currents, tikhonov_alpha)
                full_step_prediction = model_merit - merit(
                    plasma + response @ candidate - target, candidate, tikhonov_alpha
                )
                if full_step_prediction <= 1e-12 * (1.0 + model_merit):
                    outcome = "linearized_stationary"
                    break
            if calls >= max_outer_iter:
                break
            accepted = False
            fractions = (1.0, 0.5, 0.25) if optimize_shape else (1.0,)
            attempted = 0
            for tau in fractions:
                if calls >= max_outer_iter:
                    outcome = "budget_exhausted"
                    break
                state.restore(kernel, coils)
                trial_currents = state.currents + tau * (candidate - state.currents)
                prediction: float | None = None
                if response is not None and target is not None and plasma is not None:
                    prediction = model_merit - merit(
                        plasma + response @ trial_currents - target,
                        trial_currents,
                        tikhonov_alpha,
                    )
                coils.currents = trial_currents.copy()
                calls += 1
                attempted += 1
                actual_merit = None
                reason = "inner_failed"
                trial_state = None
                try:
                    inner = kernel.solve_equilibrium(
                        preserve_initial_state=True,
                        boundary_flux=compute_external_flux(kernel, coils),
                    )
                    if not finite_fields(kernel):
                        reason = "nonfinite_fields"
                    elif inner.get("converged") is not True:
                        reason = "inner_not_converged"
                    else:
                        trial_state = FreeBoundaryState.capture(kernel, coils, inner)
                        if points is None or target is None:
                            reason = "accepted"
                        else:
                            actual_merit = merit(
                                sample_grid(kernel, kernel.Psi, points) - target,
                                trial_currents,
                                tikhonov_alpha,
                            )
                            reason = (
                                "accepted"
                                if prediction is not None
                                and prediction > 0
                                and np.isfinite(actual_merit)
                                and actual_merit <= model_merit - 1e-4 * prediction
                                else "armijo_rejected"
                            )
                except Exception:
                    reason = "inner_failed"
                accepted = reason == "accepted" and trial_state is not None
                trials.append(
                    {
                        "tau": tau,
                        "predicted_decrease": prediction,
                        "actual_merit": actual_merit,
                        "accepted": accepted,
                        "reason": reason,
                    }
                )
                if accepted and trial_state is not None:
                    diff = float(np.max(np.abs(trial_state.fields["Psi"] - state.fields["Psi"])))
                    state = trial_state
                    accepted_steps += 1
                    outcome = "outer_stationary" if diff < tol else "budget_exhausted"
                    break
                state.restore(kernel, coils)
            if not accepted:
                if attempted == len(fractions):
                    outcome = "no_acceptable_step"
                break
            if outcome == "outer_stationary":
                break
    state.restore(kernel, coils)
    return result_diagnostics(
        kernel,
        coils,
        state,
        calls=calls,
        diff=diff,
        accepted_steps=accepted_steps,
        outcome=outcome,
        origin=origin,
        trial_log=trials,
        points=points,
        target=target,
        response=response,
        limits=limits,
        limiter_points=limiter_points,
        axis_point=axis_point,
        x_points=x_points,
    )
