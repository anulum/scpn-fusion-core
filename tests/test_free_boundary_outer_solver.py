# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Public Free-Boundary Outer Solve Tests
"""Real public equilibrium solves for the frozen outer-state contract."""

from __future__ import annotations

import json
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.fusion_kernel import CoilSet, FusionKernel
from scpn_fusion.core.fusion_kernel_free_boundary import compute_external_flux, green_function


@pytest.fixture
def outer_kernel(tmp_path: Path) -> FusionKernel:
    """Construct the preregistered five by five zero-source equilibrium.

    Parameters
    ----------
    tmp_path : Path
        Isolated configuration directory.

    Returns
    -------
    FusionKernel
        Actual Python SOR kernel with the frozen residual requirements.
    """
    config = {
        "reactor_name": "Public-vacuum-outer-contract",
        "grid_resolution": [5, 5],
        "dimensions": {"R_min": 4, "R_max": 8, "Z_min": -4, "Z_max": 4},
        "physics": {"plasma_current_target": 0.0, "vacuum_permeability": 4e-7 * np.pi},
        "coils": [{"name": "CS", "r": 3, "z": 0, "current": 100000}],
        "solver": {
            "max_iterations": 1000,
            "convergence_threshold": 1e-12,
            "relaxation_factor": 1.0,
            "solver_method": "sor",
            "require_gs_residual": True,
            "gs_residual_threshold": 1e-12,
            "fail_on_diverge": True,
            "sor_omega": 1.0,
        },
    }
    path = tmp_path / "equilibrium.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    kernel = FusionKernel(path)
    assert not kernel.hpc.is_available(), "Frozen fixture requires the Python SOR backend"
    return kernel


def positive_coils(kernel: FusionKernel) -> CoilSet:
    """Construct the frozen attainable wall target without fitting its amplitude.

    Parameters
    ----------
    kernel : FusionKernel
        Actual kernel on the frozen five by five grid.

    Returns
    -------
    CoilSet
        Zero initial currents with targets manufactured from declared currents.
    """
    positions = [(3.0, 0.0), (9.0, 0.0)]
    points = np.array([[5.0, -4.0], [7.0, -4.0]])
    declared = np.array([1e5, 2e5])
    response = np.column_stack(
        [
            compute_external_flux(kernel, CoilSet(positions, unit, [1, 1]))[0, [1, 3]]
            for unit in np.eye(2)
        ]
    )
    target = response @ declared
    return CoilSet(
        positions=positions,
        currents=np.zeros(2),
        turns=[1, 1],
        current_limits=np.full(2, 2e6),
        target_flux_points=points,
        target_flux_values=target,
    )


def independent_wall_error(kernel: FusionKernel, coils: CoilSet) -> float:
    """Compare actual wall samples with a scalar filament calculation.

    Parameters
    ----------
    kernel : FusionKernel
        Actual returned field state.
    coils : CoilSet
        Actual returned current request.

    Returns
    -------
    float
        Maximum wall discrepancy in Wb/rad. This is a software coherence check.
    """
    errors = []
    for iz, z in enumerate(kernel.Z):
        for ir, r in enumerate(kernel.R):
            if iz in (0, kernel.NZ - 1) or ir in (0, kernel.NR - 1):
                expected = sum(
                    float(current) * green_function(*source, float(r), float(z))
                    for source, current in zip(coils.positions, coils.currents, strict=True)
                )
                errors.append(abs(float(kernel.Psi[iz, ir]) - expected))
    return max(errors)


def test_one_solve_budget_coherent_initial(outer_kernel: FusionKernel) -> None:
    """A one-solve budget cannot install unsolved candidate currents.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Frozen actual equilibrium kernel.
    """
    coils = positive_coils(outer_kernel)
    result = outer_kernel.solve_free_boundary(
        coils,
        max_outer_iter=1,
        tol=0.0,
        optimize_shape=True,
        tikhonov_alpha=0.0,
    )
    np.testing.assert_array_equal(coils.currents, np.zeros(2))
    assert result["outer_iterations"] == 1
    assert result["outer_outcome"] == "budget_exhausted"
    assert result["accepted_steps"] == 0
    assert result["canonical_admission"] == "not_evaluated"
    assert independent_wall_error(outer_kernel, coils) < 1e-15


def test_zero_target_relative_residual_is_none(outer_kernel: FusionKernel) -> None:
    """A zero target RMS has no defined relative residual.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Frozen actual equilibrium kernel.
    """
    coils = positive_coils(outer_kernel)
    coils.target_flux_values = np.zeros(2)
    result = outer_kernel.solve_free_boundary(coils, max_outer_iter=1, optimize_shape=True)
    assert result["shape_optimization"]["flux_relative_rmse"] is None
    assert result["shape_optimization"]["flux_rmse"] == 0.0


def test_returned_arrays_do_not_alias(outer_kernel: FusionKernel) -> None:
    """Result inspection cannot mutate bound accepted fields or currents.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Frozen actual equilibrium kernel.
    """
    coils = positive_coils(outer_kernel)
    result = outer_kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=0.0,
        optimize_shape=True,
        tikhonov_alpha=0.0,
    )
    psi = outer_kernel.Psi.copy()
    currents = coils.currents.copy()
    result["fields"]["Psi"][:] = 123.0
    result["coil_currents"][:] = 0.0
    np.testing.assert_array_equal(outer_kernel.Psi, psi)
    np.testing.assert_array_equal(coils.currents, currents)


def test_nonfinite_request_refused_before_mutation(outer_kernel: FusionKernel) -> None:
    """Refuse a nonfinite target before changing currents or allocating fields.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Frozen actual equilibrium kernel.
    """
    coils = positive_coils(outer_kernel)
    coils.target_flux_values = np.array([np.nan, 0.0])
    before = outer_kernel.Psi.copy()
    with pytest.raises(ValueError, match="target_flux_values"):
        outer_kernel.solve_free_boundary(coils, optimize_shape=True)
    np.testing.assert_array_equal(outer_kernel.Psi, before)
    np.testing.assert_array_equal(coils.currents, np.zeros(2))
    assert not hasattr(outer_kernel, "B_R")


def test_recovery_after_rejected_trials(outer_kernel: FusionKernel) -> None:
    """A later attainable request recovers after an actual rejected search.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Frozen actual equilibrium kernel.
    """
    rejected, alpha = rejection_coils(outer_kernel)
    failure = outer_kernel.solve_free_boundary(
        rejected,
        max_outer_iter=4,
        tol=0.0,
        optimize_shape=True,
        tikhonov_alpha=alpha,
    )
    assert failure["outer_outcome"] == "no_acceptable_step"
    coils = positive_coils(outer_kernel)
    recovered = outer_kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=0.0,
        optimize_shape=True,
        tikhonov_alpha=0.0,
    )
    assert recovered["accepted_steps"] == 1
    assert recovered["inner_status"] == "converged"
    np.testing.assert_allclose(coils.currents, [1e5, 2e5], rtol=1e-12)
    assert independent_wall_error(outer_kernel, coils) < 1e-15


def historical_kernel(tmp_path: Path) -> FusionKernel:
    """Build the immutable historical under-budget nonlinear request.

    Parameters
    ----------
    tmp_path : Path
        Isolated configuration directory.

    Returns
    -------
    FusionKernel
        Actual seventeen by seventeen Jacobi kernel with one inner iteration.
    """
    config = {
        "reactor_name": "Free-boundary audit diagnostic",
        "grid_resolution": [17, 17],
        "dimensions": {"R_min": 4, "R_max": 8, "Z_min": -4, "Z_max": 4},
        "physics": {"plasma_current_target": 1000000, "vacuum_permeability": 4e-7 * np.pi},
        "coils": [{"name": "CS", "r": 3, "z": 0, "current": 100000}],
        "solver": {
            "max_iterations": 1,
            "convergence_threshold": 1e-12,
            "relaxation_factor": 0.1,
            "solver_method": "jacobi",
            "require_gs_residual": True,
            "gs_residual_threshold": 1e-12,
            "fail_on_diverge": True,
        },
    }
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return FusionKernel(path)


def test_historical_underbudget_state_is_unadmitted(tmp_path: Path) -> None:
    """Replay the original seventeen by seventeen stale-current counterexample.

    Parameters
    ----------
    tmp_path : Path
        Isolated historical request configuration.
    """
    kernel = historical_kernel(tmp_path)
    coils = CoilSet(
        positions=[(3.0, 0.0), (9.0, 0.0)],
        currents=np.array([1e5, 2e5]),
        turns=[1, 1],
        current_limits=np.full(2, 1e6),
        target_flux_points=np.array([[5.0, -1.0], [7.0, 1.0]]),
        target_flux_values=np.zeros(2),
    )
    result = kernel.solve_free_boundary(
        coils,
        max_outer_iter=1,
        tol=0.0,
        optimize_shape=True,
        tikhonov_alpha=0.0,
    )
    assert result["outer_outcome"] == "initial_solve_failed"
    assert result["state_origin"] == "initial_failed_solve"
    assert result["accepted_steps"] == 0
    assert result["canonical_admission"] == "not_evaluated"
    assert result["inner_status"] == "not_converged"
    np.testing.assert_array_equal(coils.currents, [1e5, 2e5])
    assert independent_wall_error(kernel, coils) < 1e-15


def test_accepted_step_coherent_snapshot(outer_kernel: FusionKernel) -> None:
    """An accepted candidate has matching currents, fields and total residuals.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Frozen actual equilibrium kernel.
    """
    coils = positive_coils(outer_kernel)
    result = outer_kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=0.0,
        optimize_shape=True,
        tikhonov_alpha=0.0,
    )
    assert result["outer_iterations"] == 2
    assert result["accepted_steps"] == 1
    assert result["inner_status"] == "converged"
    assert result["canonical_admission"] == "not_evaluated"
    np.testing.assert_allclose(coils.currents, [1e5, 2e5], rtol=1e-12)
    wall = independent_wall_error(outer_kernel, coils)
    assert wall < 1e-15
    assert result["vacuum_boundary_abs_error"] == pytest.approx(wall, abs=1e-15)
    samples: NDArray[np.float64] = outer_kernel.Psi[0, [1, 3]]
    np.testing.assert_array_equal(result["shape_optimization"]["achieved_flux"], samples)
    np.testing.assert_array_equal(result["fields"]["Psi"], outer_kernel.Psi)


def discrete_reference(boundary: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve the independent frozen zero-source discrete Dirichlet system.

    Parameters
    ----------
    boundary : NDArray[np.float64]
        Five by five external grid whose wall defines the reference.

    Returns
    -------
    NDArray[np.float64]
        Full grid from a dense linear solve, independent of production SOR.
    """
    matrix = np.zeros((9, 9))
    rhs = np.zeros(9)
    for iz in range(1, 4):
        for ir in range(1, 4):
            index = (iz - 1) * 3 + ir - 1
            matrix[index, index] = 2.5
            radius = 4 + ir
            for nz, nr, coefficient in (
                (iz, ir + 1, 1 - 0.5 / radius),
                (iz, ir - 1, 1 + 0.5 / radius),
                (iz + 1, ir, 0.25),
                (iz - 1, ir, 0.25),
            ):
                if nz in (0, 4) or nr in (0, 4):
                    rhs[index] += coefficient * boundary[nz, nr]
                else:
                    matrix[index, (nz - 1) * 3 + nr - 1] -= coefficient
    result = boundary.copy()
    result[1:-1, 1:-1] = np.linalg.solve(matrix, rhs).reshape(3, 3)
    return result


def rejection_coils(kernel: FusionKernel) -> tuple[CoilSet, float]:
    """Construct the preregistered real Armijo rejection case.

    Parameters
    ----------
    kernel : FusionKernel
        Actual frozen uniform grid.

    Returns
    -------
    tuple[CoilSet, float]
        Request and dimensional regularization fixed by the discrete reference.
    """
    coils = positive_coils(kernel)
    grid_samples = []
    reference_samples = []
    for index in range(2):
        coils.currents = np.zeros(2)
        coils.currents[index] = 1.0
        external = kernel._compute_external_flux(coils)
        grid_samples.append(float(external[2, 1]))
        reference_samples.append(float(discrete_reference(external)[2, 1]))
    g = np.array(grid_samples)
    h = np.array(reference_samples)
    direction = np.array([g[1], -g[0]]) / np.hypot(g[0], g[1])
    coils.currents = 1e6 * direction
    coils.target_flux_points = np.array([[5.0, 0.0]])
    coils.target_flux_values = np.array([float(h @ coils.currents)])
    alpha = float(h @ direction) ** 2 / 64
    return coils, alpha


def exact_reference_error_bound(field: NDArray[np.float64]) -> Fraction:
    """Apply the frozen maximum-principle bound to exact binary64 residuals.

    Parameters
    ----------
    field : NDArray[np.float64]
        Five by five grid with its actual fixed wall.

    Returns
    -------
    Fraction
        Certified maximum grid error: eight thirds times maximum PDE residual.
    """
    values = [[Fraction.from_float(float(value)) for value in row] for row in field]
    residuals = []
    for iz in range(1, 4):
        for ir in range(1, 4):
            radius = Fraction(4 + ir)
            residual = Fraction(5, 2) * values[iz][ir]
            residual -= (1 - Fraction(1, 2) / radius) * values[iz][ir + 1]
            residual -= (1 + Fraction(1, 2) / radius) * values[iz][ir - 1]
            residual -= Fraction(1, 4) * (values[iz + 1][ir] + values[iz - 1][ir])
            residuals.append(abs(residual))
    return Fraction(8, 3) * max(residuals)


@pytest.mark.parametrize("budget,trial_count", [(2, 1), (3, 2), (4, 3)])
def test_three_real_armijo_rejections_restore_initial(
    outer_kernel: FusionKernel,
    budget: int,
    trial_count: int,
) -> None:
    """Reject real inner-converged candidates and restore the accepted state.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Frozen actual equilibrium kernel.
    budget : int
        Preregistered solve budget.
    trial_count : int
        Expected executed trial count, determined by the budget.
    """
    coils, alpha = rejection_coils(outer_kernel)
    initial = coils.currents.copy()
    expected = discrete_reference(outer_kernel._compute_external_flux(coils))
    result = outer_kernel.solve_free_boundary(
        coils,
        max_outer_iter=budget,
        tol=0.0,
        optimize_shape=True,
        tikhonov_alpha=alpha,
    )
    assert result["outer_iterations"] == budget
    assert result["accepted_steps"] == 0
    assert len(result["trial_log"]) == trial_count
    assert all(row["reason"] == "armijo_rejected" for row in result["trial_log"])
    assert all(not row["accepted"] for row in result["trial_log"])
    assert result["outer_outcome"] == (
        "no_acceptable_step" if trial_count == 3 else "budget_exhausted"
    )
    np.testing.assert_array_equal(coils.currents, initial)
    discrepancy = max(
        abs(Fraction.from_float(float(actual)) - Fraction.from_float(float(reference)))
        for actual, reference in zip(outer_kernel.Psi.ravel(), expected.ravel(), strict=True)
    )
    assert discrepancy <= (
        exact_reference_error_bound(outer_kernel.Psi) + exact_reference_error_bound(expected)
    )
    assert not outer_kernel.J_phi.any()
    assert independent_wall_error(outer_kernel, coils) < 1e-15
    actual = float(outer_kernel.Psi[2, 1])
    np.testing.assert_array_equal(result["shape_optimization"]["achieved_flux"], [actual])
    assert coils.target_flux_values is not None
    residual = actual - float(coils.target_flux_values[0])
    assert result["shape_optimization"]["flux_rmse"] == abs(residual)


def test_fixed_zero_stationary_without_trials(outer_kernel: FusionKernel) -> None:
    """Eliminate fixed zero bounds without sending equal bounds to SciPy.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Frozen actual equilibrium kernel.
    """
    coils = positive_coils(outer_kernel)
    coils.current_limits = np.zeros(2)
    result = outer_kernel.solve_free_boundary(
        coils,
        max_outer_iter=1,
        optimize_shape=True,
        tikhonov_alpha=0.0,
    )
    assert result["outer_outcome"] == "linearized_stationary"
    assert result["outer_iterations"] == 1
    assert result["trial_log"] == []
    np.testing.assert_array_equal(coils.currents, np.zeros(2))
    assert result["shape_optimization"]["flux_rmse"] > 0.0
    assert result["canonical_admission"] == "not_evaluated"


def test_fixed_current_failed_diagnostics_honor_budget(tmp_path: Path) -> None:
    """Preserve declared fixed-current executions without accepting failed states.

    Parameters
    ----------
    tmp_path : Path
        Isolated historical configuration directory.
    """
    kernel = historical_kernel(tmp_path)
    coils = CoilSet(
        positions=[(3.0, 0.0), (9.0, 0.0)],
        currents=np.array([1e5, 2e5]),
        turns=[1, 1],
        current_limits=np.full(2, 1e6),
    )
    result = kernel.solve_free_boundary(coils, max_outer_iter=2, tol=0.0)
    assert result["outer_iterations"] == 2
    assert result["outer_outcome"] == "initial_solve_failed"
    assert result["state_origin"] == "failed_diagnostic_solve"
    assert result["inner_status"] == "not_converged"
    assert result["accepted_steps"] == 0
    assert result["canonical_admission"] == "not_evaluated"
    assert len(result["trial_log"]) == 1
    assert result["trial_log"][0]["role"] == "fixed_current_diagnostic"
    assert result["trial_log"][0]["accepted"] is False
    np.testing.assert_array_equal(coils.currents, [1e5, 2e5])
    assert independent_wall_error(kernel, coils) < 1e-15


def test_actual_total_flux_residual(outer_kernel: FusionKernel) -> None:
    """Report signed discrepancies from the returned total field.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Actual equilibrium kernel using the frozen rejection request.
    """
    coils, alpha = rejection_coils(outer_kernel)
    result = outer_kernel.solve_free_boundary(
        coils, max_outer_iter=4, tol=0.0, optimize_shape=True, tikhonov_alpha=alpha
    )
    shape = result["shape_optimization"]
    assert coils.target_flux_values is not None
    actual = np.array([outer_kernel.Psi[2, 1]])
    residual = actual - coils.target_flux_values
    np.testing.assert_array_equal(shape["residual"], residual)
    assert shape["flux_relative_rmse"] == pytest.approx(
        np.linalg.norm(residual) / np.linalg.norm(coils.target_flux_values)
    )
    assert shape["max_abs_flux_residual"] == abs(float(residual[0]))
    assert result["canonical_admission"] == "not_evaluated"


def test_implicit_target_is_frozen(outer_kernel: FusionKernel) -> None:
    """Retain the initial isoflux mean after an actual current change.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Actual zero-source kernel with wall sampling points.
    """
    coils = positive_coils(outer_kernel)
    coils.currents = np.array([1e5, 2e5])
    coils.target_flux_values = None
    initial_samples = outer_kernel._compute_external_flux(coils)[0, [1, 3]]
    frozen = np.full(2, np.mean(initial_samples))
    result = outer_kernel.solve_free_boundary(
        coils, max_outer_iter=2, tol=0.0, optimize_shape=True, tikhonov_alpha=0.0
    )
    assert result["accepted_steps"] == 1
    np.testing.assert_array_equal(result["shape_optimization"]["target_flux"], frozen)
    assert not np.array_equal(coils.currents, [1e5, 2e5])
    assert coils.target_flux_values is None


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_initial_runtime_failure_restores_uncalculated_fields(
    outer_kernel: FusionKernel, failure: str
) -> None:
    """Rollback actual invalid runtime configurations without solver doubles.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Real kernel whose magnetic fields have not yet been allocated.
    failure : str
        Invalid GS threshold or nonfinite relaxation negative control.
    """
    coils = positive_coils(outer_kernel)
    coils.currents = np.array([1e5, 2e5])
    before_psi = outer_kernel.Psi.copy()
    before_source = outer_kernel.J_phi.copy()
    if failure == "exception":
        outer_kernel.cfg["solver"]["gs_residual_threshold"] = 0.0
    else:
        outer_kernel.cfg["solver"]["max_iterations"] = 1
        outer_kernel.cfg["solver"]["relaxation_factor"] = float("nan")
    result = outer_kernel.solve_free_boundary(coils, max_outer_iter=2)
    assert result["outer_iterations"] == 1
    assert result["outer_outcome"] == "initial_solve_failed"
    assert result["state_origin"] == "pre_call_unsolved"
    assert result["inner_status"] == "failed"
    assert result["inner_result"] is None
    np.testing.assert_array_equal(outer_kernel.Psi, before_psi)
    np.testing.assert_array_equal(outer_kernel.J_phi, before_source)
    np.testing.assert_array_equal(coils.currents, [1e5, 2e5])
    assert not hasattr(outer_kernel, "B_R")
    assert not hasattr(outer_kernel, "B_Z")


def test_fixed_current_converged_executions_stop_at_tolerance(
    outer_kernel: FusionKernel,
) -> None:
    """Use caller tolerance on actual repeated fixed-current equilibria.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Real SOR kernel with nonzero imposed coil flux.
    """
    coils = positive_coils(outer_kernel)
    coils.currents = np.array([1e5, 2e5])
    result = outer_kernel.solve_free_boundary(coils, max_outer_iter=3, tol=1e-10)
    assert result["outer_iterations"] == 2
    assert result["outer_outcome"] == "outer_stationary"
    assert result["accepted_steps"] == 1
    assert result["inner_status"] == "converged"
    assert result["final_diff"] < 1e-10
    assert result["canonical_admission"] == "not_evaluated"
    np.testing.assert_array_equal(coils.currents, [1e5, 2e5])
    assert independent_wall_error(outer_kernel, coils) < 1e-15


@pytest.mark.parametrize("raise_invalid", [False, True])
def test_unrepresentable_subproblem_preserves_solved_state(
    outer_kernel: FusionKernel, raise_invalid: bool
) -> None:
    """Keep actual initial fields when finite targets overflow the current fit.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Real zero-source equilibrium kernel for a numerical negative control.
    raise_invalid : bool
        Raise actual NumPy invalid-operation errors instead of warning.
    """
    coils = positive_coils(outer_kernel)
    coils.target_flux_values = np.full(2, 1e308)
    warning_context = nullcontext() if raise_invalid else pytest.warns(RuntimeWarning)
    with warning_context, np.errstate(invalid="raise" if raise_invalid else "warn"):
        result = outer_kernel.solve_free_boundary(
            coils, max_outer_iter=2, optimize_shape=True, tikhonov_alpha=0.0
        )
    assert result["outer_outcome"] == "subproblem_failed"
    assert result["outer_iterations"] == 1
    assert result["accepted_steps"] == 0
    assert result["trial_log"] == []
    np.testing.assert_array_equal(coils.currents, np.zeros(2))
    np.testing.assert_array_equal(outer_kernel.Psi, np.zeros((5, 5)))
    assert result["shape_optimization"]["flux_rmse"] == pytest.approx(1e308)
    assert result["shape_optimization"]["flux_relative_rmse"] == pytest.approx(1.0)
    assert result["canonical_admission"] == "not_evaluated"


@pytest.mark.parametrize("invalid", ["alpha", "empty", "limits", "external"])
def test_invalid_requests_leave_fields_and_currents_untouched(
    outer_kernel: FusionKernel, invalid: str
) -> None:
    """Refuse invalid numerical requests before executing an equilibrium.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Actual kernel with unallocated magnetic fields.
    invalid : str
        Invalid regularization, empty geometry, infeasible current or coil coordinate.
    """
    coils = positive_coils(outer_kernel)
    alpha = 0.0
    if invalid == "alpha":
        alpha = float("nan")
    elif invalid == "empty":
        coils.positions = []
        coils.currents = np.zeros(0)
    elif invalid == "limits":
        coils.currents[0] = 3e6
    else:
        coils.positions[0] = (float("nan"), 0.0)
    before = outer_kernel.Psi.copy()
    currents = coils.currents.copy()
    with pytest.raises(ValueError):
        outer_kernel.solve_free_boundary(coils, tikhonov_alpha=alpha)
    np.testing.assert_array_equal(outer_kernel.Psi, before)
    np.testing.assert_array_equal(coils.currents, currents)
    assert not hasattr(outer_kernel, "B_R")


def test_initial_fixed_current_stationarity_needs_no_extra_solve(
    outer_kernel: FusionKernel,
) -> None:
    """Stop on an actual unchanged initial solve at the caller's tolerance.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Zero-field zero-source kernel with zero initial requested currents.
    """
    coils = positive_coils(outer_kernel)
    result = outer_kernel.solve_free_boundary(coils, max_outer_iter=3, tol=1e-10)
    assert result["outer_iterations"] == 1
    assert result["outer_outcome"] == "outer_stationary"
    assert result["accepted_steps"] == 0
    assert result["final_diff"] == 0.0
    assert result["trial_log"] == []
    assert result["canonical_admission"] == "not_evaluated"


@pytest.mark.parametrize("failure,budget", [("exception", 4), ("nonfinite", 2)])
def test_failed_fixed_current_execution_restores_last_finite_fields(
    outer_kernel: FusionKernel, tmp_path: Path, failure: str, budget: int
) -> None:
    """Keep the last actual finite diagnostic after runaway relaxation fails.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Actual kernel for an explicitly invalid relaxation negative control.
    tmp_path : Path
        Contains the same real configuration for the independent replay.
    failure : str
        Raise overflow errors or return nonfinite fields from the real solver.
    budget : int
        Two or four executions, fixed by the binary64 overflow negative control.
    """
    control = FusionKernel(tmp_path / "equilibrium.json")
    relaxation = 1e100 if failure == "exception" else 1e200
    for kernel in (outer_kernel, control):
        kernel.cfg["solver"]["max_iterations"] = 1
        kernel.cfg["solver"]["relaxation_factor"] = relaxation
    coils = positive_coils(outer_kernel)
    coils.currents = np.array([1e5, 2e5])
    previous = positive_coils(outer_kernel)
    previous.currents = coils.currents.copy()
    control.solve_free_boundary(previous, max_outer_iter=budget - 1, tol=0.0)
    warning_context = pytest.warns(RuntimeWarning) if failure == "nonfinite" else nullcontext()
    with warning_context, np.errstate(over="raise" if failure == "exception" else "warn"):
        result = outer_kernel.solve_free_boundary(coils, max_outer_iter=budget, tol=0.0)
    assert result["outer_iterations"] == budget
    assert result["inner_status"] == "not_converged"
    assert result["accepted_steps"] == 0
    assert result["trial_log"][-1]["reason"] == (
        "inner_failed" if failure == "exception" else "nonfinite_fields"
    )
    for name in ("Psi", "J_phi", "B_R", "B_Z"):
        np.testing.assert_array_equal(getattr(outer_kernel, name), getattr(control, name))
        assert np.isfinite(getattr(outer_kernel, name)).all()
    np.testing.assert_array_equal(coils.currents, previous.currents)
    assert result["canonical_admission"] == "not_evaluated"


def test_overflowed_external_grid_refused_before_mutation(outer_kernel: FusionKernel) -> None:
    """Refuse an unrepresentable ampere-turn product before imposing a wall.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Actual kernel with finite input coordinates and initial fields.
    """
    coils = positive_coils(outer_kernel)
    coils.currents = np.array([1e308, 0.0])
    coils.current_limits = None
    coils.turns = [10**20, 1]
    before = outer_kernel.Psi.copy()
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="external coil grid"):
        outer_kernel.solve_free_boundary(coils)
    np.testing.assert_array_equal(outer_kernel.Psi, before)
    np.testing.assert_array_equal(coils.currents, [1e308, 0.0])
    assert not hasattr(outer_kernel, "B_R")


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_invalid_relaxation_trial_restores_real_initial_state(
    outer_kernel: FusionKernel, failure: str
) -> None:
    """Reject actual failed trials after the zero-source initial solve succeeds.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Real kernel for an explicitly invalid relaxation negative control.
    failure : str
        Raise overflow errors or return nonfinite fields during each trial.
    """
    outer_kernel.cfg["solver"]["max_iterations"] = 2
    outer_kernel.cfg["solver"]["relaxation_factor"] = 1e200
    coils = positive_coils(outer_kernel)
    warning_context = pytest.warns(RuntimeWarning) if failure == "nonfinite" else nullcontext()
    with warning_context, np.errstate(over="raise" if failure == "exception" else "warn"):
        result = outer_kernel.solve_free_boundary(
            coils, max_outer_iter=4, tol=0.0, optimize_shape=True, tikhonov_alpha=0.0
        )
    assert result["outer_iterations"] == 4
    assert result["outer_outcome"] == "no_acceptable_step"
    assert result["inner_status"] == "converged"
    assert result["accepted_steps"] == 0
    assert [row["tau"] for row in result["trial_log"]] == [1.0, 0.5, 0.25]
    assert all(
        row["reason"] == ("inner_failed" if failure == "exception" else "nonfinite_fields")
        for row in result["trial_log"]
    )
    for name in ("Psi", "J_phi", "B_R", "B_Z"):
        np.testing.assert_array_equal(getattr(outer_kernel, name), np.zeros((5, 5)))
    np.testing.assert_array_equal(coils.currents, np.zeros(2))
    assert result["canonical_admission"] == "not_evaluated"


def test_nonzero_subnormal_target_keeps_defined_relative_residual(
    outer_kernel: FusionKernel,
) -> None:
    """Keep a nonzero target RMS when many smallest binary64 values are sampled.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Real zero-source kernel with finite repeated wall observations.
    """
    coils = positive_coils(outer_kernel)
    coils.target_flux_points = np.tile([[5.0, -4.0]], (100, 1))
    smallest = float(np.nextafter(0.0, 1.0))
    coils.target_flux_values = np.full(100, smallest)
    result = outer_kernel.solve_free_boundary(
        coils, max_outer_iter=1, optimize_shape=True, tikhonov_alpha=0.0
    )
    assert result["shape_optimization"]["flux_rmse"] == smallest
    assert result["shape_optimization"]["flux_relative_rmse"] == 1.0
    assert result["accepted_steps"] == 0
    assert result["canonical_admission"] == "not_evaluated"


def test_unrepresentable_returned_residual_remains_explicit(outer_kernel: FusionKernel) -> None:
    """Report infinite discrepancy when two finite cached values cannot be subtracted.

    Parameters
    ----------
    outer_kernel : FusionKernel
        Real kernel for a pre-call cache and runtime-failure negative control.
    """
    outer_kernel.Psi[:] = -1e308
    outer_kernel.cfg["solver"]["gs_residual_threshold"] = 0.0
    coils = positive_coils(outer_kernel)
    coils.target_flux_values = np.full(2, 1e308)
    with pytest.warns(RuntimeWarning):
        result = outer_kernel.solve_free_boundary(coils, max_outer_iter=1, optimize_shape=True)
    assert result["state_origin"] == "pre_call_unsolved"
    assert result["inner_status"] == "failed"
    assert result["shape_optimization"]["flux_rmse"] == float("inf")
    assert result["shape_optimization"]["flux_relative_rmse"] == float("inf")
    assert result["canonical_admission"] == "not_evaluated"
