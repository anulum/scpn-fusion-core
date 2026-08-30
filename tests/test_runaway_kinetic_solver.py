# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Full Runaway Kinetic Solver Tests
"""Public end-to-end evolution tests without projected axes."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.runaway_kinetic_coefficients import (
    RunawayKineticCoefficients,
)
from scpn_fusion.core.runaway_kinetic_diagnostics import (
    interval_residual,
    weighted_relative_l2,
)
from scpn_fusion.core.runaway_kinetic_grid import RunawayKineticGrid
from scpn_fusion.core.runaway_kinetic_operator import (
    RunawayKineticGeometry,
    RunawayKineticOperator,
)
from scpn_fusion.core.runaway_kinetic_solver import RunawayKineticSolver


def _source_operator(source: float) -> RunawayKineticOperator:
    grid = RunawayKineticGrid(
        radius_faces_m=np.array([0.0, 0.1, 0.2]),
        pitch_faces=np.array([-1.0, 0.0, 1.0]),
        momentum_faces_mc=np.array([0.02, 0.5, 1.0]),
    )
    cell = np.full(grid.shape, source)
    radial = np.zeros((grid.nr + 1, grid.nxi, grid.np))
    momentum = np.zeros((grid.nr, grid.nxi, grid.np + 1))
    pitch = np.zeros((grid.nr, grid.nxi + 1, grid.np))
    geometry = RunawayKineticGeometry.cylindrical(grid)
    density_source = np.sum(
        cell * geometry.density_cell_measure,
        axis=(1, 2),
    )
    coefficients = RunawayKineticCoefficients.checked(
        grid,
        radial_advection=radial,
        momentum_electric_advection=momentum,
        momentum_collision_advection=momentum,
        momentum_synchrotron_advection=momentum,
        momentum_bremsstrahlung_advection=momentum,
        pitch_electric_advection=pitch,
        pitch_synchrotron_advection=pitch,
        radial_diffusion=radial,
        momentum_diffusion=momentum,
        pitch_diffusion=pitch,
        momentum_pitch_diffusion=momentum,
        pitch_momentum_diffusion=pitch,
        avalanche_source_kernel=np.zeros(grid.shape),
        total_electron_density_m3=np.ones(grid.nr),
        total_density_avalanche_rate_s_inv=np.zeros(grid.nr),
        total_density_external_source_m3_s=density_source,
        external_source=cell,
    )
    return RunawayKineticOperator(grid, coefficients, geometry=geometry)


def test_solver_returns_full_state_all_budgets_and_moments() -> None:
    operator = _source_operator(2.0)
    solver = RunawayKineticSolver(operator, maximum_step_s=0.01)
    initial = np.ones(operator.grid.shape)
    times = np.array([0.0, 0.05, 0.1])

    result = solver.solve(initial, times)

    assert result.distribution.shape == (3, *operator.grid.shape)
    assert np.allclose(result.distribution[-1], 1.2, rtol=0.0, atol=1.0e-14)
    assert np.all(result.external_source == 2.0)
    assert np.allclose(result.total_tendency, 2.0)
    assert result.radial_transport.shape == result.distribution.shape
    assert result.pitch_scattering.shape == result.distribution.shape
    assert result.synchrotron_loss.shape == result.distribution.shape
    assert result.bremsstrahlung_loss.shape == result.distribution.shape
    assert result.avalanche_generation.shape == result.distribution.shape
    assert result.runaway_density_m3.shape == (3, operator.grid.nr)
    assert result.runaway_density_tendency_m3_s.shape == (
        3,
        operator.grid.nr,
    )
    assert result.runaway_density_radial_transport_m3_s.shape == (
        3,
        operator.grid.nr,
    )
    assert result.runaway_density_avalanche_generation_m3_s.shape == (
        3,
        operator.grid.nr,
    )
    assert result.runaway_density_external_source_m3_s.shape == (
        3,
        operator.grid.nr,
    )
    assert result.moments.density_m3.shape == (3, operator.grid.nr)
    assert result.moments.current_density_a_m2.shape == (3, operator.grid.nr)
    assert result.moments.kinetic_energy_density_j_m3.shape == (
        3,
        operator.grid.nr,
    )
    assert result.internal_steps == 10
    assert result.minimum_distribution == 1.0
    assert np.allclose(result.runaway_density_m3, result.moments.density_m3)
    assert (
        interval_residual(
            result.distribution[0],
            result.distribution[1],
            result.total_tendency[1],
            operator.geometry,
            0.05,
        )
        < 1.0e-14
    )


def test_solver_is_available_on_public_core_surface() -> None:
    from scpn_fusion import core

    assert core.RunawayKineticSolver is RunawayKineticSolver


def test_solver_couples_kinetic_avalanche_to_total_runaway_density() -> None:
    operator = _source_operator(0.0)
    grid = operator.grid
    base = operator.coefficients
    coefficients = RunawayKineticCoefficients.checked(
        grid,
        radial_advection=base.radial_advection,
        momentum_electric_advection=base.momentum_electric_advection,
        momentum_collision_advection=base.momentum_collision_advection,
        momentum_synchrotron_advection=base.momentum_synchrotron_advection,
        momentum_bremsstrahlung_advection=(base.momentum_bremsstrahlung_advection),
        pitch_electric_advection=base.pitch_electric_advection,
        pitch_synchrotron_advection=base.pitch_synchrotron_advection,
        radial_diffusion=base.radial_diffusion,
        momentum_diffusion=base.momentum_diffusion,
        pitch_diffusion=base.pitch_diffusion,
        momentum_pitch_diffusion=base.momentum_pitch_diffusion,
        pitch_momentum_diffusion=base.pitch_momentum_diffusion,
        avalanche_source_kernel=np.full(grid.shape, 1.0e-3),
        total_electron_density_m3=np.full(grid.nr, 2.0),
        total_density_avalanche_rate_s_inv=np.full(grid.nr, 2.0e-3),
        total_density_external_source_m3_s=np.zeros(grid.nr),
        external_source=base.external_source,
    )
    solver = RunawayKineticSolver(RunawayKineticOperator(grid, coefficients), maximum_step_s=0.01)
    initial_density = np.full(grid.nr, 3.0)
    result = solver.solve(
        np.ones(grid.shape),
        np.array([0.0, 0.1]),
        initial_runaway_density_m3=initial_density,
    )

    assert np.all(result.runaway_density_m3[-1] > initial_density)
    assert np.all(result.distribution[-1] > 1.0)
    assert np.all(result.avalanche_generation[-1] > 0.0)


def test_solver_rejects_nonzero_time_origin() -> None:
    solver = RunawayKineticSolver(_source_operator(0.0), maximum_step_s=0.01)

    try:
        solver.solve(np.ones(solver.operator.grid.shape), np.array([0.1, 0.2]))
    except ValueError as error:
        assert "start exactly at zero" in str(error)
    else:
        raise AssertionError("nonzero time origin was accepted")


def test_solver_rejects_unknown_backend_without_fallback() -> None:
    solver = RunawayKineticSolver(_source_operator(0.0), maximum_step_s=0.01)
    with pytest.raises(ValueError, match="exactly 'numpy' or 'rust'"):
        solver.solve(
            np.ones(solver.operator.grid.shape),
            np.array([0.0, 0.1]),
            backend="julia",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("maximum_step", "negativity_tolerance", "message"),
    [
        (0.0, 0.0, "maximum_step_s"),
        (np.nan, 0.0, "maximum_step_s"),
        (0.1, -1.0, "negativity_tolerance"),
    ],
)
def test_solver_rejects_invalid_numerical_contract(
    maximum_step: float,
    negativity_tolerance: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RunawayKineticSolver(
            _source_operator(0.0),
            maximum_step_s=maximum_step,
            negativity_tolerance=negativity_tolerance,
        )


@pytest.mark.parametrize(
    ("times", "message"),
    [
        (np.array([0.0]), "at least two"),
        (np.array([0.0, 0.2, 0.1]), "strictly increasing"),
    ],
)
def test_solver_rejects_incomplete_or_reversed_time_grid(
    times: np.ndarray[tuple[int], np.dtype[np.float64]], message: str
) -> None:
    solver = RunawayKineticSolver(_source_operator(0.0), maximum_step_s=0.01)
    with pytest.raises(ValueError, match=message):
        solver.solve(np.ones(solver.operator.grid.shape), times)


@pytest.mark.parametrize(
    ("density", "message"),
    [
        (np.ones(3), "must have shape"),
        (np.array([1.0, np.nan]), "finite and non-negative"),
        (np.array([1.0, -1.0]), "finite and non-negative"),
    ],
)
def test_solver_rejects_invalid_explicit_runaway_density(
    density: np.ndarray[tuple[int], np.dtype[np.float64]], message: str
) -> None:
    solver = RunawayKineticSolver(_source_operator(0.0), maximum_step_s=0.01)
    with pytest.raises(ValueError, match=message):
        solver.solve(
            np.ones(solver.operator.grid.shape),
            np.array([0.0, 0.1]),
            initial_runaway_density_m3=density,
        )


def test_diagnostics_reject_incomparable_shapes_and_invalid_intervals() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        weighted_relative_l2(np.ones(2), np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="finite and positive"):
        weighted_relative_l2(np.ones(2), np.ones(2), np.ones(2), floor=0.0)

    operator = _source_operator(0.0)
    state = np.ones(operator.grid.shape)
    with pytest.raises(ValueError, match="dt_s must be finite and positive"):
        interval_residual(state, state, state, operator.geometry, 0.0)


def test_solver_enforces_declared_distribution_negativity_tolerance() -> None:
    solver = RunawayKineticSolver(
        _source_operator(-10.0),
        maximum_step_s=0.1,
        negativity_tolerance=0.0,
    )
    with pytest.raises(FloatingPointError, match="negativity tolerance"):
        solver.solve(
            np.ones(solver.operator.grid.shape),
            np.array([0.0, 0.2]),
            initial_runaway_density_m3=np.full(solver.operator.grid.nr, 1.0e6),
        )
