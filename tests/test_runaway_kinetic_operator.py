# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Conservative Runaway Kinetic Operator Tests
"""Manufactured conservation and component tests for the full operator."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.runaway_kinetic_coefficients import (
    RunawayKineticCoefficients,
)
from scpn_fusion.core.runaway_kinetic_diagnostics import integrated_budget
from scpn_fusion.core.runaway_kinetic_grid import RunawayKineticGrid
from scpn_fusion.core.runaway_kinetic_operator import (
    RunawayKineticGeometry,
    RunawayKineticOperator,
)


def _grid() -> RunawayKineticGrid:
    return RunawayKineticGrid(
        radius_faces_m=np.linspace(0.0, 0.2, 4),
        pitch_faces=np.linspace(-1.0, 1.0, 5),
        momentum_faces_mc=np.linspace(0.02, 2.0, 6),
    )


def _coefficients(
    grid: RunawayKineticGrid,
    *,
    radial_diffusion: float = 0.0,
    pitch_diffusion: float = 0.0,
    avalanche_kernel: float = 0.0,
    density_avalanche_rate: float = 0.0,
    density_external_source: float = 0.0,
    external_source: float = 0.0,
) -> RunawayKineticCoefficients:
    cell = np.full(grid.shape, external_source)
    radial = np.zeros((grid.nr + 1, grid.nxi, grid.np))
    momentum = np.zeros((grid.nr, grid.nxi, grid.np + 1))
    pitch = np.zeros((grid.nr, grid.nxi + 1, grid.np))
    radial_d = np.full_like(radial, radial_diffusion)
    radial_d[0] = 0.0
    pitch_d = np.full_like(pitch, pitch_diffusion)
    pitch_d[:, 0] = 0.0
    pitch_d[:, -1] = 0.0
    return RunawayKineticCoefficients.checked(
        grid,
        radial_advection=radial,
        momentum_electric_advection=momentum,
        momentum_collision_advection=momentum,
        momentum_synchrotron_advection=momentum,
        momentum_bremsstrahlung_advection=momentum,
        pitch_electric_advection=pitch,
        pitch_synchrotron_advection=pitch,
        radial_diffusion=radial_d,
        momentum_diffusion=momentum,
        pitch_diffusion=pitch_d,
        momentum_pitch_diffusion=momentum,
        pitch_momentum_diffusion=pitch,
        avalanche_source_kernel=np.full(grid.shape, avalanche_kernel),
        total_electron_density_m3=np.full(grid.nr, 2.0),
        total_density_avalanche_rate_s_inv=np.full(grid.nr, density_avalanche_rate),
        total_density_external_source_m3_s=np.full(grid.nr, density_external_source),
        external_source=cell,
    )


def test_zero_operator_has_no_hidden_tendency() -> None:
    grid = _grid()
    tendency = RunawayKineticOperator(grid, _coefficients(grid)).evaluate(np.ones(grid.shape))

    assert np.count_nonzero(tendency.total) == 0
    assert np.count_nonzero(tendency.radial_transport) == 0
    assert np.count_nonzero(tendency.pitch_scattering) == 0


def test_pitch_scattering_conserves_particle_measure() -> None:
    grid = _grid()
    operator = RunawayKineticOperator(grid, _coefficients(grid, pitch_diffusion=0.2))
    state = np.broadcast_to((1.0 + grid.pitch**2)[None, :, None], grid.shape)
    tendency = operator.evaluate(state).pitch_scattering

    defect = float(np.sum(tendency * operator.geometry.cell_measure))
    assert abs(defect) < 1.0e-12
    assert np.max(np.abs(tendency)) > 0.0


def test_radial_transport_reports_open_edge_loss() -> None:
    grid = _grid()
    operator = RunawayKineticOperator(grid, _coefficients(grid, radial_diffusion=0.01))
    tendency = operator.evaluate(np.ones(grid.shape)).radial_transport

    assert float(np.sum(tendency * operator.geometry.cell_measure)) < 0.0
    assert np.count_nonzero(tendency[-1]) > 0


def test_avalanche_generation_depends_on_evolved_distribution_moment() -> None:
    grid = _grid()
    operator = RunawayKineticOperator(grid, _coefficients(grid, avalanche_kernel=3.0))
    one = operator.evaluate(np.ones(grid.shape)).avalanche_generation
    two = operator.evaluate(np.full(grid.shape, 2.0)).avalanche_generation

    assert np.all(one > 0.0)
    assert np.allclose(two, 2.0 * one)


def test_total_density_uses_its_integrated_avalanche_and_source_budgets() -> None:
    grid = _grid()
    operator = RunawayKineticOperator(
        grid,
        _coefficients(
            grid,
            avalanche_kernel=3.0,
            density_avalanche_rate=5.0,
            density_external_source=-7.0,
        ),
    )
    explicit_density = np.full(grid.nr, 11.0)
    tendency = operator.evaluate(np.ones(grid.shape), explicit_density)

    assert np.all(tendency.runaway_density_radial_transport_m3_s == 0.0)
    assert np.all(tendency.runaway_density_avalanche_generation_m3_s == 55.0)
    assert np.all(tendency.runaway_density_external_source_m3_s == -7.0)
    assert np.all(tendency.runaway_density_tendency_m3_s == 48.0)
    assert not np.allclose(
        tendency.runaway_density_avalanche_generation_m3_s,
        np.sum(
            tendency.avalanche_generation * operator.geometry.density_cell_measure,
            axis=(1, 2),
        ),
    )


def test_external_source_has_distinct_budget() -> None:
    grid = _grid()
    tendency = RunawayKineticOperator(grid, _coefficients(grid, external_source=7.0)).evaluate(
        np.zeros(grid.shape)
    )

    assert np.all(tendency.external_source == 7.0)
    assert np.array_equal(tendency.total, tendency.external_source)
    budget = integrated_budget(
        tendency, RunawayKineticOperator(grid, _coefficients(grid, external_source=7.0)).geometry
    )
    assert budget.external_source == budget.total


def test_component_budgets_share_total_upwind_interpolation() -> None:
    grid = _grid()
    radial = np.zeros((grid.nr + 1, grid.nxi, grid.np))
    momentum = np.zeros((grid.nr, grid.nxi, grid.np + 1))
    pitch = np.zeros((grid.nr, grid.nxi + 1, grid.np))
    electric = np.full_like(momentum, 2.0)
    collision = np.full_like(momentum, -1.0)
    coefficients = RunawayKineticCoefficients.checked(
        grid,
        radial_advection=radial,
        momentum_electric_advection=electric,
        momentum_collision_advection=collision,
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
        total_density_external_source_m3_s=np.zeros(grid.nr),
        external_source=np.zeros(grid.shape),
    )
    operator = RunawayKineticOperator(grid, coefficients)
    state = np.broadcast_to((1.0 + grid.momentum_mc)[None, None, :], grid.shape)
    tendency = operator.evaluate(state)
    total_coefficients = RunawayKineticCoefficients.checked(
        grid,
        radial_advection=radial,
        momentum_electric_advection=coefficients.momentum_advection,
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
        total_density_external_source_m3_s=np.zeros(grid.nr),
        external_source=np.zeros(grid.shape),
    )
    total_direct = (
        RunawayKineticOperator(grid, total_coefficients).evaluate(state).electric_acceleration
    )

    assert np.allclose(
        tendency.electric_acceleration + tendency.collisional_drag_diffusion,
        total_direct,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_imported_geometry_rejects_shape_sign_and_zero_cell_measure() -> None:
    grid = _grid()
    geometry = RunawayKineticGeometry.cylindrical(grid)
    supplied = {
        "cell_measure": geometry.cell_measure,
        "density_cell_measure": geometry.density_cell_measure,
        "radial_face_measure": geometry.radial_face_measure,
        "momentum_face_measure": geometry.momentum_face_measure,
        "pitch_face_measure": geometry.pitch_face_measure,
    }

    wrong_shape = dict(supplied)
    wrong_shape["radial_face_measure"] = np.zeros(grid.shape)
    with pytest.raises(ValueError, match="radial_face_measure must have shape"):
        RunawayKineticGeometry.checked(grid, **wrong_shape)

    negative = dict(supplied)
    negative["pitch_face_measure"] = -np.ones_like(geometry.pitch_face_measure)
    with pytest.raises(ValueError, match="finite non-negative"):
        RunawayKineticGeometry.checked(grid, **negative)

    zero_cell = dict(supplied)
    zero_cell["cell_measure"] = np.zeros_like(geometry.cell_measure)
    with pytest.raises(ValueError, match="strictly positive"):
        RunawayKineticGeometry.checked(grid, **zero_cell)


def test_single_cell_axes_and_explicit_runaway_density_validation() -> None:
    grid = RunawayKineticGrid(
        radius_faces_m=np.array([0.0, 0.2]),
        pitch_faces=np.array([-1.0, 1.0]),
        momentum_faces_mc=np.array([0.02, 2.0]),
    )
    operator = RunawayKineticOperator(grid, _coefficients(grid))
    tendency = operator.evaluate(np.ones(grid.shape))

    assert tendency.total.shape == (1, 1, 1)
    with pytest.raises(ValueError, match="must have shape"):
        operator.evaluate(np.ones(grid.shape), np.ones(2))
    with pytest.raises(ValueError, match="finite and non-negative"):
        operator.evaluate(np.ones(grid.shape), np.array([-1.0]))
