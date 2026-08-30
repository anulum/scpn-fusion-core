# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Runaway Kinetic Coefficient Tests
"""Tests for complete operator coefficient custody."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.runaway_kinetic_coefficients import (
    RunawayKineticCoefficients,
)
from scpn_fusion.core.runaway_kinetic_grid import RunawayKineticGrid


def _grid() -> RunawayKineticGrid:
    return RunawayKineticGrid(
        radius_faces_m=np.linspace(0.0, 0.2, 3),
        pitch_faces=np.linspace(-1.0, 1.0, 4),
        momentum_faces_mc=np.linspace(0.02, 2.0, 5),
    )


def _bundle(
    grid: RunawayKineticGrid,
    *,
    momentum_components: tuple[float, ...] = (0, 0, 0, 0),
    total_density: float = 1.0,
    density_avalanche_rate: float = 0.0,
    density_external_source: float = 0.0,
) -> RunawayKineticCoefficients:
    cell = np.zeros(grid.shape)
    radial = np.zeros((grid.nr + 1, grid.nxi, grid.np))
    momentum = np.zeros((grid.nr, grid.nxi, grid.np + 1))
    pitch = np.zeros((grid.nr, grid.nxi + 1, grid.np))
    components = [np.full_like(momentum, value) for value in momentum_components]
    return RunawayKineticCoefficients.checked(
        grid,
        radial_advection=radial,
        momentum_electric_advection=components[0],
        momentum_collision_advection=components[1],
        momentum_synchrotron_advection=components[2],
        momentum_bremsstrahlung_advection=components[3],
        pitch_electric_advection=pitch,
        pitch_synchrotron_advection=pitch,
        radial_diffusion=radial,
        momentum_diffusion=momentum,
        pitch_diffusion=pitch,
        momentum_pitch_diffusion=momentum,
        pitch_momentum_diffusion=pitch,
        avalanche_source_kernel=cell,
        total_electron_density_m3=np.full(grid.nr, total_density),
        total_density_avalanche_rate_s_inv=np.full(grid.nr, density_avalanche_rate),
        total_density_external_source_m3_s=np.full(grid.nr, density_external_source),
        external_source=cell,
    )


def test_all_momentum_terms_remain_present_in_total() -> None:
    bundle = _bundle(_grid(), momentum_components=(1.0, 2.0, 4.0, 8.0))

    assert np.all(bundle.momentum_advection == 15.0)
    assert not bundle.momentum_advection.flags.writeable


def test_coefficient_contract_rejects_wrong_axis_shape() -> None:
    grid = _grid()
    with pytest.raises(ValueError, match="radial_advection must have shape"):
        cell = np.zeros(grid.shape)
        momentum = np.zeros((grid.nr, grid.nxi, grid.np + 1))
        pitch = np.zeros((grid.nr, grid.nxi + 1, grid.np))
        RunawayKineticCoefficients.checked(
            grid,
            radial_advection=np.zeros(grid.shape),
            momentum_electric_advection=momentum,
            momentum_collision_advection=momentum,
            momentum_synchrotron_advection=momentum,
            momentum_bremsstrahlung_advection=momentum,
            pitch_electric_advection=pitch,
            pitch_synchrotron_advection=pitch,
            radial_diffusion=np.zeros((grid.nr + 1, grid.nxi, grid.np)),
            momentum_diffusion=momentum,
            pitch_diffusion=pitch,
            momentum_pitch_diffusion=momentum,
            pitch_momentum_diffusion=pitch,
            avalanche_source_kernel=cell,
            total_electron_density_m3=np.ones(grid.nr),
            total_density_avalanche_rate_s_inv=np.zeros(grid.nr),
            total_density_external_source_m3_s=np.zeros(grid.nr),
            external_source=cell,
        )


@pytest.mark.parametrize("total_density", [np.nan, -1.0])
def test_coefficient_contract_rejects_invalid_physical_density(
    total_density: float,
) -> None:
    with pytest.raises(ValueError, match="non-finite|non-negative"):
        _bundle(_grid(), total_density=total_density)


@pytest.mark.parametrize("rate", [np.nan, -1.0])
def test_coefficient_contract_rejects_invalid_total_density_avalanche_rate(
    rate: float,
) -> None:
    with pytest.raises(ValueError, match="non-finite|non-negative"):
        _bundle(_grid(), density_avalanche_rate=rate)


def test_total_density_sources_remain_separate_from_kinetic_coefficients() -> None:
    bundle = _bundle(
        _grid(),
        density_avalanche_rate=3.0,
        density_external_source=-2.0,
    )

    assert np.all(bundle.total_density_avalanche_rate_s_inv == 3.0)
    assert np.all(bundle.total_density_external_source_m3_s == -2.0)
    assert np.count_nonzero(bundle.avalanche_source_kernel) == 0
    assert not bundle.total_density_avalanche_rate_s_inv.flags.writeable
    assert not bundle.total_density_external_source_m3_s.flags.writeable
