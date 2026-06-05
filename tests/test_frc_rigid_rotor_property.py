#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Property Tests
"""Property-based invariants for the accepted no-rotation FRC contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_fusion.core.frc_rigid_rotor import (
    ELEMENTARY_CHARGE_C,
    MU_0,
    RigidRotorFRCInputs,
    solve_frc_equilibrium,
)

FloatArray: TypeAlias = NDArray[np.float64]
_F = TypeVar("_F", bound=Callable[..., object])
_TYPED_GIVEN = cast(Callable[..., Callable[[_F], _F]], given)
_TYPED_SETTINGS = cast(Callable[..., Callable[[_F], _F]], settings)


def _pressure_matched_density_m3(t_i_ev: float, t_e_ev: float, b_ext: float) -> float:
    return float(b_ext**2 / (2.0 * MU_0) / ((t_i_ev + t_e_ev) * ELEMENTARY_CHARGE_C))


def _inputs(delta: float, b_ext: float, r_s: float) -> RigidRotorFRCInputs:
    t_i_ev = 10_000.0
    t_e_ev = 5_000.0
    return RigidRotorFRCInputs(
        n0=_pressure_matched_density_m3(t_i_ev, t_e_ev, b_ext),
        T_i_eV=t_i_ev,
        T_e_eV=t_e_ev,
        theta_dot=0.0,
        R_s=r_s,
        B_ext=b_ext,
        delta=delta,
    )


def _rho(grid_points: int, r_s: float) -> FloatArray:
    return np.linspace(0.0, 2.0 * r_s, grid_points, dtype=np.float64)


_frc_delta = st.floats(min_value=0.010, max_value=0.045, allow_nan=False, allow_infinity=False)
_frc_b_ext = st.floats(min_value=2.5, max_value=9.0, allow_nan=False, allow_infinity=False)
_frc_r_s = st.floats(min_value=0.14, max_value=0.32, allow_nan=False, allow_infinity=False)
_frc_grid = st.integers(min_value=65, max_value=257)


@_TYPED_GIVEN(delta=_frc_delta, b_ext=_frc_b_ext, r_s=_frc_r_s, grid_points=_frc_grid)
@_TYPED_SETTINGS(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_no_rotation_pressure_decreases_outward_from_null(
    delta: float, b_ext: float, r_s: float, grid_points: int
) -> None:
    """The accepted pressure-balance profile peaks at the null and decreases outward."""
    inputs = _inputs(delta=delta, b_ext=b_ext, r_s=r_s)
    state = solve_frc_equilibrium(inputs, _rho(grid_points, r_s))

    outer_pressure = state.p[state.rho >= state.R_null]
    assert outer_pressure.size >= 2
    pressure_scale = max(float(np.max(np.abs(outer_pressure))), 1.0)
    assert np.all(np.diff(outer_pressure) <= pressure_scale * 1.0e-12)
    assert float(np.max(state.p)) == pytest.approx(state.peak_pressure_pa, rel=1.0e-12)
    assert state.beta_peak <= 1.0 + 1.0e-12
    assert state.separatrix_energy_closure_relative_error <= 1.0e-12
