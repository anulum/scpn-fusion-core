# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Tests
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scpn_fusion.core import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import (
    force_balance_residual,
    ion_gyroradius_m,
    null_radius,
    s_parameter,
    validate_equilibrium,
)


def _inputs(delta: float | None = 0.02, theta_dot: float = 0.0) -> RigidRotorFRCInputs:
    return RigidRotorFRCInputs(
        n0=2.0e20,
        T_i_eV=10_000.0,
        T_e_eV=5_000.0,
        theta_dot=theta_dot,
        R_s=0.20,
        B_ext=5.0,
        delta=delta,
    )


def test_inputs_are_immutable():
    inputs = _inputs()
    with pytest.raises(FrozenInstanceError):
        inputs.__setattr__("R_s", 0.3)


@pytest.mark.parametrize("n_points", [32, 64, 128, 256])
def test_no_rotation_limit_matches_steinhauer_field(n_points: int):
    inputs = _inputs(delta=0.018)
    rho = np.linspace(0.0, 0.35, n_points)

    state = solve_frc_equilibrium(inputs, rho)

    expected = -inputs.B_ext * np.tanh((rho**2 - inputs.R_s**2) / (2.0 * inputs.R_s * inputs.delta))
    assert np.allclose(state.B_z, expected, rtol=0.0, atol=1e-14)
    assert state.converged is True
    assert state.residual <= 1e-14


def test_null_radius_and_pressure_peak_track_separatrix():
    inputs = _inputs(delta=0.015)
    rho = np.linspace(0.0, 0.4, 401)

    state = solve_frc_equilibrium(inputs, rho)
    report = validate_equilibrium(state)

    assert abs(null_radius(state) - inputs.R_s) < 2.5e-4
    assert abs(state.R_null - inputs.R_s) < 2.5e-4
    assert abs(rho[np.argmax(state.p)] - inputs.R_s) < 1.1e-3
    assert report.passed


def test_default_delta_uses_thermal_ion_gyroradius():
    inputs = _inputs(delta=None)
    rho = np.linspace(0.0, 0.4, 129)

    state = solve_frc_equilibrium(inputs, rho)

    expected_delta = ion_gyroradius_m(inputs.T_i_eV, inputs.B_ext)
    assert np.isclose(state.delta, expected_delta)
    assert np.isclose(s_parameter(state), inputs.R_s / (2.0 * expected_delta))


def test_input_validation_rejects_bad_grid_and_rotating_bvp():
    inputs = _inputs()

    with pytest.raises(ValueError, match="strictly increasing"):
        solve_frc_equilibrium(inputs, np.array([0.0, 0.1, 0.1, 0.3]))

    with pytest.raises(ValueError, match="separatrix"):
        solve_frc_equilibrium(inputs, np.linspace(0.0, 0.1, 8))

    with pytest.raises(NotImplementedError, match="rotating"):
        solve_frc_equilibrium(_inputs(theta_dot=1.0), np.linspace(0.0, 0.4, 32))


def test_energy_and_pressure_balance_are_finite_positive_diagnostics():
    inputs = _inputs(delta=0.025)
    rho = np.linspace(0.0, 0.45, 257)

    state = solve_frc_equilibrium(inputs, rho)

    assert state.energy_J > 0.0
    assert state.pressure_balance_ratio > 0.0
    assert state.force_balance_residual.shape == rho.shape
    assert state.force_balance_residual_linf >= 0.0
    assert state.force_balance_residual_l2 >= 0.0
    assert np.all(np.isfinite(state.psi))
    assert np.all(state.p > 0.0)


def test_force_balance_residual_is_explicit_diagnostic_gate():
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)

    state = solve_frc_equilibrium(inputs, rho)
    residual = force_balance_residual(state)
    diagnostic_report = validate_equilibrium(state)
    strict_report = validate_equilibrium(state, force_balance_tolerance=1.0e-12)
    loose_report = validate_equilibrium(
        state,
        force_balance_tolerance=state.force_balance_residual_linf * 1.01,
    )

    assert residual.shape == rho.shape
    assert np.all(np.isfinite(residual))
    assert diagnostic_report.passed
    assert strict_report.force_balance_passed is False
    assert strict_report.passed is False
    assert loose_report.force_balance_passed is True
    assert loose_report.passed is True
