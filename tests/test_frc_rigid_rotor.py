# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Tests
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.integrate import trapezoid

from scpn_fusion.core import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import (
    MU_0,
    ampere_residual,
    force_balance_residual,
    ion_gyroradius_m,
    null_radius,
    s_parameter,
    validate_equilibrium,
)

FloatArray: TypeAlias = NDArray[np.float64]


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


def test_inputs_are_immutable() -> None:
    inputs = _inputs()
    with pytest.raises(FrozenInstanceError):
        inputs.__setattr__("R_s", 0.3)


def test_no_rotation_limit_matches_steinhauer_field() -> None:
    for n_points in [32, 64, 128, 256]:
        inputs = _inputs(delta=0.018)
        rho: FloatArray = np.linspace(0.0, 0.35, n_points)

        state = solve_frc_equilibrium(inputs, rho)

        expected = -inputs.B_ext * np.tanh((rho**2 - inputs.R_s**2) / (2.0 * inputs.R_s * inputs.delta))
        assert np.allclose(state.B_z, expected, rtol=0.0, atol=1e-14)
        assert state.converged is True
        assert state.residual <= 1e-14


def test_null_radius_and_pressure_peak_track_separatrix() -> None:
    inputs = _inputs(delta=0.015)
    rho = np.linspace(0.0, 0.4, 401)

    state = solve_frc_equilibrium(inputs, rho)
    report = validate_equilibrium(state)

    assert abs(null_radius(state) - inputs.R_s) < 2.5e-4
    assert abs(state.R_null - inputs.R_s) < 2.5e-4
    assert abs(rho[np.argmax(state.p)] - inputs.R_s) < 1.1e-3
    assert report.passed


def test_default_delta_uses_thermal_ion_gyroradius() -> None:
    inputs = _inputs(delta=None)
    rho = np.linspace(0.0, 0.4, 129)

    state = solve_frc_equilibrium(inputs, rho)

    expected_delta = ion_gyroradius_m(inputs.T_i_eV, inputs.B_ext)
    assert np.isclose(state.delta, expected_delta)
    assert s_parameter(state) > 0.0


def test_s_parameter_matches_steinhauer_integral_definition() -> None:
    inputs = _inputs(delta=0.02)
    rho: FloatArray = np.linspace(0.0, 0.4, 1025)

    state = solve_frc_equilibrium(inputs, rho)

    ion_mass_kg = 2.014 * 1.66053906660e-27
    thermal_momentum = np.sqrt(2.0 * ion_mass_kg * inputs.T_i_eV * 1.602176634e-19)
    mask = rho <= inputs.R_s
    r_clip = rho[mask]
    b_clip = state.B_z[mask]
    if r_clip[-1] < inputs.R_s:
        r_clip = np.append(r_clip, inputs.R_s)
        b_clip = np.append(b_clip, np.interp(inputs.R_s, rho, state.B_z))
    integrand = r_clip * 1.602176634e-19 * np.abs(b_clip) / thermal_momentum
    expected = trapezoid(integrand, r_clip) / inputs.R_s
    assert s_parameter(state) == pytest.approx(float(expected), rel=1.0e-12)


def test_s_parameter_increases_with_axial_field_strength() -> None:
    rho: FloatArray = np.linspace(0.0, 0.4, 513)
    low_field = solve_frc_equilibrium(
        RigidRotorFRCInputs(**{**_inputs().__dict__, "B_ext": 3.0}),
        rho,
    )
    high_field = solve_frc_equilibrium(
        RigidRotorFRCInputs(**{**_inputs().__dict__, "B_ext": 7.5}),
        rho,
    )

    assert high_field.s_parameter > low_field.s_parameter


def test_no_rotation_scalar_diagnostics_converge_with_grid_refinement() -> None:
    inputs = _inputs(delta=0.02)
    reference = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 4097))
    coarse = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 64))
    medium = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 256))
    fine = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 1024))

    for metric in (
        "R_null",
        "s_parameter",
        "energy_J",
        "pressure_balance_ratio",
    ):
        reference_value = float(getattr(reference, metric))
        coarse_error = abs(float(getattr(coarse, metric)) - reference_value)
        medium_error = abs(float(getattr(medium, metric)) - reference_value)
        fine_error = abs(float(getattr(fine, metric)) - reference_value)
        assert medium_error < coarse_error
        assert fine_error < medium_error


def test_input_validation_rejects_bad_grid_and_rotating_bvp() -> None:
    inputs = _inputs()

    with pytest.raises(ValueError, match="strictly increasing"):
        solve_frc_equilibrium(inputs, np.array([0.0, 0.1, 0.1, 0.3]))

    with pytest.raises(ValueError, match="magnetic axis"):
        solve_frc_equilibrium(inputs, np.linspace(0.01, 0.4, 32))

    with pytest.raises(ValueError, match="separatrix"):
        solve_frc_equilibrium(inputs, np.linspace(0.0, 0.1, 8))

    with pytest.raises(NotImplementedError, match="rotating"):
        solve_frc_equilibrium(_inputs(theta_dot=1.0), np.linspace(0.0, 0.4, 32))


def test_energy_and_pressure_balance_are_finite_positive_diagnostics() -> None:
    inputs = _inputs(delta=0.025)
    rho = np.linspace(0.0, 0.45, 257)

    state = solve_frc_equilibrium(inputs, rho)

    assert state.energy_J > 0.0
    assert state.pressure_balance_ratio > 0.0
    assert state.J_theta.shape == rho.shape
    assert state.ampere_residual.shape == rho.shape
    assert state.peak_j_theta_A_m2 > 0.0
    assert state.ampere_residual_linf <= 1.0e-12
    assert state.ampere_residual_l2 <= 1.0e-12
    assert state.force_balance_residual.shape == rho.shape
    assert state.force_balance_residual_linf >= 0.0
    assert state.force_balance_residual_l2 >= 0.0
    assert np.all(np.isfinite(state.psi))
    assert np.all(state.p > 0.0)


def test_toroidal_current_density_closes_ampere_law() -> None:
    inputs = _inputs(delta=0.02)
    rho: FloatArray = np.linspace(0.0, 0.4, 401)

    state = solve_frc_equilibrium(inputs, rho)

    d_bz_dr = np.gradient(state.B_z, state.rho, edge_order=2)
    dp_dr = np.gradient(state.p, state.rho, edge_order=2)
    expected_j_theta = -d_bz_dr / MU_0
    expected_force_residual = dp_dr - state.J_theta * state.B_z

    np.testing.assert_allclose(state.J_theta, expected_j_theta, rtol=1.0e-12, atol=1.0e-6)
    np.testing.assert_allclose(
        ampere_residual(state),
        MU_0 * state.J_theta + d_bz_dr,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        force_balance_residual(state),
        expected_force_residual,
        rtol=1.0e-12,
        atol=1.0e-4,
    )


def test_force_balance_residual_is_explicit_diagnostic_gate() -> None:
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)

    state = solve_frc_equilibrium(inputs, rho)
    residual = force_balance_residual(state)
    diagnostic_report = validate_equilibrium(state)
    corrupted_ampere = replace(state, ampere_residual_linf=1.0e-6)
    ampere_fail_report = validate_equilibrium(
        corrupted_ampere,
        ampere_tolerance=1.0e-9,
    )
    strict_report = validate_equilibrium(state, force_balance_tolerance=1.0e-12)
    loose_report = validate_equilibrium(
        state,
        force_balance_tolerance=state.force_balance_residual_linf * 1.01,
    )

    assert residual.shape == rho.shape
    assert np.all(np.isfinite(residual))
    assert diagnostic_report.ampere_closure_passed is True
    assert diagnostic_report.passed
    assert ampere_fail_report.ampere_closure_passed is False
    assert ampere_fail_report.passed is False
    assert strict_report.force_balance_passed is False
    assert strict_report.passed is False
    assert loose_report.force_balance_passed is True
    assert loose_report.passed is True
