# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Tests
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from runpy import run_path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.integrate import trapezoid

from scpn_fusion.core import (
    RigidRotorFRCInputs,
    psi_normalized_profile as public_psi_normalized_profile,
    solve_frc_equilibrium,
)
from scpn_fusion.core.frc_rigid_rotor import (
    ELEMENTARY_CHARGE_C,
    MU_0,
    ampere_residual,
    beta_profile,
    density_profile,
    force_balance_residual,
    flux_derivative_residual,
    frc_no_rotation_jax_observables,
    ion_gyroradius_m,
    psi_normalized_profile,
    null_radius,
    pressure_balance_residual,
    pressure_gradient_residual,
    s_parameter,
    validate_equilibrium,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _pressure_matched_density_m3(t_i_ev: float, t_e_ev: float, b_ext: float) -> float:
    external_pressure = b_ext**2 / (2.0 * MU_0)
    return float(external_pressure / ((t_i_ev + t_e_ev) * ELEMENTARY_CHARGE_C))


def _inputs(delta: float | None = 0.02, theta_dot: float = 0.0) -> RigidRotorFRCInputs:
    t_i_ev = 10_000.0
    t_e_ev = 5_000.0
    b_ext = 5.0
    return RigidRotorFRCInputs(
        n0=_pressure_matched_density_m3(t_i_ev, t_e_ev, b_ext),
        T_i_eV=t_i_ev,
        T_e_eV=t_e_ev,
        theta_dot=theta_dot,
        R_s=0.20,
        B_ext=b_ext,
        delta=delta,
    )


def _frc_quickstart_namespace() -> dict[str, Any]:
    example_path = (
        Path(__file__).resolve().parents[1] / "examples" / "03_frc_rigid_rotor_quickstart.py"
    )
    return run_path(str(example_path))


def test_inputs_are_immutable() -> None:
    inputs = _inputs()
    with pytest.raises(FrozenInstanceError):
        inputs.__setattr__("R_s", 0.3)


def test_no_rotation_limit_matches_steinhauer_field() -> None:
    for n_points in [32, 64, 128, 256]:
        inputs = _inputs(delta=0.018)
        rho: FloatArray = np.linspace(0.0, 0.35, n_points)

        state = solve_frc_equilibrium(inputs, rho)

        expected = -inputs.B_ext * np.tanh(
            (rho**2 - inputs.R_s**2) / (2.0 * inputs.R_s * inputs.delta)
        )
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
    assert state.target_separatrix_radius_m == inputs.R_s
    assert state.separatrix_radius_error_m < 2.5e-4
    assert state.field_reversal_passed is True
    assert report.target_separatrix_radius_m == inputs.R_s
    assert report.field_reversal_passed is True
    assert report.null_error_m < 2.5e-4
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
    low_b_ext = 3.0
    high_b_ext = 7.5
    low_field = solve_frc_equilibrium(
        RigidRotorFRCInputs(
            **{
                **_inputs().__dict__,
                "B_ext": low_b_ext,
                "n0": _pressure_matched_density_m3(10_000.0, 5_000.0, low_b_ext),
            }
        ),
        rho,
    )
    high_field = solve_frc_equilibrium(
        RigidRotorFRCInputs(
            **{
                **_inputs().__dict__,
                "B_ext": high_b_ext,
                "n0": _pressure_matched_density_m3(10_000.0, 5_000.0, high_b_ext),
            }
        ),
        rho,
    )

    assert high_field.s_parameter > low_field.s_parameter


def test_jax_no_rotation_observables_match_numpy_and_have_finite_gradients() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    x_grid: FloatArray = np.linspace(0.0, 2.0, 401)
    inputs = _inputs(delta=0.02)
    rho = x_grid * inputs.R_s
    numpy_state = solve_frc_equilibrium(inputs, rho)

    jax_state = frc_no_rotation_jax_observables(
        x_grid,
        n0=inputs.n0,
        T_i_eV=inputs.T_i_eV,
        T_e_eV=inputs.T_e_eV,
        R_s=inputs.R_s,
        B_ext=inputs.B_ext,
        delta=inputs.delta,
    )

    np.testing.assert_allclose(np.asarray(jax_state["rho"]), numpy_state.rho, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(jax_state["B_z"]),
        numpy_state.B_z,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax_state["pressure"]),
        numpy_state.p,
        rtol=1.0e-12,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(jax_state["psi_normalized"]),
        numpy_state.psi_normalized,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(jax_state["energy_J"]) == pytest.approx(numpy_state.energy_J, rel=1.0e-12)
    assert float(jax_state["s_parameter"]) == pytest.approx(
        numpy_state.s_parameter,
        rel=1.0e-12,
    )

    def energy_for(field_t: jnp.ndarray, radius_m: jnp.ndarray) -> jnp.ndarray:
        matched_density = field_t**2 / (2.0 * MU_0)
        matched_density = matched_density / ((inputs.T_i_eV + inputs.T_e_eV) * ELEMENTARY_CHARGE_C)
        observables = frc_no_rotation_jax_observables(
            x_grid,
            n0=matched_density,
            T_i_eV=inputs.T_i_eV,
            T_e_eV=inputs.T_e_eV,
            R_s=radius_m,
            B_ext=field_t,
            delta=inputs.delta,
        )
        return cast(jnp.ndarray, observables["energy_J"])

    grad_b_ext, grad_r_s = jax.grad(energy_for, argnums=(0, 1))(
        jnp.asarray(inputs.B_ext),
        jnp.asarray(inputs.R_s),
    )

    assert np.isfinite(float(grad_b_ext))
    assert np.isfinite(float(grad_r_s))
    assert float(grad_b_ext) > 0.0
    assert float(grad_r_s) > 0.0


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
        "pressure_gradient_residual_linf",
        "beta_peak",
        "beta_separatrix_average",
        "particle_line_density_m1",
        "separatrix_pressure_energy_J_m",
        "separatrix_magnetic_deficit_energy_J_m",
        "separatrix_energy_closure_relative_error",
        "separatrix_bz_gradient_T_m",
        "separatrix_current_density_A_m2",
        "separatrix_gradient_relative_error",
        "separatrix_current_density_relative_error",
        "sheet_current_integral_A_m",
        "sheet_current_integral_relative_error",
        "psi_axis_Wb",
        "psi_separatrix_Wb",
        "psi_normalized_axis_error",
        "psi_normalized_separatrix",
        "psi_normalized_separatrix_error",
        "psi_normalized_residual_linf",
    ):
        reference_value = float(getattr(reference, metric))
        coarse_error = abs(float(getattr(coarse, metric)) - reference_value)
        medium_error = abs(float(getattr(medium, metric)) - reference_value)
        fine_error = abs(float(getattr(fine, metric)) - reference_value)
        exact_tolerance = max(abs(reference_value), 1.0) * 1.0e-13
        if medium_error <= exact_tolerance and fine_error <= exact_tolerance:
            continue
        assert medium_error <= coarse_error + exact_tolerance
        assert fine_error <= medium_error + exact_tolerance
        assert medium_error < coarse_error or medium_error <= exact_tolerance
        assert fine_error < medium_error or fine_error <= exact_tolerance


def test_input_validation_rejects_bad_grid_and_rotating_bvp() -> None:
    inputs = _inputs()

    with pytest.raises(ValueError, match="strictly increasing"):
        solve_frc_equilibrium(inputs, np.array([0.0, 0.1, 0.1, 0.3]))

    with pytest.raises(ValueError, match="magnetic axis"):
        solve_frc_equilibrium(inputs, np.linspace(0.01, 0.4, 32))

    with pytest.raises(ValueError, match="separatrix"):
        solve_frc_equilibrium(inputs, np.linspace(0.0, 0.1, 8))

    with pytest.raises(ValueError, match="outside the separatrix"):
        solve_frc_equilibrium(inputs, np.linspace(0.0, inputs.R_s, 8))

    with pytest.raises(NotImplementedError, match="rotating"):
        solve_frc_equilibrium(_inputs(theta_dot=1.0), np.linspace(0.0, 0.4, 32))


def test_energy_and_pressure_balance_are_finite_positive_diagnostics() -> None:
    inputs = _inputs(delta=0.025)
    rho = np.linspace(0.0, 0.45, 257)

    state = solve_frc_equilibrium(inputs, rho)

    assert state.energy_J > 0.0
    assert state.pressure_balance_ratio > 0.0
    assert state.pressure_balance_residual.shape == rho.shape
    assert state.pressure_balance_residual_linf <= 1.0e-12
    assert state.pressure_balance_residual_l2 <= 1.0e-12
    assert state.pressure_gradient_analytic_Pa_m.shape == rho.shape
    assert state.pressure_gradient_residual.shape == rho.shape
    assert state.pressure_gradient_residual_linf <= 2.0e-2
    assert state.pressure_gradient_residual_l2 <= 2.0e-2
    assert state.peak_pressure_pa > 0.0
    assert state.density_m3.shape == rho.shape
    assert state.density_peak_m3 > 0.0
    assert state.input_density_m3 == pytest.approx(inputs.n0)
    assert abs(state.central_density_residual_m3) <= state.input_density_m3 * 1.0e-3
    assert state.central_density_relative_error <= 1.0e-3
    assert state.beta.shape == rho.shape
    assert 0.0 < state.beta_peak <= 1.0 + 1.0e-12
    assert 0.0 < state.beta_separatrix_average <= state.beta_peak
    assert state.particle_line_density_m1 > 0.0
    assert state.separatrix_pressure_energy_J_m > 0.0
    assert state.separatrix_magnetic_deficit_energy_J_m > 0.0
    assert state.separatrix_energy_closure_relative_error <= 1.0e-12
    assert state.input_thermal_pressure_pa > 0.0
    assert state.thermal_pressure_ratio == pytest.approx(1.0)
    assert state.target_separatrix_radius_m == inputs.R_s
    assert state.separatrix_radius_error_m <= float(np.max(np.diff(rho)))
    assert state.field_reversal_passed is True
    assert state.flux_derivative_residual.shape == rho.shape
    assert state.flux_derivative_residual_linf <= 2.0e-2
    assert state.flux_derivative_residual_l2 <= 2.0e-2
    assert state.J_theta.shape == rho.shape
    assert state.ampere_residual.shape == rho.shape
    assert state.peak_j_theta_A_m2 > 0.0
    assert state.ampere_residual_linf <= 2.0e-2
    assert state.ampere_residual_l2 <= 2.0e-2
    assert state.separatrix_expected_bz_gradient_T_m == pytest.approx(-inputs.B_ext / inputs.delta)
    assert state.separatrix_expected_current_density_A_m2 == pytest.approx(
        inputs.B_ext / (MU_0 * inputs.delta)
    )
    assert state.separatrix_gradient_relative_error <= 2.0e-2
    assert state.separatrix_current_density_relative_error <= 2.0e-2
    expected_sheet_current = (state.B_z[0] - state.B_z[-1]) / MU_0
    assert state.expected_sheet_current_integral_A_m == pytest.approx(float(expected_sheet_current))
    assert state.sheet_current_integral_A_m == pytest.approx(
        float(trapezoid(state.J_theta, state.rho)), rel=1.0e-12
    )
    assert state.sheet_current_integral_relative_error <= 2.0e-2
    assert state.psi_normalized.shape == rho.shape
    assert state.psi_axis_Wb == pytest.approx(0.0, abs=1.0e-14)
    assert state.psi_separatrix_Wb > 0.0
    assert state.psi_normalized_axis_error <= 1.0e-12
    assert state.psi_normalized_separatrix == pytest.approx(1.0, abs=1.0e-12)
    assert state.psi_normalized_separatrix_error <= 1.0e-12
    assert state.psi_normalized_residual_linf <= 1.0e-12
    assert state.psi_normalized_monotonic_passed is True
    assert state.psi_normalized_bounds_passed is True
    assert state.force_balance_residual.shape == rho.shape
    assert state.force_balance_residual_linf >= 0.0
    assert state.force_balance_residual_l2 >= 0.0
    assert np.all(np.isfinite(state.psi))
    assert np.all(state.p > 0.0)


def test_toroidal_current_density_matches_steinhauer_derivative() -> None:
    delta = 0.02
    inputs = _inputs(delta=delta)
    rho: FloatArray = np.linspace(0.0, 0.4, 401)

    state = solve_frc_equilibrium(inputs, rho)

    argument = (rho**2 - inputs.R_s**2) / (2.0 * inputs.R_s * delta)
    expected_j_theta = (
        inputs.B_ext * (1.0 - np.tanh(argument) ** 2) * rho / (MU_0 * inputs.R_s * delta)
    )
    d_bz_dr = np.gradient(state.B_z, state.rho, edge_order=2)
    dp_dr = np.gradient(state.p, state.rho, edge_order=2)
    expected_force_residual = dp_dr - state.J_theta * state.B_z
    d_bz_dr_analytic = -inputs.B_ext * (1.0 - np.tanh(argument) ** 2) * rho / (inputs.R_s * delta)
    expected_pressure_gradient = -(state.B_z * d_bz_dr_analytic) / MU_0
    expected_pressure_gradient_residual = dp_dr - expected_pressure_gradient
    expected_separatrix_gradient = -inputs.B_ext / delta
    expected_separatrix_current_density = inputs.B_ext / (MU_0 * delta)
    interpolated_gradient = float(np.interp(inputs.R_s, rho, d_bz_dr))
    interpolated_current_density = float(np.interp(inputs.R_s, rho, state.J_theta))

    np.testing.assert_allclose(state.J_theta, expected_j_theta, rtol=1.0e-12, atol=1.0e-6)
    np.testing.assert_allclose(
        ampere_residual(state),
        MU_0 * state.J_theta + d_bz_dr,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        pressure_gradient_residual(state),
        expected_pressure_gradient_residual,
        rtol=1.0e-12,
        atol=1.0e-4,
    )
    np.testing.assert_allclose(
        state.pressure_gradient_analytic_Pa_m,
        expected_pressure_gradient,
        rtol=1.0e-12,
        atol=1.0e-4,
    )
    np.testing.assert_allclose(
        force_balance_residual(state),
        expected_force_residual,
        rtol=1.0e-12,
        atol=1.0e-4,
    )
    assert state.separatrix_bz_gradient_T_m == pytest.approx(interpolated_gradient, rel=1.0e-12)
    assert state.separatrix_expected_bz_gradient_T_m == pytest.approx(expected_separatrix_gradient)
    assert state.separatrix_gradient_relative_error == pytest.approx(
        abs(interpolated_gradient - expected_separatrix_gradient)
        / abs(expected_separatrix_gradient),
        rel=1.0e-12,
    )
    assert state.separatrix_current_density_A_m2 == pytest.approx(
        interpolated_current_density, rel=1.0e-12
    )
    assert state.separatrix_expected_current_density_A_m2 == pytest.approx(
        expected_separatrix_current_density
    )
    assert state.separatrix_current_density_relative_error == pytest.approx(
        abs(interpolated_current_density - expected_separatrix_current_density)
        / abs(expected_separatrix_current_density),
        rel=1.0e-12,
    )
    expected_sheet_current = float((state.B_z[0] - state.B_z[-1]) / MU_0)
    assert state.expected_sheet_current_integral_A_m == pytest.approx(expected_sheet_current)
    assert state.sheet_current_integral_A_m == pytest.approx(
        float(trapezoid(state.J_theta, state.rho)), rel=1.0e-12
    )
    assert state.sheet_current_integral_relative_error == pytest.approx(
        abs(state.sheet_current_integral_A_m - expected_sheet_current)
        / abs(expected_sheet_current),
        rel=1.0e-12,
    )


def test_cylindrical_flux_matches_steinhauer_primitive() -> None:
    delta = 0.02
    inputs = _inputs(delta=delta)
    rho: FloatArray = np.linspace(0.0, 0.4, 401)

    state = solve_frc_equilibrium(inputs, rho)

    argument = (rho**2 - inputs.R_s**2) / (2.0 * inputs.R_s * delta)
    log_cosh = np.log(np.cosh(argument))
    expected_psi = -inputs.B_ext * inputs.R_s * delta * (log_cosh - float(log_cosh[0]))
    dpsi_dr = np.gradient(state.psi, state.rho, edge_order=2)

    expected_psi_normalized = (expected_psi - float(expected_psi[0])) / (
        float(np.interp(inputs.R_s, rho, expected_psi)) - float(expected_psi[0])
    )

    np.testing.assert_allclose(state.psi, expected_psi, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(
        psi_normalized_profile(state), expected_psi_normalized, rtol=1.0e-14, atol=1.0e-14
    )
    np.testing.assert_allclose(
        public_psi_normalized_profile(state), expected_psi_normalized, rtol=1.0e-14, atol=1.0e-14
    )
    assert state.psi_axis_Wb == pytest.approx(float(expected_psi[0]), abs=1.0e-14)
    assert state.psi_separatrix_Wb == pytest.approx(
        float(np.interp(inputs.R_s, rho, expected_psi)), rel=1.0e-14
    )
    assert state.psi_normalized_axis_error <= 1.0e-14
    assert state.psi_normalized_separatrix == pytest.approx(1.0, abs=1.0e-14)
    assert state.psi_normalized_separatrix_error <= 1.0e-14
    assert state.psi_normalized_residual_linf <= 1.0e-14
    assert state.psi_normalized_monotonic_passed is True
    assert state.psi_normalized_bounds_passed is True
    np.testing.assert_allclose(
        flux_derivative_residual(state),
        dpsi_dr - state.rho * state.B_z,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert state.flux_derivative_residual_linf <= 2.0e-2


def test_pressure_profile_matches_local_magnetic_pressure_balance() -> None:
    inputs = _inputs(delta=0.02)
    rho: FloatArray = np.linspace(0.0, 0.4, 401)

    state = solve_frc_equilibrium(inputs, rho)

    external_pressure = inputs.B_ext**2 / (2.0 * MU_0)
    expected_pressure = external_pressure - state.B_z**2 / (2.0 * MU_0)
    expected_input_pressure = inputs.n0 * (inputs.T_i_eV + inputs.T_e_eV) * 1.602176634e-19
    expected_density = expected_pressure / ((inputs.T_i_eV + inputs.T_e_eV) * ELEMENTARY_CHARGE_C)
    expected_beta = expected_pressure / external_pressure
    separatrix_mask = rho <= inputs.R_s
    expected_pressure_energy = float(
        trapezoid(
            expected_pressure[separatrix_mask] * 2.0 * np.pi * rho[separatrix_mask],
            rho[separatrix_mask],
        )
    )
    expected_magnetic_deficit_energy = float(
        trapezoid(
            (external_pressure - state.B_z[separatrix_mask] ** 2 / (2.0 * MU_0))
            * 2.0
            * np.pi
            * rho[separatrix_mask],
            rho[separatrix_mask],
        )
    )

    np.testing.assert_allclose(state.p, expected_pressure, rtol=1.0e-14, atol=1.0e-8)
    np.testing.assert_allclose(density_profile(state), expected_density, rtol=1.0e-14, atol=1.0e-8)
    np.testing.assert_allclose(beta_profile(state), expected_beta, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(
        pressure_balance_residual(state),
        state.p + state.B_z**2 / (2.0 * MU_0) - external_pressure,
        rtol=0.0,
        atol=1.0e-8,
    )
    assert state.pressure_balance_residual_linf <= 1.0e-12
    assert state.peak_pressure_pa == pytest.approx(external_pressure, rel=1.0e-4)
    assert state.density_peak_m3 == pytest.approx(inputs.n0, rel=1.0e-4)
    assert state.input_density_m3 == pytest.approx(inputs.n0)
    assert state.central_density_residual_m3 == pytest.approx(0.0, abs=inputs.n0 * 1.0e-4)
    assert state.central_density_relative_error <= 1.0e-4
    assert state.beta_peak == pytest.approx(1.0, rel=1.0e-4)
    assert state.beta_separatrix_average == pytest.approx(
        float(
            trapezoid(
                expected_beta[separatrix_mask] * 2.0 * np.pi * rho[separatrix_mask],
                rho[separatrix_mask],
            )
            / (np.pi * inputs.R_s**2)
        ),
        rel=1.0e-12,
    )
    assert state.particle_line_density_m1 == pytest.approx(
        float(
            trapezoid(
                expected_density[separatrix_mask] * 2.0 * np.pi * rho[separatrix_mask],
                rho[separatrix_mask],
            )
        ),
        rel=1.0e-12,
    )
    assert state.separatrix_pressure_energy_J_m == pytest.approx(
        expected_pressure_energy, rel=1.0e-12
    )
    assert state.separatrix_magnetic_deficit_energy_J_m == pytest.approx(
        expected_magnetic_deficit_energy,
        rel=1.0e-12,
    )
    assert state.separatrix_energy_closure_relative_error <= 1.0e-12
    assert state.input_thermal_pressure_pa == pytest.approx(expected_input_pressure)
    assert state.thermal_pressure_ratio == pytest.approx(
        expected_input_pressure / external_pressure
    )


def test_ampere_residual_refines_with_grid() -> None:
    inputs = _inputs(delta=0.02)

    coarse = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 101))
    medium = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 201))
    fine = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 401))

    assert medium.ampere_residual_linf < coarse.ampere_residual_linf
    assert fine.ampere_residual_linf < medium.ampere_residual_linf


def test_flux_derivative_residual_refines_with_grid() -> None:
    inputs = _inputs(delta=0.02)

    coarse = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 101))
    medium = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 201))
    fine = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 401))

    assert medium.flux_derivative_residual_linf < coarse.flux_derivative_residual_linf
    assert fine.flux_derivative_residual_linf < medium.flux_derivative_residual_linf


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
    assert diagnostic_report.pressure_balance_passed is True
    assert diagnostic_report.pressure_gradient_passed is True
    assert diagnostic_report.flux_closure_passed is True
    assert diagnostic_report.psi_normalized_passed is True
    assert diagnostic_report.psi_normalized_monotonic_passed is True
    assert diagnostic_report.psi_normalized_bounds_passed is True
    assert diagnostic_report.ampere_closure_passed is True
    assert diagnostic_report.field_reversal_passed is True
    assert diagnostic_report.passed
    assert ampere_fail_report.ampere_closure_passed is False
    assert ampere_fail_report.passed is False
    corrupted_psi_normalized = replace(state, psi_normalized_separatrix_error=0.5)
    psi_normalized_fail_report = validate_equilibrium(
        corrupted_psi_normalized,
        psi_normalized_tolerance=1.0e-3,
    )
    assert psi_normalized_fail_report.psi_normalized_passed is False
    assert psi_normalized_fail_report.passed is False
    assert strict_report.force_balance_passed is False
    assert strict_report.passed is False
    assert loose_report.force_balance_passed is True
    assert loose_report.passed is True


def test_pressure_balance_gate_is_fail_closed() -> None:
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)
    state = solve_frc_equilibrium(inputs, rho)

    corrupted = replace(state, pressure_balance_residual_linf=1.0)
    report = validate_equilibrium(corrupted, pressure_balance_tolerance=1.0e-3)

    assert report.pressure_balance_passed is False
    assert report.passed is False


def test_pressure_gradient_gate_is_fail_closed() -> None:
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)
    state = solve_frc_equilibrium(inputs, rho)

    corrupted = replace(state, pressure_gradient_residual_linf=1.0)
    report = validate_equilibrium(corrupted, pressure_gradient_tolerance=1.0e-3)

    assert report.pressure_gradient_passed is False
    assert report.passed is False


def test_density_consistency_gate_is_fail_closed() -> None:
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)
    state = solve_frc_equilibrium(inputs, rho)

    corrupted = replace(state, central_density_relative_error=0.5)
    report = validate_equilibrium(corrupted, density_tolerance=1.0e-3)

    assert report.density_consistency_passed is False
    assert report.passed is False


def test_beta_limit_gate_is_fail_closed() -> None:
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)
    state = solve_frc_equilibrium(inputs, rho)

    corrupted = replace(state, beta_peak=1.5)
    report = validate_equilibrium(corrupted, beta_limit_tolerance=1.0e-3)

    assert report.beta_limit_passed is False
    assert report.passed is False


def test_energy_inventory_gate_is_fail_closed() -> None:
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)
    state = solve_frc_equilibrium(inputs, rho)

    corrupted = replace(state, separatrix_energy_closure_relative_error=0.5)
    report = validate_equilibrium(corrupted, energy_inventory_tolerance=1.0e-6)

    assert report.energy_inventory_passed is False
    assert report.passed is False


def test_current_sheet_gate_is_fail_closed() -> None:
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)
    state = solve_frc_equilibrium(inputs, rho)

    corrupted = replace(state, separatrix_gradient_relative_error=0.5)
    report = validate_equilibrium(corrupted, current_sheet_tolerance=1.0e-3)

    assert report.current_sheet_passed is False
    assert report.passed is False


def test_sheet_current_gate_is_fail_closed() -> None:
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)
    state = solve_frc_equilibrium(inputs, rho)

    corrupted = replace(state, sheet_current_integral_relative_error=1.0)
    report = validate_equilibrium(corrupted, sheet_current_tolerance=1.0e-3)

    assert report.sheet_current_passed is False
    assert report.passed is False


def test_flux_closure_gate_is_fail_closed() -> None:
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)
    state = solve_frc_equilibrium(inputs, rho)

    corrupted = replace(state, flux_derivative_residual_linf=1.0)
    report = validate_equilibrium(corrupted, flux_tolerance=1.0e-3)

    assert report.flux_closure_passed is False
    assert report.passed is False


def test_validation_recomputes_separatrix_target_and_field_reversal() -> None:
    inputs = _inputs(delta=0.02)
    rho = np.linspace(0.0, 0.4, 401)
    state = solve_frc_equilibrium(inputs, rho)

    wrong_target = replace(
        state,
        target_separatrix_radius_m=inputs.R_s + 0.02,
        separatrix_radius_error_m=0.0,
    )
    wrong_target_report = validate_equilibrium(wrong_target)

    non_reversing = replace(
        state,
        B_z=np.abs(state.B_z),
        field_reversal_passed=True,
    )
    non_reversing_report = validate_equilibrium(non_reversing)

    assert wrong_target_report.null_error_m > 0.01
    assert wrong_target_report.passed is False
    assert non_reversing_report.field_reversal_passed is False
    assert non_reversing_report.passed is False


def test_frc_quickstart_reproduces_steinhauer_no_rotation_contract(tmp_path: Path) -> None:
    namespace = _frc_quickstart_namespace()
    run_case = cast(Any, namespace["run_case"])
    main = cast(Any, namespace["main"])

    result = run_case(grid_points=401)
    rho = cast(FloatArray, result["rho_m"])
    b_z = cast(FloatArray, result["B_z_T"])
    pressure = cast(FloatArray, result["pressure_Pa"])
    inputs = cast(dict[str, float], result["inputs"])
    diagnostics = cast(dict[str, float | bool | str], result["diagnostics"])

    expected_b_z = -inputs["B_ext_T"] * np.tanh(
        (rho**2 - inputs["R_s_m"] ** 2) / (2.0 * inputs["R_s_m"] * inputs["delta_m"])
    )
    expected_pressure = inputs["B_ext_T"] ** 2 / (2.0 * MU_0) - b_z**2 / (2.0 * MU_0)

    np.testing.assert_allclose(b_z, expected_b_z, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(pressure, expected_pressure, rtol=1.0e-14, atol=1.0e-8)
    assert diagnostics["validation_passed"] is True
    assert diagnostics["field_reversal_passed"] is True
    assert float(diagnostics["pressure_balance_residual_linf"]) <= 1.0e-12
    assert float(diagnostics["separatrix_energy_closure_relative_error"]) <= 1.0e-12

    output_json = tmp_path / "frc_quickstart_summary.json"
    assert main(["--grid-points", "401", "--output-json", str(output_json)]) == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["diagnostics"]["validation_passed"] is True
    assert payload["inputs"]["grid_points"] == 401
