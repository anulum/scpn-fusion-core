# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Equilibrium Closures tests
"""Unit tests for the shared FRC rigid-rotor closure math and state accessors.

The suite is deliberately JAX-independent: the ``_jax_*`` primitives are driven
with NumPy as the array backend (``jnp = np``) so the branches that production
only reaches through the JAX observable path are covered directly, without the
coverage-tracer JAX import crash. Every closure, residual, geometry helper, and
state accessor is exercised, including the guarded error paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.frc_rigid_rotor import solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor_closures import (
    _ampere_current_closure_residual,
    _axial_field_derivative_from_steinhauer,
    _clip_to_separatrix,
    _cylindrical_flux_from_steinhauer,
    _field_reversal_passed,
    _flux_derivative_closure_residual,
    _jax_log_cosh,
    _jax_steinhauer_psi_at_x,
    _log_cosh,
    _pressure_balance_residual,
    _pressure_gradient_closure_residual,
    _pressure_gradient_from_steinhauer,
    _psi_normalized_bounds_passed,
    _psi_normalized_closure_residual,
    _psi_normalized_monotonic_passed,
    _radial_force_balance_residual,
    _s_parameter_from_profile,
    _toroidal_current_density_from_steinhauer,
    _zero_crossing_radius,
    ampere_residual,
    beta_profile,
    density_profile,
    flux_derivative_residual,
    force_balance_residual,
    null_radius,
    pressure_balance_residual,
    pressure_gradient_residual,
    psi_normalized_profile,
    s_parameter,
)
from scpn_fusion.core.frc_rigid_rotor_contracts import (
    FRCEquilibriumState,
    RigidRotorFRCInputs,
)


def _solved_state() -> FRCEquilibriumState:
    """Return a real no-rotation FRC equilibrium for accessor and profile tests."""
    inputs = RigidRotorFRCInputs(
        n0=1.0e19,
        T_i_eV=200.0,
        T_e_eV=180.0,
        theta_dot=0.0,
        R_s=0.3,
        B_ext=0.5,
    )
    rho_grid = np.linspace(0.0, 0.45, 64, dtype=np.float64)
    return solve_frc_equilibrium(inputs, rho_grid)


class TestSteinhauerPrimitives:
    def test_log_cosh_matches_reference(self) -> None:
        values = np.array([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=np.float64)
        expected = np.log(np.cosh(values))
        np.testing.assert_allclose(_log_cosh(values), expected, atol=1e-12)

    def test_jax_log_cosh_with_numpy_backend(self) -> None:
        values = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
        # NumPy satisfies the JAX-traceable surface used by the primitive.
        result = _jax_log_cosh(np, values)
        np.testing.assert_allclose(result, np.log(np.cosh(values)), atol=1e-12)

    def test_jax_steinhauer_psi_reduces_to_zero_at_axis(self) -> None:
        psi_axis = _jax_steinhauer_psi_at_x(np, 0.0, 0.5, 0.3, 0.02)
        # At x -> the axis argument the two log-cosh terms cancel exactly.
        argument = (0.0 - 1.0) * 0.3 / (2.0 * 0.02)
        axis_argument = -0.3 / (2.0 * 0.02)
        assert argument == axis_argument
        assert float(psi_axis) == 0.0

    def test_cylindrical_flux_vanishes_at_axis(self) -> None:
        argument = np.array([-5.0, -2.0, 0.0, 2.0], dtype=np.float64)
        psi = _cylindrical_flux_from_steinhauer(argument, 0.5, 0.3, 0.02)
        assert float(psi[0]) == 0.0

    def test_toroidal_current_and_field_derivative_are_consistent(self) -> None:
        rho = np.linspace(0.01, 0.4, 32, dtype=np.float64)
        argument = (rho**2 - 0.09) / (2.0 * 0.3 * 0.02)
        j_theta = _toroidal_current_density_from_steinhauer(rho, argument, 0.5, 0.3, 0.02)
        dbz_dr = _axial_field_derivative_from_steinhauer(rho, argument, 0.5, 0.3, 0.02)
        mu_0 = 4.0 * np.pi * 1e-7
        # J_theta = -mu_0^-1 dB_z/dr by construction.
        np.testing.assert_allclose(mu_0 * j_theta, -dbz_dr, atol=1e-18)

    def test_pressure_gradient_from_steinhauer(self) -> None:
        b_z = np.array([0.4, 0.2, 0.0, -0.2], dtype=np.float64)
        dbz_dr = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)
        mu_0 = 4.0 * np.pi * 1e-7
        expected = -(b_z * dbz_dr) / mu_0
        np.testing.assert_allclose(_pressure_gradient_from_steinhauer(b_z, dbz_dr), expected)


class TestClosureResiduals:
    def test_ampere_and_flux_and_gradient_residuals_are_finite(self) -> None:
        state = _solved_state()
        assert np.all(
            np.isfinite(_ampere_current_closure_residual(state.rho, state.B_z, state.J_theta))
        )
        assert np.all(
            np.isfinite(_flux_derivative_closure_residual(state.rho, state.psi, state.B_z))
        )
        analytic = _pressure_gradient_from_steinhauer(
            state.B_z, np.gradient(state.B_z, state.rho, edge_order=2)
        )
        assert np.all(
            np.isfinite(_pressure_gradient_closure_residual(state.rho, state.p, analytic))
        )

    def test_pressure_and_force_balance_residuals(self) -> None:
        state = _solved_state()
        pb = _pressure_balance_residual(state.p, state.B_z, 0.5)
        fb = _radial_force_balance_residual(state.rho, state.B_z, state.J_theta, state.p)
        assert pb.shape == state.p.shape
        assert fb.shape == state.p.shape

    def test_psi_normalized_closure_residual_matches_definition(self) -> None:
        psi = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        psi_normalized = (psi - 0.0) / (1.0 - 0.0)
        residual = _psi_normalized_closure_residual(psi, 0.0, 1.0, psi_normalized)
        np.testing.assert_allclose(residual, np.zeros_like(psi), atol=1e-15)

    def test_psi_normalized_closure_residual_rejects_zero_span(self) -> None:
        psi = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="psi separatrix span must be non-zero"):
            _psi_normalized_closure_residual(psi, 0.4, 0.4, psi)


class TestGeometryHelpers:
    def test_clip_to_separatrix_appends_interpolated_endpoint(self) -> None:
        rho = np.array([0.0, 0.1, 0.2, 0.4], dtype=np.float64)
        values = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float64)
        r_clip, value_clip = _clip_to_separatrix(rho, values, 0.3)
        assert float(r_clip[-1]) == 0.3
        assert float(value_clip[-1]) == pytest.approx(4.0)

    def test_clip_to_separatrix_exact_endpoint_no_append(self) -> None:
        rho = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        r_clip, value_clip = _clip_to_separatrix(rho, values, 0.3)
        assert float(r_clip[-1]) == 0.3
        assert r_clip.size == 4

    def test_clip_to_separatrix_rejects_empty_interval(self) -> None:
        rho = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(ValueError, match="rho_grid must contain points below R_s"):
            _clip_to_separatrix(rho, values, 0.5)

    def test_s_parameter_from_profile_positive(self) -> None:
        state = _solved_state()
        value = _s_parameter_from_profile(state.rho, state.B_z, 0.3, 200.0)
        assert value > 0.0

    def test_s_parameter_from_profile_guards(self) -> None:
        rho = np.array([0.0, 0.1, 0.2, 0.4], dtype=np.float64)
        b_z = np.array([0.4, 0.2, 0.0, -0.2], dtype=np.float64)
        with pytest.raises(ValueError, match="R_s must be positive"):
            _s_parameter_from_profile(rho, b_z, 0.0, 200.0)
        with pytest.raises(ValueError, match="T_i_eV must be positive"):
            _s_parameter_from_profile(rho, b_z, 0.3, 0.0)
        with pytest.raises(ValueError, match="mass_amu must be positive"):
            _s_parameter_from_profile(rho, b_z, 0.3, 200.0, mass_amu=0.0)

    def test_psi_normalized_monotonic_and_bounds(self) -> None:
        rho = np.array([0.0, 0.1, 0.2, 0.4], dtype=np.float64)
        psi_n = np.array([0.0, 0.4, 0.8, 1.2], dtype=np.float64)
        assert _psi_normalized_monotonic_passed(rho, psi_n, 0.3, 1e-9) is True
        assert _psi_normalized_bounds_passed(rho, psi_n, 0.3, 1e-9) is True

    def test_zero_crossing_radius_interpolates(self) -> None:
        rho = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        values = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        assert _zero_crossing_radius(rho, values) == pytest.approx(1.5)

    def test_zero_crossing_radius_no_sign_change_returns_argmin(self) -> None:
        rho = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        values = np.array([3.0, 2.0, 0.5, 4.0], dtype=np.float64)
        assert _zero_crossing_radius(rho, values) == pytest.approx(2.0)

    def test_zero_crossing_radius_equal_endpoints_returns_left(self) -> None:
        rho = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        # -0.0 and 0.0 differ in sign bit but compare equal -> the flat branch.
        values = np.array([-0.0, 0.0, 1.0, 2.0], dtype=np.float64)
        assert _zero_crossing_radius(rho, values) == 0.0

    def test_field_reversal_passed_true(self) -> None:
        state = _solved_state()
        assert _field_reversal_passed(state.rho, state.B_z, 0.3) is True

    def test_field_reversal_passed_no_inner_or_outer_points(self) -> None:
        rho = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        b_z = np.array([0.4, -0.2, -0.5], dtype=np.float64)
        assert _field_reversal_passed(rho, b_z, 1.0) is False
        rho_inner = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        assert _field_reversal_passed(rho_inner, b_z, 1.0) is False


class TestStateAccessors:
    def test_null_radius_matches_separatrix(self) -> None:
        state = _solved_state()
        assert null_radius(state) == pytest.approx(0.3, abs=0.02)

    def test_s_parameter_accessor_and_guard(self) -> None:
        state = _solved_state()
        assert s_parameter(state) == pytest.approx(state.s_parameter)
        with pytest.raises(ValueError, match="mass_amu must be positive"):
            s_parameter(state, mass_amu=0.0)

    def test_residual_and_profile_accessors_return_state_fields(self) -> None:
        state = _solved_state()
        assert force_balance_residual(state) is state.force_balance_residual
        assert ampere_residual(state) is state.ampere_residual
        assert flux_derivative_residual(state) is state.flux_derivative_residual
        assert psi_normalized_profile(state) is state.psi_normalized
        assert pressure_balance_residual(state) is state.pressure_balance_residual
        assert pressure_gradient_residual(state) is state.pressure_gradient_residual
        assert density_profile(state) is state.density_m3
        assert beta_profile(state) is state.beta
