# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Equilibrium Solver tests
"""Unit tests for the FRC rigid-rotor solver and its input guards.

The suite is JAX-independent: it drives the NumPy solver path, the rotating
centrifugal closure, and every guard branch directly. The differentiable
``frc_no_rotation_jax_observables`` body itself requires the optional JAX import,
which crashes under the coverage tracer (documented jax×coverage-tracer bug), so
it is exercised by the behavioural (no-``--cov``) suite in
``test_frc_rigid_rotor.py`` rather than here. Its NumPy input guards
(``_validate_normalized_grid``, ``_normalised_separatrix_grid``,
``_validate_positive_concrete``, ``_validate_nonzero_concrete``) are covered here
by direct calls so that only the JAX-import-dependent body remains coverage-blocked.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.frc_rigid_rotor_contracts import RigidRotorFRCInputs
from scpn_fusion.core.frc_rigid_rotor_solver import (
    _fill_equilibrium_state,
    _normalised_separatrix_grid,
    _rotating_pressure_from_density_closure,
    _validate_grid,
    _validate_inputs,
    _validate_nonzero_concrete,
    _validate_normalized_grid,
    _validate_positive_concrete,
    ion_gyroradius_m,
    solve_frc_equilibrium,
)


def _inputs(*, theta_dot: float = 0.0) -> RigidRotorFRCInputs:
    return RigidRotorFRCInputs(
        n0=1.0e19,
        T_i_eV=200.0,
        T_e_eV=180.0,
        theta_dot=theta_dot,
        R_s=0.3,
        B_ext=0.5,
    )


def _grid() -> NDArray[np.float64]:
    return np.linspace(0.0, 0.45, 64, dtype=np.float64)


class TestIonGyroradius:
    def test_returns_positive_radius(self) -> None:
        assert ion_gyroradius_m(200.0, 0.5) > 0.0

    def test_rejects_nonpositive_temperature(self) -> None:
        with pytest.raises(ValueError, match="T_i_eV must be positive"):
            ion_gyroradius_m(0.0, 0.5)

    def test_rejects_zero_field(self) -> None:
        with pytest.raises(ValueError, match="B_T must be non-zero"):
            ion_gyroradius_m(200.0, 0.0)


class TestSolveNoRotation:
    def test_solves_and_reverses_field(self) -> None:
        state = solve_frc_equilibrium(_inputs(), _grid())
        assert state.converged
        assert state.theta_dot == 0.0
        assert state.field_reversal_passed
        assert state.model == "steinhauer_2011_no_rotation_analytical"

    def test_explicit_delta_bypasses_gyroradius(self) -> None:
        inputs = RigidRotorFRCInputs(
            n0=1.0e19,
            T_i_eV=200.0,
            T_e_eV=180.0,
            theta_dot=0.0,
            R_s=0.3,
            B_ext=0.5,
            delta=0.02,
        )
        state = solve_frc_equilibrium(inputs, _grid())
        assert state.delta == 0.02


class TestSolveRotating:
    def test_rotating_branch_uses_centrifugal_closure(self) -> None:
        state = solve_frc_equilibrium(_inputs(theta_dot=2.0e5), _grid())
        assert state.theta_dot == 2.0e5
        assert state.model == "rostoker_qerushi_2002_rotating_rigid_rotor"
        # Non-negative pressure profile under the centrifugal density modulation.
        assert np.all(state.p >= 0.0)


class TestRotatingPressureClosure:
    def test_reduces_to_static_at_zero_rotation_scale(self) -> None:
        rho = _grid()
        static = np.ones_like(rho)
        rotated = _rotating_pressure_from_density_closure(rho, static, 1.0e3, 380.0)
        # Small drive keeps the factor close to unity and non-negative.
        assert np.all(rotated >= static - 1e-9)

    def test_rejects_supersonic_overflow(self) -> None:
        rho = np.linspace(0.0, 5.0, 8, dtype=np.float64)
        static = np.ones_like(rho)
        with pytest.raises(ValueError, match="rotating-closure validity cap"):
            _rotating_pressure_from_density_closure(rho, static, 5.0e7, 1.0)


class TestFillEquilibriumStateGuard:
    def test_rejects_degenerate_flux_span(self) -> None:
        rho = np.array([0.0, 0.1, 0.2, 0.4], dtype=np.float64)
        flat_psi = np.zeros_like(rho)
        b_z = np.array([0.4, 0.1, -0.1, -0.4], dtype=np.float64)
        j_theta = np.ones_like(rho)
        p = np.array([1.0, 2.0, 1.0, 0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="psi separatrix span must be non-zero"):
            _fill_equilibrium_state(
                _inputs(),
                rho,
                flat_psi,
                b_z,
                j_theta,
                p,
                1e-10,
                0.0,
                0.02,
                np.zeros_like(rho),
            )


class TestValidateInputs:
    def test_accepts_valid_inputs(self) -> None:
        _validate_inputs(_inputs(), 1e-10)

    def test_rejects_nonpositive_density(self) -> None:
        with pytest.raises(ValueError, match="n0 must be positive"):
            _validate_inputs(RigidRotorFRCInputs(0.0, 200.0, 180.0, 0.0, 0.3, 0.5), 1e-10)

    def test_rejects_nonpositive_temperatures(self) -> None:
        with pytest.raises(ValueError, match="temperatures must be positive"):
            _validate_inputs(RigidRotorFRCInputs(1.0e19, 0.0, 180.0, 0.0, 0.3, 0.5), 1e-10)

    def test_rejects_nonpositive_radius(self) -> None:
        with pytest.raises(ValueError, match="R_s must be positive"):
            _validate_inputs(RigidRotorFRCInputs(1.0e19, 200.0, 180.0, 0.0, 0.0, 0.5), 1e-10)

    def test_rejects_zero_field(self) -> None:
        with pytest.raises(ValueError, match="B_ext must be non-zero"):
            _validate_inputs(RigidRotorFRCInputs(1.0e19, 200.0, 180.0, 0.0, 0.3, 0.0), 1e-10)

    def test_rejects_nonpositive_delta(self) -> None:
        with pytest.raises(ValueError, match="delta must be positive"):
            _validate_inputs(
                RigidRotorFRCInputs(1.0e19, 200.0, 180.0, 0.0, 0.3, 0.5, delta=-0.01), 1e-10
            )


class TestValidateGrid:
    def test_accepts_valid_grid(self) -> None:
        rho = _validate_grid(_grid(), 0.3)
        assert rho.ndim == 1

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            _validate_grid(np.zeros((2, 2), dtype=np.float64), 0.3)

    def test_rejects_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least four points"):
            _validate_grid(np.array([0.0, 0.1, 0.4], dtype=np.float64), 0.3)

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite values"):
            _validate_grid(np.array([0.0, 0.1, np.inf, 0.4], dtype=np.float64), 0.3)

    def test_rejects_nonzero_origin(self) -> None:
        with pytest.raises(ValueError, match="start at the magnetic axis"):
            _validate_grid(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64), 0.3)

    def test_rejects_non_increasing(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            _validate_grid(np.array([0.0, 0.2, 0.2, 0.4], dtype=np.float64), 0.3)

    def test_rejects_grid_inside_separatrix(self) -> None:
        with pytest.raises(ValueError, match="outside the separatrix radius"):
            _validate_grid(np.array([0.0, 0.1, 0.2, 0.25], dtype=np.float64), 0.3)


class TestValidateNormalizedGrid:
    def test_accepts_valid_grid(self) -> None:
        x = _validate_normalized_grid(np.array([0.0, 0.4, 0.8, 1.2], dtype=np.float64))
        assert x[0] == 0.0

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            _validate_normalized_grid(np.zeros((2, 2), dtype=np.float64))

    def test_rejects_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least four points"):
            _validate_normalized_grid(np.array([0.0, 0.5, 1.2], dtype=np.float64))

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite values"):
            _validate_normalized_grid(np.array([0.0, 0.4, np.nan, 1.2], dtype=np.float64))

    def test_rejects_nonzero_origin(self) -> None:
        with pytest.raises(ValueError, match="must start at 0"):
            _validate_normalized_grid(np.array([0.1, 0.4, 0.8, 1.2], dtype=np.float64))

    def test_rejects_non_increasing(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            _validate_normalized_grid(np.array([0.0, 0.4, 0.4, 1.2], dtype=np.float64))

    def test_rejects_grid_inside_separatrix(self) -> None:
        with pytest.raises(ValueError, match="outside the separatrix x=1"):
            _validate_normalized_grid(np.array([0.0, 0.3, 0.6, 0.9], dtype=np.float64))


class TestNormalisedSeparatrixGrid:
    def test_appends_separatrix_endpoint(self) -> None:
        x = _normalised_separatrix_grid(np.array([0.0, 0.4, 0.8, 1.2], dtype=np.float64))
        assert float(x[-1]) == 1.0

    def test_keeps_exact_separatrix_endpoint(self) -> None:
        x = _normalised_separatrix_grid(np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64))
        assert float(x[-1]) == 1.0
        assert x.size == 3

    def test_rejects_grid_without_interior_points(self) -> None:
        with pytest.raises(ValueError, match="points below x=1"):
            _normalised_separatrix_grid(np.array([1.5, 2.0, 2.5], dtype=np.float64))


class TestConcreteGuards:
    def test_positive_concrete_ignores_non_numeric(self) -> None:
        # Non-numeric (e.g. a traced value) is passed through unchecked.
        _validate_positive_concrete("traced", "x")

    def test_positive_concrete_accepts_positive(self) -> None:
        _validate_positive_concrete(1.5, "x")

    def test_positive_concrete_rejects_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="x must be positive"):
            _validate_positive_concrete(-1.0, "x")

    def test_positive_concrete_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="x must be positive"):
            _validate_positive_concrete(np.inf, "x")

    def test_nonzero_concrete_ignores_non_numeric(self) -> None:
        _validate_nonzero_concrete("traced", "b")

    def test_nonzero_concrete_accepts_nonzero(self) -> None:
        _validate_nonzero_concrete(0.5, "b")

    def test_nonzero_concrete_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="b must be non-zero"):
            _validate_nonzero_concrete(0.0, "b")

    def test_nonzero_concrete_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="b must be non-zero"):
            _validate_nonzero_concrete(np.nan, "b")
