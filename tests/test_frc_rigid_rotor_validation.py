# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Equilibrium Validation tests
"""Branch-level unit tests for the FRC rigid-rotor acceptance validation.

A real solved equilibrium is used as the passing baseline; each acceptance gate
is then flipped to its failing outcome with :func:`dataclasses.replace`, so both
the pass and fail branch of every gate — including the rotating-state momentum
gate and the finite/isfinite guards — is exercised directly.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from scpn_fusion.core.frc_rigid_rotor_contracts import (
    ELEMENTARY_CHARGE_C,
    MU_0,
    FRCEquilibriumState,
    RigidRotorFRCInputs,
)
from scpn_fusion.core.frc_rigid_rotor_solver import solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor_validation import validate_equilibrium

# Pressure-matched, sub-sonic configuration proven to satisfy every acceptance
# gate (mirrors the accepted no-rotation contract used across the FRC suite).
_T_I_EV = 10_000.0
_T_E_EV = 5_000.0
_B_EXT = 5.0


def _base_inputs(*, theta_dot: float = 0.0) -> RigidRotorFRCInputs:
    external_pressure = _B_EXT**2 / (2.0 * MU_0)
    n0 = external_pressure / ((_T_I_EV + _T_E_EV) * ELEMENTARY_CHARGE_C)
    return RigidRotorFRCInputs(
        n0=n0,
        T_i_eV=_T_I_EV,
        T_e_eV=_T_E_EV,
        theta_dot=theta_dot,
        R_s=0.20,
        B_ext=_B_EXT,
        delta=0.02,
    )


def _solved(*, theta_dot: float = 0.0) -> FRCEquilibriumState:
    grid = np.linspace(0.0, 0.4, 401, dtype=np.float64)
    return solve_frc_equilibrium(_base_inputs(theta_dot=theta_dot), grid)


def _mutate(state: FRCEquilibriumState, **changes: Any) -> FRCEquilibriumState:
    return dataclasses.replace(state, **changes)


class TestPassingBaseline:
    def test_no_rotation_state_passes_all_gates(self) -> None:
        report = validate_equilibrium(_solved())
        assert report.passed
        assert report.finite
        assert report.monotonic_grid
        assert report.field_reversal_passed
        assert report.theta_dot == 0.0

    def test_rotating_state_passes_with_diagnostic_only_magnetostatics(self) -> None:
        report = validate_equilibrium(_solved(theta_dot=3.0e5))
        assert report.passed
        assert report.rotation_force_balance_passed
        assert report.theta_dot == 3.0e5


class TestGeometricGates:
    def test_non_finite_state_fails(self) -> None:
        state = _solved()
        bad = state.p.copy()
        bad[0] = np.nan
        report = validate_equilibrium(_mutate(state, p=bad))
        assert not report.finite
        assert not report.passed

    def test_non_monotonic_grid_fails(self) -> None:
        state = _solved()
        rho = state.rho.copy()
        rho[2] = rho[1]
        report = validate_equilibrium(_mutate(state, rho=rho))
        assert not report.monotonic_grid
        assert not report.passed

    def test_no_field_reversal_fails(self) -> None:
        state = _solved()
        report = validate_equilibrium(_mutate(state, B_z=np.abs(state.B_z) + 1.0))
        assert not report.field_reversal_passed
        assert not report.passed


class TestClosureGates:
    def test_flux_closure_failure(self) -> None:
        report = validate_equilibrium(_mutate(_solved(), flux_derivative_residual_linf=1.0e9))
        assert not report.flux_closure_passed
        assert not report.passed

    def test_ampere_closure_failure(self) -> None:
        report = validate_equilibrium(_mutate(_solved(), ampere_residual_linf=1.0e9))
        assert not report.ampere_closure_passed
        assert not report.passed

    def test_pressure_balance_failure_no_rotation(self) -> None:
        report = validate_equilibrium(_mutate(_solved(), pressure_balance_residual_linf=1.0e9))
        assert not report.pressure_balance_passed
        assert not report.passed

    def test_pressure_gradient_failure_no_rotation(self) -> None:
        report = validate_equilibrium(_mutate(_solved(), pressure_gradient_residual_linf=1.0e9))
        assert not report.pressure_gradient_passed

    def test_energy_inventory_failure_no_rotation(self) -> None:
        report = validate_equilibrium(
            _mutate(_solved(), separatrix_energy_closure_relative_error=1.0e9)
        )
        assert not report.energy_inventory_passed

    def test_density_consistency_failure_no_rotation(self) -> None:
        report = validate_equilibrium(_mutate(_solved(), central_density_relative_error=1.0e9))
        assert not report.density_consistency_passed


class TestCurrentSheetGates:
    def test_current_sheet_relative_error_failure(self) -> None:
        report = validate_equilibrium(_mutate(_solved(), separatrix_gradient_relative_error=1.0e9))
        assert not report.current_sheet_passed

    def test_current_sheet_non_finite_failure(self) -> None:
        report = validate_equilibrium(
            _mutate(_solved(), separatrix_current_density_relative_error=np.nan)
        )
        assert not report.current_sheet_passed

    def test_sheet_current_relative_error_failure(self) -> None:
        report = validate_equilibrium(
            _mutate(_solved(), sheet_current_integral_relative_error=1.0e9)
        )
        assert not report.sheet_current_passed

    def test_sheet_current_non_finite_failure(self) -> None:
        report = validate_equilibrium(
            _mutate(_solved(), sheet_current_integral_relative_error=np.nan)
        )
        assert not report.sheet_current_passed


class TestFluxNormalisationGates:
    def test_psi_normalized_tolerance_failure(self) -> None:
        report = validate_equilibrium(_mutate(_solved(), psi_normalized_axis_error=1.0))
        assert not report.psi_normalized_passed

    def test_psi_normalized_non_finite_failure(self) -> None:
        report = validate_equilibrium(_mutate(_solved(), psi_axis_Wb=np.nan))
        assert not report.psi_normalized_passed

    def test_psi_normalized_monotonic_flag_failure(self) -> None:
        report = validate_equilibrium(_mutate(_solved(), psi_normalized_monotonic_passed=False))
        assert not report.psi_normalized_passed


class TestBetaAndForceGates:
    def test_beta_limit_failure(self) -> None:
        report = validate_equilibrium(_mutate(_solved(), beta_peak=2.0))
        assert not report.beta_limit_passed

    def test_force_balance_tolerance_enforced_when_set(self) -> None:
        state = _mutate(_solved(), force_balance_residual_linf=1.0e9)
        report = validate_equilibrium(state, force_balance_tolerance=1.0e-6)
        assert not report.force_balance_passed
        assert not report.passed

    def test_force_balance_skipped_when_tolerance_none(self) -> None:
        state = _mutate(_solved(), force_balance_residual_linf=1.0e9)
        report = validate_equilibrium(state)
        assert report.force_balance_passed


class TestRotationGate:
    def test_rotation_force_balance_relative_failure(self) -> None:
        state = _solved(theta_dot=3.0e5)
        report = validate_equilibrium(_mutate(state, rotation_force_balance_residual_linf=1.0e9))
        assert not report.rotation_force_balance_passed
        assert not report.passed

    def test_rotation_force_balance_non_finite_failure(self) -> None:
        state = _solved(theta_dot=3.0e5)
        report = validate_equilibrium(_mutate(state, rotation_force_balance_residual_linf=np.nan))
        assert not report.rotation_force_balance_passed
