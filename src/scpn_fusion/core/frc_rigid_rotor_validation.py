# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Equilibrium Validation
"""Acceptance validation for the FRC rigid-rotor equilibrium.

Consumer layer of the ``frc_rigid_rotor`` package: turns a solved
:class:`FRCEquilibriumState` into an :class:`FRCValidationReport` of finite,
geometric, closure, current-sheet, flux-normalisation, and force-balance gates.
Imports the null-radius and field-reversal helpers from
``frc_rigid_rotor_closures`` and the data contracts from
``frc_rigid_rotor_contracts``, so it carries no solver logic and stays free of an
import cycle. Re-exported by the ``frc_rigid_rotor`` facade.
"""

from __future__ import annotations

import numpy as np

from .frc_rigid_rotor_closures import _field_reversal_passed, null_radius
from .frc_rigid_rotor_contracts import FRCEquilibriumState, FRCValidationReport


def validate_equilibrium(
    state: FRCEquilibriumState,
    *,
    tolerance: float = 1e-6,
    flux_tolerance: float = 2e-2,
    pressure_balance_tolerance: float = 2e-2,
    pressure_gradient_tolerance: float = 2e-2,
    density_tolerance: float = 2e-2,
    beta_limit_tolerance: float = 2e-2,
    energy_inventory_tolerance: float = 1e-10,
    ampere_tolerance: float = 2e-2,
    current_sheet_tolerance: float = 2e-2,
    sheet_current_tolerance: float = 2e-2,
    psi_normalized_tolerance: float = 1e-10,
    force_balance_tolerance: float | None = None,
    rotation_force_balance_tolerance: float = 2e-2,
) -> FRCValidationReport:
    """Validate finite values, null placement, pressure peaking, and force balance.

    For a rotating state (``theta_dot != 0``) the authoritative momentum check is
    the centrifugal force-balance residual ``dp/dr - J_theta B_z - rho omega^2 r``;
    the magnetostatic-only identities (constant ``p + B_z^2/2 mu_0`` pressure
    balance, the no-rotation pressure gradient, and the magnetostatic energy
    closure) do not hold under rotation and are reported as diagnostics rather
    than pass/fail gates.
    """
    is_rotating = state.theta_dot != 0.0
    finite = all(
        bool(np.all(np.isfinite(values)))
        for values in (
            state.rho,
            state.psi,
            state.psi_normalized,
            state.B_z,
            state.B_theta,
            state.J_theta,
            state.p,
            state.density_m3,
            state.beta,
            state.pressure_balance_residual,
            state.pressure_gradient_analytic_Pa_m,
            state.pressure_gradient_residual,
            state.flux_derivative_residual,
            state.ampere_residual,
            state.force_balance_residual,
        )
    )
    monotonic_grid = bool(np.all(np.diff(state.rho) > 0.0))
    r_null = null_radius(state)
    null_error = abs(r_null - state.target_separatrix_radius_m)
    field_reversal_passed = _field_reversal_passed(
        state.rho,
        state.B_z,
        state.target_separatrix_radius_m,
    )
    pressure_peak_radius = float(state.rho[int(np.argmax(state.p))])
    pressure_peak_error = abs(pressure_peak_radius - state.R_null)
    edge_field_error = abs(abs(float(state.B_z[-1])) - abs(float(state.B_z[0])))
    radial_spacing = float(np.max(np.diff(state.rho)))
    geometric_passed = (
        finite
        and monotonic_grid
        and field_reversal_passed
        and null_error <= max(tolerance, radial_spacing)
        and pressure_peak_error <= max(tolerance, 2.0 * radial_spacing)
    )
    # Magnetostatic identities only gate the no-rotation contract; under rotation
    # the centrifugal source makes them non-zero by construction (diagnostics only).
    magnetostatic_pressure_balance_passed = (
        state.pressure_balance_residual_linf <= pressure_balance_tolerance
    )
    magnetostatic_pressure_gradient_passed = (
        state.pressure_gradient_residual_linf <= pressure_gradient_tolerance
    )
    magnetostatic_energy_inventory_passed = (
        state.separatrix_energy_closure_relative_error <= energy_inventory_tolerance
    )
    pressure_balance_passed = is_rotating or magnetostatic_pressure_balance_passed
    pressure_gradient_passed = is_rotating or magnetostatic_pressure_gradient_passed
    density_consistency_passed = (
        is_rotating or state.central_density_relative_error <= density_tolerance
    )
    beta_limit_passed = state.beta_peak <= 1.0 + beta_limit_tolerance
    energy_inventory_passed = is_rotating or magnetostatic_energy_inventory_passed
    rotation_force_balance_passed = (not is_rotating) or (
        np.isfinite(state.rotation_force_balance_residual_linf)
        and state.rotation_force_balance_residual_linf <= rotation_force_balance_tolerance
    )
    flux_closure_passed = state.flux_derivative_residual_linf <= flux_tolerance
    ampere_closure_passed = state.ampere_residual_linf <= ampere_tolerance
    current_sheet_passed = (
        np.isfinite(state.separatrix_gradient_relative_error)
        and np.isfinite(state.separatrix_current_density_relative_error)
        and state.separatrix_gradient_relative_error <= current_sheet_tolerance
        and state.separatrix_current_density_relative_error <= current_sheet_tolerance
    )
    sheet_current_passed = (
        np.isfinite(state.sheet_current_integral_relative_error)
        and state.sheet_current_integral_relative_error <= sheet_current_tolerance
    )
    psi_normalized_passed = (
        np.isfinite(state.psi_axis_Wb)
        and np.isfinite(state.psi_separatrix_Wb)
        and np.isfinite(state.psi_normalized_axis_error)
        and np.isfinite(state.psi_normalized_separatrix)
        and np.isfinite(state.psi_normalized_separatrix_error)
        and np.isfinite(state.psi_normalized_residual_linf)
        and state.psi_normalized_axis_error <= psi_normalized_tolerance
        and state.psi_normalized_separatrix_error <= psi_normalized_tolerance
        and state.psi_normalized_residual_linf <= psi_normalized_tolerance
        and state.psi_normalized_monotonic_passed
        and state.psi_normalized_bounds_passed
    )
    force_balance_passed = (
        force_balance_tolerance is None
        or state.force_balance_residual_linf <= force_balance_tolerance
    )
    passed = (
        geometric_passed
        and pressure_balance_passed
        and pressure_gradient_passed
        and density_consistency_passed
        and beta_limit_passed
        and energy_inventory_passed
        and flux_closure_passed
        and ampere_closure_passed
        and current_sheet_passed
        and sheet_current_passed
        and psi_normalized_passed
        and force_balance_passed
        and rotation_force_balance_passed
    )
    return FRCValidationReport(
        finite=finite,
        monotonic_grid=monotonic_grid,
        null_error_m=float(null_error),
        target_separatrix_radius_m=state.target_separatrix_radius_m,
        field_reversal_passed=bool(field_reversal_passed),
        psi_axis_Wb=state.psi_axis_Wb,
        psi_separatrix_Wb=state.psi_separatrix_Wb,
        psi_normalized_axis_error=state.psi_normalized_axis_error,
        psi_normalized_separatrix=state.psi_normalized_separatrix,
        psi_normalized_separatrix_error=state.psi_normalized_separatrix_error,
        psi_normalized_residual_linf=state.psi_normalized_residual_linf,
        psi_normalized_monotonic_passed=bool(state.psi_normalized_monotonic_passed),
        psi_normalized_bounds_passed=bool(state.psi_normalized_bounds_passed),
        psi_normalized_passed=bool(psi_normalized_passed),
        pressure_peak_error_m=float(pressure_peak_error),
        edge_field_error_T=float(edge_field_error),
        pressure_balance_ratio=state.pressure_balance_ratio,
        pressure_balance_residual_linf=state.pressure_balance_residual_linf,
        pressure_balance_residual_l2=state.pressure_balance_residual_l2,
        pressure_balance_passed=bool(pressure_balance_passed),
        pressure_gradient_residual_linf=state.pressure_gradient_residual_linf,
        pressure_gradient_residual_l2=state.pressure_gradient_residual_l2,
        pressure_gradient_passed=bool(pressure_gradient_passed),
        thermal_pressure_ratio=state.thermal_pressure_ratio,
        density_peak_m3=state.density_peak_m3,
        input_density_m3=state.input_density_m3,
        central_density_residual_m3=state.central_density_residual_m3,
        central_density_relative_error=state.central_density_relative_error,
        density_consistency_passed=bool(density_consistency_passed),
        beta_peak=state.beta_peak,
        beta_separatrix_average=state.beta_separatrix_average,
        particle_line_density_m1=state.particle_line_density_m1,
        beta_limit_passed=bool(beta_limit_passed),
        separatrix_pressure_energy_J_m=state.separatrix_pressure_energy_J_m,
        separatrix_magnetic_deficit_energy_J_m=state.separatrix_magnetic_deficit_energy_J_m,
        separatrix_energy_closure_relative_error=state.separatrix_energy_closure_relative_error,
        energy_inventory_passed=bool(energy_inventory_passed),
        flux_derivative_residual_linf=state.flux_derivative_residual_linf,
        flux_derivative_residual_l2=state.flux_derivative_residual_l2,
        flux_closure_passed=bool(flux_closure_passed),
        ampere_residual_linf=state.ampere_residual_linf,
        ampere_residual_l2=state.ampere_residual_l2,
        ampere_closure_passed=bool(ampere_closure_passed),
        separatrix_bz_gradient_T_m=state.separatrix_bz_gradient_T_m,
        separatrix_expected_bz_gradient_T_m=state.separatrix_expected_bz_gradient_T_m,
        separatrix_gradient_relative_error=state.separatrix_gradient_relative_error,
        separatrix_current_density_A_m2=state.separatrix_current_density_A_m2,
        separatrix_expected_current_density_A_m2=state.separatrix_expected_current_density_A_m2,
        separatrix_current_density_relative_error=state.separatrix_current_density_relative_error,
        sheet_current_integral_A_m=state.sheet_current_integral_A_m,
        expected_sheet_current_integral_A_m=state.expected_sheet_current_integral_A_m,
        sheet_current_integral_relative_error=state.sheet_current_integral_relative_error,
        sheet_current_passed=bool(sheet_current_passed),
        current_sheet_passed=bool(current_sheet_passed),
        force_balance_residual_linf=state.force_balance_residual_linf,
        force_balance_residual_l2=state.force_balance_residual_l2,
        force_balance_passed=bool(force_balance_passed),
        passed=bool(passed),
        theta_dot=float(state.theta_dot),
        rotation_force_balance_residual_linf=float(state.rotation_force_balance_residual_linf),
        rotation_force_balance_residual_l2=float(state.rotation_force_balance_residual_l2),
        rotation_force_balance_passed=bool(rotation_force_balance_passed),
    )
