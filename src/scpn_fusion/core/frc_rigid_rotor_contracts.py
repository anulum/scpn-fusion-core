# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Equilibrium Contracts
"""Data contracts and physical constants for the FRC rigid-rotor equilibrium.

Base layer of the ``frc_rigid_rotor`` package: physical constants, the input and
state dataclasses, and the machine-readable rotating-closure acceptance status.
Carries no solver or validation logic, so the solver and validation modules can
import it without a cycle. Re-exported by the ``frc_rigid_rotor`` facade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

MU_0 = 4.0 * np.pi * 1e-7
ELEMENTARY_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_KG = 1.66053906660e-27
DEUTERIUM_MASS_AMU = 2.014

FloatArray: TypeAlias = NDArray[np.float64]


def _empty_float_array() -> FloatArray:
    """Return an empty float64 array used as a dataclass default for diagnostics."""
    return np.empty(0, dtype=np.float64)


ROTATING_FRC_BVP_STATUS = "implemented_rostoker_qerushi_1d_rotating_closure"
ROTATING_FRC_BVP_REQUIRED_REFERENCE = (
    "Rostoker & Qerushi 2002 Phys. Plasmas 9 3057 (one space dimension, one ion); "
    "US 6,664,740 B2 rigid-rotor density closure"
)
ROTATING_FRC_BVP_SOLVER_ACTION = (
    "solve_outer_anchored_centrifugal_force_balance_for_nonzero_theta_dot"
)
ROTATING_FRC_BVP_CLAIM_BOUNDARY = (
    "The rotating rigid-rotor equilibrium solves the source-verified Rostoker & Qerushi "
    "(2002) one-dimensional one-ion centrifugal force balance "
    "d/dr[p + B_z^2/(2 mu_0)] = rho omega^2 r with the rigid-rotor density closure, "
    "reducing bit-exactly to the accepted Steinhauer no-rotation contract at "
    "theta_dot == 0. Verbatim Steinhauer 2011 Figure 3 digitised parity is NOT claimed "
    "and remains a separate external-parity gate; C-2U performance/topology references "
    "are context only, not a figure-parity certification."
)
ROTATING_FRC_BVP_ROTATING_REFERENCE = (
    "Rostoker & Qerushi 2002; US 6,664,740 B2 rigid-rotor density closure "
    "n_j = n_j(0) exp[(m_j omega_j^2 r^2/2 - e_j phi + e_j omega_j psi)/T_j]"
)
ROTATING_FRC_BVP_NON_CLOSING_REFERENCES = (
    "Romero 2018",
    "Baltz 2017 C-2U positive-net-heating table",
    "Slough 2011 Fig. 5",
)


@dataclass(frozen=True)
class RigidRotorFRCInputs:
    """Physical inputs for the FRC rigid-rotor analytical equilibrium."""

    n0: float
    T_i_eV: float
    T_e_eV: float
    theta_dot: float
    R_s: float
    B_ext: float
    delta: float | None = None


@dataclass(frozen=True)
class FRCEquilibriumState:
    """Radial FRC equilibrium state returned by :func:`solve_frc_equilibrium`."""

    rho: FloatArray
    psi: FloatArray
    psi_normalized: FloatArray
    B_z: FloatArray
    B_theta: FloatArray
    J_theta: FloatArray
    p: FloatArray
    density_m3: FloatArray
    beta: FloatArray
    R_null: float
    target_separatrix_radius_m: float
    separatrix_radius_error_m: float
    separatrix_index: int
    field_reversal_passed: bool
    s_parameter: float
    energy_J: float
    converged: bool
    residual: float
    delta: float
    psi_axis_Wb: float
    psi_separatrix_Wb: float
    psi_normalized_axis_error: float
    psi_normalized_separatrix: float
    psi_normalized_separatrix_error: float
    psi_normalized_residual_linf: float
    psi_normalized_monotonic_passed: bool
    psi_normalized_bounds_passed: bool
    pressure_balance_ratio: float
    pressure_balance_residual: FloatArray
    pressure_balance_residual_linf: float
    pressure_balance_residual_l2: float
    pressure_gradient_analytic_Pa_m: FloatArray
    pressure_gradient_residual: FloatArray
    pressure_gradient_residual_linf: float
    pressure_gradient_residual_l2: float
    peak_pressure_pa: float
    density_peak_m3: float
    input_density_m3: float
    central_density_residual_m3: float
    central_density_relative_error: float
    beta_peak: float
    beta_separatrix_average: float
    particle_line_density_m1: float
    separatrix_pressure_energy_J_m: float
    separatrix_magnetic_deficit_energy_J_m: float
    separatrix_energy_closure_relative_error: float
    input_thermal_pressure_pa: float
    thermal_pressure_ratio: float
    flux_derivative_residual: FloatArray
    flux_derivative_residual_linf: float
    flux_derivative_residual_l2: float
    ampere_residual: FloatArray
    ampere_residual_linf: float
    ampere_residual_l2: float
    peak_j_theta_A_m2: float
    separatrix_bz_gradient_T_m: float
    separatrix_expected_bz_gradient_T_m: float
    separatrix_gradient_relative_error: float
    separatrix_current_density_A_m2: float
    separatrix_expected_current_density_A_m2: float
    separatrix_current_density_relative_error: float
    sheet_current_integral_A_m: float
    expected_sheet_current_integral_A_m: float
    sheet_current_integral_relative_error: float
    force_balance_residual: FloatArray
    force_balance_residual_linf: float
    force_balance_residual_l2: float
    model: str
    # --- Rotating rigid-rotor (Rostoker & Qerushi 2002) diagnostics ---
    # For ``theta_dot == 0`` these carry the trivial no-rotation values (zeros /
    # the no-rotation model tag) so the accepted contract is byte-unchanged.
    theta_dot: float = 0.0
    rotation_reference: str = ""
    centrifugal_source_Pa_m: FloatArray = field(default_factory=_empty_float_array)
    rotation_force_balance_residual: FloatArray = field(default_factory=_empty_float_array)
    rotation_force_balance_residual_linf: float = 0.0
    rotation_force_balance_residual_l2: float = 0.0
    rotation_mach_number: float = 0.0
    rotation_pressure_peak_radius_m: float = 0.0
    pressure_clipped_fraction: float = 0.0


@dataclass(frozen=True)
class FRCValidationReport:
    """Validation diagnostics for the analytical FRC state."""

    finite: bool
    monotonic_grid: bool
    null_error_m: float
    target_separatrix_radius_m: float
    field_reversal_passed: bool
    psi_axis_Wb: float
    psi_separatrix_Wb: float
    psi_normalized_axis_error: float
    psi_normalized_separatrix: float
    psi_normalized_separatrix_error: float
    psi_normalized_residual_linf: float
    psi_normalized_monotonic_passed: bool
    psi_normalized_bounds_passed: bool
    psi_normalized_passed: bool
    pressure_peak_error_m: float
    edge_field_error_T: float
    pressure_balance_ratio: float
    pressure_balance_residual_linf: float
    pressure_balance_residual_l2: float
    pressure_balance_passed: bool
    pressure_gradient_residual_linf: float
    pressure_gradient_residual_l2: float
    pressure_gradient_passed: bool
    thermal_pressure_ratio: float
    density_peak_m3: float
    input_density_m3: float
    central_density_residual_m3: float
    central_density_relative_error: float
    density_consistency_passed: bool
    beta_peak: float
    beta_separatrix_average: float
    particle_line_density_m1: float
    beta_limit_passed: bool
    separatrix_pressure_energy_J_m: float
    separatrix_magnetic_deficit_energy_J_m: float
    separatrix_energy_closure_relative_error: float
    energy_inventory_passed: bool
    flux_derivative_residual_linf: float
    flux_derivative_residual_l2: float
    flux_closure_passed: bool
    ampere_residual_linf: float
    ampere_residual_l2: float
    ampere_closure_passed: bool
    separatrix_bz_gradient_T_m: float
    separatrix_expected_bz_gradient_T_m: float
    separatrix_gradient_relative_error: float
    separatrix_current_density_A_m2: float
    separatrix_expected_current_density_A_m2: float
    separatrix_current_density_relative_error: float
    sheet_current_integral_A_m: float
    expected_sheet_current_integral_A_m: float
    sheet_current_integral_relative_error: float
    sheet_current_passed: bool
    current_sheet_passed: bool
    force_balance_residual_linf: float
    force_balance_residual_l2: float
    force_balance_passed: bool
    passed: bool
    # --- Rotating rigid-rotor validation (trivial for theta_dot == 0) ---
    theta_dot: float = 0.0
    rotation_force_balance_residual_linf: float = 0.0
    rotation_force_balance_residual_l2: float = 0.0
    rotation_force_balance_passed: bool = True


def rotating_frc_bvp_acceptance_status() -> dict[str, object]:
    """Return the acceptance status for the rotating rigid-rotor equilibrium.

    This is a machine-readable claim boundary for B.8/FUS-C.1. It reports that
    the rotating closure is implemented against the source-verified Rostoker &
    Qerushi (2002) one-dimensional one-ion centrifugal force balance, reducing
    bit-exactly to the accepted Steinhauer no-rotation contract at
    ``theta_dot == 0``, while explicitly excluding verbatim Steinhauer 2011
    Figure 3 digitised parity (a separate external-parity gate).
    """
    return {
        "status": ROTATING_FRC_BVP_STATUS,
        "accepted_contract": "steinhauer_2011_no_rotation_analytical",
        "rotating_bvp_implemented": True,
        "rotating_closure_reference": ROTATING_FRC_BVP_ROTATING_REFERENCE,
        "solver_action": ROTATING_FRC_BVP_SOLVER_ACTION,
        "required_reference": ROTATING_FRC_BVP_REQUIRED_REFERENCE,
        "reduces_to_no_rotation_contract": True,
        "steinhauer_figure3_parity_claimed": False,
        "non_closing_references": ROTATING_FRC_BVP_NON_CLOSING_REFERENCES,
        "claim_boundary": ROTATING_FRC_BVP_CLAIM_BOUNDARY,
    }
