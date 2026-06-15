# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Equilibrium
"""Field-reversed-configuration rigid-rotor equilibrium helpers.

This module implements the Steinhauer no-rotation analytical limit for the
FRC rigid-rotor workstream. Rotating BVP support is intentionally fail-closed
until the dedicated FUS-C.1 BVP implementation lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid

MU_0 = 4.0 * np.pi * 1e-7
ELEMENTARY_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_KG = 1.66053906660e-27
DEUTERIUM_MASS_AMU = 2.014

FloatArray: TypeAlias = NDArray[np.float64]

ROTATING_FRC_BVP_STATUS = "blocked_missing_verified_steinhauer_rotating_closure"
ROTATING_FRC_BVP_REQUIRED_REFERENCE = "Steinhauer 2011 Section II.B plus Figure 3 closure"
ROTATING_FRC_BVP_SOLVER_ACTION = "raise_not_implemented_for_nonzero_theta_dot"
ROTATING_FRC_BVP_CLAIM_BOUNDARY = (
    "The certified production contract is the Steinhauer 2011 no-rotation analytical "
    "FRC equilibrium. The rotating rigid-rotor BVP remains fail-closed until the "
    "Steinhauer Section II.B closure and Figure 3 reference are verified; C-2U "
    "performance and topology references are context only, not a rotating-BVP solver "
    "certification."
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


def rotating_frc_bvp_acceptance_status() -> dict[str, object]:
    """Return the fail-closed acceptance status for the unresolved rotating BVP.

    This is a machine-readable claim boundary for B.8/FUS-C.1. It deliberately
    reports the current accepted no-rotation contract and the exact external
    reference still required before nonzero ``theta_dot`` support can be
    certified.
    """
    return {
        "status": ROTATING_FRC_BVP_STATUS,
        "accepted_contract": "steinhauer_2011_no_rotation_analytical",
        "rotating_bvp_implemented": False,
        "solver_action": ROTATING_FRC_BVP_SOLVER_ACTION,
        "required_reference": ROTATING_FRC_BVP_REQUIRED_REFERENCE,
        "non_closing_references": ROTATING_FRC_BVP_NON_CLOSING_REFERENCES,
        "claim_boundary": ROTATING_FRC_BVP_CLAIM_BOUNDARY,
    }


def ion_gyroradius_m(T_i_eV: float, B_T: float, *, mass_amu: float = DEUTERIUM_MASS_AMU) -> float:
    """Return thermal ion gyroradius in metres using ``sqrt(2 m_i T_i) / (e B)``."""
    if T_i_eV <= 0.0:
        raise ValueError("T_i_eV must be positive")
    if B_T == 0.0:
        raise ValueError("B_T must be non-zero")
    ion_mass_kg = mass_amu * ATOMIC_MASS_KG
    thermal_momentum = np.sqrt(2.0 * ion_mass_kg * T_i_eV * ELEMENTARY_CHARGE_C)
    return float(thermal_momentum / (ELEMENTARY_CHARGE_C * abs(B_T)))


def _fill_equilibrium_state(
    inputs: RigidRotorFRCInputs,
    rho: FloatArray,
    psi: FloatArray,
    B_z: FloatArray,
    J_theta: FloatArray,
    p: FloatArray,
    tolerance: float,
    residual: float,
    delta: float,
    dBz_dr_analytic: FloatArray,
) -> FRCEquilibriumState:
    B_theta = np.zeros_like(B_z)
    dBz_dr = np.gradient(B_z, rho, edge_order=2)
    separatrix_bz_gradient_T_m = float(np.interp(inputs.R_s, rho, dBz_dr))
    separatrix_expected_bz_gradient_T_m = float(-inputs.B_ext / delta)
    separatrix_gradient_relative_error = abs(separatrix_bz_gradient_T_m - separatrix_expected_bz_gradient_T_m) / max(abs(separatrix_expected_bz_gradient_T_m), tolerance)
    separatrix_current_density_A_m2 = float(np.interp(inputs.R_s, rho, J_theta))
    separatrix_expected_current_density_A_m2 = float(inputs.B_ext / (MU_0 * delta))
    separatrix_current_density_relative_error = abs(separatrix_current_density_A_m2 - separatrix_expected_current_density_A_m2) / max(abs(separatrix_expected_current_density_A_m2), tolerance)
    sheet_current_integral_A_m = float(trapezoid(J_theta, rho))
    expected_sheet_current_integral_A_m = float((B_z[0] - B_z[-1]) / MU_0)
    sheet_current_integral_relative_error = abs(sheet_current_integral_A_m - expected_sheet_current_integral_A_m) / max(abs(expected_sheet_current_integral_A_m), tolerance)
    ampere_residual = _ampere_current_closure_residual(rho, B_z, J_theta)
    ampere_scale = max(tolerance, float(np.max(np.abs(dBz_dr))), float(MU_0 * np.max(np.abs(J_theta))))
    ampere_residual_linf = float(np.max(np.abs(ampere_residual)) / ampere_scale)
    ampere_residual_l2 = float(np.sqrt(np.mean((ampere_residual / ampere_scale) ** 2)))
    psi_axis_Wb = float(psi[0])
    psi_separatrix_Wb = float(np.interp(inputs.R_s, rho, psi))
    psi_span_Wb = psi_separatrix_Wb - psi_axis_Wb
    if abs(psi_span_Wb) <= tolerance:
        raise ValueError("psi separatrix span must be non-zero")
    psi_normalized = (psi - psi_axis_Wb) / psi_span_Wb
    psi_normalized_axis_error = abs(float(psi_normalized[0]))
    psi_normalized_separatrix = float(np.interp(inputs.R_s, rho, psi_normalized))
    psi_normalized_separatrix_error = abs(psi_normalized_separatrix - 1.0)
    psi_normalized_residual = _psi_normalized_closure_residual(psi, psi_axis_Wb, psi_separatrix_Wb, psi_normalized)
    psi_normalized_residual_linf = float(np.max(np.abs(psi_normalized_residual)))
    psi_normalized_monotonic_passed = _psi_normalized_monotonic_passed(rho, psi_normalized, inputs.R_s, tolerance)
    psi_normalized_bounds_passed = _psi_normalized_bounds_passed(rho, psi_normalized, inputs.R_s, tolerance)
    flux_derivative_residual = _flux_derivative_closure_residual(rho, psi, B_z)
    dpsi_dr = np.gradient(psi, rho, edge_order=2)
    flux_scale = max(tolerance, float(np.max(np.abs(rho * B_z))), float(np.max(np.abs(dpsi_dr))))
    flux_derivative_residual_linf = float(np.max(np.abs(flux_derivative_residual)) / flux_scale)
    flux_derivative_residual_l2 = float(np.sqrt(np.mean((flux_derivative_residual / flux_scale) ** 2)))
    r_null = _zero_crossing_radius(rho, B_z)
    separatrix_radius_error = abs(r_null - inputs.R_s)
    separatrix_index = int(np.argmin(np.abs(rho - r_null)))
    field_reversal = _field_reversal_passed(rho, B_z, inputs.R_s)

    input_thermal_pressure = inputs.n0 * (inputs.T_i_eV + inputs.T_e_eV) * ELEMENTARY_CHARGE_C
    external_magnetic_pressure = inputs.B_ext**2 / (2.0 * MU_0)
    thermal_energy_j = (inputs.T_i_eV + inputs.T_e_eV) * ELEMENTARY_CHARGE_C
    density_m3 = p / thermal_energy_j
    density_peak_m3 = float(np.max(density_m3))
    central_density_residual_m3 = density_peak_m3 - inputs.n0
    central_density_relative_error = abs(central_density_residual_m3) / max(density_peak_m3, inputs.n0, tolerance)
    beta = p / max(external_magnetic_pressure, tolerance)
    beta_peak = float(np.max(beta))
    beta_r_clip, beta_clip = _clip_to_separatrix(rho, beta, inputs.R_s)
    beta_separatrix_average = float(trapezoid(beta_clip * 2.0 * np.pi * beta_r_clip, beta_r_clip) / (np.pi * inputs.R_s**2))
    density_r_clip, density_clip = _clip_to_separatrix(rho, density_m3, inputs.R_s)
    particle_line_density_m1 = float(trapezoid(density_clip * 2.0 * np.pi * density_r_clip, density_r_clip))
    pressure_r_clip, pressure_clip = _clip_to_separatrix(rho, p, inputs.R_s)
    separatrix_pressure_energy_J_m = float(trapezoid(pressure_clip * 2.0 * np.pi * pressure_r_clip, pressure_r_clip))
    magnetic_deficit = external_magnetic_pressure - B_z**2 / (2.0 * MU_0)
    deficit_r_clip, deficit_clip = _clip_to_separatrix(rho, magnetic_deficit, inputs.R_s)
    separatrix_magnetic_deficit_energy_J_m = float(trapezoid(deficit_clip * 2.0 * np.pi * deficit_r_clip, deficit_r_clip))
    separatrix_energy_closure_relative_error = abs(separatrix_pressure_energy_J_m - separatrix_magnetic_deficit_energy_J_m) / max(abs(separatrix_pressure_energy_J_m), abs(separatrix_magnetic_deficit_energy_J_m), tolerance)
    pressure_balance_residual = _pressure_balance_residual(p, B_z, inputs.B_ext)
    pressure_balance_residual_linf = float(np.max(np.abs(pressure_balance_residual)) / max(external_magnetic_pressure, tolerance))
    pressure_balance_residual_l2 = float(np.sqrt(np.mean((pressure_balance_residual / max(external_magnetic_pressure, tolerance)) ** 2)))

    pressure_gradient_analytic = _pressure_gradient_from_steinhauer(B_z, dBz_dr_analytic)
    pressure_gradient_residual = _pressure_gradient_closure_residual(rho, p, pressure_gradient_analytic)
    finite_pressure_gradient = np.gradient(p, rho, edge_order=2)
    pressure_gradient_scale = max(tolerance, float(np.max(np.abs(finite_pressure_gradient))), float(np.max(np.abs(pressure_gradient_analytic))))
    pressure_gradient_residual_linf = float(np.max(np.abs(pressure_gradient_residual)) / pressure_gradient_scale)
    pressure_gradient_residual_l2 = float(np.sqrt(np.mean((pressure_gradient_residual / pressure_gradient_scale) ** 2)))
    force_residual = _radial_force_balance_residual(rho, B_z, J_theta, p)
    residual_scale = max(tolerance, float(np.max(np.abs(np.gradient(p, rho, edge_order=2)))), float(np.max(np.abs(J_theta * B_z))))
    force_balance_residual_linf = float(np.max(np.abs(force_residual)) / residual_scale)
    force_balance_residual_l2 = float(np.sqrt(np.mean((force_residual / residual_scale) ** 2)))

    magnetic_energy_density = B_z**2 / (2.0 * MU_0)
    # Stored energy density = magnetic field energy + plasma internal energy. For
    # an ideal plasma the internal energy density is (3/2) p, not p.
    total_energy_density = magnetic_energy_density + 1.5 * p
    energy_J_per_m = float(trapezoid(total_energy_density * 2.0 * np.pi * rho, rho))

    pressure_integral = float(trapezoid(p * 2.0 * np.pi * rho, rho))
    external_pressure_energy = (inputs.B_ext**2 / (2.0 * MU_0)) * np.pi * inputs.R_s**2
    pressure_balance_ratio = pressure_integral / max(external_pressure_energy, tolerance)
    s_value = _s_parameter_from_profile(rho, B_z, inputs.R_s, inputs.T_i_eV)

    return FRCEquilibriumState(
        rho=rho,
        psi=psi,
        psi_normalized=psi_normalized,
        B_z=B_z,
        B_theta=B_theta,
        J_theta=J_theta,
        p=p,
        density_m3=density_m3,
        beta=beta,
        R_null=r_null,
        target_separatrix_radius_m=inputs.R_s,
        separatrix_radius_error_m=separatrix_radius_error,
        separatrix_index=separatrix_index,
        field_reversal_passed=field_reversal,
        s_parameter=s_value,
        energy_J=energy_J_per_m,
        converged=True,
        residual=residual,
        delta=delta,
        psi_axis_Wb=psi_axis_Wb,
        psi_separatrix_Wb=psi_separatrix_Wb,
        psi_normalized_axis_error=psi_normalized_axis_error,
        psi_normalized_separatrix=psi_normalized_separatrix,
        psi_normalized_separatrix_error=psi_normalized_separatrix_error,
        psi_normalized_residual_linf=psi_normalized_residual_linf,
        psi_normalized_monotonic_passed=psi_normalized_monotonic_passed,
        psi_normalized_bounds_passed=psi_normalized_bounds_passed,
        pressure_balance_ratio=pressure_balance_ratio,
        pressure_balance_residual=pressure_balance_residual,
        pressure_balance_residual_linf=pressure_balance_residual_linf,
        pressure_balance_residual_l2=pressure_balance_residual_l2,
        pressure_gradient_analytic_Pa_m=pressure_gradient_analytic,
        pressure_gradient_residual=pressure_gradient_residual,
        pressure_gradient_residual_linf=pressure_gradient_residual_linf,
        pressure_gradient_residual_l2=pressure_gradient_residual_l2,
        peak_pressure_pa=float(np.max(p)),
        density_peak_m3=density_peak_m3,
        input_density_m3=inputs.n0,
        central_density_residual_m3=central_density_residual_m3,
        central_density_relative_error=central_density_relative_error,
        beta_peak=beta_peak,
        beta_separatrix_average=beta_separatrix_average,
        particle_line_density_m1=particle_line_density_m1,
        separatrix_pressure_energy_J_m=separatrix_pressure_energy_J_m,
        separatrix_magnetic_deficit_energy_J_m=separatrix_magnetic_deficit_energy_J_m,
        separatrix_energy_closure_relative_error=separatrix_energy_closure_relative_error,
        input_thermal_pressure_pa=input_thermal_pressure,
        thermal_pressure_ratio=input_thermal_pressure / max(external_magnetic_pressure, tolerance),
        flux_derivative_residual=flux_derivative_residual,
        flux_derivative_residual_linf=flux_derivative_residual_linf,
        flux_derivative_residual_l2=flux_derivative_residual_l2,
        ampere_residual=ampere_residual,
        ampere_residual_linf=ampere_residual_linf,
        ampere_residual_l2=ampere_residual_l2,
        peak_j_theta_A_m2=float(np.max(np.abs(J_theta))),
        separatrix_bz_gradient_T_m=separatrix_bz_gradient_T_m,
        separatrix_expected_bz_gradient_T_m=separatrix_expected_bz_gradient_T_m,
        separatrix_gradient_relative_error=separatrix_gradient_relative_error,
        separatrix_current_density_A_m2=separatrix_current_density_A_m2,
        separatrix_expected_current_density_A_m2=separatrix_expected_current_density_A_m2,
        separatrix_current_density_relative_error=separatrix_current_density_relative_error,
        sheet_current_integral_A_m=sheet_current_integral_A_m,
        expected_sheet_current_integral_A_m=expected_sheet_current_integral_A_m,
        sheet_current_integral_relative_error=sheet_current_integral_relative_error,
        force_balance_residual=force_residual,
        force_balance_residual_linf=force_balance_residual_linf,
        force_balance_residual_l2=force_balance_residual_l2,
        model="steinhauer_2011_no_rotation_analytical",
    )


def solve_frc_equilibrium(
    inputs: RigidRotorFRCInputs,
    rho_grid: FloatArray,
    *,
    solver: Literal["numpy", "rust"] = "numpy",
    tolerance: float = 1e-10,
    max_iter: int = 200,
) -> FRCEquilibriumState:
    _validate_inputs(inputs, tolerance)
    rho = _validate_grid(rho_grid, inputs.R_s)
    delta = inputs.delta if inputs.delta is not None else ion_gyroradius_m(inputs.T_i_eV, inputs.B_ext)

    if inputs.theta_dot != 0.0:
        raise NotImplementedError(
            "rotating rigid-rotor BVP support is fail-closed pending the verified "
            "FUS-C.1 derivation; only the theta_dot == 0 Steinhauer analytical limit "
            "is certified for production use."
        )

    argument = (rho**2 - inputs.R_s**2) / (2.0 * inputs.R_s * delta)
    B_z = -inputs.B_ext * np.tanh(argument)
    J_theta = _toroidal_current_density_from_steinhauer(rho, argument, inputs.B_ext, inputs.R_s, delta)
    psi = _cylindrical_flux_from_steinhauer(argument, inputs.B_ext, inputs.R_s, delta)
    external_magnetic_pressure = inputs.B_ext**2 / (2.0 * MU_0)
    p = np.maximum(external_magnetic_pressure - B_z**2 / (2.0 * MU_0), 0.0)
    residual = float(np.max(np.abs(B_z - (-inputs.B_ext * np.tanh(argument)))))

    dBz_dr_analytic = _axial_field_derivative_from_steinhauer(rho, argument, inputs.B_ext, inputs.R_s, delta)
    return _fill_equilibrium_state(inputs, rho, psi, B_z, J_theta, p, tolerance, residual, delta, dBz_dr_analytic)



def frc_no_rotation_jax_observables(
    rho_normalized_grid: FloatArray,
    *,
    n0: Any,
    T_i_eV: float,
    T_e_eV: float,
    R_s: Any,
    B_ext: Any,
    delta: Any | None,
    mass_amu: float = DEUTERIUM_MASS_AMU,
) -> dict[str, Any]:
    """Return differentiable observables for the accepted no-rotation FRC contract.

    The independent grid is ``x = r / R_s``. Keeping the grid normalised makes
    gradients with respect to ``R_s`` well-defined because the separatrix
    interval remains the fixed domain ``0 <= x <= 1``. The implemented equations
    are the same Steinhauer no-rotation field, cylindrical flux primitive,
    magnetic-pressure-balance profile, and Eq. 27 ``s`` integral used by the
    NumPy and Rust solver paths.

    This helper intentionally does not implement the rotating rigid-rotor BVP.
    """
    try:
        from jax import config as jax_config

        cast(Any, jax_config).update("jax_enable_x64", True)
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "frc_no_rotation_jax_observables requires the optional JAX dependency"
        ) from exc

    x_np = _validate_normalized_grid(rho_normalized_grid)
    _validate_positive_concrete(T_i_eV, "T_i_eV")
    _validate_positive_concrete(T_e_eV, "T_e_eV")
    _validate_positive_concrete(mass_amu, "mass_amu")
    _validate_positive_concrete(R_s, "R_s")
    _validate_nonzero_concrete(B_ext, "B_ext")
    _validate_positive_concrete(n0, "n0")
    if delta is not None:
        _validate_positive_concrete(delta, "delta")

    x = jnp.asarray(x_np, dtype=jnp.float64)
    r_s = jnp.asarray(R_s, dtype=jnp.float64)
    b_ext = jnp.asarray(B_ext, dtype=jnp.float64)
    t_i = jnp.asarray(T_i_eV, dtype=jnp.float64)
    t_e = jnp.asarray(T_e_eV, dtype=jnp.float64)
    density_at_null = jnp.asarray(n0, dtype=jnp.float64)
    ion_mass_kg = jnp.asarray(mass_amu * ATOMIC_MASS_KG, dtype=jnp.float64)
    element_charge = jnp.asarray(ELEMENTARY_CHARGE_C, dtype=jnp.float64)
    mu0 = jnp.asarray(MU_0, dtype=jnp.float64)
    pi = jnp.asarray(np.pi, dtype=jnp.float64)
    layer = (
        jnp.sqrt(2.0 * ion_mass_kg * t_i * element_charge) / (element_charge * jnp.abs(b_ext))
        if delta is None
        else jnp.asarray(delta, dtype=jnp.float64)
    )

    rho = x * r_s
    argument = (x * x - 1.0) * r_s / (2.0 * layer)
    tanh_argument = jnp.tanh(argument)
    b_z = -b_ext * tanh_argument
    b_theta = jnp.zeros_like(b_z)
    j_theta = b_ext * (1.0 - tanh_argument * tanh_argument) * x / (mu0 * layer)
    psi = -b_ext * r_s * layer * (_jax_log_cosh(jnp, argument) - _jax_log_cosh(jnp, argument[0]))
    psi_axis = psi[0]
    psi_separatrix = _jax_steinhauer_psi_at_x(jnp, 1.0, b_ext, r_s, layer)
    psi_normalized = (psi - psi_axis) / (psi_separatrix - psi_axis)

    external_pressure = b_ext * b_ext / (2.0 * mu0)
    pressure = jnp.maximum(external_pressure - b_z * b_z / (2.0 * mu0), 0.0)
    thermal_energy_j = (t_i + t_e) * element_charge
    density = pressure / thermal_energy_j
    beta = pressure / external_pressure
    magnetic_energy_density = b_z * b_z / (2.0 * mu0)
    # Internal energy density is (3/2) p (see the NumPy path).
    energy_integrand = (magnetic_energy_density + 1.5 * pressure) * 2.0 * pi * rho
    energy_j_per_m = jnp.trapezoid(energy_integrand, rho)
    pressure_integrand = pressure * 2.0 * pi * rho
    pressure_balance_ratio = jnp.trapezoid(pressure_integrand, rho) / (
        external_pressure * pi * r_s * r_s
    )

    x_sep = jnp.asarray(_normalised_separatrix_grid(x_np), dtype=jnp.float64)
    rho_sep = x_sep * r_s
    argument_sep = (x_sep * x_sep - 1.0) * r_s / (2.0 * layer)
    b_z_sep = -b_ext * jnp.tanh(argument_sep)
    pressure_sep = jnp.maximum(external_pressure - b_z_sep * b_z_sep / (2.0 * mu0), 0.0)
    separatrix_pressure_energy = jnp.trapezoid(pressure_sep * 2.0 * pi * rho_sep, rho_sep)
    magnetic_deficit_sep = external_pressure - b_z_sep * b_z_sep / (2.0 * mu0)
    separatrix_magnetic_deficit_energy = jnp.trapezoid(
        magnetic_deficit_sep * 2.0 * pi * rho_sep,
        rho_sep,
    )
    thermal_momentum = jnp.sqrt(2.0 * ion_mass_kg * t_i * element_charge)
    s_integrand = rho_sep * element_charge * jnp.abs(b_z_sep) / thermal_momentum
    s_value = jnp.trapezoid(s_integrand, rho_sep) / r_s

    return {
        "model": "steinhauer_2011_no_rotation_analytical_jax",
        "rho": rho,
        "rho_normalized": x,
        "B_z": b_z,
        "B_theta": b_theta,
        "J_theta": j_theta,
        "psi": psi,
        "psi_normalized": psi_normalized,
        "pressure": pressure,
        "density_m3": density,
        "density_at_null_m3": density_at_null,
        "beta": beta,
        "energy_J": energy_j_per_m,
        "pressure_balance_ratio": pressure_balance_ratio,
        "separatrix_pressure_energy_J_m": separatrix_pressure_energy,
        "separatrix_magnetic_deficit_energy_J_m": separatrix_magnetic_deficit_energy,
        "s_parameter": s_value,
        "delta_m": layer,
    }


def null_radius(state: FRCEquilibriumState) -> float:
    """Return the interpolated radius where the axial field reverses."""
    return _zero_crossing_radius(state.rho, state.B_z)


def s_parameter(state: FRCEquilibriumState, mass_amu: float = DEUTERIUM_MASS_AMU) -> float:
    """Return the Steinhauer Eq. 27 ``s`` parameter carried by this FRC state."""
    if mass_amu <= 0.0:
        raise ValueError("mass_amu must be positive")
    return float(state.s_parameter)


def force_balance_residual(state: FRCEquilibriumState) -> FloatArray:
    """Return radial ``dp/dr - (J x B)_r`` force-balance residual in Pa/m."""
    return state.force_balance_residual


def ampere_residual(state: FRCEquilibriumState) -> FloatArray:
    """Return radial Ampere closure residual ``mu_0 J_theta + dB_z/dr`` in T/m."""
    return state.ampere_residual


def flux_derivative_residual(state: FRCEquilibriumState) -> FloatArray:
    """Return cylindrical flux closure residual ``dpsi/dr - r B_z`` in T m."""
    return state.flux_derivative_residual


def psi_normalized_profile(state: FRCEquilibriumState) -> FloatArray:
    """Return separatrix-normalised cylindrical flux ``psi_N`` with ``psi_N(R_s)=1``."""
    return state.psi_normalized


def pressure_balance_residual(state: FRCEquilibriumState) -> FloatArray:
    """Return local pressure-balance residual ``p + B_z^2/(2 mu_0) - B_ext^2/(2 mu_0)``."""
    return state.pressure_balance_residual


def pressure_gradient_residual(state: FRCEquilibriumState) -> FloatArray:
    """Return finite-grid ``dp/dr`` minus analytical magnetic-pressure gradient."""
    return state.pressure_gradient_residual


def density_profile(state: FRCEquilibriumState) -> FloatArray:
    """Return solved density profile ``n(r) = p(r) / ((T_i + T_e) e)`` in m^-3."""
    return state.density_m3


def beta_profile(state: FRCEquilibriumState) -> FloatArray:
    """Return local plasma beta profile relative to the external axial-field pressure."""
    return state.beta


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
) -> FRCValidationReport:
    """Validate finite values, null placement, pressure peaking, and optional force balance."""
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
    pressure_balance_passed = state.pressure_balance_residual_linf <= pressure_balance_tolerance
    pressure_gradient_passed = state.pressure_gradient_residual_linf <= pressure_gradient_tolerance
    density_consistency_passed = state.central_density_relative_error <= density_tolerance
    beta_limit_passed = state.beta_peak <= 1.0 + beta_limit_tolerance
    energy_inventory_passed = (
        state.separatrix_energy_closure_relative_error <= energy_inventory_tolerance
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
    )


def _validate_inputs(inputs: RigidRotorFRCInputs, tolerance: float) -> None:
    if inputs.n0 <= 0.0:
        raise ValueError("n0 must be positive")
    if inputs.T_i_eV <= 0.0 or inputs.T_e_eV <= 0.0:
        raise ValueError("ion and electron temperatures must be positive")
    if inputs.R_s <= 0.0:
        raise ValueError("R_s must be positive")
    if inputs.B_ext == 0.0:
        raise ValueError("B_ext must be non-zero")
    if inputs.delta is not None and inputs.delta <= 0.0:
        raise ValueError("delta must be positive when provided")
    # allow theta_dot


def _validate_grid(rho_grid: FloatArray, R_s: float) -> FloatArray:
    rho = cast(FloatArray, np.asarray(rho_grid, dtype=np.float64))
    if rho.ndim != 1:
        raise ValueError("rho_grid must be one-dimensional")
    if rho.size < 4:
        raise ValueError("rho_grid must contain at least four points")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho_grid must contain finite values")
    if rho[0] != 0.0:
        raise ValueError("rho_grid must start at the magnetic axis radius 0")
    if not np.all(np.diff(rho) > 0.0):
        raise ValueError("rho_grid must be strictly increasing")
    if rho[-1] <= R_s:
        raise ValueError("rho_grid must include radii outside the separatrix radius R_s")
    return rho


def _validate_normalized_grid(rho_normalized_grid: FloatArray) -> FloatArray:
    x = cast(FloatArray, np.asarray(rho_normalized_grid, dtype=np.float64))
    if x.ndim != 1:
        raise ValueError("rho_normalized_grid must be one-dimensional")
    if x.size < 4:
        raise ValueError("rho_normalized_grid must contain at least four points")
    if not np.all(np.isfinite(x)):
        raise ValueError("rho_normalized_grid must contain finite values")
    if x[0] != 0.0:
        raise ValueError("rho_normalized_grid must start at 0")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("rho_normalized_grid must be strictly increasing")
    if x[-1] <= 1.0:
        raise ValueError("rho_normalized_grid must extend outside the separatrix x=1")
    return x


def _normalised_separatrix_grid(x: FloatArray) -> FloatArray:
    stop = int(np.searchsorted(x, 1.0, side="right"))
    x_sep = x[:stop]
    if x_sep.size == 0:
        raise ValueError("rho_normalized_grid must contain points below x=1")
    if x_sep[-1] < 1.0:
        x_sep = np.append(x_sep, 1.0)
    return x_sep


def _validate_positive_concrete(value: Any, name: str) -> None:
    if not isinstance(value, int | float | np.floating):
        return
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive")


def _validate_nonzero_concrete(value: Any, name: str) -> None:
    if not isinstance(value, int | float | np.floating):
        return
    numeric = float(value)
    if not np.isfinite(numeric) or numeric == 0.0:
        raise ValueError(f"{name} must be non-zero")


def _s_parameter_from_profile(
    rho: FloatArray,
    B_z: FloatArray,
    R_s: float,
    T_i_eV: float,
    *,
    mass_amu: float = DEUTERIUM_MASS_AMU,
) -> float:
    """Return Steinhauer Eq. 27 ``s = R_s^-1 integral_0^R_s r / rho_i(r) dr``."""
    if R_s <= 0.0:
        raise ValueError("R_s must be positive")
    if T_i_eV <= 0.0:
        raise ValueError("T_i_eV must be positive")
    if mass_amu <= 0.0:
        raise ValueError("mass_amu must be positive")
    r_clip, b_clip = _clip_to_separatrix(rho, B_z, R_s)
    ion_mass_kg = mass_amu * ATOMIC_MASS_KG
    thermal_momentum = np.sqrt(2.0 * ion_mass_kg * T_i_eV * ELEMENTARY_CHARGE_C)
    inverse_gyroradius = ELEMENTARY_CHARGE_C * np.abs(b_clip) / thermal_momentum
    integrand = r_clip * inverse_gyroradius
    return float(trapezoid(integrand, r_clip) / R_s)


def _clip_to_separatrix(
    rho: FloatArray, values: FloatArray, R_s: float
) -> tuple[FloatArray, FloatArray]:
    """Return profile samples on ``[0, R_s]`` with an interpolated separatrix endpoint."""
    stop = int(np.searchsorted(rho, R_s, side="right"))
    r_clip = rho[:stop]
    value_clip = values[:stop]
    if r_clip.size == 0:
        raise ValueError("rho_grid must contain points below R_s")
    if r_clip[-1] < R_s:
        r_clip = np.append(r_clip, R_s)
        value_clip = np.append(value_clip, np.interp(R_s, rho, values))
    return r_clip, value_clip


def _cylindrical_flux_from_steinhauer(
    argument: FloatArray,
    B_ext: float,
    R_s: float,
    delta: float,
) -> FloatArray:
    """Return analytical ``psi(r) = integral_0^r r' B_z(r') dr'`` for Eq. 7."""
    log_cosh_argument = _log_cosh(argument)
    axis_log_cosh = float(log_cosh_argument[0])
    return -B_ext * R_s * delta * (log_cosh_argument - axis_log_cosh)


def _log_cosh(values: FloatArray) -> FloatArray:
    """Return numerically stable ``log(cosh(values))``."""
    abs_values = np.abs(values)
    return cast(FloatArray, abs_values + np.log1p(np.exp(-2.0 * abs_values)) - np.log(2.0))


def _jax_log_cosh(jnp: Any, values: Any) -> Any:
    """Return a JAX-traceable numerically stable ``log(cosh(values))``."""
    abs_values = jnp.abs(values)
    return abs_values + jnp.log1p(jnp.exp(-2.0 * abs_values)) - jnp.log(2.0)


def _jax_steinhauer_psi_at_x(jnp: Any, x: float, b_ext: Any, r_s: Any, delta: Any) -> Any:
    """Return the JAX-traceable Steinhauer flux primitive at normalised radius ``x``."""
    argument = (x * x - 1.0) * r_s / (2.0 * delta)
    axis_argument = -r_s / (2.0 * delta)
    return -b_ext * r_s * delta * (_jax_log_cosh(jnp, argument) - _jax_log_cosh(jnp, axis_argument))


def _toroidal_current_density_from_steinhauer(
    rho: FloatArray,
    argument: FloatArray,
    B_ext: float,
    R_s: float,
    delta: float,
) -> FloatArray:
    """Return analytical ``J_theta = -mu_0^-1 dB_z/dr`` for Eq. 7."""
    tanh_argument = np.tanh(argument)
    sech_squared = 1.0 - tanh_argument**2
    return B_ext * sech_squared * rho / (MU_0 * R_s * delta)


def _axial_field_derivative_from_steinhauer(
    rho: FloatArray,
    argument: FloatArray,
    B_ext: float,
    R_s: float,
    delta: float,
) -> FloatArray:
    """Return analytical ``dB_z/dr`` for the accepted Steinhauer field."""
    tanh_argument = np.tanh(argument)
    sech_squared = 1.0 - tanh_argument**2
    return -B_ext * sech_squared * rho / (R_s * delta)


def _pressure_gradient_from_steinhauer(B_z: FloatArray, dBz_dr: FloatArray) -> FloatArray:
    """Return analytical ``dp/dr = -(B_z / mu_0) dB_z/dr`` in Pa/m."""
    return -(B_z * dBz_dr) / MU_0


def _pressure_gradient_closure_residual(
    rho: FloatArray,
    p: FloatArray,
    pressure_gradient_analytic: FloatArray,
) -> FloatArray:
    """Return finite-grid ``dp/dr`` minus the analytical pressure gradient."""
    finite_pressure_gradient = np.gradient(p, rho, edge_order=2)
    return cast(FloatArray, finite_pressure_gradient - pressure_gradient_analytic)


def _ampere_current_closure_residual(
    rho: FloatArray, B_z: FloatArray, J_theta: FloatArray
) -> FloatArray:
    """Return ``mu_0 J_theta + dB_z/dr``; zero means Ampere's law closes."""
    d_bz_dr = np.gradient(B_z, rho, edge_order=2)
    return cast(FloatArray, MU_0 * J_theta + d_bz_dr)


def _flux_derivative_closure_residual(
    rho: FloatArray, psi: FloatArray, B_z: FloatArray
) -> FloatArray:
    """Return ``dpsi/dr - r B_z``; zero means the flux primitive closes."""
    dpsi_dr = np.gradient(psi, rho, edge_order=2)
    return cast(FloatArray, dpsi_dr - rho * B_z)


def _psi_normalized_closure_residual(
    psi: FloatArray,
    psi_axis_Wb: float,
    psi_separatrix_Wb: float,
    psi_normalized: FloatArray,
) -> FloatArray:
    """Return consistency residual for ``psi_N = (psi - psi_axis)/(psi_sep - psi_axis)``."""
    span = psi_separatrix_Wb - psi_axis_Wb
    if span == 0.0:
        raise ValueError("psi separatrix span must be non-zero")
    expected = (psi - psi_axis_Wb) / span
    return psi_normalized - expected


def _psi_normalized_monotonic_passed(
    rho: FloatArray,
    psi_normalized: FloatArray,
    R_s: float,
    tolerance: float,
) -> bool:
    """Return whether ``psi_N`` is monotone on the resolved separatrix interval."""
    _, psi_clip = _clip_to_separatrix(rho, psi_normalized, R_s)
    return bool(np.all(np.diff(psi_clip) >= -tolerance))


def _psi_normalized_bounds_passed(
    rho: FloatArray,
    psi_normalized: FloatArray,
    R_s: float,
    tolerance: float,
) -> bool:
    """Return whether ``psi_N`` remains inside the expected [0, 1] interval on ``[0, R_s]``."""
    _, psi_clip = _clip_to_separatrix(rho, psi_normalized, R_s)
    return bool(np.min(psi_clip) >= -tolerance and np.max(psi_clip) <= 1.0 + tolerance)


def _pressure_balance_residual(p: FloatArray, B_z: FloatArray, B_ext: float) -> FloatArray:
    """Return ``p + B_z^2/(2 mu_0) - B_ext^2/(2 mu_0)`` in Pa."""
    return p + B_z**2 / (2.0 * MU_0) - B_ext**2 / (2.0 * MU_0)


def _radial_force_balance_residual(
    rho: FloatArray,
    B_z: FloatArray,
    J_theta: FloatArray,
    p: FloatArray,
) -> FloatArray:
    dp_dr = np.gradient(p, rho, edge_order=2)
    j_cross_b_r = J_theta * B_z
    return cast(FloatArray, dp_dr - j_cross_b_r)


def _zero_crossing_radius(rho: FloatArray, values: FloatArray) -> float:
    sign_changes = np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:]))
    if sign_changes.size == 0:
        return float(rho[int(np.argmin(np.abs(values)))])
    i = int(sign_changes[0])
    y0 = float(values[i])
    y1 = float(values[i + 1])
    if y1 == y0:
        return float(rho[i])
    weight = -y0 / (y1 - y0)
    return float(rho[i] + weight * (rho[i + 1] - rho[i]))


def _field_reversal_passed(rho: FloatArray, B_z: FloatArray, R_s: float) -> bool:
    """Return whether the axial field reverses sign across the separatrix."""
    inner_indices = np.flatnonzero(rho < R_s)
    outer_indices = np.flatnonzero(rho > R_s)
    if inner_indices.size == 0 or outer_indices.size == 0:
        return False
    inner_field = float(B_z[int(inner_indices[-1])])
    outer_field = float(B_z[int(outer_indices[0])])
    return bool(
        np.isfinite(inner_field) and np.isfinite(outer_field) and inner_field * outer_field < 0.0
    )
