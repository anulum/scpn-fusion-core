# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Equilibrium Solver
"""Solver and input-guard layer for the FRC rigid-rotor equilibrium.

Consumer layer of the ``frc_rigid_rotor`` package: builds an
:class:`FRCEquilibriumState` from :class:`RigidRotorFRCInputs` on a radial grid.
Covers the accepted Steinhauer no-rotation contract and the Rostoker & Qerushi
(2002) rotating rigid-rotor closure, the differentiable JAX observable helper,
and the solver/observable input guards. Imports the shared analytical closures
from ``frc_rigid_rotor_closures`` and the data contracts from
``frc_rigid_rotor_contracts``, so it carries no validation logic and stays free
of an import cycle. Re-exported by the ``frc_rigid_rotor`` facade.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from scipy.integrate import trapezoid

from .frc_rigid_rotor_closures import (
    _ampere_current_closure_residual,
    _axial_field_derivative_from_steinhauer,
    _clip_to_separatrix,
    _cylindrical_flux_from_steinhauer,
    _field_reversal_passed,
    _flux_derivative_closure_residual,
    _jax_log_cosh,
    _jax_steinhauer_psi_at_x,
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
)
from .frc_rigid_rotor_contracts import (
    ATOMIC_MASS_KG,
    DEUTERIUM_MASS_AMU,
    ELEMENTARY_CHARGE_C,
    FRCEquilibriumState,
    FloatArray,
    MU_0,
    ROTATING_FRC_BVP_ROTATING_REFERENCE,
    RigidRotorFRCInputs,
)


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
    *,
    mass_amu: float = DEUTERIUM_MASS_AMU,
    pressure_clipped_fraction: float = 0.0,
) -> FRCEquilibriumState:
    B_theta = np.zeros_like(B_z)
    dBz_dr = np.gradient(B_z, rho, edge_order=2)
    separatrix_bz_gradient_T_m = float(np.interp(inputs.R_s, rho, dBz_dr))
    separatrix_expected_bz_gradient_T_m = float(-inputs.B_ext / delta)
    separatrix_gradient_relative_error = abs(
        separatrix_bz_gradient_T_m - separatrix_expected_bz_gradient_T_m
    ) / max(abs(separatrix_expected_bz_gradient_T_m), tolerance)
    separatrix_current_density_A_m2 = float(np.interp(inputs.R_s, rho, J_theta))
    separatrix_expected_current_density_A_m2 = float(inputs.B_ext / (MU_0 * delta))
    separatrix_current_density_relative_error = abs(
        separatrix_current_density_A_m2 - separatrix_expected_current_density_A_m2
    ) / max(abs(separatrix_expected_current_density_A_m2), tolerance)
    sheet_current_integral_A_m = float(trapezoid(J_theta, rho))
    expected_sheet_current_integral_A_m = float((B_z[0] - B_z[-1]) / MU_0)
    sheet_current_integral_relative_error = abs(
        sheet_current_integral_A_m - expected_sheet_current_integral_A_m
    ) / max(abs(expected_sheet_current_integral_A_m), tolerance)
    ampere_residual = _ampere_current_closure_residual(rho, B_z, J_theta)
    ampere_scale = max(
        tolerance, float(np.max(np.abs(dBz_dr))), float(MU_0 * np.max(np.abs(J_theta)))
    )
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
    psi_normalized_residual = _psi_normalized_closure_residual(
        psi, psi_axis_Wb, psi_separatrix_Wb, psi_normalized
    )
    psi_normalized_residual_linf = float(np.max(np.abs(psi_normalized_residual)))
    psi_normalized_monotonic_passed = _psi_normalized_monotonic_passed(
        rho, psi_normalized, inputs.R_s, tolerance
    )
    psi_normalized_bounds_passed = _psi_normalized_bounds_passed(
        rho, psi_normalized, inputs.R_s, tolerance
    )
    flux_derivative_residual = _flux_derivative_closure_residual(rho, psi, B_z)
    dpsi_dr = np.gradient(psi, rho, edge_order=2)
    flux_scale = max(tolerance, float(np.max(np.abs(rho * B_z))), float(np.max(np.abs(dpsi_dr))))
    flux_derivative_residual_linf = float(np.max(np.abs(flux_derivative_residual)) / flux_scale)
    flux_derivative_residual_l2 = float(
        np.sqrt(np.mean((flux_derivative_residual / flux_scale) ** 2))
    )
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
    central_density_relative_error = abs(central_density_residual_m3) / max(
        density_peak_m3, inputs.n0, tolerance
    )
    beta = p / max(external_magnetic_pressure, tolerance)
    beta_peak = float(np.max(beta))
    beta_r_clip, beta_clip = _clip_to_separatrix(rho, beta, inputs.R_s)
    beta_separatrix_average = float(
        trapezoid(beta_clip * 2.0 * np.pi * beta_r_clip, beta_r_clip) / (np.pi * inputs.R_s**2)
    )
    density_r_clip, density_clip = _clip_to_separatrix(rho, density_m3, inputs.R_s)
    particle_line_density_m1 = float(
        trapezoid(density_clip * 2.0 * np.pi * density_r_clip, density_r_clip)
    )
    pressure_r_clip, pressure_clip = _clip_to_separatrix(rho, p, inputs.R_s)
    separatrix_pressure_energy_J_m = float(
        trapezoid(pressure_clip * 2.0 * np.pi * pressure_r_clip, pressure_r_clip)
    )
    magnetic_deficit = external_magnetic_pressure - B_z**2 / (2.0 * MU_0)
    deficit_r_clip, deficit_clip = _clip_to_separatrix(rho, magnetic_deficit, inputs.R_s)
    separatrix_magnetic_deficit_energy_J_m = float(
        trapezoid(deficit_clip * 2.0 * np.pi * deficit_r_clip, deficit_r_clip)
    )
    separatrix_energy_closure_relative_error = abs(
        separatrix_pressure_energy_J_m - separatrix_magnetic_deficit_energy_J_m
    ) / max(
        abs(separatrix_pressure_energy_J_m), abs(separatrix_magnetic_deficit_energy_J_m), tolerance
    )
    pressure_balance_residual = _pressure_balance_residual(p, B_z, inputs.B_ext)
    pressure_balance_residual_linf = float(
        np.max(np.abs(pressure_balance_residual)) / max(external_magnetic_pressure, tolerance)
    )
    pressure_balance_residual_l2 = float(
        np.sqrt(
            np.mean((pressure_balance_residual / max(external_magnetic_pressure, tolerance)) ** 2)
        )
    )

    pressure_gradient_analytic = _pressure_gradient_from_steinhauer(B_z, dBz_dr_analytic)
    pressure_gradient_residual = _pressure_gradient_closure_residual(
        rho, p, pressure_gradient_analytic
    )
    finite_pressure_gradient = np.gradient(p, rho, edge_order=2)
    pressure_gradient_scale = max(
        tolerance,
        float(np.max(np.abs(finite_pressure_gradient))),
        float(np.max(np.abs(pressure_gradient_analytic))),
    )
    pressure_gradient_residual_linf = float(
        np.max(np.abs(pressure_gradient_residual)) / pressure_gradient_scale
    )
    pressure_gradient_residual_l2 = float(
        np.sqrt(np.mean((pressure_gradient_residual / pressure_gradient_scale) ** 2))
    )
    force_residual = _radial_force_balance_residual(rho, B_z, J_theta, p)
    residual_scale = max(
        tolerance,
        float(np.max(np.abs(np.gradient(p, rho, edge_order=2)))),
        float(np.max(np.abs(J_theta * B_z))),
    )
    force_balance_residual_linf = float(np.max(np.abs(force_residual)) / residual_scale)
    force_balance_residual_l2 = float(np.sqrt(np.mean((force_residual / residual_scale) ** 2)))

    # Mass density from the local pressure and the (T_i + T_e) closure: with a
    # single ion species rho = m_i n and n = p / ((T_i + T_e) e), so rho = kappa p.
    ion_mass_kg = mass_amu * ATOMIC_MASS_KG
    thermal_energy_per_particle_J = (inputs.T_i_eV + inputs.T_e_eV) * ELEMENTARY_CHARGE_C
    mass_density = ion_mass_kg * (p / thermal_energy_per_particle_J)

    omega = inputs.theta_dot
    # Verified rotating rigid-rotor force balance (Rostoker & Qerushi 2002):
    #   d/dr[p + B_z^2/(2 mu_0)] = rho omega^2 r   (centrifugal source on the RHS).
    # Written with the analytic current this is  dp/dr - J_theta B_z - rho omega^2 r.
    centrifugal_source = mass_density * omega**2 * rho
    rotation_force_residual = (
        _radial_force_balance_residual(rho, B_z, J_theta, p) - centrifugal_source
    )
    rotation_residual_scale = max(
        tolerance,
        float(np.max(np.abs(np.gradient(p, rho, edge_order=2)))),
        float(np.max(np.abs(J_theta * B_z))),
        float(np.max(np.abs(centrifugal_source))),
    )
    rotation_force_balance_residual_linf = float(
        np.max(np.abs(rotation_force_residual)) / rotation_residual_scale
    )
    rotation_force_balance_residual_l2 = float(
        np.sqrt(np.mean((rotation_force_residual / rotation_residual_scale) ** 2))
    )
    # Rotation Mach number relative to the isothermal ion-acoustic speed at R_s.
    sound_speed = np.sqrt(thermal_energy_per_particle_J / ion_mass_kg)
    rotation_mach_number = float(abs(omega) * inputs.R_s / max(sound_speed, tolerance))
    rotation_pressure_peak_radius_m = float(rho[int(np.argmax(p))])

    magnetic_energy_density = B_z**2 / (2.0 * MU_0)
    # Stored energy density = magnetic field energy + plasma internal energy + bulk
    # rotational kinetic energy. For an ideal plasma the internal energy density is
    # (3/2) p, not p; the rotational term (1/2) rho (omega r)^2 vanishes at omega = 0.
    rotational_kinetic_density = 0.5 * mass_density * (omega * rho) ** 2
    total_energy_density = magnetic_energy_density + 1.5 * p + rotational_kinetic_density
    energy_J_per_m = float(trapezoid(total_energy_density * 2.0 * np.pi * rho, rho))

    pressure_integral = float(trapezoid(p * 2.0 * np.pi * rho, rho))
    external_pressure_energy = (inputs.B_ext**2 / (2.0 * MU_0)) * np.pi * inputs.R_s**2
    pressure_balance_ratio = pressure_integral / max(external_pressure_energy, tolerance)
    s_value = _s_parameter_from_profile(rho, B_z, inputs.R_s, inputs.T_i_eV)
    is_rotating = omega != 0.0

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
        model=(
            "rostoker_qerushi_2002_rotating_rigid_rotor"
            if is_rotating
            else "steinhauer_2011_no_rotation_analytical"
        ),
        theta_dot=float(omega),
        rotation_reference=(ROTATING_FRC_BVP_ROTATING_REFERENCE if is_rotating else ""),
        centrifugal_source_Pa_m=centrifugal_source,
        rotation_force_balance_residual=rotation_force_residual,
        rotation_force_balance_residual_linf=rotation_force_balance_residual_linf,
        rotation_force_balance_residual_l2=rotation_force_balance_residual_l2,
        rotation_mach_number=rotation_mach_number,
        rotation_pressure_peak_radius_m=rotation_pressure_peak_radius_m,
        pressure_clipped_fraction=float(pressure_clipped_fraction),
    )


def solve_frc_equilibrium(
    inputs: RigidRotorFRCInputs,
    rho_grid: FloatArray,
    *,
    solver: Literal["numpy", "rust"] = "numpy",
    tolerance: float = 1e-10,
    max_iter: int = 200,
) -> FRCEquilibriumState:
    """Solve the fail-closed Steinhauer no-rotation rigid-rotor FRC equilibrium.

    Two verified closures are covered on a finite, strictly increasing radial
    grid that starts at the magnetic axis and extends beyond the requested
    separatrix radius:

    * ``theta_dot == 0`` — the accepted Steinhauer no-rotation magnetostatic
      pressure balance ``p = B_ext^2/(2 mu_0) - B_z^2/(2 mu_0)`` (byte-unchanged
      production contract).
    * ``theta_dot != 0`` — the Rostoker & Qerushi (2002) one-dimensional one-ion
      rotating rigid-rotor closure. The rigid-rotor axial field/flux ansatz is
      retained and the pressure inherits the verified centrifugal density factor
      ``exp[(1/2) m_i omega^2 r^2 / ((T_i + T_e) e)]`` via
      :func:`_rotating_pressure_from_density_closure`. As ``theta_dot -> 0`` the
      factor tends to unity and the solution reduces analytically to the
      no-rotation contract, with the centrifugal force-balance residual reported
      as the fixed-field consistency diagnostic.
    """
    _validate_inputs(inputs, tolerance)
    rho = _validate_grid(rho_grid, inputs.R_s)
    delta = (
        inputs.delta if inputs.delta is not None else ion_gyroradius_m(inputs.T_i_eV, inputs.B_ext)
    )

    argument = (rho**2 - inputs.R_s**2) / (2.0 * inputs.R_s * delta)
    B_z = -inputs.B_ext * np.tanh(argument)
    J_theta = _toroidal_current_density_from_steinhauer(
        rho, argument, inputs.B_ext, inputs.R_s, delta
    )
    psi = _cylindrical_flux_from_steinhauer(argument, inputs.B_ext, inputs.R_s, delta)
    external_magnetic_pressure = inputs.B_ext**2 / (2.0 * MU_0)

    magnetostatic_pressure = np.maximum(external_magnetic_pressure - B_z**2 / (2.0 * MU_0), 0.0)
    if inputs.theta_dot == 0.0:
        p = magnetostatic_pressure
        pressure_clipped_fraction = 0.0
    else:
        # Rostoker & Qerushi (2002) rigid-rotor centrifugal density modulation on
        # the diamagnetic FRC profile; manifestly non-negative and reducing to the
        # magnetostatic contract as theta_dot -> 0.
        p = _rotating_pressure_from_density_closure(
            rho,
            magnetostatic_pressure,
            inputs.theta_dot,
            inputs.T_i_eV + inputs.T_e_eV,
        )
        pressure_clipped_fraction = 0.0

    residual = float(np.max(np.abs(B_z - (-inputs.B_ext * np.tanh(argument)))))

    dBz_dr_analytic = _axial_field_derivative_from_steinhauer(
        rho, argument, inputs.B_ext, inputs.R_s, delta
    )
    return _fill_equilibrium_state(
        inputs,
        rho,
        psi,
        B_z,
        J_theta,
        p,
        tolerance,
        residual,
        delta,
        dBz_dr_analytic,
        pressure_clipped_fraction=pressure_clipped_fraction,
    )


def _rotating_pressure_from_density_closure(
    rho: FloatArray,
    magnetostatic_pressure: FloatArray,
    theta_dot: float,
    thermal_sum_eV: float,
    *,
    mass_amu: float = DEUTERIUM_MASS_AMU,
    exponent_cap: float = 300.0,
) -> FloatArray:
    """Return the rotating rigid-rotor pressure from the centrifugal density closure.

    The verified Rostoker & Qerushi (2002) / US 6,664,740 B2 rigid-rotor density
    closure adds a centrifugal factor to the equilibrium density,
    ``n(r) = n_static(r) exp[(1/2) m_i omega^2 r^2 / (T_i + T_e)]``. With the
    isothermal ``(T_i + T_e)`` closure the pressure inherits the same factor,

    ``p(r) = p_static(r) exp[(1/2) m_i omega^2 r^2 / ((T_i + T_e) e)]``,

    where ``p_static`` is the accepted magnetostatic FRC profile. The factor is
    unity at ``omega = 0`` (recovering the no-rotation contract) and grows with
    radius, redistributing pressure outward under rotation. The result is
    manifestly non-negative because ``p_static >= 0`` and the exponential is
    positive. The additional convention-dependent rotation-flux term
    ``e omega psi`` of the full closure is deliberately excluded here (its sign
    convention is not verifiable from the open-access sources); the residual it
    would remove is reported as the fixed-field consistency diagnostic.

    Parameters
    ----------
    rho:
        Radial grid in metres.
    magnetostatic_pressure:
        Accepted no-rotation FRC pressure profile ``p_static(r)`` in Pa.
    theta_dot:
        Rigid ion rotation rate ``omega`` in rad/s (non-zero).
    thermal_sum_eV:
        ``T_i + T_e`` in eV, setting the centrifugal scale.
    mass_amu:
        Ion mass in atomic mass units (deuterium by default).
    exponent_cap:
        Numerical overflow guard on the peak exponent; strongly super-sonic
        drives that exceed it fall outside the reduced-closure validity range.

    Returns
    -------
    numpy.ndarray
        The rotating pressure profile in Pa (non-negative).
    """
    ion_mass_kg = mass_amu * ATOMIC_MASS_KG
    thermal_energy_J = thermal_sum_eV * ELEMENTARY_CHARGE_C
    exponent = 0.5 * ion_mass_kg * theta_dot**2 * rho**2 / thermal_energy_J
    e_max = float(exponent[-1])
    if e_max > exponent_cap:
        raise ValueError(
            "rotation drive exceeds the rotating-closure validity cap "
            f"((1/2) m_i omega^2 r_max^2 / (T_i + T_e) = {e_max:.3g} > "
            f"{exponent_cap:.3g}); the reduced rigid-rotor closure is only valid "
            "for sub-sonic to transonic rotation."
        )
    return cast(FloatArray, magnetostatic_pressure * np.exp(exponent))


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
