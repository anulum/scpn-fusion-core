# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Equilibrium Closures
"""Shared analytical closures and diagnostics for the FRC rigid-rotor equilibrium.

Middle layer of the ``frc_rigid_rotor`` package: the source-verified Steinhauer
analytical field/flux primitives, the closure-consistency residuals, the profile
and geometry helpers, and the public state accessors. Depends only on the data
contracts, so both the solver and the validation modules import it without a
cycle. Re-exported by the ``frc_rigid_rotor`` facade.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.integrate import trapezoid

from .frc_rigid_rotor_contracts import (
    ATOMIC_MASS_KG,
    DEUTERIUM_MASS_AMU,
    ELEMENTARY_CHARGE_C,
    MU_0,
    FRCEquilibriumState,
    FloatArray,
)


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
