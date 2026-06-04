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
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid

MU_0 = 4.0 * np.pi * 1e-7
ELEMENTARY_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_KG = 1.66053906660e-27
DEUTERIUM_MASS_AMU = 2.014

FloatArray: TypeAlias = NDArray[np.float64]


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
    B_z: FloatArray
    B_theta: FloatArray
    J_theta: FloatArray
    p: FloatArray
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
    pressure_balance_ratio: float
    pressure_balance_residual: FloatArray
    pressure_balance_residual_linf: float
    pressure_balance_residual_l2: float
    peak_pressure_pa: float
    input_thermal_pressure_pa: float
    thermal_pressure_ratio: float
    flux_derivative_residual: FloatArray
    flux_derivative_residual_linf: float
    flux_derivative_residual_l2: float
    ampere_residual: FloatArray
    ampere_residual_linf: float
    ampere_residual_l2: float
    peak_j_theta_A_m2: float
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
    pressure_peak_error_m: float
    edge_field_error_T: float
    pressure_balance_ratio: float
    pressure_balance_residual_linf: float
    pressure_balance_residual_l2: float
    pressure_balance_passed: bool
    thermal_pressure_ratio: float
    flux_derivative_residual_linf: float
    flux_derivative_residual_l2: float
    flux_closure_passed: bool
    ampere_residual_linf: float
    ampere_residual_l2: float
    ampere_closure_passed: bool
    force_balance_residual_linf: float
    force_balance_residual_l2: float
    force_balance_passed: bool
    passed: bool


def ion_gyroradius_m(T_i_eV: float, B_T: float, *, mass_amu: float = DEUTERIUM_MASS_AMU) -> float:
    """Return thermal ion gyroradius in metres using ``sqrt(2 m_i T_i) / (e B)``."""
    if T_i_eV <= 0.0:
        raise ValueError("T_i_eV must be positive")
    if B_T == 0.0:
        raise ValueError("B_T must be non-zero")
    ion_mass_kg = mass_amu * ATOMIC_MASS_KG
    thermal_momentum = np.sqrt(2.0 * ion_mass_kg * T_i_eV * ELEMENTARY_CHARGE_C)
    return float(thermal_momentum / (ELEMENTARY_CHARGE_C * abs(B_T)))


def solve_frc_equilibrium(
    inputs: RigidRotorFRCInputs,
    rho_grid: FloatArray,
    *,
    solver: Literal["numpy"] = "numpy",
    tolerance: float = 1e-10,
    max_iter: int = 200,
) -> FRCEquilibriumState:
    """Solve the Steinhauer no-rotation FRC analytical limit on a radial grid.

    The implemented magnetic field is Steinhauer (2011), Eq. 7:
    ``B_z(r) = -B_ext tanh((r^2 - R_s^2) / (2 R_s delta))``.
    Full rotating rigid-rotor BVP support is deliberately rejected until its
    dedicated implementation and parity tests are added.
    """
    del max_iter
    if solver != "numpy":
        raise ValueError("only the numpy analytical no-rotation path is implemented")
    _validate_inputs(inputs, tolerance)
    rho = _validate_grid(rho_grid, inputs.R_s)
    delta = inputs.delta if inputs.delta is not None else ion_gyroradius_m(inputs.T_i_eV, inputs.B_ext)

    argument = (rho**2 - inputs.R_s**2) / (2.0 * inputs.R_s * delta)
    B_z = -inputs.B_ext * np.tanh(argument)
    B_theta = np.zeros_like(B_z)
    J_theta = _toroidal_current_density_from_steinhauer(
        rho,
        argument,
        inputs.B_ext,
        inputs.R_s,
        delta,
    )
    ampere_residual = _ampere_current_closure_residual(rho, B_z, J_theta)
    ampere_scale = max(
        tolerance,
        float(np.max(np.abs(np.gradient(B_z, rho, edge_order=2)))),
        float(MU_0 * np.max(np.abs(J_theta))),
    )
    ampere_residual_linf = float(np.max(np.abs(ampere_residual)) / ampere_scale)
    ampere_residual_l2 = float(np.sqrt(np.mean((ampere_residual / ampere_scale) ** 2)))
    psi = _cylindrical_flux_from_steinhauer(argument, inputs.B_ext, inputs.R_s, delta)
    flux_derivative_residual = _flux_derivative_closure_residual(rho, psi, B_z)
    dpsi_dr = cast(FloatArray, np.gradient(psi, rho, edge_order=2))
    flux_scale = max(tolerance, float(np.max(np.abs(rho * B_z))), float(np.max(np.abs(dpsi_dr))))
    flux_derivative_residual_linf = float(np.max(np.abs(flux_derivative_residual)) / flux_scale)
    flux_derivative_residual_l2 = float(np.sqrt(np.mean((flux_derivative_residual / flux_scale) ** 2)))
    r_null = _zero_crossing_radius(rho, B_z)
    separatrix_radius_error = abs(r_null - inputs.R_s)
    separatrix_index = int(np.argmin(np.abs(rho - r_null)))
    field_reversal = _field_reversal_passed(rho, B_z, inputs.R_s)

    input_thermal_pressure = inputs.n0 * (inputs.T_i_eV + inputs.T_e_eV) * ELEMENTARY_CHARGE_C
    external_magnetic_pressure = inputs.B_ext**2 / (2.0 * MU_0)
    p = cast(FloatArray, external_magnetic_pressure - B_z**2 / (2.0 * MU_0))
    p = cast(FloatArray, np.maximum(p, 0.0))
    pressure_balance_residual = _pressure_balance_residual(p, B_z, inputs.B_ext)
    pressure_balance_residual_linf = float(np.max(np.abs(pressure_balance_residual)) / max(external_magnetic_pressure, tolerance))
    pressure_balance_residual_l2 = float(
        np.sqrt(np.mean((pressure_balance_residual / max(external_magnetic_pressure, tolerance)) ** 2))
    )
    force_residual = _radial_force_balance_residual(rho, B_z, J_theta, p)
    residual_scale = max(
        tolerance,
        float(np.max(np.abs(np.gradient(p, rho, edge_order=2)))),
        float(np.max(np.abs(J_theta * B_z))),
    )
    force_balance_residual_linf = float(np.max(np.abs(force_residual)) / residual_scale)
    force_balance_residual_l2 = float(np.sqrt(np.mean((force_residual / residual_scale) ** 2)))

    magnetic_energy_density = B_z**2 / (2.0 * MU_0)
    total_energy_density = magnetic_energy_density + p
    energy_J_per_m = float(trapezoid(total_energy_density * 2.0 * np.pi * rho, rho))

    pressure_integral = float(trapezoid(p * 2.0 * np.pi * rho, rho))
    external_pressure_energy = (inputs.B_ext**2 / (2.0 * MU_0)) * np.pi * inputs.R_s**2
    pressure_balance_ratio = pressure_integral / max(external_pressure_energy, tolerance)
    s_value = _s_parameter_from_profile(rho, B_z, inputs.R_s, inputs.T_i_eV)

    return FRCEquilibriumState(
        rho=rho,
        psi=psi,
        B_z=B_z,
        B_theta=B_theta,
        J_theta=J_theta,
        p=p,
        R_null=r_null,
        target_separatrix_radius_m=inputs.R_s,
        separatrix_radius_error_m=float(separatrix_radius_error),
        separatrix_index=separatrix_index,
        field_reversal_passed=field_reversal,
        s_parameter=s_value,
        energy_J=energy_J_per_m,
        converged=True,
        residual=float(np.max(np.abs(B_z - (-inputs.B_ext * np.tanh(argument))))),
        delta=delta,
        pressure_balance_ratio=float(pressure_balance_ratio),
        pressure_balance_residual=pressure_balance_residual,
        pressure_balance_residual_linf=pressure_balance_residual_linf,
        pressure_balance_residual_l2=pressure_balance_residual_l2,
        peak_pressure_pa=float(np.max(p)),
        input_thermal_pressure_pa=float(input_thermal_pressure),
        thermal_pressure_ratio=float(input_thermal_pressure / max(external_magnetic_pressure, tolerance)),
        flux_derivative_residual=flux_derivative_residual,
        flux_derivative_residual_linf=flux_derivative_residual_linf,
        flux_derivative_residual_l2=flux_derivative_residual_l2,
        ampere_residual=ampere_residual,
        ampere_residual_linf=ampere_residual_linf,
        ampere_residual_l2=ampere_residual_l2,
        peak_j_theta_A_m2=float(np.max(np.abs(J_theta))),
        force_balance_residual=force_residual,
        force_balance_residual_linf=force_balance_residual_linf,
        force_balance_residual_l2=force_balance_residual_l2,
        model="steinhauer_2011_no_rotation_analytical",
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


def pressure_balance_residual(state: FRCEquilibriumState) -> FloatArray:
    """Return local pressure-balance residual ``p + B_z^2/(2 mu_0) - B_ext^2/(2 mu_0)``."""
    return state.pressure_balance_residual


def validate_equilibrium(
    state: FRCEquilibriumState,
    *,
    tolerance: float = 1e-6,
    flux_tolerance: float = 2e-2,
    pressure_balance_tolerance: float = 2e-2,
    ampere_tolerance: float = 2e-2,
    force_balance_tolerance: float | None = None,
) -> FRCValidationReport:
    """Validate finite values, null placement, pressure peaking, and optional force balance."""
    finite = all(
        bool(np.all(np.isfinite(values)))
        for values in (
            state.rho,
            state.psi,
            state.B_z,
            state.B_theta,
            state.J_theta,
            state.p,
            state.pressure_balance_residual,
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
    flux_closure_passed = state.flux_derivative_residual_linf <= flux_tolerance
    ampere_closure_passed = state.ampere_residual_linf <= ampere_tolerance
    force_balance_passed = (
        force_balance_tolerance is None or state.force_balance_residual_linf <= force_balance_tolerance
    )
    passed = (
        geometric_passed
        and pressure_balance_passed
        and flux_closure_passed
        and ampere_closure_passed
        and force_balance_passed
    )
    return FRCValidationReport(
        finite=finite,
        monotonic_grid=monotonic_grid,
        null_error_m=float(null_error),
        target_separatrix_radius_m=state.target_separatrix_radius_m,
        field_reversal_passed=bool(field_reversal_passed),
        pressure_peak_error_m=float(pressure_peak_error),
        edge_field_error_T=float(edge_field_error),
        pressure_balance_ratio=state.pressure_balance_ratio,
        pressure_balance_residual_linf=state.pressure_balance_residual_linf,
        pressure_balance_residual_l2=state.pressure_balance_residual_l2,
        pressure_balance_passed=bool(pressure_balance_passed),
        thermal_pressure_ratio=state.thermal_pressure_ratio,
        flux_derivative_residual_linf=state.flux_derivative_residual_linf,
        flux_derivative_residual_l2=state.flux_derivative_residual_l2,
        flux_closure_passed=bool(flux_closure_passed),
        ampere_residual_linf=state.ampere_residual_linf,
        ampere_residual_l2=state.ampere_residual_l2,
        ampere_closure_passed=bool(ampere_closure_passed),
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
    if abs(inputs.theta_dot) > tolerance:
        raise NotImplementedError("rotating rigid-rotor BVP support is not implemented yet")


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
    inverse_gyroradius = cast(FloatArray, ELEMENTARY_CHARGE_C * np.abs(b_clip) / thermal_momentum)
    integrand = cast(FloatArray, r_clip * inverse_gyroradius)
    return float(trapezoid(integrand, r_clip) / R_s)


def _clip_to_separatrix(rho: FloatArray, values: FloatArray, R_s: float) -> tuple[FloatArray, FloatArray]:
    """Return profile samples on ``[0, R_s]`` with an interpolated separatrix endpoint."""
    stop = int(np.searchsorted(rho, R_s, side="right"))
    r_clip = cast(FloatArray, rho[:stop])
    value_clip = cast(FloatArray, values[:stop])
    if r_clip.size == 0:
        raise ValueError("rho_grid must contain points below R_s")
    if r_clip[-1] < R_s:
        r_clip = cast(FloatArray, np.append(r_clip, R_s))
        value_clip = cast(FloatArray, np.append(value_clip, np.interp(R_s, rho, values)))
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
    return cast(FloatArray, -B_ext * R_s * delta * (log_cosh_argument - axis_log_cosh))


def _log_cosh(values: FloatArray) -> FloatArray:
    """Return numerically stable ``log(cosh(values))``."""
    abs_values = cast(FloatArray, np.abs(values))
    return cast(FloatArray, abs_values + np.log1p(np.exp(-2.0 * abs_values)) - np.log(2.0))


def _toroidal_current_density_from_steinhauer(
    rho: FloatArray,
    argument: FloatArray,
    B_ext: float,
    R_s: float,
    delta: float,
) -> FloatArray:
    """Return analytical ``J_theta = -mu_0^-1 dB_z/dr`` for Eq. 7."""
    tanh_argument = cast(FloatArray, np.tanh(argument))
    sech_squared = cast(FloatArray, 1.0 - tanh_argument**2)
    return cast(FloatArray, B_ext * sech_squared * rho / (MU_0 * R_s * delta))


def _ampere_current_closure_residual(rho: FloatArray, B_z: FloatArray, J_theta: FloatArray) -> FloatArray:
    """Return ``mu_0 J_theta + dB_z/dr``; zero means Ampere's law closes."""
    d_bz_dr = cast(FloatArray, np.gradient(B_z, rho, edge_order=2))
    return cast(FloatArray, MU_0 * J_theta + d_bz_dr)


def _flux_derivative_closure_residual(rho: FloatArray, psi: FloatArray, B_z: FloatArray) -> FloatArray:
    """Return ``dpsi/dr - r B_z``; zero means the flux primitive closes."""
    dpsi_dr = cast(FloatArray, np.gradient(psi, rho, edge_order=2))
    return cast(FloatArray, dpsi_dr - rho * B_z)


def _pressure_balance_residual(p: FloatArray, B_z: FloatArray, B_ext: float) -> FloatArray:
    """Return ``p + B_z^2/(2 mu_0) - B_ext^2/(2 mu_0)`` in Pa."""
    return cast(FloatArray, p + B_z**2 / (2.0 * MU_0) - B_ext**2 / (2.0 * MU_0))


def _radial_force_balance_residual(
    rho: FloatArray,
    B_z: FloatArray,
    J_theta: FloatArray,
    p: FloatArray,
) -> FloatArray:
    dp_dr = cast(FloatArray, np.gradient(p, rho, edge_order=2))
    j_cross_b_r = cast(FloatArray, J_theta * B_z)
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
    return bool(np.isfinite(inner_field) and np.isfinite(outer_field) and inner_field * outer_field < 0.0)
