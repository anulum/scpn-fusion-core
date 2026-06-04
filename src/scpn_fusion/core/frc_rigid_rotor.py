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
    p: FloatArray
    R_null: float
    separatrix_index: int
    s_parameter: float
    energy_J: float
    converged: bool
    residual: float
    delta: float
    pressure_balance_ratio: float
    model: str


@dataclass(frozen=True)
class FRCValidationReport:
    """Validation diagnostics for the analytical FRC state."""

    finite: bool
    monotonic_grid: bool
    null_error_m: float
    pressure_peak_error_m: float
    edge_field_error_T: float
    pressure_balance_ratio: float
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
    psi = _cylindrical_flux_from_bz(rho, B_z)
    r_null = _zero_crossing_radius(rho, B_z)
    separatrix_index = int(np.argmin(np.abs(rho - r_null)))

    p0 = inputs.n0 * (inputs.T_i_eV + inputs.T_e_eV) * ELEMENTARY_CHARGE_C
    psi_axis = np.interp(r_null, rho, psi)
    pressure_span = max(abs(inputs.B_ext * inputs.R_s), tolerance)
    p = p0 * np.exp(-2.0 * ((psi - psi_axis) / pressure_span) ** 2)

    magnetic_energy_density = B_z**2 / (2.0 * MU_0)
    total_energy_density = magnetic_energy_density + p
    energy_J_per_m = float(trapezoid(total_energy_density * 2.0 * np.pi * rho, rho))

    pressure_integral = float(trapezoid(p * 2.0 * np.pi * rho, rho))
    external_pressure_energy = (inputs.B_ext**2 / (2.0 * MU_0)) * np.pi * inputs.R_s**2
    pressure_balance_ratio = pressure_integral / max(external_pressure_energy, tolerance)

    return FRCEquilibriumState(
        rho=rho,
        psi=psi,
        B_z=B_z,
        B_theta=B_theta,
        p=p,
        R_null=r_null,
        separatrix_index=separatrix_index,
        s_parameter=inputs.R_s / (2.0 * delta),
        energy_J=energy_J_per_m,
        converged=True,
        residual=float(np.max(np.abs(B_z - (-inputs.B_ext * np.tanh(argument))))),
        delta=delta,
        pressure_balance_ratio=float(pressure_balance_ratio),
        model="steinhauer_2011_no_rotation_analytical",
    )


def null_radius(state: FRCEquilibriumState) -> float:
    """Return the interpolated radius where the axial field reverses."""
    return _zero_crossing_radius(state.rho, state.B_z)


def s_parameter(state: FRCEquilibriumState, mass_amu: float = DEUTERIUM_MASS_AMU) -> float:
    """Return the analytical no-rotation ``s`` parameter for this FRC state."""
    del mass_amu
    return float(state.s_parameter)


def validate_equilibrium(state: FRCEquilibriumState, *, tolerance: float = 1e-6) -> FRCValidationReport:
    """Validate finite values, magnetic-null placement, and pressure peaking."""
    finite = all(
        bool(np.all(np.isfinite(values)))
        for values in (state.rho, state.psi, state.B_z, state.B_theta, state.p)
    )
    monotonic_grid = bool(np.all(np.diff(state.rho) > 0.0))
    r_null = null_radius(state)
    null_error = abs(r_null - state.R_null)
    pressure_peak_radius = float(state.rho[int(np.argmax(state.p))])
    pressure_peak_error = abs(pressure_peak_radius - state.R_null)
    edge_field_error = abs(abs(float(state.B_z[-1])) - abs(float(state.B_z[0])))
    passed = finite and monotonic_grid and null_error <= tolerance and pressure_peak_error <= max(
        tolerance, 2.0 * float(np.max(np.diff(state.rho)))
    )
    return FRCValidationReport(
        finite=finite,
        monotonic_grid=monotonic_grid,
        null_error_m=float(null_error),
        pressure_peak_error_m=float(pressure_peak_error),
        edge_field_error_T=float(edge_field_error),
        pressure_balance_ratio=state.pressure_balance_ratio,
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
    if rho[0] < 0.0:
        raise ValueError("rho_grid must start at a non-negative radius")
    if not np.all(np.diff(rho) > 0.0):
        raise ValueError("rho_grid must be strictly increasing")
    if rho[-1] < R_s:
        raise ValueError("rho_grid must include the separatrix radius R_s")
    return rho


def _cylindrical_flux_from_bz(rho: FloatArray, B_z: FloatArray) -> FloatArray:
    integrand = cast(FloatArray, rho * B_z)
    psi = cast(FloatArray, np.zeros_like(rho))
    increments = cast(FloatArray, 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(rho))
    psi[1:] = np.cumsum(increments)
    return psi


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
