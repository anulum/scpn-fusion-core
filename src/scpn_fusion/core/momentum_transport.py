# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Momentum Transport
"""Momentum transport, rotation diagnostics, and ExB shear helper models."""

from __future__ import annotations


import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def nbi_torque(
    P_nbi_profile: FloatArray, R0: float, v_beam: float, theta_inj_deg: float
) -> FloatArray:
    """Return the NBI torque deposition profile [N m / m^3]."""
    if v_beam <= 0.0:
        return np.zeros_like(P_nbi_profile)

    theta_rad = np.radians(theta_inj_deg)
    # T = P * R₀ sin(θ) / v_beam [N·m/m³]
    return np.asarray(P_nbi_profile * R0 * np.sin(theta_rad) / v_beam)


def intrinsic_rotation_torque(
    grad_Ti: FloatArray, grad_ne: FloatArray, R0: float, a: float
) -> FloatArray:
    """Return the residual-stress (Rice scaling) intrinsic rotation torque.

    The residual stress scales with the ion temperature gradient, T_intr ~ grad_Ti.
    """
    # Rice scaling: residual stress ∝ ∇T_i
    return -1e-3 * grad_Ti


def exb_shearing_rate(
    omega_phi: FloatArray, B_theta: FloatArray, B0: float, R0: float, rho: FloatArray, a: float
) -> FloatArray:
    """Return the ExB shearing rate [rad/s].

    omega_ExB = (R B_theta / B) d/dr (E_r / (R B_theta)).
    """
    # Assuming core is rotation dominated: E_r ≈ v_phi * B_theta = (R0 * omega_phi) * B_theta
    # So E_r / (R0 B_theta) ≈ omega_phi
    # Thus omega_ExB ≈ (R0 B_theta / B0) * d(omega_phi)/dr

    drho = rho[1] - rho[0] if len(rho) > 1 else 0.1
    dr = drho * a

    domega_dr = np.gradient(omega_phi, dr)

    # B = sqrt(B0^2 + B_theta^2) ≈ B0 for standard tokamak
    B_tot = np.sqrt(B0**2 + B_theta**2)

    rate = (R0 * B_theta / np.maximum(B_tot, 1e-6)) * domega_dr
    return np.asarray(np.abs(rate))


def turbulence_suppression_factor(omega_ExB: FloatArray, gamma_max: FloatArray) -> FloatArray:
    """Return the reduction factor on anomalous transport.

    F = 1 / (1 + (omega_ExB / gamma_max)^2).
    """
    # Guard against zero gamma
    gamma_safe = np.maximum(gamma_max, 1e-6)
    return np.asarray(1.0 / (1.0 + (omega_ExB / gamma_safe) ** 2))


def radial_electric_field(
    ne: FloatArray,
    Ti_keV: FloatArray,
    omega_phi: FloatArray,
    B_theta: FloatArray,
    B0: float,
    R0: float,
    rho: FloatArray,
    a: float,
    v_theta: FloatArray | None = None,
) -> FloatArray:
    """Return E_r [V/m] from the neoclassical radial force balance.

    E_r = (1/Z_i e n_i) dp_i/dr + v_phi B_theta - v_theta B_phi.
    """
    if v_theta is None:
        v_theta_arr = np.zeros_like(omega_phi, dtype=float)
    else:
        v_theta_arr = np.asarray(v_theta, dtype=float)
        if v_theta_arr.shape != np.asarray(omega_phi).shape or not np.all(np.isfinite(v_theta_arr)):
            raise ValueError("v_theta must be a finite profile matching omega_phi")

    e_charge = 1.602e-19
    p_i = ne * 1e19 * Ti_keV * 1e3 * e_charge

    drho = rho[1] - rho[0] if len(rho) > 1 else 0.1
    dp_dr = np.gradient(p_i, drho * a)

    # Z_i = 1 for main ions
    term1 = dp_dr / np.maximum(1.0 * e_charge * ne * 1e19, 1e-6)

    # v_phi = R0 * omega_phi
    v_phi = R0 * omega_phi
    term2 = v_phi * B_theta - v_theta_arr * B0

    return np.asarray(term1 + term2)


class RotationDiagnostics:
    """Diagnostics for toroidal rotation and RWM stabilisation margin."""

    @staticmethod
    def mach_number(omega_phi: FloatArray, Ti_keV: FloatArray, R0: float) -> FloatArray:
        """Return toroidal Mach number from angular rotation and ion temperature."""
        v_phi = omega_phi * R0
        e_charge = 1.602e-19
        m_i = 2.0 * 1.67e-27

        c_s = np.sqrt(Ti_keV * 1e3 * e_charge / m_i)
        return np.asarray(np.abs(v_phi) / np.maximum(c_s, 1e-3))

    @staticmethod
    def rwm_stabilization_criterion(omega_phi: FloatArray, tau_wall: float) -> bool:
        """Return whether core rotation satisfies the wall-time stabilisation rule."""
        omega = np.asarray(omega_phi, dtype=float)
        if omega.ndim != 1 or omega.size == 0 or not np.all(np.isfinite(omega)):
            raise ValueError("omega_phi must be a finite one-dimensional profile")
        if not np.isfinite(tau_wall) or tau_wall <= 0.0:
            raise ValueError("tau_wall must be finite and positive")

        core_count = max(1, min(omega.size // 5, omega.size))
        omega_core = float(np.mean(np.abs(omega[:core_count])))
        return bool(omega_core * tau_wall >= 1.0)


class MomentumTransportSolver:
    """Implicit radial angular-momentum transport solver."""

    def __init__(self, rho: FloatArray, R0: float, a: float, B0: float, prandtl: float = 0.7):
        """Initialise geometry, magnetic field, transport grid, and rotation state."""
        self.rho = np.asarray(rho, dtype=float)
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.prandtl = prandtl
        self.nr = len(self.rho)
        if self.nr < 3:
            raise ValueError("rho must contain at least three radial points")
        if not np.all(np.isfinite(self.rho)) or not np.all(np.diff(self.rho) > 0.0):
            raise ValueError("rho must be finite and strictly increasing")
        if not np.isclose(self.rho[0], 0.0) or not np.isclose(self.rho[-1], 1.0):
            raise ValueError("rho must span the normalised interval [0, 1]")
        drho = np.diff(self.rho)
        if not np.allclose(drho, drho[0], rtol=1e-6, atol=1e-12):
            raise ValueError("rho grid must be uniformly spaced")
        for name, value in {"R0": R0, "a": a, "B0": B0, "prandtl": prandtl}.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.drho = float(drho[0])

        self.omega_phi = np.zeros(self.nr)

    def step(
        self,
        dt: float,
        chi_i: FloatArray,
        ne: FloatArray,
        Ti_keV: FloatArray,
        T_nbi: FloatArray,
        T_intrinsic: FloatArray,
    ) -> FloatArray:
        """Advance the rotation profile by one implicit time step.

        Solves d(rho_m R^2 omega)/dt + (1/r) d/dr(r Pi_phi) = T_tot with momentum
        density rho_m = n_i m_i and flux Pi_phi = -chi_phi d(rho_m R^2 omega)/dr,
        assuming zero pinch velocity.
        """
        import scipy.linalg

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        chi_i = self._require_profile("chi_i", chi_i, non_negative=True)
        ne = self._require_profile("ne", ne, positive=True)
        Ti_keV = self._require_profile("Ti_keV", Ti_keV, positive=True)
        T_nbi = self._require_profile("T_nbi", T_nbi)
        T_intrinsic = self._require_profile("T_intrinsic", T_intrinsic)

        chi_phi = self.prandtl * chi_i

        m_i = 2.0 * 1.67e-27
        n_m3 = ne * 1e19
        rho_m = n_m3 * m_i

        # Momentum density L = rho_m R0^2 omega
        L = rho_m * self.R0**2 * self.omega_phi

        T_tot = T_nbi + T_intrinsic

        dr = self.drho * self.a

        diag = np.zeros(self.nr)
        upper = np.zeros(self.nr)
        lower = np.zeros(self.nr)
        rhs = np.zeros(self.nr)

        diag[0] = 1.0
        upper[0] = -1.0
        rhs[0] = 0.0

        diag[-1] = 1.0
        rhs[-1] = 0.0  # No slip at edge

        for i in range(1, self.nr - 1):
            r_val = self.rho[i] * self.a

            c_plus = chi_phi[i] / dr**2 + chi_phi[i] / (2.0 * r_val * dr)
            c_minus = chi_phi[i] / dr**2 - chi_phi[i] / (2.0 * r_val * dr)
            c_0 = -2.0 * chi_phi[i] / dr**2

            lower[i] = -dt * c_minus
            diag[i] = 1.0 - dt * c_0
            upper[i] = -dt * c_plus

            rhs[i] = L[i] + dt * T_tot[i]

        ab = np.zeros((3, self.nr))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]

        L_new = scipy.linalg.solve_banded((1, 1), ab, rhs)

        # L = rho_m R0^2 omega
        self.omega_phi = L_new / np.maximum(rho_m * self.R0**2, 1e-12)

        return self.omega_phi

    def _require_profile(
        self,
        name: str,
        value: FloatArray,
        *,
        non_negative: bool = False,
        positive: bool = False,
    ) -> FloatArray:
        arr = np.asarray(value, dtype=float)
        if arr.shape != (self.nr,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite profile with shape ({self.nr},)")
        if positive and np.any(arr <= 0.0):
            raise ValueError(f"{name} must be positive")
        if non_negative and np.any(arr < 0.0):
            raise ValueError(f"{name} must be non-negative")
        return arr
