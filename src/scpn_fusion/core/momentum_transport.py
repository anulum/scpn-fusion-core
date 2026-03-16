# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Momentum Transport
from __future__ import annotations


import numpy as np


def nbi_torque(
    P_nbi_profile: np.ndarray, R0: float, v_beam: float, theta_inj_deg: float
) -> np.ndarray:
    """
    Torque deposition from NBI [N m / m^3].
    """
    if v_beam <= 0.0:
        return np.zeros_like(P_nbi_profile)

    theta_rad = np.radians(theta_inj_deg)
    # T = P * R₀ sin(θ) / v_beam [N·m/m³]
    return np.asarray(P_nbi_profile * R0 * np.sin(theta_rad) / v_beam)


def intrinsic_rotation_torque(
    grad_Ti: np.ndarray, grad_ne: np.ndarray, R0: float, a: float
) -> np.ndarray:
    """
    Residual stress model torque (Rice scaling heuristic).
    T_intr ~ grad_Ti
    """
    # Rice scaling: residual stress ∝ ∇T_i
    return -1e-3 * grad_Ti


def exb_shearing_rate(
    omega_phi: np.ndarray, B_theta: np.ndarray, B0: float, R0: float, rho: np.ndarray, a: float
) -> np.ndarray:
    """
    ExB shearing rate [rad/s].
    omega_ExB = (R B_theta / B) * d/dr (E_r / (R B_theta))
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


def turbulence_suppression_factor(omega_ExB: np.ndarray, gamma_max: np.ndarray) -> np.ndarray:
    """
    Reduction factor on anomalous transport.
    F = 1 / (1 + (omega_ExB / gamma_max)^2)
    """
    # Guard against zero gamma
    gamma_safe = np.maximum(gamma_max, 1e-6)
    return np.asarray(1.0 / (1.0 + (omega_ExB / gamma_safe) ** 2))


def radial_electric_field(
    ne: np.ndarray,
    Ti_keV: np.ndarray,
    omega_phi: np.ndarray,
    B_theta: np.ndarray,
    B0: float,
    R0: float,
    rho: np.ndarray,
    a: float,
) -> np.ndarray:
    """
    E_r [V/m] from radial force balance (neoclassical).
    E_r = (1/Z_i e n_i) dp_i/dr + v_φ B_θ (v_θ ≈ 0).
    """
    e_charge = 1.602e-19
    p_i = ne * 1e19 * Ti_keV * 1e3 * e_charge

    drho = rho[1] - rho[0] if len(rho) > 1 else 0.1
    dp_dr = np.gradient(p_i, drho * a)

    # Z_i = 1 for main ions
    term1 = dp_dr / np.maximum(1.0 * e_charge * ne * 1e19, 1e-6)

    # v_phi = R0 * omega_phi
    v_phi = R0 * omega_phi
    term2 = v_phi * B_theta

    return np.asarray(term1 + term2)


class RotationDiagnostics:
    @staticmethod
    def mach_number(omega_phi: np.ndarray, Ti_keV: np.ndarray, R0: float) -> np.ndarray:
        v_phi = omega_phi * R0
        e_charge = 1.602e-19
        m_i = 2.0 * 1.67e-27

        c_s = np.sqrt(Ti_keV * 1e3 * e_charge / m_i)
        return np.asarray(np.abs(v_phi) / np.maximum(c_s, 1e-3))

    @staticmethod
    def rwm_stabilization_criterion(omega_phi: np.ndarray, tau_wall: float) -> bool:
        # Simplified criterion: omega_phi * tau_wall > 1% of something?
        # Typically Omega * tau_wall > O(1) for stabilization
        omega_core = np.abs(omega_phi[0])
        return bool(omega_core * tau_wall > 0.01)


class MomentumTransportSolver:
    def __init__(self, rho: np.ndarray, R0: float, a: float, B0: float, prandtl: float = 0.7):
        self.rho = rho
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.prandtl = prandtl
        self.nr = len(rho)
        self.drho = rho[1] - rho[0]

        self.omega_phi = np.zeros(self.nr)

    def step(
        self,
        dt: float,
        chi_i: np.ndarray,
        ne: np.ndarray,
        Ti_keV: np.ndarray,
        T_nbi: np.ndarray,
        T_intrinsic: np.ndarray,
    ) -> np.ndarray:
        """
        Advance rotation profile.
        rho_m = n_i m_i
        d(rho_m R^2 omega)/dt + 1/r d/dr(r Pi_phi) = T_tot
        Pi_phi = -chi_phi d(rho_m R^2 omega)/dr + V * ...
        Assuming V_pinch = 0 for simplicity.
        """
        import scipy.linalg

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
