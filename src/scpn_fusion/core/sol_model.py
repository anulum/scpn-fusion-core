# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SOL Two-Point Model (Eich heat flux scaling)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SOLSolution:
    """Two-point SOL solution: upstream/target conditions and heat-flux width."""

    T_upstream_eV: float
    T_target_eV: float
    n_target_19: float
    q_parallel_MW_m2: float
    lambda_q_mm: float


def eich_heat_flux_width(P_SOL_MW: float, R0: float, B_pol: float, epsilon: float) -> float:
    """Eich scaling: lambda_q [mm] = 1.35 P^-0.02 R0^0.04 Bpol^-0.92 eps^0.42."""
    if P_SOL_MW <= 0.0 or B_pol <= 0.0 or R0 <= 0.0 or epsilon <= 0.0:
        return 1.0
    return float(1.35 * (P_SOL_MW**-0.02) * (R0**0.04) * (B_pol**-0.92) * (epsilon**0.42))


def peak_target_heat_flux(
    P_SOL_MW: float, R0: float, lambda_q_m: float, f_expansion: float = 5.0, alpha_deg: float = 3.0
) -> float:
    """Peak target heat flux [MW/m^2]."""
    if lambda_q_m <= 0.0:
        return 0.0
    alpha_rad = np.radians(alpha_deg)
    q_peak = P_SOL_MW / (4.0 * np.pi * R0 * lambda_q_m * f_expansion) * np.sin(alpha_rad)
    return float(q_peak)


class TwoPointSOL:
    """Spitzer-Harm two-point model with Eich heat-flux width scaling."""

    def __init__(self, R0: float, a: float, q95: float, B_pol: float, kappa: float = 1.0):
        self.R0 = R0
        self.a = a
        self.q95 = q95
        self.B_pol = B_pol
        self.kappa = kappa
        self.epsilon = a / R0
        self.L_par = np.pi * q95 * R0

    def solve(self, P_SOL_MW: float, n_u_19: float, f_rad: float = 0.0) -> SOLSolution:
        """Solve for target temperature, density, and parallel heat flux."""
        lambda_q_mm = eich_heat_flux_width(P_SOL_MW, self.R0, self.B_pol, self.epsilon)
        lambda_q_m = lambda_q_mm * 1e-3

        B_ratio = self.q95 / self.epsilon
        q_par_u_W_m2 = (P_SOL_MW * 1e6) / (4.0 * np.pi * self.R0 * lambda_q_m) * B_ratio

        # Spitzer-Harm conduction: T_u = (7/2 L q / kappa0)^(2/7)
        kappa_0 = 2000.0
        T_u = ((3.5 * self.L_par * q_par_u_W_m2) / kappa_0) ** (2.0 / 7.0)

        q_par_t_W_m2 = max(q_par_u_W_m2 * (1.0 - f_rad), 1e3)

        gamma_sh = 7.0
        e_charge = 1.602e-19
        m_i = 2.0 * 1.6726e-27  # Deuterium

        n_u = n_u_19 * 1e19

        denom = n_u * T_u * gamma_sh * e_charge * np.sqrt(2.0 * e_charge / m_i)

        if denom <= 0.0:
            T_t = 0.1
        else:
            sqrt_Tt = 2.0 * q_par_t_W_m2 / denom
            T_t = sqrt_Tt**2

        T_t = min(T_t, T_u)
        n_t = n_u * T_u / (2.0 * max(T_t, 0.1))

        return SOLSolution(
            T_upstream_eV=float(T_u),
            T_target_eV=float(T_t),
            n_target_19=float(n_t / 1e19),
            q_parallel_MW_m2=float(q_par_u_W_m2 / 1e6),
            lambda_q_mm=float(lambda_q_mm),
        )
