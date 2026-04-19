# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Current Diffusion Equation
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

# Sauter, Angioni & Lin-Liu (1999/2002) neoclassical resistivity
MU_0 = 4.0 * np.pi * 1e-7


def neoclassical_resistivity(
    Te_keV: float, ne_19: float, Z_eff: float, epsilon: float, q: float = 1.0, R0: float = 2.0
) -> float:
    """Sauter neoclassical parallel resistivity [Ohm-m]."""
    Te_keV = max(Te_keV, 1e-3)
    ne_19 = max(ne_19, 1e-3)
    epsilon = max(epsilon, 1e-6)

    ln_Lambda = 17.0
    eta_Spitzer = 1.65e-9 * Z_eff * ln_Lambda / (Te_keV**1.5)

    # Trapped fraction — Sauter 2002 Eq. 14
    f_t = 1.0 - (1.0 - epsilon) ** 2 / (np.sqrt(1.0 - epsilon**2) * (1.0 + 1.46 * np.sqrt(epsilon)))
    f_t = max(0.0, min(f_t, 1.0))

    e_charge = 1.602e-19
    m_e = 9.109e-31
    v_te = np.sqrt(2.0 * Te_keV * 1e3 * e_charge / m_e)
    nu_ei = (
        (ne_19 * 1e19)
        * Z_eff
        * e_charge**4
        * ln_Lambda
        / (12.0 * np.pi**1.5 * (8.854e-12) ** 2 * np.sqrt(m_e) * (Te_keV * 1e3 * e_charge) ** 1.5)
    )
    # Sauter 2002 Eq. 13 — electron collisionality (needed for banana/plateau regime selection)
    nu_star_e = nu_ei * max(q, 0.5) * R0 / (epsilon**1.5 * v_te)  # noqa: F841

    # Neoclassical correction — Sauter 2002 Eq. 15
    C_R = 1.0 - (1.0 + 0.36 / Z_eff) * f_t + (0.59 / Z_eff) * f_t**2

    eta_neo = eta_Spitzer / (1.0 - f_t) * C_R
    return float(max(eta_neo, eta_Spitzer))


def q_from_psi(rho: np.ndarray, psi: np.ndarray, R0: float, a: float, B0: float) -> np.ndarray:
    """q(rho) = -rho a^2 B0 / (R0 dpsi/drho), L'Hopital at axis."""
    nr = len(rho)
    q = np.zeros(nr)
    drho = rho[1] - rho[0]

    dpsi_drho = np.gradient(psi, drho)

    for i in range(1, nr):
        denom = R0 * dpsi_drho[i]
        if abs(denom) < 1e-12:
            q[i] = q[i - 1] if i > 1 else 1.0
        else:
            q[i] = -rho[i] * a**2 * B0 / denom

    # L'Hopital at rho=0: q(0) = -a^2 B0 / (R0 d^2psi/drho^2)
    d2psi = (psi[2] - 2 * psi[1] + psi[0]) / (drho**2)
    if abs(d2psi) > 1e-12:
        q[0] = -(a**2) * B0 / (R0 * d2psi)
    else:
        q[0] = q[1]

    q = np.abs(q)
    return q


def resistive_diffusion_time(a: float, eta: float) -> float:
    """tau_R = mu0 a^2 / eta [seconds]."""
    return MU_0 * a**2 / max(eta, 1e-12)


class CurrentDiffusionSolver:
    """Implicit Crank-Nicolson solver for 1D poloidal flux diffusion."""

    def __init__(self, rho: np.ndarray, R0: float, a: float, B0: float):
        self.rho = rho
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.nr = len(rho)
        self.drho = rho[1] - rho[0]
        self.psi = np.zeros(self.nr)
        for i in range(1, self.nr):
            r = self.rho[i]
            q_r = 1.0 + 2.0 * r**2
            dpsi = -r * self.a**2 * self.B0 / (self.R0 * q_r)
            self.psi[i] = self.psi[i - 1] + dpsi * self.drho
        self.psi -= self.psi[-1]

    def step(
        self,
        dt: float,
        Te: np.ndarray,
        ne: np.ndarray,
        Z_eff: float,
        j_bs: np.ndarray,
        j_cd: np.ndarray,
        j_ext: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance poloidal flux psi by one timestep dt."""
        if j_ext is None:
            j_ext = np.zeros(self.nr)

        j_tot_source = j_bs + j_cd + j_ext

        q_prof = q_from_psi(self.rho, self.psi, self.R0, self.a, self.B0)
        eta_prof = np.zeros(self.nr)
        for i in range(self.nr):
            epsilon = self.rho[i] * self.a / self.R0
            eta_prof[i] = neoclassical_resistivity(Te[i], ne[i], Z_eff, epsilon, q_prof[i], self.R0)

        D = eta_prof / (MU_0 * self.a**2)

        alpha = dt / 2.0
        drho2 = self.drho**2

        sub = np.zeros(self.nr)
        diag = np.zeros(self.nr)
        sup = np.zeros(self.nr)
        rhs = np.zeros(self.nr)

        # Axis BC: L(psi)_0 = 4 D_0 (psi_1 - psi_0) / drho^2
        diag[0] = 1.0 + alpha * 4.0 * D[0] / drho2
        sup[0] = -alpha * 4.0 * D[0] / drho2
        rhs[0] = (
            self.psi[0]
            + alpha * 4.0 * D[0] * (self.psi[1] - self.psi[0]) / drho2
            + dt * self.R0 * eta_prof[0] * j_tot_source[0]
        )

        for i in range(1, self.nr - 1):
            r = self.rho[i]
            coeff_prev = D[i] * (1.0 / drho2 - 1.0 / (2.0 * r * self.drho))
            coeff_curr = D[i] * (-2.0 / drho2)
            coeff_next = D[i] * (1.0 / drho2 + 1.0 / (2.0 * r * self.drho))

            sub[i] = -alpha * coeff_prev
            diag[i] = 1.0 - alpha * coeff_curr
            sup[i] = -alpha * coeff_next

            L_psi_n = (
                coeff_prev * self.psi[i - 1]
                + coeff_curr * self.psi[i]
                + coeff_next * self.psi[i + 1]
            )
            rhs[i] = self.psi[i] + alpha * L_psi_n + dt * self.R0 * eta_prof[i] * j_tot_source[i]

        # Edge Dirichlet BC
        diag[-1] = 1.0
        sub[-1] = 0.0
        rhs[-1] = self.psi[-1]

        ab = np.zeros((3, self.nr))
        ab[0, 1:] = sup[:-1]
        ab[1, :] = diag
        ab[2, :-1] = sub[1:]

        self.psi = solve_banded((1, 1), ab, rhs)
        return self.psi
