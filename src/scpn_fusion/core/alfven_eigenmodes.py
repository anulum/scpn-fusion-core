# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Alfven Eigenmodes
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlfvenGap:
    rho_location: float
    omega_lower: float
    omega_upper: float
    m_coupling: int


class AlfvenContinuum:
    """
    Shear Alfven wave continuum and gap structure.
    """

    def __init__(
        self,
        rho: np.ndarray,
        q: np.ndarray,
        ne: np.ndarray,
        B0: float,
        R0: float,
        m_i_amu: float = 2.5,
        a: float = 2.0,
    ):
        self.rho = rho
        self.q = q
        self.ne = ne
        self.B0 = B0
        self.R0 = R0
        self.a = a

        m_i = m_i_amu * 1.67e-27
        m_e = 9.11e-31

        # Mass density
        self.rho_mass = ne * 1e19 * (m_i + m_e)

        # v_A(rho)
        mu_0 = 4.0 * np.pi * 1e-7
        self.v_A = B0 / np.sqrt(mu_0 * np.maximum(self.rho_mass, 1e-12))

    def alfven_speed(self, rho_eval: float) -> float:
        return float(np.interp(rho_eval, self.rho, self.v_A))

    def continuum(self, m: int, n: int) -> np.ndarray:
        """
        omega_A(rho) = |n*q - m| / (q*R0) * v_A
        """
        k_par = np.abs(n * self.q - m) / np.maximum(self.q * self.R0, 1e-6)
        return np.asarray(k_par * self.v_A)

    def find_gaps(self, n: int) -> list[AlfvenGap]:
        """
        Locate TAE gaps where q = (2m+1)/(2n).
        """
        gaps = []
        for m in range(1, 10):
            q_gap = (2.0 * m + 1.0) / (2.0 * n)

            # Find where q crosses q_gap
            q_diff = self.q - q_gap
            crossings = np.where(np.diff(np.sign(q_diff)))[0]

            for idx in crossings:
                r1, r2 = self.rho[idx], self.rho[idx + 1]
                q1, q2 = self.q[idx], self.q[idx + 1]
                if q1 == q2:
                    continue

                frac = (q_gap - q1) / (q2 - q1)
                rho_gap = r1 + frac * (r2 - r1)

                v_A_gap = self.alfven_speed(rho_gap)
                omega_0 = v_A_gap / (2.0 * q_gap * self.R0)

                # Gap width proxy (depends on inverse aspect ratio epsilon)
                eps = rho_gap * self.a / self.R0
                omega_lower = omega_0 * (1.0 - eps)
                omega_upper = omega_0 * (1.0 + eps)

                gaps.append(
                    AlfvenGap(
                        rho_location=float(rho_gap),
                        omega_lower=float(omega_lower),
                        omega_upper=float(omega_upper),
                        m_coupling=m,
                    )
                )

        return gaps


class TAEMode:
    def __init__(self, n: int, q_rational: float, v_A: float, R0: float):
        self.n = n
        self.q = q_rational
        self.v_A = v_A
        self.R0 = R0

    def frequency(self) -> float:
        return self.v_A / (2.0 * self.q * self.R0)

    def frequency_kHz(self) -> float:
        return self.frequency() / (2.0 * np.pi * 1e3)


class FastParticleDrive:
    def __init__(self, E_fast_keV: float, n_fast_frac: float, m_fast_amu: float = 4.0):
        self.E_fast_keV = E_fast_keV
        self.n_fast_frac = n_fast_frac
        self.m_fast = m_fast_amu * 1.67e-27

        E_J = E_fast_keV * 1e3 * 1.602e-19
        self.v_fast = np.sqrt(2.0 * E_J / self.m_fast)

    def beta_fast(self, ne_20: float, B0: float) -> float:
        n_e = ne_20 * 1e20
        n_fast = n_e * self.n_fast_frac
        E_J = self.E_fast_keV * 1e3 * 1.602e-19
        p_fast = (2.0 / 3.0) * n_fast * E_J

        mu_0 = 4.0 * np.pi * 1e-7
        p_mag = B0**2 / (2.0 * mu_0)

        return float(p_fast / p_mag)

    def resonance_function(self, v_fast: float, v_A: float) -> float:
        """
        F(v_f / v_A) peaking near 1 or 1/3 (for sidebands).
        """
        x = v_fast / v_A
        # Simple heuristic resonance curve peaking at x=1 and x=1/3
        f_main = np.exp(-(((x - 1.0) / 0.2) ** 2))
        f_side = 0.5 * np.exp(-(((x - 0.33) / 0.1) ** 2))
        return float(f_main + f_side)

    def growth_rate(self, tae: TAEMode, beta_fast: float) -> float:
        """
        gamma_fast / omega = beta_f * q^2 * F
        """
        omega = tae.frequency()
        F = self.resonance_function(self.v_fast, tae.v_A)
        return float(omega * beta_fast * tae.q**2 * F)


@dataclass
class TAEStabilityResult:
    n: int
    m: int
    frequency_kHz: float
    gamma_drive: float
    gamma_damp: float
    gamma_net: float
    unstable: bool


class AlfvenStabilityAnalysis:
    def __init__(self, continuum: AlfvenContinuum, fast_params: FastParticleDrive):
        self.continuum = continuum
        self.fast_params = fast_params

    def tae_stability(self, n_range: range = range(1, 6)) -> list[TAEStabilityResult]:
        results = []
        for n in n_range:
            gaps = self.continuum.find_gaps(n)
            for gap in gaps:
                v_A = self.continuum.alfven_speed(gap.rho_location)
                q_gap = (2.0 * gap.m_coupling + 1.0) / (2.0 * n)
                tae = TAEMode(n, q_gap, v_A, self.continuum.R0)

                omega = tae.frequency()
                freq_kHz = tae.frequency_kHz()

                # Approximate ne_20
                idx = int(np.searchsorted(self.continuum.rho, gap.rho_location))
                idx = min(idx, len(self.continuum.ne) - 1)
                ne_20 = self.continuum.ne[idx] / 10.0  # if ne is in 10^19

                b_fast = self.fast_params.beta_fast(max(ne_20, 0.1), self.continuum.B0)

                gamma_drive = self.fast_params.growth_rate(tae, b_fast)

                # Simple damping model ~ 1% of omega
                gamma_damp = 0.01 * omega

                gamma_net = gamma_drive - gamma_damp
                unstable = gamma_net > 0.0

                results.append(
                    TAEStabilityResult(
                        n=n,
                        m=gap.m_coupling,
                        frequency_kHz=freq_kHz,
                        gamma_drive=gamma_drive,
                        gamma_damp=gamma_damp,
                        gamma_net=gamma_net,
                        unstable=unstable,
                    )
                )

        return results

    def critical_beta_fast(self, n: int) -> float:
        """Find beta_fast where gamma_net = 0 for the most unstable mode."""
        res = self.tae_stability(range(n, n + 1))
        if not res:
            return float("inf")

        # Find mode with best resonance
        best_mode = max(res, key=lambda r: r.gamma_drive)
        if best_mode.gamma_drive == 0:
            return float("inf")

        # beta_crit = b_fast * (gamma_damp / gamma_drive)
        # We need the current beta_fast used to compute it.
        # But gamma_drive is linear in beta_fast in our model: gamma = C * beta
        # So beta_crit = gamma_damp / C = gamma_damp / (gamma_drive / beta_current)

        v_A = self.continuum.alfven_speed(0.5)  # rough
        b_current = self.fast_params.beta_fast(1.0, self.continuum.B0)  # nominal

        if b_current == 0:
            return float("inf")

        return float(b_current * (best_mode.gamma_damp / best_mode.gamma_drive))

    def alpha_particle_loss_estimate(self, gamma_net: float, tau_sd: float = 0.5) -> float:
        """Fraction of alpha power lost."""
        if gamma_net <= 0:
            return 0.0

        # Simple heuristic: loss scales with gamma * tau_slowing_down
        loss = min(1.0, 1e-4 * gamma_net * tau_sd)
        return float(loss)


def rsae_frequency(q_min: float, n: int, m: int, v_A: float, R0: float) -> float:
    """
    RSAE frequency.
    omega_RSAE = omega_BAE + |n q_min - m| v_A / (q_min R0)
    """
    omega_BAE = 0.1 * v_A / R0  # Highly simplified
    omega_RSAE = omega_BAE + abs(n * q_min - m) * v_A / (q_min * R0)
    return float(omega_RSAE)
