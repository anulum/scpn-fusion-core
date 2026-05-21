# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Shear Alfven continuum, TAE gap, and fast-particle drive utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ELEMENTARY_CHARGE = 1.602176634e-19
_ATOMIC_MASS_KG = 1.66053906660e-27


@dataclass
class AlfvenGap:
    """Radial location and frequency bounds for a toroidicity-induced gap."""

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
        kappa: float = 1.7,
        gap_width_scale: float = 1.0,
    ):
        if not np.isfinite(B0) or B0 <= 0.0:
            raise ValueError("B0 must be finite and > 0.")
        if not np.isfinite(R0) or R0 <= 0.0:
            raise ValueError("R0 must be finite and > 0.")
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("a must be finite and > 0.")
        if not np.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("kappa must be finite and > 0.")
        if not np.isfinite(gap_width_scale) or gap_width_scale <= 0.0:
            raise ValueError("gap_width_scale must be finite and > 0.")
        self.rho = rho
        self.q = q
        self.ne = ne
        self.B0 = B0
        self.R0 = R0
        self.a = a
        self.kappa = float(kappa)
        self.gap_width_scale = float(gap_width_scale)

        m_i = m_i_amu * 1.67e-27
        m_e = 9.11e-31

        # Mass density
        self.rho_mass = ne * 1e19 * (m_i + m_e)

        # v_A(rho)
        mu_0 = 4.0 * np.pi * 1e-7
        self.v_A = B0 / np.sqrt(mu_0 * np.maximum(self.rho_mass, 1e-12))

    def alfven_speed(self, rho_eval: float) -> float:
        """Interpolate local Alfven speed at the requested normalised radius."""
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

                # Shaping-aware toroidicity gap width model.
                eps = rho_gap * self.a / self.R0
                delta_gap = self.gap_width_scale * eps * np.sqrt(self.kappa)
                delta_gap = float(np.clip(delta_gap, 1.0e-6, 0.9))
                omega_lower = omega_0 * (1.0 - delta_gap)
                omega_upper = omega_0 * (1.0 + delta_gap)

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
    """Single toroidal Alfven eigenmode frequency estimate."""

    def __init__(self, n: int, q_rational: float, v_A: float, R0: float):
        self.n = n
        self.q = q_rational
        self.v_A = v_A
        self.R0 = R0

    def frequency(self) -> float:
        """Return angular TAE frequency in radians per second."""
        return self.v_A / (2.0 * self.q * self.R0)

    def frequency_kHz(self) -> float:
        """Return cyclic TAE frequency in kilohertz."""
        return self.frequency() / (2.0 * np.pi * 1e3)


class FastParticleDrive:
    """Fast-particle pressure and resonance drive model for TAE stability."""

    def __init__(
        self,
        E_fast_keV: float,
        n_fast_frac: float,
        m_fast_amu: float = 4.0,
        *,
        main_resonance_width: float = 0.2,
        sideband_resonance_width: float = 0.1,
        sideband_weight: float = 0.5,
    ):
        self.E_fast_keV = E_fast_keV
        self.n_fast_frac = n_fast_frac
        self.m_fast = m_fast_amu * 1.67e-27
        self.main_resonance_width = float(main_resonance_width)
        self.sideband_resonance_width = float(sideband_resonance_width)
        self.sideband_weight = float(sideband_weight)
        if not np.isfinite(self.main_resonance_width) or self.main_resonance_width <= 0.0:
            raise ValueError("main_resonance_width must be finite and > 0.")
        if not np.isfinite(self.sideband_resonance_width) or self.sideband_resonance_width <= 0.0:
            raise ValueError("sideband_resonance_width must be finite and > 0.")
        if not np.isfinite(self.sideband_weight) or self.sideband_weight < 0.0:
            raise ValueError("sideband_weight must be finite and >= 0.")

        E_J = E_fast_keV * 1e3 * 1.602e-19
        self.v_fast = np.sqrt(2.0 * E_J / self.m_fast)

    def beta_fast(self, ne_20: float, B0: float) -> float:
        """Return fast-particle beta from electron density and magnetic field."""
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
        if not np.isfinite(v_fast) or v_fast <= 0.0:
            raise ValueError("v_fast must be finite and > 0.")
        if not np.isfinite(v_A) or v_A <= 0.0:
            raise ValueError("v_A must be finite and > 0.")
        x = v_fast / v_A
        f_main = np.exp(-(((x - 1.0) / self.main_resonance_width) ** 2))
        f_side = self.sideband_weight * np.exp(
            -(((x - 1.0 / 3.0) / self.sideband_resonance_width) ** 2)
        )
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
    """Net TAE drive and damping result for one mode candidate."""

    n: int
    m: int
    frequency_kHz: float
    gamma_drive: float
    gamma_damp: float
    gamma_net: float
    unstable: bool


class AlfvenStabilityAnalysis:
    """Evaluate fast-particle drive against continuum TAE gaps."""

    def __init__(self, continuum: AlfvenContinuum, fast_params: FastParticleDrive):
        self.continuum = continuum
        self.fast_params = fast_params

    def tae_stability(self, n_range: range = range(1, 6)) -> list[TAEStabilityResult]:
        """Return TAE stability estimates for all gaps in the requested n range."""
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
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        gaps = self.continuum.find_gaps(n)
        if not gaps:
            return float("inf")
        beta_crit_best = float("inf")
        for gap in gaps:
            q_gap = (2.0 * gap.m_coupling + 1.0) / (2.0 * n)
            v_A = self.continuum.alfven_speed(gap.rho_location)
            resonance = self.fast_params.resonance_function(self.fast_params.v_fast, v_A)
            coeff = q_gap**2 * resonance
            if coeff <= 0.0:
                continue
            # From gamma_drive = omega*beta_f*q^2*F and gamma_damp = 0.01*omega:
            # beta_crit = 0.01 / (q^2 F).
            beta_crit = 0.01 / coeff
            if beta_crit < beta_crit_best:
                beta_crit_best = beta_crit
        return float(beta_crit_best)

    def alpha_particle_loss_estimate(
        self,
        gamma_net: float,
        tau_sd: float = 0.5,
        *,
        nonlinear_saturation: float = 1.0e-4,
        transport_threshold: float = 1.0,
    ) -> float:
        """Fraction of alpha power lost."""
        gamma = float(gamma_net)
        tau = float(tau_sd)
        sat = float(nonlinear_saturation)
        threshold = float(transport_threshold)
        if not np.isfinite(gamma):
            raise ValueError("gamma_net must be finite.")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau_sd must be finite and > 0.")
        if not np.isfinite(sat) or sat <= 0.0:
            raise ValueError("nonlinear_saturation must be finite and > 0.")
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("transport_threshold must be finite and > 0.")
        if gamma <= 0.0:
            return 0.0

        drive = gamma * tau / threshold
        # Saturating transport-loss model: 1-exp(-sat*drive)
        loss = 1.0 - np.exp(-sat * drive)
        return float(np.clip(loss, 0.0, 1.0))


def _require_positive(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return scalar


def bae_accumulation_frequency(
    *,
    Ti_keV: float,
    Te_keV: float,
    m_i_amu: float,
    R0: float,
) -> float:
    """Return the kinetic BAE accumulation-point angular frequency.

    Uses the large-aspect-ratio low-frequency estimate
    omega_BAE = sqrt(7/4 + Te/Ti) * v_ti / R0, with
    v_ti = sqrt(2 Ti / m_i). Temperatures are in keV and the result is rad/s.
    """
    Ti = _require_positive("Ti_keV", Ti_keV)
    Te = _require_positive("Te_keV", Te_keV)
    ion_mass = _require_positive("m_i_amu", m_i_amu) * _ATOMIC_MASS_KG
    major_radius = _require_positive("R0", R0)
    thermal_speed = np.sqrt(2.0 * Ti * 1.0e3 * _ELEMENTARY_CHARGE / ion_mass)
    compressibility = np.sqrt(1.75 + Te / Ti)
    return float(compressibility * thermal_speed / major_radius)


def rsae_frequency(
    q_min: float,
    n: int,
    m: int,
    v_A: float,
    R0: float,
    *,
    Ti_keV: float = 10.0,
    Te_keV: float = 10.0,
    m_i_amu: float = 2.5,
) -> float:
    """
    RSAE frequency.
    omega_RSAE = omega_BAE + |n q_min - m| v_A / (q_min R0)
    """
    q_minimum = _require_positive("q_min", q_min)
    alfven_speed = _require_positive("v_A", v_A)
    major_radius = _require_positive("R0", R0)
    if n == 0:
        raise ValueError("toroidal mode number n must be nonzero.")
    omega_BAE = bae_accumulation_frequency(
        Ti_keV=Ti_keV,
        Te_keV=Te_keV,
        m_i_amu=m_i_amu,
        R0=major_radius,
    )
    omega_RSAE = omega_BAE + abs(n * q_minimum - m) * alfven_speed / (q_minimum * major_radius)
    return float(omega_RSAE)
