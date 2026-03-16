# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Relativistic Runaway Electron Dynamics
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


E_CHARGE = 1.602176634e-19  # C
M_E = 9.1093837e-31  # kg
C_LIGHT = 2.99792458e8  # m/s
EPS_0 = 8.8541878128e-12  # F/m


@dataclass
class RunawayParams:
    ne_20: float  # electron density [10^20 m^-3]
    Te_keV: float  # electron temperature [keV]
    E_par: float  # parallel electric field [V/m]
    Z_eff: float  # effective charge
    B0: float  # magnetic field [T]
    R0: float  # major radius [m]
    a: float = 2.0  # minor radius [m]


def dreicer_field(ne_20: float, Te_keV: float, coulomb_log: float = 15.0) -> float:
    """
    Dreicer field E_D = n_e e^3 lnΛ / (4π ε₀² T_e).
    Connor & Hastie, Nucl. Fusion 15, 415 (1975).
    """
    n_e = ne_20 * 1e20
    Te_J = Te_keV * 1e3 * E_CHARGE
    return float(n_e * E_CHARGE**3 * coulomb_log / (4.0 * np.pi * EPS_0**2 * Te_J))


def critical_field(ne_20: float, coulomb_log: float = 15.0) -> float:
    """
    Critical field E_c = n_e e^3 lnΛ / (4π ε₀² m_e c²).
    Rosenbluth & Putvinski, Nucl. Fusion 37, 1355 (1997).
    E_D / E_c = m_e c² / T_e ≈ 51 at T_e = 10 keV.
    """
    n_e = ne_20 * 1e20
    return float(n_e * E_CHARGE**3 * coulomb_log / (4.0 * np.pi * EPS_0**2 * M_E * C_LIGHT**2))


def dreicer_generation_rate(params: RunawayParams, coulomb_log: float = 15.0) -> float:
    """
    Primary (seed) runaway electrons generation rate [m^-3 s^-1].
    Connor & Hastie (1975).
    """
    if params.E_par <= 0.0 or params.Te_keV <= 0.0 or params.ne_20 <= 0.0:
        return 0.0

    n_e = params.ne_20 * 1e20
    Te_J = params.Te_keV * 1e3 * E_CHARGE

    v_te = np.sqrt(2.0 * Te_J / M_E)

    # E_D based on thermal velocity: E_D = n_e e^3 lnL / (4 pi eps0^2 Te)
    E_D = n_e * E_CHARGE**3 * coulomb_log / (4.0 * np.pi * EPS_0**2 * Te_J)

    if params.E_par / E_D < 1e-4:
        return 0.0

    nu_ee = n_e * E_CHARGE**4 * coulomb_log / (4.0 * np.pi * EPS_0**2 * M_E**2 * v_te**3)

    Z = params.Z_eff
    E_ratio = params.E_par / E_D

    h_z = (Z + 1.0) / 16.0 * (Z + 1.0 + 2.0 * np.sqrt(1.0 + 1.0 / Z) + E_ratio)

    C_D = 0.35

    exponent = -E_D / (4.0 * params.E_par) - np.sqrt((1.0 + Z) * E_D / params.E_par)

    # Avoid overflow/underflow
    if exponent < -500:
        return 0.0

    rate = C_D * n_e * nu_ee * (E_ratio) ** (-h_z) * np.exp(exponent)
    return float(max(0.0, rate))


def avalanche_growth_rate(params: RunawayParams, n_RE: float, coulomb_log: float = 15.0) -> float:
    """
    Avalanche multiplication rate [m^-3 s^-1].
    Rosenbluth & Putvinski, Nucl. Fusion 37, 1355 (1997), Eq. 66.
    dn_RE/dt = n_RE (E/E_c - 1) / τ_c where τ_c already includes lnΛ.
    """
    if n_RE <= 0.0 or params.E_par <= 0.0:
        return 0.0

    E_c = critical_field(params.ne_20, coulomb_log)
    if params.E_par <= E_c:
        return 0.0

    n_e = params.ne_20 * 1e20
    tau_c = 4.0 * np.pi * EPS_0**2 * M_E**2 * C_LIGHT**3 / (n_e * E_CHARGE**4 * coulomb_log)

    rate = n_RE * (params.E_par / E_c - 1.0) / tau_c
    return float(max(0.0, rate))


def hot_tail_seed(
    Te_pre_keV: float, Te_post_keV: float, ne_20: float, quench_time_ms: float
) -> float:
    """
    Seed RE density from thermal quench hot-tail mechanism [m^-3].
    Smith, H.M. et al., Phys. Plasmas 15, 072502 (2008).
    Faster TQ → lower v_c/v_te → more seed electrons.
    """
    if Te_post_keV >= Te_pre_keV or Te_post_keV <= 0:
        return 0.0

    ratio = Te_pre_keV / Te_post_keV
    # v_c/v_te parametric fit to Smith 2008 Fig. 3
    V_C_V_TE_REF = 4.0  # at τ_q = 1 ms reference
    v_c_v_te = V_C_V_TE_REF * (quench_time_ms / 1.0) ** 0.2

    n_e = ne_20 * 1e20
    exponent = -(v_c_v_te**2)
    if exponent < -500:
        return 0.0

    n_seed = n_e * v_c_v_te**3 * np.exp(exponent) * ratio**1.5
    return float(max(0.0, n_seed))


class RunawayEvolution:
    def __init__(self, params: RunawayParams):
        self.params = params

    def step(self, dt: float, n_RE: float, E_par: float) -> float:
        self.params.E_par = E_par

        rate_D = dreicer_generation_rate(self.params)
        rate_A = avalanche_growth_rate(self.params, n_RE)

        dn_RE = (rate_D + rate_A) * dt
        return float(n_RE + dn_RE)

    def evolve(
        self,
        n_RE_0: float,
        E_par_profile: Callable[[float], float],
        t_span: tuple[float, float],
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_start, t_end = t_span
        n_steps = int(np.ceil((t_end - t_start) / dt))

        t_arr = np.linspace(t_start, t_end, n_steps + 1)
        n_RE_arr = np.zeros(n_steps + 1)
        n_RE_arr[0] = n_RE_0

        for i in range(n_steps):
            E_par = E_par_profile(t_arr[i])
            n_RE_arr[i + 1] = self.step(dt, n_RE_arr[i], E_par)

        return t_arr, n_RE_arr

    def current_fraction(self, n_RE: float, I_p_MA: float) -> float:
        """Fraction of total plasma current carried by REs (v ≈ c)."""
        if I_p_MA <= 0.0:
            return 0.0
        j_RE = E_CHARGE * n_RE * C_LIGHT
        I_RE = j_RE * np.pi * self.params.a**2
        return float(min(1.0, I_RE / (I_p_MA * 1e6)))


class RunawayMitigationAssessment:
    @staticmethod
    def required_density_for_suppression(
        E_par: float, Z_eff: float, coulomb_log: float = 15.0
    ) -> float:
        """
        Density [10^20 m^-3] needed to make E_c > E_par.
        """
        if E_par <= 0.0:
            return 0.0
        # E_c = n_e * e^3 lnL / (4 pi eps0^2 m_e c^2)
        # n_e = E_par * (4 pi eps0^2 m_e c^2) / (e^3 lnL)
        n_e_m3 = E_par * (4.0 * np.pi * EPS_0**2 * M_E * C_LIGHT**2) / (E_CHARGE**3 * coulomb_log)
        return float(n_e_m3 / 1e20)

    @staticmethod
    def maximum_re_energy(B0: float, R0: float) -> float:
        """Maximum RE energy [MeV] limited by synchrotron radiation or drift orbit."""
        # While the runaway orbit limit is E ~ e B0 R0 c (which is ~10 GeV for ITER),
        # synchrotron radiation limits the maximum energy to ~25-30 MeV in ITER.
        # We return a heuristic based on this limit.
        return 25.0

    @staticmethod
    def wall_heat_load(n_RE: float, E_max_MeV: float, A_wet: float, volume: float = 800.0) -> float:
        """Deposited energy density [MJ/m^2] assuming instant loss."""
        # Total energy = n_RE * Volume * E_avg
        # Assume E_avg ~ E_max / 2
        E_avg_J = (E_max_MeV / 2.0) * 1e6 * E_CHARGE
        W_total_J = n_RE * volume * E_avg_J

        if A_wet <= 0.0:
            return float("inf")

        load_MJ_m2 = (W_total_J / 1e6) / A_wet
        return float(load_MJ_m2)
