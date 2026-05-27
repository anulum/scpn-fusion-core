# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Relativistic Runaway Electron Dynamics
"""Runaway electron utility contracts for DREAM-style reduced-order workflows."""
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
    """Scalar plasma parameters used by reduced-order runaway balances."""
    ne_20: float  # electron density [10^20 m^-3]
    Te_keV: float  # electron temperature [keV]
    E_par: float  # parallel electric field [V/m]
    Z_eff: float  # effective charge
    B0: float  # magnetic field [T]
    R0: float  # major radius [m]
    a: float = 2.0  # minor radius [m]


@dataclass(frozen=True)
class DreamFluidBalance:
    """DREAM-style fluid runaway density balance for a single plasma state."""

    dreicer_source: float
    avalanche_source: float
    loss_source: float
    total_source: float
    runaway_fraction: float
    growth_time_s: float


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
    Te_pre_keV: float,
    Te_post_keV: float,
    ne_20: float,
    quench_time_ms: float,
    *,
    vc_vte_ref: float = 4.0,
    quench_exponent: float = 0.2,
) -> float:
    """
    Seed RE density from thermal quench hot-tail mechanism [m^-3].
    Smith, H.M. et al., Phys. Plasmas 15, 072502 (2008).
    Faster TQ → lower v_c/v_te → more seed electrons.
    """
    te_pre = float(Te_pre_keV)
    te_post = float(Te_post_keV)
    ne = float(ne_20)
    tau_q = float(quench_time_ms)
    vc_ref = float(vc_vte_ref)
    q_exp = float(quench_exponent)
    if not np.isfinite(te_pre) or te_pre <= 0.0:
        raise ValueError("Te_pre_keV must be finite and > 0.")
    if not np.isfinite(te_post) or te_post <= 0.0:
        raise ValueError("Te_post_keV must be finite and > 0.")
    if not np.isfinite(ne) or ne <= 0.0:
        raise ValueError("ne_20 must be finite and > 0.")
    if not np.isfinite(tau_q) or tau_q <= 0.0:
        raise ValueError("quench_time_ms must be finite and > 0.")
    if not np.isfinite(vc_ref) or vc_ref <= 0.0:
        raise ValueError("vc_vte_ref must be finite and > 0.")
    if not np.isfinite(q_exp) or q_exp <= 0.0:
        raise ValueError("quench_exponent must be finite and > 0.")
    if te_post >= te_pre:
        return 0.0

    ratio = te_pre / te_post
    # v_c/v_te parametric fit to Smith 2008 Fig. 3
    v_c_v_te = vc_ref * (tau_q / 1.0) ** q_exp

    n_e = ne * 1e20
    exponent = -(v_c_v_te**2)
    if exponent < -500:
        return 0.0

    n_seed = n_e * v_c_v_te**3 * np.exp(exponent) * ratio**1.5
    return float(max(0.0, n_seed))


def dream_fluid_density_balance(
    params: RunawayParams,
    n_RE: float,
    *,
    loss_time_s: float = np.inf,
    max_runaway_fraction: float = 1.0,
    coulomb_log: float = 15.0,
) -> DreamFluidBalance:
    """Evaluate the scalar density balance used by DREAM-style fluid runs.

    The contract is ``dn_RE/dt = S_Dreicer + gamma_avalanche n_RE - n_RE/tau_loss``.
    It is a fluid benchmark contract, not a kinetic DREAM distribution solver.
    """
    n_re = float(n_RE)
    tau_loss = float(loss_time_s)
    max_fraction = float(max_runaway_fraction)
    if not np.isfinite(n_re) or n_re < 0.0:
        raise ValueError("n_RE must be finite and >= 0.")
    if not np.isfinite(max_fraction) or max_fraction <= 0.0 or max_fraction > 1.0:
        raise ValueError("max_runaway_fraction must be finite and in (0, 1].")
    if np.isnan(tau_loss) or tau_loss <= 0.0:
        raise ValueError("loss_time_s must be positive or infinite.")

    n_e = params.ne_20 * 1e20
    if not np.isfinite(n_e) or n_e <= 0.0:
        raise ValueError("electron density must be finite and > 0.")
    if n_re > n_e * max_fraction:
        raise ValueError("n_RE exceeds the configured runaway density cap.")

    dreicer_source = dreicer_generation_rate(params, coulomb_log)
    avalanche_source = avalanche_growth_rate(params, n_re, coulomb_log)
    loss_source = 0.0 if np.isinf(tau_loss) else n_re / tau_loss
    total_source = dreicer_source + avalanche_source - loss_source
    growth_time_s = n_re / total_source if total_source > 0.0 and n_re > 0.0 else np.inf

    return DreamFluidBalance(
        dreicer_source=float(dreicer_source),
        avalanche_source=float(avalanche_source),
        loss_source=float(loss_source),
        total_source=float(total_source),
        runaway_fraction=float(n_re / n_e),
        growth_time_s=float(growth_time_s),
    )


class RunawayEvolution:
    """Time-domain reduced-order solver for runaway electron density evolution."""

    def __init__(self, params: RunawayParams):
        self.params = params

    def balance(
        self,
        n_RE: float,
        E_par: float,
        *,
        loss_time_s: float = np.inf,
        max_runaway_fraction: float = 1.0,
    ) -> DreamFluidBalance:
        """Return the instantaneous DREAM-style fluid density balance."""
        self.params.E_par = float(E_par)
        return dream_fluid_density_balance(
            self.params,
            n_RE,
            loss_time_s=loss_time_s,
            max_runaway_fraction=max_runaway_fraction,
        )

    def step(
        self,
        dt: float,
        n_RE: float,
        E_par: float,
        *,
        loss_time_s: float = np.inf,
        max_runaway_fraction: float = 1.0,
    ) -> float:
        """Advance one explicit time step of the reduced-order runaway balance."""
        dt_s = float(dt)
        n_re = float(n_RE)
        e_par = float(E_par)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt must be finite and > 0.")
        if not np.isfinite(n_re) or n_re < 0.0:
            raise ValueError("n_RE must be finite and >= 0.")
        if not np.isfinite(e_par):
            raise ValueError("E_par must be finite.")

        balance = self.balance(
            n_re,
            e_par,
            loss_time_s=loss_time_s,
            max_runaway_fraction=max_runaway_fraction,
        )
        n_cap = self.params.ne_20 * 1e20 * float(max_runaway_fraction)

        return float(min(n_cap, max(n_re + balance.total_source * dt_s, 0.0)))

    def evolve(
        self,
        n_RE_0: float,
        E_par_profile: Callable[[float], float],
        t_span: tuple[float, float],
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate runaway density over a profile-driven E∥(t) history."""
        if not isinstance(t_span, tuple) or len(t_span) != 2:
            raise ValueError("t_span must be a (t_start, t_end) tuple.")
        t_start, t_end = float(t_span[0]), float(t_span[1])
        dt_s = float(dt)
        n0 = float(n_RE_0)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt must be finite and > 0.")
        if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
            raise ValueError("t_span must be finite with t_end > t_start.")
        if not np.isfinite(n0) or n0 < 0.0:
            raise ValueError("n_RE_0 must be finite and >= 0.")

        n_steps = int(np.ceil((t_end - t_start) / dt_s))
        if n_steps < 1:
            raise ValueError("integration requires at least one time step.")

        t_arr = np.linspace(t_start, t_end, n_steps + 1)
        n_RE_arr = np.zeros(n_steps + 1)
        n_RE_arr[0] = n0

        for i in range(n_steps):
            E_par = E_par_profile(t_arr[i])
            if not np.isfinite(E_par):
                raise ValueError("E_par_profile must return finite values.")
            n_RE_arr[i + 1] = self.step(dt_s, n_RE_arr[i], float(E_par))

        return t_arr, n_RE_arr

    def current_fraction(self, n_RE: float, I_p_MA: float) -> float:
        """Fraction of total plasma current carried by REs (v ≈ c)."""
        n_re = float(n_RE)
        i_p_ma = float(I_p_MA)
        if not np.isfinite(n_re) or n_re < 0.0:
            raise ValueError("n_RE must be finite and >= 0.")
        if not np.isfinite(i_p_ma):
            raise ValueError("I_p_MA must be finite.")
        if i_p_ma <= 0.0:
            return 0.0
        j_RE = E_CHARGE * n_re * C_LIGHT
        I_RE = j_RE * np.pi * self.params.a**2
        return float(min(1.0, I_RE / (i_p_ma * 1e6)))


class RunawayMitigationAssessment:
    """Collection of reduced-order mitigation formulas and guard checks."""

    @staticmethod
    def required_density_for_suppression(
        E_par: float, Z_eff: float, coulomb_log: float = 15.0
    ) -> float:
        """
        Density [10^20 m^-3] needed to make E_c > E_par.
        """
        e_par = float(E_par)
        z_eff = float(Z_eff)
        ln_lambda = float(coulomb_log)
        if not np.isfinite(e_par) or e_par <= 0.0:
            return 0.0
        if not np.isfinite(z_eff) or z_eff <= 0.0:
            raise ValueError("Z_eff must be finite and > 0.")
        if not np.isfinite(ln_lambda) or ln_lambda <= 0.0:
            raise ValueError("coulomb_log must be finite and > 0.")
        # E_c = n_e * e^3 lnL / (4 pi eps0^2 m_e c^2)
        # Effective collisional drag scales with impurity content.
        collision_multiplier = 0.5 * (1.0 + z_eff)
        coeff = collision_multiplier * E_CHARGE**3 * ln_lambda
        # n_e = E_par * (4 pi eps0^2 m_e c^2) / (e^3 lnL)
        n_e_m3 = e_par * (4.0 * np.pi * EPS_0**2 * M_E * C_LIGHT**2) / coeff
        return float(n_e_m3 / 1e20)

    @staticmethod
    def maximum_re_energy(
        B0: float,
        R0: float,
        *,
        ne_20: float = 1.0,
        Z_eff: float = 1.5,
        pitch_angle_rad: float = 0.35,
    ) -> float:
        """Maximum RE energy [MeV] from orbit and synchrotron limits."""
        b = float(B0)
        r = float(R0)
        ne = float(ne_20)
        zeff = float(Z_eff)
        pitch = float(pitch_angle_rad)
        if not np.isfinite(b) or b <= 0.0:
            raise ValueError("B0 must be finite and > 0.")
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError("R0 must be finite and > 0.")
        if not np.isfinite(ne) or ne <= 0.0:
            raise ValueError("ne_20 must be finite and > 0.")
        if not np.isfinite(zeff) or zeff <= 0.0:
            raise ValueError("Z_eff must be finite and > 0.")
        if not np.isfinite(pitch) or pitch <= 0.0 or pitch >= (0.5 * np.pi):
            raise ValueError("pitch_angle_rad must be finite and in (0, pi/2).")

        # Drift-orbit limit: E_orbit[eV] ~ B[T] R[m] c[m/s].
        orbit_limit_mev = (b * r * C_LIGHT) * 1.0e-6

        # Synchrotron-loss constrained practical limit around ITER conditions.
        sin_pitch = np.sin(pitch)
        synch_limit_mev = 28.0
        synch_limit_mev *= (b / 5.3) ** (1.0 / 3.0)
        synch_limit_mev *= (r / 6.2) ** (2.0 / 3.0)
        synch_limit_mev *= (ne / 1.0) ** (-1.0 / 6.0)
        synch_limit_mev *= (zeff / 1.5) ** (-1.0 / 6.0)
        synch_limit_mev *= (max(sin_pitch, 1e-3) / np.sin(0.35)) ** (-1.0 / 3.0)

        return float(max(0.1, min(orbit_limit_mev, synch_limit_mev)))

    @staticmethod
    def wall_heat_load(
        n_RE: float,
        E_max_MeV: float,
        A_wet: float,
        volume: float = 800.0,
        *,
        mean_energy_MeV: float | None = None,
    ) -> float:
        """Deposited runaway-electron beam energy density [MJ/m^2]."""
        n_re = float(n_RE)
        e_max = float(E_max_MeV)
        area = float(A_wet)
        vol = float(volume)
        if not np.isfinite(n_re) or n_re < 0.0:
            raise ValueError("n_RE must be finite and >= 0.")
        if not np.isfinite(e_max) or e_max <= 0.0:
            raise ValueError("E_max_MeV must be finite and > 0.")
        if not np.isfinite(vol) or vol <= 0.0:
            raise ValueError("volume must be finite and > 0.")
        if mean_energy_MeV is None:
            e_mean = 0.5 * e_max
        else:
            e_mean = float(mean_energy_MeV)
            if not np.isfinite(e_mean) or e_mean <= 0.0 or e_mean > e_max:
                raise ValueError("mean_energy_MeV must be finite and within (0, E_max_MeV]")

        E_avg_J = e_mean * 1e6 * E_CHARGE
        W_total_J = n_re * vol * E_avg_J

        if area <= 0.0:
            return float("inf")

        load_MJ_m2 = (W_total_J / 1e6) / area
        return float(load_MJ_m2)
