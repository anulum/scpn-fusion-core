# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disruption Sequence Model
"""Reduced disruption sequence model spanning TQ, CQ, runaway, and halo phases."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ellipe

from scpn_fusion.core.runaway_electrons import RunawayEvolution, RunawayParams, hot_tail_seed


def _require_positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass
class DisruptionConfig:
    """Input plasma and machine parameters for a disruption-sequence run."""

    R0: float
    a: float
    B0: float
    kappa: float
    Ip_MA: float
    W_th_MJ: float
    Te_pre_keV: float
    ne_pre_20: float
    dBr_over_B_trigger: float


@dataclass
class TQResult:
    """Thermal-quench duration, residual temperature, and wall heat load."""

    tau_tq_ms: float
    post_tq_Te_eV: float
    heat_load_MJ_m2: float


@dataclass
class CQResult:
    """Current-quench timescale and time traces for plasma current and electric field."""

    cq_duration_ms: float
    Ip_trace: np.ndarray
    E_par_trace: np.ndarray


@dataclass
class REResult:
    """Runaway-electron beam current, stored energy, and termination heat load."""

    I_RE_MA: float
    W_RE_MJ: float
    wall_heat_load_MJ_m2: float


@dataclass
class HaloResult:
    """Halo-current force metrics and ITER-style load-limit status."""

    f_halo: float
    tpf: float
    F_z_MN: float
    F_sideways_MN: float
    within_iter_limits: bool


@dataclass
class DisruptionResult:
    """Aggregated thermal, current, runaway, and halo disruption outputs."""

    tq_result: TQResult
    cq_result: CQResult
    re_result: REResult
    halo_result: HaloResult
    total_duration_ms: float
    wall_heat_load_MJ_m2: float
    vessel_force_MN: float


class ThermalQuench:
    """Thermal-quench phase model based on stochastic magnetic transport."""

    def __init__(self, W_th_MJ: float, a: float, R0: float, q: float, B0: float):
        self.W_th_MJ = W_th_MJ
        self.a = a
        self.R0 = R0
        self.q = q
        self.B0 = B0

    def rechester_rosenbluth_chi(self, dBr_over_B: float, v_e: float) -> float:
        """
        Stochastic heat transport chi [m^2/s].
        chi_stoch = v_e * pi * q * R0 * (delta B_r / B)^2
        """
        return v_e * math.pi * self.q * self.R0 * (dBr_over_B**2)

    def quench_timescale(self, dBr_over_B: float, Te_pre_keV: float) -> float:
        """tau_TQ [s]"""
        if dBr_over_B <= 0.0:
            return float("inf")

        # v_e = sqrt(2 * Te / m_e)
        e_charge = 1.602e-19
        m_e = 9.109e-31
        Te_J = Te_pre_keV * 1e3 * e_charge
        v_e = math.sqrt(2.0 * Te_J / m_e)

        chi = self.rechester_rosenbluth_chi(dBr_over_B, v_e)

        # tau ~ a^2 / chi
        return self.a**2 / chi

    def heat_deposition(
        self, W_th_MJ: float, A_wall_m2: float, peaking_factor: float = 3.0
    ) -> float:
        """Peak heat flux [MJ/m^2]"""
        if A_wall_m2 <= 0.0:
            return float("inf")
        return (W_th_MJ / A_wall_m2) * peaking_factor

    def post_tq_temperature(
        self, Te_pre_keV: float, tau_tq_ms: float, tau_radiation_ms: float = 0.5
    ) -> float:
        """Residual Te [eV] after quench and radiation cooling."""
        Te_pre_eV = _require_positive("Te_pre_keV", Te_pre_keV) * 1e3
        tau_tq_ms = _require_positive("tau_tq_ms", tau_tq_ms)
        tau_radiation_ms = _require_positive("tau_radiation_ms", tau_radiation_ms)

        cooling_exposure = tau_tq_ms / tau_radiation_ms
        radiation_multiplier = 44.0
        cold_floor_eV = 5.0 + 45.0 * math.exp(-cooling_exposure)
        return cold_floor_eV + (Te_pre_eV - cold_floor_eV) * math.exp(
            -radiation_multiplier * cooling_exposure
        )


class CurrentQuench:
    """Current-quench phase model using post-quench Spitzer resistance."""

    def __init__(self, Ip_MA: float, L_plasma_uH: float, R0: float, a: float, kappa: float = 1.0):
        self.Ip_MA = Ip_MA
        self.L_plasma_H = L_plasma_uH * 1e-6
        self.R0 = R0
        self.a = a
        self.kappa = kappa

    def resistivity_post_tq(self, Te_eV: float, Z_eff: float) -> float:
        """Spitzer resistivity [Ohm m]"""
        ln_Lambda = 10.0  # Typically lower in cold plasma
        # eta = 1.65e-9 * Z_eff * ln_Lambda / (Te_keV^1.5)
        Te_keV = max(Te_eV / 1000.0, 1e-6)
        return float(1.65e-9 * Z_eff * ln_Lambda / (Te_keV**1.5))

    def cq_timescale(self, Te_eV: float, Z_eff: float) -> float:
        """tau_CQ = L / R_p [ms]"""
        eta = self.resistivity_post_tq(Te_eV, Z_eff)
        # R_p = eta * 2 R0 / (a^2 kappa) — toroidal resistance
        R_p = eta * 2.0 * self.R0 / (self.a**2 * self.kappa)

        tau_s = self.L_plasma_H / R_p
        return tau_s * 1000.0

    def induced_electric_field(self, dIp_dt: float) -> float:
        """E_par = L / (2 pi R0) * |dIp/dt| [V/m] where Ip is in Amps."""
        return self.L_plasma_H / (2.0 * math.pi * self.R0) * abs(dIp_dt)

    def evolve(self, Te_post_tq_eV: float, Z_eff: float, dt: float, n_steps: int) -> CQResult:
        """Integrate exponential current decay and induced electric field traces."""
        tau_cq_ms = self.cq_timescale(Te_post_tq_eV, Z_eff)
        tau_cq_s = tau_cq_ms / 1000.0

        t_arr = np.linspace(0, n_steps * dt, n_steps)
        Ip_trace = self.Ip_MA * 1e6 * np.exp(-t_arr / tau_cq_s)

        dIp_dt = -(self.Ip_MA * 1e6) / tau_cq_s * np.exp(-t_arr / tau_cq_s)
        E_par_trace = self.L_plasma_H / (2.0 * math.pi * self.R0) * np.abs(dIp_dt)

        return CQResult(
            cq_duration_ms=tau_cq_ms,
            Ip_trace=np.asarray(Ip_trace / 1e6),
            E_par_trace=np.asarray(E_par_trace),
        )


class REBeamPhase:
    """Runaway-electron beam current, energy, and termination-load calculations."""

    def __init__(self, re_evolution: RunawayEvolution):
        self.re = re_evolution

    def beam_current(self, n_RE: float, v_par: float, A_beam: float) -> float:
        """I_RE [MA]"""
        e_charge = 1.602e-19
        I_A = n_RE * e_charge * v_par * A_beam
        return I_A / 1e6

    def beam_energy(self, n_RE: float, E_max_MeV: float, V_beam: float) -> float:
        """W_RE [MJ]"""
        e_charge = 1.602e-19
        E_avg_J = (E_max_MeV / 2.0) * 1e6 * e_charge
        W_J = n_RE * V_beam * E_avg_J
        return W_J / 1e6

    def termination_heat_load(self, W_RE_MJ: float, A_deposition_m2: float) -> float:
        """Return localized runaway-electron termination heat load in MJ/m^2."""
        if A_deposition_m2 <= 0.0:
            return float("inf")
        return W_RE_MJ / A_deposition_m2


class HaloCurrentModel:
    """Halo-current load model for vertical and asymmetric sideways vessel forces."""

    def __init__(self, Ip_MA: float, R0: float, B0: float, kappa: float):
        self.Ip_MA = Ip_MA
        self.R0 = R0
        self.B0 = B0
        self.kappa = kappa

    def halo_fraction(self, dZ_dt: float, tau_cq_ms: float) -> float:
        """Fraction of initial Ip."""
        # Empirical scaling: faster VDE -> higher f_halo
        # f_halo scales roughly as ~ 1 / tau_cq
        # Usually between 0.1 and 0.5.
        f_halo = 0.5 * math.exp(-tau_cq_ms / 100.0) + 0.1 * abs(dZ_dt)
        return min(max(f_halo, 0.1), 0.6)

    def toroidal_peaking_factor(self, n_mode: int = 1) -> float:
        """TPF for n=1 asymmetry"""
        if n_mode < 1:
            raise ValueError("n_mode must be >= 1")
        base_asymmetry = 1.45 / (n_mode**0.55)
        elongation_drive = 0.18 * max(self.kappa - 1.0, 0.0)
        return min(2.8, max(1.05, 1.0 + base_asymmetry + elongation_drive))

    def vertical_force(self, f_halo: float, tpf: float) -> float:
        """Vertical vessel load from toroidal halo-current asymmetry."""
        if not math.isfinite(f_halo) or not 0.0 <= f_halo <= 1.0:
            raise ValueError("f_halo must be finite and within [0, 1]")
        _require_positive("tpf", tpf)
        I_halo = f_halo * self.Ip_MA * 1e6
        F_N = I_halo * self.B0 * (2.0 * math.pi * self.R0) * tpf
        return F_N / 1e6

    def sideways_force(self, f_halo: float, tpf: float) -> float:
        """F_sideways [MN]"""
        # Roughly comparable to vertical force for n=1
        return self.vertical_force(f_halo, tpf) * 0.5

    def iter_limit_check(self, f_halo: float, tpf: float) -> bool:
        """f_halo * TPF < 0.75"""
        return (f_halo * tpf) < 0.75


class DisruptionSequence:
    """End-to-end reduced disruption scenario executor."""

    def __init__(self, config: DisruptionConfig):
        self.config = config
        self.V_plasma = 2.0 * math.pi**2 * config.R0 * config.a**2 * config.kappa
        self.A_wall = self._elliptic_torus_area(config.R0, config.a, config.kappa)

    @staticmethod
    def _elliptic_torus_area(R0: float, a: float, kappa: float) -> float:
        """Return toroidal area from the exact perimeter of an elliptical cross-section."""
        _require_positive("R0", R0)
        _require_positive("a", a)
        _require_positive("kappa", kappa)
        minor = a
        vertical = a * kappa
        major_axis = max(minor, vertical)
        minor_axis = min(minor, vertical)
        eccentricity_sq = 1.0 - (minor_axis / major_axis) ** 2
        perimeter = 4.0 * major_axis * float(ellipe(eccentricity_sq))
        return 2.0 * math.pi * R0 * perimeter

    def run(self, spi_density_target: float | None = None) -> DisruptionResult:
        """Run the disruption phases and optionally apply SPI density mitigation."""
        # Phase 1: TQ
        tq = ThermalQuench(self.config.W_th_MJ, self.config.a, self.config.R0, 3.0, self.config.B0)
        tau_tq_s = tq.quench_timescale(self.config.dBr_over_B_trigger, self.config.Te_pre_keV)
        tau_tq_ms = tau_tq_s * 1000.0

        post_T = tq.post_tq_temperature(self.config.Te_pre_keV, tau_tq_ms)
        tq_load = tq.heat_deposition(self.config.W_th_MJ, self.A_wall, 3.0)

        tq_res = TQResult(tau_tq_ms, post_T, tq_load)

        # Mitigation effect on parameters
        if spi_density_target is not None:
            ne_20 = spi_density_target
            Z_eff = 1.0  # High density SPI (e.g. D2) dilution, or high Z if Ne/Ar
            # For simplicity, just use the target density
        else:
            ne_20 = self.config.ne_pre_20
            Z_eff = 1.5

        # Phase 2: CQ
        cq = CurrentQuench(
            self.config.Ip_MA, 10.0, self.config.R0, self.config.a, self.config.kappa
        )

        # We want to simulate ~150 ms
        dt = 1e-3
        n_steps = 150
        cq_res = cq.evolve(post_T, Z_eff, dt, n_steps)

        # Phase 3: RE
        re_params = RunawayParams(
            ne_20, post_T / 1000.0, 0.0, Z_eff, self.config.B0, self.config.R0
        )
        re_model = RunawayEvolution(re_params)

        # Initial seed from the Smith hot-tail model.
        n_RE = hot_tail_seed(self.config.Te_pre_keV, post_T / 1000.0, ne_20, tau_tq_ms)

        # Evolve REs using E_par from CQ
        for E_p in cq_res.E_par_trace:
            n_RE = re_model.step(dt, n_RE, E_p)

        re_phase = REBeamPhase(re_model)
        A_beam = math.pi * (self.config.a / 2) ** 2 * self.config.kappa  # beam cross section
        V_beam = 2.0 * math.pi * self.config.R0 * A_beam
        I_RE = re_phase.beam_current(n_RE, 3e8, A_beam)
        W_RE = re_phase.beam_energy(n_RE, 25.0, V_beam)

        # Conservation clamp
        I_RE = min(I_RE, self.config.Ip_MA)

        re_load = re_phase.termination_heat_load(W_RE, 1.0)  # Localized termination

        re_res = REResult(I_RE, W_RE, re_load)

        # Phase 4: Halo
        halo = HaloCurrentModel(
            self.config.Ip_MA, self.config.R0, self.config.B0, self.config.kappa
        )
        f_halo = halo.halo_fraction(0.1, cq_res.cq_duration_ms)
        tpf = halo.toroidal_peaking_factor()

        F_z = halo.vertical_force(f_halo, tpf)
        F_side = halo.sideways_force(f_halo, tpf)
        within = halo.iter_limit_check(f_halo, tpf)

        halo_res = HaloResult(f_halo, tpf, F_z, F_side, within)

        tot_dur = tau_tq_ms + cq_res.cq_duration_ms
        tot_heat = tq_load + re_load

        return DisruptionResult(tq_res, cq_res, re_res, halo_res, tot_dur, tot_heat, F_z)

    def with_mitigation(self, spi_density_target: float) -> DisruptionResult:
        """Run the same sequence with a specified SPI density target."""
        return self.run(spi_density_target)
