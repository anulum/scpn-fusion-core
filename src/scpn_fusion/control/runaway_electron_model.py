# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Runaway Electron Model
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Runaway electron model extracted from ``halo_re_physics`` monolith."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


_E_CHARGE = 1.602e-19
_M_ELECTRON = 9.109e-31
_C_LIGHT = 2.998e8
_EPSILON0 = 8.854e-12
_LN_LAMBDA = 15.0
_MU0 = 4.0 * np.pi * 1e-7


def _as_finite_float(name: str, value: float) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _as_positive_float(name: str, value: float) -> float:
    out = _as_finite_float(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return out


def _as_non_negative_float(name: str, value: float) -> float:
    out = _as_finite_float(name, value)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return out


@dataclass
class RunawayElectronResult:
    """Time-resolved runaway electron simulation output."""

    time_ms: list[float]
    runaway_current_ma: list[float]
    dreicer_rate_per_s: list[float]
    avalanche_rate_per_s: list[float]
    electric_field_v_m: list[float]
    peak_re_current_ma: float
    final_re_current_ma: float
    avalanche_gain: float


class RunawayElectronModel:
    """Rosenbluth-Putvinski runaway electron avalanche model."""

    def __init__(
        self,
        n_e: float = 1e20,
        T_e_keV: float = 20.0,
        z_eff: float = 1.0,
        major_radius_m: float = 6.2,
        magnetic_field_t: float = 5.3,
        enable_relativistic_losses: bool = True,
        neon_mol: float = 0.0,
        runaway_beam_radius_m: float = 0.02,
    ) -> None:
        self.n_e_free = _as_positive_float("n_e", n_e)
        self.T_e0 = _as_positive_float("T_e_keV", T_e_keV)
        self.R0 = _as_positive_float("major_radius_m", major_radius_m)
        self.B_t = _as_positive_float("magnetic_field_t", magnetic_field_t)
        self.enable_relativistic_losses = bool(enable_relativistic_losses)
        self.runaway_beam_radius_m = _as_positive_float(
            "runaway_beam_radius_m", runaway_beam_radius_m
        )

        v_th = np.sqrt(2.0 * self.T_e0 * 1e3 * _E_CHARGE / _M_ELECTRON)
        self.tau_coll = (
            6.0
            * np.pi**2
            * _EPSILON0**2
            * _M_ELECTRON**2
            * v_th**3
            / (self.n_e_free * _E_CHARGE**4 * _LN_LAMBDA)
        )

        t_e_joules = self.T_e0 * 1e3 * _E_CHARGE
        self.E_D = (
            self.n_e_free
            * _E_CHARGE**3
            * _LN_LAMBDA
            / (4.0 * np.pi * _EPSILON0**2 * t_e_joules)
        )

        self.neon_mol = 0.0
        self.n_e_tot = self.n_e_free
        self.Z_eff = 1.0
        self.E_c = 0.0
        self.tau_av = 0.0
        self.max_runaway_fraction = 0.08
        self._update_impurity_state(neon_mol=neon_mol, z_eff=z_eff)

    def _update_impurity_state(self, *, neon_mol: float, z_eff: float) -> None:
        neon_mol = _as_non_negative_float("neon_mol", neon_mol)
        z_eff = _as_finite_float("z_eff", z_eff)
        if z_eff < 1.0:
            raise ValueError(f"z_eff must be >= 1.0, got {z_eff!r}")

        n_neon = neon_mol * 5.0e21
        self.neon_mol = neon_mol
        self.n_e_tot = self.n_e_free + n_neon
        self.Z_eff = z_eff

        self.E_c = (
            self.n_e_tot
            * _E_CHARGE**3
            * _LN_LAMBDA
            / (4.0 * np.pi * _EPSILON0**2 * _M_ELECTRON * _C_LIGHT**2)
        )

        self.tau_av = (
            _M_ELECTRON * _C_LIGHT / (_E_CHARGE * max(self.E_c, 1e-6)) * _LN_LAMBDA
        ) * (1.0 + 1.5 * (self.Z_eff - 1.0))
        self.max_runaway_fraction = float(
            np.clip(
                0.10 / (1.0 + 0.40 * max(self.Z_eff - 1.0, 0.0) + 10.0 * self.neon_mol),
                1e-5,
                0.08,
            )
        )

    def _dreicer_rate(self, E: float, T_e_keV: float) -> float:
        if not np.isfinite(E) or not np.isfinite(T_e_keV):
            return 0.0
        if E <= 0 or T_e_keV <= 0.01:
            return 0.0

        t_joules = T_e_keV * 1e3 * _E_CHARGE
        e_d = (
            self.n_e_free
            * _E_CHARGE**3
            * _LN_LAMBDA
            / (4.0 * np.pi * _EPSILON0**2 * t_joules)
        )

        ratio = e_d / max(E, 1e-6)
        if not np.isfinite(ratio) or ratio <= 0.0 or ratio > 200.0:
            return 0.0

        h_z = 3.0 * (self.Z_eff + 1.0) / 16.0
        nu_eff = float(np.sqrt(max((1.0 + self.Z_eff) * ratio / 2.0, 0.0)))
        c_d = 0.35
        ratio_term = float(np.exp(-h_z * np.log(max(ratio, 1e-20))))
        exp_arg = float(np.clip(-ratio / 4.0 - nu_eff, -700.0, 0.0))
        rate = (
            (self.n_e_free / max(self.tau_coll, 1e-20))
            * c_d
            * ratio_term
            * np.exp(exp_arg)
        )
        if not np.isfinite(rate):
            return 0.0
        return max(float(rate), 0.0)

    def _avalanche_rate(self, E: float, n_re: float) -> float:
        if not np.isfinite(E) or not np.isfinite(n_re):
            return 0.0
        if E <= self.E_c or n_re <= 0:
            return 0.0

        deconfinement_factor = 0.001 if self.neon_mol > 0.3 else 1.0
        growth = n_re * (E / self.E_c - 1.0) / (max(self.tau_av, 1e-20) * _LN_LAMBDA)
        if not np.isfinite(growth):
            return 0.0
        return max(float(growth * deconfinement_factor), 0.0)

    def _fokker_planck_generation(
        self, E: float, n_re: float, T_e_keV: float = 0.5
    ) -> float:
        if not np.isfinite(E) or not np.isfinite(n_re):
            return 0.0
        if E <= self.E_c or T_e_keV <= 0.01:
            return 0.0

        t_joules = T_e_keV * 1e3 * _E_CHARGE
        e_d = (
            self.n_e_free
            * _E_CHARGE**3
            * _LN_LAMBDA
            / (4.0 * np.pi * _EPSILON0**2 * t_joules)
        )
        u_c_sq = e_d / max(E, 1e-6)
        exp_arg = -u_c_sq / 2.0
        if exp_arg < -700.0:
            return 0.0

        hot_tail_prefactor = 2.5e-6 / (
            1.0 + 6.0 * self.neon_mol + 0.6 * max(self.Z_eff - 1.0, 0.0)
        )
        rate = (
            hot_tail_prefactor
            * (self.n_e_free / max(self.tau_coll, 1e-20))
            * np.exp(exp_arg)
        )
        return float(max(rate, 0.0))

    def _relativistic_loss_rate(self, *, E: float, n_re: float) -> float:
        if not self.enable_relativistic_losses:
            return 0.0
        if not np.isfinite(E) or not np.isfinite(n_re) or n_re <= 0.0:
            return 0.0

        e_ratio = max(E / max(self.E_c, 1e-9), 0.0)
        gamma_eff = 1.0 + 4.0 * max(e_ratio - 1.0, 0.0)
        tau_sync = 0.08 / max(self.B_t * self.B_t * gamma_eff, 1e-12)
        tau_brem = 0.12 / max(
            (1.0 + 0.08 * self.Z_eff) * (self.n_e_tot / 1e20) * gamma_eff, 1e-12
        )
        tau_rel = max(min(tau_sync, tau_brem), 1e-6)
        loss = n_re / tau_rel
        if not np.isfinite(loss):
            return 0.0
        return float(max(loss, 0.0))

    def simulate(
        self,
        plasma_current_ma: float = 15.0,
        tau_cq_s: float = 0.01,
        T_e_quench_keV: float = 0.5,
        neon_z_eff: float = 3.0,
        neon_mol: Optional[float] = None,
        duration_s: float = 0.05,
        dt_s: float = 1e-5,
        seed_re_fraction: float = 1e-8,
    ) -> RunawayElectronResult:
        plasma_current_ma = _as_positive_float("plasma_current_ma", plasma_current_ma)
        tau_cq_s = _as_positive_float("tau_cq_s", tau_cq_s)
        t_e = _as_positive_float("T_e_quench_keV", T_e_quench_keV)
        duration_s = _as_positive_float("duration_s", duration_s)
        dt = _as_positive_float("dt_s", dt_s)
        if dt > duration_s:
            raise ValueError(f"dt_s ({dt}) must be <= duration_s ({duration_s})")

        seed_re_fraction = _as_finite_float("seed_re_fraction", seed_re_fraction)
        if not (0.0 < seed_re_fraction <= 1.0):
            raise ValueError(
                f"seed_re_fraction must be in (0, 1], got {seed_re_fraction!r}"
            )
        neon_mol_eff = (
            self.neon_mol if neon_mol is None else _as_non_negative_float("neon_mol", neon_mol)
        )
        self._update_impurity_state(neon_mol=neon_mol_eff, z_eff=neon_z_eff)

        n_steps = max(int(duration_s / dt), 10)
        ip = plasma_current_ma * 1e6
        ip0 = ip
        l_p = _MU0 * self.R0 * (np.log(8.0 * self.R0 / 2.0) - 2.0 + 0.5)
        n_re = self.n_e_free * seed_re_fraction

        time_ms: list[float] = []
        re_current_ma: list[float] = []
        dreicer_rates: list[float] = []
        avalanche_rates: list[float] = []
        e_field_list: list[float] = []

        for step in range(n_steps):
            t = step * dt
            time_ms.append(t * 1e3)

            i_ohmic = max(ip - (re_current_ma[-1] * 1e6 if re_current_ma else 0.0), 0.0)
            d_i_ohmic_dt = -i_ohmic / tau_cq_s
            e_tor = l_p * abs(d_i_ohmic_dt) / (2.0 * np.pi * self.R0)
            e_field_list.append(e_tor)

            gamma_d = self._dreicer_rate(e_tor, t_e)
            gamma_av = self._avalanche_rate(e_tor, n_re)
            gamma_fp = self._fokker_planck_generation(e_tor, n_re, T_e_keV=t_e)
            dreicer_rates.append(gamma_d)
            avalanche_rates.append(gamma_av)
            rel_loss = self._relativistic_loss_rate(E=e_tor, n_re=n_re)
            mitigation_source_factor = 1.0 / (
                1.0 + 4.0 * self.neon_mol + 0.5 * max(self.Z_eff - 1.0, 0.0)
            )

            loss_rate = n_re / max(self.tau_av * 5.0, 1e-12) if e_tor < self.E_c else 0.0
            if not np.isfinite(loss_rate):
                loss_rate = 0.0

            source_rate = (gamma_d + gamma_av + gamma_fp) * mitigation_source_factor
            dn_re = (source_rate - loss_rate - rel_loss) * dt
            if not np.isfinite(dn_re):
                dn_re = 0.0
            n_re = max(n_re + dn_re, 0.0)
            if not np.isfinite(n_re):
                n_re = 0.0
            n_re = min(n_re, self.n_e_free * self.max_runaway_fraction)

            i_re_val = (
                _E_CHARGE
                * n_re
                * _C_LIGHT
                * np.pi
                * self.runaway_beam_radius_m**2
            )
            if not np.isfinite(i_re_val):
                i_re_val = ip0
            i_re_val = min(i_re_val, ip0)
            re_current_ma.append(i_re_val / 1e6)

            ip += d_i_ohmic_dt * dt
            ip = max(ip, 0.0)

        peak_re = max(re_current_ma) if re_current_ma else 0.0
        final_re = re_current_ma[-1] if re_current_ma else 0.0
        avalanche_gain = n_re / max(self.n_e_free * seed_re_fraction, 1e-30) if n_re > 0 else 1.0
        avalanche_gain = max(float(avalanche_gain), 1.0)

        return RunawayElectronResult(
            time_ms=time_ms,
            runaway_current_ma=re_current_ma,
            dreicer_rate_per_s=dreicer_rates,
            avalanche_rate_per_s=avalanche_rates,
            electric_field_v_m=e_field_list,
            peak_re_current_ma=peak_re,
            final_re_current_ma=final_re,
            avalanche_gain=avalanche_gain,
        )


__all__ = ["RunawayElectronModel", "RunawayElectronResult"]
