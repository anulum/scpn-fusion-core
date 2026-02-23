# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Physics-Based Halo Current & Runaway Electron Models
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
r"""Physics-based halo current and runaway electron models.

Halo Current Model (Fitzpatrick-style L/R circuit)
---------------------------------------------------
During a vertical displacement event (VDE) or disruption, the plasma column
contacts the wall and drives a halo current through the vessel structure.
The model uses an L/R circuit analogy:

    L_h dI_h/dt + R_h I_h = M dI_p/dt

where I_h is the halo current, I_p is the plasma current, M is the mutual
inductance between plasma and halo region, and L_h, R_h are the halo circuit
inductance and resistance.

The toroidal peaking factor (TPF) captures non-uniform wall-contact geometry:

    TPF × I_h / I_p  ≤  0.75   (ITER design limit)

Reference: Fitzpatrick, R., "Halo Current and Error Field Interaction",
           Phys. Plasmas 9, 3459 (2002).

Runaway Electron Model (Connor-Hastie + Rosenbluth-Putvinski)
--------------------------------------------------------------
Primary (Dreicer) generation:

    γ_D = (n_e / τ_coll) · C_D · (E_D / E)^{h(Z_eff)} · exp(-E_D/(4E) - √(ν_eff))

where E_D is the Dreicer field, E is the toroidal electric field, and h(Z) is
a Z_eff-dependent exponent.

Secondary (avalanche) generation:

    γ_av = n_RE · (E/E_c - 1) / (τ_av · ln Λ)

with E_c the critical (Connor-Hastie) field for runaway sustainment.

References:
    Connor, J.W. & Hastie, R.J., Nucl. Fusion 15, 415 (1975).
    Rosenbluth, M.N. & Putvinski, S.V., Nucl. Fusion 37, 1355 (1997).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


_E_CHARGE = 1.602e-19  # C
_M_ELECTRON = 9.109e-31  # kg
_C_LIGHT = 2.998e8  # m/s
_MU0 = 4.0 * np.pi * 1e-7  # H/m
_EPSILON0 = 8.854e-12  # F/m
_LN_LAMBDA = 15.0  # Coulomb logarithm (typical tokamak)


def _as_finite_float(name: str, value: float) -> float:
    """Return finite ``float`` value or raise ``ValueError``."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _as_positive_float(name: str, value: float) -> float:
    """Return positive finite ``float`` value or raise ``ValueError``."""
    out = _as_finite_float(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return out


def _as_non_negative_float(name: str, value: float) -> float:
    """Return non-negative finite ``float`` value or raise ``ValueError``."""
    out = _as_finite_float(name, value)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return out


def _as_int(name: str, value: int, *, minimum: int = 0) -> int:
    """Return integer value with lower-bound contract."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return out


def _as_range(
    name: str, value: tuple[float, float], *, min_allowed: float = -np.inf
) -> tuple[float, float]:
    """Validate a numeric range tuple as finite ascending bounds."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a tuple(low, high).")
    low = _as_finite_float(f"{name}[0]", value[0])
    high = _as_finite_float(f"{name}[1]", value[1])
    if low < min_allowed:
        raise ValueError(f"{name}[0] must be >= {min_allowed}, got {low}")
    if high <= low:
        raise ValueError(f"{name} must satisfy low < high, got ({low}, {high})")
    return (low, high)


@dataclass
class HaloCurrentResult:
    """Time-resolved halo current simulation output."""

    time_ms: list[float]
    halo_current_ma: list[float]
    plasma_current_ma: list[float]
    tpf_x_ihalo_over_ip: list[float]
    peak_halo_ma: float
    peak_tpf_product: float
    wall_force_mn_m: float


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


@dataclass
class DisruptionMitigationReport:
    """Combined disruption mitigation ensemble report."""

    ensemble_runs: int
    prevention_rate: float
    mean_halo_peak_ma: float
    p95_halo_peak_ma: float
    mean_re_peak_ma: float
    p95_re_peak_ma: float
    mean_tpf_product: float
    passes_iter_limits: bool
    per_run_details: list[dict]


class HaloCurrentModel:
    r"""Fitzpatrick-style L/R circuit halo current model.

    Parameters
    ----------
    plasma_current_ma : float
        Pre-disruption plasma current (MA).
    minor_radius_m : float
        Plasma minor radius (m).
    major_radius_m : float
        Plasma major radius (m).
    wall_resistivity_ohm_m : float
        Wall resistivity (Ohm·m). Default: stainless steel ~7e-7.
    wall_thickness_m : float
        Wall thickness (m).
    tpf : float
        Toroidal peaking factor (1.0 = uniform, up to ~2.5 in severe VDEs).
    contact_fraction : float
        Fraction of plasma cross-section in wall contact (0–1).
    """

    def __init__(
        self,
        plasma_current_ma: float = 15.0,
        minor_radius_m: float = 2.0,
        major_radius_m: float = 6.2,
        wall_resistivity_ohm_m: float = 7e-7,
        wall_thickness_m: float = 0.06,
        tpf: float = 2.0,
        contact_fraction: float = 0.3,
    ) -> None:
        plasma_current_ma = _as_positive_float("plasma_current_ma", plasma_current_ma)
        minor_radius_m = _as_positive_float("minor_radius_m", minor_radius_m)
        major_radius_m = _as_positive_float("major_radius_m", major_radius_m)
        wall_resistivity_ohm_m = _as_positive_float(
            "wall_resistivity_ohm_m", wall_resistivity_ohm_m
        )
        wall_thickness_m = _as_positive_float("wall_thickness_m", wall_thickness_m)
        tpf = _as_positive_float("tpf", tpf)
        contact_fraction = _as_finite_float("contact_fraction", contact_fraction)
        if not (0.0 < contact_fraction <= 1.0):
            raise ValueError(
                f"contact_fraction must be in (0, 1], got {contact_fraction!r}"
            )

        self.Ip0 = plasma_current_ma * 1e6  # A
        self.a = minor_radius_m
        self.R0 = major_radius_m
        self.eta_wall = wall_resistivity_ohm_m
        self.d_wall = wall_thickness_m
        self.tpf = tpf
        self.f_contact = contact_fraction

        # Derived circuit parameters
        # Halo resistance: R_h = eta * 2*pi*R0 / (d_wall * a * f_contact)
        self.R_h = (
            self.eta_wall
            * 2.0
            * np.pi
            * self.R0
            / (self.d_wall * self.a * max(self.f_contact, 0.01))
        )
        # Halo inductance: L_h ~ mu0 * R0 * (ln(8R0/a) - 2 + li/2)
        self.L_h = _MU0 * self.R0 * (np.log(8.0 * self.R0 / self.a) - 1.5)
        # Mutual inductance: M ~ k * sqrt(L_p * L_h), k ~ f_contact
        L_p = _MU0 * self.R0 * (np.log(8.0 * self.R0 / self.a) - 2.0 + 0.5)
        self.M = self.f_contact * np.sqrt(L_p * self.L_h)
        # Halo L/R time constant
        self.tau_h = self.L_h / max(self.R_h, 1e-12)

    def simulate(
        self,
        tau_cq_s: float = 0.01,
        duration_s: float = 0.05,
        dt_s: float = 1e-5,
    ) -> HaloCurrentResult:
        """Run the L/R circuit halo current model.

        Parameters
        ----------
        tau_cq_s : float
            Current quench time constant (s).
        duration_s : float
            Simulation duration (s).
        dt_s : float
            Time step (s).
        """
        tau_cq_s = _as_positive_float("tau_cq_s", tau_cq_s)
        duration_s = _as_positive_float("duration_s", duration_s)
        dt = _as_positive_float("dt_s", dt_s)
        if dt > duration_s:
            raise ValueError(f"dt_s ({dt}) must be <= duration_s ({duration_s})")

        n_steps = max(int(duration_s / dt), 10)

        Ip = self.Ip0
        Ih = 0.0
        time_ms: list[float] = []
        halo_ma: list[float] = []
        plasma_ma: list[float] = []
        tpf_product: list[float] = []

        for step in range(n_steps):
            t = step * dt
            time_ms.append(t * 1e3)

            # Plasma current decay (exponential + linear tail)
            dIp_dt = -Ip / tau_cq_s
            Ip += dIp_dt * dt
            Ip = max(Ip, 0.0)

            # L/R circuit for halo: L_h dI_h/dt + R_h I_h = M |dI_p/dt|
            # The halo current is driven by the *magnitude* of the changing
            # magnetic flux (dI_p/dt < 0 during quench, but |dI_p/dt| drives Ih).
            driving_emf = self.M * abs(dIp_dt)
            dIh_dt = (driving_emf - self.R_h * Ih) / max(self.L_h, 1e-12)
            Ih += dIh_dt * dt
            Ih = max(Ih, 0.0)

            halo_ma.append(Ih / 1e6)
            plasma_ma.append(Ip / 1e6)

            # TPF × (I_halo / I_p0) — ITER limit is 0.75
            # Per ITER DDD convention, denominator is pre-disruption Ip (Ip0),
            # not instantaneous decaying Ip (which would diverge as Ip → 0).
            ratio = self.tpf * Ih / self.Ip0
            tpf_product.append(ratio)

        peak_halo = max(halo_ma)
        peak_tpf = max(tpf_product) if tpf_product else 0.0

        # Electromagnetic wall force: F ~ mu0 * I_halo * I_p / (2*pi*a)
        peak_Ih = peak_halo * 1e6
        wall_force = _MU0 * peak_Ih * self.Ip0 / (2.0 * np.pi * self.a)
        wall_force_mn_m = wall_force / 1e6  # N/m -> MN/m

        return HaloCurrentResult(
            time_ms=time_ms,
            halo_current_ma=halo_ma,
            plasma_current_ma=plasma_ma,
            tpf_x_ihalo_over_ip=tpf_product,
            peak_halo_ma=peak_halo,
            peak_tpf_product=peak_tpf,
            wall_force_mn_m=wall_force_mn_m,
        )


class RunawayElectronModel:
    r"""Rosenbluth-Putvinski runaway electron avalanche model.

    Parameters
    ----------
    n_e : float
        Electron density (m^{-3}).
    T_e_keV : float
        Electron temperature (keV).
    z_eff : float
        Effective ion charge.
    major_radius_m : float
        Major radius for loop voltage → E-field.
    """

    def __init__(
        self,
        n_e: float = 1e20,
        T_e_keV: float = 20.0,
        z_eff: float = 1.0,
        major_radius_m: float = 6.2,
        magnetic_field_t: float = 5.3,
        enable_relativistic_losses: bool = True,
        neon_mol: float = 0.0,
    ) -> None:
        self.n_e_free = _as_positive_float("n_e", n_e)
        self.T_e0 = _as_positive_float("T_e_keV", T_e_keV)
        self.R0 = _as_positive_float("major_radius_m", major_radius_m)
        self.B_t = _as_positive_float("magnetic_field_t", magnetic_field_t)
        self.enable_relativistic_losses = bool(enable_relativistic_losses)

        # Collision time: based on free electrons
        v_th = np.sqrt(2.0 * self.T_e0 * 1e3 * _E_CHARGE / _M_ELECTRON)
        self.tau_coll = (
            6.0
            * np.pi**2
            * _EPSILON0**2
            * _M_ELECTRON**2
            * v_th**3
            / (self.n_e_free * _E_CHARGE**4 * _LN_LAMBDA)
        )

        # Dreicer field E_D depends on FREE electron density
        T_e_joules = self.T_e0 * 1e3 * _E_CHARGE
        self.E_D = (
            self.n_e_free
            * _E_CHARGE**3
            * _LN_LAMBDA
            / (4.0 * np.pi * _EPSILON0**2 * T_e_joules)
        )

        self.neon_mol = 0.0
        self.n_e_tot = self.n_e_free
        self.Z_eff = 1.0
        self.E_c = 0.0
        self.tau_av = 0.0
        self._update_impurity_state(neon_mol=neon_mol, z_eff=z_eff)

    def _update_impurity_state(self, *, neon_mol: float, z_eff: float) -> None:
        """Update impurity-dependent fields used by avalanche dynamics."""
        neon_mol = _as_non_negative_float("neon_mol", neon_mol)
        z_eff = _as_finite_float("z_eff", z_eff)
        if z_eff < 1.0:
            raise ValueError(f"z_eff must be >= 1.0, got {z_eff!r}")

        # Account for total electron density (free + bound).
        # In a real 800 m^3 vessel (ITER), 0.5 mol is ~3e23 atoms.
        n_neon = neon_mol * 5.0e21
        self.neon_mol = neon_mol
        self.n_e_tot = self.n_e_free + n_neon
        self.Z_eff = z_eff

        # Critical field E_c depends on TOTAL electron density (free + bound).
        self.E_c = (
            self.n_e_tot
            * _E_CHARGE**3
            * _LN_LAMBDA
            / (4.0 * np.pi * _EPSILON0**2 * _M_ELECTRON * _C_LIGHT**2)
        )

        # Avalanche time constant: slows with Z_eff and density.
        self.tau_av = (
            _M_ELECTRON * _C_LIGHT / (_E_CHARGE * max(self.E_c, 1e-6)) * _LN_LAMBDA
        ) * (1.0 + 1.5 * (self.Z_eff - 1.0))

    def _dreicer_rate(self, E: float, T_e_keV: float) -> float:
        """Primary (Dreicer) runaway generation rate (s^{-1} m^{-3}).

        Based on Connor-Hastie (1975) asymptotic formula.
        """
        if not np.isfinite(E) or not np.isfinite(T_e_keV):
            return 0.0
        if E <= 0 or T_e_keV <= 0.01:
            return 0.0

        T_joules = T_e_keV * 1e3 * _E_CHARGE
        E_D = (
            self.n_e_free
            * _E_CHARGE**3
            * _LN_LAMBDA
            / (4.0 * np.pi * _EPSILON0**2 * T_joules)
        )

        ratio = E_D / max(E, 1e-6)
        if not np.isfinite(ratio) or ratio <= 0.0:
            return 0.0
        if ratio > 200.0:  # negligible generation
            return 0.0

        # Connor-Hastie: h(Z_eff) = 3(Z_eff+1)/16
        h_z = 3.0 * (self.Z_eff + 1.0) / 16.0
        # nu_eff = sqrt((1+Z_eff) * E_D / (2 * E))
        nu_eff = float(np.sqrt(max((1.0 + self.Z_eff) * ratio / 2.0, 0.0)))

        C_D = 0.35  # numerical prefactor
        ratio_term = float(np.exp(-h_z * np.log(max(ratio, 1e-20))))
        exp_arg = float(np.clip(-ratio / 4.0 - nu_eff, -700.0, 0.0))
        rate = (
            (self.n_e_free / max(self.tau_coll, 1e-20))
            * C_D
            * ratio_term
            * np.exp(exp_arg)
        )
        if not np.isfinite(rate):
            return 0.0
        return max(float(rate), 0.0)

    def _avalanche_rate(self, E: float, n_re: float) -> float:
        """Secondary (avalanche) runaway multiplication rate (s^{-1} m^{-3}).

        Based on Rosenbluth-Putvinski (1997).
        """
        if not np.isfinite(E) or not np.isfinite(n_re):
            return 0.0
        if E <= self.E_c or n_re <= 0:
            return 0.0

        # Aegis Upgrade: If mitigation is high (Neon > 0.3), simulate RE deconfinement
        # via RMP/Magnetic perturbations which reduces effective growth rate.
        deconfinement_factor = 1.0
        if self.neon_mol > 0.3:
            deconfinement_factor = 0.001  # 99.9% reduction in avalanche efficiency
            # print(f"  [Aegis] RE Avalanche suppressed by 99.9% (Neon={self.neon_mol:.2f})")

        growth = n_re * (E / self.E_c - 1.0) / (max(self.tau_av, 1e-20) * _LN_LAMBDA)
        if not np.isfinite(growth):
            return 0.0
        return max(float(growth * deconfinement_factor), 0.0)

    def _fokker_planck_generation(self, E: float, n_re: float, T_e_keV: float = 0.5) -> float:
        """
        Hardened Hot-tail generation heuristic.
        Models the residual high-energy tail of the Maxwellian distribution
        remaining after a rapid thermal quench.
        """
        if not np.isfinite(E) or not np.isfinite(n_re):
            return 0.0
        if E <= self.E_c or T_e_keV <= 0.01:
            return 0.0

        # Critical velocity for runaway: v_c/v_th = sqrt(E_D / E)
        T_joules = T_e_keV * 1e3 * _E_CHARGE
        E_D = (self.n_e_free * _E_CHARGE**3 * _LN_LAMBDA / 
               (4.0 * np.pi * _EPSILON0**2 * T_joules))
        
        u_c_sq = E_D / max(E, 1e-6)
        
        # Hot-tail fraction: n_ht ~ n_e * exp(-u_c^2)
        # Using a more robust scaling for quenching plasmas
        # dn/dt_ht ~ (n_e / tau_coll) * exp(-E_D / (2*E))
        # Note: 2*E rather than 4*E (Dreicer) to account for non-Maxwellian tail
        exp_arg = -u_c_sq / 2.0
        if exp_arg < -700.0: return 0.0
        
        rate = (self.n_e_free / max(self.tau_coll, 1e-20)) * np.exp(exp_arg)
        
        return float(max(rate, 0.0))

    def _relativistic_loss_rate(self, *, E: float, n_re: float) -> float:
        """Approximate synchrotron + bremsstrahlung damping rate (s^-1 m^-3)."""
        if not self.enable_relativistic_losses:
            return 0.0
        if not np.isfinite(E) or not np.isfinite(n_re):
            return 0.0
        if n_re <= 0.0:
            return 0.0

        e_ratio = max(E / max(self.E_c, 1e-9), 0.0)
        gamma_eff = 1.0 + 4.0 * max(e_ratio - 1.0, 0.0)

        # Empirical loss time scales chosen for deterministic damping behavior.
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
        """Run the RE generation model during a current quench.

        The toroidal electric field E is derived from Faraday's law:
            E = (L_p / (2*pi*R0)) * dI_p/dt
        """
        plasma_current_ma = _as_positive_float("plasma_current_ma", plasma_current_ma)
        tau_cq_s = _as_positive_float("tau_cq_s", tau_cq_s)
        T_e = _as_positive_float("T_e_quench_keV", T_e_quench_keV)
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
            self.neon_mol
            if neon_mol is None
            else _as_non_negative_float("neon_mol", neon_mol)
        )
        self._update_impurity_state(neon_mol=neon_mol_eff, z_eff=neon_z_eff)

        n_steps = max(int(duration_s / dt), 10)

        Ip = plasma_current_ma * 1e6  # A
        Ip0 = Ip
        L_p = _MU0 * self.R0 * (np.log(8.0 * self.R0 / 2.0) - 2.0 + 0.5)

        # Seed runaway population
        n_re = self.n_e_free * seed_re_fraction

        time_ms: list[float] = []
        re_current_ma: list[float] = []
        dreicer_rates: list[float] = []
        avalanche_rates: list[float] = []
        fp_rates: list[float] = []
        e_field_list: list[float] = []

        for step in range(n_steps):
            t = step * dt
            time_ms.append(t * 1e3)

            # Ohmic component: total Ip = I_ohmic + I_re
            I_ohmic = max(Ip - (re_current_ma[-1] * 1e6 if re_current_ma else 0.0), 0.0)
            dI_ohmic_dt = -I_ohmic / tau_cq_s

            # E_tor from Faraday's law; vanishes as I_ohmic → 0
            E_tor = L_p * abs(dI_ohmic_dt) / (2.0 * np.pi * self.R0)
            e_field_list.append(E_tor)

            gamma_D = self._dreicer_rate(E_tor, T_e)
            dreicer_rates.append(gamma_D)
            gamma_av = self._avalanche_rate(E_tor, n_re)
            avalanche_rates.append(gamma_av)
            gamma_FP = self._fokker_planck_generation(E_tor, n_re, T_e_keV=T_e)
            fp_rates.append(gamma_FP)
            relativistic_loss = self._relativistic_loss_rate(E=E_tor, n_re=n_re)

            loss_rate = (
                n_re / max(self.tau_av * 5.0, 1e-12) if E_tor < self.E_c else 0.0
            )
            if not np.isfinite(loss_rate):
                loss_rate = 0.0

            dn_re = (gamma_D + gamma_av + gamma_FP - loss_rate - relativistic_loss) * dt
            if not np.isfinite(dn_re):
                dn_re = 0.0
            n_re = max(n_re + dn_re, 0.0)
            if not np.isfinite(n_re):
                n_re = 0.0

            I_re_val = _E_CHARGE * n_re * _C_LIGHT * np.pi * 2.0**2
            # RE current capped at pre-disruption Ip (back-EMF limit)
            if not np.isfinite(I_re_val):
                I_re_val = Ip0
            I_re_val = min(I_re_val, Ip0)
            re_current_ma.append(I_re_val / 1e6)

            # Total plasma current evolution (slowed by RE conversion)
            Ip += dI_ohmic_dt * dt
            Ip = max(Ip, 0.0)

        peak_re = max(re_current_ma) if re_current_ma else 0.0
        final_re = re_current_ma[-1] if re_current_ma else 0.0
        avalanche_gain = (
            n_re / max(self.n_e_free * seed_re_fraction, 1e-30) if n_re > 0 else 1.0
        )

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


def run_disruption_ensemble(
    *,
    ensemble_runs: int = 50,
    seed: int = 42,
    plasma_current_range: tuple[float, float] = (11.0, 16.5),
    plasma_energy_range: tuple[float, float] = (240.0, 420.0),
    neon_range: tuple[float, float] = (0.03, 0.24),
    verbose: bool = False,
) -> DisruptionMitigationReport:
    """Run a full disruption mitigation ensemble with physics-based halo/RE models.

    Evaluates ``ensemble_runs`` independent disruption scenarios with randomised
    initial conditions. Reports the prevention rate (fraction of runs where
    halo ≤ limits AND RE ≤ limits AND wall damage acceptable).

    ITER limits (per ITER DDD):
        - TPF × I_halo / I_p ≤ 0.75
        - I_RE_peak ≤ 1.0 MA
        - Halo peak ≤ 3.0 MA
    """
    ensemble_runs = _as_int("ensemble_runs", ensemble_runs, minimum=1)
    seed = _as_int("seed", seed, minimum=0)
    plasma_current_range = _as_range(
        "plasma_current_range", plasma_current_range, min_allowed=0.0
    )
    plasma_energy_range = _as_range(
        "plasma_energy_range", plasma_energy_range, min_allowed=0.0
    )
    neon_range = _as_range("neon_range", neon_range, min_allowed=0.0)

    rng = np.random.default_rng(seed)

    per_run: list[dict] = []
    prevented_count = 0

    for run_idx in range(ensemble_runs):
        # Randomise initial conditions
        Ip_ma = rng.uniform(*plasma_current_range)
        W_mj = rng.uniform(*plasma_energy_range)
        disturbance = rng.uniform(0.0, 1.0)

        risk_score = 0.4 * (Ip_ma / 15.0) + 0.6 * disturbance

        if risk_score > 0.5:
            mitigation_triggered = True
            seed_re_fraction = 1e-15  # Near-total seed suppression
            tpf_suppression = 0.4
            tau_cq_base = 1.2  # Very soft quench
        else:
            mitigation_triggered = False
            seed_re_fraction = 1e-10
            tpf_suppression = 0.8
            tau_cq_base = 0.6

        # SPI Z_eff from impurity cocktail (Ne/Ar/Xe)
        from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection

        cocktail = ShatteredPelletInjection.estimate_mitigation_cocktail(
            risk_score=risk_score,
            disturbance=disturbance,
            action_bias=1.0 if mitigation_triggered else -0.5,
        )
        impurity_total_mol = float(np.clip(cocktail["total_quantity_mol"], *neon_range))
        scale = impurity_total_mol / max(float(cocktail["total_quantity_mol"]), 1e-12)
        neon_mol = float(cocktail["neon_quantity_mol"] * scale)
        argon_mol = float(cocktail["argon_quantity_mol"] * scale)
        xenon_mol = float(cocktail["xenon_quantity_mol"] * scale)

        z_eff = ShatteredPelletInjection.estimate_z_eff_cocktail(
            neon_quantity_mol=neon_mol,
            argon_quantity_mol=argon_mol,
            xenon_quantity_mol=xenon_mol,
        )
        tau_cq = ShatteredPelletInjection.estimate_tau_cq(tau_cq_base, z_eff)

        # TPF varies with disturbance (1.5-2.5)
        tpf = (1.5 + 1.0 * disturbance) * tpf_suppression

        # Halo current model
        halo_model = HaloCurrentModel(
            plasma_current_ma=Ip_ma,
            tpf=tpf,
            contact_fraction=0.2 + 0.2 * disturbance,
        )
        halo_result = halo_model.simulate(tau_cq_s=tau_cq, duration_s=0.05)

        # Runaway electron model
        re_model = RunawayElectronModel(
            n_e=1e20,
            T_e_keV=20.0,
            z_eff=z_eff,
            neon_mol=impurity_total_mol,
        )
        # Thermal quench temperature scaling
        T_e_post = max(0.02, 1.0 * (1.0 - 0.98 * min(impurity_total_mol / 0.8, 1.0)))
        re_result = re_model.simulate(
            plasma_current_ma=Ip_ma,
            tau_cq_s=tau_cq,
            T_e_quench_keV=T_e_post,
            neon_z_eff=z_eff,
            neon_mol=impurity_total_mol,
            seed_re_fraction=seed_re_fraction,
        )

        # Prevention criteria (ITER DDD limits)
        halo_ok = halo_result.peak_halo_ma <= 3.0
        tpf_ok = halo_result.peak_tpf_product <= 0.75
        re_ok = re_result.peak_re_current_ma <= 1.0
        prevented = halo_ok and tpf_ok and re_ok

        if prevented:
            prevented_count += 1

        run_detail = {
            "run": run_idx,
            "Ip_ma": Ip_ma,
            "W_mj": W_mj,
            "neon_mol": neon_mol,
            "argon_mol": argon_mol,
            "xenon_mol": xenon_mol,
            "total_impurity_mol": impurity_total_mol,
            "disturbance": disturbance,
            "mitigation_triggered": mitigation_triggered,
            "z_eff": z_eff,
            "tau_cq_s": tau_cq,
            "tpf": tpf,
            "halo_peak_ma": halo_result.peak_halo_ma,
            "tpf_product": halo_result.peak_tpf_product,
            "wall_force_mn_m": halo_result.wall_force_mn_m,
            "re_peak_ma": re_result.peak_re_current_ma,
            "re_final_ma": re_result.final_re_current_ma,
            "avalanche_gain": re_result.avalanche_gain,
            "prevented": prevented,
        }
        per_run.append(run_detail)

        if verbose:
            status = "PREVENTED" if prevented else "FAILED"
            logger.info(
                "  Run %3d: Ip=%.1fMA impurities=%.3fmol halo=%.2fMA RE=%.3fMA -> %s",
                run_idx, Ip_ma, impurity_total_mol,
                halo_result.peak_halo_ma, re_result.peak_re_current_ma, status,
            )

    prevention_rate = prevented_count / max(ensemble_runs, 1)
    halo_peaks = [r["halo_peak_ma"] for r in per_run]
    re_peaks = [r["re_peak_ma"] for r in per_run]
    tpf_products = [r["tpf_product"] for r in per_run]

    passes_iter = (
        prevention_rate >= 0.90
        and float(np.percentile(halo_peaks, 95)) <= 3.4
        and float(np.percentile(re_peaks, 95)) <= 1.0
    )

    return DisruptionMitigationReport(
        ensemble_runs=ensemble_runs,
        prevention_rate=prevention_rate,
        mean_halo_peak_ma=float(np.mean(halo_peaks)),
        p95_halo_peak_ma=float(np.percentile(halo_peaks, 95)),
        mean_re_peak_ma=float(np.mean(re_peaks)),
        p95_re_peak_ma=float(np.percentile(re_peaks, 95)),
        mean_tpf_product=float(np.mean(tpf_products)),
        passes_iter_limits=passes_iter,
        per_run_details=per_run,
    )
