# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Edge Localized Mode (ELM) Model
"""Reduced peeling-ballooning, ELM crash, and RMP suppression models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require_positive(name: str, value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


class PeelingBallooningBoundary:
    """
    Evaluates stability against peeling-ballooning modes.
    """

    def __init__(self, q95: float, kappa: float, delta: float, a: float, R0: float):
        self.q95 = q95
        self.kappa = kappa
        self.delta = delta
        self.a = a
        self.R0 = R0

    def peeling_limit(self, j_edge: float, n_mode: int = 10) -> float:
        """
        Critical edge current density limit for peeling stability.
        """
        if n_mode < 1:
            raise ValueError("n_mode must be >= 1")
        q95 = _require_positive("q95", self.q95)
        minor_radius = _require_positive("a", self.a)
        major_radius = _require_positive("R0", self.R0)
        aspect = major_radius / minor_radius
        shaping = 1.0 + 0.22 * (self.kappa - 1.0) + 0.35 * self.delta**2
        mode_factor = 1.0 + 0.08 * np.log1p(n_mode)
        aspect_factor = np.sqrt(max(aspect, 1.0) / 3.0)
        j_crit = 1.0e6 * shaping * mode_factor * aspect_factor / max(q95, 2.0)
        return float(j_crit)

    def ballooning_limit(self, s_edge: float) -> float:
        """
        Critical alpha (pressure gradient) at edge.
        """
        # alpha_crit ~ s_edge
        # With shaping, kappa and delta increase the limit
        # Sauter et al., Phys. Plasmas 6, 2834 (1999): F_shape uses δ²
        alpha_crit = 0.5 * max(s_edge, 0.1) * (1.0 + self.kappa**2 * (1.0 + 2.0 * self.delta**2))
        return float(alpha_crit)

    def is_unstable(self, alpha_edge: float, j_edge: float, s_edge: float) -> bool:
        """
        Combined criterion for peeling-ballooning.
        """
        j_crit = self.peeling_limit(j_edge)
        a_crit = self.ballooning_limit(s_edge)

        # Elliptical coupling boundary
        j_norm = max(0.0, j_edge / j_crit)
        a_norm = max(0.0, alpha_edge / a_crit)

        coupling = 0.35 * j_norm * a_norm
        return (j_norm**2 + a_norm**2 + coupling) > 1.0

    def stability_margin(self, alpha_edge: float, j_edge: float, s_edge: float) -> float:
        """
        Distance to boundary (positive = stable).
        """
        j_crit = self.peeling_limit(j_edge)
        a_crit = self.ballooning_limit(s_edge)

        j_norm = max(0.0, j_edge / j_crit)
        a_norm = max(0.0, alpha_edge / a_crit)

        coupling = 0.35 * j_norm * a_norm
        return float(1.0 - np.sqrt(j_norm**2 + a_norm**2 + coupling))


@dataclass
class ELMCrashResult:
    """Pedestal state and heat-flux outputs from one ELM crash."""

    delta_W_MJ: float
    T_ped_post: float
    n_ped_post: float
    peak_heat_flux_MW_m2: float
    duration_ms: float


class ELMCrashModel:
    """
    Applies the Type I ELM crash to pedestal profiles.
    """

    def __init__(self, f_elm_fraction: float = 0.08):
        self.f_elm_fraction = f_elm_fraction

    def crash(self, T_ped: float, n_ped: float, W_ped: float, A_wet: float = 1.0) -> ELMCrashResult:
        """Apply a fractional Type-I ELM crash to pedestal scalar quantities."""
        delta_W_MJ = self.f_elm_fraction * W_ped

        # T and n drop proportionally
        # W ~ n T, if both drop by sqrt(1 - f), W drops by (1 - f)
        drop_factor = np.sqrt(1.0 - self.f_elm_fraction)

        T_ped_post = T_ped * drop_factor
        n_ped_post = n_ped * drop_factor

        duration_ms = 0.25

        peak_heat_flux = (delta_W_MJ / A_wet) / (duration_ms * 1e-3)

        return ELMCrashResult(delta_W_MJ, T_ped_post, n_ped_post, peak_heat_flux, duration_ms)

    def apply_to_profiles(
        self, rho: np.ndarray, Te: np.ndarray, ne: np.ndarray, rho_ped: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the crash drop factor to profile samples outside the pedestal top."""
        Te_new = Te.copy()
        ne_new = ne.copy()

        idx_ped = np.searchsorted(rho, rho_ped)

        drop_factor = np.sqrt(1.0 - self.f_elm_fraction)

        for i in range(idx_ped, len(rho)):
            # Flattening outside the pedestal top
            Te_new[i] *= drop_factor
            ne_new[i] *= drop_factor

        return Te_new, ne_new


@dataclass
class ELMEvent:
    """Timestamped ELM event summary emitted by the pedestal cycler."""

    time: float
    delta_W_MJ: float
    f_elm_Hz: float
    crash_type: str


class RMPSuppression:
    """
    Resonant Magnetic Perturbation effects on ELMs.
    """

    def __init__(self, n_coils: int = 3, I_rmp_kA: float = 90.0, n_toroidal: int = 3):
        self.n_coils = n_coils
        self.I_rmp_kA = I_rmp_kA
        self.n_toroidal = n_toroidal

    def chirikov_parameter(
        self, q_profile: np.ndarray, rho: np.ndarray, delta_B_r: float, B0: float, R0: float
    ) -> float:
        """
        σ_Chir = sum (w_mn / dr_mn)
        w_mn = 4 * sqrt(R0 * q' * delta_B_r / (n * B0))
        """
        q_profile = np.asarray(q_profile, dtype=float)
        rho = np.asarray(rho, dtype=float)
        if q_profile.ndim != 1 or rho.ndim != 1 or q_profile.size != rho.size:
            raise ValueError("q_profile and rho must be one-dimensional arrays with equal length")
        if q_profile.size < 3:
            raise ValueError("q_profile and rho must contain at least three samples")
        if not np.all(np.isfinite(q_profile)) or not np.all(np.isfinite(rho)):
            raise ValueError("q_profile and rho must be finite")
        if np.any(np.diff(rho) <= 0.0) or np.any(np.diff(q_profile) < 0.0):
            raise ValueError("rho and q_profile must be monotonic increasing")
        if delta_B_r <= 0.0 or B0 <= 0.0 or self.n_toroidal <= 0:
            return 0.0
        B0 = _require_positive("B0", B0)
        R0 = _require_positive("R0", R0)

        q_min = float(q_profile[0])
        q_max = float(q_profile[-1])
        m_min = int(np.ceil(self.n_toroidal * q_min))
        m_max = int(np.floor(self.n_toroidal * q_max))
        if m_max < m_min:
            return 0.0

        dq_drho = np.gradient(q_profile, rho)
        resonant_rho: list[float] = []
        widths: list[float] = []
        for m_mode in range(m_min, m_max + 1):
            q_res = m_mode / self.n_toroidal
            rho_res = float(np.interp(q_res, q_profile, rho))
            shear = float(np.interp(rho_res, rho, np.abs(dq_drho)))
            if shear <= 0.0:
                continue
            local_q = float(np.interp(rho_res, rho, q_profile))
            width = 4.0 * np.sqrt(R0 * local_q * abs(delta_B_r) / (self.n_toroidal * B0 * shear))
            resonant_rho.append(rho_res)
            widths.append(float(width))

        if len(widths) < 2:
            return float(widths[0]) if widths else 0.0

        rho_arr = np.asarray(resonant_rho)
        width_arr = np.asarray(widths)
        spacing = np.empty_like(rho_arr)
        spacing[0] = rho_arr[1] - rho_arr[0]
        spacing[-1] = rho_arr[-1] - rho_arr[-2]
        if len(rho_arr) > 2:
            spacing[1:-1] = 0.5 * (rho_arr[2:] - rho_arr[:-2])
        return float(np.sum(width_arr / np.maximum(spacing, 1e-6)))

    def suppressed(self, sigma_chir: float) -> bool:
        """Return whether Chirikov overlap is high enough for ELM suppression."""
        return sigma_chir > 1.0

    def pedestal_transport_enhancement(self, sigma_chir: float) -> float:
        """Estimate pedestal transport increase from Chirikov overlap above unity."""
        alpha = 2.0
        return 1.0 + alpha * max(0.0, sigma_chir - 1.0)

    def density_pump_out(self, sigma_chir: float) -> float:
        """Return the reduced-density fraction for suppressed RMP operation."""
        if sigma_chir > 1.0:
            return 0.2  # 20% reduction
        return 0.0


def elm_power_balance_frequency(P_SOL_MW: float, W_ped_MJ: float, f_elm_fraction: float) -> float:
    """Estimate ELM frequency from SOL power and fractional pedestal energy loss."""
    if W_ped_MJ <= 0.0 or f_elm_fraction <= 0.0:
        return 0.0
    return P_SOL_MW / (f_elm_fraction * W_ped_MJ)


class ELMCycler:
    """
    Tracks pedestal state and triggers ELM events.
    """

    def __init__(self, pb_boundary: PeelingBallooningBoundary, crash_model: ELMCrashModel):
        self.pb = pb_boundary
        self.crash_model = crash_model
        self.time = 0.0
        self.last_crash = 0.0

    def step(
        self,
        dt: float,
        alpha_edge: float,
        j_edge: float,
        s_edge: float,
        T_ped: float,
        n_ped: float,
        W_ped: float,
    ) -> ELMEvent | None:
        """Advance time and emit an ELM event when the boundary is exceeded."""
        self.time += dt

        if self.pb.is_unstable(alpha_edge, j_edge, s_edge):
            res = self.crash_model.crash(T_ped, n_ped, W_ped)

            f_elm = 1.0 / max(self.time - self.last_crash, 1e-6)
            self.last_crash = self.time

            return ELMEvent(
                time=self.time, delta_W_MJ=res.delta_W_MJ, f_elm_Hz=f_elm, crash_type="Type I"
            )

        return None
