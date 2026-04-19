# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Edge Localized Mode (ELM) Model
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        Critical edge current density limit (simplified scaling).
        In reality, depends strongly on q95 and collisionality.
        """
        # Critical j_edge ~ 1/q95
        j_crit = 1.0e6 / max(self.q95, 2.0)
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

        # Simplified PB boundary: j_norm^2 + a_norm^2 > 1
        return (j_norm**2 + a_norm**2) > 1.0

    def stability_margin(self, alpha_edge: float, j_edge: float, s_edge: float) -> float:
        """
        Distance to boundary (positive = stable).
        """
        j_crit = self.peeling_limit(j_edge)
        a_crit = self.ballooning_limit(s_edge)

        j_norm = max(0.0, j_edge / j_crit)
        a_norm = max(0.0, alpha_edge / a_crit)

        return float(1.0 - np.sqrt(j_norm**2 + a_norm**2))


@dataclass
class ELMCrashResult:
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
        # Very simplified representation of overlap across outer rational surfaces
        q_edge = q_profile[-1]

        # Approximate shear at edge
        if len(rho) > 1:
            dq_drho = (q_profile[-1] - q_profile[-2]) / (rho[-1] - rho[-2])
        else:
            dq_drho = 10.0

        shear = dq_drho * rho[-1] / q_edge

        if delta_B_r <= 0.0 or B0 <= 0.0 or self.n_toroidal == 0:
            return 0.0

        # Island width approximation (w_mn in rho space roughly)
        w_mn = 4.0 * np.sqrt(R0 * shear * delta_B_r / (self.n_toroidal * B0))

        # Distance between resonances dr_mn ~ 1 / (n q')
        dr_mn = 1.0 / (self.n_toroidal * max(dq_drho, 1e-3))

        return float(w_mn / dr_mn)

    def suppressed(self, sigma_chir: float) -> bool:
        return sigma_chir > 1.0

    def pedestal_transport_enhancement(self, sigma_chir: float) -> float:
        alpha = 2.0
        return 1.0 + alpha * max(0.0, sigma_chir - 1.0)

    def density_pump_out(self, sigma_chir: float) -> float:
        if sigma_chir > 1.0:
            return 0.2  # 20% reduction
        return 0.0


def elm_power_balance_frequency(P_SOL_MW: float, W_ped_MJ: float, f_elm_fraction: float) -> float:
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
        self.time += dt

        if self.pb.is_unstable(alpha_edge, j_edge, s_edge):
            res = self.crash_model.crash(T_ped, n_ped, W_ped)

            f_elm = 1.0 / max(self.time - self.last_crash, 1e-6)
            self.last_crash = self.time

            return ELMEvent(
                time=self.time, delta_W_MJ=res.delta_W_MJ, f_elm_Hz=f_elm, crash_type="Type I"
            )

        return None
