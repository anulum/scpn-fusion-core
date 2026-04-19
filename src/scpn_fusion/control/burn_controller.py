# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Burn Control and Alpha Heating Feedback
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_fusion.core.uncertainty import bosch_hale_reactivity


class AlphaHeating:
    def __init__(self, R0: float, a: float, kappa: float = 1.0):
        self.R0 = R0
        self.a = a
        self.kappa = kappa
        self.E_alpha_J = 3.52 * 1e6 * 1.602e-19

    def power_density(
        self, ne_20: np.ndarray, Te_keV: np.ndarray, Ti_keV: np.ndarray
    ) -> np.ndarray:
        """
        p_alpha [MW/m^3] for 50:50 DT mixture.
        """
        # ne in m^-3
        ne_m3 = ne_20 * 1e20
        # nD = nT = ne / 2
        nD = ne_m3 / 2.0
        nT = ne_m3 / 2.0

        sigv = bosch_hale_reactivity(Ti_keV)

        # p_alpha_W = n_D n_T <sigma v> E_alpha
        p_alpha_W = nD * nT * sigv * self.E_alpha_J
        return p_alpha_W / 1e6

    def power(
        self, ne_20: np.ndarray, Te_keV: np.ndarray, Ti_keV: np.ndarray, rho: np.ndarray
    ) -> float:
        """
        P_alpha [MW] integrated over volume.
        """
        p_dens = self.power_density(ne_20, Te_keV, Ti_keV)

        # dV = 4 pi^2 R0 a^2 kappa rho drho
        dV = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.kappa * rho

        _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        P_tot = _trapz(p_dens * dV, rho)
        return float(P_tot)

    def Q(self, P_alpha_MW: float, P_aux_MW: float) -> float:
        """Fusion gain."""
        if P_aux_MW <= 0.0:
            return float("inf") if P_alpha_MW > 0 else 0.0
        return 5.0 * P_alpha_MW / P_aux_MW


class BurnStabilityAnalysis:
    def __init__(self, alpha_heating: AlphaHeating):
        self.alpha_heating = alpha_heating

    def reactivity_exponent(self, Ti_keV: float) -> float:
        """
        d(ln <sigma v>) / d(ln T)
        Evaluated via finite difference.
        """
        if Ti_keV <= 0.1:
            return 10.0

        dT = 0.01 * Ti_keV
        sv_arr_plus = np.asarray(bosch_hale_reactivity(np.array([Ti_keV + dT])))
        sv_arr_minus = np.asarray(bosch_hale_reactivity(np.array([Ti_keV - dT])))
        sv_plus = sv_arr_plus[0]
        sv_minus = sv_arr_minus[0]

        if sv_minus <= 0 or sv_plus <= 0:
            return 10.0

        d_ln_sv = np.log(sv_plus) - np.log(sv_minus)
        d_ln_T = np.log(Ti_keV + dT) - np.log(Ti_keV - dT)

        return float(d_ln_sv / d_ln_T)

    def is_thermally_stable(self, Ti_keV: float) -> bool:
        """True if exponent < 2"""
        return self.reactivity_exponent(Ti_keV) < 2.0

    def stability_boundary_keV(self) -> float:
        """Find T where exponent = 2."""
        # Bisection between 5 and 30 keV
        T_low = 5.0
        T_high = 30.0

        for _ in range(20):
            T_mid = (T_low + T_high) / 2.0
            if self.reactivity_exponent(T_mid) > 2.0:
                T_low = T_mid
            else:
                T_high = T_mid

        return T_high


class BurnController:
    def __init__(
        self, Q_target: float = 10.0, T_target_keV: float = 20.0, P_aux_max_MW: float = 73.0
    ):
        self.Q_target = Q_target
        self.T_target = T_target_keV
        self.P_aux_max = P_aux_max_MW

        self.integral_T = 0.0

        self.K_T_p = -5.0  # MW/keV proportional gain
        self.K_T_i = -1.0  # MW/(keV·s) integral gain

        self.last_P_aux = P_aux_max_MW / 2.0

    def step(self, Q_meas: float, T_meas_keV: float, P_alpha_MW: float, dt: float) -> float:
        # Emergency cooling
        if T_meas_keV > 30.0:
            self.last_P_aux = 0.0
            return 0.0

        e_T = T_meas_keV - self.T_target
        self.integral_T += e_T * dt

        # P_aux = P_ff + P_fb
        # If we just want to stabilize T:
        P_fb = self.K_T_p * e_T + self.K_T_i * self.integral_T

        # Base power to sustain target
        # Simplified: rely on integral to find it
        P_cmd = self.P_aux_max / 2.0 + P_fb

        P_cmd = np.clip(P_cmd, 0.0, self.P_aux_max)
        self.last_P_aux = P_cmd

        return float(P_cmd)


@dataclass
class BurnPoint:
    Te_keV: float
    P_alpha_MW: float
    P_loss_MW: float
    Q: float
    stable: bool


class SubignitedBurnPoint:
    def __init__(self, alpha_heating: AlphaHeating):
        self.alpha = alpha_heating
        self.stability = BurnStabilityAnalysis(alpha_heating)

    def find_operating_point(
        self, ne_20: float, P_aux_MW: float, tau_E_s: float
    ) -> list[BurnPoint]:
        """
        Solve P_alpha(T) + P_aux = P_loss(T)
        P_loss = 3 n T V / tau_E
        We just scan T to find intersections.
        """
        T_scan = np.linspace(1.0, 40.0, 400)
        points = []

        V = 2.0 * np.pi**2 * self.alpha.R0 * self.alpha.a**2 * self.alpha.kappa
        e_charge = 1.602e-19

        # Assume flat profiles for 0D model
        P_alphas = np.zeros_like(T_scan)
        P_losses = np.zeros_like(T_scan)

        for i, T in enumerate(T_scan):
            ne_arr = np.array([ne_20])
            T_arr = np.array([T])
            # For 0D, power is just density * volume
            p_dens = self.alpha.power_density(ne_arr, T_arr, T_arr)[0]
            P_alphas[i] = p_dens * V

            # P_loss = 3 n T V / tau_E
            W_J = 3.0 * (ne_20 * 1e20) * (T * 1e3 * e_charge) * V
            P_losses[i] = (W_J / tau_E_s) / 1e6

        P_net = P_alphas + P_aux_MW - P_losses

        # Find zero crossings
        crossings = np.where(np.diff(np.sign(P_net)))[0]

        for idx in crossings:
            T_cross = T_scan[idx]
            P_a = P_alphas[idx]
            P_l = P_losses[idx]
            Q = self.alpha.Q(P_a, P_aux_MW)

            # Stable if dP_net/dT < 0 (loss grows faster than source)
            dP_net_dT = P_net[idx + 1] - P_net[idx]
            stable = dP_net_dT < 0

            points.append(
                BurnPoint(
                    Te_keV=float(T_cross),
                    P_alpha_MW=float(P_a),
                    P_loss_MW=float(P_l),
                    Q=float(Q),
                    stable=bool(stable),
                )
            )

        return points
