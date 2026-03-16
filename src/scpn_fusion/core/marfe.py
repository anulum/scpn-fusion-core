# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MARFE Radiation Front Stability
from __future__ import annotations

import math

import numpy as np

from scpn_fusion.core.impurity_transport import CoolingCurve


class RadiationCondensation:
    def __init__(self, impurity: str, ne_20: float, f_imp: float):
        self.impurity = impurity
        self.ne_20 = ne_20
        self.f_imp = f_imp
        self.curve = CoolingCurve(impurity)

    def _dL_dT(self, Te_eV: float) -> float:
        dT = 0.01 * Te_eV
        L_plus = self.curve.L_z(np.array([Te_eV + dT]))[0]
        L_minus = self.curve.L_z(np.array([Te_eV - dT]))[0]
        return float((L_plus - L_minus) / (2.0 * dT))

    def growth_rate(self, Te_eV: float, k_par: float, kappa_par: float) -> float:
        """
        gamma = -(kappa_par * k_par^2 + n^2 * dL/dT) / (n * c_v)
        """
        ne = self.ne_20 * 1e20
        n_imp = ne * self.f_imp

        dL_dT = self._dL_dT(Te_eV)

        # c_v = 3/2 (1 + f_imp) roughly
        c_v = 1.5 * (1.0 + self.f_imp) * 1.602e-19  # J/eV

        # n^2 L assumes L is defined per electron per impurity
        rad_term = ne * n_imp * dL_dT
        cond_term = kappa_par * k_par**2

        gamma = -(cond_term + rad_term) / (ne * c_v)
        return float(gamma)

    def is_unstable(self, Te_eV: float, k_par: float, kappa_par: float) -> bool:
        return self.growth_rate(Te_eV, k_par, kappa_par) > 0.0

    def critical_density(self, Te_eV: float, k_par: float, kappa_par: float) -> float:
        """
        n_crit where gamma = 0.
        n^2 f_imp dL/dT = -kappa_par k_par^2
        """
        dL_dT = self._dL_dT(Te_eV)
        if dL_dT >= 0.0:
            return float("inf")  # Stable at all densities

        n_crit_sq = -(kappa_par * k_par**2) / (self.f_imp * dL_dT)
        return float(np.sqrt(n_crit_sq) / 1e20)


class MARFEFrontModel:
    def __init__(self, L_par: float, kappa_par: float, q_perp: float, impurity: str, f_imp: float):
        self.L_par = L_par
        self.kappa_par = kappa_par
        self.q_perp = q_perp
        self.f_imp = f_imp

        self.n_s = 50
        self.s = np.linspace(0, L_par, self.n_s)
        self.ds = self.s[1] - self.s[0]

        self.T = np.ones(self.n_s) * 100.0  # start hot
        self.curve = CoolingCurve(impurity)

    def step(self, dt: float, ne_20: float) -> np.ndarray:
        import scipy.linalg

        ne = ne_20 * 1e20
        n_imp = ne * self.f_imp

        # Rad source
        L = self.curve.L_z(self.T)
        P_rad = ne * n_imp * L

        # Heat eq: c_v n dT/dt = kappa d^2T/ds^2 - P_rad + q_perp
        c_v_n = 1.5 * ne * 1.602e-19

        alpha = self.kappa_par / c_v_n

        diag = np.zeros(self.n_s)
        upper = np.zeros(self.n_s)
        lower = np.zeros(self.n_s)
        rhs = np.zeros(self.n_s)

        diag[0] = 1.0
        rhs[0] = 100.0  # Core boundary

        diag[-1] = 1.0
        upper[-1] = -1.0
        rhs[-1] = 0.0  # Symmetry at target or X-point

        for i in range(1, self.n_s - 1):
            c_diff = alpha * dt / self.ds**2
            lower[i] = -c_diff
            diag[i] = 1.0 + 2.0 * c_diff
            upper[i] = -c_diff

            rhs[i] = self.T[i] + dt / c_v_n * (self.q_perp - P_rad[i])

        ab = np.zeros((3, self.n_s))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]

        self.T = scipy.linalg.solve_banded((1, 1), ab, rhs)
        self.T = np.maximum(self.T, 1.0)
        return self.T

    def equilibrium(self, ne_20: float) -> np.ndarray:
        for _ in range(1000):
            self.step(1e-4, ne_20)
        return self.T

    def is_marfe(self) -> bool:
        # A MARFE is a localized cold spot.
        # If the minimum T is < 20 eV and there is a steep gradient, it's collapsed.
        min_T = np.min(self.T)
        max_T = np.max(self.T)
        return bool(min_T < 20.0 and max_T > 50.0)


class DensityLimitPredictor:
    @staticmethod
    def greenwald_limit(Ip_MA: float, a: float) -> float:
        """n_GW in 10^20 m^-3."""
        if a <= 0.0:
            return float("inf")
        return float(Ip_MA / (math.pi * a**2))

    @staticmethod
    def marfe_limit(Ip_MA: float, a: float, P_SOL_MW: float, impurity: str, f_imp: float) -> float:
        """Heuristic scaling mapping P_SOL and f_imp to a density limit."""
        # Typically n_crit ~ sqrt(P_SOL / f_imp)
        # We tie it to Greenwald scaling
        n_gw = DensityLimitPredictor.greenwald_limit(Ip_MA, a)

        # Base factor for clean plasma
        factor = math.sqrt(max(P_SOL_MW, 1.0)) / (10.0 * math.sqrt(max(f_imp, 1e-5)))
        return float(n_gw * factor)


class MARFEStabilityDiagram:
    def __init__(self, R0: float, a: float, q95: float, impurity: str):
        self.R0 = R0
        self.a = a
        self.q95 = q95
        self.impurity = impurity

    def scan_density_power(self, ne_range: np.ndarray, P_SOL_range: np.ndarray) -> np.ndarray:
        result = np.zeros((len(ne_range), len(P_SOL_range)))

        # Simple heuristic limit:
        # If n > n_marfe_crit(P) -> unstable (-1)
        for i, ne in enumerate(ne_range):
            for j, P in enumerate(P_SOL_range):
                # mock Ip=15 MA
                n_crit = DensityLimitPredictor.marfe_limit(15.0, self.a, P, self.impurity, 1e-4)
                if ne > n_crit:
                    result[i, j] = -1
                else:
                    result[i, j] = 1

        return result
