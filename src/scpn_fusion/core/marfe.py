# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MARFE Radiation Front Stability
"""MARFE radiation-condensation, front evolution, and density-limit helpers."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.impurity_transport import CoolingCurve

FloatArray: TypeAlias = NDArray[np.float64]


class RadiationCondensation:
    """Local radiation-condensation growth model for impurity-cooled plasma."""

    def __init__(self, impurity: str, ne_20: float, f_imp: float):
        """Initialize impurity cooling curve and density/fraction parameters."""
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
        """Return whether the condensation growth rate is positive."""
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
    """One-dimensional parallel heat-balance model for a MARFE front."""

    def __init__(self, L_par: float, kappa_par: float, q_perp: float, impurity: str, f_imp: float):
        """Initialize parallel grid, heat conduction, heating, and impurity state."""
        self.L_par = L_par
        self.kappa_par = kappa_par
        self.q_perp = q_perp
        self.f_imp = f_imp

        self.n_s = 50
        self.s = np.linspace(0, L_par, self.n_s)
        self.ds = self.s[1] - self.s[0]

        self.T = np.ones(self.n_s) * 100.0  # start hot
        self.curve = CoolingCurve(impurity)

    def step(self, dt: float, ne_20: float) -> FloatArray:
        """Advance the parallel temperature profile by one implicit step."""
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

    def equilibrium(self, ne_20: float) -> FloatArray:
        """Relax the front model toward a steady temperature profile."""
        for _ in range(1000):
            self.step(1e-4, ne_20)
        return self.T

    def is_marfe(self) -> bool:
        """Return whether the current profile contains a collapsed cold front."""
        # A MARFE is a localized cold spot.
        # If the minimum T is < 20 eV and there is a steep gradient, it's collapsed.
        min_T = np.min(self.T)
        max_T = np.max(self.T)
        return bool(min_T < 20.0 and max_T > 50.0)


class DensityLimitPredictor:
    """Greenwald and MARFE density-limit scaling helpers."""

    @staticmethod
    def greenwald_limit(Ip_MA: float, a: float) -> float:
        """n_GW in 10^20 m^-3."""
        if a <= 0.0:
            return float("inf")
        return float(Ip_MA / (math.pi * a**2))

    @staticmethod
    def marfe_limit(
        Ip_MA: float,
        a: float,
        P_SOL_MW: float,
        impurity: str,
        f_imp: float,
        *,
        Te_eV: float | None = None,
        k_par: float | None = None,
        kappa_par: float | None = None,
    ) -> float:
        """Return MARFE density limit from condensation physics when local data exist."""
        if Te_eV is not None or k_par is not None or kappa_par is not None:
            if Te_eV is None or k_par is None or kappa_par is None:
                raise ValueError("Te_eV, k_par, and kappa_par must be supplied together")
            rc = RadiationCondensation(impurity, ne_20=1.0, f_imp=f_imp)
            return rc.critical_density(Te_eV=Te_eV, k_par=k_par, kappa_par=kappa_par)

        # Typically n_crit ~ sqrt(P_SOL / f_imp)
        # We tie it to Greenwald scaling
        n_gw = DensityLimitPredictor.greenwald_limit(Ip_MA, a)

        # Base factor for clean plasma
        factor = math.sqrt(max(P_SOL_MW, 1.0)) / (10.0 * math.sqrt(max(f_imp, 1e-5)))
        return float(n_gw * factor)


class MARFEStabilityDiagram:
    """Density-power stability map for MARFE onset."""

    def __init__(
        self,
        R0: float,
        a: float,
        q95: float,
        impurity: str,
        Ip_MA: float = 15.0,
        f_imp: float = 1e-4,
    ):
        """Initialize tokamak geometry, current, q95, and impurity parameters."""
        if not np.isfinite(R0) or float(R0) <= 0.0:
            raise ValueError("R0 must be finite and > 0.")
        if not np.isfinite(a) or float(a) <= 0.0:
            raise ValueError("a must be finite and > 0.")
        if not np.isfinite(q95) or float(q95) <= 0.0:
            raise ValueError("q95 must be finite and > 0.")
        if not np.isfinite(Ip_MA) or float(Ip_MA) <= 0.0:
            raise ValueError("Ip_MA must be finite and > 0.")
        if not np.isfinite(f_imp) or float(f_imp) <= 0.0:
            raise ValueError("f_imp must be finite and > 0.")
        self.R0 = R0
        self.a = a
        self.q95 = q95
        self.impurity = impurity
        self.Ip_MA = float(Ip_MA)
        self.f_imp = float(f_imp)

    def scan_density_power(self, ne_range: FloatArray, P_SOL_range: FloatArray) -> FloatArray:
        """Classify density-power grid points as MARFE-stable or unstable."""
        ne_arr = np.asarray(ne_range, dtype=float)
        psol_arr = np.asarray(P_SOL_range, dtype=float)
        if ne_arr.ndim != 1 or psol_arr.ndim != 1:
            raise ValueError("ne_range and P_SOL_range must be one-dimensional arrays.")
        if ne_arr.size == 0 or psol_arr.size == 0:
            raise ValueError("ne_range and P_SOL_range must not be empty.")
        if np.any(~np.isfinite(ne_arr)) or np.any(~np.isfinite(psol_arr)):
            raise ValueError("ne_range and P_SOL_range must be finite.")
        if np.any(ne_arr < 0.0) or np.any(psol_arr <= 0.0):
            raise ValueError("ne_range must be >= 0 and P_SOL_range must be > 0.")

        result = np.zeros((len(ne_range), len(P_SOL_range)))

        # Simple heuristic limit:
        # If n > n_marfe_crit(P) -> unstable (-1)
        q95_scale = float(np.sqrt(np.clip(self.q95 / 3.0, 0.5, 2.0)))
        for i, ne in enumerate(ne_arr):
            for j, P in enumerate(psol_arr):
                n_crit = (
                    DensityLimitPredictor.marfe_limit(
                        self.Ip_MA, self.a, float(P), self.impurity, self.f_imp
                    )
                    * q95_scale
                )
                if ne > n_crit:
                    result[i, j] = -1
                else:
                    result[i, j] = 1

        return result
