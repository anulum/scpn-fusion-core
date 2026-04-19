# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Species Impurity Transport
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ImpuritySpecies:
    element: str
    Z_nucleus: int
    mass_amu: float
    source_rate: float = 0.0


class CoolingCurve:
    """
    Parametric cooling rate L_Z(Te) [W m^3].
    """

    def __init__(self, element: str):
        self.element = element

    def L_z(self, Te_eV: np.ndarray) -> np.ndarray:
        log_Te = np.log(Te_eV)
        if self.element == "W":
            # Putterich et al. 2010 fit — peaks near 1500 eV and 50 eV
            L = 1e-31 * np.exp(-(((log_Te - np.log(1500.0)) / 1.5) ** 2))
            L += 3e-33 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2))
            return np.asarray(L)
        if self.element == "C":
            return np.asarray(1e-32 * np.exp(-(((log_Te - np.log(10.0)) / 0.5) ** 2)))
        if self.element == "Ar":
            return np.asarray(1e-32 * np.exp(-(((log_Te - np.log(200.0)) / 1.0) ** 2)))
        if self.element == "Ne":
            return np.asarray(1e-32 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2)))
        return np.zeros_like(Te_eV)


def neoclassical_impurity_pinch(
    Z: int,
    ne: np.ndarray,
    Te_eV: np.ndarray,
    Ti_eV: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    R0: float,
    a: float,
    epsilon: np.ndarray,
) -> np.ndarray:
    """
    V_neo [m/s] (negative = inward).
    Hirshman & Sigmar, Nucl. Fusion 21, 1079 (1981).
    V_neo = -Z D_neo [Z ∇n/n + (Z/2 - H_Z) ∇T/T]
    """
    drho = rho[1] - rho[0] if len(rho) > 1 else 0.1

    grad_ne_over_n = np.gradient(ne, drho * a) / np.maximum(ne, 1e-6)
    grad_Ti_over_T = np.gradient(Ti_eV, drho * a) / np.maximum(Ti_eV, 1e-6)

    D_NEO = 0.1  # m²/s, banana regime nominal scale
    D_neo = D_NEO * np.ones_like(rho)

    H_Z = 0.5  # screening factor, banana regime trace impurities

    V_neo = -D_neo * (Z * grad_ne_over_n + (Z / 2.0 - H_Z) * grad_Ti_over_T)
    return np.asarray(V_neo)


def total_radiated_power(
    ne: np.ndarray,
    n_impurity: dict[str, np.ndarray],
    Te_eV: np.ndarray,
    rho: np.ndarray,
    R0: float,
    a: float,
) -> float:
    """
    P_rad in MW.
    """
    p_rad_density = np.zeros_like(rho)

    for element, n_Z in n_impurity.items():
        curve = CoolingCurve(element)
        L = curve.L_z(Te_eV)
        # p_rad = n_e * n_z * L_z
        p_rad_density += ne * n_Z * L

    # Integrate over volume: dV = 4 pi^2 R0 a^2 rho drho
    vol_element = 4.0 * np.pi**2 * R0 * a**2 * rho
    _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    P_rad_W = _trapz(p_rad_density * vol_element, rho)

    return float(P_rad_W / 1e6)


def tungsten_accumulation_diagnostic(n_W: np.ndarray, ne: np.ndarray) -> dict[str, Any]:
    c_W_core = float(n_W[0] / max(ne[0], 1e-6))
    c_W_edge = float(n_W[-1] / max(ne[-1], 1e-6))

    peaking_factor = c_W_core / max(c_W_edge, 1e-12)

    if c_W_core < 1e-5:
        danger = "safe"
    elif c_W_core < 5e-5:
        danger = "warning"
    else:
        danger = "critical"

    return {
        "c_W_core": c_W_core,
        "c_W_edge": c_W_edge,
        "peaking_factor": peaking_factor,
        "danger_level": danger,
    }


class ImpurityTransportSolver:
    def __init__(self, rho: np.ndarray, R0: float, a: float, species: list[ImpuritySpecies]):
        self.rho = rho
        self.R0 = R0
        self.a = a
        self.species = species

        self.nr = len(rho)
        self.drho = rho[1] - rho[0]

        self.n_z = {s.element: np.zeros(self.nr) for s in species}

    def step(
        self,
        dt: float,
        ne: np.ndarray,
        Te_eV: np.ndarray,
        Ti_eV: np.ndarray,
        D_anom: float,
        V_pinch: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        1D Transport advance for each species.
        Uses simple upwind/centered differences.
        """
        import scipy.linalg

        dr = self.drho * self.a

        for s in self.species:
            n = self.n_z[s.element]
            V = V_pinch.get(s.element, np.zeros(self.nr))

            # D_total = D_anom + D_neo
            D = D_anom * np.ones(self.nr)

            # Implicit advance
            diag = np.zeros(self.nr)
            upper = np.zeros(self.nr)
            lower = np.zeros(self.nr)
            rhs = np.zeros(self.nr)

            # Boundary conditions
            diag[0] = 1.0
            upper[0] = -1.0
            rhs[0] = 0.0  # dn/dr = 0 at axis

            diag[-1] = 1.0
            rhs[-1] = s.source_rate * dt / dr  # Very simplified edge source mapping

            # Interior
            for i in range(1, self.nr - 1):
                r_val = self.rho[i] * self.a

                # Diffusion term
                coeff_D_plus = D[i] / dr**2 + D[i] / (2.0 * r_val * dr)
                coeff_D_minus = D[i] / dr**2 - D[i] / (2.0 * r_val * dr)
                coeff_D_0 = -2.0 * D[i] / dr**2

                # Convection term (upwind)
                if V[i] > 0:
                    coeff_V_0 = -V[i] / dr - V[i] / r_val
                    coeff_V_minus = V[i] / dr
                    coeff_V_plus = 0.0
                else:
                    coeff_V_0 = V[i] / dr - V[i] / r_val
                    coeff_V_plus = -V[i] / dr
                    coeff_V_minus = 0.0

                lower[i] = -dt * (coeff_D_minus + coeff_V_minus)
                diag[i] = 1.0 - dt * (coeff_D_0 + coeff_V_0)
                upper[i] = -dt * (coeff_D_plus + coeff_V_plus)

                rhs[i] = n[i]

            # Solve
            ab = np.zeros((3, self.nr))
            ab[0, 1:] = upper[:-1]
            ab[1, :] = diag
            ab[2, :-1] = lower[1:]

            n_new = scipy.linalg.solve_banded((1, 1), ab, rhs)

            # Replace
            self.n_z[s.element] = np.maximum(n_new, 0.0)

        return self.n_z
