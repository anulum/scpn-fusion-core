# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Species Impurity Transport
"""Multi-species impurity transport, cooling, radiation, and accumulation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ImpuritySpecies:
    """Impurity species metadata and edge source parameters."""

    element: str
    Z_nucleus: int
    mass_amu: float
    source_rate: float = 0.0
    source_decay_width_rho: float = 0.05

    def __post_init__(self) -> None:
        if self.Z_nucleus < 1:
            raise ValueError("Z_nucleus must be positive")
        if not np.isfinite(self.mass_amu) or self.mass_amu <= 0.0:
            raise ValueError("mass_amu must be finite and positive")
        if not np.isfinite(self.source_rate) or self.source_rate < 0.0:
            raise ValueError("source_rate must be finite and non-negative")
        if not np.isfinite(self.source_decay_width_rho) or self.source_decay_width_rho <= 0.0:
            raise ValueError("source_decay_width_rho must be finite and positive")


class CoolingCurve:
    """
    Parametric cooling rate L_Z(Te) [W m^3].
    """

    def __init__(self, element: str):
        self.element = element

    def L_z(self, Te_eV: np.ndarray) -> np.ndarray:
        """Evaluate the element cooling curve for electron temperatures."""
        Te = np.asarray(Te_eV, dtype=float)
        valid = np.isfinite(Te) & (Te > 0.0)
        if not np.any(valid):
            return np.zeros_like(Te, dtype=float)

        log_Te = np.zeros_like(Te, dtype=float)
        log_Te[valid] = np.log(Te[valid])
        if self.element == "W":
            # Putterich et al. 2010 fit — peaks near 1500 eV and 50 eV
            L = 1e-31 * np.exp(-(((log_Te - np.log(1500.0)) / 1.5) ** 2))
            L += 3e-33 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        if self.element == "C":
            L = 1e-32 * np.exp(-(((log_Te - np.log(10.0)) / 0.5) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        if self.element == "Ar":
            L = 1e-32 * np.exp(-(((log_Te - np.log(200.0)) / 1.0) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        if self.element == "Ne":
            L = 1e-32 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        return np.zeros_like(Te)


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
    V_neo = -D_neo [Z/L_n + (Z/2 - H_Z)/L_Ti]
    with inverse scale lengths 1/L_x = -d ln(x)/dr.
    """
    if Z < 1:
        raise ValueError("Z must be positive")
    rho_arr = np.asarray(rho, dtype=float)
    ne_arr = np.asarray(ne, dtype=float)
    ti_arr = np.asarray(Ti_eV, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    eps_arr = np.asarray(epsilon, dtype=float)
    arrays = (rho_arr, ne_arr, ti_arr, q_arr, eps_arr)
    if any(arr.shape != rho_arr.shape for arr in arrays):
        raise ValueError("rho, ne, Ti_eV, q, and epsilon must have matching shapes")
    if rho_arr.size < 3:
        raise ValueError("rho must contain at least three points")
    if not np.all(np.isfinite(rho_arr)) or not np.all(np.diff(rho_arr) > 0.0):
        raise ValueError("rho must be finite and strictly increasing")
    if not np.all(np.isfinite(ne_arr)) or np.any(ne_arr <= 0.0):
        raise ValueError("ne must be finite and positive")
    if not np.all(np.isfinite(ti_arr)) or np.any(ti_arr <= 0.0):
        raise ValueError("Ti_eV must be finite and positive")
    if not np.all(np.isfinite(q_arr)) or np.any(q_arr <= 0.0):
        raise ValueError("q must be finite and positive")
    if not np.all(np.isfinite(eps_arr)) or np.any(eps_arr < 0.0):
        raise ValueError("epsilon must be finite and non-negative")
    if not np.isfinite(R0) or R0 <= 0.0:
        raise ValueError("R0 must be finite and positive")
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("a must be finite and positive")

    dr = (rho_arr[1] - rho_arr[0]) * a
    inv_Ln = -np.gradient(np.log(ne_arr), dr)
    inv_LTi = -np.gradient(np.log(ti_arr), dr)

    D_NEO = 0.1  # m²/s, banana-regime nominal scale
    D_neo = D_NEO * q_arr**2 / np.sqrt(np.maximum(eps_arr, 0.05))

    H_Z = 0.5  # screening factor, banana regime trace impurities

    V_neo = -D_neo * (Z * inv_Ln + (Z / 2.0 - H_Z) * inv_LTi)
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
    """Return core/edge tungsten concentration and accumulation danger level."""
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
    """Implicit radial impurity transport solver for multiple species."""

    def __init__(self, rho: np.ndarray, R0: float, a: float, species: list[ImpuritySpecies]):
        """Initialize geometry, species inventory, and radial impurity state."""
        self.rho = np.asarray(rho, dtype=float)
        self.R0 = R0
        self.a = a
        self.species = species

        self.nr = len(self.rho)
        if self.nr < 3:
            raise ValueError("rho must contain at least three radial points")
        if not np.all(np.isfinite(self.rho)):
            raise ValueError("rho must contain only finite values")
        if not np.all(np.diff(self.rho) > 0.0):
            raise ValueError("rho must be strictly increasing")
        if not np.isclose(self.rho[0], 0.0) or not np.isclose(self.rho[-1], 1.0):
            raise ValueError("rho must span the normalised interval [0, 1]")
        if not np.isfinite(R0) or R0 <= 0.0:
            raise ValueError("R0 must be finite and positive")
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("a must be finite and positive")

        drho = np.diff(self.rho)
        if not np.allclose(drho, drho[0], rtol=1e-6, atol=1e-12):
            raise ValueError("rho grid must be uniformly spaced for the banded transport solve")
        self.drho = float(drho[0])

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
            source = self._edge_source_density(s)

            # Boundary conditions
            diag[0] = 1.0
            upper[0] = -1.0
            rhs[0] = 0.0  # dn/dr = 0 at axis

            diag[-1] = 1.0
            rhs[-1] = n[-1] + dt * source[-1]

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

                rhs[i] = n[i] + dt * source[i]

            # Solve
            ab = np.zeros((3, self.nr))
            ab[0, 1:] = upper[:-1]
            ab[1, :] = diag
            ab[2, :-1] = lower[1:]

            n_new = scipy.linalg.solve_banded((1, 1), ab, rhs)

            # Replace
            self.n_z[s.element] = np.maximum(n_new, 0.0)

        return self.n_z

    def _edge_source_density(self, species: ImpuritySpecies) -> np.ndarray:
        """Return a volume-normalised edge source density [m^-3 s^-1]."""
        if species.source_rate == 0.0:
            return np.zeros(self.nr)

        width = max(species.source_decay_width_rho, self.drho)
        profile = np.exp(-(1.0 - self.rho) / width)
        profile[self.rho < max(0.0, 1.0 - 8.0 * width)] = 0.0

        vol_element = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.rho
        _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        profile_volume = float(_trapz(profile * vol_element, self.rho))
        if profile_volume <= 0.0 or not np.isfinite(profile_volume):
            raise ValueError("edge source profile has zero normalisation")

        edge_area = 4.0 * np.pi**2 * self.R0 * self.a
        total_particles_per_second = species.source_rate * edge_area
        return np.asarray(profile * total_particles_per_second / profile_volume)
