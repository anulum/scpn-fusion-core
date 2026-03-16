# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Gyrokinetic Species & Collision Operator
"""
Species definitions and collision operator for the linear GK solver.

Velocity-space discretisation uses (energy, lambda) coordinates with
Gauss-Legendre quadrature.  The collision operator implements a
simplified Sugama model (pitch-angle scattering + energy diffusion).

References:
  - Sugama & Watanabe, Phys. Plasmas 13 (2006) 012501
  - Abel et al., Rep. Prog. Phys. 76 (2013) 116201, §4.4
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_E_CHARGE = 1.602176634e-19  # C
_M_PROTON = 1.67262192369e-27  # kg
_M_ELECTRON = 9.1093837015e-31  # kg


@dataclass
class GKSpecies:
    """Single plasma species for the GK solver.

    Parameters
    ----------
    mass_amu : float
        Mass in atomic mass units (2.0 for deuterium).
    charge_e : float
        Charge in units of e (+1 for ions, -1 for electrons).
    temperature_keV : float
        Temperature [keV].
    density_19 : float
        Density [10^19 m^-3].
    R_L_T : float
        Normalised temperature gradient R/L_T.
    R_L_n : float
        Normalised density gradient R/L_n.
    is_adiabatic : bool
        If True, use adiabatic response instead of kinetic.
    """

    mass_amu: float
    charge_e: float
    temperature_keV: float
    density_19: float
    R_L_T: float
    R_L_n: float
    is_adiabatic: bool = False

    @property
    def mass_kg(self) -> float:
        return self.mass_amu * _M_PROTON

    @property
    def thermal_speed(self) -> float:
        """v_th = sqrt(2 T / m) [m/s]."""
        T_J = self.temperature_keV * 1e3 * _E_CHARGE
        return float(np.sqrt(2.0 * T_J / self.mass_kg))

    @property
    def larmor_radius(self) -> float:
        """rho_s = m v_th / (|q| B) — requires B to be set externally.

        Returns rho_s / B for later scaling.
        """
        return self.mass_kg * self.thermal_speed / (abs(self.charge_e) * _E_CHARGE)


def deuterium_ion(
    T_keV: float = 8.0, n_19: float = 10.0, R_L_T: float = 6.9, R_L_n: float = 2.2
) -> GKSpecies:
    return GKSpecies(
        mass_amu=2.0, charge_e=1.0, temperature_keV=T_keV, density_19=n_19, R_L_T=R_L_T, R_L_n=R_L_n
    )


def electron(
    T_keV: float = 8.0,
    n_19: float = 10.0,
    R_L_T: float = 6.9,
    R_L_n: float = 2.2,
    adiabatic: bool = True,
) -> GKSpecies:
    mass_amu = _M_ELECTRON / _M_PROTON
    return GKSpecies(
        mass_amu=mass_amu,
        charge_e=-1.0,
        temperature_keV=T_keV,
        density_19=n_19,
        R_L_T=R_L_T,
        R_L_n=R_L_n,
        is_adiabatic=adiabatic,
    )


@dataclass
class VelocityGrid:
    """Energy-lambda velocity-space grid with Gauss-Legendre weights.

    Energy E normalised to T_s.  Lambda = mu B0 / E ∈ [0, 1].
    """

    n_energy: int = 16
    n_lambda: int = 24

    def __post_init__(self) -> None:
        # Gauss-Legendre on [0, E_max] for energy (E_max ~ 6 T)
        e_nodes, e_weights = np.polynomial.legendre.leggauss(self.n_energy)
        self.E_max = 6.0
        self.energy = 0.5 * self.E_max * (e_nodes + 1.0)  # map [-1,1] → [0, E_max]
        self.energy_weights = 0.5 * self.E_max * e_weights

        # Gauss-Legendre on [0, 1] for lambda
        l_nodes, l_weights = np.polynomial.legendre.leggauss(self.n_lambda)
        self.lam = 0.5 * (l_nodes + 1.0)
        self.lambda_weights = 0.5 * l_weights

    @property
    def n_total(self) -> int:
        return self.n_energy * self.n_lambda


def bessel_j0(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """J_0(x) via scipy or polynomial approximation.

    For the GK solver, x = k_perp * rho_s * sqrt(2 * lambda * E).
    At small x (long wavelength limit), J_0 → 1.
    """
    from scipy.special import j0

    return np.asarray(j0(x), dtype=np.float64)


def collision_frequencies(
    species: GKSpecies,
    n_e_19: float,
    T_e_keV: float,
    Z_eff: float = 1.0,
    ln_lambda: float = 17.0,
) -> tuple[float, float]:
    """Compute deflection and energy-diffusion collision frequencies.

    Returns (nu_D, nu_E) normalised to v_th / R.

    Simplified Sugama model:
      nu_D = nu_ii for ions, nu_ei for electrons
      nu_E = nu_D * (m_target / m_field) correction
    """
    n_e = n_e_19 * 1e19  # m^-3
    T_e_J = T_e_keV * 1e3 * _E_CHARGE
    T_s_J = species.temperature_keV * 1e3 * _E_CHARGE

    # Braginskii ion-ion: nu_ii = 4 sqrt(pi) n Z^4 e^4 ln(Lambda) / (3 m^0.5 (2T)^1.5)
    # Simplified to order-of-magnitude for the linearised operator
    v_th = species.thermal_speed
    eps_0 = 8.8541878128e-12

    q_s = abs(species.charge_e) * _E_CHARGE
    nu_ref = (
        n_e
        * q_s**4
        * ln_lambda
        / (6.0 * np.pi**1.5 * eps_0**2 * species.mass_kg**0.5 * (2.0 * T_s_J) ** 1.5)
    )

    nu_D = float(Z_eff * nu_ref)
    nu_E = nu_D  # simplified: energy diffusion ~ pitch-angle for like-species
    return nu_D, nu_E


def pitch_angle_operator(
    n_lambda: int,
    lam: NDArray[np.float64],
    B_ratio: float = 1.0,
) -> NDArray[np.float64]:
    """Pitch-angle scattering operator matrix L_pitch on lambda grid.

    L_pitch f = (1/2) d/d(xi) (1-xi^2) df/d(xi)
    where xi = v_∥ / v = sqrt(1 - lambda * B/B0).

    Returns shape (n_lambda, n_lambda).
    """
    xi = np.sqrt(np.maximum(1.0 - lam * B_ratio, 0.0))
    d_lam = np.diff(lam, prepend=0.0, append=1.0)
    d_lam_mid = 0.5 * (d_lam[:-1] + d_lam[1:])

    # Second-order FD for (1-xi^2) d^2f/d(xi^2) ≈ lambda * d^2f/d(lambda^2)
    L = np.zeros((n_lambda, n_lambda))
    for j in range(1, n_lambda - 1):
        h = d_lam_mid[j]
        if h < 1e-30:
            continue
        coeff = lam[j] * (1.0 - lam[j] * B_ratio) / (h * h)
        L[j, j - 1] = coeff
        L[j, j] = -2.0 * coeff
        L[j, j + 1] = coeff

    # Boundary: df/d(lambda) = 0 at lambda=0 and lambda=1/B_ratio (trapped-passing boundary)
    return L
