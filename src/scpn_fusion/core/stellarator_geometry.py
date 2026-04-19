# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stellarator Geometry
"""
Stellarator flux-surface geometry in Boozer coordinates, neoclassical
transport, and ISS04 confinement scaling.

Provides a W7-X preset and general stellarator configuration for
equilibrium and transport calculations without assuming axisymmetry.

References
----------
- Boozer, A. H., Phys. Fluids 24 (1981) 1999.
- Yamada et al., Nucl. Fusion 45 (2005) 1684.  (ISS04 scaling)
- Nemov et al., Phys. Plasmas 6 (1999) 4622.  (effective ripple)
- Beidler et al., Nucl. Fusion 51 (2011) 076001. (W7-X neoclassical)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core._validators import require_positive_float


@dataclass
class StellaratorConfig:
    """Stellarator device and magnetic configuration parameters.

    Parameters
    ----------
    N_fp : int
        Number of toroidal field periods.
    R0 : float
        Major radius [m].
    a : float
        Average minor radius [m].
    B0 : float
        On-axis magnetic field [T].
    iota_0 : float
        Rotational transform at magnetic axis (s=0).
    iota_a : float
        Rotational transform at plasma edge (s=1).
    mirror_ratio : float
        Helical mirror ratio epsilon_h = delta_B / B0.
    helical_excursion : float
        Helical axis excursion amplitude [m].
    """

    N_fp: int = 5
    R0: float = 5.5
    a: float = 0.53
    B0: float = 2.5
    iota_0: float = 0.87
    iota_a: float = 1.0
    mirror_ratio: float = 0.07
    helical_excursion: float = 0.05
    name: str = "custom"


def w7x_config() -> StellaratorConfig:
    """Wendelstein 7-X standard configuration.

    Klinger et al., Nucl. Fusion 59 (2019) 112004.
    """
    return StellaratorConfig(
        N_fp=5,
        R0=5.5,
        a=0.53,
        B0=2.5,
        iota_0=0.87,
        iota_a=1.0,
        mirror_ratio=0.07,
        helical_excursion=0.05,
        name="W7-X",
    )


def iota_profile(
    config: StellaratorConfig, s: float | NDArray[np.float64]
) -> float | NDArray[np.float64]:
    """Rotational transform iota(s) via linear interpolation.

    Stellarators use iota = 1/q directly.

    Parameters
    ----------
    config : StellaratorConfig
    s : float or array
        Normalised toroidal flux label, s in [0, 1].
    """
    return config.iota_0 + (config.iota_a - config.iota_0) * np.asarray(s)


def stellarator_flux_surface(
    config: StellaratorConfig,
    s: float,
    n_theta: int = 64,
    n_phi: int = 64,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute a single flux surface in Boozer coordinates.

    Parameters
    ----------
    config : StellaratorConfig
    s : float
        Normalised toroidal flux, s in (0, 1].
    n_theta, n_phi : int
        Poloidal and toroidal grid resolution.

    Returns
    -------
    R : ndarray, shape (n_theta, n_phi)
        Major radius [m].
    Z : ndarray, shape (n_theta, n_phi)
        Vertical position [m].
    B : ndarray, shape (n_theta, n_phi)
        Magnetic field magnitude [T].
    """
    s = require_positive_float("s", s)
    r = config.a * np.sqrt(s)
    iota = float(iota_profile(config, s))

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    # Boozer representation: helical excursion modulates axis position
    delta_R = config.helical_excursion * np.cos(config.N_fp * PH)
    R = config.R0 + r * np.cos(TH) + delta_R
    Z = r * np.sin(TH) + config.helical_excursion * np.sin(config.N_fp * PH)

    # |B| with toroidal and helical modulation
    epsilon_t = r / config.R0
    epsilon_h = config.mirror_ratio * np.sqrt(s)
    B = config.B0 * (
        1.0 - epsilon_t * np.cos(TH) - epsilon_h * np.cos(config.N_fp * PH - iota * TH)
    )

    return R, Z, B


def effective_ripple(config: StellaratorConfig, s: float) -> float:
    """Effective helical ripple epsilon_eff for neoclassical transport.

    Nemov et al., Phys. Plasmas 6 (1999) 4622, Eq. 30.
    Simplified analytic proxy: epsilon_eff ~ epsilon_h^(3/2) / sqrt(N_fp).
    Full computation requires field-line tracing (DKES/NEO-2).

    Parameters
    ----------
    config : StellaratorConfig
    s : float
        Normalised toroidal flux, s in (0, 1].

    Returns
    -------
    float
        Effective ripple epsilon_eff (dimensionless, 0 < eps_eff < 1).
    """
    s = require_positive_float("s", s)
    epsilon_h = config.mirror_ratio * np.sqrt(s)
    eps_eff = epsilon_h**1.5 / np.sqrt(config.N_fp)
    return float(np.clip(eps_eff, 0.0, 1.0))


def iss04_scaling(
    config: StellaratorConfig,
    n_e: float,
    P_heat: float,
) -> float:
    """ISS04 energy confinement scaling law for stellarators.

    Yamada et al., Nucl. Fusion 45 (2005) 1684, Eq. 4:
        tau_E = 0.134 * a^2.28 * R^0.64 * P^-0.61 * n_e19^0.54
                * B^0.84 * iota_2/3^0.41

    Parameters
    ----------
    config : StellaratorConfig
    n_e : float
        Line-averaged electron density [10^19 m^-3].
    P_heat : float
        Absorbed heating power [MW].

    Returns
    -------
    float
        Predicted energy confinement time [s].
    """
    n_e = require_positive_float("n_e", n_e)
    P_heat = require_positive_float("P_heat", P_heat)

    # iota evaluated at s = 2/3 (standard ISS04 reference radius)
    ISS04_S_REF = 2.0 / 3.0  # Yamada et al. 2005, Eq. 4
    iota_ref = float(iota_profile(config, ISS04_S_REF))

    tau = (
        0.134
        * config.a**2.28
        * config.R0**0.64
        * P_heat**-0.61
        * n_e**0.54
        * config.B0**0.84
        * iota_ref**0.41
    )
    return float(tau)


def stellarator_neoclassical_chi(
    config: StellaratorConfig,
    s: float,
    T_keV: float,
    n_e19: float,
) -> float:
    """Neoclassical thermal diffusivity in the 1/nu regime.

    Beidler et al., Nucl. Fusion 51 (2011) 076001.
    chi_neo ~ epsilon_eff^(3/2) * v_th^2 / (nu * R * N_fp)

    where v_th = sqrt(T / m_i) and nu = n_e * ln_Lambda * e^4 / (m_i^2 * v_th^3).

    Parameters
    ----------
    config : StellaratorConfig
    s : float
        Normalised toroidal flux, s in (0, 1].
    T_keV : float
        Ion temperature [keV].
    n_e19 : float
        Electron density [10^19 m^-3].

    Returns
    -------
    float
        Neoclassical chi [m^2/s].
    """
    s = require_positive_float("s", s)
    T_keV = require_positive_float("T_keV", T_keV)
    n_e19 = require_positive_float("n_e19", n_e19)

    eps_eff = effective_ripple(config, s)

    M_I = 3.344e-27  # deuteron mass [kg]
    E_KEV_TO_J = 1.602e-16  # keV → J
    v_th = np.sqrt(T_keV * E_KEV_TO_J / M_I)

    COULOMB_LOG = 17.0  # Wesson, Tokamaks 4th ed., Ch. 14
    E_CHARGE = 1.602e-19  # [C]
    EPS_0 = 8.854e-12  # [F/m]
    n_e_m3 = n_e19 * 1e19

    # Braginskii collision frequency: nu_ii ~ n * e^4 * ln_Lambda / (eps_0^2 * m_i^2 * v_th^3)
    nu_ii = n_e_m3 * E_CHARGE**4 * COULOMB_LOG / (4.0 * np.pi * EPS_0**2 * M_I**2 * v_th**3)

    # 1/nu regime: chi = epsilon_eff^(3/2) * v_th^2 / (nu * R * N_fp)
    chi = eps_eff**1.5 * v_th**2 / (nu_ii * config.R0 * config.N_fp)

    return float(chi)
