# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stellarator Geometry
"""Stellarator Boozer-coordinate geometry, neoclassical transport, and ISS04 scaling.

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

    def __post_init__(self) -> None:
        """Validate field-period count and the positive device dimensions."""
        if not isinstance(self.N_fp, int) or self.N_fp < 1:
            raise ValueError("N_fp must be an integer >= 1.")
        for field_name in ("R0", "a", "B0"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if self.a >= self.R0:
            raise ValueError("a must be smaller than R0.")
        for field_name in ("iota_0", "iota_a"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive iota.")
        if not np.isfinite(self.mirror_ratio) or not 0.0 <= self.mirror_ratio < 1.0:
            raise ValueError("mirror_ratio must be finite and in [0, 1).")
        if not np.isfinite(self.helical_excursion) or self.helical_excursion < 0.0:
            raise ValueError("helical_excursion must be finite and non-negative.")
        if self.helical_excursion + self.a >= self.R0:
            raise ValueError("helical_excursion plus a must stay inside R0.")


def _require_flux_label(name: str, s: float, *, include_axis: bool = False) -> float:
    s_val = float(s)
    lower_ok = s_val >= 0.0 if include_axis else s_val > 0.0
    if not np.isfinite(s_val) or not lower_ok or s_val > 1.0:
        interval = "[0, 1]" if include_axis else "(0, 1]"
        raise ValueError(f"{name} must be finite and in {interval}.")
    return s_val


def _require_grid_resolution(name: str, value: int) -> int:
    if not isinstance(value, int) or value < 8:
        raise ValueError(f"{name} must be an integer >= 8.")
    return value


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
        Stellarator device geometry and rotational-transform profile parameters.
    s : float or array
        Normalised toroidal flux label, s in [0, 1].

    """
    s_arr = np.asarray(s, dtype=float)
    if not np.all(np.isfinite(s_arr)) or np.any(s_arr < 0.0) or np.any(s_arr > 1.0):
        raise ValueError("s must be finite and in [0, 1].")
    return config.iota_0 + (config.iota_a - config.iota_0) * s_arr


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
        Stellarator device geometry and rotational-transform profile parameters.
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
    s = _require_flux_label("s", s)
    n_theta = _require_grid_resolution("n_theta", n_theta)
    n_phi = _require_grid_resolution("n_phi", n_phi)
    r = config.a * np.sqrt(s)
    iota = float(iota_profile(config, s))

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    # Boozer representation: helical excursion modulates axis position
    delta_R = config.helical_excursion * np.cos(config.N_fp * PH)
    R = config.R0 + r * np.cos(TH) + delta_R
    Z = r * np.sin(TH) + config.helical_excursion * np.sin(config.N_fp * PH)

    # |B| with toroidal, helical mirror, and helical-axis curvature modulation.
    epsilon_t = r / config.R0
    epsilon_h = config.mirror_ratio * np.sqrt(s)
    axis_curvature = (config.helical_excursion / config.R0) * np.sqrt(s)
    B = config.B0 * (
        1.0
        - epsilon_t * np.cos(TH)
        - epsilon_h * np.cos(config.N_fp * PH - iota * TH)
        - axis_curvature * np.cos(config.N_fp * PH)
    )

    return R, Z, B


def effective_ripple(config: StellaratorConfig, s: float) -> float:
    """Effective helical ripple epsilon_eff for neoclassical transport.

    Field-spectrum estimate inspired by Nemov et al., Phys. Plasmas 6 (1999)
    4622.  The toroidally averaged field component is removed on each
    poloidal ring, leaving the non-axisymmetric helical |B| spectrum that
    drives 1/nu stellarator transport.  The returned scalar preserves the
    expected zero-ripple axisymmetric limit, increases with radial flux label,
    and responds to both mirror modulation and helical-axis excursion.

    Parameters
    ----------
    config : StellaratorConfig
        Stellarator device geometry and rotational-transform profile parameters.
    s : float
        Normalised toroidal flux, s in (0, 1].

    Returns
    -------
    float
        Effective ripple epsilon_eff (dimensionless, 0 < eps_eff < 1).

    """
    s = _require_flux_label("s", s)
    if config.mirror_ratio == 0.0 and config.helical_excursion == 0.0:
        return 0.0

    _, _, B = stellarator_flux_surface(config, s, n_theta=96, n_phi=max(64, 16 * config.N_fp))
    B_mean = float(np.mean(B))
    if not np.isfinite(B_mean) or B_mean <= 0.0:
        raise ValueError("computed magnetic field mean must be finite and positive.")

    b_norm = B / B_mean - 1.0
    axisymmetric_component = np.mean(b_norm, axis=1, keepdims=True)
    nonaxisymmetric = b_norm - axisymmetric_component
    rms_nonaxisymmetric = float(np.sqrt(np.mean(nonaxisymmetric**2)))

    phi_spectrum = np.fft.rfft(nonaxisymmetric, axis=1)
    harmonic_idx = min(config.N_fp, phi_spectrum.shape[1] - 1)
    harmonic_power = float(np.mean(np.abs(phi_spectrum[:, harmonic_idx]) ** 2))
    total_power = float(np.mean(np.sum(np.abs(phi_spectrum[:, 1:]) ** 2, axis=1)))
    spectral_concentration = harmonic_power / max(total_power, 1e-30)

    helical_strength = (
        np.sqrt(2.0) * rms_nonaxisymmetric * np.sqrt(max(spectral_concentration, 0.0))
    )
    aspect_factor = np.sqrt(max(config.a / config.R0, 1e-12))
    eps_eff = helical_strength**1.5 * aspect_factor / np.sqrt(config.N_fp)
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
        Stellarator device geometry and rotational-transform profile parameters.
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
        Stellarator device geometry and rotational-transform profile parameters.
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
