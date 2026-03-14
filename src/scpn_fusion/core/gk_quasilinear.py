# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Quasilinear Transport Flux Model
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Quasilinear flux model converting linear eigenvalues to transport
coefficients.

Implements the mixing-length saturation rule and velocity-space
weighted quasilinear fluxes.  The output is chi_i, chi_e, D_e in
physical units [m^2/s].

References:
  - Bourdelle et al., Phys. Plasmas 14 (2007) 112501 — QuaLiKiz
  - Staebler et al., Phys. Plasmas 14 (2007) 055909 — TGLF saturation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.gk_eigenvalue import LinearGKResult
from scpn_fusion.core.gk_interface import GKOutput
from scpn_fusion.core.gk_species import GKSpecies

_E_CHARGE = 1.602176634e-19
_M_PROTON = 1.67262192369e-27


def mixing_length_saturation(
    gamma: NDArray[np.float64],
    omega_r: NDArray[np.float64],
    k_y: NDArray[np.float64],
    gamma_floor: float = 1e-6,
) -> NDArray[np.float64]:
    """Mixing-length saturation: |phi_k|^2 = gamma / (k_perp^2 * |omega_r|).

    Returns saturation amplitude array, shape (n_ky,).
    """
    abs_omega = np.maximum(np.abs(omega_r), gamma_floor)
    k_perp_sq = k_y**2  # k_perp ~ k_y in flux-tube
    phi_sq = np.where(gamma > 0, gamma / (k_perp_sq * abs_omega), 0.0)
    return phi_sq


def quasilinear_fluxes_from_spectrum(
    result: LinearGKResult,
    ion: GKSpecies,
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    saturation: str = "mixing_length",
) -> GKOutput:
    """Convert linear GK spectrum to physical transport fluxes.

    Parameters
    ----------
    result : LinearGKResult
        Output from solve_linear_gk().
    ion : GKSpecies
        Main ion species (for gyro-Bohm normalisation).
    R0, a, B0 : float
        Geometry and field for dimensional conversion.
    saturation : str
        Saturation model: "mixing_length" (default).
    """
    if len(result.k_y) == 0:
        return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=True, dominant_mode="stable")

    phi_sq = mixing_length_saturation(result.gamma, result.omega_r, result.k_y)

    # Quasilinear weights
    # chi_i ~ sum_ky gamma_k * phi_sq_k * (omega_*Ti / omega_r_k)
    # chi_e ~ sum_ky gamma_k * phi_sq_k * (omega_*Te / omega_r_k)
    chi_i_norm = 0.0
    chi_e_norm = 0.0
    D_e_norm = 0.0

    for i in range(len(result.k_y)):
        if result.gamma[i] <= 0:
            continue
        ky = result.k_y[i]
        omega_r = result.omega_r[i]
        if abs(omega_r) < 1e-10:
            continue

        amp = result.gamma[i] * phi_sq[i]

        mt = result.mode_type[i]
        if mt == "ITG":
            omega_star_i = -ky * ion.R_L_T / max(ion.temperature_keV / ion.temperature_keV, 0.1)
            chi_i_norm += amp * abs(omega_star_i / omega_r)
        elif mt == "TEM":
            omega_star_e = ky * ion.R_L_T  # electron R/L_T as proxy
            omega_star_n = ky * ion.R_L_n
            chi_e_norm += amp * abs(omega_star_e / omega_r)
            D_e_norm += amp * abs(omega_star_n / omega_r)
        elif mt == "ETG":
            omega_star_e = ky * ion.R_L_T
            chi_e_norm += amp * abs(omega_star_e / omega_r) / 60.0**2

    # Gyro-Bohm normalisation: chi_gB = rho_s^2 * c_s / a
    m_i = ion.mass_amu * _M_PROTON
    T_i_J = ion.temperature_keV * 1e3 * _E_CHARGE
    c_s = np.sqrt(T_i_J / m_i)
    rho_s = m_i * c_s / (_E_CHARGE * B0)
    chi_gB = rho_s**2 * c_s / a

    chi_i = float(chi_i_norm * chi_gB)
    chi_e = float(chi_e_norm * chi_gB)
    D_e = float(D_e_norm * chi_gB)

    # Dominant mode from peak growth rate
    if result.gamma_max > 0:
        idx_max = int(np.argmax(result.gamma))
        dominant = result.mode_type[idx_max]
    else:
        dominant = "stable"

    return GKOutput(
        chi_i=chi_i,
        chi_e=chi_e,
        D_e=D_e,
        gamma=result.gamma,
        omega_r=result.omega_r,
        k_y=result.k_y,
        dominant_mode=dominant,
        converged=True,
    )


def critical_gradient_scan(
    R_L_Ti_values: NDArray[np.float64],
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    q: float = 1.4,
    s_hat: float = 0.78,
    n_ky: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scan R/L_Ti to find the critical gradient and transport stiffness.

    Returns (R_L_Ti_array, gamma_max_array).
    """
    from scpn_fusion.core.gk_eigenvalue import solve_linear_gk
    from scpn_fusion.core.gk_species import deuterium_ion, electron

    gamma_max_list = []
    for rlt in R_L_Ti_values:
        species = [deuterium_ion(R_L_T=rlt), electron()]
        result = solve_linear_gk(
            species_list=species,
            R0=R0,
            a=a,
            B0=B0,
            q=q,
            s_hat=s_hat,
            n_ky_ion=n_ky,
            n_ky_etg=0,
            n_theta=32,
            n_period=1,
        )
        gamma_max_list.append(result.gamma_max)

    return R_L_Ti_values, np.array(gamma_max_list)
