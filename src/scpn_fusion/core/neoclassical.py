# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neoclassical Transport Module
"""
Neoclassical transport and bootstrap current models.

Implements the Chang-Hinton ion thermal diffusivity, collisionality
scaling, and the Sauter bootstrap current model for general axisymmetry.
"""

from __future__ import annotations

import numpy as np

# ─── Physical constants (CODATA 2018) ───────────────────────────────
_E_CHARGE = 1.602176634e-19  # C
_M_PROTON = 1.67262192369e-27  # kg
_M_ELECTRON = 9.1093837015e-31  # kg
_EPS0 = 8.8541878128e-12  # F/m
_LN_LAMBDA = 17.0  # Coulomb logarithm, Wesson Ch. 14


def collisionality(
    n_e_19: float,
    T_kev: float,
    q: float,
    R: float,
    epsilon: float,
    mass_amu: float = 2.0,
    z_eff: float = 1.0,
) -> float:
    """Compute the dimensionless neoclassical collisionality nu_star.

    nu_star = nu_ii * q * R / (epsilon^1.5 * v_th)

    Parameters
    ----------
    n_e_19 : float
        Electron density [10^19 m^-3].
    T_kev : float
        Species temperature [keV].
    q : float
        Safety factor.
    R : float
        Major radius [m].
    epsilon : float
        Inverse aspect ratio r/R.
    mass_amu : float
        Species mass in AMU (default 2.0 for D).
    z_eff : float
        Effective charge.

    Returns
    -------
    float — Dimensionless collisionality.
    """
    if T_kev <= 0.01 or epsilon < 1e-6 or n_e_19 <= 0:
        return 0.0

    T_J = T_kev * 1.602176634e-16
    m = mass_amu * _M_PROTON
    v_th = np.sqrt(2.0 * T_J / m)

    n_e = n_e_19 * 1e19
    # Ion-ion collision frequency
    nu_ii = (n_e * z_eff**2 * _E_CHARGE**4 * _LN_LAMBDA) / (
        12.0 * np.pi**1.5 * _EPS0**2 * np.sqrt(m) * T_J**1.5
    )

    return float(nu_ii * q * R / (epsilon**1.5 * v_th))


def chang_hinton_chi(
    q: float,
    epsilon: float,
    nu_star: float,
    rho_i: float,
    nu_ii: float,
) -> float:
    """Compute Chang-Hinton neoclassical ion thermal diffusivity.

    Reference: Chang, C.S. and Hinton, F.L., Phys. Fluids 25, 1493 (1982).

    Parameters
    ----------
    q : float
        Safety factor.
    epsilon : float
        Inverse aspect ratio r/R.
    nu_star : float
        Ion collisionality.
    rho_i : float
        Ion Larmor radius [m].
    nu_ii : float
        Ion-ion collision frequency [s^-1].

    Returns
    -------
    float — Ion thermal diffusivity [m^2/s].
    """
    if epsilon < 1e-6:
        return 0.0

    eps32 = epsilon**1.5
    alpha_sh = epsilon

    # Chang & Hinton, Eq. 10
    chi = (
        0.66
        * (1.0 + 1.54 * alpha_sh)
        * q**2
        * rho_i**2
        * nu_ii
        / (eps32 * (1.0 + 0.74 * nu_star ** (2.0 / 3.0)))
    )
    return float(chi)


def banana_plateau_chi(
    q: float,
    epsilon: float,
    nu_star: float,
    z_eff: float,
) -> float:
    """Dimensionless banana-plateau neoclassical scaling factor.

    Returns a dimensionless quantity proportional to the ion thermal
    diffusivity in the banana-plateau regime. Multiply by rho_i^2 * v_ti / R
    to obtain physical chi_i [m^2/s].

    Parameters
    ----------
    q : float
        Safety factor.
    epsilon : float
        Inverse aspect ratio.
    nu_star : float
        Collisionality.
    z_eff : float
        Effective charge.
    """
    if epsilon < 1e-6:
        return 0.0
    # Hinton & Hazeltine, Rev. Mod. Phys. 48, 239 (1976), Eq. 4.61
    # chi_bp ~ q^2 * epsilon^{-1.5} * (1 + 0.6*z_eff) / (1 + nu_star)
    return float(q**2 * epsilon ** (-1.5) * (1.0 + 0.6 * z_eff) / (1.0 + nu_star))


def sauter_bootstrap(
    rho: np.ndarray,
    Te: np.ndarray,
    Ti: np.ndarray,
    ne: np.ndarray,
    q: np.ndarray,
    R0: float,
    a: float,
    B0: float = 5.3,
    z_eff: float = 1.5,
) -> np.ndarray:
    """Compute the Sauter bootstrap current density profile.

    Reference: Sauter, O. et al., Phys. Plasmas 6, 2834 (1999).

    Parameters
    ----------
    rho : np.ndarray
        Normalised radial grid.
    Te, Ti : np.ndarray
        Electron and ion temperatures [keV].
    ne : np.ndarray
        Electron density [10^19 m^-3].
    q : np.ndarray
        Safety factor profile.
    R0, a : float
        Major and minor radii [m].
    B0 : float
        Toroidal magnetic field [T].
    z_eff : float
        Effective charge.

    Returns
    -------
    np.ndarray — Bootstrap current density [A/m^2].
    """
    n = len(rho)
    j_bs = np.zeros(n)

    # Precompute gradients (central differences)
    drho = rho[1] - rho[0]
    dne_drho = np.gradient(ne * 1e19, drho)
    dTe_drho = np.gradient(Te, drho)
    dTi_drho = np.gradient(Ti, drho)

    for i in range(1, n - 1):
        eps = rho[i] * a / R0
        if eps < 1e-6 or Te[i] <= 0.1 or ne[i] <= 0.1 or q[i] <= 0.5:
            continue

        # Trapped fraction (Sauter Eq. 14)
        f_t = 1.0 - (1.0 - eps) ** 2 / (np.sqrt(1.0 - eps**2) * (1.0 + 1.46 * np.sqrt(eps)))

        # Collisionality nu_star_e
        v_the = np.sqrt(2.0 * Te[i] * 1.602e-16 / _M_ELECTRON)
        nu_ee = (ne[i] * 1e19 * _E_CHARGE**4 * _LN_LAMBDA) / (
            12.0 * np.pi**1.5 * _EPS0**2 * _M_ELECTRON**0.5 * (Te[i] * 1.602e-16) ** 1.5
        )
        nu_star_e = nu_ee * q[i] * R0 / (eps**1.5 * v_the)

        # Sauter coefficients alpha, L31, L32, L34 (approximate)
        # Simplified for this module
        X = f_t / (1.0 + (1.0 + 0.3 * np.sqrt(nu_star_e)) * nu_star_e)
        L31 = X

        # dp/dρ in Pa
        grad_p = (Te[i] + Ti[i]) * 1.602e-16 * dne_drho[i] + ne[i] * 1e19 * 1.602e-16 * (
            dTe_drho[i] + dTi_drho[i]
        )

        # B_pol ≈ B₀ ε / q  (from q = rB_T / (R₀B_θ))
        B_pol = B0 * eps / max(q[i], 0.1)
        if B_pol < 1e-10:
            continue

        # j_bs ≈ -L31 (dp/dr) / B_pol, with dp/dr = (dp/dρ)/a
        # Sign preserved: negative grad_p (normal profile) gives positive j_bs
        j_bs[i] = -L31 * grad_p / (a * B_pol)

    return j_bs
