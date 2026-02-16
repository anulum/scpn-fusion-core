# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — MHD Stability: Mercier & Ballooning Criteria
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Pressure-driven MHD stability analysis: q-profile, Mercier criterion,
and first-stability ballooning boundary.

References
----------
- Freidberg, *Ideal MHD*, Cambridge (2014), Ch. 12
- Connor, Hastie & Taylor, Phys. Rev. Lett. 40:396 (1978)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class QProfile:
    """Safety-factor profile and derived quantities."""

    rho: NDArray[np.float64]
    q: NDArray[np.float64]
    shear: NDArray[np.float64]
    alpha_mhd: NDArray[np.float64]
    q_min: float
    q_min_rho: float
    q_edge: float


@dataclass
class MercierResult:
    """Mercier interchange stability result."""

    rho: NDArray[np.float64]
    D_M: NDArray[np.float64]
    stable: NDArray[np.bool_]
    first_unstable_rho: float | None


@dataclass
class BallooningResult:
    """First-stability ballooning boundary result."""

    rho: NDArray[np.float64]
    s: NDArray[np.float64]
    alpha: NDArray[np.float64]
    alpha_crit: NDArray[np.float64]
    stable: NDArray[np.bool_]
    margin: NDArray[np.float64]


# ── Q-profile computation ───────────────────────────────────────────

def compute_q_profile(
    rho: NDArray[np.float64],
    ne: NDArray[np.float64],
    Ti: NDArray[np.float64],
    Te: NDArray[np.float64],
    R0: float,
    a: float,
    B0: float,
    Ip_MA: float,
) -> QProfile:
    """Compute the safety-factor profile from cylindrical approximation.

    Uses a parabolic current profile:
        I_enclosed(rho) = Ip * (2*rho^2 - rho^4)

    and the cylindrical safety factor:
        q(rho) = rho * a * B0 / (R0 * B_theta(rho))

    Parameters
    ----------
    rho : array — normalised radius [0, 1]
    ne : array — electron density [10^19 m^-3]
    Ti, Te : array — ion/electron temperature [keV]
    R0 : float — major radius [m]
    a : float — minor radius [m]
    B0 : float — toroidal field on axis [T]
    Ip_MA : float — total plasma current [MA]

    Returns
    -------
    QProfile
    """
    mu0 = 4.0 * np.pi * 1e-7
    Ip = Ip_MA * 1e6  # MA -> A
    n = len(rho)

    # Parabolic current profile: j(rho) ∝ (1 - rho^2)
    # => I_enclosed(rho) = Ip * (2*rho^2 - rho^4)
    rho_safe = np.maximum(rho, 1e-10)
    I_enc = Ip * (2.0 * rho_safe**2 - rho_safe**4)

    # B_theta(rho) = mu0 * I_enc / (2 * pi * rho * a)
    B_theta = mu0 * I_enc / (2.0 * np.pi * rho_safe * a)
    B_theta = np.maximum(B_theta, 1e-12)

    # q(rho) = rho * a * B0 / (R0 * B_theta)
    q = rho_safe * a * B0 / (R0 * B_theta)

    # Fix axis: q(0) = q(eps) (L'Hopital limit gives q0 = a^2 B0 / (R0 mu0 Ip / pi))
    # Analytic: q0 = 2 pi a^2 B0 / (mu0 R0 Ip * 2) for parabolic j
    q0 = np.pi * a**2 * B0 / (mu0 * R0 * Ip)
    q[0] = q0

    # Magnetic shear: s = (rho/q) * dq/drho
    dq = np.gradient(q, rho_safe)
    shear = (rho_safe / q) * dq
    shear[0] = 0.0  # Zero shear at axis by symmetry

    # Normalised pressure gradient (alpha_MHD)
    # alpha = -2 mu0 R0 q^2 / B0^2 * dp/dr
    # p = n_e * (Ti + Te) in keV * 10^19 m^-3 => convert to Pa
    e_keV_to_J = 1.602176634e-16
    p_Pa = ne * 1e19 * (Ti + Te) * e_keV_to_J  # pressure in Pa
    dp_drho = np.gradient(p_Pa, rho_safe)  # dp/d(rho)
    dp_dr = dp_drho / a  # dp/dr [Pa/m]

    alpha_mhd = -2.0 * mu0 * R0 * q**2 / (B0**2) * dp_dr
    alpha_mhd = np.maximum(alpha_mhd, 0.0)  # Physical: alpha >= 0

    q_min_idx = int(np.argmin(q))
    q_min = float(q[q_min_idx])
    q_min_rho = float(rho[q_min_idx])
    q_edge_val = float(q[-1])

    return QProfile(
        rho=rho, q=q, shear=shear, alpha_mhd=alpha_mhd,
        q_min=q_min, q_min_rho=q_min_rho, q_edge=q_edge_val,
    )


# ── Mercier criterion ────────────────────────────────────────────────

def mercier_stability(qp: QProfile) -> MercierResult:
    """Evaluate the Mercier interchange stability criterion.

    The Mercier index is (Freidberg, Ch. 12):
        D_M = s*(s - 1) + alpha*(1 - s/2)

    Stable where D_M >= 0.

    Parameters
    ----------
    qp : QProfile

    Returns
    -------
    MercierResult
    """
    s = qp.shear
    alpha = qp.alpha_mhd

    D_M = s * (s - 1.0) + alpha * (1.0 - s / 2.0)

    stable = D_M >= 0.0

    # Find first unstable location (skip axis where s=0 makes D_M=0)
    first_unstable_rho: float | None = None
    for i in range(2, len(qp.rho)):
        if not stable[i]:
            first_unstable_rho = float(qp.rho[i])
            break

    return MercierResult(
        rho=qp.rho, D_M=D_M, stable=stable,
        first_unstable_rho=first_unstable_rho,
    )


# ── Ballooning stability ────────────────────────────────────────────

def ballooning_stability(qp: QProfile) -> BallooningResult:
    """Evaluate the first ballooning stability boundary.

    The critical normalised pressure gradient (Connor-Hastie-Taylor 1978):
        alpha_crit(s) = s*(1 - s/2)   for s < 1
                      = 0.6*s          for s >= 1

    Stable where alpha <= alpha_crit.

    Parameters
    ----------
    qp : QProfile

    Returns
    -------
    BallooningResult
    """
    s = qp.shear
    alpha = qp.alpha_mhd

    alpha_crit = np.where(s < 1.0, s * (1.0 - s / 2.0), 0.6 * s)
    alpha_crit = np.maximum(alpha_crit, 0.0)

    stable = alpha <= alpha_crit
    margin = alpha_crit - alpha

    return BallooningResult(
        rho=qp.rho, s=s, alpha=alpha,
        alpha_crit=alpha_crit, stable=stable, margin=margin,
    )
