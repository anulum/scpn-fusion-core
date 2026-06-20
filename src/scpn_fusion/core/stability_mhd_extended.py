# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MHD Stability: Extended Criteria
"""Extended MHD stability criteria: Troyon, NTM, RWM, peeling-ballooning.

Separated from stability_mhd.py for module-size compliance.
All result dataclasses remain in stability_mhd.py.

References
----------
- Troyon et al., Plasma Phys. Control. Fusion 26:209 (1984)
- La Haye, Phys. Plasmas 13:055501 (2006)
- Snyder et al., Phys. Plasmas 9:2037 (2002)
- Snyder et al., Nucl. Fusion 51:103016 (2011)

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .stability_mhd import (
    NTMResult,
    PeelingBallooningResult,
    QProfile,
    RWMResult,
    TroyonResult,
)


def _validate_qp_basic(qp: QProfile) -> None:
    rho = np.asarray(qp.rho, dtype=np.float64)
    q = np.asarray(qp.q, dtype=np.float64)
    shear = np.asarray(qp.shear, dtype=np.float64)
    alpha = np.asarray(qp.alpha_mhd, dtype=np.float64)
    if rho.ndim != 1 or rho.size < 3:
        raise ValueError("qp.rho must be one-dimensional with at least 3 points.")
    if not (q.shape == shear.shape == alpha.shape == rho.shape):
        raise ValueError("qp arrays must share the same one-dimensional shape.")
    if (
        not np.all(np.isfinite(rho))
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(shear))
        or not np.all(np.isfinite(alpha))
    ):
        raise ValueError("qp arrays must contain finite values.")
    if not np.all(np.diff(rho) > 0.0):
        raise ValueError("qp.rho must be strictly increasing.")
    if np.any(q <= 0.0):
        raise ValueError("qp.q must be strictly positive.")
    if not np.isfinite(float(qp.q_edge)) or float(qp.q_edge) <= 0.0:
        raise ValueError("qp.q_edge must be finite and > 0.")
    if not np.isclose(float(qp.q_edge), float(q[-1]), rtol=1e-12, atol=1e-12):
        raise ValueError("qp.q_edge must match qp.q[-1].")


def troyon_beta_limit(
    beta_t: float,
    Ip_MA: float,
    a: float,
    B0: float,
    g_nowall: float = 2.8,
    g_wall: float = 3.5,
) -> TroyonResult:
    r"""Evaluate the Troyon normalised-beta limit.

    The normalised beta is defined as:

    .. math::
        \beta_N = \frac{\beta_t}{I_N}  \quad\text{where}\quad
        I_N = \frac{I_p\,[\text{MA}]}{a\,[\text{m}]\,B_0\,[\text{T}]}

    The factor of 100 converts fractional *beta_t* into the conventional
    percent-based normalisation (units: % m T / MA).

    Stability requires beta_N < g, where g ~ 2.8 (no-wall) or g ~ 3.5
    (ideal-wall).

    Parameters
    ----------
    beta_t : float
        Total toroidal beta (dimensionless, e.g. 0.025).
    Ip_MA : float
        Plasma current [MA].
    a : float
        Minor radius [m].
    B0 : float
        Toroidal magnetic field on axis [T].
    g_nowall : float
        Troyon coefficient without wall (default 2.8).
    g_wall : float
        Troyon coefficient with ideal wall (default 3.5).

    Returns
    -------
    TroyonResult

    References
    ----------
    Troyon et al., Plasma Phys. Control. Fusion 26:209 (1984)

    """
    for name, value in {"beta_t": beta_t, "Ip_MA": Ip_MA, "a": a, "B0": B0}.items():
        val = float(value)
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite.")
        if name != "beta_t" and val <= 0.0:
            raise ValueError(f"{name} must be > 0.")
        if name == "beta_t" and val < 0.0:
            raise ValueError("beta_t must be >= 0.")
    for name, value in {"g_nowall": g_nowall, "g_wall": g_wall}.items():
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be finite and > 0.")
    g_nowall = float(g_nowall)
    g_wall = float(g_wall)
    if g_wall <= g_nowall:
        raise ValueError("g_wall must be greater than g_nowall.")

    beta_t = float(beta_t)
    Ip_MA = float(Ip_MA)
    a = float(a)
    B0 = float(B0)

    I_N = Ip_MA / (a * B0)
    beta_N = 100.0 * beta_t / I_N

    beta_N_crit_nw = g_nowall
    beta_N_crit_w = g_wall

    stable_nw = beta_N < beta_N_crit_nw
    stable_w = beta_N < beta_N_crit_w
    margin_nw = beta_N_crit_nw - beta_N

    return TroyonResult(
        beta_N=beta_N,
        beta_N_crit_nowall=beta_N_crit_nw,
        beta_N_crit_wall=beta_N_crit_w,
        stable_nowall=stable_nw,
        stable_wall=stable_w,
        margin_nowall=margin_nw,
    )


def ntm_stability(
    qp: QProfile,
    j_bs: NDArray[np.float64],
    j_total: NDArray[np.float64],
    a: float,
    r_s_delta_prime: float = -2.0,
) -> NTMResult:
    r"""Reduced-order neoclassical tearing mode (NTM) seeding analysis.

    The modified Rutherford equation for island width *w* is (reduced-order):

    .. math::
        \tau_R\,\frac{dw}{dt}
            = r_s\,\Delta'(w)
            + \frac{j_\text{bs}}{j_\phi}\,\frac{a}{w}

    An NTM is potentially unstable at a given radius when the
    bootstrap drive exceeds the classical stabilisation.  The marginal
    island width (below which the island shrinks) is:

    .. math::
        w_\text{marg} = -\frac{j_\text{bs} / j_\phi}{r_s \Delta'}\,a

    Parameters
    ----------
    qp : QProfile
        Safety-factor profile providing the rational-surface locations.
    j_bs : array
        Bootstrap current density [A/m^2].
    j_total : array
        Total current density [A/m^2].
    a : float
        Minor radius [m].
    r_s_delta_prime : float
        Classical tearing stability index
        (negative = classically stable, default -2.0).

    Returns
    -------
    NTMResult

    References
    ----------
    La Haye, Phys. Plasmas 13:055501 (2006)
    Sauter et al., Phys. Plasmas 4:1654 (1997)

    """
    _validate_qp_basic(qp)
    j_bs = np.asarray(j_bs, dtype=np.float64)
    j_total = np.asarray(j_total, dtype=np.float64)
    if j_bs.shape != qp.rho.shape or j_total.shape != qp.rho.shape:
        raise ValueError("j_bs and j_total must match qp.rho shape.")
    if not np.all(np.isfinite(j_bs)) or not np.all(np.isfinite(j_total)):
        raise ValueError("j_bs and j_total must be finite.")
    a = float(a)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("a must be finite and > 0.")
    rdp = float(r_s_delta_prime)
    if not np.isfinite(rdp):
        raise ValueError("r_s_delta_prime must be finite.")

    j_total_safe = np.where(np.abs(j_total) > 1e-6, j_total, 1e-6)
    j_bs_frac = j_bs / j_total_safe

    delta_prime = np.full_like(qp.rho, rdp)
    j_bs_drive = j_bs_frac.copy()

    denom = np.where(np.abs(delta_prime) > 1e-10, delta_prime, -1e-10)
    w_marginal = -j_bs_frac * a / denom
    w_marginal = np.maximum(w_marginal, 0.0)

    # "Unstable" here means a finite seed island exceeding w_marginal can grow
    # via bootstrap drive — triggerability, not spontaneous growth from zero.
    ntm_unstable = (w_marginal > 0.0) & (j_bs_frac > 0.0) & (delta_prime < 0.0)

    most_unstable_rho: float | None = None
    if np.any(ntm_unstable):
        idx = int(np.argmax(np.where(ntm_unstable, w_marginal, 0.0)))
        most_unstable_rho = float(qp.rho[idx])

    return NTMResult(
        rho=qp.rho,
        delta_prime=delta_prime,
        j_bs_drive=j_bs_drive,
        w_marginal=w_marginal,
        ntm_unstable=ntm_unstable,
        most_unstable_rho=most_unstable_rho,
    )


def rwm_stability(
    beta_N: float,
    g_nowall: float = 2.8,
    g_wall: float = 3.5,
) -> RWMResult:
    """Evaluate Resistive Wall Mode (RWM) stability.

    The RWM occurs when beta_N exceeds the no-wall limit but remains
    below the ideal-wall limit.  The mode grows on the resistive
    time-scale of the wall (tau_wall).

    Parameters
    ----------
    beta_N : float
        Normalised beta [% m T / MA].
    g_nowall : float
        Troyon no-wall limit (default 2.8).
    g_wall : float
        Troyon ideal-wall limit (default 3.5).

    Returns
    -------
    RWMResult

    """
    beta_n = float(beta_N)
    g_nw = float(g_nowall)
    g_w = float(g_wall)
    if not np.isfinite(beta_n):
        raise ValueError("beta_N must be finite.")
    if not np.isfinite(g_nw) or g_nw <= 0.0:
        raise ValueError("g_nowall must be finite and > 0.")
    if not np.isfinite(g_w) or g_w <= 0.0:
        raise ValueError("g_wall must be finite and > 0.")
    if g_w <= g_nw:
        raise ValueError("g_wall must be greater than g_nowall.")

    stable = beta_n < g_nw

    if beta_n > g_nw:
        denom = max(g_w - beta_n, 0.01)
        growth_rate = (beta_n - g_nw) / denom
    else:
        growth_rate = 0.0

    return RWMResult(
        beta_N=beta_n,
        beta_N_crit_nowall=g_nw,
        beta_N_crit_wall=g_w,
        stable=stable,
        mode_growth_rate=float(growth_rate),
    )


def peeling_ballooning_stability(
    qp: QProfile,
    j_edge: float,
    p_ped_Pa: float,
    R0: float,
    a: float,
    B0: float,
    kappa: float = 1.7,
    delta: float = 0.3,
) -> PeelingBallooningResult:
    r"""Coupled peeling-ballooning stability for H-mode pedestal.

    The instability boundary in normalised (j, alpha) space is approximately
    elliptical with shaping corrections:

    .. math::
        \left(\frac{j_\parallel}{j_{\rm crit}}\right)^2
        + \left(\frac{\alpha}{\alpha_{\rm crit}}\right)^2 = 1

    Parameters
    ----------
    qp : QProfile
        Pre-computed safety-factor profile.
    j_edge : float
        Edge parallel current density [A/m^2].
    p_ped_Pa : float
        Pedestal-top pressure [Pa].
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    B0 : float
        Toroidal field on axis [T].
    kappa : float
        Elongation (default 1.7).
    delta : float
        Triangularity (default 0.3).

    Returns
    -------
    PeelingBallooningResult

    References
    ----------
    Snyder et al., Phys. Plasmas 9:2037 (2002)
    Snyder et al., Nucl. Fusion 51:103016 (2011)

    """
    _validate_qp_basic(qp)
    edge_current = float(j_edge)
    p_ped = float(p_ped_Pa)
    r_major = float(R0)
    r_minor = float(a)
    b_axis = float(B0)
    elongation = float(kappa)
    triangularity = float(delta)
    if not np.isfinite(edge_current):
        raise ValueError("j_edge must be finite.")
    if not np.isfinite(p_ped) or p_ped < 0.0:
        raise ValueError("p_ped_Pa must be finite and >= 0.")
    for name, value in {"R0": r_major, "a": r_minor, "B0": b_axis, "kappa": elongation}.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0.")
    if not np.isfinite(triangularity) or abs(triangularity) >= 1.0:
        raise ValueError("delta must be finite with |delta| < 1.")

    mu0 = 4.0 * np.pi * 1e-7
    q_edge = max(qp.q_edge, 1.01)

    # Shaping factor: Snyder 2002 Fig. 5 + Snyder 2011 EPED calibration
    f_shape = (1.0 + 0.5 * (elongation - 1.0)) * (1.0 + 0.8 * triangularity)

    B_pol_denom = 2.0 * np.pi * r_minor * np.sqrt((1.0 + elongation**2) / 2.0)
    Ip_approx = 2.0 * np.pi * r_minor * b_axis / (mu0 * q_edge * r_major)
    B_pol = mu0 * Ip_approx / B_pol_denom

    j_crit = 2.0 * B_pol * f_shape / (mu0 * q_edge**2 * r_major)
    j_crit = max(j_crit, 1e-6)

    s_edge = float(qp.shear[-1]) if len(qp.shear) > 0 else 2.0
    s_edge = max(s_edge, 0.1)
    if s_edge < 1.0:
        alpha_crit_base = s_edge * (1.0 - s_edge / 2.0)
    else:
        alpha_crit_base = 0.6 * s_edge
    alpha_crit = max(alpha_crit_base * (1.0 + 0.3 * (elongation - 1.0)), 0.01)

    # Pedestal alpha_MHD with Delta_ped ~ 0.05
    Delta_ped = 0.05
    dp_dr = p_ped / max(Delta_ped * r_minor, 1e-3)
    alpha_ped = 2.0 * mu0 * r_major * q_edge**2 / (b_axis**2) * dp_dr

    j_norm = abs(edge_current) / j_crit
    alpha_norm = alpha_ped / alpha_crit

    pb_radius = float(np.sqrt(j_norm**2 + alpha_norm**2))
    stability_distance = 1.0 - pb_radius
    stable = stability_distance > 0.0

    elm_type: str | None = None
    if not stable:
        if alpha_norm > j_norm:
            elm_type = "type_I"
        else:
            elm_type = "type_III"

    return PeelingBallooningResult(
        j_edge_norm=float(j_norm),
        alpha_edge_norm=float(alpha_norm),
        stability_distance=float(stability_distance),
        stable=stable,
        elm_type=elm_type,
    )
