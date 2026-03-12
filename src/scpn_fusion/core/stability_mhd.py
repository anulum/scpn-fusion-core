# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — MHD Stability: Seven-Criterion Suite
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
MHD stability analysis suite — seven criteria:

1. **Mercier** — interchange stability (D_M >= 0)
2. **Ballooning** — first-stability boundary (Connor-Hastie-Taylor)
3. **Kruskal-Shafranov** — external kink (q_edge > 1)
4. **Troyon** — normalised beta limit (beta_N < g)
5. **NTM** — neoclassical tearing mode seeding threshold
6. **RWM** — resistive wall mode (beta_N between no-wall and ideal-wall limits)
7. **Peeling-ballooning** — coupled pedestal stability (ELM boundary)

Criteria 4–7 live in stability_mhd_extended.py and are re-exported here
for backward compatibility.

References
----------
- Freidberg, *Ideal MHD*, Cambridge (2014), Ch. 12
- Connor, Hastie & Taylor, Phys. Rev. Lett. 40:396 (1978)
- Kruskal & Schwarzschild, Proc. R. Soc. Lond. A 223:348 (1954)
- Troyon et al., Plasma Phys. Control. Fusion 26:209 (1984)
- La Haye, Phys. Plasmas 13:055501 (2006)
- Snyder et al., Phys. Plasmas 9:2037 (2002)
- Snyder et al., Nucl. Fusion 51:103016 (2011)
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class KruskalShafranovResult:
    """External kink stability result (Kruskal-Shafranov criterion)."""

    q_edge: float
    stable: bool  # True if q_edge > 1
    margin: float  # q_edge - 1


@dataclass
class TroyonResult:
    """Troyon normalised-beta-limit result."""

    beta_N: float  # Normalised beta [% m T / MA]
    beta_N_crit_nowall: float  # Critical beta_N without wall (g = 2.8)
    beta_N_crit_wall: float  # Critical beta_N with ideal wall (g = 3.5)
    stable_nowall: bool
    stable_wall: bool
    margin_nowall: float  # beta_N_crit_nowall - beta_N


@dataclass
class NTMResult:
    """Neoclassical tearing mode seeding analysis result."""

    rho: NDArray[np.float64]
    delta_prime: NDArray[np.float64]  # Classical stability index (< 0 = stable)
    j_bs_drive: NDArray[np.float64]  # Bootstrap current fraction drive
    w_marginal: NDArray[np.float64]  # Marginal island width [m]
    ntm_unstable: NDArray[np.bool_]
    most_unstable_rho: float | None


@dataclass
class RWMResult:
    """Resistive Wall Mode stability result."""

    beta_N: float
    beta_N_crit_nowall: float
    beta_N_crit_wall: float
    stable: bool
    mode_growth_rate: float  # dimensionless: gamma*tau_w ~ (beta-beta_nw)/(beta_w-beta)


@dataclass
class PeelingBallooningResult:
    """Coupled peeling-ballooning stability at the H-mode pedestal.

    Snyder et al., Phys. Plasmas 9:2037 (2002); Nucl. Fusion 51:103016 (2011).
    """

    j_edge_norm: float  # j_edge / j_crit (peeling drive)
    alpha_edge_norm: float  # alpha / alpha_crit (ballooning drive)
    stability_distance: float  # distance from PB boundary (>0 = stable)
    stable: bool
    elm_type: str | None  # "type_I", "type_III", or None


@dataclass
class StabilitySummary:
    """Combined result from all MHD stability criteria."""

    mercier: MercierResult
    ballooning: BallooningResult
    kruskal_shafranov: KruskalShafranovResult
    troyon: TroyonResult | None
    ntm: NTMResult | None
    rwm: RWMResult | None
    peeling_ballooning: PeelingBallooningResult | None
    n_criteria_checked: int
    n_criteria_stable: int
    overall_stable: bool


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
    kappa: float = 1.0,
    delta: float = 0.0,
) -> QProfile:
    """Compute the safety-factor profile from a shape-aware approximation.

    Uses a parabolic current profile and a Uckan-style geometric correction
    for elongation (kappa) and triangularity (delta).

    Parameters
    ----------
    rho : array — normalised radius [0, 1]
    ne : array — electron density [10^19 m^-3]
    Ti, Te : array — ion/electron temperature [keV]
    R0 : float — major radius [m]
    a : float — minor radius [m]
    B0 : float — toroidal field on axis [T]
    Ip_MA : float — total plasma current [MA]
    kappa : float — elongation
    delta : float — triangularity

    Returns
    -------
    QProfile
    """
    mu0 = 4.0 * np.pi * 1e-7
    Ip = Ip_MA * 1e6  # MA -> A
    epsilon = a / R0

    # Uckan-style shape correction proxy.
    f_shape = (1.0 + kappa**2 * (1.0 + 2.0 * delta**2 - 1.2 * delta**3)) / 2.0
    f_aspect = (1.17 - 0.65 * epsilon) / (1.0 - epsilon**2)
    f_total = f_shape * f_aspect

    # Parabolic current profile: j(rho) ~ (1 - rho^2)
    # => I_enclosed(rho) = Ip * (2*rho^2 - rho^4)
    rho_safe = np.maximum(rho, 1e-10)
    I_enc = Ip * (2.0 * rho_safe**2 - rho_safe**4)

    B_theta = mu0 * I_enc / (2.0 * np.pi * rho_safe * a)
    B_theta = np.maximum(B_theta, 1e-12)

    q_cyl = rho_safe * a * B0 / (R0 * B_theta)
    q = q_cyl * f_total

    q0 = f_total * np.pi * a**2 * B0 / (mu0 * R0 * Ip)
    q[0] = q0

    # Magnetic shear: s = (rho/q) * dq/drho
    dq = np.gradient(q, rho_safe)
    shear = (rho_safe / q) * dq
    shear[0] = 0.0  # Zero shear at axis by symmetry

    # Normalised pressure gradient (alpha_MHD)
    e_keV_to_J = 1.602176634e-16
    p_Pa = ne * 1e19 * (Ti + Te) * e_keV_to_J
    dp_drho = np.gradient(p_Pa, rho_safe)
    dp_dr = dp_drho / a

    alpha_mhd = -2.0 * mu0 * R0 * q**2 / (B0**2) * dp_dr

    q_min_idx = int(np.argmin(q))
    q_min = float(q[q_min_idx])
    q_min_rho = float(rho[q_min_idx])
    q_edge_val = float(q[-1])

    return QProfile(
        rho=rho,
        q=q,
        shear=shear,
        alpha_mhd=alpha_mhd,
        q_min=q_min,
        q_min_rho=q_min_rho,
        q_edge=q_edge_val,
    )


# ── Mercier criterion ────────────────────────────────────────────────


def mercier_stability(qp: QProfile) -> MercierResult:
    """Evaluate the Mercier interchange stability criterion (Suydam form).

    D_M = s^2 / 4 - alpha_mhd
    Stable where D_M >= 0.
    """
    s = qp.shear
    alpha = qp.alpha_mhd

    D_M = (s**2 / 4.0) - alpha

    stable = D_M >= 0.0

    first_unstable_rho: float | None = None
    for i in range(5, len(qp.rho)):  # skip axis region
        if not stable[i]:
            first_unstable_rho = float(qp.rho[i])
            break

    return MercierResult(
        rho=qp.rho,
        D_M=D_M.astype(np.float64),
        stable=stable,
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
        rho=qp.rho,
        s=s,
        alpha=alpha,
        alpha_crit=alpha_crit,
        stable=stable,
        margin=margin,
    )


# ── Kruskal-Shafranov criterion ────────────────────────────────────


def kruskal_shafranov_stability(qp: QProfile) -> KruskalShafranovResult:
    """Evaluate the Kruskal-Shafranov external kink stability criterion.

    The plasma is stable against the m=1/n=1 external kink mode when the
    edge safety factor satisfies q(edge) > 1.

    Parameters
    ----------
    qp : QProfile

    Returns
    -------
    KruskalShafranovResult

    References
    ----------
    Kruskal & Schwarzschild, Proc. R. Soc. Lond. A 223:348 (1954)
    Shafranov, Sov. Phys. Tech. Phys. 15:175 (1970)
    """
    stable = qp.q_edge > 1.0
    margin = qp.q_edge - 1.0
    return KruskalShafranovResult(
        q_edge=qp.q_edge,
        stable=stable,
        margin=margin,
    )


# ── Extended criteria (Troyon, NTM, RWM, PB) ───────────────────────
# Implementations live in stability_mhd_extended.py; re-exported here.

from .stability_mhd_extended import (  # noqa: E402, F401
    ntm_stability,
    peeling_ballooning_stability,
    rwm_stability,
    troyon_beta_limit,
)


# ── Full stability check (all 7 criteria) ──────────────────────────


def run_full_stability_check(
    qp: QProfile,
    beta_t: float | None = None,
    Ip_MA: float | None = None,
    a: float | None = None,
    B0: float | None = None,
    R0: float | None = None,
    j_bs: NDArray[np.float64] | None = None,
    j_total: NDArray[np.float64] | None = None,
    j_edge: float | None = None,
    p_ped_Pa: float | None = None,
    kappa: float = 1.7,
    delta: float = 0.3,
) -> StabilitySummary:
    """Run all available MHD stability criteria and return a summary.

    Mercier, ballooning, and Kruskal-Shafranov are always evaluated.
    Troyon requires ``beta_t``, ``Ip_MA``, ``a``, and ``B0``.
    NTM requires ``j_bs``, ``j_total``, and ``a``.
    Peeling-ballooning requires ``j_edge``, ``p_ped_Pa``, ``R0``, ``a``, ``B0``.

    Parameters
    ----------
    qp : QProfile — pre-computed safety-factor profile
    beta_t : float, optional — total toroidal beta (dimensionless)
    Ip_MA : float, optional — plasma current [MA]
    a : float, optional — minor radius [m]
    B0 : float, optional — toroidal field on axis [T]
    R0 : float, optional — major radius [m]
    j_bs : array, optional — bootstrap current density [A/m^2]
    j_total : array, optional — total current density [A/m^2]
    j_edge : float, optional — edge parallel current density [A/m^2]
    p_ped_Pa : float, optional — pedestal-top pressure [Pa]
    kappa : float — elongation (default 1.7)
    delta : float — triangularity (default 0.3)

    Returns
    -------
    StabilitySummary
    """
    mr = mercier_stability(qp)
    br = ballooning_stability(qp)
    ks = kruskal_shafranov_stability(qp)

    n_checked = 3
    n_stable = 0

    mercier_ok = mr.first_unstable_rho is None
    if mercier_ok:
        n_stable += 1

    ballooning_ok = bool(np.all(br.stable))
    if ballooning_ok:
        n_stable += 1

    if ks.stable:
        n_stable += 1

    troyon_result: TroyonResult | None = None
    if beta_t is not None and Ip_MA is not None and a is not None and B0 is not None:
        troyon_result = troyon_beta_limit(beta_t, Ip_MA, a, B0)
        n_checked += 1
        if troyon_result.stable_nowall:
            n_stable += 1

    ntm_result: NTMResult | None = None
    if j_bs is not None and j_total is not None and a is not None:
        ntm_result = ntm_stability(qp, j_bs, j_total, a)
        n_checked += 1
        if not np.any(ntm_result.ntm_unstable):
            n_stable += 1

    rwm_result: RWMResult | None = None
    if troyon_result is not None:
        rwm_result = rwm_stability(troyon_result.beta_N)
        n_checked += 1
        if rwm_result.stable:
            n_stable += 1

    pb_result: PeelingBallooningResult | None = None
    if (
        j_edge is not None
        and p_ped_Pa is not None
        and R0 is not None
        and a is not None
        and B0 is not None
    ):
        pb_result = peeling_ballooning_stability(
            qp,
            j_edge,
            p_ped_Pa,
            R0,
            a,
            B0,
            kappa=kappa,
            delta=delta,
        )
        n_checked += 1
        if pb_result.stable:
            n_stable += 1

    overall = n_stable == n_checked

    return StabilitySummary(
        mercier=mr,
        ballooning=br,
        kruskal_shafranov=ks,
        troyon=troyon_result,
        ntm=ntm_result,
        rwm=rwm_result,
        peeling_ballooning=pb_result,
        n_criteria_checked=n_checked,
        n_criteria_stable=n_stable,
        overall_stable=overall,
    )
