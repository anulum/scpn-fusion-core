# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — MHD Stability: Five-Criterion Suite
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
MHD stability analysis suite — five criteria:

1. **Mercier** — interchange stability (D_M >= 0)
2. **Ballooning** — first-stability boundary (Connor-Hastie-Taylor)
3. **Kruskal-Shafranov** — external kink (q_edge > 1)
4. **Troyon** — normalised beta limit (beta_N < g)
5. **NTM** — neoclassical tearing mode seeding threshold

References
----------
- Freidberg, *Ideal MHD*, Cambridge (2014), Ch. 12
- Connor, Hastie & Taylor, Phys. Rev. Lett. 40:396 (1978)
- Kruskal & Schwarzschild, Proc. R. Soc. Lond. A 223:348 (1954)
- Troyon et al., Plasma Phys. Control. Fusion 26:209 (1984)
- La Haye, Phys. Plasmas 13:055501 (2006)
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


@dataclass
class KruskalShafranovResult:
    """External kink stability result (Kruskal-Shafranov criterion)."""

    q_edge: float
    stable: bool       # True if q_edge > 1
    margin: float      # q_edge - 1


@dataclass
class TroyonResult:
    """Troyon normalised-beta-limit result."""

    beta_N: float                # Normalised beta [% m T / MA]
    beta_N_crit_nowall: float    # Critical beta_N without wall (g = 2.8)
    beta_N_crit_wall: float      # Critical beta_N with ideal wall (g = 3.5)
    stable_nowall: bool
    stable_wall: bool
    margin_nowall: float         # beta_N_crit_nowall - beta_N


@dataclass
class NTMResult:
    """Neoclassical tearing mode seeding analysis result."""

    rho: NDArray[np.float64]
    delta_prime: NDArray[np.float64]       # Classical stability index (< 0 = stable)
    j_bs_drive: NDArray[np.float64]        # Bootstrap current fraction drive
    w_marginal: NDArray[np.float64]        # Marginal island width [m]
    ntm_unstable: NDArray[np.bool_]
    most_unstable_rho: float | None


@dataclass
class RWMResult:
    """Resistive Wall Mode stability result."""

    beta_N: float
    beta_N_crit_nowall: float
    beta_N_crit_wall: float
    stable: bool
    mode_growth_rate: float # 1/tau_wall units


@dataclass
class StabilitySummary:
    """Combined result from all MHD stability criteria."""

    mercier: MercierResult
    ballooning: BallooningResult
    kruskal_shafranov: KruskalShafranovResult
    troyon: TroyonResult | None
    ntm: NTMResult | None
    rwm: RWMResult | None
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
    """Evaluate the Mercier interchange stability criterion (Suydam form).

    D_M = s^2 / 4 - alpha_mhd
    Stable where D_M >= 0.
    """
    s = qp.shear
    alpha = qp.alpha_mhd

    # Suydam/Mercier index: shear stabilisation vs pressure-gradient drive
    D_M = (s**2 / 4.0) - alpha

    stable = D_M >= 0.0

    # Find first unstable location (skip axis where s=0)
    first_unstable_rho: float | None = None
    for i in range(5, len(qp.rho)): # skip axis region
        if not stable[i]:
            first_unstable_rho = float(qp.rho[i])
            break

    return MercierResult(
        rho=qp.rho, D_M=D_M.astype(np.float64), stable=stable,
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


# ── Kruskal-Shafranov criterion ────────────────────────────────────

def kruskal_shafranov_stability(qp: QProfile) -> KruskalShafranovResult:
    """Evaluate the Kruskal-Shafranov external kink stability criterion.

    The plasma is stable against the m=1/n=1 external kink mode when the
    edge safety factor satisfies q(edge) > 1.  Physically, when q_edge < 1
    the magnetic field-line pitch allows helical perturbations that are
    not stabilised by line-bending, driving a global kink.

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
        q_edge=qp.q_edge, stable=stable, margin=margin,
    )


# ── Troyon beta limit ──────────────────────────────────────────────

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
    beta_t : float — total toroidal beta (dimensionless, e.g. 0.025)
    Ip_MA : float — plasma current [MA]
    a : float — minor radius [m]
    B0 : float — toroidal magnetic field on axis [T]
    g_nowall : float — Troyon coefficient without wall (default 2.8)
    g_wall : float — Troyon coefficient with ideal wall (default 3.5)

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
    beta_t = float(beta_t)
    Ip_MA = float(Ip_MA)
    a = float(a)
    B0 = float(B0)

    I_N = Ip_MA / max(a * B0, 1e-10)  # normalised current [MA / (m T)]
    beta_N = 100.0 * beta_t / max(I_N, 1e-10)  # [% m T / MA]

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


# ── NTM seeding threshold ──────────────────────────────────────────

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

    The first term is classically stabilising when
    :math:`r_s \Delta' < 0`.  The second term is the bootstrap-current
    drive that destabilises the island once seeded.

    An NTM is potentially unstable at a given radius when the
    bootstrap drive exceeds the classical stabilisation.  The marginal
    island width (below which the island shrinks) is:

    .. math::
        w_\text{marg} = -\frac{j_\text{bs} / j_\phi}{r_s \Delta'}\,a

    NTM instability occurs where :math:`w_\text{marg} > 0` (positive
    bootstrap drive with negative classical :math:`\Delta'`).

    Parameters
    ----------
    qp : QProfile
    j_bs : array — bootstrap current density [A/m^2]
    j_total : array — total current density [A/m^2]
    a : float — minor radius [m]
    r_s_delta_prime : float — classical tearing stability index
        (negative = classically stable, default -2.0)

    Returns
    -------
    NTMResult

    References
    ----------
    La Haye, Phys. Plasmas 13:055501 (2006)
    Sauter et al., Phys. Plasmas 4:1654 (1997)
    """
    j_bs = np.asarray(j_bs, dtype=np.float64)
    j_total = np.asarray(j_total, dtype=np.float64)
    if j_bs.shape != qp.rho.shape or j_total.shape != qp.rho.shape:
        raise ValueError("j_bs and j_total must match qp.rho shape.")
    if not np.all(np.isfinite(j_bs)) or not np.all(np.isfinite(j_total)):
        raise ValueError("j_bs and j_total must be finite.")
    a = float(a)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("a must be finite and > 0.")

    j_total_safe = np.where(np.abs(j_total) > 1e-6, j_total, 1e-6)
    j_bs_frac = j_bs / j_total_safe  # bootstrap fraction

    # delta_prime array: here we use a uniform classical index along the
    # profile as a simplification.  A full tearing-mode code would solve
    # the outer-region equation at each rational surface.
    delta_prime = np.full_like(qp.rho, r_s_delta_prime)

    # Bootstrap drive strength
    j_bs_drive = j_bs_frac.copy()

    # Marginal island width: w_marg = -(j_bs/j_phi) * a / (r_s * Delta')
    # Only meaningful where delta_prime < 0 (classically stable baseline).
    denom = np.where(np.abs(delta_prime) > 1e-10, delta_prime, -1e-10)
    w_marginal = -j_bs_frac * a / denom
    w_marginal = np.maximum(w_marginal, 0.0)  # physical: width >= 0

    # NTM unstable where bootstrap drives a positive marginal width
    ntm_unstable = (w_marginal > 0.0) & (j_bs_frac > 0.0) & (delta_prime < 0.0)

    most_unstable_rho: float | None = None
    if np.any(ntm_unstable):
        # Pick radius with largest marginal island width
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


# ── RWM stability ──────────────────────────────────────────────────

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
    beta_N : float — normalised beta [% m T / MA]
    g_nowall : float — Troyon no-wall limit (default 2.8)
    g_wall : float — Troyon ideal-wall limit (default 3.5)

    Returns
    -------
    RWMResult
    """
    stable = beta_N < g_nowall
    
    # Growth rate estimate (heuristic dispersion relation)
    # gamma * tau_w ~ (beta - beta_nowall) / (beta_wall - beta)
    if beta_N > g_nowall:
        denom = max(g_wall - beta_N, 0.01)
        growth_rate = (beta_N - g_nowall) / denom
    else:
        growth_rate = 0.0

    return RWMResult(
        beta_N=beta_N,
        beta_N_crit_nowall=g_nowall,
        beta_N_crit_wall=g_wall,
        stable=stable,
        mode_growth_rate=float(growth_rate),
    )


# ── Full stability check (all 5 criteria) ──────────────────────────

def run_full_stability_check(
    qp: QProfile,
    beta_t: float | None = None,
    Ip_MA: float | None = None,
    a: float | None = None,
    B0: float | None = None,
    j_bs: NDArray[np.float64] | None = None,
    j_total: NDArray[np.float64] | None = None,
) -> StabilitySummary:
    """Run all available MHD stability criteria and return a summary.

    Mercier, ballooning, and Kruskal-Shafranov are always evaluated.
    Troyon requires ``beta_t``, ``Ip_MA``, ``a``, and ``B0``.
    NTM requires ``j_bs``, ``j_total``, and ``a``.

    Parameters
    ----------
    qp : QProfile — pre-computed safety-factor profile
    beta_t : float, optional — total toroidal beta (dimensionless)
    Ip_MA : float, optional — plasma current [MA]
    a : float, optional — minor radius [m]
    B0 : float, optional — toroidal field on axis [T]
    j_bs : array, optional — bootstrap current density [A/m^2]
    j_total : array, optional — total current density [A/m^2]

    Returns
    -------
    StabilitySummary
    """
    mr = mercier_stability(qp)
    br = ballooning_stability(qp)
    ks = kruskal_shafranov_stability(qp)

    n_checked = 3
    n_stable = 0

    # Mercier: stable if no unstable point found (past the axis)
    mercier_ok = mr.first_unstable_rho is None
    if mercier_ok:
        n_stable += 1

    # Ballooning: stable if all points are stable
    ballooning_ok = bool(np.all(br.stable))
    if ballooning_ok:
        n_stable += 1

    # KS: direct boolean
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

    overall = n_stable == n_checked

    return StabilitySummary(
        mercier=mr,
        ballooning=br,
        kruskal_shafranov=ks,
        troyon=troyon_result,
        ntm=ntm_result,
        rwm=rwm_result,
        n_criteria_checked=n_checked,
        n_criteria_stable=n_stable,
        overall_stable=overall,
    )
