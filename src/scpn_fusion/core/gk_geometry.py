# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Miller Flux-Tube Geometry
"""
Miller parameterisation of local magnetic equilibrium geometry for
flux-tube gyrokinetic calculations.

Computes metric coefficients, field-line curvature, and the Jacobian
on a ballooning-angle grid from (R0, a, kappa, delta, q, s_hat, ...).

Reference: Miller et al., Phys. Plasmas 5 (1998) 973.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class MillerGeometry:
    """Flux-tube geometry on a ballooning-angle grid.

    All arrays have shape ``(n_theta,)``.
    """

    theta: NDArray[np.float64]  # ballooning angle grid
    R: NDArray[np.float64]  # major radius R(theta) [m]
    Z: NDArray[np.float64]  # vertical position Z(theta) [m]
    B_mag: NDArray[np.float64]  # |B|(theta) [T]
    jacobian: NDArray[np.float64]  # flux-surface Jacobian
    g_rr: NDArray[np.float64]  # |grad r|^2
    g_rt: NDArray[np.float64]  # grad r . grad theta
    g_tt: NDArray[np.float64]  # |grad theta|^2
    kappa_n: NDArray[np.float64]  # normal curvature
    kappa_g: NDArray[np.float64]  # geodesic curvature
    b_dot_grad_theta: NDArray[np.float64]  # B . grad(theta) / B


def miller_geometry(
    R0: float,
    a: float,
    rho: float,
    kappa: float = 1.0,
    delta: float = 0.0,
    s_kappa: float = 0.0,
    s_delta: float = 0.0,
    q: float = 1.4,
    s_hat: float = 0.78,
    alpha_MHD: float = 0.0,
    dR_dr: float = 0.0,
    B0: float = 5.3,
    n_theta: int = 64,
    n_period: int = 2,
) -> MillerGeometry:
    """Compute Miller geometry on a ballooning-angle grid.

    Parameters
    ----------
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    rho : float
        Normalised flux coordinate (r/a).
    kappa, delta : float
        Elongation and triangularity.
    s_kappa, s_delta : float
        Shear of elongation/triangularity: (r/kappa)(dkappa/dr), etc.
    q, s_hat : float
        Safety factor and magnetic shear s = (r/q)(dq/dr).
    alpha_MHD : float
        Shafranov shift parameter alpha = -q^2 R0 (dp/dr) / (B0^2 / 2mu0).
    dR_dr : float
        Shafranov shift gradient dR_axis/dr (typically negative).
    B0 : float
        Toroidal field at R0 [T].
    n_theta : int
        Grid points per 2*pi period.
    n_period : int
        Number of poloidal periods (ballooning copies).
    """
    r = rho * a
    theta = np.linspace(-n_period * np.pi, n_period * np.pi, n_theta * n_period, endpoint=False)

    # Miller et al. Eq. (1)-(2): flux surface shape
    delta_angle = np.arcsin(delta)
    R_s = R0 + r * np.cos(theta + delta_angle * np.sin(theta)) + dR_dr * r
    Z_s = kappa * r * np.sin(theta)

    # Derivatives w.r.t. theta
    dR_dt = -r * np.sin(theta + delta_angle * np.sin(theta)) * (1 + delta_angle * np.cos(theta))
    dZ_dt = kappa * r * np.cos(theta)

    # Derivatives w.r.t. r (at constant theta)
    dR_dr_tot = np.cos(theta + delta_angle * np.sin(theta)) + dR_dr
    dZ_dr_r = kappa * np.sin(theta)

    # Jacobian of (r, theta) → (R, Z): J = dR/dr * dZ/dtheta - dR/dtheta * dZ/dr
    jac = dR_dr_tot * dZ_dt - dR_dt * dZ_dr_r
    jac = np.where(np.abs(jac) < 1e-30, 1e-30, jac)

    # |grad r|^2 = (dR/dtheta)^2 + (dZ/dtheta)^2) / J^2
    g_rr = (dR_dt**2 + dZ_dt**2) / jac**2

    # grad r . grad theta = -(dR/dr * dR/dtheta + dZ/dr * dZ/dtheta) / J^2
    g_rt = -(dR_dr_tot * dR_dt + dZ_dr_r * dZ_dt) / jac**2

    # |grad theta|^2 = (dR/dr^2 + dZ/dr^2) / J^2
    g_tt = (dR_dr_tot**2 + dZ_dr_r**2) / jac**2

    # Toroidal field: B_phi = B0 * R0 / R (vacuum approximation)
    B_phi = B0 * R0 / R_s

    # Poloidal field: B_p = r / (q * R_s * |J/r|) — simplified
    abs_jac_over_r = np.abs(jac) / max(r, 1e-6)
    B_p = 1.0 / (q * abs_jac_over_r + 1e-30)

    B_mag = np.sqrt(B_phi**2 + B_p**2)

    # b . grad(theta) = B_p / (R_s * |grad theta|) ≈ 1 / (q * R_s) simplified
    b_dot_grad_theta = 1.0 / (q * R_s)

    # Curvature — Miller Eqs. (18)-(19)
    # Normal curvature kappa_n ≈ -(1/R)(cos(theta) + s_hat*theta*sin(theta))
    # Geodesic curvature kappa_g ≈ -(1/R)(sin(theta) - s_hat*theta*cos(theta) - alpha_MHD*sin(theta))
    inv_R = 1.0 / R_s
    kappa_n = -inv_R * (np.cos(theta) + (s_hat * theta - alpha_MHD) * np.sin(theta))
    kappa_g = -inv_R * (np.sin(theta) - (s_hat * theta - alpha_MHD) * np.cos(theta))

    return MillerGeometry(
        theta=theta,
        R=R_s,
        Z=Z_s,
        B_mag=B_mag,
        jacobian=jac,
        g_rr=g_rr,
        g_rt=g_rt,
        g_tt=g_tt,
        kappa_n=kappa_n,
        kappa_g=kappa_g,
        b_dot_grad_theta=b_dot_grad_theta,
    )


def circular_geometry(
    R0: float = 2.78,
    a: float = 1.0,
    rho: float = 0.5,
    q: float = 1.4,
    s_hat: float = 0.78,
    B0: float = 2.0,
    n_theta: int = 64,
    n_period: int = 2,
) -> MillerGeometry:
    """Circular cross-section limit (kappa=1, delta=0).

    Useful for verification against analytic results and the
    Cyclone Base Case (Dimits et al. 2000).
    """
    return miller_geometry(
        R0=R0,
        a=a,
        rho=rho,
        kappa=1.0,
        delta=0.0,
        s_kappa=0.0,
        s_delta=0.0,
        q=q,
        s_hat=s_hat,
        alpha_MHD=0.0,
        dR_dr=0.0,
        B0=B0,
        n_theta=n_theta,
        n_period=n_period,
    )
