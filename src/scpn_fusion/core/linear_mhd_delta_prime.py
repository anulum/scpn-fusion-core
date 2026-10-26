# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Cylindrical Tearing Index by Outer-Region Shooting
r"""Tearing stability index :math:`\Delta'` of a Wesson cylinder by two-sided shooting.

The ideal outer equation is solved independently of the eigenvalue solver
(:mod:`scpn_fusion.core.linear_mhd_1d`) and serves as the reference for the
cylindrical Furth–Killeen–Rosenbluth check.  Near the resonant surface the
nonzero outer solution is :math:`\psi = C[1 + a x\ln|x| + b_\pm x + \dots]`,
with :math:`x = r - r_s` and :math:`a = (m/r_s)J_0'(r_s)/F'(r_s)`, so
:math:`\psi'/\psi` holds :math:`a\ln|x|` (symmetric, cancels in the jump) and
:math:`-a^2x\ln^2|x|`, which leaves :math:`-2a^2\epsilon\ln^2\epsilon` in the
jump; the extrapolation basis is therefore
:math:`\{1, \epsilon\ln\epsilon, \epsilon, \epsilon\ln^2\epsilon\}`.

References
----------
Furth, Killeen & Rosenbluth, Phys. Fluids 6, 459 (1963).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from scpn_fusion.core.linear_mhd_1d import FloatArray, WessonCylinder


@dataclass(frozen=True)
class DeltaPrimeFit:
    """Extrapolated :math:`\\Delta'`, the fitted jumps and the least-squares residual."""

    value: float
    excisions: FloatArray
    jumps: FloatArray
    coefficients: FloatArray
    residual: float


def cylinder_delta_prime_fit(
    cylinder: WessonCylinder,
    m: int,
    n: int,
    *,
    excisions: FloatArray | None = None,
) -> DeltaPrimeFit:
    """Return :math:`\\Delta'` of the ideal outer equation by two-sided shooting, with fit details.

    The outer equation :math:`L\\psi-\\kappa_bJ_0'\\psi/F=0` is integrated from
    the axis (:math:`\\psi\\sim r^m`) and from the wall (:math:`\\psi=0`) to
    :math:`r_s\\mp\\epsilon`; the jump of :math:`\\psi'/\\psi` is extrapolated to
    :math:`\\epsilon\\to0` with the basis
    :math:`\\{1,\\epsilon\\ln\\epsilon,\\epsilon,\\epsilon\\ln^2\\epsilon\\}` (module docstring).
    The default excisions span :math:`10^{-4}\\ldots10^{-6}`: on the reference
    case the window :math:`10^{-3}\\ldots10^{-5}` still carries higher-order
    terms (2.7e-5 relative), while :math:`10^{-4}\\ldots10^{-6}` and
    :math:`10^{-5}\\ldots10^{-7}` agree to 3e-7.
    """
    geom = cylinder.geometry(m, n)
    rs = geom.resonant_x
    eps_values = (
        np.geomspace(1e-4, 1e-6, 9)
        if excisions is None
        else np.asarray(excisions, dtype=np.float64)
    )
    r_start = 1e-4
    if (
        eps_values.ndim != 1
        or eps_values.size < 4
        or not np.all(np.isfinite(eps_values))
        or np.any(eps_values <= 0.0)
        or np.unique(eps_values).size != eps_values.size
    ):
        raise ValueError("excisions must hold at least four finite, positive, distinct values")
    if np.any(rs - eps_values <= r_start) or np.any(rs + eps_values >= 1.0):
        raise ValueError(
            "every excision must keep r_s - eps above the axis start and r_s + eps < 1"
        )

    def rhs(r: float, y: FloatArray) -> FloatArray:
        radius = np.array([r])
        f = geom.f_parallel(radius)[0]
        jp = cylinder.current_prime(radius)[0]
        return np.array([y[1], -y[1] / r + (m * m / (r * r)) * y[0] + (m / r) * jp * y[0] / f])

    jumps = []
    for eps in eps_values:
        left = solve_ivp(
            rhs,
            (r_start, rs - eps),
            [r_start**m, m * r_start ** (m - 1)],
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        right = solve_ivp(
            rhs, (1.0, rs + eps), [0.0, -1.0], method="DOP853", rtol=1e-12, atol=1e-14
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            jump = right.y[1, -1] / right.y[0, -1] - left.y[1, -1] / left.y[0, -1]
        if not (left.success and right.success) or not np.isfinite(jump):
            raise RuntimeError(f"outer-equation shooting failed or diverged at excision {eps:g}")
        jumps.append(jump)
    log_eps = np.log(eps_values)
    design = np.stack(
        [np.ones_like(eps_values), eps_values * log_eps, eps_values, eps_values * log_eps**2],
        axis=1,
    )
    jump_array = np.asarray(jumps, dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(design, jump_array, rcond=None)
    residual = float(np.linalg.norm(design @ coefficients - jump_array))
    return DeltaPrimeFit(
        value=float(coefficients[0]),
        excisions=eps_values,
        jumps=jump_array,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        residual=residual,
    )


def cylinder_delta_prime(
    cylinder: WessonCylinder,
    m: int,
    n: int,
    *,
    excisions: FloatArray | None = None,
) -> float:
    """Return :math:`\\Delta'` from :func:`cylinder_delta_prime_fit`."""
    return cylinder_delta_prime_fit(cylinder, m, n, excisions=excisions).value
