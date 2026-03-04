# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Integrated Transport Runtime Utilities
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np


def thomas_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """O(n) tridiagonal solver (Thomas algorithm)."""
    n = len(d)
    cp = np.empty(n - 1)
    dp = np.empty(n)

    b0 = float(b[0])
    if (not np.isfinite(b0)) or abs(b0) < 1e-30:
        b0 = 1e-30
    cp0 = float(c[0]) / b0
    dp0 = float(d[0]) / b0
    cp[0] = cp0 if np.isfinite(cp0) else 0.0
    dp[0] = dp0 if np.isfinite(dp0) else 0.0

    for i in range(1, n):
        m = b[i] - a[i - 1] * (cp[i - 1] if i - 1 < len(cp) else 0.0)
        if (not np.isfinite(m)) or abs(m) < 1e-30:
            m = 1e-30
        numer = d[i] - a[i - 1] * dp[i - 1]
        if not np.isfinite(numer):
            numer = 0.0
        dp_i = numer / m
        dp[i] = dp_i if np.isfinite(dp_i) else 0.0
        if i < n - 1:
            cp_i = c[i] / m
            cp[i] = cp_i if np.isfinite(cp_i) else 0.0

    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x_i = dp[i] - cp[i] * x[i + 1]
        x[i] = x_i if np.isfinite(x_i) else 0.0
    return x


def explicit_diffusion_rhs(
    *,
    rho: np.ndarray,
    drho: float,
    T: np.ndarray,
    chi: np.ndarray,
) -> np.ndarray:
    """Compute explicit diffusion operator L_h(T) = (1/r) d/dr(r chi dT/dr)."""
    n = len(T)
    Lh = np.zeros(n)
    dr = drho

    r = rho[1:n - 1]
    chi_ip = 0.5 * (chi[1:n - 1] + chi[2:n])
    chi_im = 0.5 * (chi[1:n - 1] + chi[0:n - 2])
    r_ip = r + 0.5 * dr
    r_im = r - 0.5 * dr

    flux_ip = chi_ip * r_ip * (T[2:n] - T[1:n - 1]) / dr
    flux_im = chi_im * r_im * (T[1:n - 1] - T[0:n - 2]) / dr
    Lh[1:n - 1] = (flux_ip - flux_im) / (r * dr)
    return Lh


def build_cn_tridiag(
    *,
    rho: np.ndarray,
    drho: float,
    chi: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build tridiagonal coefficients for Crank-Nicolson LHS."""
    n = len(rho)
    dr = drho
    a = np.zeros(n - 1)
    b = np.ones(n)
    c = np.zeros(n - 1)

    r = rho[1:n - 1]
    chi_ip = 0.5 * (chi[1:n - 1] + chi[2:n])
    chi_im = 0.5 * (chi[1:n - 1] + chi[0:n - 2])
    r_ip = r + 0.5 * dr
    r_im = r - 0.5 * dr

    coeff_ip = chi_ip * r_ip / (r * dr * dr)
    coeff_im = chi_im * r_im / (r * dr * dr)

    b[1:n - 1] = 1.0 + 0.5 * dt * (coeff_ip + coeff_im)
    c[1:n - 1] = -0.5 * dt * coeff_ip
    a[0:n - 2] = -0.5 * dt * coeff_im
    return a, b, c


def sanitize_with_fallback(
    arr: np.ndarray,
    reference: np.ndarray,
    *,
    floor: float | None = None,
    ceil: float | None = None,
) -> tuple[np.ndarray, int]:
    """Replace non-finite entries and enforce optional lower/upper bounds."""
    out = np.asarray(arr, dtype=np.float64).copy()
    ref = np.asarray(reference, dtype=np.float64)
    bad = ~np.isfinite(out)
    recovered = int(np.count_nonzero(bad))
    if recovered > 0:
        out[bad] = ref[bad]
    if floor is not None:
        np.maximum(out, floor, out=out)
    if ceil is not None:
        np.minimum(out, ceil, out=out)
    return out, recovered


__all__ = [
    "thomas_solve",
    "explicit_diffusion_rhs",
    "build_cn_tridiag",
    "sanitize_with_fallback",
]
