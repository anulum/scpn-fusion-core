# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Runtime Utilities
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

    r = rho[1 : n - 1]
    chi_ip = 0.5 * (chi[1 : n - 1] + chi[2:n])
    chi_im = 0.5 * (chi[1 : n - 1] + chi[0 : n - 2])
    r_ip = r + 0.5 * dr
    r_im = r - 0.5 * dr

    flux_ip = chi_ip * r_ip * (T[2:n] - T[1 : n - 1]) / dr
    flux_im = chi_im * r_im * (T[1 : n - 1] - T[0 : n - 2]) / dr
    Lh[1 : n - 1] = (flux_ip - flux_im) / (r * dr)
    return Lh


class _CNGridCache:
    """Precomputed grid-constant factors for Crank-Nicolson tridiag assembly.

    ``geo_ip[i]`` = r_{i+1/2} / (r_i * dr²), ``geo_im[i]`` = r_{i-1/2} / (r_i * dr²),
    both defined over interior indices i=1..n-2.
    """

    __slots__ = ("geo_ip", "geo_im", "_key")

    def __init__(self, rho: np.ndarray, drho: float) -> None:
        n = len(rho)
        r = rho[1 : n - 1]
        r_ip = r + 0.5 * drho
        r_im = r - 0.5 * drho
        inv_r_dr2 = 1.0 / (r * drho * drho)
        self.geo_ip = r_ip * inv_r_dr2
        self.geo_im = r_im * inv_r_dr2
        self._key = (n, drho)

    def matches(self, rho: np.ndarray, drho: float) -> bool:
        return self._key == (len(rho), drho)


_cn_grid_cache: _CNGridCache | None = None


def build_cn_tridiag(
    *,
    rho: np.ndarray,
    drho: float,
    chi: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build tridiagonal coefficients for Crank-Nicolson LHS."""
    global _cn_grid_cache  # noqa: PLW0603
    if _cn_grid_cache is None or not _cn_grid_cache.matches(rho, drho):
        _cn_grid_cache = _CNGridCache(rho, drho)
    cache = _cn_grid_cache

    n = len(rho)
    a = np.zeros(n - 1)
    b = np.ones(n)
    c = np.zeros(n - 1)

    chi_ip = 0.5 * (chi[1 : n - 1] + chi[2:n])
    chi_im = 0.5 * (chi[1 : n - 1] + chi[0 : n - 2])

    coeff_ip = chi_ip * cache.geo_ip
    coeff_im = chi_im * cache.geo_im

    b[1 : n - 1] = 1.0 + 0.5 * dt * (coeff_ip + coeff_im)
    c[1 : n - 1] = -0.5 * dt * coeff_ip
    a[0 : n - 2] = -0.5 * dt * coeff_im
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
