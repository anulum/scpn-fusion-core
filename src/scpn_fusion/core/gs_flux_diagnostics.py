# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Checked Flux Operator and Current Diagnostics
"""Measure the existing centred GS stencil without masking arithmetic failures."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _validate_operator_inputs_np(
    psi: FloatArray,
    R_min: float,
    R_max: float,
    Z_min: float,
    Z_max: float,
) -> tuple[FloatArray, int, int, float, float, FloatArray]:
    """Validate nonempty flux support and the actual endpoint-inclusive grid."""
    psi_arr = np.asarray(psi, dtype=np.float64)
    if psi_arr.ndim != 2:
        raise ValueError("psi must be a two-dimensional array")
    nz, nr = psi_arr.shape
    if nr < 3 or nz < 3:
        raise ValueError("psi shape must be at least 3x3 for centred GS diagnostics")
    if not (np.isfinite(R_min) and np.isfinite(R_max) and 0.0 < R_min < R_max):
        raise ValueError("R_min and R_max must be finite, positive, increasing values")
    if not (np.isfinite(Z_min) and np.isfinite(Z_max) and Z_min < Z_max):
        raise ValueError("Z_min and Z_max must be finite, increasing values")
    if not np.isfinite(psi_arr).all():
        raise ValueError("psi must contain only finite values")
    dr = (R_max - R_min) / (nr - 1)
    dz = (Z_max - Z_min) / (nz - 1)
    if not (np.isfinite(dr) and np.isfinite(dz) and dr > 0.0 and dz > 0.0):
        raise ValueError("operator grid spacing must be finite and positive")
    radius = np.linspace(R_min, R_max, nr, dtype=np.float64)
    height = np.linspace(Z_min, Z_max, nz, dtype=np.float64)
    if not (np.all(np.diff(radius) > 0.0) and np.all(np.diff(height) > 0.0)):
        raise ValueError("operator grid axes have collapsed adjacent nodes")
    return psi_arr, nr, nz, dr, dz, radius


def gs_delta_star_np(
    psi: FloatArray,
    R_min: float,
    R_max: float,
    Z_min: float,
    Z_max: float,
) -> FloatArray:
    r"""Evaluate ``Delta*psi = psi_RR - psi_R/R + psi_ZZ`` on the interior.

    Parameters
    ----------
    psi : FloatArray
        Finite flux matrix with at least three rows and columns.
    R_min, R_max : float
        Finite positive ordered bounds of the radial mesh.
    Z_min, Z_max : float
        Finite ordered bounds of the vertical mesh.

    Returns
    -------
    FloatArray
        The original centred stencil, with zero boundary cells.

    Raises
    ------
    ValueError
        Flux support or actual mesh geometry is invalid.
    RuntimeError
        A stencil coefficient or evaluated value is nonfinite.

    """
    values, nr, nz, dr, dz, radius = _validate_operator_inputs_np(psi, R_min, R_max, Z_min, Z_max)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            dr2, dz2 = dr * dr, dz * dz
            if not (np.isfinite(dr2) and np.isfinite(dz2) and dr2 > 0.0 and dz2 > 0.0):
                raise RuntimeError("operator spacing square is nonfinite or degenerate")
            coefficients = (1.0 / dr2, 1.0 / dz2, 2.0 / dr2 + 2.0 / dz2)
            denominator = 2.0 * dr * radius[None, 1:-1]
            if not np.isfinite(coefficients).all():
                raise RuntimeError("operator coefficient is nonfinite or degenerate")
            np.reciprocal(denominator)
            radial = (values[1:-1, 2:] - 2.0 * values[1:-1, 1:-1] + values[1:-1, :-2]) / dr2
            drift = (values[1:-1, 2:] - values[1:-1, :-2]) / denominator
            vertical = (values[2:, 1:-1] - 2.0 * values[1:-1, 1:-1] + values[:-2, 1:-1]) / dz2
            interior = radial - drift + vertical
    except (FloatingPointError, OverflowError, ZeroDivisionError) as failure:
        raise RuntimeError("operator arithmetic is nonfinite") from failure
    result = np.zeros_like(values)
    result[1 : nz - 1, 1 : nr - 1] = interior
    return result


def gs_toroidal_current_density_np(
    psi: FloatArray,
    R_min: float,
    R_max: float,
    Z_min: float,
    Z_max: float,
    mu0: float = 4e-7 * np.pi,
) -> FloatArray:
    r"""Return the signed current implied by ``Delta*psi = -mu0 R J_phi``.

    Parameters
    ----------
    psi : FloatArray
        Finite flux matrix on the declared rectangular mesh.
    R_min, R_max, Z_min, Z_max : float
        Bounds used by the original centred operator.
    mu0 : float, optional
        Finite positive permeability of the explicitly configured model.

    Returns
    -------
    FloatArray
        Finite signed current density with zero boundary cells.

    Raises
    ------
    ValueError
        Permeability, flux support or mesh geometry is invalid.
    RuntimeError
        The operator, permeability-radius product or current is nonfinite.

    """
    if not (np.isfinite(mu0) and mu0 > 0.0):
        raise ValueError("mu0 must be finite and positive")
    values, nr, nz, _, _, radius = _validate_operator_inputs_np(psi, R_min, R_max, Z_min, Z_max)
    delta = gs_delta_star_np(values, R_min, R_max, Z_min, Z_max)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            denominator = mu0 * radius[None, 1:-1]
            if np.any(denominator == 0.0):
                raise RuntimeError("current denominator is nonfinite or zero")
            interior = -delta[1:-1, 1:-1] / denominator
    except (FloatingPointError, OverflowError, ZeroDivisionError) as failure:
        raise RuntimeError("current arithmetic is nonfinite") from failure
    result = np.zeros_like(values)
    result[1 : nz - 1, 1 : nr - 1] = interior
    return result


def gs_total_toroidal_current_np(
    psi: FloatArray,
    R_min: float,
    R_max: float,
    Z_min: float,
    Z_max: float,
    mu0: float = 4e-7 * np.pi,
) -> float:
    """Integrate signed interior current with the original rectangular weights.

    Parameters
    ----------
    psi : FloatArray
        Finite flux matrix on the declared mesh.
    R_min, R_max, Z_min, Z_max : float
        Ordered finite mesh bounds.
    mu0 : float, optional
        Finite positive configured model permeability.

    Returns
    -------
    float
        Finite signed toroidal current using the original reduction order.

    Raises
    ------
    ValueError
        Permeability, flux support or mesh geometry is invalid.
    RuntimeError
        Density or integrated-current arithmetic is nonfinite.

    """
    _, _, _, dr, dz, _ = _validate_operator_inputs_np(psi, R_min, R_max, Z_min, Z_max)
    density = gs_toroidal_current_density_np(psi, R_min, R_max, Z_min, Z_max, mu0)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            total = float(np.sum(density[1:-1, 1:-1]) * dr * dz)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as failure:
        raise RuntimeError("integrated current arithmetic is nonfinite") from failure
    return total
