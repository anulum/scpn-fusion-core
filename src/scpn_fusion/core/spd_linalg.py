# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Symmetric Positive-Definite Solves
"""Factorised solves for symmetric positive-definite (SPD) matrices.

Innovation covariances, Tikhonov normal matrices, inductance matrices and
moment Gram matrices are SPD by construction. Solving with them through a
Cholesky factor is cheaper and more accurate than forming an explicit inverse.

Contract
--------
* The input must be square, finite and symmetric within
  ``1e2 * eps * ||A||_F``; it is then symmetrised.
* The matrix is scaled by its diagonal, ``A_s = D^-1 A D^-1`` with
  ``D = sqrt(diag A)``, so the SPD test and the conditioning record do not
  depend on units.
* The gate is Cholesky success. No eigenvalue floor is imposed. The LAPACK
  ``pocon`` reciprocal condition estimate of ``A_s`` is recorded; a
  non-finite or zero estimate is refused.
* Every refusal raises :class:`NonSPDMatrixError`; nothing falls back silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.linalg.lapack import dpocon

from scpn_fusion.core.fusion_kernel_numerics import FloatArray
from scpn_fusion.exceptions import FusionCoreError

_EPS = float(np.finfo(np.float64).eps)
SYMMETRY_TOLERANCE_FACTOR = 1.0e2
"""Symmetry and PSD tolerances are this factor times eps times the matrix norm."""


class NonSPDMatrixError(FusionCoreError, ValueError):
    """A matrix required to be symmetric positive (semi-)definite is not."""


def _square_finite(a: Any, name: str) -> FloatArray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise NonSPDMatrixError(f"{name} must be a non-empty square matrix.")
    if not np.all(np.isfinite(arr)):
        raise NonSPDMatrixError(f"{name} must contain finite values only.")
    return arr


def symmetrize(a: Any, *, name: str = "matrix") -> FloatArray:
    """Return ``(A + A^T) / 2`` after checking ``A`` is symmetric within tolerance.

    Raises
    ------
    NonSPDMatrixError
        If ``A`` is not square and finite, or its asymmetry exceeds
        ``1e2 * eps * ||A||_F``.
    """
    arr = _square_finite(a, name)
    norm = float(np.linalg.norm(arr))
    asym = float(np.linalg.norm(arr - arr.T))
    if asym > SYMMETRY_TOLERANCE_FACTOR * _EPS * norm:
        raise NonSPDMatrixError(f"{name} is not symmetric (||A - A^T||_F = {asym:.3e}).")
    return np.asarray(0.5 * (arr + arr.T), dtype=np.float64)


def require_psd(a: Any, *, name: str = "matrix") -> FloatArray:
    """Return the symmetrised matrix after checking it is positive semi-definite.

    The smallest eigenvalue may fall below zero by at most
    ``1e2 * eps * ||A||_2`` (rounding), never more.
    """
    sym = symmetrize(a, name=name)
    eigenvalues = np.linalg.eigvalsh(sym)
    scale = float(np.max(np.abs(eigenvalues)))
    if float(eigenvalues[0]) < -SYMMETRY_TOLERANCE_FACTOR * _EPS * scale:
        raise NonSPDMatrixError(
            f"{name} is not positive semi-definite (min eigenvalue {eigenvalues[0]:.3e})."
        )
    return sym


@dataclass(frozen=True)
class SPDFactor:
    """Cholesky factor of a diagonally scaled SPD matrix.

    Attributes
    ----------
    lower_factor : FloatArray
        Packed output of :func:`scipy.linalg.cho_factor` (lower) for ``A_s``.
    scale : FloatArray
        ``sqrt(diag A)``; ``A = diag(scale) A_s diag(scale)``.
    rcond : float
        LAPACK ``pocon`` estimate of the reciprocal 1-norm condition of ``A_s``.
    matrix : FloatArray
        The symmetrised matrix ``A``, used for one step of iterative refinement.
    """

    lower_factor: FloatArray
    scale: FloatArray
    rcond: float
    matrix: FloatArray

    @property
    def order(self) -> int:
        """Matrix order ``n``."""
        return int(self.scale.shape[0])

    def _factor_solve(self, rhs: FloatArray) -> FloatArray:
        scale = self.scale if rhs.ndim == 1 else self.scale[:, None]
        return np.asarray(
            cho_solve((self.lower_factor, True), rhs / scale) / scale, dtype=np.float64
        )

    def solve(self, b: Any) -> FloatArray:
        """Return ``X`` with ``A X = B`` for a vector or matrix right-hand side.

        One step of iterative refinement, ``X += A^-1 (B - A X)``, brings the
        forward error to the level of an explicit inverse while keeping the
        backward stability of the Cholesky solve.
        """
        rhs = np.asarray(b, dtype=np.float64)
        if rhs.shape[0] != self.order:
            raise ValueError(f"right-hand side has {rhs.shape[0]} rows, expected {self.order}.")
        x = self._factor_solve(rhs)
        return np.asarray(x + self._factor_solve(rhs - self.matrix @ x), dtype=np.float64)

    def inverse(self) -> FloatArray:
        """Return ``A^-1`` (symmetric); prefer :meth:`solve` when only products are needed."""
        inv = self.solve(np.eye(self.order))
        return np.asarray(0.5 * (inv + inv.T), dtype=np.float64)


def _usable_rcond(rcond: float, info: int, name: str) -> float:
    """Return ``rcond`` if LAPACK succeeded and it is finite and positive."""
    if info != 0 or not np.isfinite(rcond) or rcond <= 0.0:
        raise NonSPDMatrixError(f"{name} is numerically singular (rcond estimate {rcond!r}).")
    return rcond


def factor_spd(a: Any, *, name: str = "matrix") -> SPDFactor:
    """Factor an SPD matrix under the module contract.

    Raises
    ------
    NonSPDMatrixError
        If the matrix is not square, finite and symmetric, has a non-positive
        diagonal entry, fails Cholesky, or has a non-finite or zero ``rcond``.
    """
    sym = symmetrize(a, name=name)
    diag = np.diag(sym)
    if np.any(diag <= 0.0):
        raise NonSPDMatrixError(f"{name} has a non-positive diagonal entry.")
    scale = np.sqrt(diag)
    scaled = sym / np.outer(scale, scale)
    try:
        lower, _ = cho_factor(scaled, lower=True, check_finite=False)
    except LinAlgError as exc:
        raise NonSPDMatrixError(f"{name} is not positive definite (Cholesky failed).") from exc
    anorm = float(np.max(np.sum(np.abs(scaled), axis=0)))
    rcond_estimate, info = dpocon(lower, anorm, uplo="L")
    rcond = _usable_rcond(float(rcond_estimate), int(info), name)
    return SPDFactor(
        lower_factor=np.asarray(lower, dtype=np.float64), scale=scale, rcond=rcond, matrix=sym
    )


def spd_solve(a: Any, b: Any, *, name: str = "matrix") -> FloatArray:
    """Return ``X`` with ``A X = B`` for SPD ``A``; see :func:`factor_spd`."""
    return factor_spd(a, name=name).solve(b)


def joseph_update(p: Any, k: Any, h: Any, r: Any) -> FloatArray:
    """Return the Joseph-form covariance ``(I - K H) P (I - K H)^T + K R K^T``.

    The result is symmetrised. Unlike ``(I - K H) P`` it stays positive
    semi-definite for any gain, including a gain perturbed by rounding.
    """
    pm = np.asarray(p, dtype=np.float64)
    km = np.asarray(k, dtype=np.float64)
    hm = np.asarray(h, dtype=np.float64)
    rm = np.asarray(r, dtype=np.float64)
    a = np.eye(pm.shape[0]) - km @ hm
    updated = a @ pm @ a.T + km @ rm @ km.T
    return np.asarray(0.5 * (updated + updated.T), dtype=np.float64)


__all__ = [
    "NonSPDMatrixError",
    "SPDFactor",
    "SYMMETRY_TOLERANCE_FACTOR",
    "factor_spd",
    "joseph_update",
    "require_psd",
    "spd_solve",
    "symmetrize",
]
