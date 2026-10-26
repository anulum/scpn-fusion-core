# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Unstable-Eigenvalue Search for Linear MHD Pencils
r"""Verified search for unstable eigenvalues of a generalised problem :math:`\lambda Bx=Ax`.

Used by the linear reduced-MHD solvers (:mod:`scpn_fusion.core.linear_mhd_1d`,
:mod:`scpn_fusion.core.linear_mhd_2d`), but independent of how the sparse
pencil was assembled.

:func:`unstable_search` solves problems with at most ``dense_limit`` unknowns
completely by dense QZ.  Larger problems use shift-and-invert Arnoldi over real
positive shifts :math:`\sigma_{\max}f^{k}`, lowered to ``sigma_min`` and raised
to ``sigma_ceil``; every root with positive real part inside its disk
:math:`|z-\sigma|<\sigma` is collected over all shifts (the ideal Alfvén
continuum lies on or left of the imaginary axis, so such a root is a discrete
unstable mode, not a continuum point).  Every candidate is verified by its
eigenvector residual.  The iterative path cannot prove that no stronger root
exists outside the searched disks, so its result is marked incomplete and
"no root" means "none found in the searched domain", not stability.

Arnoldi starts from a fixed pseudo-random vector, so the iterative path is
reproducible up to the summation order of the linear-algebra libraries.  A root seen from a distant shift is only as accurate as the
shift-and-invert transform allows there (up to about :math:`10^{-6}` relative
against the dense spectrum was observed for the cylindrical tearing root of the
benchmark smoke case), so each iterative candidate is first refined by
inverse iteration at a shift :math:`\lambda(1+10^{-3})` next to it and only
then verified; like the dense path it keeps only roots with
:math:`\mathrm{Re}\,\lambda` above ``sigma_min``.  Repeat sightings of one
root then agree far inside ``merge_tol``, and the sighting with the smallest
residual is kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from scipy.linalg import eig

ComplexArray = NDArray[np.complex128]

_START_SEED = 0
_REFINE_OFFSET = 1e-3
_REFINE_STEPS = 4


class GeneralisedEigenproblem(Protocol):
    """Sparse pencil :math:`(A, B)` of the problem :math:`\\lambda Bx=Ax`."""

    @property
    def a(self) -> sp.csc_matrix:
        """Return the operator ``A``."""

    @property
    def b(self) -> sp.csc_matrix:
        """Return the mass operator ``B``."""


def _positive(name: str, value: float) -> float:
    """Return ``value`` as a float or reject a non-finite or non-positive input."""
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _start_vector(size: int) -> ComplexArray:
    """Return the fixed pseudo-random Arnoldi start vector of length ``size``."""
    rng = np.random.default_rng(_START_SEED)
    return np.asarray(
        rng.standard_normal(size) + 1j * rng.standard_normal(size), dtype=np.complex128
    )


def eigenvalues_near(
    problem: GeneralisedEigenproblem, sigma: complex, count: int = 6, *, maxiter: int = 2000
) -> ComplexArray:
    """Return ``count`` eigenvalues nearest ``sigma`` by shift-and-invert Arnoldi."""
    factor = spla.splu((problem.a - sigma * problem.b).tocsc())
    operator = spla.LinearOperator(
        problem.a.shape, matvec=lambda v: factor.solve(problem.b @ v), dtype=np.complex128
    )
    mu = spla.eigs(
        operator,
        k=count,
        which="LM",
        return_eigenvectors=False,
        tol=1e-10,
        ncv=40,
        maxiter=maxiter,
        v0=_start_vector(problem.a.shape[0]),
    )
    return np.asarray(sigma + 1.0 / mu, dtype=np.complex128)


def all_eigenvalues(problem: GeneralisedEigenproblem) -> ComplexArray:
    """Return the finite part of the full dense spectrum (small problems only)."""
    values = eig(problem.a.toarray(), problem.b.toarray(), right=False)
    return np.asarray(values[np.isfinite(values)], dtype=np.complex128)


@dataclass(frozen=True)
class SpectralSearch:
    """Verified unstable eigenvalues and the domain that was searched for them.

    ``rejected`` counts eligible roots that failed verification (non-finite or
    residual above the bound); ``complete`` is true only for a dense search
    that rejected none, so an empty result with ``rejected > 0`` is a
    numerical-qualification failure, not an empty unstable set.  ``shifts``
    lists the shifts actually used (empty on the dense path); ``merge_tol`` is
    the relative resolution below which repeat sightings were merged
    (iterative path only).
    """

    candidates: tuple[complex, ...]
    residuals: tuple[float, ...]
    method: str
    domain: str
    complete: bool
    rejected: int = 0
    shifts: tuple[float, ...] = ()
    merge_tol: float = 0.0

    @property
    def most_unstable(self) -> complex | None:
        """Return the candidate with the largest real part, or ``None`` if none was found."""
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda z: z.real)


def _relative_residual(
    problem: GeneralisedEigenproblem,
    lam: complex,
    x: ComplexArray,
    norm_a: float,
    norm_b: float,
) -> float:
    """Return ``|Ax - lam Bx| / ((|A|_1 + |lam| |B|_1) |x|)``."""
    vec = np.asarray(x, dtype=np.complex128)
    defect = problem.a @ vec - lam * (problem.b @ vec)
    scale = (norm_a + abs(lam) * norm_b) * float(np.linalg.norm(vec))
    return float(np.linalg.norm(defect) / scale)


def _pairs_near(
    problem: GeneralisedEigenproblem, sigma: float, count: int, maxiter: int
) -> tuple[ComplexArray, ComplexArray]:
    """Return eigenvalues and eigenvectors near ``sigma`` (converged ones only on ARPACK failure)."""
    factor = spla.splu((problem.a - sigma * problem.b).tocsc())
    operator = spla.LinearOperator(
        problem.a.shape, matvec=lambda v: factor.solve(problem.b @ v), dtype=np.complex128
    )
    try:
        mu, vectors = spla.eigs(
            operator,
            k=count,
            which="LM",
            tol=1e-10,
            ncv=40,
            maxiter=maxiter,
            v0=_start_vector(problem.a.shape[0]),
        )
    except spla.ArpackNoConvergence as exc:
        mu = np.asarray(exc.eigenvalues, dtype=np.complex128)
        vectors = np.asarray(exc.eigenvectors, dtype=np.complex128)
    return np.asarray(sigma + 1.0 / mu, dtype=np.complex128), np.asarray(
        vectors, dtype=np.complex128
    )


def _refine(
    problem: GeneralisedEigenproblem, lam: complex, x: ComplexArray
) -> tuple[complex, ComplexArray]:
    """Refine an eigenpair by inverse iteration at the shift ``lam (1 + 1e-3)``.

    The shifted operator has ``lam`` as its dominant eigenvalue by a factor of
    about ``1e3`` over any root farther than ``|lam|`` away, so a few steps
    reduce the eigenvector error far below the residual bound.  The eigenvalue
    is the Rayleigh quotient of the shift-inverted operator.
    """
    sigma = lam + _REFINE_OFFSET * abs(lam)
    factor = spla.splu((problem.a - sigma * problem.b).tocsc())
    vec = np.asarray(x, dtype=np.complex128)
    for _ in range(_REFINE_STEPS):
        vec = np.asarray(factor.solve(problem.b @ vec), dtype=np.complex128)
        vec /= np.linalg.norm(vec)
    image = np.asarray(factor.solve(problem.b @ vec), dtype=np.complex128)
    mu = complex(np.vdot(vec, image))
    return sigma + 1.0 / mu, vec


def unstable_search(
    problem: GeneralisedEigenproblem,
    *,
    sigma_max: float = 0.3,
    sigma_min: float = 1e-12,
    sigma_ceil: float = 10.0,
    factor: float = 4.0,
    count: int = 6,
    dense_limit: int = 2000,
    residual_tol: float = 1e-8,
    merge_tol: float = 1e-10,
    maxiter: int = 2000,
) -> SpectralSearch:
    """Return every verified unstable eigenvalue found, with the searched domain and completeness."""
    top = _positive("sigma_max", sigma_max)
    floor = _positive("sigma_min", sigma_min)
    ceil = _positive("sigma_ceil", sigma_ceil)
    if factor <= 1.0:
        raise ValueError("factor must be > 1")
    if floor > top or ceil < top:
        raise ValueError("sigma_min <= sigma_max <= sigma_ceil is required")
    if not math.isfinite(merge_tol) or merge_tol < 0.0:
        raise ValueError("merge_tol must be finite and >= 0")
    norm_a = float(spla.norm(problem.a, 1))
    norm_b = float(spla.norm(problem.b, 1))
    found: list[tuple[complex, float]] = []
    rejected = 0

    def verified(lam: complex, x: ComplexArray) -> float | None:
        nonlocal rejected
        res = _relative_residual(problem, lam, x, norm_a, norm_b)
        if not res <= residual_tol:  # also rejects a NaN residual
            rejected += 1
            return None
        return res

    if problem.a.shape[0] <= dense_limit:
        values, vectors = eig(problem.a.toarray(), problem.b.toarray())
        for k, lam in enumerate(values):
            if np.isfinite(lam) and lam.real > floor:
                res = verified(complex(lam), vectors[:, k])
                if res is not None:
                    found.append((complex(lam), res))
        found.sort(key=lambda item: -item[0].real)
        return SpectralSearch(
            tuple(z for z, _ in found),
            tuple(r for _, r in found),
            "dense-qz",
            f"all finite generalised eigenvalues with Re > {floor:g}",
            rejected == 0,
            rejected,
        )
    shifts = [top]
    while shifts[-1] / factor >= floor:
        shifts.append(shifts[-1] / factor)
    while shifts[0] * factor <= ceil:
        shifts.insert(0, shifts[0] * factor)
    for sigma in shifts:
        values, vectors = _pairs_near(problem, sigma, count, maxiter)
        for k, seen in enumerate(values):
            if not (seen.real > 0.0 and abs(seen - sigma) < sigma):
                continue
            lam, vec = _refine(problem, complex(seen), vectors[:, k])
            if not lam.real > floor:  # a numerical zero, as on the dense path
                continue
            res = verified(lam, vec)
            if res is None:
                continue
            same = [i for i, (z, _) in enumerate(found) if abs(lam - z) <= merge_tol * abs(z)]
            if not same:
                found.append((lam, res))
            elif res < found[same[0]][1]:
                found[same[0]] = (lam, res)
    found.sort(key=lambda item: -item[0].real)
    return SpectralSearch(
        tuple(z for z, _ in found),
        tuple(r for _, r in found),
        "shift-invert",
        f"union of the disks |z - s| < s over the listed shifts (sigma_min {floor:g},"
        f" sigma_ceil {ceil:g}, factor {factor:g}), {count} roots per shift",
        False,
        rejected,
        tuple(shifts),
        merge_tol,
    )


def most_unstable(
    problem: GeneralisedEigenproblem,
    *,
    sigma_max: float = 0.3,
    sigma_min: float = 1e-12,
    sigma_ceil: float = 10.0,
    factor: float = 4.0,
    dense_limit: int = 2000,
    maxiter: int = 2000,
) -> complex | None:
    """Return the most unstable verified eigenvalue found by :func:`unstable_search`, or ``None``.

    ``None`` means no unstable root was found in the searched domain; on the
    iterative path that does not establish stability (see :class:`SpectralSearch`).
    """
    return unstable_search(
        problem,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        sigma_ceil=sigma_ceil,
        factor=factor,
        dense_limit=dense_limit,
        maxiter=maxiter,
    ).most_unstable
