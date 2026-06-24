# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Exactness-class reproduction comparator (fleet WS-2)
"""The exactness-class reproduction comparator — FUSION's WS-2 reference implementation.

The fleet honesty model says an in-browser recompute "matches" a committed claim, but its two
pins are in tension for floating-point kernels: H2 (digest equality) and H3 (tolerance-aware).
Float kernels are not bit-reproducible across toolchains (x86 / ARM / WASM-SIMD, FMA contraction,
compiler flags, libm), so a *correct* recompute yields a different digest and a naive digest check
fires a false drift alarm. WS-2 resolves this with a per-claim **exactness class** that decides how
the verifier compares:

* ``bit-exact`` — the recomputed value's content digest must equal the committed digest. For
  integer / fixed-point / genuinely bit-reproducible kernels (the producer asserts toolchain
  independence; CI must cross-check it).
* ``tolerance`` — the recomputed value must match the committed reference value within a band
  (``atol + rtol * |ref|``, NumPy ``allclose`` semantics). The band is static ``{rtol, atol}`` or a
  scalar reduced from a UQ envelope (``reduce_to_scalar(uq)``, supplied by the producer). For float
  kernels — the majority of FUSION and the fleet.
* ``stochastic`` — reproduction needs the committed seed and RNG identity, after which it reduces to
  ``bit-exact`` or ``tolerance``. (This module compares the reduced form; seeding is the caller's.)

This is a pure comparison function over (recomputed, committed, exactness-class) — no I/O, no
signing (signing is WS-1). It returns an honest verdict, including ``unverifiable`` when the
exactness class is absent, never silently defaulting (a default would false-drift floats or mask a
bit-exact regression). The browser ``@anulum/verify`` lib mirrors this verdict-for-verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class ExactnessClass(StrEnum):
    """How a claim's reproduction is to be compared. Declared per claim; never defaulted."""

    BIT_EXACT = "bit-exact"
    TOLERANCE = "tolerance"
    STOCHASTIC = "stochastic"


class ReproVerdict(StrEnum):
    """The verdict of comparing a recomputed value against a committed claim."""

    MATCH = "match"  # bit-exact digest equality held
    WITHIN_TOLERANCE = "within-tolerance"  # float recompute inside the declared band
    DRIFT = "drift"  # recompute disagrees beyond the rule — LOUD, tamper-evident
    UNVERIFIABLE = "unverifiable"  # no exactness class declared — cannot compare honestly


@dataclass(frozen=True)
class ComparisonResult:
    """One reproduction comparison verdict with a human-readable rationale.

    Parameters
    ----------
    verdict
        The :class:`ReproVerdict`.
    detail
        A one-line explanation (the band, the digest mismatch, the absent class).
    """

    verdict: ReproVerdict
    detail: str

    @property
    def reproduced(self) -> bool:
        """Return whether the recompute counts as an honest reproduction.

        ``True`` for a bit-exact match or a within-tolerance float recompute; ``False`` for drift
        or an unverifiable (un-declared) claim.
        """
        return self.verdict in (ReproVerdict.MATCH, ReproVerdict.WITHIN_TOLERANCE)


def compare_bit_exact(recomputed_digest: str, committed_digest: str) -> ComparisonResult:
    """Compare two content digests for a bit-exact claim.

    Parameters
    ----------
    recomputed_digest
        The ``sha256:<hex>`` digest of the in-browser recompute.
    committed_digest
        The committed claim's digest.

    Returns
    -------
    ComparisonResult
        ``MATCH`` on exact digest equality, else ``DRIFT`` (loud, tamper-evident).
    """
    if recomputed_digest == committed_digest:
        return ComparisonResult(ReproVerdict.MATCH, "bit-exact digest equality")
    return ComparisonResult(
        ReproVerdict.DRIFT,
        f"bit-exact digest mismatch: {recomputed_digest!r} != {committed_digest!r}",
    )


def _finite_aware_close(
    recomputed: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    rtol: float,
    atol: float,
) -> bool:
    """Return whether two arrays agree within tolerance, treating NaN≡NaN and ±Inf by sign.

    IEEE ``NaN != NaN`` and ``inf - inf = nan`` would make :func:`numpy.allclose` reject a faithful
    reproduction of a result that legitimately contains non-finite values, so non-finite positions
    are matched structurally (same NaN mask, same signed-infinity mask) and only the finite
    positions are compared within ``atol + rtol * |reference|``.
    """
    if recomputed.shape != reference.shape:
        return False
    nan_a, nan_b = np.isnan(recomputed), np.isnan(reference)
    if not np.array_equal(nan_a, nan_b):
        return False
    posinf_a, posinf_b = np.isposinf(recomputed), np.isposinf(reference)
    neginf_a, neginf_b = np.isneginf(recomputed), np.isneginf(reference)
    if not (np.array_equal(posinf_a, posinf_b) and np.array_equal(neginf_a, neginf_b)):
        return False
    finite = ~(nan_a | posinf_a | neginf_a)
    if not np.any(finite):
        return True  # all positions non-finite and structurally matched
    return bool(
        np.all(
            np.abs(recomputed[finite] - reference[finite])
            <= atol + rtol * np.abs(reference[finite])
        )
    )


def compare_tolerance(
    recomputed: NDArray[np.float64] | float,
    reference: NDArray[np.float64] | float,
    *,
    rtol: float = 1.0e-9,
    atol: float = 0.0,
) -> ComparisonResult:
    """Compare a recomputed float value/array against a committed reference within tolerance.

    Parameters
    ----------
    recomputed
        The in-browser recompute (scalar or array).
    reference
        The committed reference value(s).
    rtol, atol
        The relative and absolute tolerance band (``atol + rtol * |reference|``). For a
        UQ-grounded band, pass ``atol = reduce_to_scalar(uq)`` and ``rtol = 0.0``.

    Returns
    -------
    ComparisonResult
        ``WITHIN_TOLERANCE`` if the recompute is inside the band, else ``DRIFT``.

    Raises
    ------
    ValueError
        If ``rtol`` or ``atol`` is negative (a tolerance band must be non-negative).
    """
    if rtol < 0.0 or atol < 0.0:
        raise ValueError(f"tolerance band must be non-negative, got rtol={rtol!r}, atol={atol!r}")
    a = np.asarray(recomputed, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if _finite_aware_close(a, b, rtol=rtol, atol=atol):
        return ComparisonResult(
            ReproVerdict.WITHIN_TOLERANCE, f"within tolerance (rtol={rtol:g}, atol={atol:g})"
        )
    return ComparisonResult(ReproVerdict.DRIFT, f"beyond tolerance (rtol={rtol:g}, atol={atol:g})")


def reproduce(
    exactness_class: ExactnessClass | None,
    *,
    recomputed_digest: str | None = None,
    committed_digest: str | None = None,
    recomputed_value: NDArray[np.float64] | float | None = None,
    reference_value: NDArray[np.float64] | float | None = None,
    rtol: float = 1.0e-9,
    atol: float = 0.0,
) -> ComparisonResult:
    """Dispatch a reproduction comparison on the claim's exactness class.

    A ``stochastic`` claim must be reduced by the caller (seed the RNG, then pass the reduced
    ``bit-exact`` or ``tolerance`` class with its payload); passing ``STOCHASTIC`` here is itself an
    error because the reduction is the caller's responsibility.

    Parameters
    ----------
    exactness_class
        The declared class, or ``None`` (which yields ``UNVERIFIABLE`` — never a silent default).
    recomputed_digest, committed_digest
        Required for ``bit-exact``.
    recomputed_value, reference_value
        Required for ``tolerance``.
    rtol, atol
        The tolerance band for ``tolerance``.

    Returns
    -------
    ComparisonResult
        The verdict; ``UNVERIFIABLE`` when no class is declared.

    Raises
    ------
    ValueError
        If the class is ``STOCHASTIC`` (must be reduced first), or the payload required for the
        declared class is missing.
    """
    if exactness_class is None:
        return ComparisonResult(
            ReproVerdict.UNVERIFIABLE, "no exactness class declared (absent = loud unverifiable)"
        )
    if exactness_class is ExactnessClass.STOCHASTIC:
        raise ValueError(
            "stochastic claims must be reduced to bit-exact or tolerance by the caller "
            "(seed the RNG first), then dispatched with the reduced class"
        )
    if exactness_class is ExactnessClass.BIT_EXACT:
        if recomputed_digest is None or committed_digest is None:
            raise ValueError("bit-exact reproduction requires recomputed_digest + committed_digest")
        return compare_bit_exact(recomputed_digest, committed_digest)
    if recomputed_value is None or reference_value is None:
        raise ValueError("tolerance reproduction requires recomputed_value + reference_value")
    return compare_tolerance(recomputed_value, reference_value, rtol=rtol, atol=atol)


__all__ = [
    "ComparisonResult",
    "ExactnessClass",
    "ReproVerdict",
    "compare_bit_exact",
    "compare_tolerance",
    "reproduce",
]
