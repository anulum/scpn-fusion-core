# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Exactness-class reproduction comparator tests (WS-2)
"""Tests for the exactness-class reproduction comparator (fleet WS-2)."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.studio.exactness import (
    ComparisonResult,
    ExactnessClass,
    ReproVerdict,
    compare_bit_exact,
    compare_tolerance,
    reproduce,
)


def test_bit_exact_match() -> None:
    """Equal digests are a bit-exact match and count as reproduced."""
    result = compare_bit_exact("sha256:abc", "sha256:abc")
    assert result.verdict is ReproVerdict.MATCH
    assert result.reproduced is True


def test_bit_exact_mismatch_is_loud_drift() -> None:
    """A digest mismatch is loud drift and is not a reproduction."""
    result = compare_bit_exact("sha256:aaa", "sha256:bbb")
    assert result.verdict is ReproVerdict.DRIFT
    assert result.reproduced is False
    assert "mismatch" in result.detail


def test_tolerance_scalar_within_band() -> None:
    """A float recompute one ULP off matches within a relative tolerance."""
    result = compare_tolerance(1.0 + 1e-12, 1.0, rtol=1e-9, atol=0.0)
    assert result.verdict is ReproVerdict.WITHIN_TOLERANCE
    assert result.reproduced is True


def test_tolerance_array_within_band() -> None:
    """An array recompute inside the band is within tolerance."""
    ref = np.array([1.0, 2.0, 3.0])
    got = ref + 1e-11
    assert compare_tolerance(got, ref, rtol=1e-9).verdict is ReproVerdict.WITHIN_TOLERANCE


def test_tolerance_beyond_band_is_drift() -> None:
    """A recompute outside the band is drift."""
    result = compare_tolerance(1.5, 1.0, rtol=1e-9, atol=0.0)
    assert result.verdict is ReproVerdict.DRIFT
    assert "beyond tolerance" in result.detail


def test_tolerance_uq_grounded_absolute_band() -> None:
    """A UQ-grounded band passes atol = reduce_to_scalar(uq) with rtol = 0."""
    # reduce_to_scalar(uq) -> 0.05 (e.g. a k-sigma envelope half-width)
    assert compare_tolerance(0.93, 0.90, rtol=0.0, atol=0.05).reproduced is True
    assert compare_tolerance(0.99, 0.90, rtol=0.0, atol=0.05).verdict is ReproVerdict.DRIFT


def test_tolerance_rejects_negative_band() -> None:
    """A negative tolerance band is rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        compare_tolerance(1.0, 1.0, rtol=-1e-9)
    with pytest.raises(ValueError, match="non-negative"):
        compare_tolerance(1.0, 1.0, atol=-1.0)


def test_tolerance_shape_mismatch_is_drift() -> None:
    """An array shape change is drift before any value comparison."""
    result = compare_tolerance(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert result.verdict is ReproVerdict.DRIFT


def test_tolerance_matches_structural_nan_and_inf() -> None:
    """NaN positions match NaN, signed infinities match by sign; finite parts within band."""
    ref = np.array([np.nan, np.inf, -np.inf, 1.0])
    got = np.array([np.nan, np.inf, -np.inf, 1.0 + 1e-12])
    assert compare_tolerance(got, ref, rtol=1e-9).verdict is ReproVerdict.WITHIN_TOLERANCE


def test_tolerance_mismatched_nan_mask_is_drift() -> None:
    """A NaN where the reference is finite (or vice versa) is drift, not within tolerance."""
    ref = np.array([np.nan, 1.0])
    got = np.array([0.0, 1.0])
    assert compare_tolerance(got, ref).verdict is ReproVerdict.DRIFT


def test_tolerance_mismatched_signed_infinity_is_drift() -> None:
    """+Inf reproduced where the reference had -Inf is drift."""
    ref = np.array([np.inf, 1.0])
    got = np.array([-np.inf, 1.0])
    assert compare_tolerance(got, ref).verdict is ReproVerdict.DRIFT


def test_tolerance_all_non_finite_structurally_matches() -> None:
    """An all-non-finite result reproduces when the non-finite structure matches exactly."""
    ref = np.array([np.nan, np.inf])
    got = np.array([np.nan, np.inf])
    assert compare_tolerance(got, ref).verdict is ReproVerdict.WITHIN_TOLERANCE


def test_reproduce_dispatches_bit_exact() -> None:
    """``reproduce`` routes a bit-exact claim to the digest comparison."""
    result = reproduce(
        ExactnessClass.BIT_EXACT, recomputed_digest="sha256:x", committed_digest="sha256:x"
    )
    assert result.verdict is ReproVerdict.MATCH


def test_reproduce_dispatches_tolerance() -> None:
    """``reproduce`` routes a tolerance claim to the band comparison."""
    result = reproduce(
        ExactnessClass.TOLERANCE, recomputed_value=1.0 + 1e-12, reference_value=1.0, rtol=1e-9
    )
    assert result.verdict is ReproVerdict.WITHIN_TOLERANCE


def test_reproduce_absent_class_is_unverifiable() -> None:
    """No declared exactness class yields a loud unverifiable verdict, never a silent default."""
    result = reproduce(None)
    assert result.verdict is ReproVerdict.UNVERIFIABLE
    assert result.reproduced is False


def test_reproduce_stochastic_must_be_reduced_first() -> None:
    """A stochastic class cannot be compared directly — the caller must seed and reduce it."""
    with pytest.raises(ValueError, match="reduced to bit-exact or tolerance"):
        reproduce(ExactnessClass.STOCHASTIC)


def test_reproduce_bit_exact_requires_both_digests() -> None:
    """A bit-exact dispatch without both digests is an error."""
    with pytest.raises(ValueError, match="requires recomputed_digest"):
        reproduce(ExactnessClass.BIT_EXACT, recomputed_digest="sha256:x")


def test_reproduce_tolerance_requires_both_values() -> None:
    """A tolerance dispatch without both values is an error."""
    with pytest.raises(ValueError, match="requires recomputed_value"):
        reproduce(ExactnessClass.TOLERANCE, recomputed_value=1.0)


def test_exactness_class_and_verdict_wire_values() -> None:
    """The enum string values are the locked wire vocabulary."""
    assert ExactnessClass.BIT_EXACT == "bit-exact"
    assert ExactnessClass.TOLERANCE == "tolerance"
    assert ExactnessClass.STOCHASTIC == "stochastic"
    assert ReproVerdict.WITHIN_TOLERANCE == "within-tolerance"
    assert ComparisonResult(ReproVerdict.MATCH, "x").reproduced is True
