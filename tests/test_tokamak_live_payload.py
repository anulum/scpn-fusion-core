# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tokamak Live Payload Tests
"""Tests for tokamak live payload coercion helpers."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import pytest

from scpn_fusion.io.tokamak_live_payload import to_scalar, to_trace

FloatArray: TypeAlias = NDArray[np.float64]


def test_to_scalar_returns_last_finite_value_from_payload() -> None:
    """Scalar coercion returns the last finite value from array payloads."""
    payload: FloatArray = np.array([[np.nan, 1.25], [2.5, np.inf]], dtype=np.float64)

    assert to_scalar(payload) == 2.5


def test_to_scalar_rejects_empty_payload() -> None:
    """Scalar coercion fails closed on empty node payloads."""
    with pytest.raises(ValueError, match="Empty MDSplus node payload"):
        to_scalar(np.array([], dtype=np.float64))


def test_to_scalar_rejects_payload_without_finite_values() -> None:
    """Scalar coercion fails closed when no finite values are available."""
    with pytest.raises(ValueError, match="No finite values"):
        to_scalar([np.nan, np.inf, -np.inf])


def test_to_trace_repeats_scalar_payload_before_resampling() -> None:
    """Trace coercion expands scalar payloads before calling the resampler."""
    captured: list[FloatArray] = []

    def resample(trace: FloatArray, points: int) -> FloatArray:
        """Record scalar-expanded traces and return the requested point count."""
        captured.append(trace.copy())
        return np.full(points, trace[0], dtype=np.float64)

    trace = to_trace(3.5, 4, resample_1d=resample)

    np.testing.assert_array_equal(captured[0], np.full(8, 3.5, dtype=np.float64))
    np.testing.assert_array_equal(trace, np.full(4, 3.5, dtype=np.float64))


def test_to_trace_flattens_filters_and_resamples_payload() -> None:
    """Trace coercion flattens arrays and drops non-finite samples before resampling."""
    captured: list[FloatArray] = []

    def resample(trace: FloatArray, points: int) -> FloatArray:
        """Record finite traces and return a deterministic prefix."""
        captured.append(trace.copy())
        return trace[:points]

    trace = to_trace([[1.0, np.nan], [2.0, 3.0]], 2, resample_1d=resample)

    np.testing.assert_array_equal(captured[0], np.array([1.0, 2.0, 3.0], dtype=np.float64))
    np.testing.assert_array_equal(trace, np.array([1.0, 2.0], dtype=np.float64))


def test_to_trace_rejects_payloads_with_too_few_finite_values() -> None:
    """Trace coercion requires at least two finite samples."""
    with pytest.raises(ValueError, match="at least 2 finite values"):
        to_trace([np.nan, 1.0], 4, resample_1d=lambda trace, points: trace[:points])
