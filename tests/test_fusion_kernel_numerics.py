# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FusionKernel Numeric Helper Tests
"""Contract tests for the shared FusionKernel numeric helpers.

Cover ``sanitize_numeric_array`` (non-finite replacement plus finite-overflow
clipping, including a custom cap) and ``stable_rms`` (overflow-free RMS with its
empty and all-zero short circuits). Pure NumPy scalar/array helpers with no
compute hot path and no Rust/Julia/Go counterpart, so no polyglot parity surface
is involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.fusion_kernel_numerics import (
    NUMERIC_SANITIZE_CAP,
    sanitize_numeric_array,
    stable_rms,
)


class TestSanitizeNumericArray:
    """Replace non-finite entries and clip finite values into the cap band."""

    def test_replaces_non_finite_entries(self) -> None:
        """NaN maps to zero and infinities map to the signed cap."""
        out = sanitize_numeric_array(np.array([np.nan, np.inf, -np.inf, 1.5]))
        assert out[0] == 0.0
        assert out[1] == NUMERIC_SANITIZE_CAP
        assert out[2] == -NUMERIC_SANITIZE_CAP
        assert out[3] == pytest.approx(1.5)

    def test_clips_finite_values_beyond_cap(self) -> None:
        """A finite magnitude above the cap is clipped to the signed cap."""
        out = sanitize_numeric_array(np.array([1.0e260, -1.0e260]))
        assert out[0] == NUMERIC_SANITIZE_CAP
        assert out[1] == -NUMERIC_SANITIZE_CAP

    def test_respects_custom_cap(self) -> None:
        """A custom cap bounds both the infinity replacement and clipping."""
        out = sanitize_numeric_array(np.array([np.inf, 10.0, -50.0]), cap=5.0)
        assert out[0] == 5.0
        assert out[1] == 5.0
        assert out[2] == -5.0

    def test_passes_through_in_range_values_as_float64(self) -> None:
        """In-range values are returned unchanged with float64 dtype."""
        out = sanitize_numeric_array(np.array([-1, 0, 2], dtype=np.int64))
        assert out.dtype == np.float64
        assert list(out) == [-1.0, 0.0, 2.0]


class TestStableRMS:
    """Compute an overflow-free RMS with empty and all-zero short circuits."""

    def test_matches_direct_rms_for_moderate_values(self) -> None:
        """The scaled RMS equals the direct RMS for well-scaled inputs."""
        arr = np.array([3.0, 4.0])
        assert stable_rms(arr) == pytest.approx(np.sqrt((9.0 + 16.0) / 2.0))

    def test_empty_array_returns_zero(self) -> None:
        """An empty array has an RMS of exactly zero."""
        assert stable_rms(np.array([])) == 0.0

    def test_all_zero_array_returns_zero(self) -> None:
        """An all-zero array short circuits to zero before scaling."""
        assert stable_rms(np.zeros(4)) == 0.0

    def test_avoids_overflow_for_large_magnitudes(self) -> None:
        """A magnitude whose square overflows float64 still yields a finite RMS."""
        arr = np.array([1.0e200, 1.0e200])
        result = stable_rms(arr)
        assert np.isfinite(result)
        assert result == pytest.approx(1.0e200)

    def test_sanitizes_before_reducing(self) -> None:
        """A non-finite entry is capped before the RMS reduction runs."""
        result = stable_rms(np.array([np.inf]))
        assert result == pytest.approx(NUMERIC_SANITIZE_CAP)
