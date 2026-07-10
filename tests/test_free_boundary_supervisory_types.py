# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct guard tests for the free-boundary supervisory type helpers.

Covers the vector/mask normalisation validators and the safety-margin
estimator's shape and finiteness guards, plus a nominal margin estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control._free_boundary_supervisory_types import (
    FreeBoundarySafetyMargins,
    _normalize_mask,
    _normalize_vector,
    estimate_free_boundary_safety_margins,
)


def _margins(**overrides: object) -> FreeBoundarySafetyMargins:
    kwargs: dict[str, object] = {
        "corrected_state": np.array([6.0, 0.0, 5.02, -3.48]),
        "target_state": np.array([6.0, 0.0, 5.02, -3.48]),
        "bias_hat": np.array([0.01, 0.01, 0.01, 0.01]),
        "coil_currents": np.array([0.5, -0.5, 0.4, -0.4]),
        "target_ip_ma": 8.0,
        "q95_floor": 4.0,
        "beta_n_ceiling": 2.0,
        "disruption_risk_ceiling": 0.7,
    }
    kwargs.update(overrides)
    return estimate_free_boundary_safety_margins(**kwargs)  # type: ignore[arg-type]


class TestNormalizeVector:
    """The finite-vector normaliser validates length and finiteness."""

    def test_none_returns_default_fill(self) -> None:
        """A ``None`` value expands to a default-filled vector."""
        assert np.array_equal(_normalize_vector(None, length=3, name="v"), np.zeros(3))

    def test_wrong_length_raises(self) -> None:
        """A vector of the wrong length is rejected."""
        with pytest.raises(ValueError, match="must have length 3"):
            _normalize_vector([1.0, 2.0], length=3, name="v")

    def test_non_finite_raises(self) -> None:
        """A non-finite vector element is rejected."""
        with pytest.raises(ValueError, match="must contain finite values"):
            _normalize_vector([1.0, float("nan")], length=2, name="v")


class TestNormalizeMask:
    """The boolean-mask normaliser validates length and finiteness."""

    def test_none_returns_false_mask(self) -> None:
        """A ``None`` value expands to an all-false mask."""
        assert not _normalize_mask(None, length=2, name="m").any()

    def test_truthy_values_become_bools(self) -> None:
        """Non-zero entries map to ``True`` and zeros to ``False``."""
        assert list(_normalize_mask([1.0, 0.0, 2.0], length=3, name="m")) == [True, False, True]

    def test_wrong_length_raises(self) -> None:
        """A mask of the wrong length is rejected."""
        with pytest.raises(ValueError, match="must have length 3"):
            _normalize_mask([1.0, 0.0], length=3, name="m")

    def test_non_finite_raises(self) -> None:
        """A non-finite mask element is rejected."""
        with pytest.raises(ValueError, match="must contain finite values"):
            _normalize_mask([1.0, float("nan")], length=2, name="m")


class TestEstimateSafetyMargins:
    """Shape and finiteness guards of the safety-margin estimator."""

    def test_nominal_estimate_is_finite(self) -> None:
        """A well-formed call returns finite safety margins."""
        margins = _margins()
        assert isinstance(margins, FreeBoundarySafetyMargins)
        assert np.isfinite(margins.q95)
        assert np.isfinite(margins.beta_n)

    def test_wrong_geometry_size_raises(self) -> None:
        """A corrected or target state that is not a 4-vector is rejected."""
        with pytest.raises(ValueError, match="must be finite 4-vectors"):
            _margins(corrected_state=np.array([1.0, 2.0, 3.0]))

    def test_wrong_bias_or_empty_currents_raises(self) -> None:
        """A non-4-vector bias or empty coil-current array is rejected."""
        with pytest.raises(ValueError, match="bias_hat must be a finite 4-vector"):
            _margins(coil_currents=np.array([]))

    def test_non_finite_inputs_raise(self) -> None:
        """A non-finite geometry input is rejected."""
        with pytest.raises(ValueError, match="safety-margin inputs must be finite"):
            _margins(corrected_state=np.array([6.0, 0.0, 5.02, float("nan")]))

    @pytest.mark.parametrize("history", [[], [0.5, float("inf")]])
    def test_invalid_risk_history_raises(self, history: list[float]) -> None:
        """An empty or non-finite risk-signal history is rejected."""
        with pytest.raises(ValueError, match="risk_signal_history must contain finite"):
            _margins(risk_signal_history=history)
