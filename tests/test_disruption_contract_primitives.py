# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Contract Primitive Tests
# ----------------------------------------------------------------------
"""Tests for reusable disruption-contract primitive validators/utilities."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.disruption_contract_primitives import (
    gaussian_interval,
    require_1d_array,
    require_fraction,
    require_int,
    require_positive_float,
    synthetic_disruption_signal,
)


def test_require_int_rejects_bool() -> None:
    with pytest.raises(ValueError, match="integer"):
        require_int("episodes", True, 1)


def test_require_fraction_bounds() -> None:
    assert require_fraction("risk", 0.25) == 0.25
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        require_fraction("risk", 1.5)


def test_require_positive_float_rejects_zero() -> None:
    with pytest.raises(ValueError, match="> 0"):
        require_positive_float("dt", 0.0)


def test_require_1d_array_expected_size_contract() -> None:
    arr = require_1d_array("signal", [1.0, 2.0, 3.0], expected_size=3)
    assert arr.shape == (3,)
    with pytest.raises(ValueError, match="must have 4 samples"):
        require_1d_array("signal", [1.0, 2.0, 3.0], expected_size=4)


def test_gaussian_interval_clamps_bounds() -> None:
    lo, hi = gaussian_interval(mean=0.5, sigma=0.2, lower_bound=0.0, upper_bound=1.0)
    assert 0.0 <= lo <= hi <= 1.0


def test_synthetic_disruption_signal_is_deterministic_for_seed() -> None:
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    signal_a, toroidal_a = synthetic_disruption_signal(rng=rng_a, disturbance=0.6, window=128)
    signal_b, toroidal_b = synthetic_disruption_signal(rng=rng_b, disturbance=0.6, window=128)
    np.testing.assert_allclose(signal_a, signal_b)
    assert toroidal_a == toroidal_b
