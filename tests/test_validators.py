# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Shared Validators Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core._validators import (
    require_1d_array,
    require_bounded_float,
    require_finite_array,
    require_finite_float,
    require_fraction,
    require_int,
    require_non_negative_float,
    require_positive_float,
    require_range,
)


# ── require_finite_float ────────────────────────────────────────────


class TestRequireFiniteFloat:
    def test_accepts_int(self):
        assert require_finite_float("x", 3) == 3.0

    def test_accepts_float(self):
        assert require_finite_float("x", 2.5) == 2.5

    def test_accepts_negative(self):
        assert require_finite_float("x", -1.5) == -1.5

    def test_accepts_zero(self):
        assert require_finite_float("x", 0) == 0.0

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            require_finite_float("x", float("nan"))

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="finite"):
            require_finite_float("x", float("inf"))

    def test_rejects_neg_inf(self):
        with pytest.raises(ValueError, match="finite"):
            require_finite_float("x", float("-inf"))

    def test_accepts_numpy_scalar(self):
        assert require_finite_float("x", np.float64(1.5)) == pytest.approx(1.5)

    def test_rejects_string(self):
        with pytest.raises((ValueError, TypeError)):
            require_finite_float("x", "hello")


# ── require_positive_float ──────────────────────────────────────────


class TestRequirePositiveFloat:
    def test_accepts_positive(self):
        assert require_positive_float("x", 1.0) == 1.0

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="> 0"):
            require_positive_float("x", 0.0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="> 0"):
            require_positive_float("x", -1.0)

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            require_positive_float("x", float("nan"))


# ── require_non_negative_float ──────────────────────────────────────


class TestRequireNonNegativeFloat:
    def test_accepts_zero(self):
        assert require_non_negative_float("x", 0.0) == 0.0

    def test_accepts_positive(self):
        assert require_non_negative_float("x", 5.0) == 5.0

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match=">= 0"):
            require_non_negative_float("x", -0.01)


# ── require_int ─────────────────────────────────────────────────────


class TestRequireInt:
    def test_accepts_int(self):
        assert require_int("n", 5) == 5

    def test_accepts_numpy_int(self):
        assert require_int("n", np.int64(3)) == 3

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="integer"):
            require_int("n", True)

    def test_rejects_float(self):
        with pytest.raises(ValueError, match="integer"):
            require_int("n", 3.5)

    def test_minimum_enforced(self):
        assert require_int("n", 5, minimum=3) == 5
        with pytest.raises(ValueError, match=">= 3"):
            require_int("n", 2, minimum=3)

    def test_minimum_boundary(self):
        assert require_int("n", 0, minimum=0) == 0

    def test_no_minimum_bool_error_message(self):
        with pytest.raises(ValueError, match="must be an integer."):
            require_int("n", True)

    def test_minimum_bool_error_message(self):
        with pytest.raises(ValueError, match="must be an integer >= 1"):
            require_int("n", False, minimum=1)


# ── require_fraction ────────────────────────────────────────────────


class TestRequireFraction:
    def test_accepts_zero(self):
        assert require_fraction("f", 0.0) == 0.0

    def test_accepts_one(self):
        assert require_fraction("f", 1.0) == 1.0

    def test_accepts_half(self):
        assert require_fraction("f", 0.5) == 0.5

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            require_fraction("f", -0.1)

    def test_rejects_above_one(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            require_fraction("f", 1.01)


# ── require_range ───────────────────────────────────────────────────


class TestRequireRange:
    def test_valid_range(self):
        assert require_range("r", (1.0, 2.0)) == (1.0, 2.0)

    def test_rejects_inverted(self):
        with pytest.raises(ValueError, match="low < high"):
            require_range("r", (2.0, 1.0))

    def test_rejects_equal(self):
        with pytest.raises(ValueError, match="low < high"):
            require_range("r", (1.0, 1.0))

    def test_min_allowed(self):
        assert require_range("r", (0.5, 1.5), min_allowed=0.0) == (0.5, 1.5)
        with pytest.raises(ValueError, match=">="):
            require_range("r", (-0.1, 1.0), min_allowed=0.0)

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            require_range("r", (float("nan"), 1.0))


# ── require_bounded_float ───────────────────────────────────────────


class TestRequireBoundedFloat:
    def test_no_bounds(self):
        assert require_bounded_float("x", 100.0) == 100.0

    def test_low_inclusive(self):
        assert require_bounded_float("x", 0.0, low=0.0) == 0.0
        with pytest.raises(ValueError, match=">="):
            require_bounded_float("x", -0.01, low=0.0)

    def test_low_exclusive(self):
        with pytest.raises(ValueError, match=">"):
            require_bounded_float("x", 0.0, low=0.0, low_exclusive=True)
        assert require_bounded_float("x", 0.01, low=0.0, low_exclusive=True) == pytest.approx(0.01)

    def test_high_inclusive(self):
        assert require_bounded_float("x", 1.0, high=1.0) == 1.0
        with pytest.raises(ValueError, match="<="):
            require_bounded_float("x", 1.01, high=1.0)

    def test_high_exclusive(self):
        with pytest.raises(ValueError, match="<"):
            require_bounded_float("x", 1.0, high=1.0, high_exclusive=True)

    def test_combined_bounds(self):
        val = require_bounded_float("x", 0.5, low=0.0, high=1.0)
        assert val == 0.5

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            require_bounded_float("x", float("nan"))

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="finite"):
            require_bounded_float("x", float("inf"))


# ── require_finite_array ────────────────────────────────────────────


class TestRequireFiniteArray:
    def test_accepts_list(self):
        arr = require_finite_array("a", [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])
        assert arr.dtype == np.float64

    def test_rejects_nan_in_array(self):
        with pytest.raises(ValueError, match="finite"):
            require_finite_array("a", [1.0, float("nan")])

    def test_ndim_constraint(self):
        require_finite_array("a", [1.0, 2.0], ndim=1)
        with pytest.raises(ValueError, match="1D"):
            require_finite_array("a", [[1.0, 2.0]], ndim=1)

    def test_shape_constraint(self):
        require_finite_array("a", [[1.0, 2.0], [3.0, 4.0]], shape=(2, 2))
        with pytest.raises(ValueError, match="shape"):
            require_finite_array("a", [[1.0, 2.0]], shape=(2, 2))

    def test_accepts_numpy_array(self):
        arr = require_finite_array("a", np.array([1.0, 2.0]))
        assert arr.dtype == np.float64

    def test_empty_array(self):
        arr = require_finite_array("a", [])
        assert arr.size == 0


# ── require_1d_array ────────────────────────────────────────────────


class TestRequire1dArray:
    def test_valid_array(self):
        arr = require_1d_array("a", [1.0, 2.0, 3.0])
        assert arr.shape == (3,)

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="1D"):
            require_1d_array("a", [[1.0, 2.0]])

    def test_minimum_size(self):
        require_1d_array("a", [1.0, 2.0], minimum_size=2)
        with pytest.raises(ValueError, match="at least 3"):
            require_1d_array("a", [1.0, 2.0], minimum_size=3)

    def test_expected_size(self):
        require_1d_array("a", [1.0, 2.0, 3.0], expected_size=3)
        with pytest.raises(ValueError, match="must have 4 samples"):
            require_1d_array("a", [1.0, 2.0, 3.0], expected_size=4)

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            require_1d_array("a", [1.0, float("inf")])

    def test_default_minimum_size_is_1(self):
        with pytest.raises(ValueError, match="at least 1"):
            require_1d_array("a", [])
