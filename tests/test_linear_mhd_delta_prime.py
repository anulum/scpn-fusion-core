# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the cylindrical tearing index computed by outer-region shooting."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.linear_mhd_1d import WessonCylinder
from scpn_fusion.core.linear_mhd_delta_prime import cylinder_delta_prime, cylinder_delta_prime_fit


def test_cylinder_delta_prime_is_stable_across_excision_windows() -> None:
    """With the eps ln^2 eps term two windows agree to 1e-6 (9.69474)."""
    cyl = WessonCylinder(1.2, 1.5, 10.0)
    default = cylinder_delta_prime_fit(cyl, 2, 1)
    narrow = cylinder_delta_prime(cyl, 2, 1, excisions=np.geomspace(1e-5, 1e-7, 9))
    assert default.value == pytest.approx(narrow, rel=1e-6)
    assert default.value == pytest.approx(9.69474, abs=1e-4)
    assert default.coefficients.size == 4 and default.residual < 1e-4
    assert default.jumps.shape == default.excisions.shape == (9,)


def test_cylinder_delta_prime_refuses_invalid_excisions() -> None:
    """Too few, non-finite or repeated excisions, or ones reaching the axis start or wall, are refused."""
    cyl = WessonCylinder(1.2, 1.5, 10.0)
    rs = cyl.resonant_radius(2, 1)
    for bad in (
        np.array([1e-3, 1e-4, 1e-5]),
        np.array([1e-3, 1e-4, 1e-5, float("nan")]),
        np.array([1e-3, 1e-4, 1e-5, 1e-5]),
    ):
        with pytest.raises(ValueError, match="finite, positive, distinct"):
            cylinder_delta_prime(cyl, 2, 1, excisions=bad)
    for reach in (rs, 1.0 - rs):
        with pytest.raises(ValueError, match="axis start"):
            cylinder_delta_prime(cyl, 2, 1, excisions=np.array([reach, 1e-3, 1e-4, 1e-5]))


class _SteepEdgeCylinder(WessonCylinder):
    """A cylinder whose current gradient jumps to 1e300 outside r = 0.99 (a stiff edge input)."""

    def current_prime(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the Wesson gradient inside r = 0.99 and 1e300 outside."""
        rr = np.asarray(r, dtype=np.float64)
        return np.asarray(np.where(rr > 0.99, 1e300, super().current_prime(rr)), dtype=np.float64)


def test_cylinder_delta_prime_refuses_a_failed_shooting() -> None:
    """An edge too stiff to integrate makes the wall-side shooting fail: refused, not extrapolated."""
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(RuntimeError, match="failed or diverged"),
    ):
        cylinder_delta_prime(_SteepEdgeCylinder(1.2, 1.5, 10.0), 2, 1)
