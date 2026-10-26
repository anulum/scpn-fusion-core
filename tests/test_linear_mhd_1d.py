# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real-solver tests for the 1-D linear resistive reduced-MHD geometry, assembly and growth rate.

The full acceptance runs (eta down to 1e-8) take minutes and live in
``validation/linear_mhd_reduced_model_acceptance.py``; these tests use coarse grids
and eta = 1e-4 so they stay fast while exercising every path for real.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.linear_mhd_1d import (
    WessonCylinder,
    assemble,
    clustered_grid,
    growth_rate,
    harris_slab,
)
from scpn_fusion.core.tearing_mode_theory import HarrisSheetTearingModel

ETA = 1e-4


def test_harris_tearing_is_unstable_and_close_to_fkr() -> None:
    """At eta = 1e-4 the growth rate is within 10 % of FKR (0.92 measured, not yet asymptotic)."""
    gamma = growth_rate(harris_slab(0.5, 20.0), ETA, n_side=40)
    reference = HarrisSheetTearingModel(
        sheet_half_width_m=1.0, wave_number_per_m=0.5, finite_width_coefficient=1.0
    ).fkr_growth_rate_per_s(alfven_time_s=1.0, lundquist_number=1.0 / ETA)
    assert 0.90 < gamma / reference < 1.0


def test_negative_delta_prime_sheet_has_no_unstable_root() -> None:
    """ka = 1.2 gives Delta' < 0; the search returns 0.0 above its floor."""
    assert growth_rate(harris_slab(1.2, 20.0), ETA, n_side=20, sigma_min=1e-3) == 0.0


def test_cylinder_resistive_kink_and_tearing_are_unstable() -> None:
    """m = 1 (axis value only) and m = 2 (value and slope fixed on axis) both grow."""
    kink = WessonCylinder(0.9, 1.0, 10.0).geometry(1, 1)
    tearing = WessonCylinder(1.2, 1.5, 10.0).geometry(2, 1)
    assert kink.axis_value_only is True and tearing.axis_value_only is False
    assert growth_rate(tearing, ETA, n_side=40) > 0.0
    # At eta = 1e-4 the kink layer is far from its asymptote (0.31 of eta^(1/3) F'^(2/3)
    # measured; 0.97 at eta = 1e-8 in the acceptance run), so only the sign and
    # the bound by the asymptotic rate are asserted here.
    gamma_kink = growth_rate(kink, ETA, n_side=40)
    reference_kink = ETA ** (1 / 3) * abs(
        kink.f_parallel_prime(np.array([kink.resonant_x]))[0]
    ) ** (2 / 3)
    assert 0.0 < gamma_kink < reference_kink


def test_cylinder_profile_quantities() -> None:
    """q(0) = q0, q(1) = q0 (nu + 1), and the on-axis limits are continuous."""
    cyl = WessonCylinder(1.2, 1.5, 10.0)
    r = np.array([0.0, 1e-7, 0.5, 1.0])
    q = cyl.q(r)
    assert q[0] == pytest.approx(1.2) and q[-1] == pytest.approx(1.2 * 2.5)
    assert cyl.b_theta(r)[1] == pytest.approx(0.5 * cyl.j0 * 1e-7)
    assert cyl.b_theta_prime(r)[0] == pytest.approx(0.5 * cyl.j0)
    assert cyl.current(np.array([1.0]))[0] == 0.0
    with pytest.raises(ValueError, match="not inside"):
        cyl.resonant_radius(5, 1)
    with pytest.raises(ValueError, match="m and n"):
        cyl.geometry(0, 1)
    with pytest.raises(ValueError, match="q0"):
        WessonCylinder(0.0, 1.0, 10.0)


def test_grid_and_input_validation() -> None:
    """Grid construction covers the uniform and boundary-centred cases and rejects bad input."""
    uniform = clustered_grid(0.0, 1.0, 0.5, 0.1, 10)
    assert uniform[0] == 0.0 and uniform[-1] == 1.0 and np.all(np.diff(uniform) > 0)
    edge = clustered_grid(0.0, 1.0, 0.0, 1e-3, 20)
    assert edge[0] == 0.0 and edge[1] == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="n_side"):
        clustered_grid(0.0, 1.0, 0.5, 0.1, 1)
    with pytest.raises(ValueError, match="x_c"):
        clustered_grid(0.0, 1.0, 2.0, 0.1, 10)
    with pytest.raises(ValueError, match="h_min"):
        clustered_grid(0.0, 1.0, 0.5, 0.0, 10)
    with pytest.raises(ValueError, match="nodes"):
        assemble(harris_slab(0.5, 20.0), np.array([0.0, 1.0]), ETA)
    with pytest.raises(ValueError, match="eta"):
        assemble(harris_slab(0.5, 20.0), np.array([-1.0, 0.0, 1.0]), float("nan"))
    with pytest.raises(ValueError, match="ka"):
        harris_slab(-1.0, 20.0)
