# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real-solver tests for the coupled-harmonic reduced-MHD solver on a shaped cross-section.

The shaped-section runs (eta down to 1e-9, minutes to hours) are not part of this suite;
these tests use coarse grids and eta = 1e-4 to exercise every path.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.linear_mhd_1d import (
    WessonCylinder,
    clustered_grid,
    growth_rate,
    most_unstable,
)
from scpn_fusion.core.linear_mhd_2d import (
    ShapedCylinder,
    assemble_coupled,
    coupled_growth_rate,
    harmonic_fractions,
)

ETA = 1e-4


def _shift(r: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.1 * (1.0 - r**2)


def _shift_prime(r: NDArray[np.float64]) -> NDArray[np.float64]:
    return -0.2 * r


def _shaped(profile: WessonCylinder) -> ShapedCylinder:
    return ShapedCylinder(
        q=profile.q,
        major_radius=10.0,
        elongation=1.6,
        triangularity=0.3,
        shift=_shift,
        shift_prime=_shift_prime,
    )


def test_circular_single_harmonic_equals_the_one_dimensional_solver() -> None:
    """With kappa = 1, delta = 0 and one harmonic the 2-D operator reduces to the 1-D one."""
    profile = WessonCylinder(1.2, 1.5, 10.0)
    rs = profile.resonant_radius(2, 1)
    gamma_2d = coupled_growth_rate(
        ShapedCylinder(q=profile.q, major_radius=10.0), ETA, 1, [2], resonant_r=rs, n_side=30
    )
    gamma_1d = growth_rate(profile.geometry(2, 1), ETA, n_side=30)
    assert gamma_2d == pytest.approx(gamma_1d, rel=1e-6)


def test_straight_field_line_jacobian_is_a_flux_function_and_matches_the_area() -> None:
    """The (r, theta*) Jacobian is constant on each surface and 2 pi J = dA/dr."""
    cyl = _shaped(WessonCylinder(1.02, 1.5, 10.0))
    radii = np.array([0.2, 0.7])
    jac = cyl.metric(radii)["jac"]
    assert np.all(np.ptp(jac, axis=1) <= 1e-12 * jac.mean(axis=1))

    def area(r: float) -> float:
        tg = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        ang = tg + 0.3 * np.sin(tg)
        x = _shift(np.array(r)) + r * np.cos(ang)
        y = 1.6 * r * np.sin(tg)
        x_t = -r * np.sin(ang) * (1.0 + 0.3 * np.cos(tg))
        y_t = 1.6 * r * np.cos(tg)
        return float(np.pi * np.mean(x * y_t - y * x_t))

    for r, j in zip(radii, jac.mean(axis=1), strict=True):
        d_area = (area(r + 1e-4) - area(r - 1e-4)) / 2e-4
        assert 2.0 * np.pi * j == pytest.approx(d_area, rel=1e-9)


def test_shaped_tearing_is_unstable_and_led_by_its_resonant_harmonic() -> None:
    """Operator/geometry experiment: the shaped base state is refused unless flagged.

    With the flag the 2/1 tearing grows and m = 2 carries most of the nodal psi.
    """
    profile = WessonCylinder(1.02, 1.5, 10.0)
    rs = profile.resonant_radius(2, 1)
    nodes = clustered_grid(0.0, 1.0, rs, 0.02 * ETA ** (1 / 3), 30)
    with pytest.raises(ValueError, match="not force balanced"):
        assemble_coupled(_shaped(profile), nodes, ETA, 1, [1, 2, 3])
    disc = assemble_coupled(_shaped(profile), nodes, ETA, 1, [1, 2, 3], allow_unbalanced_base=True)
    lam = most_unstable(disc)
    assert lam is not None and lam.real > 0.0
    shares = harmonic_fractions(disc, lam)
    assert set(shares) == {1, 2, 3}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert max(shares, key=shares.__getitem__) == 2


def test_non_resonant_case_reports_zero_growth() -> None:
    """With q > 2 everywhere the 2/1 harmonic has no resonant surface and does not grow."""
    profile = WessonCylinder(2.5, 1.0, 10.0)
    cyl = ShapedCylinder(q=profile.q, major_radius=10.0)
    assert coupled_growth_rate(cyl, ETA, 1, [2], resonant_r=0.5, n_side=20, sigma_min=1e-3) == 0.0


def test_input_validation() -> None:
    """Shape, discretisation and harmonic inputs are checked."""
    q = WessonCylinder(1.2, 1.5, 10.0).q
    with pytest.raises(ValueError, match="major_radius"):
        ShapedCylinder(q=q, major_radius=0.0)
    with pytest.raises(ValueError, match="elongation"):
        ShapedCylinder(q=q, major_radius=10.0, elongation=0.0)
    with pytest.raises(ValueError, match="triangularity"):
        ShapedCylinder(q=q, major_radius=10.0, triangularity=1.0)
    with pytest.raises(ValueError, match="together"):
        ShapedCylinder(q=q, major_radius=10.0, shift=_shift)
    with pytest.raises(ValueError, match="n_theta"):
        ShapedCylinder(q=q, major_radius=10.0, n_theta=4)
    with pytest.raises(ValueError, match="finite"):
        ShapedCylinder(q=q, major_radius=float("nan"))
    overlapping = ShapedCylinder(
        q=q, major_radius=10.0, shift=lambda r: -2.0 * r, shift_prime=lambda r: -2.0 + 0.0 * r
    )
    with pytest.raises(ValueError, match="Jacobian"):
        overlapping.metric(np.array([0.5]))
    cyl = ShapedCylinder(q=q, major_radius=10.0)
    nodes = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="eta"):
        assemble_coupled(cyl, nodes, 0.0, 1, [2])
    with pytest.raises(ValueError, match="m_values"):
        assemble_coupled(cyl, nodes, ETA, 1, [0, 2])
    with pytest.raises(ValueError, match="m_values"):
        assemble_coupled(cyl, nodes, ETA, 1, [2, 2])
    with pytest.raises(ValueError, match="nodes"):
        assemble_coupled(cyl, np.linspace(0.1, 1.0, 5), ETA, 1, [2])


def test_equilibrium_gate_separates_balanced_and_unbalanced_base_states() -> None:
    """J0 varies along shaped surfaces (O(1)) but not along circles (round-off only)."""
    profile = WessonCylinder(1.02, 1.5, 10.0)
    radii = np.array([0.25, 0.5, 0.9])
    assert ShapedCylinder(q=profile.q, major_radius=10.0).equilibrium_residual(radii) < 1e-8
    assert _shaped(profile).equilibrium_residual(radii) > 0.1


def test_near_axis_metric_is_the_circular_one_and_the_axis_is_refused() -> None:
    """The adaptive step keeps r - h > 0 near the axis; r <= 0 is outside the domain."""
    cyl = ShapedCylinder(q=lambda r: np.ones_like(r), major_radius=10.0)
    radii = np.array([5e-6, 1e-3, 0.5])
    metric = cyl.metric(radii)
    assert np.allclose(metric["jac"], radii[:, None], rtol=1e-6, atol=0.0)
    assert np.allclose(metric["grr"], 1.0, rtol=1e-6)
    assert np.allclose(metric["gtt"] * radii[:, None] ** 2, 1.0, rtol=1e-6)
    assert np.all(np.isfinite(cyl.current(radii)))
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError, match="radii must be finite and > 0"):
            cyl.metric(np.array([bad]))


def test_non_finite_base_state_is_refused_even_when_flagged() -> None:
    """A NaN safety factor cannot pass the gate, with or without the override (re-review R3)."""
    cyl = ShapedCylinder(q=lambda r: np.full_like(r, np.nan), major_radius=10.0)
    nodes = np.linspace(0.0, 1.0, 5)
    for flag in (False, True):
        with pytest.raises(ValueError, match="non-finite"):
            assemble_coupled(cyl, nodes, ETA, 1, [2], allow_unbalanced_base=flag)
    with pytest.raises(ValueError, match="non-finite"):
        cyl.equilibrium_residual(np.array([0.5]))
