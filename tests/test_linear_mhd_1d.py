# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real-solver tests for the 1-D linear resistive reduced-MHD eigenvalue solver.

The full acceptance runs (eta down to 1e-8) take minutes and live in
``validation/linear_mhd_reduced_model_acceptance.py``; these tests use coarse grids
and eta = 1e-4 so they stay fast while exercising every path for real.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.typing import NDArray

from scpn_fusion.core.linear_mhd_1d import (
    Discretisation,
    WessonCylinder,
    all_eigenvalues,
    assemble,
    clustered_grid,
    cylinder_delta_prime,
    cylinder_delta_prime_fit,
    eigenvalues_near,
    growth_rate,
    harris_slab,
    most_unstable,
    unstable_search,
)
from scpn_fusion.core.tearing_mode_theory import HarrisSheetTearingModel

ETA = 1e-4


def _slab_discretisation(ka: float = 0.5, n_side: int = 40) -> Discretisation:
    geom = harris_slab(ka, 20.0)
    nodes = clustered_grid(geom.x_min, geom.x_max, geom.resonant_x, 0.02 * ETA ** (1 / 3), n_side)
    return assemble(geom, nodes, ETA)


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


def _diagonal(values: list[float], b_diagonal: list[float] | None = None) -> Discretisation:
    """Return a real generalised problem with a known spectrum (A, B diagonal)."""
    a = sp.diags(np.asarray(values, dtype=np.complex128)).tocsc()
    b_values = (
        np.ones(len(values)) if b_diagonal is None else np.asarray(b_diagonal, dtype=np.float64)
    )
    b = sp.diags(b_values.astype(np.complex128)).tocsc()
    return Discretisation(nodes=np.linspace(0.0, 1.0, 3), a=a, b=b)


HIDDEN_INSIDE = [0.55, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24] + [-float(k) for k in range(1, 14)]
BEYOND_FIRST_DISK = [0.65, 0.01] + [-float(k) for k in range(1, 19)]


@pytest.mark.parametrize("spectrum, expected", [(HIDDEN_INSIDE, 0.55), (BEYOND_FIRST_DISK, 0.65)])
def test_search_finds_the_reviewer_counter_examples_on_both_paths(
    spectrum: list[float], expected: float
) -> None:
    """Six closer roots, or a root beyond the first disk, no longer hide the strongest root."""
    dense = unstable_search(_diagonal(spectrum))
    assert dense.method == "dense-qz" and dense.complete
    assert dense.most_unstable == pytest.approx(expected, abs=1e-12)
    iterative = unstable_search(_diagonal(spectrum), dense_limit=0)
    assert iterative.method == "shift-invert" and not iterative.complete
    assert "disks" in iterative.domain
    assert iterative.most_unstable == pytest.approx(expected, abs=1e-10)
    assert max(iterative.residuals) <= 1e-8
    assert len(set(np.round(iterative.candidates, 8))) == len(iterative.candidates)


def test_shift_invert_matches_the_dense_spectrum() -> None:
    """The iterative search, the dense search and the full dense spectrum agree on the slab."""
    disc = _slab_discretisation()
    dense = all_eigenvalues(disc)
    found = most_unstable(disc, dense_limit=0)
    assert found is not None
    assert abs(found.real - dense.real.max()) <= 1e-8 * dense.real.max()
    assert most_unstable(disc) == pytest.approx(found, rel=1e-8)
    near = eigenvalues_near(disc, found.real * 1.01, count=1)
    assert np.min(np.abs(near - found)) < 1e-8


def test_rejected_roots_void_the_completeness_claim() -> None:
    """A root that fails verification is counted, and the dense result is then not complete."""
    disc = _slab_discretisation(n_side=6)
    accepted = unstable_search(disc)
    assert accepted.complete and accepted.rejected == 0
    assert accepted.most_unstable is not None and accepted.most_unstable.real > 0.0
    refused = unstable_search(disc, residual_tol=0.0)
    assert refused.candidates == () and refused.rejected >= 1 and not refused.complete
    iterative = unstable_search(disc, dense_limit=0, residual_tol=0.0)
    assert iterative.candidates == () and iterative.rejected >= 1
    singular = unstable_search(_diagonal([0.4, 0.2, -1.0], b_diagonal=[1.0, 0.0, 1.0]))
    assert singular.candidates == pytest.approx((0.4,)) and singular.complete


def test_dense_search_keeps_distinct_close_roots() -> None:
    """Exact distinct roots 1e-9 apart are both returned; the stronger one leads (re-review R2)."""
    search = unstable_search(_diagonal([0.55, 0.550000001, -1.0]))
    assert len(search.candidates) == 2
    assert search.most_unstable == pytest.approx(0.550000001, abs=1e-15)


def test_shift_ladder_is_reported_exactly() -> None:
    """The reported shifts are the ladder sigma_max * 4^k inside [sigma_min, sigma_ceil]."""
    search = unstable_search(
        _diagonal(HIDDEN_INSIDE), dense_limit=0, sigma_min=1e-3, sigma_ceil=10.0
    )
    expected = sorted((0.3 * 4.0**k for k in range(-4, 3)), reverse=True)
    assert search.shifts == pytest.approx(tuple(expected))
    assert search.merge_tol == 1e-10 and "sigma_ceil 10" in search.domain
    with pytest.raises(ValueError, match="merge_tol"):
        unstable_search(_diagonal(HIDDEN_INSIDE), dense_limit=0, merge_tol=float("nan"))


def test_unconverged_arpack_is_tolerated_and_reported_as_no_root() -> None:
    """With one Arnoldi iteration nothing converges; the search falls through to None."""
    disc = _slab_discretisation(n_side=20)
    assert (
        most_unstable(disc, sigma_max=0.3, sigma_min=0.1, sigma_ceil=0.3, dense_limit=0, maxiter=1)
        is None
    )


def test_search_rejects_inconsistent_shift_settings() -> None:
    """A non-contracting factor or misordered shift bounds are refused."""
    disc = _slab_discretisation(n_side=10)
    with pytest.raises(ValueError, match="factor"):
        most_unstable(disc, factor=1.0)
    with pytest.raises(ValueError, match="sigma_min <= sigma_max <= sigma_ceil"):
        unstable_search(disc, sigma_max=0.3, sigma_ceil=0.1)


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


def test_cylinder_delta_prime_is_stable_across_excision_windows() -> None:
    """With the eps ln^2 eps term two windows agree to 1e-6 (9.69474)."""
    cyl = WessonCylinder(1.2, 1.5, 10.0)
    default = cylinder_delta_prime_fit(cyl, 2, 1)
    narrow = cylinder_delta_prime(cyl, 2, 1, excisions=np.geomspace(1e-5, 1e-7, 9))
    assert default.value == pytest.approx(narrow, rel=1e-6)
    assert default.value == pytest.approx(9.69474, abs=1e-4)
    assert default.coefficients.size == 4 and default.residual < 1e-4
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
