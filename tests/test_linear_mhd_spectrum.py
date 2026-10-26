# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the verified unstable-eigenvalue search of linear MHD pencils.

Pencils with a known spectrum (diagonal A and B) pin the search logic exactly;
the assembled Harris-sheet problem checks the dense and iterative paths against
the full dense spectrum of a real discretisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp

from scpn_fusion.core.linear_mhd_1d import (
    Discretisation,
    WessonCylinder,
    assemble,
    clustered_grid,
    harris_slab,
)
from scpn_fusion.core.linear_mhd_spectrum import (
    all_eigenvalues,
    eigenvalues_near,
    most_unstable,
    unstable_search,
)

ETA = 1e-4


@dataclass(frozen=True)
class _Pencil:
    """A bare pencil: the search needs only ``a`` and ``b``."""

    a: sp.csc_matrix
    b: sp.csc_matrix


def _diagonal(values: list[float], b_diagonal: list[float] | None = None) -> _Pencil:
    """Return a real generalised problem with a known spectrum (A, B diagonal)."""
    a = sp.diags(np.asarray(values, dtype=np.complex128)).tocsc()
    b_values = (
        np.ones(len(values)) if b_diagonal is None else np.asarray(b_diagonal, dtype=np.float64)
    )
    b = sp.diags(b_values.astype(np.complex128)).tocsc()
    return _Pencil(a=a, b=b)


def _slab_discretisation(ka: float = 0.5, n_side: int = 40) -> Discretisation:
    geom = harris_slab(ka, 20.0)
    nodes = clustered_grid(geom.x_min, geom.x_max, geom.resonant_x, 0.02 * ETA ** (1 / 3), n_side)
    return assemble(geom, nodes, ETA)


HIDDEN_INSIDE = [0.55, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24] + [-float(k) for k in range(1, 14)]
BEYOND_FIRST_DISK = [0.65, 0.01] + [-float(k) for k in range(1, 19)]


@pytest.mark.parametrize("spectrum, expected", [(HIDDEN_INSIDE, 0.55), (BEYOND_FIRST_DISK, 0.65)])
def test_search_finds_the_strongest_root_on_both_paths(
    spectrum: list[float], expected: float
) -> None:
    """Six closer roots, or a root beyond the first disk, do not hide the strongest root."""
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
    assert refused.most_unstable is None
    iterative = unstable_search(disc, dense_limit=0, residual_tol=0.0)
    assert iterative.candidates == () and iterative.rejected >= 1
    singular = unstable_search(_diagonal([0.4, 0.2, -1.0], b_diagonal=[1.0, 0.0, 1.0]))
    assert singular.candidates == pytest.approx((0.4,)) and singular.complete


def test_dense_search_keeps_distinct_close_roots() -> None:
    """Exact distinct roots 1e-9 apart are both returned; the stronger one leads."""
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
    """A non-contracting factor, misordered or non-finite shift bounds are refused."""
    disc = _slab_discretisation(n_side=10)
    with pytest.raises(ValueError, match="factor"):
        most_unstable(disc, factor=1.0)
    with pytest.raises(ValueError, match="sigma_min <= sigma_max <= sigma_ceil"):
        unstable_search(disc, sigma_max=0.3, sigma_ceil=0.1)
    with pytest.raises(ValueError, match="sigma_min must be finite"):
        unstable_search(disc, sigma_min=float("nan"))


def _cylinder_tearing(n_side: int = 10) -> Discretisation:
    """Return the 2/1 tearing problem of the benchmark smoke case (eta = 1e-6)."""
    geom = WessonCylinder(1.2, 1.5, 10.0).geometry(2, 1)
    eta = 1e-6
    nodes = clustered_grid(geom.x_min, geom.x_max, geom.resonant_x, 0.02 * eta ** (1 / 3), n_side)
    return assemble(geom, nodes, eta)


def test_iterative_search_is_reproducible_and_matches_dense_qz() -> None:
    """The refined iterative root matches dense QZ, repeats, and numerical zeros are excluded."""
    disc = _cylinder_tearing()
    dense = unstable_search(disc).most_unstable
    first = unstable_search(disc, dense_limit=0)
    second = unstable_search(disc, dense_limit=0)
    assert dense is not None and first.most_unstable is not None
    assert len(first.candidates) == len(second.candidates) == 1
    assert first.candidates[0] == pytest.approx(second.candidates[0], rel=1e-13)
    assert abs(first.most_unstable - dense) <= 1e-10 * abs(dense)
    assert first.residuals[0] <= 1e-12
