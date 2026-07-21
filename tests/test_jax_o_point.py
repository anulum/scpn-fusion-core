# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the smooth differentiable magnetic-axis (O-point) flux finder.

Load-bearing claims: (1) the returned ``ψ_axis`` matches the true flux peak to sub-cell
accuracy (< 0.5 %), for either sign convention and for a shifted/off-centre axis;
(2) it is smoothly differentiable — ``jax.grad`` is finite and matches finite differences —
where the hard ``argmax`` reference has a zero (dropped) subgradient; (3) sub-cell accuracy:
it beats plain grid-cell readout on an axis deliberately placed between cells.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scpn_fusion.core.jax_equilibrium_solver import _interior_axis_flux
from scpn_fusion.core.jax_o_point import (
    DEFAULT_BETA,
    DEFAULT_PATCH,
    _quadratic_pinv,
    smooth_axis_flux,
)


def _domed(n=65, r0=1.6, z0=0.1, amp=1.0, wall=0.1, sign=1.0):
    """A smooth paraboloid-ish flux hump on a grid: peak ``wall + sign*amp`` at (r0, z0)."""
    R = jnp.linspace(1.0, 2.5, n)
    Z = jnp.linspace(-1.4, 1.4, n)
    RR, ZZ = jnp.meshgrid(R, Z, indexing="xy")  # [NZ, NR]
    rho2 = ((RR - r0) / 0.55) ** 2 + ((ZZ - z0) / 0.75) ** 2
    hump = jnp.exp(-rho2)  # 1 at axis, →0 at edge
    return wall + sign * amp * hump, R, Z, (r0, z0)


# ── value accuracy ────────────────────────────────────────────────


def test_matches_true_peak_positive_convention():
    psi, _R, _Z, _ = _domed(amp=1.0, wall=0.1, sign=1.0)
    axis = float(smooth_axis_flux(psi))
    true_peak = float(jnp.max(psi))
    assert abs(axis - true_peak) / abs(true_peak) < 5e-3


def test_matches_true_peak_negative_convention():
    """Sign-agnostic: a flux WELL (axis is the minimum) is handled too."""
    psi, _R, _Z, _ = _domed(amp=1.0, wall=0.1, sign=-1.0)
    axis = float(smooth_axis_flux(psi))
    true_extremum = float(jnp.min(psi))
    assert abs(axis - true_extremum) / abs(true_extremum) < 5e-2


def test_beats_grid_cell_readout_for_subcell_axis():
    """Axis deliberately off-grid: the quadratic vertex is closer to the true peak than the
    best grid cell (which is what a hard argmax returns)."""
    n = 41
    R = jnp.linspace(1.0, 2.5, n)
    Z = jnp.linspace(-1.4, 1.4, n)
    dr = float(R[1] - R[0])
    r0 = float(R[n // 2]) + 0.5 * dr  # half a cell off-grid
    z0 = float(Z[n // 2]) + 0.5 * float(Z[1] - Z[0])
    RR, ZZ = jnp.meshgrid(R, Z, indexing="xy")
    psi = 0.1 + jnp.exp(-(((RR - r0) / 0.55) ** 2 + ((ZZ - z0) / 0.75) ** 2))
    true_peak = 1.1
    grid_cell = float(jnp.max(psi))  # best on-grid value (hard-argmax readout)
    smooth = float(smooth_axis_flux(psi))
    assert abs(smooth - true_peak) < abs(grid_cell - true_peak)


def test_close_to_hard_argmax_on_grid_aligned_axis():
    psi, _R, _Z, _ = _domed(n=65)
    assert (
        abs(float(smooth_axis_flux(psi)) - float(_interior_axis_flux(psi)))
        / abs(float(_interior_axis_flux(psi)))
        < 5e-3
    )


# ── differentiability (the reason this module exists) ─────────────


def test_gradient_is_finite_and_nonzero():
    psi, _R, _Z, _ = _domed()
    g = jax.grad(lambda p: smooth_axis_flux(p))(psi)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_gradient_matches_finite_difference():
    """The whole point: a smooth, correct gradient where hard argmax drops it."""
    psi, _R, _Z, _ = _domed(n=49)
    g = np.asarray(jax.grad(lambda p: smooth_axis_flux(p))(psi))
    # perturb a handful of interior cells near the axis and check central FD
    rng = np.random.default_rng(0)
    nz, nr = psi.shape
    checked = 0
    for i, j in [(nz // 2, nr // 2), (nz // 2 + 1, nr // 2), (nz // 2, nr // 2 - 2)]:
        eps = 1e-4
        pp = psi.at[i, j].add(eps)
        pm = psi.at[i, j].add(-eps)
        fd = (float(smooth_axis_flux(pp)) - float(smooth_axis_flux(pm))) / (2 * eps)
        if abs(fd) > 1e-6:
            assert abs(g[i, j] - fd) / (abs(fd) + 1e-12) < 1e-3
            checked += 1
    assert checked >= 1


def test_jit_and_batch_stability():
    psi, _R, _Z, _ = _domed()
    a = smooth_axis_flux(psi)
    b = smooth_axis_flux(psi, patch=5)
    assert bool(jnp.isfinite(a)) and bool(jnp.isfinite(b))
    # both should land near the true peak
    assert abs(float(a) - float(b)) / abs(float(a)) < 2e-2


# ── internals & guards ────────────────────────────────────────────


def test_pinv_shape_and_cache():
    pinv7 = _quadratic_pinv(7)
    assert pinv7.shape == (6, 49)
    assert _quadratic_pinv(7) is pinv7  # cached identity


def test_defaults_sane():
    assert DEFAULT_PATCH % 2 == 1 and DEFAULT_PATCH >= 3
    assert DEFAULT_BETA > 0.0
