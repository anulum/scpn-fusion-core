# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests for the differentiable compact B-spline profile basis
"""Self-contained tests for :mod:`scpn_fusion.core.jax_profile_basis`.

Covers: design-matrix shape + partition of unity, clamped endpoints, the exact-gradient linear
map (``∂profile/∂coeffs = B``), profile reconstruction from fitted coefficients, the cache, and
configurable ``n_coeff`` / ``degree`` — all cheap (no equilibrium solve).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from scpn_fusion.core.jax_profile_basis import (
    DEFAULT_N_COEFF,
    bspline_design_matrix,
    evaluate_profile,
)


def test_design_matrix_shape_and_partition_of_unity() -> None:
    psin = np.linspace(0.0, 1.0, 40)
    b = bspline_design_matrix(psin, n_coeff=12)
    assert b.shape == (40, 12)
    assert np.allclose(b.sum(axis=1), 1.0, atol=1e-10)  # partition of unity


def test_clamped_endpoints_pass_through_end_coefficients() -> None:
    """A clamped knot vector anchors the profile to the first/last coefficient at ψ_N = 0 / 1."""
    psin = np.array([0.0, 1.0])
    b = bspline_design_matrix(psin, n_coeff=10)
    coeffs = jnp.asarray(np.arange(10, dtype=float) + 1.0)
    vals = evaluate_profile(coeffs, b)
    assert abs(float(vals[0]) - 1.0) < 1e-10  # coeffs[0]
    assert abs(float(vals[1]) - 10.0) < 1e-10  # coeffs[-1]


def test_evaluate_is_linear_in_coefficients() -> None:
    psin = np.linspace(0.0, 1.0, 25)
    b = bspline_design_matrix(psin, n_coeff=12)
    c = jnp.asarray(np.linspace(-3.0, 2.0, 12))
    assert np.allclose(
        np.asarray(evaluate_profile(3.0 * c, b)),
        3.0 * np.asarray(evaluate_profile(c, b)),
        atol=1e-10,
    )


def test_gradient_is_the_exact_design_matrix() -> None:
    """``∂(wᵀ·profile)/∂coeffs = Bᵀ w`` exactly — the fixed basis gives a noise-free gradient."""
    psin = np.linspace(0.0, 1.0, 30)
    b = bspline_design_matrix(psin, n_coeff=12)
    w = jnp.asarray(np.linspace(0.1, 1.0, 30))
    c = jnp.asarray(np.zeros(12))
    g = jax.grad(lambda coeffs: jnp.sum(w * evaluate_profile(coeffs, b)))(c)
    assert np.allclose(np.asarray(g), b.T @ np.asarray(w), atol=1e-10)


def test_reconstructs_a_smooth_target_profile() -> None:
    """Least-squares-fitted coefficients reproduce a smooth target p'(ψ_N) closely."""
    psin = np.linspace(0.0, 1.0, 60)
    target = -8.0e4 * (1.0 - psin**2)  # a plausible p' shape, peaked on axis, zero at edge
    b = bspline_design_matrix(psin, n_coeff=12)
    coeffs, *_ = np.linalg.lstsq(b, target, rcond=None)
    recon = np.asarray(evaluate_profile(jnp.asarray(coeffs), b))
    rel = np.max(np.abs(recon - target)) / (np.max(np.abs(target)) + 1e-30)
    assert rel < 5e-3


def test_cache_returns_same_array_and_is_reused() -> None:
    psin = np.linspace(0.0, 1.0, 20)
    a = bspline_design_matrix(psin, n_coeff=DEFAULT_N_COEFF)
    b = bspline_design_matrix(psin, n_coeff=DEFAULT_N_COEFF)
    assert a is b  # cached
    assert bspline_design_matrix(psin, n_coeff=8) is not a  # distinct key


def test_cached_design_is_read_only() -> None:
    """The cached design is shared across callers, so it must be frozen against mutation."""
    b = bspline_design_matrix(np.linspace(0.0, 1.0, 20), n_coeff=10)
    assert not b.flags.writeable
    with pytest.raises(ValueError):
        b[0, 0] = 9.0


def test_degree_and_ncoeff_configurable() -> None:
    psin = np.linspace(0.0, 1.0, 15)
    for n_coeff in (6, 12, 16):
        b = bspline_design_matrix(psin, n_coeff=n_coeff, degree=3)
        assert b.shape == (15, n_coeff)
        assert np.allclose(b.sum(axis=1), 1.0, atol=1e-10)
