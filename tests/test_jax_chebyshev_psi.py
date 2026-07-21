# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests for the compact Chebyshev ψ representation
"""Self-contained tests for :mod:`scpn_fusion.core.jax_chebyshev_psi`.

A small synthetic grid exercises the whole coefficient ⇄ field contract: the constant tensor-product
design, exact recovery of an in-basis field, faithful compression of a smooth off-basis field, the
exact-gradient linearity of the synthesis map, and differentiability of both directions (so the
compact ψ block drops into ``jax.grad`` the same way the B-spline profile coefficients do).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from scpn_fusion.core.jax_chebyshev_psi import (
    DEFAULT_N_R,
    DEFAULT_N_Z,
    _map_to_unit,
    chebyshev_psi_design_matrix,
    evaluate_psi,
    fit_psi_coeffs,
    psi_from_coeffs,
)
from scpn_fusion.core.jax_o_point import smooth_axis_flux

# ── A small synthetic equilibrium grid (non-square, to catch axis swaps) ─────
_R = np.linspace(1.0, 2.5, 33)
_Z = np.linspace(-1.4, 1.4, 29)
_NZ, _NR = _Z.size, _R.size


def _gaussian_field(width: float = 0.8) -> np.ndarray:
    """A genuinely off-basis (analytic, non-polynomial) smooth flux blob on the (NZ, NR) grid.

    A Gaussian is *not* in the Chebyshev span at any finite order, so it is a real test of spectral
    compression — a narrower ``width`` needs more modes for the same fidelity.
    """
    rr, zz = np.meshgrid(_R, _Z)
    return np.exp(-((rr - 1.7) ** 2 + zz**2) / width)


def _smooth_field() -> np.ndarray:
    """The default realistic, broad flux blob (spectrally well-compressed by 16×34 modes)."""
    return _gaussian_field(0.8)


# ── Coordinate mapping ────────────────────────────────────────────────────


def test_map_to_unit_spans_the_chebyshev_domain() -> None:
    x = _map_to_unit(_R)
    assert x[0] == pytest.approx(-1.0)
    assert x[-1] == pytest.approx(1.0)
    assert np.all(np.diff(x) > 0)  # monotone preserved


def test_map_to_unit_degenerate_span_is_centred_not_nan() -> None:
    """A single-valued grid maps to the domain centre (0), not a divide-by-zero."""
    out = _map_to_unit(np.full(4, 1.7))
    assert np.all(out == 0.0)


# ── Design matrix ─────────────────────────────────────────────────────────


def test_design_matrix_shape_and_defaults() -> None:
    phi = chebyshev_psi_design_matrix(_R, _Z)
    assert phi.shape == (_NZ * _NR, DEFAULT_N_Z * DEFAULT_N_R)
    assert (DEFAULT_N_Z, DEFAULT_N_R) == (16, 34)  # the contract's 16×34


def test_design_matrix_is_cached_by_identity() -> None:
    a = chebyshev_psi_design_matrix(_R, _Z, n_r=8, n_z=6)
    b = chebyshev_psi_design_matrix(_R, _Z, n_r=8, n_z=6)
    assert a is b  # same concrete array returned, not rebuilt


def test_first_coefficient_is_the_constant_offset() -> None:
    """``T_0 ≡ 1`` ⇒ the (jz=0, jr=0) coefficient is a uniform field offset."""
    phi = chebyshev_psi_design_matrix(_R, _Z, n_r=5, n_z=4)
    c = jnp.zeros(4 * 5).at[0].set(3.5)
    field = evaluate_psi(c, phi)
    assert np.allclose(np.asarray(field), 3.5)


# ── Analysis ⇄ synthesis round trip ───────────────────────────────────────


def test_in_basis_field_is_recovered_exactly() -> None:
    """A field synthesised from known coefficients is recovered to round-off (full-rank Φ)."""
    n_r, n_z = 8, 7
    phi = chebyshev_psi_design_matrix(_R, _Z, n_r=n_r, n_z=n_z)
    rng = np.random.default_rng(0)
    c_true = jnp.asarray(rng.standard_normal(n_z * n_r))
    psi = evaluate_psi(c_true, phi)
    c_fit = fit_psi_coeffs(psi, phi)
    assert np.allclose(np.asarray(c_fit), np.asarray(c_true), atol=1e-9)
    # and the reconstruction reproduces the field itself
    assert np.allclose(np.asarray(evaluate_psi(c_fit, phi)), np.asarray(psi), atol=1e-9)


def test_smooth_off_basis_field_compresses_faithfully() -> None:
    """A broad off-basis Gaussian is captured within a small relative residual by 16×34 modes
    (measured ~2e-7; asserted < 1e-5 with margin — honest, not machine-exact)."""
    psi = _gaussian_field(0.8)
    phi = chebyshev_psi_design_matrix(_R, _Z)
    c = fit_psi_coeffs(psi, phi)
    recon = np.asarray(psi_from_coeffs(c, phi, _NZ, _NR))
    rel = np.max(np.abs(recon - psi)) / (np.max(np.abs(psi)) + 1e-30)
    assert rel < 1e-5
    # genuine compression: far fewer coefficients than grid points
    assert c.shape[0] < _NZ * _NR


def test_more_modes_strictly_improve_an_under_resolved_fit() -> None:
    """On a narrow Gaussian (genuinely under-resolved at low order) fidelity improves by orders of
    magnitude as the basis grows — a real spectral-convergence trend, not round-off noise."""
    psi = _gaussian_field(0.30)
    err = []
    for n in (4, 8, 14):
        phi = chebyshev_psi_design_matrix(_R, _Z, n_r=n, n_z=n)
        recon = np.asarray(psi_from_coeffs(fit_psi_coeffs(psi, phi), phi, _NZ, _NR))
        err.append(np.max(np.abs(recon - psi)))
    assert err[0] > err[1] > err[2]
    assert err[0] / err[2] > 100.0  # orders-of-magnitude spectral improvement


def test_psi_from_coeffs_has_grid_shape() -> None:
    phi = chebyshev_psi_design_matrix(_R, _Z, n_r=6, n_z=5)
    c = jnp.ones(5 * 6)
    assert psi_from_coeffs(c, phi, _NZ, _NR).shape == (_NZ, _NR)


# ── Differentiability (the reason it exists) ──────────────────────────────


def test_synthesis_is_exact_gradient_linear() -> None:
    """``∂ψ/∂c`` equals the design matrix exactly (no interpolation gradient noise)."""
    phi = chebyshev_psi_design_matrix(_R, _Z, n_r=6, n_z=5)
    c0 = jnp.zeros(5 * 6)
    jac = jax.jacobian(lambda c: evaluate_psi(c, phi))(c0)
    assert np.allclose(np.asarray(jac), np.asarray(phi))


def test_grad_flows_from_axis_flux_to_compact_coefficients() -> None:
    """End-to-end: ``jax.grad`` of a field functional reaches the compact ψ coefficients, finite
    and non-zero — the compact block behaves like any other differentiated unknown."""
    phi = chebyshev_psi_design_matrix(_R, _Z, n_r=10, n_z=8)
    c = fit_psi_coeffs(_smooth_field(), phi)

    def loss(coeffs: jnp.ndarray) -> jnp.ndarray:
        return smooth_axis_flux(psi_from_coeffs(coeffs, phi, _NZ, _NR))

    g = jax.grad(loss)(c)
    assert g.shape == c.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.sum(jnp.abs(g))) > 0.0


def test_analysis_map_is_differentiable_in_the_field() -> None:
    """The least-squares analysis ``c = fit(ψ)`` is itself differentiable in ψ (finite grad)."""
    phi = chebyshev_psi_design_matrix(_R, _Z, n_r=6, n_z=5)
    psi0 = jnp.asarray(_smooth_field())

    def loss(psi: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(fit_psi_coeffs(psi, phi) ** 2)

    g = jax.grad(loss)(psi0)
    assert g.shape == psi0.shape
    assert bool(jnp.all(jnp.isfinite(g)))
