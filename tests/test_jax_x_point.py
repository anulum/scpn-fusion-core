# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests for the smooth differentiable X-point (separatrix) flux finder
"""Self-contained tests for :mod:`scpn_fusion.core.jax_x_point`.

Synthetic flux fields with a genuine saddle (no external fixture), so the tests run in CI. The
field superposes two same-sign Gaussians — a plasma O-point (upper) and a divertor-coil flux
(lower) — which produces a real col (saddle, ``∇ψ = 0``) between them. Ground truth is the flux
at the hard ``|∇ψ|²`` minimum over the lower off-axis region; the smooth soft-argmin finder must
agree with it. Covers: saddle-value accuracy vs the hard minimum, axis exclusion, sub-cell
smoothness/differentiability, the near-wall edge-proximity robustness CONTROL's second-eye
flagged, and the lower/upper-band switch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from scpn_fusion.core.jax_x_point import DEFAULT_LOWER_FRAC, smooth_xpoint_flux


def _grids(nz: int = 65, nr: int = 65) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.linspace(1.0, 2.5, nr), jnp.linspace(-1.4, 1.4, nz)


def _saddle_field(
    R: jnp.ndarray,
    Z: jnp.ndarray,
    o_rz: tuple[float, float] = (1.7, 0.4),
    x_rz: tuple[float, float] = (1.5, -0.9),
    o_amp: float = 1.0,
    x_amp: float = 0.6,
    width: float = 0.18,
) -> jnp.ndarray:
    """Plasma O-point (``o_rz``) + divertor-coil flux (``x_rz``), both same sign → a real saddle
    (col) between them in the lower region; the global max is the O-point."""
    rr, zz = jnp.meshgrid(R, Z)
    o = o_amp * jnp.exp(-((rr - o_rz[0]) ** 2 + (zz - o_rz[1]) ** 2) / width)
    x = x_amp * jnp.exp(-((rr - x_rz[0]) ** 2 + (zz - x_rz[1]) ** 2) / width)
    return o + x + 0.05


def _hard_saddle_flux(
    psi: jnp.ndarray, R: jnp.ndarray, Z: jnp.ndarray, lower_frac: float = DEFAULT_LOWER_FRAC
) -> float:
    """Ground-truth separatrix flux: ψ at the hard ``|∇ψ|²`` minimum over the lower off-axis
    interior (excluding the axis disk and the wall band) — what the smooth finder approximates."""
    nz, nr = psi.shape
    d_r = float(R[1] - R[0])
    d_z = float(Z[1] - Z[0])
    gz = np.zeros((nz, nr))
    gr = np.zeros((nz, nr))
    p = np.asarray(psi)
    gz[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * d_z)
    gr[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * d_r)
    grad2 = gr**2 + gz**2
    ai, aj = np.unravel_index(int(np.argmax(p)), p.shape)
    band = int(round(lower_frac * nz))
    ii, jj = np.mgrid[0:nz, 0:nr]
    region = (
        (ii < band)
        & (ii >= 2)
        & (ii < nz - 2)
        & (jj >= 2)
        & (jj < nr - 2)
        & (((ii - ai) ** 2 + (jj - aj) ** 2) > 36)
    )
    masked = np.where(region, grad2, np.inf)
    si, sj = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return float(p[si, sj])


def test_matches_hard_saddle_flux() -> None:
    """The smooth soft-argmin agrees with the hard ``|∇ψ|²``-minimum saddle flux."""
    R, Z = _grids()
    psi = _saddle_field(R, Z)
    est = float(smooth_xpoint_flux(psi, R, Z))
    truth = _hard_saddle_flux(psi, R, Z)
    span = float(jnp.max(psi) - jnp.min(psi))
    assert abs(est - truth) / span < 0.05


def test_estimate_sits_below_the_axis() -> None:
    """The separatrix flux is a col — clearly below the O-point maximum, never at it."""
    R, Z = _grids()
    psi = _saddle_field(R, Z)
    est = float(smooth_xpoint_flux(psi, R, Z))
    span = float(jnp.max(psi) - jnp.min(psi))
    assert (float(jnp.max(psi)) - est) / span > 0.3


def test_axis_exclusion_makes_it_band_robust() -> None:
    """With the axis-disk exclusion the estimate is stable across the search-band fraction
    (the fragility that a bare band suffered) — 0.50/0.55/0.60 all agree."""
    R, Z = _grids()
    psi = _saddle_field(R, Z)
    vals = [float(smooth_xpoint_flux(psi, R, Z, lower_frac=lf)) for lf in (0.50, 0.55, 0.60)]
    span = float(jnp.max(psi) - jnp.min(psi))
    assert (max(vals) - min(vals)) / span < 0.03


def test_gradient_is_finite_and_averaging() -> None:
    """``ψ_bndry`` is smoothly differentiable in ``ψ`` (needed for the coupled adjoint); as a
    softmax-weighted average of ψ its gradient sums to one."""
    R, Z = _grids()
    psi = _saddle_field(R, Z)
    g = jax.grad(lambda p: smooth_xpoint_flux(p, R, Z))(psi)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.sum(jnp.abs(g))) > 0.0
    assert abs(float(jnp.sum(g)) - 1.0) < 1e-6


def test_edge_proximity_saddle_near_lower_wall() -> None:
    """CONTROL second-eye caveat: a divertor X-point only a few cells from the wall is still
    located (the edge band keeps the boundary from contaminating the estimate)."""
    R, Z = _grids()
    psi = _saddle_field(R, Z, o_rz=(1.7, 0.3), x_rz=(1.5, -1.15))
    est = float(smooth_xpoint_flux(psi, R, Z))
    truth = _hard_saddle_flux(psi, R, Z)
    span = float(jnp.max(psi) - jnp.min(psi))
    assert abs(est - truth) / span < 0.08


def test_upper_band_switch() -> None:
    """``search_below=False`` finds an upper single-null saddle instead."""
    R, Z = _grids()
    psi = _saddle_field(R, Z, o_rz=(1.7, -0.4), x_rz=(1.5, 0.9))
    est = float(smooth_xpoint_flux(psi, R, Z, search_below=False))
    truth = _hard_saddle_flux(psi[::-1], R, Z)  # mirror → reuse the lower-region ground truth
    span = float(jnp.max(psi) - jnp.min(psi))
    assert abs(est - truth) / span < 0.06


def test_jit_and_default_frac() -> None:
    """The default search band is a sensible interior fraction and the call is jit-stable."""
    assert 0.3 < DEFAULT_LOWER_FRAC < 0.65
    R, Z = _grids(49, 49)
    psi = _saddle_field(R, Z, o_rz=(1.7, 0.3), x_rz=(1.5, -0.8))
    a = smooth_xpoint_flux(psi, R, Z)
    b = jax.jit(lambda p: smooth_xpoint_flux(p, R, Z))(psi)
    assert np.allclose(float(a), float(b), atol=1e-10)
