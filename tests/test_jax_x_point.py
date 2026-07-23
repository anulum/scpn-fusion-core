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
(lower) — which produces a real col (saddle, ``∇ψ = 0``) between them. Ground truth comes from
a high-resolution evaluation along the analytic inter-peak saddle line. Covers: saddle-value
accuracy, axis exclusion, sub-cell differentiability, remote-null rejection, near-wall
robustness, and the lower/upper-band switch.
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


def _reference_saddle_flux(
    *,
    o_rz: tuple[float, float] = (1.7, 0.4),
    x_rz: tuple[float, float] = (1.5, -0.9),
    o_amp: float = 1.0,
    x_amp: float = 0.6,
    width: float = 0.18,
) -> float:
    """High-resolution analytic saddle value between the two Gaussian peaks.

    The saddle lies on the line joining the Gaussian centres. Along that line it is the
    inter-peak minimum; in the perpendicular direction it is a maximum.
    """
    t = np.linspace(0.05, 0.95, 100_001)
    r = o_rz[0] + t * (x_rz[0] - o_rz[0])
    z = o_rz[1] + t * (x_rz[1] - o_rz[1])
    o = o_amp * np.exp(-((r - o_rz[0]) ** 2 + (z - o_rz[1]) ** 2) / width)
    x = x_amp * np.exp(-((r - x_rz[0]) ** 2 + (z - x_rz[1]) ** 2) / width)
    return float(np.min(o + x + 0.05))


def test_matches_analytic_saddle_flux() -> None:
    """The sub-cell estimator agrees with the analytic inter-peak saddle flux."""
    R, Z = _grids()
    psi = _saddle_field(R, Z)
    est = float(smooth_xpoint_flux(psi, R, Z))
    truth = _reference_saddle_flux()
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
    """``ψ_bndry`` is locally differentiable in ``ψ`` and translation-equivariant, so the
    gradient sums to one as required by the coupled adjoint."""
    R, Z = _grids()
    psi = _saddle_field(R, Z)
    g = jax.grad(lambda p: smooth_xpoint_flux(p, R, Z))(psi)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.sum(jnp.abs(g))) > 0.0
    assert abs(float(jnp.sum(g)) - 1.0) < 1e-6


def test_remote_lower_saddle_does_not_override_plasma_separatrix() -> None:
    """A lower-band saddle outside the plasma-local search radius may have an even smaller
    sampled gradient, but it is a wall/coil null and must not replace the separatrix."""
    R, Z = _grids()
    psi = _saddle_field(R, Z)
    rr, zz = jnp.meshgrid(R, Z)
    remote = 0.35 * jnp.exp(-((rr - 2.30) ** 2 + (zz + 0.85) ** 2) / 0.008)
    remote += 0.30 * jnp.exp(-((rr - 2.42) ** 2 + (zz + 1.18) ** 2) / 0.008)
    perturbed = psi + remote
    estimate = float(smooth_xpoint_flux(perturbed, R, Z))
    baseline = float(smooth_xpoint_flux(psi, R, Z))
    span = float(jnp.max(psi) - jnp.min(psi))
    assert abs(estimate - baseline) / span < 0.08


def test_edge_proximity_saddle_near_lower_wall() -> None:
    """CONTROL second-eye caveat: a divertor X-point only a few cells from the wall is still
    located (the edge band keeps the boundary from contaminating the estimate)."""
    R, Z = _grids()
    o_rz = (1.7, 0.3)
    x_rz = (1.5, -1.15)
    psi = _saddle_field(R, Z, o_rz=o_rz, x_rz=x_rz)
    est = float(smooth_xpoint_flux(psi, R, Z))
    truth = _reference_saddle_flux(o_rz=o_rz, x_rz=x_rz)
    span = float(jnp.max(psi) - jnp.min(psi))
    assert abs(est - truth) / span < 0.08


def test_upper_band_switch() -> None:
    """``search_below=False`` finds an upper single-null saddle instead."""
    R, Z = _grids()
    o_rz = (1.7, -0.4)
    x_rz = (1.5, 0.9)
    psi = _saddle_field(R, Z, o_rz=o_rz, x_rz=x_rz)
    est = float(smooth_xpoint_flux(psi, R, Z, search_below=False))
    truth = _reference_saddle_flux(o_rz=o_rz, x_rz=x_rz)
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
