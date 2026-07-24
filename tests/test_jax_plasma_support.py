# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Topology + AD tests for axis-connected plasma support."""

from __future__ import annotations

from typing import Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_jax_config_update = cast(Callable[[str, bool], None], jax.config.update)
_jax_config_update("jax_enable_x64", True)

from scpn_fusion.core.jax_free_boundary_gs import (
    general_gs_source,
    normalised_flux,
    normalised_flux_unclipped,
)
from scpn_fusion.core.jax_free_boundary_predictive import (
    _plasma_current,
    build_response_matrix,
    solve_predictive_equilibrium,
)
from scpn_fusion.core.jax_plasma_support import (
    _default_soft_fill_steps,
    hard_axis_connected_from_psi_n,
    hard_axis_connected_mask,
    soft_axis_connected_support,
    soft_lcfs_weight,
)


def _synthetic_diverted_psi(
    nz: int = 49,
    nr: int = 49,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build a topology toy: axis-connected core + disconnected private-flux island.

    Both regions have ``ψ_N < 1`` under axis/boundary normalisation, but only the
    core is path-connected to the axis. This is the diverted private-flux failure
    mode of a pure ``ψ_N < 1`` mask.
    """
    z = np.linspace(-1.0, 1.0, nz)
    r = np.linspace(0.5, 1.5, nr)
    rr, zz = np.meshgrid(r, z, indexing="xy")
    # Core: high peak at (1.0, 0.15) — axis above midplane.
    core = np.exp(-((rr - 1.0) ** 2 + (zz - 0.15) ** 2) / 0.05)
    # Separatrix-like collar so the private island is flux-disconnected.
    # Private flux island below: intermediate amplitude, spatially separate.
    private = 0.45 * np.exp(-((rr - 1.0) ** 2 + (zz + 0.75) ** 2) / 0.02)
    psi = core + private
    # Axis and boundary scalars for ψ_N.
    psi_axis = float(np.max(psi))
    # Boundary below private peak so both core flanks and private sit under ψ_N < 1.
    psi_bndry = 0.20 * psi_axis
    return psi.astype(np.float64), psi, psi_axis, psi_bndry


def test_hard_flood_fill_excludes_disconnected_private_island() -> None:
    """Hard 4-connected fill from the axis must drop the private-flux blob."""
    psi, _, psi_axis, psi_bndry = _synthetic_diverted_psi()
    psi_n = np.asarray(
        normalised_flux(jnp.asarray(psi), jnp.asarray(psi_axis), jnp.asarray(psi_bndry))
    )
    level = psi_n < 1.0
    # Seed = deepest core cell (min ψ_N inside level set).
    connected = hard_axis_connected_from_psi_n(psi_n)
    # Private island centre ~ row of z=-0.75.
    nz, nr = psi_n.shape
    z = np.linspace(-1.0, 1.0, nz)
    r = np.linspace(0.5, 1.5, nr)
    i_priv = int(np.argmin(np.abs(z + 0.75)))
    j_priv = int(np.argmin(np.abs(r - 1.0)))
    i_axis = int(np.argmin(np.abs(z - 0.15)))
    j_axis = int(np.argmin(np.abs(r - 1.0)))
    assert level[i_priv, j_priv], "fixture must put private cell inside ψ_N < 1"
    assert level[i_axis, j_axis], "fixture must put axis cell inside ψ_N < 1"
    assert connected[i_axis, j_axis]
    assert not connected[i_priv, j_priv], "private island must be excluded"
    # Pure level set still includes private — the bug Tomáš reported.
    assert int(level.sum()) > int(connected.sum())


def test_hard_seed_outside_level_set_is_empty() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    out = hard_axis_connected_mask(mask, seed_i=0, seed_j=0)
    assert not out.any()


def test_soft_support_suppresses_private_island() -> None:
    """Soft AD-safe support must also suppress the disconnected private island."""
    psi, _, psi_axis, psi_bndry = _synthetic_diverted_psi()
    psi_j = jnp.asarray(psi)
    psi_n = normalised_flux_unclipped(
        psi_j,
        jnp.asarray(psi_axis),
        jnp.asarray(psi_bndry),
    )
    soft = np.asarray(soft_axis_connected_support(psi_j, psi_n, cutoff_width=0.03))
    hard = hard_axis_connected_from_psi_n(np.asarray(psi_n))
    nz, nr = soft.shape
    z = np.linspace(-1.0, 1.0, nz)
    r = np.linspace(0.5, 1.5, nr)
    i_priv = int(np.argmin(np.abs(z + 0.75)))
    j_priv = int(np.argmin(np.abs(r - 1.0)))
    i_axis = int(np.argmin(np.abs(z - 0.15)))
    j_axis = int(np.argmin(np.abs(r - 1.0)))
    assert soft[i_axis, j_axis] > 0.5
    assert soft[i_priv, j_priv] < 0.05
    # Soft agrees with hard topology on a high threshold.
    soft_bool = soft > 0.2
    iou_num = float(np.sum(soft_bool & hard))
    iou_den = float(np.sum(soft_bool | hard)) + 1e-30
    assert iou_num / iou_den > 0.5
    # Private cell stays out of the soft high set.
    assert not soft_bool[i_priv, j_priv]


def test_soft_support_is_finite_and_bounded() -> None:
    psi, _, psi_axis, psi_bndry = _synthetic_diverted_psi(33, 33)
    psi_j = jnp.asarray(psi)
    psi_n = normalised_flux_unclipped(
        psi_j,
        jnp.asarray(psi_axis),
        jnp.asarray(psi_bndry),
    )
    soft = soft_axis_connected_support(psi_j, psi_n, 0.03)
    assert bool(jnp.all(jnp.isfinite(soft)))
    assert float(jnp.min(soft)) >= 0.0
    assert float(jnp.max(soft)) <= 1.0 + 1e-12


def test_soft_support_gradients_are_finite() -> None:
    """Reverse-mode through the soft flood fill must stay finite (AD-safe)."""
    psi, _, psi_axis, psi_bndry = _synthetic_diverted_psi(33, 33)
    psi_j = jnp.asarray(psi)

    def loss(p: jnp.ndarray) -> jnp.ndarray:
        pn = normalised_flux_unclipped(
            p,
            jnp.asarray(psi_axis),
            jnp.asarray(psi_bndry),
        )
        return jnp.sum(soft_axis_connected_support(p, pn, 0.05))

    g = jax.grad(loss)(psi_j)
    assert bool(jnp.all(jnp.isfinite(g)))
    # Non-vacuous: axis region should carry sensitivity.
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_soft_lcfs_weight_matches_tanh_definition() -> None:
    psi_n = jnp.linspace(0.0, 1.5, 64)
    w = 0.03
    got = soft_lcfs_weight(psi_n, w)
    ref = 0.5 * (1.0 - jnp.tanh((psi_n - 1.0) / w))
    assert float(jnp.max(jnp.abs(got - ref))) < 1e-14


def test_unclipped_lcfs_distance_decays_outside_instead_of_stalling_at_half() -> None:
    psi = jnp.asarray([1.0, 0.2, 0.0])
    raw = normalised_flux_unclipped(psi, jnp.asarray(1.0), jnp.asarray(0.2))
    clipped = normalised_flux(psi, jnp.asarray(1.0), jnp.asarray(0.2))
    assert np.asarray(raw) == pytest.approx(np.asarray([0.0, 1.0, 1.25]))
    assert np.asarray(clipped) == pytest.approx(np.asarray([0.0, 1.0, 1.0]))
    assert float(soft_lcfs_weight(raw, 0.03)[-1]) < 1.0e-6
    assert float(soft_lcfs_weight(clipped, 0.03)[-1]) == pytest.approx(0.5)


def test_equal_depth_disconnected_basin_is_not_independently_seeded() -> None:
    nz = nr = 49
    z = jnp.linspace(-1.0, 1.0, nz)
    r = jnp.linspace(-1.0, 1.0, nr)
    zz, rr = jnp.meshgrid(z, r, indexing="ij")
    sigma2 = 0.08**2
    upper = jnp.exp(-((zz - 0.55) ** 2 + rr**2) / (2.0 * sigma2))
    lower = jnp.exp(-((zz + 0.55) ** 2 + rr**2) / (2.0 * sigma2))
    psi = jnp.maximum(upper, lower)
    axis = jnp.max(psi)
    boundary = jnp.asarray(0.2, dtype=psi.dtype)
    raw = normalised_flux_unclipped(psi, axis, boundary)
    soft = soft_axis_connected_support(psi, raw, 0.03)
    hard = hard_axis_connected_from_psi_n(np.asarray(raw))
    upper_i = int(jnp.argmin(jnp.abs(z - 0.55)))
    lower_i = int(jnp.argmin(jnp.abs(z + 0.55)))
    centre_j = int(jnp.argmin(jnp.abs(r)))
    hard_centres = (bool(hard[upper_i, centre_j]), bool(hard[lower_i, centre_j]))
    assert sum(hard_centres) == 1
    selected_i, rejected_i = (upper_i, lower_i) if hard_centres[0] else (lower_i, upper_i)
    assert float(soft[selected_i, centre_j]) > 0.5
    assert float(soft[rejected_i, centre_j]) < 1.0e-6


def test_default_soft_fill_budget_is_capped() -> None:
    assert _default_soft_fill_steps((33, 33)) == 32
    assert _default_soft_fill_steps((257, 257)) == 96


def test_general_gs_source_axis_connected_kills_private_source() -> None:
    """Default general_gs_source must not put source on a private-flux island."""
    psi, _, psi_axis, psi_bndry = _synthetic_diverted_psi()
    R = jnp.linspace(0.5, 1.5, psi.shape[1])
    psin = jnp.linspace(0.0, 1.0, 6)
    pprime = jnp.full(6, -1.0e4)
    ffprime = jnp.full(6, -0.5)
    src_conn = np.asarray(
        general_gs_source(
            jnp.asarray(psi),
            R,
            jnp.asarray(psi_axis),
            jnp.asarray(psi_bndry),
            psin,
            pprime,
            ffprime,
            axis_connected=True,
        )
    )
    src_level = np.asarray(
        general_gs_source(
            jnp.asarray(psi),
            R,
            jnp.asarray(psi_axis),
            jnp.asarray(psi_bndry),
            psin,
            pprime,
            ffprime,
            axis_connected=False,
        )
    )
    nz, nr = psi.shape
    z = np.linspace(-1.0, 1.0, nz)
    r = np.linspace(0.5, 1.5, nr)
    i_priv = int(np.argmin(np.abs(z + 0.75)))
    j_priv = int(np.argmin(np.abs(r - 1.0)))
    # Level-set path still sources private island; connected path must not.
    assert abs(src_level[i_priv, j_priv]) > 1e-6
    assert abs(src_conn[i_priv, j_priv]) < 1e-3 * (np.max(np.abs(src_conn)) + 1e-30)


def test_historical_level_set_path_still_available() -> None:
    """axis_connected=False preserves the pre-fix ψ_N < 1 contract."""
    psi, _, psi_axis, psi_bndry = _synthetic_diverted_psi(33, 33)
    R = jnp.linspace(0.5, 1.5, psi.shape[1])
    psin = jnp.linspace(0.0, 1.0, 6)
    pprime = jnp.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
    ffprime = jnp.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
    src = np.asarray(
        general_gs_source(
            jnp.asarray(psi),
            R,
            jnp.asarray(psi_axis),
            jnp.asarray(psi_bndry),
            psin,
            pprime,
            ffprime,
            axis_connected=False,
        )
    )
    psi_n = np.asarray(
        normalised_flux(jnp.asarray(psi), jnp.asarray(psi_axis), jnp.asarray(psi_bndry))
    )
    pp = np.interp(psi_n, np.asarray(psin), np.asarray(pprime))
    ffp = np.interp(psi_n, np.asarray(psin), np.asarray(ffprime))
    r2d = np.asarray(R)[np.newaxis, :]
    mu0 = 4.0e-7 * np.pi
    expected = np.where(psi_n < 1.0, -(mu0 * r2d**2 * pp + ffp), 0.0)
    np.testing.assert_allclose(src, expected, rtol=1e-6, atol=1e-9)


def test_plasma_current_ip_holds_on_synthetic_diverted_solve() -> None:
    """Predictive solve still holds Ip after axis-connected support wiring."""
    r = jnp.linspace(1.0, 2.5, 33)
    z = jnp.linspace(-1.4, 1.4, 33)
    coil_r = jnp.array([1.2, 2.3, 1.2, 2.3, 1.6, 1.5])
    coil_z = jnp.array([0.9, 0.9, -0.9, -0.9, 1.3, -1.35])
    coil_i = jnp.array([-3.0e5, -3.0e5, -3.0e5, -3.0e5, -1.0e5, -6.0e5])
    psin = jnp.linspace(0.0, 1.0, 6)
    pprime = jnp.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
    ffprime = jnp.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
    ip = 1.0e6
    m, b, s = build_response_matrix(r, z)
    psi = solve_predictive_equilibrium(
        coil_i, pprime, ffprime, r, z, coil_r, coil_z, psin, ip, m, b, s, n_iter=150
    )
    assert bool(jnp.all(jnp.isfinite(psi)))
    d_r = float(r[1] - r[0])
    d_z = float(z[1] - z[0])
    dA = d_r * d_z
    axis = jnp.max(psi)  # rough; _plasma_current recomputes smooth axis internally via caller
    # Recompute through public residual path: use solve's Ip via integrated current helper.
    from scpn_fusion.core.jax_o_point import smooth_axis_flux
    from scpn_fusion.core.jax_x_point import smooth_xpoint_flux

    ax = smooth_axis_flux(psi)
    bd = smooth_xpoint_flux(psi, r, z)
    j = _plasma_current(
        psi, r, ax, bd, psin, pprime, ffprime, jnp.asarray(ip), dA, 0.03, 4.0e-7 * jnp.pi
    )
    ip_got = float(jnp.sum(j) * dA)
    assert abs(ip_got - ip) / ip < 1e-3
    assert bool(jnp.all(jnp.isfinite(j)))
