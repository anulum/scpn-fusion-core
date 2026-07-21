# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests for the JAX geometric-multigrid GS preconditioner
"""Self-contained tests for :mod:`scpn_fusion.core.jax_multigrid_precond`.

Covers the contract a Krylov preconditioner must honour (exact linearity, exact identity on the
wall ring), its quality (a single V-cycle contracts the GS residual by a large factor; the
preconditioned BiCGSTAB needs measurably fewer iterations — an iteration COUNT, so the metric
is host-load independent), equality of the preconditioned and plain solutions, the fail-closed
guards, and the honest degradation on grids that cannot coarsen. Wall-clock speed is
deliberately NOT asserted here — that is the dedicated-hardware benchmark lane.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from scpn_fusion.core.jax_free_boundary_predictive import _gs_operator_flat
from scpn_fusion.core.jax_multigrid_precond import (
    _build_hierarchy,
    build_gs_mg_preconditioner,
)

_NR = 65
_NZ = 65
_R = jnp.linspace(1.0, 2.5, _NR)
_Z = jnp.linspace(-1.4, 1.4, _NZ)
_DR = float(_R[1] - _R[0])
_DZ = float(_Z[1] - _Z[0])
_SHAPE = (_NZ, _NR)


def _operator(psi_flat: jnp.ndarray) -> jnp.ndarray:
    return _gs_operator_flat(psi_flat, _SHAPE, _R, jnp.asarray(_DR), jnp.asarray(_DZ))


def _test_rhs() -> jnp.ndarray:
    """A GS-like right-hand side: smooth interior source + a non-trivial Dirichlet ring."""
    rr, zz = jnp.meshgrid(_R, _Z)
    interior = -3.0e-1 * rr * jnp.exp(-((rr - 1.7) ** 2 + zz**2) / 0.15)
    ring = 0.05 * rr + 0.02 * zz
    rhs = interior.at[0, :].set(ring[0, :]).at[-1, :].set(ring[-1, :])
    rhs = rhs.at[:, 0].set(ring[:, 0]).at[:, -1].set(ring[:, -1])
    return rhs.reshape(-1)


def _min_iters_to_tol(m: object, rhs: jnp.ndarray, rel_tol: float, max_cap: int = 4000) -> int:
    """Smallest BiCGSTAB ``maxiter`` whose returned iterate satisfies the residual tolerance.

    Measured by re-running the solve at candidate iteration caps and checking the TRUE relative
    residual ``‖A x − b‖/‖b‖`` — an iteration count, independent of host load."""
    b_norm = float(jnp.linalg.norm(rhs))

    def achieved(maxiter: int) -> bool:
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            _operator, rhs, tol=0.0, atol=1e-300, maxiter=maxiter, M=m
        )
        res = float(jnp.linalg.norm(_operator(x) - rhs)) / b_norm
        return bool(np.isfinite(res)) and res <= rel_tol

    lo, hi = 1, 1
    while not achieved(hi):
        lo, hi = hi, hi * 2
        if hi > max_cap:
            pytest.fail(f"BiCGSTAB did not reach rel_tol={rel_tol} within {max_cap} iters")
    while lo < hi:
        mid = (lo + hi) // 2
        if achieved(mid):
            hi = mid
        else:
            lo = mid + 1
    return hi


# ── Preconditioner contract: linearity + exact ring identity ──────


def test_preconditioner_is_exactly_linear() -> None:
    m = build_gs_mg_preconditioner(_SHAPE, _R, _DR, _DZ)
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal(_NZ * _NR))
    y = jnp.asarray(rng.standard_normal(_NZ * _NR))
    a, b = 2.75, -0.4
    combined = np.asarray(m(a * x + b * y))
    separate = a * np.asarray(m(x)) + b * np.asarray(m(y))
    scale = np.max(np.abs(separate)) + 1e-300
    assert np.max(np.abs(combined - separate)) / scale < 1e-12


def test_wall_ring_rows_are_inverted_exactly() -> None:
    """``A`` has identity wall rows, so ``M`` must return ``r`` on the ring bit-exactly."""
    m = build_gs_mg_preconditioner(_SHAPE, _R, _DR, _DZ)
    r = _test_rhs()
    z = np.asarray(m(r)).reshape(_SHAPE)
    r2 = np.asarray(r).reshape(_SHAPE)
    assert np.array_equal(z[0, :], r2[0, :]) and np.array_equal(z[-1, :], r2[-1, :])
    assert np.array_equal(z[:, 0], r2[:, 0]) and np.array_equal(z[:, -1], r2[:, -1])


# ── Preconditioner quality (host-load-independent metrics) ────────


def test_single_vcycle_contracts_the_residual() -> None:
    """On ring-free vectors — the representative Krylov workload once the identity wall rows
    are satisfied — one application of ``M`` must already solve most of the system:
    ``‖A M r − r‖ ≪ ‖r‖``, with a grid-independent factor (the multigrid signature).

    Ring-HEAVY vectors are deliberately not asserted here: strong Dirichlet ring data couples
    into the near-ring interior at ``O(1/h²)`` and a single V-cycle leaves a leftover larger
    than ``‖r‖`` (measured ≈ 15 ‖r‖ at 65² during development). That is expected of a
    preconditioner and harmless in the Krylov loop — the outer iteration absorbs it, as the
    iteration-count and solution-equality tests below pin down."""
    rng = np.random.default_rng(11)
    for n in (33, 65):
        r_axis = jnp.linspace(1.0, 2.5, n)
        z_axis = jnp.linspace(-1.4, 1.4, n)
        d_r = float(r_axis[1] - r_axis[0])
        d_z = float(z_axis[1] - z_axis[0])

        def op(pf: jnp.ndarray, _n=n, _r=r_axis, _dr=d_r, _dz=d_z) -> jnp.ndarray:
            return _gs_operator_flat(pf, (_n, _n), _r, jnp.asarray(_dr), jnp.asarray(_dz))

        rr, zz = jnp.meshgrid(r_axis, z_axis)
        smooth = -3.0e-1 * rr * jnp.exp(-((rr - 1.7) ** 2 + zz**2) / 0.15)
        noise = jnp.asarray(rng.standard_normal((n, n)))
        m = build_gs_mg_preconditioner((n, n), r_axis, d_r, d_z)
        for field in (smooth, noise):
            ring_free = (
                field.at[0, :].set(0.0).at[-1, :].set(0.0).at[:, 0].set(0.0).at[:, -1].set(0.0)
            )
            r = ring_free.reshape(-1)
            contraction = float(jnp.linalg.norm(op(m(r)) - r) / jnp.linalg.norm(r))
            assert contraction < 0.2, (n, contraction)


def test_preconditioning_cuts_krylov_iterations_at_least_in_half() -> None:
    """Iteration-count comparison on the 65² GS system (counts are load-independent; the
    wall-clock claim is reserved for the dedicated-hardware benchmark)."""
    rhs = _test_rhs()
    plain = _min_iters_to_tol(None, rhs, 1e-8)
    m = build_gs_mg_preconditioner(_SHAPE, _R, _DR, _DZ)
    preconditioned = _min_iters_to_tol(m, rhs, 1e-8)
    assert preconditioned * 2 <= plain, (plain, preconditioned)


def test_preconditioned_solution_matches_plain_solution() -> None:
    """The preconditioner must not move the answer, only the convergence path."""
    rhs = _test_rhs()
    m = build_gs_mg_preconditioner(_SHAPE, _R, _DR, _DZ)
    x_plain, _ = jax.scipy.sparse.linalg.bicgstab(_operator, rhs, tol=1e-12, atol=0.0, maxiter=4000)
    x_mg, _ = jax.scipy.sparse.linalg.bicgstab(
        _operator, rhs, tol=1e-12, atol=0.0, maxiter=4000, M=m
    )
    scale = float(jnp.max(jnp.abs(x_plain))) + 1e-300
    assert float(jnp.max(jnp.abs(x_mg - x_plain))) / scale < 1e-8


# ── Level hierarchy ───────────────────────────────────────────────


def test_hierarchy_coarsens_odd_grids_to_min_grid() -> None:
    levels = _build_hierarchy(np.asarray(_R, dtype=np.float64), _NZ, _DR, _DZ, 5)
    assert [lv.shape for lv in levels] == [(65, 65), (33, 33), (17, 17), (9, 9), (5, 5)]


def test_even_grid_degrades_to_single_level_smoother() -> None:
    """A grid that cannot coarsen still yields a valid (weaker) linear preconditioner."""
    nz = nr = 16
    r = jnp.linspace(1.0, 2.5, nr)
    levels = _build_hierarchy(np.asarray(r, dtype=np.float64), nz, 0.1, 0.1, 5)
    assert len(levels) == 1
    m = build_gs_mg_preconditioner((nz, nr), r, 0.1, 0.1)
    rng = np.random.default_rng(3)
    v = jnp.asarray(rng.standard_normal(nz * nr))
    out = np.asarray(m(v))
    assert out.shape == (nz * nr,) and np.all(np.isfinite(out))


# ── Fail-closed guards ────────────────────────────────────────────


def test_rejects_mismatched_r_grid_length() -> None:
    with pytest.raises(ValueError, match="R_grid must be 1-D of length"):
        build_gs_mg_preconditioner(_SHAPE, _R[:-1], _DR, _DZ)


def test_rejects_degenerate_grid_and_spacings() -> None:
    with pytest.raises(ValueError, match="at least 3x3"):
        build_gs_mg_preconditioner((2, 5), jnp.linspace(1.0, 2.0, 5), 0.1, 0.1)
    with pytest.raises(ValueError, match="finite and > 0"):
        build_gs_mg_preconditioner(_SHAPE, _R, 0.0, _DZ)
    with pytest.raises(ValueError, match="finite and > 0"):
        build_gs_mg_preconditioner(_SHAPE, _R, _DR, float("nan"))


def test_rejects_nonpositive_tuning_parameters() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        build_gs_mg_preconditioner(_SHAPE, _R, _DR, _DZ, n_vcycles=0)
    with pytest.raises(ValueError, match=">= 1"):
        build_gs_mg_preconditioner(_SHAPE, _R, _DR, _DZ, pre_smooth=0)
