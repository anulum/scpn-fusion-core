# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests for the compiled (lax.while_loop) predictive forward solve
"""Self-contained tests for :mod:`scpn_fusion.core.jax_predictive_forward_compiled`.

The compiled forward must reach the SAME fixed point as the eager reference solver (the
whole point of the implementation is removing host-loop overhead, not changing physics):
equivalence is asserted at span-relative tolerance for both preconditioner settings, warm
repeats must be deterministic, the equilibrium must hold the target Ip, and the fail-closed
guards (non-uniform grids, bad settings, wrong psi_init shape) must raise. Wall-clock is
deliberately NOT asserted (host-load dependent); the committed evidence generator records
indicative local timings separately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from scpn_fusion.core.jax_free_boundary_gs import MU0_SI
from scpn_fusion.core.jax_free_boundary_predictive import (
    _plasma_current,
    build_response_matrix,
    solve_predictive_equilibrium,
)
from scpn_fusion.core.jax_o_point import smooth_axis_flux
from scpn_fusion.core.jax_predictive_forward_compiled import (
    _make_batched_runner,
    solve_predictive_equilibrium_batched,
    solve_predictive_equilibrium_compiled,
)
from scpn_fusion.core.jax_x_point import smooth_xpoint_flux

# The synthetic diverted case of the predictive test suite, verbatim.
_R = jnp.linspace(1.0, 2.5, 33)
_Z = jnp.linspace(-1.4, 1.4, 33)
_COIL_R = jnp.array([1.2, 2.3, 1.2, 2.3, 1.6, 1.5])
_COIL_Z = jnp.array([0.9, 0.9, -0.9, -0.9, 1.3, -1.35])
_COIL_I = jnp.array([-3.0e5, -3.0e5, -3.0e5, -3.0e5, -1.0e5, -6.0e5])
_PSIN = jnp.linspace(0.0, 1.0, 6)
_PPRIME = jnp.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
_FFPRIME = jnp.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
_IP = 1.0e6


@pytest.fixture(scope="module")
def response():
    return build_response_matrix(_R, _Z)


def _solve_compiled(response, **kw):
    m, b, s = response
    return solve_predictive_equilibrium_compiled(
        _COIL_I, _PPRIME, _FFPRIME, _R, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s, **kw
    )


@pytest.fixture(scope="module")
def compiled_psi(response):
    return _solve_compiled(response, n_iter=150)


def test_compiled_matches_eager_fixed_point(response, compiled_psi) -> None:
    """The compiled loop reaches the eager solver's equilibrium (implementation change,
    not a physics change). Measured during development: ~2e-9 span-relative."""
    m, b, s = response
    psi_eager = solve_predictive_equilibrium(
        _COIL_I, _PPRIME, _FFPRIME, _R, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s, n_iter=150
    )
    axis = float(smooth_axis_flux(psi_eager))
    xp = float(smooth_xpoint_flux(psi_eager, _R, _Z))
    span = abs(axis - xp)
    assert float(jnp.max(jnp.abs(compiled_psi - psi_eager))) / span < 1e-6


def test_compiled_plain_matches_eager_plain(response) -> None:
    """Equivalence also holds with the preconditioner OFF (same inner Krylov path)."""
    m, b, s = response
    psi_c = _solve_compiled(response, n_iter=150, use_mg_preconditioner=False)
    psi_e = solve_predictive_equilibrium(
        _COIL_I, _PPRIME, _FFPRIME, _R, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s, n_iter=150
    )
    span = float(jnp.max(psi_e) - jnp.min(psi_e))
    assert float(jnp.max(jnp.abs(psi_c - psi_e))) / span < 1e-6


def test_warm_repeat_is_deterministic(response, compiled_psi) -> None:
    """A second warm call through the cached runner reproduces the field bit-exactly."""
    psi2 = _solve_compiled(response, n_iter=150)
    assert float(jnp.max(jnp.abs(psi2 - compiled_psi))) == 0.0


def test_compiled_solve_holds_target_ip(compiled_psi) -> None:
    axis = smooth_axis_flux(compiled_psi)
    xp = smooth_xpoint_flux(compiled_psi, _R, _Z)
    dA = jnp.asarray((_R[1] - _R[0]) * (_Z[1] - _Z[0]))
    j = _plasma_current(
        compiled_psi, _R, axis, xp, _PSIN, _PPRIME, _FFPRIME, jnp.asarray(_IP), dA, 0.03, MU0_SI
    )
    assert abs(float(jnp.sum(j) * dA) - _IP) / _IP < 1e-6


def test_compiled_solve_is_finite_and_forms_plasma(compiled_psi) -> None:
    assert bool(jnp.all(jnp.isfinite(compiled_psi)))
    assert float(smooth_axis_flux(compiled_psi)) > float(smooth_xpoint_flux(compiled_psi, _R, _Z))


def test_mg_richardson_inner_matches_bicgstab_fixed_point(response, compiled_psi) -> None:
    """The MG-Richardson inner solver (1 matvec + 1 V-cycle per cycle) reaches the SAME
    fixed point as the BiCGSTAB reference path — measured ~1e-9 during development at
    cycle counts 2-4 on 33² and 129²; pinned here at span tolerance for cycles=2."""
    psi_mgr = _solve_compiled(response, n_iter=150, inner_solver="mg_richardson", inner_cycles=2)
    span = float(jnp.max(compiled_psi) - jnp.min(compiled_psi))
    assert float(jnp.max(jnp.abs(psi_mgr - compiled_psi))) / span < 1e-6


def test_mg_richardson_guards_fail_closed(response) -> None:
    with pytest.raises(ValueError, match="unknown inner_solver"):
        _solve_compiled(response, inner_solver="jacobi")
    with pytest.raises(ValueError, match="requires use_mg_preconditioner"):
        _solve_compiled(response, inner_solver="mg_richardson", use_mg_preconditioner=False)
    with pytest.raises(ValueError, match="inner_cycles"):
        _solve_compiled(response, inner_solver="mg_richardson", inner_cycles=0)


# ── Fail-closed guards ────────────────────────────────────────────


def test_non_uniform_grid_fails_closed(response) -> None:
    m, b, s = response
    r_bad = jnp.concatenate([_R[:-1], jnp.array([_R[-1] + 0.3])])
    with pytest.raises(ValueError, match="uniformly spaced"):
        solve_predictive_equilibrium_compiled(
            _COIL_I, _PPRIME, _FFPRIME, r_bad, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s
        )


def test_bad_settings_fail_closed(response) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        _solve_compiled(response, n_iter=0)
    with pytest.raises(ValueError, match="finite and > 0"):
        _solve_compiled(response, tol=float("nan"))


def test_wrong_psi_init_shape_fails_closed(response) -> None:
    with pytest.raises(ValueError, match="psi_init shape"):
        _solve_compiled(response, psi_init=jnp.zeros((5, 7)))


# ── Iteration diagnostics + Anderson normal-equations solver ──────


def test_return_iterations_matches_field_and_bounds(response, compiled_psi) -> None:
    """The diagnostic path returns the identical field plus a measured outer count."""
    psi, k = _solve_compiled(response, n_iter=150, return_iterations=True)
    assert float(jnp.max(jnp.abs(psi - compiled_psi))) == 0.0
    assert 1 <= k <= 150


def test_warm_start_uses_fewer_iterations(response, compiled_psi) -> None:
    """A converged warm start with ip_ramp=1 must beat the cold iteration count — this is
    the measured lever of the warm-start evidence lane (129²: 56 cold vs 11 warm)."""
    _, k_cold = _solve_compiled(response, n_iter=150, return_iterations=True)
    _, k_warm = _solve_compiled(
        response, n_iter=150, psi_init=compiled_psi, ip_ramp=1, return_iterations=True
    )
    assert k_warm < k_cold


def test_normal_eq_matches_lstsq_fixed_point(response, compiled_psi) -> None:
    """The Gram normal-equations Anderson solve (batchable, no tall SVD) reaches the same
    equilibrium as the lstsq reference — measured ~2e-9 span-relative at 129²."""
    psi_ne = _solve_compiled(response, n_iter=150, anderson_solver="normal_eq")
    span = float(jnp.max(compiled_psi) - jnp.min(compiled_psi))
    assert float(jnp.max(jnp.abs(psi_ne - compiled_psi))) / span < 1e-6


def test_unknown_anderson_solver_fails_closed(response) -> None:
    with pytest.raises(ValueError, match="unknown anderson_solver"):
        _solve_compiled(response, anderson_solver="qr")


# ── Batched (vmap) solve — the MCMC/ensemble lane ─────────────────


@pytest.fixture(scope="module")
def batch_case(response, compiled_psi):
    """±0.2 % parameter perturbations around the converged base — the MCMC-proposal
    pattern. Measured on this case: warm solves converge in 10-11 outer iterations.
    (A −1 % perturbation does NOT converge warm within 300 iterations at 33² — residual
    plateau near the softmax X-point extraction; the +1 % twin converges in 11. Batch
    elements are therefore compared ONLY where the single solve itself converges.)"""
    m, b, s = response
    factors = jnp.array([0.998, 1.0, 1.002])[:, None]
    ci = _COIL_I[jnp.newaxis, :] * factors
    pp = _PPRIME[jnp.newaxis, :] * factors
    ff = _FFPRIME[jnp.newaxis, :] * factors
    out = solve_predictive_equilibrium_batched(
        ci,
        pp,
        ff,
        _R,
        _Z,
        _COIL_R,
        _COIL_Z,
        _PSIN,
        _IP,
        m,
        b,
        s,
        psi_init=compiled_psi,
        ip_ramp=1,
        n_iter=150,
    )
    return ci, pp, ff, out


def test_batched_elements_match_single_solves(response, compiled_psi, batch_case) -> None:
    """Element i of the batch equals the single solve for sample i (same warm start and
    inner settings) — batching must not change any element's physics."""
    ci, pp, ff, out = batch_case
    assert out.shape == (3, _Z.size, _R.size)
    m, b, s = response
    span = float(jnp.max(compiled_psi) - jnp.min(compiled_psi))
    for i in (0, 2):
        single, k = solve_predictive_equilibrium_compiled(
            ci[i],
            pp[i],
            ff[i],
            _R,
            _Z,
            _COIL_R,
            _COIL_Z,
            _PSIN,
            _IP,
            m,
            b,
            s,
            psi_init=compiled_psi,
            ip_ramp=1,
            n_iter=150,
            inner_solver="mg_richardson",
            inner_cycles=2,
            return_iterations=True,
        )
        assert k < 150  # the comparison is only meaningful for a CONVERGED single solve
        assert float(jnp.max(jnp.abs(out[i] - single))) / span < 1e-6


def test_batched_runner_is_cached_and_deterministic(response, compiled_psi, batch_case) -> None:
    """A second batched call with the same static settings must HIT the runner cache (the
    measured batch cliff was per-call re-jitting: ~3 min recompile at 129² on a GTX 1060)
    and reproduce the batch bit-exactly."""
    ci, pp, ff, out = batch_case
    m, b, s = response
    hits_before = _make_batched_runner.cache_info().hits
    out2 = solve_predictive_equilibrium_batched(
        ci,
        pp,
        ff,
        _R,
        _Z,
        _COIL_R,
        _COIL_Z,
        _PSIN,
        _IP,
        m,
        b,
        s,
        psi_init=compiled_psi,
        ip_ramp=1,
        n_iter=150,
    )
    assert _make_batched_runner.cache_info().hits == hits_before + 1
    assert float(jnp.max(jnp.abs(out2 - out))) == 0.0


def test_batched_cold_start_runs_per_sample_vacuum_init(response) -> None:
    """Without psi_init every sample starts from its own vacuum field (shared_init=False
    branch); the solve must stay finite and form a plasma in each element."""
    m, b, s = response
    factors = jnp.array([0.995, 1.005])[:, None]
    out = solve_predictive_equilibrium_batched(
        _COIL_I[jnp.newaxis, :] * factors,
        _PPRIME[jnp.newaxis, :] * factors,
        _FFPRIME[jnp.newaxis, :] * factors,
        _R,
        _Z,
        _COIL_R,
        _COIL_Z,
        _PSIN,
        _IP,
        m,
        b,
        s,
        n_iter=150,
    )
    assert bool(jnp.all(jnp.isfinite(out)))
    for i in range(2):
        assert float(smooth_axis_flux(out[i])) > float(smooth_xpoint_flux(out[i], _R, _Z))


def test_batched_bad_inputs_fail_closed(response, compiled_psi) -> None:
    m, b, s = response

    def call(ci, pp, ff, **kw):
        return solve_predictive_equilibrium_batched(
            ci, pp, ff, _R, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s, **kw
        )

    two = jnp.stack([_COIL_I, _COIL_I])
    two_pp = jnp.stack([_PPRIME, _PPRIME])
    two_ff = jnp.stack([_FFPRIME, _FFPRIME])
    with pytest.raises(ValueError, match="must be 2-D"):
        call(_COIL_I, two_pp, two_ff)
    with pytest.raises(ValueError, match="batch dimensions must agree"):
        call(two, two_pp[:1], two_ff)
    with pytest.raises(ValueError, match="unknown inner_solver"):
        call(two, two_pp, two_ff, inner_solver="jacobi")
    with pytest.raises(ValueError, match="requires use_mg_preconditioner"):
        call(two, two_pp, two_ff, inner_solver="mg_richardson", use_mg_preconditioner=False)
    with pytest.raises(ValueError, match="unknown anderson_solver"):
        call(two, two_pp, two_ff, anderson_solver="qr")
    with pytest.raises(ValueError, match=">= 1"):
        call(two, two_pp, two_ff, n_iter=0)
    with pytest.raises(ValueError, match="finite and > 0"):
        call(two, two_pp, two_ff, tol=float("nan"))
    with pytest.raises(ValueError, match="inner_cycles"):
        call(two, two_pp, two_ff, inner_cycles=0)
    with pytest.raises(ValueError, match="psi_init shape"):
        call(two, two_pp, two_ff, psi_init=jnp.zeros((5, 7)))
