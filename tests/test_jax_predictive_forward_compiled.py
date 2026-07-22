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
