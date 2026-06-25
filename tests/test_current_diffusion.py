# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Current Diffusion Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.current_diffusion import (
    CurrentDiffusionSolver,
    FluxEvolutionTrajectory,
    _as_profile,
    _call_tau,
    _validate_radial_grid,
    neoclassical_resistivity,
    q_from_psi,
    resistive_diffusion_time,
    solve_flux_evolution_nonadiabatic,
)


def test_neoclassical_resistivity() -> None:
    eta = neoclassical_resistivity(Te_keV=1.0, ne_19=1.0, Z_eff=1.5, epsilon=0.1)
    assert eta > 0.0
    assert 1e-8 < eta < 1e-7


def test_q_from_psi() -> None:
    rho = np.linspace(0, 1, 50)
    R0 = 2.0
    a = 0.5
    B0 = 1.0
    psi = -(a**2) * B0 / R0 * rho**2 / 2.0

    q = q_from_psi(rho, psi, R0, a, B0)
    assert np.allclose(q, 1.0, atol=0.1)


def test_resistive_diffusion_time() -> None:
    tau = resistive_diffusion_time(a=2.0, eta=1e-8)
    assert tau > 100.0


def test_pure_ohmic_relaxation() -> None:
    rho = np.linspace(0, 1, 50)
    solver = CurrentDiffusionSolver(rho, R0=2.0, a=0.5, B0=1.0)

    Te = np.ones(50)
    ne = np.ones(50)
    j_bs = np.zeros(50)
    j_cd = np.zeros(50)

    psi_initial = solver.psi.copy()

    dt = 10.0
    for _ in range(10):
        solver.step(dt, Te, ne, 1.5, j_bs, j_cd)

    assert not np.allclose(solver.psi, psi_initial)
    q = q_from_psi(rho, solver.psi, 2.0, 0.5, 1.0)
    assert np.all(q > 0)


def test_conservation_and_steady_state() -> None:
    rho = np.linspace(0, 1, 50)
    solver = CurrentDiffusionSolver(rho, R0=2.0, a=0.5, B0=1.0)

    Te = np.ones(50)
    ne = np.ones(50)
    j_bs = np.zeros(50)
    j_cd = np.ones(50) * 1e5

    dt = 1.0
    for _ in range(100):
        solver.step(dt, Te, ne, 1.5, j_bs, j_cd)

    psi_new = solver.psi.copy()
    solver.step(dt, Te, ne, 1.5, j_bs, j_cd)
    assert np.allclose(psi_new, solver.psi, rtol=1e-3, atol=1e-3)


def test_bootstrap_steepens_q() -> None:
    rho = np.linspace(0, 1, 50)
    solver = CurrentDiffusionSolver(rho, R0=2.0, a=0.5, B0=1.0)

    Te = np.ones(50)
    ne = np.ones(50)
    j_bs = np.exp(-(((rho - 0.5) / 0.1) ** 2)) * 1e6
    j_cd = np.zeros(50)

    for _ in range(5):
        solver.step(10.0, Te, ne, 1.5, j_bs, j_cd)

    q = q_from_psi(rho, solver.psi, 2.0, 0.5, 1.0)
    assert q[0] > 0


# ---------------------------------------------------------------------------
# Non-adiabatic MIF/FRC flux carrier — solve_flux_evolution_nonadiabatic.
#
# The integrator advances ``dpsi/dt = -psi/tau_psi + R_null E_theta - eta J_theta``
# with an analytic per-cell integrating factor over each step. The four
# closed-form regimes below pin the scheme against exact solutions; the
# remaining tests exercise every validation branch.
# ---------------------------------------------------------------------------


def _valid_flux_kwargs() -> dict[str, object]:
    """A fully-valid, fully-quiescent keyword set (zero drive, no damping)."""
    return {
        "tau_psi_fn": lambda t, r: np.full(r.size, np.inf),
        "R_null_t": lambda t: 1.0,
        "E_theta_t": lambda t, r: np.zeros(r.size),
        "eta_spitzer_fn": lambda r: np.zeros(r.size),
        "J_theta_t": lambda t, r: np.zeros(r.size),
        "dt": 0.1,
        "n_steps": 3,
    }


def test_flux_evolution_zero_drive_is_invariant() -> None:
    rho = np.linspace(0.0, 1.0, 6)
    psi0 = np.linspace(0.5, 0.0, 6)
    traj = solve_flux_evolution_nonadiabatic(rho, psi0, **_valid_flux_kwargs())  # type: ignore[arg-type]

    assert isinstance(traj, FluxEvolutionTrajectory)
    assert traj.time_s.shape == (4,)
    assert traj.psi.shape == (4, 6)
    assert traj.source_increment.shape == (3, 6)
    assert traj.dt_s == 0.1
    # No drive, infinite damping time -> psi is frozen at psi0 for all steps.
    for snapshot in traj.psi:
        assert np.allclose(snapshot, psi0, rtol=0.0, atol=1e-15)
    assert np.allclose(traj.hall_drive, 0.0)
    assert np.allclose(traj.resistive_loss, 0.0)
    assert np.allclose(traj.source, 0.0)
    assert np.allclose(traj.damping_rate, 0.0)
    assert np.allclose(traj.source_increment, 0.0)
    assert np.allclose(traj.damping_decrement, 0.0)
    assert np.allclose(traj.update_residual, 0.0)


def test_flux_evolution_undamped_constant_source_ramps_linearly() -> None:
    rho = np.linspace(0.0, 1.0, 4)
    psi0 = np.full(4, 2.0)
    dt, n_steps = 0.2, 5
    # source = R_null E_theta - eta J_theta = 1.0*0.5 - 0.2*1.0 = 0.3.
    traj = solve_flux_evolution_nonadiabatic(
        rho,
        psi0,
        tau_psi_fn=lambda t, r: np.full(r.size, np.inf),
        R_null_t=lambda t: 1.0,
        E_theta_t=lambda t, r: np.full(r.size, 0.5),
        eta_spitzer_fn=lambda r: np.full(r.size, 0.2),
        J_theta_t=lambda t, r: np.full(r.size, 1.0),
        dt=dt,
        n_steps=n_steps,
    )
    source = 0.3
    for n in range(n_steps + 1):
        assert np.allclose(traj.psi[n], 2.0 + source * dt * n, atol=1e-12)
    assert np.allclose(traj.hall_drive, 0.5)
    assert np.allclose(traj.resistive_loss, 0.2)
    assert np.allclose(traj.source, 0.3)
    # Exact discrete balance psi[n+1] = psi[n] - damping_decrement + source_increment.
    for n in range(n_steps):
        assert np.allclose(
            traj.psi[n + 1],
            traj.psi[n] - traj.damping_decrement[n] + traj.source_increment[n],
            atol=1e-15,
        )
    assert np.allclose(traj.update_residual, 0.0)


def test_flux_evolution_pure_damping_matches_analytic_exponential() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    psi0 = np.full(5, 4.0)
    tau = 10.0
    dt, n_steps = 0.5, 8
    traj = solve_flux_evolution_nonadiabatic(
        rho,
        psi0,
        tau_psi_fn=lambda t, r: np.full(r.size, tau),
        R_null_t=lambda t: 0.0,
        E_theta_t=lambda t, r: np.zeros(r.size),
        eta_spitzer_fn=lambda r: np.zeros(r.size),
        J_theta_t=lambda t, r: np.zeros(r.size),
        dt=dt,
        n_steps=n_steps,
    )
    assert np.allclose(traj.damping_rate, 1.0 / tau)
    for n in range(n_steps + 1):
        t = dt * n
        assert np.allclose(traj.psi[n], 4.0 * np.exp(-t / tau), rtol=1e-12, atol=1e-12)


def test_flux_evolution_constant_damping_and_source_is_exact() -> None:
    rho = np.linspace(0.0, 1.0, 4)
    psi0 = np.full(4, 1.0)
    tau = 5.0
    gamma = 1.0 / tau
    source = 0.4  # R_null=1.0, E_theta=0.4, eta=0 -> S=0.4.
    dt, n_steps = 0.25, 10
    traj = solve_flux_evolution_nonadiabatic(
        rho,
        psi0,
        tau_psi_fn=lambda t, r: np.full(r.size, tau),
        R_null_t=lambda t: 1.0,
        E_theta_t=lambda t, r: np.full(r.size, source),
        eta_spitzer_fn=lambda r: np.zeros(r.size),
        J_theta_t=lambda t, r: np.zeros(r.size),
        dt=dt,
        n_steps=n_steps,
    )
    psi_ss = source / gamma  # integrating-factor steady state = S*tau.
    for n in range(n_steps + 1):
        t = dt * n
        expected = psi_ss + (1.0 - psi_ss) * np.exp(-gamma * t)
        assert np.allclose(traj.psi[n], expected, rtol=1e-12, atol=1e-12)


def test_flux_evolution_mixed_damping_scalar_inputs_and_one_arg_tau() -> None:
    rho = np.linspace(0.0, 1.0, 4)
    psi0 = 3.0  # scalar -> broadcast by _as_profile.
    dt, n_steps = 0.1, 3

    def tau_one_arg(t: float) -> np.ndarray:
        # Single-argument callable -> exercises the _call_tau TypeError fallback.
        return np.array([np.inf, 10.0, 10.0, 10.0])

    traj = solve_flux_evolution_nonadiabatic(
        rho,
        psi0,
        tau_psi_fn=tau_one_arg,
        R_null_t=lambda t: 2.0,
        E_theta_t=lambda t, r: 0.5,  # scalar profile
        eta_spitzer_fn=lambda r: 0.0,
        J_theta_t=lambda t, r: 0.0,
        dt=dt,
        n_steps=n_steps,
    )
    assert np.allclose(traj.psi[0], 3.0)
    # Cell 0 is undamped (tau = inf); the others damp at 1/tau.
    assert traj.damping_rate[0, 0] == 0.0
    assert np.allclose(traj.damping_rate[0, 1:], 0.1)
    # Undamped cell with constant source S = 2.0*0.5 = 1.0 ramps linearly.
    assert np.allclose(traj.psi[n_steps, 0], 3.0 + n_steps * dt * 1.0, atol=1e-12)
    # Damped cells relax below the undamped ramp.
    assert np.all(traj.psi[n_steps, 1:] < traj.psi[n_steps, 0])


def test_flux_evolution_rejects_nonpositive_dt() -> None:
    rho = np.linspace(0.0, 1.0, 4)
    psi0 = np.zeros(4)
    kwargs = _valid_flux_kwargs()
    kwargs["dt"] = 0.0
    with pytest.raises(ValueError, match="dt must be positive"):
        solve_flux_evolution_nonadiabatic(rho, psi0, **kwargs)  # type: ignore[arg-type]
    kwargs["dt"] = np.nan
    with pytest.raises(ValueError, match="dt must be positive"):
        solve_flux_evolution_nonadiabatic(rho, psi0, **kwargs)  # type: ignore[arg-type]


def test_flux_evolution_rejects_nonpositive_n_steps() -> None:
    rho = np.linspace(0.0, 1.0, 4)
    psi0 = np.zeros(4)
    kwargs = _valid_flux_kwargs()
    kwargs["n_steps"] = 0
    with pytest.raises(ValueError, match="n_steps must be positive"):
        solve_flux_evolution_nonadiabatic(rho, psi0, **kwargs)  # type: ignore[arg-type]


def test_flux_evolution_rejects_bad_tau() -> None:
    rho = np.linspace(0.0, 1.0, 4)
    psi0 = np.zeros(4)
    kwargs = _valid_flux_kwargs()
    # A finite non-positive tau passes the _as_profile finite/+inf gate and is
    # rejected by the solver's own positivity contract (line 106).
    kwargs["tau_psi_fn"] = lambda t, r: -5.0
    with pytest.raises(ValueError, match="positive finite values or np.inf"):
        solve_flux_evolution_nonadiabatic(rho, psi0, **kwargs)  # type: ignore[arg-type]
    kwargs["tau_psi_fn"] = lambda t, r: np.zeros(r.size)
    with pytest.raises(ValueError, match="positive finite values or np.inf"):
        solve_flux_evolution_nonadiabatic(rho, psi0, **kwargs)  # type: ignore[arg-type]


def test_flux_evolution_rejects_bad_r_null() -> None:
    rho = np.linspace(0.0, 1.0, 4)
    psi0 = np.zeros(4)
    kwargs = _valid_flux_kwargs()
    kwargs["R_null_t"] = lambda t: -1.0
    with pytest.raises(ValueError, match="finite non-negative radius"):
        solve_flux_evolution_nonadiabatic(rho, psi0, **kwargs)  # type: ignore[arg-type]
    kwargs["R_null_t"] = lambda t: np.nan
    with pytest.raises(ValueError, match="finite non-negative radius"):
        solve_flux_evolution_nonadiabatic(rho, psi0, **kwargs)  # type: ignore[arg-type]


def test_flux_evolution_rejects_negative_eta() -> None:
    rho = np.linspace(0.0, 1.0, 4)
    psi0 = np.zeros(4)
    kwargs = _valid_flux_kwargs()
    kwargs["eta_spitzer_fn"] = lambda r: np.full(r.size, -1.0)
    with pytest.raises(ValueError, match="non-negative resistivity"):
        solve_flux_evolution_nonadiabatic(rho, psi0, **kwargs)  # type: ignore[arg-type]


def test_validate_radial_grid_errors() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        _validate_radial_grid(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="at least two points"):
        _validate_radial_grid(np.array([0.0]))
    with pytest.raises(ValueError, match="finite values"):
        _validate_radial_grid(np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_radial_grid(np.array([1.0, 0.0]))
    grid = _validate_radial_grid(np.array([0.0, 0.5, 1.0]))
    assert grid.dtype == np.float64
    assert grid.shape == (3,)


def test_as_profile_scalar_broadcast_and_errors() -> None:
    broadcast = _as_profile(2.0, 3, "x")
    assert broadcast.shape == (3,)
    assert np.all(broadcast == 2.0)
    with pytest.raises(ValueError, match=r"x must be scalar or shape \(3,\)"):
        _as_profile(np.array([1.0, 2.0]), 3, "x")
    with pytest.raises(ValueError, match="x must contain finite values"):
        _as_profile(np.array([1.0, np.nan, 2.0]), 3, "x")
    allowed = _as_profile(np.array([np.inf, 1.0]), 2, "tau", allow_infinite=True)
    assert np.isposinf(allowed[0])
    with pytest.raises(ValueError, match="finite values or np.inf"):
        _as_profile(np.array([-np.inf, 1.0]), 2, "tau", allow_infinite=True)


def test_call_tau_two_arg_and_one_arg_dispatch() -> None:
    rho = np.array([0.0, 1.0])
    two_arg = _call_tau(lambda t, r: r * 2.0, 0.0, rho)
    assert np.allclose(two_arg, rho * 2.0)
    one_arg = _call_tau(lambda t: 5.0, 0.0, rho)
    assert one_arg == 5.0


def test_q_from_psi_flat_profile_is_degenerate() -> None:
    # A flat psi makes dpsi/drho and d2psi/drho2 vanish everywhere, exercising
    # both the |denom|<eps fallback (q[i]=q[i-1], q[1]=1.0) and the axis
    # |d2psi|<eps fallback (q[0]=q[1]).
    rho = np.linspace(0.0, 1.0, 5)
    psi = np.full(5, 0.3)
    q = q_from_psi(rho, psi, R0=2.0, a=0.5, B0=1.0)
    assert q.shape == (5,)
    assert np.allclose(q, 1.0)


def test_q_from_psi_linear_profile_uses_axis_lhopital_fallback() -> None:
    # A linear psi keeps dpsi/drho finite (main branch) but zeroes d2psi/drho2,
    # so the on-axis L'Hopital term falls back to q[0] = q[1].
    rho = np.linspace(0.0, 1.0, 5)
    psi = -0.7 * rho
    q = q_from_psi(rho, psi, R0=2.0, a=0.5, B0=1.0)
    assert q[0] == q[1]
    assert np.all(q > 0.0)
