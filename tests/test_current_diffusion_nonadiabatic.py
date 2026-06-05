# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Non-Adiabatic Flux Evolution Tests
from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import pytest

from scpn_fusion.core.current_diffusion import solve_flux_evolution_nonadiabatic

FloatArray: TypeAlias = NDArray[np.float64]


def _zero_profile(_: float, rho: FloatArray) -> FloatArray:
    return np.zeros_like(rho)


def _zero_eta(rho: FloatArray) -> FloatArray:
    return np.zeros_like(rho)


def test_zero_drive_zero_damping_preserves_flux_exactly() -> None:
    rho: FloatArray = np.linspace(0.0, 1.0, 32)
    psi0: FloatArray = -0.2 * rho**2 + 0.03 * rho

    trajectory = solve_flux_evolution_nonadiabatic(
        rho,
        psi0,
        tau_psi_fn=lambda _: np.inf,
        R_null_t=lambda _: 0.0,
        E_theta_t=_zero_profile,
        eta_spitzer_fn=_zero_eta,
        J_theta_t=_zero_profile,
        dt=1.0e-7,
        n_steps=8,
    )

    assert trajectory.psi.shape == (9, rho.size)
    assert trajectory.source_increment.shape == (8, rho.size)
    assert trajectory.damping_decrement.shape == (8, rho.size)
    assert trajectory.update_residual.shape == (8, rho.size)
    np.testing.assert_array_equal(trajectory.psi[0], psi0)
    np.testing.assert_array_equal(trajectory.psi[-1], psi0)
    np.testing.assert_array_equal(trajectory.source, np.zeros_like(trajectory.source))
    np.testing.assert_array_equal(
        trajectory.psi[1:],
        trajectory.psi[:-1] - trajectory.damping_decrement + trajectory.source_increment,
    )
    np.testing.assert_array_equal(
        trajectory.update_residual,
        np.zeros_like(trajectory.update_residual),
    )


def test_positive_hall_drive_increases_flux_by_integrated_source() -> None:
    rho: FloatArray = np.linspace(0.0, 1.0, 24)
    psi0 = np.zeros_like(rho)
    r_null = 0.18

    def e_theta(_: float, grid: FloatArray) -> FloatArray:
        return 20.0 + 3.0 * grid

    trajectory = solve_flux_evolution_nonadiabatic(
        rho,
        psi0,
        tau_psi_fn=lambda _: np.inf,
        R_null_t=lambda _: r_null,
        E_theta_t=e_theta,
        eta_spitzer_fn=_zero_eta,
        J_theta_t=_zero_profile,
        dt=2.0e-8,
        n_steps=5,
    )

    expected = 5 * 2.0e-8 * r_null * e_theta(0.0, rho)
    np.testing.assert_allclose(trajectory.psi[-1], expected, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        np.sum(trajectory.source_increment, axis=0),
        expected,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.all(trajectory.hall_drive > 0.0)
    assert np.all(trajectory.resistive_loss == 0.0)


def test_resistive_loss_decreases_flux_under_positive_current() -> None:
    rho: FloatArray = np.linspace(0.0, 1.0, 20)
    psi0 = np.ones_like(rho) * 0.5
    eta = 2.5e-6
    current = 1.2e5
    dt = 1.0e-8
    n_steps = 6

    trajectory = solve_flux_evolution_nonadiabatic(
        rho,
        psi0,
        tau_psi_fn=lambda _: np.inf,
        R_null_t=lambda _: 0.0,
        E_theta_t=_zero_profile,
        eta_spitzer_fn=lambda grid: np.full_like(grid, eta),
        J_theta_t=lambda _, grid: np.full_like(grid, current),
        dt=dt,
        n_steps=n_steps,
    )

    expected = psi0 - n_steps * dt * eta * current
    np.testing.assert_allclose(trajectory.psi[-1], expected, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        np.sum(trajectory.source_increment, axis=0),
        -n_steps * dt * eta * current,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.all(trajectory.resistive_loss > 0.0)


def test_damping_recovers_constant_tau_exponential_solution() -> None:
    rho: FloatArray = np.linspace(0.0, 1.0, 18)
    psi0: FloatArray = 0.2 + 0.1 * rho
    tau = 4.0e-7
    dt = 1.0e-8
    n_steps = 12

    trajectory = solve_flux_evolution_nonadiabatic(
        rho,
        psi0,
        tau_psi_fn=lambda _: tau,
        R_null_t=lambda _: 0.0,
        E_theta_t=_zero_profile,
        eta_spitzer_fn=_zero_eta,
        J_theta_t=_zero_profile,
        dt=dt,
        n_steps=n_steps,
    )

    expected = psi0 * np.exp(-trajectory.time_s[-1] / tau)
    np.testing.assert_allclose(trajectory.psi[-1], expected, rtol=1.0e-14, atol=1.0e-15)
    np.testing.assert_allclose(
        np.sum(trajectory.damping_decrement, axis=0),
        psi0 - expected,
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_flux_update_budget_closes_for_mixed_drive_and_damping() -> None:
    rho: FloatArray = np.linspace(0.0, 1.0, 21)
    psi0: FloatArray = 0.25 - 0.04 * rho**2
    dt = 5.0e-9

    trajectory = solve_flux_evolution_nonadiabatic(
        rho,
        psi0,
        tau_psi_fn=lambda _t, grid: 3.0e-7 + 1.0e-7 * grid,
        R_null_t=lambda t: 0.18 + 0.01 * t / dt,
        E_theta_t=lambda t, grid: 12.0 + t / dt + 0.5 * grid,
        eta_spitzer_fn=lambda grid: 1.0e-6 * (1.0 + 120.0 * grid),
        J_theta_t=lambda t, grid: 8.0e4 * (1.0 - 0.4 * grid) + 1.0e3 * t / dt,
        dt=dt,
        n_steps=7,
    )

    reconstructed = trajectory.psi[:-1] - trajectory.damping_decrement + trajectory.source_increment
    np.testing.assert_allclose(trajectory.psi[1:], reconstructed, rtol=0.0, atol=1.0e-15)
    assert float(np.max(np.abs(trajectory.update_residual))) <= 1.0e-15
    assert np.any(trajectory.source_increment > 0.0)
    assert np.any(trajectory.source_increment < 0.0)
    assert np.all(np.isfinite(trajectory.damping_decrement))


def test_nonadiabatic_flux_input_validation() -> None:
    invalid_cases: list[tuple[FloatArray, FloatArray, Callable[[float], float], float, int]] = [
        (np.array([0.0, 0.5, 0.4]), np.zeros(3), lambda _: np.inf, 1.0e-8, 1),
        (np.linspace(0.0, 1.0, 4), np.zeros(3), lambda _: np.inf, 1.0e-8, 1),
        (np.linspace(0.0, 1.0, 4), np.zeros(4), lambda _: -1.0, 1.0e-8, 1),
        (np.linspace(0.0, 1.0, 4), np.zeros(4), lambda _: np.inf, 0.0, 1),
        (np.linspace(0.0, 1.0, 4), np.zeros(4), lambda _: np.inf, 1.0e-8, 0),
    ]

    for rho, psi0, tau, dt, n_steps in invalid_cases:
        with pytest.raises(ValueError):
            solve_flux_evolution_nonadiabatic(
                rho,
                psi0,
                tau_psi_fn=tau,
                R_null_t=lambda _: 0.0,
                E_theta_t=_zero_profile,
                eta_spitzer_fn=_zero_eta,
                J_theta_t=_zero_profile,
                dt=dt,
                n_steps=n_steps,
            )
