# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pulsed Hall-MHD Tests
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

from scpn_fusion.core import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, FRCEquilibriumState, MU_0
from scpn_fusion.core.hall_mhd_pulsed import (
    HallMHDPulsedConfig,
    axial_field_from_flux,
    faraday_e_theta_from_b_ramp,
    gkeyll_small_hall_acceptance_status,
    initial_hall_mhd_pulsed_state,
    ono_fig4_acceptance_status,
    run_hall_mhd_pulsed,
    spitzer_resistivity_ohm_m,
    step_hall_mhd_pulsed,
)


def _equilibrium() -> FRCEquilibriumState:
    t_i = 10_000.0
    t_e = 5_000.0
    b_ext = 5.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    rho: NDArray[np.float64] = np.linspace(0.0, 0.4, 129, dtype=np.float64)
    return solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.2,
            B_ext=b_ext,
            delta=0.02,
        ),
        rho,
    )


def _config(**kwargs: object) -> HallMHDPulsedConfig:
    eq = _equilibrium()
    values = {
        "equilibrium": eq,
        "B_ext_t": lambda t: 5.0 + 1.0e5 * t,
        "tau_psi_s": 5.0e-6,
        "electron_temperature_eV": 5_000.0,
    }
    values.update(kwargs)
    return HallMHDPulsedConfig(**values)


def test_faraday_drive_matches_circular_loop_formula() -> None:
    rho: NDArray[np.float64] = np.array([0.0, 0.1, 0.2], dtype=np.float64)

    e_theta = faraday_e_theta_from_b_ramp(rho, lambda t: 5.0 + 2.0e4 * t, 1.0e-6)

    np.testing.assert_allclose(e_theta, -0.5 * rho * 2.0e4, rtol=1.0e-10, atol=1.0e-9)


def test_spitzer_resistivity_temperature_scaling() -> None:
    eta_100 = spitzer_resistivity_ohm_m(100.0)[()]
    eta_400 = spitzer_resistivity_ohm_m(400.0)[()]

    assert eta_400 == pytest.approx(eta_100 / 8.0, rel=1.0e-14)


def test_initial_state_reconstructs_axial_field_from_flux() -> None:
    config = _config(B_ext_t=lambda _t: 5.0)

    state = initial_hall_mhd_pulsed_state(config)

    np.testing.assert_allclose(state.B_z, axial_field_from_flux(config.equilibrium.rho, state.psi))
    assert state.closure_status == "axisymmetric_ono_flux"
    assert state.external_parity_status == "blocked_missing_public_same_case_reference"


def test_implicit_damping_matches_closed_form_decay() -> None:
    config = _config(
        B_ext_t=lambda _t: 5.0,
        tau_psi_s=2.0e-6,
        E_theta_t=lambda _t, rho: np.zeros_like(rho),
        J_theta_t=lambda _t, rho: np.zeros_like(rho),
    )
    state = initial_hall_mhd_pulsed_state(config)
    dt = 1.0e-7
    n_steps = 12

    trajectory = run_hall_mhd_pulsed(state, config, dt, n_steps)

    expected = state.psi / (1.0 + dt / config.tau_psi_s) ** n_steps
    np.testing.assert_allclose(trajectory[-1].psi, expected, rtol=1.0e-12, atol=0.0)
    assert trajectory[-1].energy_proxy_J_m < state.energy_proxy_J_m


def test_hall_drive_increases_flux_under_positive_e_theta() -> None:
    config = _config(
        B_ext_t=lambda _t: 5.0,
        tau_psi_s=1.0e9,
        E_theta_t=lambda _t, rho: np.full_like(rho, 2.5),
        J_theta_t=lambda _t, rho: np.zeros_like(rho),
    )
    state = initial_hall_mhd_pulsed_state(config)

    next_state = step_hall_mhd_pulsed(state, config, 1.0e-8)

    assert float(np.mean(next_state.psi - state.psi)) > 0.0
    assert next_state.hall_drive_l2 > 0.0
    assert next_state.source_residual_linf < 1.0e-9


def test_small_hall_limit_recovers_resistive_sink() -> None:
    j_value = 2.0e5
    config = _config(
        B_ext_t=lambda _t: 5.0,
        tau_psi_s=1.0e9,
        hall_scale=0.0,
        E_theta_t=lambda _t, rho: np.full_like(rho, 10.0),
        J_theta_t=lambda _t, rho: np.full_like(rho, j_value),
    )
    state = initial_hall_mhd_pulsed_state(config)
    dt = 1.0e-8

    next_state = step_hall_mhd_pulsed(state, config, dt)

    eta = spitzer_resistivity_ohm_m(config.electron_temperature_eV)[()]
    expected = (state.psi - dt * eta * j_value) / (1.0 + dt / config.tau_psi_s)
    np.testing.assert_allclose(next_state.psi, expected, rtol=1.0e-12, atol=1.0e-18)
    assert next_state.hall_drive_l2 == 0.0


def test_external_reference_statuses_are_fail_closed() -> None:
    assert (
        gkeyll_small_hall_acceptance_status()["status"]
        == "blocked_missing_public_same_case_reference"
    )
    assert ono_fig4_acceptance_status()["status"] == "blocked_missing_public_digitised_reference"


def test_hall_mhd_inputs_fail_closed() -> None:
    config = _config()
    state = initial_hall_mhd_pulsed_state(config)

    with pytest.raises(ValueError, match="positive"):
        step_hall_mhd_pulsed(state, config, 0.0)
    with pytest.raises(ValueError, match="positive"):
        spitzer_resistivity_ohm_m(0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        axial_field_from_flux(np.array([0.0, 0.0, 1.0]), np.ones(3))
    with pytest.raises(ValueError, match="positive"):
        bad_config = HallMHDPulsedConfig(
            equilibrium=config.equilibrium,
            B_ext_t=lambda _t: 5.0,
            tau_psi_s=0.0,
            electron_temperature_eV=5_000.0,
        )
        initial_hall_mhd_pulsed_state(bad_config)
