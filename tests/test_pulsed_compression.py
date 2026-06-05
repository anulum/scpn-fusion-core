# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pulsed Compression Tests
from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
import pytest

from scpn_fusion.core import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import FRCEquilibriumState
from scpn_fusion.core.pulsed_compression import (
    CoilGeometry,
    PulsedCompressionConfig,
    adiabatic_temperature_update_eV,
    coil_field_t,
    initial_coil_circuit_state,
    initial_pulsed_compression_state,
    plasma_volume_m3,
    run_coil_circuit,
    run_pulsed_compression,
    run_voltage_driven_pulsed_compression,
    slough_fig5_acceptance_status,
    spitzer_resistivity_ohm_m,
    step_coil_circuit,
    step_pulsed_compression,
)
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0


def _equilibrium() -> FRCEquilibriumState:
    b_ext = 5.0
    t_i = 10_000.0
    t_e = 5_000.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    rho: NDArray[np.float64] = np.linspace(0.0, 0.4, 65, dtype=np.float64)
    return cast(
        FRCEquilibriumState,
        solve_frc_equilibrium(
            RigidRotorFRCInputs(
                n0=n0, T_i_eV=t_i, T_e_eV=t_e, theta_dot=0.0, R_s=0.2, B_ext=b_ext, delta=0.02
            ),
            rho,
        ),
    )


def _coil(current: float = 2.0e5) -> tuple[CoilGeometry, float]:
    coil = CoilGeometry(
        N_turns=80,
        L_coil_m=1.0,
        R_coil_m=0.35,
        L_inductance_H=2.0e-6,
        R_resistance_ohm=0.02,
        bank_voltage_max_V=20_000.0,
    )
    return coil, current


def _config(current: float = 2.0e5) -> PulsedCompressionConfig:
    coil, current_a = _coil(current)
    return PulsedCompressionConfig(
        equilibrium=_equilibrium(),
        coil=coil,
        coil_current_t=lambda _t: current_a,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=10_000.0,
        electron_temperature_eV=5_000.0,
        tau_psi_s=np.inf,
    )


def test_coil_field_matches_uniform_solenoid_formula() -> None:
    coil, current = _coil()
    expected = MU_0 * coil.N_turns * current / coil.L_coil_m

    assert coil_field_t(coil, current) == pytest.approx(expected, rel=1.0e-15)


def test_coil_circuit_matches_exact_rl_solution_and_energy_balance() -> None:
    coil, _current = _coil()
    voltage = 12_000.0
    dt_s = 25.0e-9
    initial_current = 2.5e5
    initial = initial_coil_circuit_state(coil, initial_current)

    next_state = step_coil_circuit(initial, coil, voltage, dt_s)

    tau = coil.L_inductance_H / coil.R_resistance_ohm
    expected = voltage / coil.R_resistance_ohm + (
        initial_current - voltage / coil.R_resistance_ohm
    ) * np.exp(-dt_s / tau)
    assert next_state.current_A == pytest.approx(expected, rel=1.0e-14)
    assert next_state.magnetic_energy_J > initial.magnetic_energy_J
    assert next_state.ohmic_loss_J > 0.0
    assert next_state.source_work_J > 0.0
    assert abs(next_state.energy_balance_residual) < 1.0e-12


def test_adiabatic_temperature_preserves_invariant() -> None:
    old_volume = plasma_volume_m3(0.2, 1.0)
    new_volume = plasma_volume_m3(0.15, 1.0)
    temperature = 1000.0
    gamma = 5.0 / 3.0

    updated = adiabatic_temperature_update_eV(temperature, old_volume, new_volume, gamma)

    assert updated * new_volume ** (gamma - 1.0) == pytest.approx(
        temperature * old_volume ** (gamma - 1.0),
        rel=1.0e-14,
    )


def test_pressure_imbalance_accelerates_radially() -> None:
    config = _config(current=1.0)
    state = initial_pulsed_compression_state(config)

    next_state = step_pulsed_compression(state, config, 1.0e-9)

    force_area = 2.0 * np.pi * state.R_s_m * config.plasma_length_m
    expected_acceleration = (
        (state.internal_pressure_Pa - next_state.external_magnetic_pressure_Pa)
        * force_area
        / config.plasma_mass_kg
    )
    assert next_state.dR_s_dt_m_s > state.dR_s_dt_m_s
    assert next_state.radial_acceleration_m_s2 == pytest.approx(expected_acceleration)
    assert next_state.radial_acceleration_m_s2 > 0.0
    assert next_state.R_s_m > state.R_s_m
    assert next_state.flux_coupling_status == "ono_nonadiabatic_flux_carrier"
    assert next_state.flux_budget_claim_status == "passed"
    assert next_state.flux_update_residual_abs_max <= 1.0e-12
    assert abs(next_state.energy_balance_residual) < 1.0e-12


def test_external_compression_heats_and_shrinks_plasma() -> None:
    config = _config(current=5.0e5)
    state = initial_pulsed_compression_state(config)

    next_state = step_pulsed_compression(state, config, 2.0e-8)

    assert next_state.R_s_m < state.R_s_m
    assert next_state.radial_acceleration_m_s2 < 0.0
    assert next_state.T_i_eV > state.T_i_eV
    assert next_state.T_e_eV > state.T_e_eV
    assert next_state.compression_work_J > state.compression_work_J
    assert abs(next_state.energy_balance_residual) < 1.0e-12


def test_run_pulsed_compression_tracks_flux_history() -> None:
    config = _config(current=5.0e5)
    initial = initial_pulsed_compression_state(config)

    states = run_pulsed_compression(initial, config, 1.0e-8, 4)

    assert len(states) == 5
    assert states[-1].t_s == pytest.approx(4.0e-8)
    assert all(state.flux_psi.shape == initial.flux_psi.shape for state in states)
    assert all(np.isfinite(state.radial_acceleration_m_s2) for state in states)
    assert np.isfinite(states[-1].flux_psi_checksum)
    assert all(state.flux_budget_claim_status == "passed" for state in states[1:])
    assert max(state.flux_update_residual_abs_max for state in states[1:]) <= 1.0e-12


def test_flux_budget_exposes_source_damping_and_residual() -> None:
    config = PulsedCompressionConfig(
        equilibrium=_equilibrium(),
        coil=_coil(current=1.0)[0],
        coil_current_t=lambda _t: 1.0,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=10_000.0,
        electron_temperature_eV=5_000.0,
        tau_psi_s=np.inf,
        E_theta_t=lambda _t, rho: np.full_like(rho, 2.0),
        J_theta_t=lambda _t, rho: np.zeros_like(rho),
    )
    initial = initial_pulsed_compression_state(config)

    next_state = step_pulsed_compression(initial, config, 1.0e-9)

    expected_source_checksum = float(
        np.sum(np.full_like(initial.flux_psi, next_state.R_s_m * 2.0e-9))
    )
    assert next_state.flux_coupling_status == "ono_nonadiabatic_flux_carrier"
    assert next_state.flux_budget_claim_status == "passed"
    assert next_state.flux_source_increment_checksum == pytest.approx(expected_source_checksum)
    assert next_state.flux_damping_decrement_checksum == pytest.approx(0.0)
    assert next_state.flux_update_residual_abs_max <= 1.0e-12


def test_voltage_driven_coil_circuit_feeds_pulsed_compression() -> None:
    config = _config(current=1.0)

    result = run_voltage_driven_pulsed_compression(
        config,
        lambda _t: 20_000.0,
        1.0e-9,
        32,
        initial_current_A=5.0e5,
    )

    assert len(result.coil_circuit) == 33
    assert len(result.compression) == 33
    assert result.coil_circuit[-1].current_A > result.coil_circuit[0].current_A
    assert result.compression[-1].B_ext_T == pytest.approx(
        coil_field_t(config.coil, result.coil_circuit[-2].current_A),
        rel=1.0e-14,
    )
    assert result.compression[-1].R_s_m < result.compression[0].R_s_m
    assert result.compression[-1].compression_work_J > 0.0
    assert abs(result.coil_circuit[-1].energy_balance_residual) < 1.0e-12


def test_spitzer_resistivity_scaling_and_blocked_reference_status() -> None:
    cold = float(spitzer_resistivity_ohm_m(100.0)[()])
    hot = float(spitzer_resistivity_ohm_m(400.0)[()])

    assert cold / hot == pytest.approx((400.0 / 100.0) ** 1.5, rel=1.0e-14)
    assert slough_fig5_acceptance_status()["status"] == "blocked_missing_public_digitised_reference"


def test_pulsed_compression_inputs_fail_closed() -> None:
    config = _config()
    with pytest.raises(ValueError, match="finite"):
        coil_field_t(config.coil, np.inf)
    with pytest.raises(ValueError, match="positive"):
        adiabatic_temperature_update_eV(1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        run_pulsed_compression(initial_pulsed_compression_state(config), config, 1.0e-8, 0)
    with pytest.raises(ValueError, match="bank_voltage"):
        run_coil_circuit(config.coil, lambda _t: config.coil.bank_voltage_max_V + 1.0, 1.0e-9, 1)
