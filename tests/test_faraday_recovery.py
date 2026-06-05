# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Faraday Recovery Tests
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from scpn_fusion.core import (
    CoilGeometry,
    FaradayRecoveryTrajectoryPoint,
    RigidRotorFRCInputs,
    coil_source_work_from_voltage_driven_compression,
    compression_flux_budget_from_pulsed_compression,
    compression_flux_budget_from_voltage_driven_compression,
    compression_work_from_pulsed_compression,
    compression_work_from_voltage_driven_compression,
    faraday_back_emf,
    faraday_back_emf_from_values,
    faraday_trajectory_from_pulsed_compression,
    faraday_trajectory_from_voltage_driven_compression,
    initial_pulsed_compression_state,
    integrated_recovery_energy,
    magnetic_flux_wb,
    run_pulsed_compression,
    run_voltage_driven_pulsed_compression,
    solve_frc_equilibrium,
)
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0
from scpn_fusion.core.pulsed_compression import PulsedCompressionConfig


def test_constant_radius_and_field_have_zero_back_emf() -> None:
    emf = faraday_back_emf(
        lambda _t: 0.20,
        lambda _t: 20.0,
        12,
        1.0e-6,
        dR_s_dt=lambda _t: 0.0,
        dB_ext_dt=lambda _t: 0.0,
    )

    assert emf == pytest.approx(0.0, abs=0.0)


def test_closed_form_constant_field_radial_expansion() -> None:
    emf = faraday_back_emf_from_values(
        separatrix_radius_m=0.20,
        b_ext_t=20.0,
        d_radius_dt_m_s=1.5e4,
        d_b_ext_dt_t_s=0.0,
        N_turns=8,
    )

    expected = -8.0 * math.pi * (2.0 * 20.0 * 0.20 * 1.5e4)
    assert emf == pytest.approx(expected, rel=1.0e-15)


def test_callable_finite_difference_matches_linear_history() -> None:
    radius_0 = 0.18
    speed = -2.0e4
    field_0 = 19.0
    field_rate = 3.0e6
    time_s = 2.5e-7

    emf = faraday_back_emf(
        lambda t: radius_0 + speed * t,
        lambda t: field_0 + field_rate * t,
        5,
        time_s,
        finite_difference_dt_s=1.0e-10,
    )
    expected = faraday_back_emf_from_values(
        radius_0 + speed * time_s,
        field_0 + field_rate * time_s,
        speed,
        field_rate,
        5,
    )

    assert emf == pytest.approx(expected, rel=1.0e-9)


def test_integrated_recovery_energy_matches_analytical_linear_radius_case() -> None:
    turns = 6
    resistance = 0.08
    b_ext = 20.0
    radius_0 = 0.15
    speed = 4.0e3
    duration = 1.0e-6
    times = np.linspace(0.0, duration, 257)
    trajectory = [
        FaradayRecoveryTrajectoryPoint(
            t_s=float(t), separatrix_radius_m=radius_0 + speed * t, b_ext_t=b_ext
        )
        for t in times
    ]

    report = integrated_recovery_energy(trajectory, turns, resistance)

    coefficient = turns * math.pi * 2.0 * b_ext * speed
    expected = coefficient * coefficient / resistance
    expected *= ((radius_0 + speed * duration) ** 3 - radius_0**3) / (3.0 * speed)
    assert report.recovered_energy_j == pytest.approx(expected, rel=2.0e-5)
    assert report.budget_claim_status == "blocked_missing_compression_work"
    assert report.energy_budget_passed is None
    assert report.source_budget_claim_status == "blocked_missing_coil_source_work"
    assert report.source_energy_budget_passed is None
    assert report.flux_derivative_closure_passed is True
    assert report.flux_derivative_residual_linf <= 2.0e-2
    assert len(report.flux_derivative_residual_wb_s) == len(trajectory)
    assert report.max_abs_flux_rate_field_term_wb_s == pytest.approx(0.0, abs=0.0)
    assert report.max_abs_flux_rate_radial_term_wb_s > 0.0
    assert report.max_abs_flux_rate_total_wb_s == pytest.approx(
        report.max_abs_flux_rate_radial_term_wb_s
    )
    assert report.samples[-1].flux_rate_total_wb_s == pytest.approx(
        -report.samples[-1].back_emf_v / turns
    )


def test_faraday_recovery_flags_inconsistent_derivative_sidecars() -> None:
    trajectory = [
        FaradayRecoveryTrajectoryPoint(
            t_s=float(t),
            separatrix_radius_m=0.2 + 1.0e3 * t,
            b_ext_t=20.0,
            d_radius_dt_m_s=0.0,
            d_b_ext_dt_t_s=0.0,
        )
        for t in np.linspace(0.0, 1.0e-6, 17)
    ]

    report = integrated_recovery_energy(trajectory, 8, 0.1)

    assert report.flux_derivative_closure_passed is False
    assert report.flux_derivative_residual_linf > 0.5
    assert report.compression_flux_budget_claim_status == "blocked_missing_compression_flux_budget"
    assert report.compression_flux_budget_passed is None


def test_integrated_recovery_energy_reports_budget_when_work_is_supplied() -> None:
    trajectory = [
        FaradayRecoveryTrajectoryPoint(
            t_s=0.0, separatrix_radius_m=0.20, b_ext_t=20.0, d_radius_dt_m_s=0.0, d_b_ext_dt_t_s=0.0
        ),
        FaradayRecoveryTrajectoryPoint(
            t_s=1.0e-6,
            separatrix_radius_m=0.20,
            b_ext_t=20.0,
            d_radius_dt_m_s=0.0,
            d_b_ext_dt_t_s=0.0,
        ),
    ]

    report = integrated_recovery_energy(trajectory, 4, 0.1, compression_work_j=1.0e-12)

    assert report.recovered_energy_j == pytest.approx(0.0, abs=0.0)
    assert report.energy_budget_passed is False
    assert report.budget_claim_status == "failed"
    assert report.source_budget_claim_status == "blocked_missing_coil_source_work"
    assert report.compression_flux_budget_claim_status == "blocked_missing_compression_flux_budget"


def test_faraday_recovery_evaluates_pulsed_compression_work_sidecar() -> None:
    b_ext = 5.0
    t_i = 10_000.0
    t_e = 5_000.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    equilibrium = solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.20,
            B_ext=b_ext,
            delta=0.02,
        ),
        np.linspace(0.0, 0.4, 65, dtype=np.float64),
    )
    coil = CoilGeometry(
        N_turns=80,
        L_coil_m=1.0,
        R_coil_m=0.35,
        L_inductance_H=2.0e-6,
        R_resistance_ohm=0.02,
        bank_voltage_max_V=20_000.0,
    )
    config = PulsedCompressionConfig(
        equilibrium=equilibrium,
        coil=coil,
        coil_current_t=lambda _t: 5.0e5,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=t_i,
        electron_temperature_eV=t_e,
        tau_psi_s=np.inf,
    )
    states = run_pulsed_compression(initial_pulsed_compression_state(config), config, 1.0e-9, 16)

    trajectory = faraday_trajectory_from_pulsed_compression(states)
    compression_work = compression_work_from_pulsed_compression(states)
    compression_flux_budget = compression_flux_budget_from_pulsed_compression(states)
    report = integrated_recovery_energy(
        trajectory,
        coil.N_turns,
        coil.R_resistance_ohm,
        compression_work_j=compression_work,
        compression_flux_budget=compression_flux_budget,
    )

    assert len(trajectory) == len(states)
    assert trajectory[-1].separatrix_radius_m == pytest.approx(states[-1].R_s_m)
    assert trajectory[-1].d_radius_dt_m_s == pytest.approx(states[-1].dR_s_dt_m_s)
    assert compression_work > 0.0
    assert report.compression_work_j == pytest.approx(compression_work)
    assert report.energy_budget_passed is not None
    assert report.energy_budget_relative_error is not None
    assert np.isfinite(report.energy_budget_relative_error)
    assert report.budget_claim_status in {"passed", "failed"}
    assert report.source_budget_claim_status == "blocked_missing_coil_source_work"
    assert report.source_energy_budget_passed is None
    assert compression_flux_budget.budget_claim_status == "passed"
    assert compression_flux_budget.update_residual_abs_max <= 1.0e-12
    assert report.compression_flux_budget == compression_flux_budget
    assert report.compression_flux_budget_passed is True
    assert report.compression_flux_budget_claim_status == "passed"
    assert isinstance(report.flux_derivative_closure_passed, bool)
    assert np.isfinite(report.flux_derivative_residual_linf)
    assert np.isfinite(report.flux_derivative_residual_l2)
    assert report.max_abs_flux_rate_total_wb_s > 0.0


def test_faraday_recovery_evaluates_voltage_driven_source_sidecars() -> None:
    b_ext = 5.0
    t_i = 10_000.0
    t_e = 5_000.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    equilibrium = solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.20,
            B_ext=b_ext,
            delta=0.02,
        ),
        np.linspace(0.0, 0.4, 65, dtype=np.float64),
    )
    coil = CoilGeometry(
        N_turns=80,
        L_coil_m=1.0,
        R_coil_m=0.35,
        L_inductance_H=2.0e-6,
        R_resistance_ohm=0.02,
        bank_voltage_max_V=20_000.0,
    )
    config = PulsedCompressionConfig(
        equilibrium=equilibrium,
        coil=coil,
        coil_current_t=lambda _t: 1.0,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=t_i,
        electron_temperature_eV=t_e,
        tau_psi_s=np.inf,
    )
    result = run_voltage_driven_pulsed_compression(
        config,
        lambda _t: 20_000.0,
        1.0e-9,
        32,
        initial_current_A=5.0e5,
    )

    trajectory = faraday_trajectory_from_voltage_driven_compression(result)
    compression_work = compression_work_from_voltage_driven_compression(result)
    compression_flux_budget = compression_flux_budget_from_voltage_driven_compression(result)
    source_work = coil_source_work_from_voltage_driven_compression(result)
    report = integrated_recovery_energy(
        trajectory,
        coil.N_turns,
        coil.R_resistance_ohm,
        compression_work_j=compression_work,
        coil_source_work_j=source_work,
        compression_flux_budget=compression_flux_budget,
    )

    assert len(trajectory) == len(result.compression)
    assert compression_work == pytest.approx(result.compression[-1].compression_work_J)
    assert source_work == pytest.approx(result.coil_circuit[-1].source_work_J)
    assert source_work > 0.0
    assert report.compression_work_j == pytest.approx(compression_work)
    assert report.coil_source_work_j == pytest.approx(source_work)
    assert report.energy_budget_relative_error is not None
    assert np.isfinite(report.energy_budget_relative_error)
    assert report.source_energy_budget_relative_error is not None
    assert np.isfinite(report.source_energy_budget_relative_error)
    assert report.budget_claim_status in {"passed", "failed"}
    assert report.source_budget_claim_status in {"passed", "failed"}
    assert report.compression_flux_budget_passed is True
    assert report.compression_flux_budget_claim_status == "passed"
    assert isinstance(report.flux_derivative_closure_passed, bool)
    assert np.isfinite(report.flux_derivative_residual_linf)
    assert np.isfinite(report.flux_derivative_residual_l2)
    assert report.max_abs_flux_rate_total_wb_s > 0.0


def test_faraday_recovery_propagates_failed_compression_flux_budget() -> None:
    b_ext = 5.0
    t_i = 10_000.0
    t_e = 5_000.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    equilibrium = solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.20,
            B_ext=b_ext,
            delta=0.02,
        ),
        np.linspace(0.0, 0.4, 65, dtype=np.float64),
    )
    coil = CoilGeometry(
        N_turns=80,
        L_coil_m=1.0,
        R_coil_m=0.35,
        L_inductance_H=2.0e-6,
        R_resistance_ohm=0.02,
        bank_voltage_max_V=20_000.0,
    )
    config = PulsedCompressionConfig(
        equilibrium=equilibrium,
        coil=coil,
        coil_current_t=lambda _t: 5.0e5,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=t_i,
        electron_temperature_eV=t_e,
        tau_psi_s=np.inf,
    )
    states = list(
        run_pulsed_compression(initial_pulsed_compression_state(config), config, 1.0e-9, 4)
    )
    states[-1] = replace(
        states[-1],
        flux_budget_claim_status="failed",
        flux_update_residual_abs_max=1.0e-3,
    )
    compression_flux_budget = compression_flux_budget_from_pulsed_compression(states)
    report = integrated_recovery_energy(
        faraday_trajectory_from_pulsed_compression(states),
        coil.N_turns,
        coil.R_resistance_ohm,
        compression_work_j=compression_work_from_pulsed_compression(states),
        compression_flux_budget=compression_flux_budget,
    )

    assert compression_flux_budget.budget_claim_status == "failed"
    assert report.compression_flux_budget_passed is False
    assert report.compression_flux_budget_claim_status == "failed"


def test_faraday_recovery_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        faraday_back_emf_from_values(0.2, 20.0, 1.0, 0.0, 0)
    with pytest.raises(ValueError, match="positive"):
        magnetic_flux_wb(0.0, 20.0)
    with pytest.raises(ValueError, match="at least two"):
        integrated_recovery_energy([FaradayRecoveryTrajectoryPoint(0.0, 0.2, 20.0)], 2, 0.1)
    with pytest.raises(ValueError, match="strictly increasing"):
        integrated_recovery_energy(
            [
                FaradayRecoveryTrajectoryPoint(0.0, 0.2, 20.0),
                FaradayRecoveryTrajectoryPoint(0.0, 0.21, 20.0),
            ],
            2,
            0.1,
        )
    with pytest.raises(ValueError, match="all supplied or all omitted"):
        integrated_recovery_energy(
            [
                FaradayRecoveryTrajectoryPoint(0.0, 0.2, 20.0, d_radius_dt_m_s=1.0),
                FaradayRecoveryTrajectoryPoint(1.0e-6, 0.21, 20.0),
            ],
            2,
            0.1,
        )
    with pytest.raises(ValueError, match="at least two states"):
        faraday_trajectory_from_pulsed_compression([object()])
    with pytest.raises(ValueError, match="voltage-driven compression result is missing"):
        faraday_trajectory_from_voltage_driven_compression(object())
    with pytest.raises(ValueError, match="at least two samples"):
        coil_source_work_from_voltage_driven_compression({"coil_circuit": []})
