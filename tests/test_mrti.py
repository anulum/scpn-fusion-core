# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MRTI Tests
from __future__ import annotations

from math import isclose

import numpy as np
from numpy.typing import NDArray
import pytest

from scpn_fusion.core import (
    RigidRotorFRCInputs,
    solve_frc_equilibrium,
    MRTISpectrumTracker,
    effective_acceleration_from_pulsed_compression,
    effective_acceleration_from_radius_rate,
    mrti_growth_rate,
    track_mrti_from_pulsed_compression,
)
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0
from scpn_fusion.core.mrti import MU_0 as MRTI_MU_0
from scpn_fusion.core.pulsed_compression import (
    CoilGeometry,
    PulsedCompressionConfig,
    initial_pulsed_compression_state,
    run_pulsed_compression,
)


def test_mrti_growth_rate_hydrodynamic_limit() -> None:
    k_modes: NDArray[np.float64] = np.array([1.0, 4.0, 16.0], dtype=np.float64)
    acceleration = 2.5e6

    gamma = mrti_growth_rate(k_modes, acceleration, B_perp=0.0, rho_kg_m3=1.0e-3)

    np.testing.assert_allclose(gamma, np.sqrt(k_modes * acceleration), rtol=1.0e-12, atol=0.0)


def test_mrti_growth_rate_magnetic_tension_stabilizes_short_modes() -> None:
    density = 1.2e-3
    b_perp = 1.0e-3
    acceleration = 8.0e6
    k_modes: NDArray[np.float64] = np.array([1.0, 2.0e7], dtype=np.float64)
    raw = k_modes * acceleration - (k_modes * k_modes * b_perp * b_perp) / (MRTI_MU_0 * density)

    gamma = mrti_growth_rate(k_modes, acceleration, B_perp=b_perp, rho_kg_m3=density)

    assert gamma[0] > 0.0
    assert raw[1] < 0.0
    assert gamma[1] == 0.0


def test_spectrum_tracker_matches_constant_acceleration_exponential() -> None:
    tracker = MRTISpectrumTracker(k_modes_m_inv=[2.0, 8.0], initial_perturbation_m=2.0e-9)
    dt = 2.5e-7
    steps = 12
    acceleration = 4.0e6

    state = tracker.state()
    for _ in range(steps):
        state = tracker.step(dt, acceleration)

    gamma = np.sqrt(np.array([2.0, 8.0], dtype=np.float64) * acceleration)
    expected = 2.0e-9 * np.exp(gamma * dt * steps)
    np.testing.assert_allclose(state.amplitudes_m, expected, rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(state.log_amplitudes, np.log(expected), rtol=1.0e-12, atol=1.0e-12)
    assert state.fastest_growing_k_m_inv == 8.0
    assert not state.saturation_warning
    assert state.time_of_breach_s is None
    assert state.amplitude_overflow_limited is False


def test_spectrum_tracker_keeps_extreme_growth_finite_in_log_space() -> None:
    tracker = MRTISpectrumTracker(k_modes_m_inv=[1.0, 4.0], initial_perturbation_m=1.0e-9)

    state = tracker.step(1.0, 1.0e8)

    assert np.all(np.isfinite(state.amplitudes_m))
    assert np.isfinite(state.max_amplitude_m)
    assert np.all(np.isfinite(state.log_amplitudes))
    assert state.max_log_amplitude > np.log(np.finfo(np.float64).max)
    assert state.amplitude_overflow_limited is True
    assert state.saturation_warning is True


def test_spectrum_tracker_records_first_saturation_breach() -> None:
    tracker = MRTISpectrumTracker(
        k_modes_m_inv=[10.0, 40.0],
        initial_perturbation_m=1.0e-8,
        saturation_threshold_m=1.0e-6,
    )

    state = tracker.step(1.0e-4, 1.0e8)

    assert state.saturation_warning
    assert state.time_of_breach_s == pytest.approx(state.t_s)
    assert tracker.saturation_threshold_breached()


def test_effective_acceleration_from_radius_rate_recovers_linear_ramp() -> None:
    time_s: NDArray[np.float64] = np.linspace(0.0, 1.0e-6, 9, dtype=np.float64)
    acceleration = -3.25e11
    speed = 5.0e4 + acceleration * time_s

    estimated = effective_acceleration_from_radius_rate(time_s, speed, smoothing_window=3)

    np.testing.assert_allclose(estimated, acceleration, rtol=1.0e-12, atol=2.0e-4)


def test_pulsed_compression_trajectory_drives_mrti_tracker() -> None:
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
    config = PulsedCompressionConfig(
        equilibrium=equilibrium,
        coil=CoilGeometry(
            N_turns=80,
            L_coil_m=1.0,
            R_coil_m=0.35,
            L_inductance_H=2.0e-6,
            R_resistance_ohm=0.02,
            bank_voltage_max_V=20_000.0,
        ),
        coil_current_t=lambda _t: 5.0e5,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=t_i,
        electron_temperature_eV=t_e,
        tau_psi_s=np.inf,
    )
    states = run_pulsed_compression(initial_pulsed_compression_state(config), config, 1.0e-9, 16)

    acceleration = effective_acceleration_from_pulsed_compression(states)
    tracker = MRTISpectrumTracker(k_max_m_inv=1.0e4, n_modes=64)
    snapshots = track_mrti_from_pulsed_compression(states, tracker)

    assert len(acceleration) == len(states)
    assert len(snapshots) == len(states) - 1
    assert np.all(np.isfinite(acceleration))
    assert float(np.max(acceleration)) > 0.0
    assert snapshots[-1].t_s == pytest.approx(states[-1].t_s)
    assert snapshots[-1].max_amplitude_m >= snapshots[0].max_amplitude_m


def test_mrti_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mrti_growth_rate([-1.0], 1.0)
    with pytest.raises(ValueError, match="positive"):
        mrti_growth_rate([1.0], 1.0, rho_kg_m3=0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        effective_acceleration_from_radius_rate([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="positive odd"):
        effective_acceleration_from_radius_rate([0.0, 1.0], [1.0, 2.0], smoothing_window=2)
    with pytest.raises(ValueError, match="at least two pulsed-compression"):
        effective_acceleration_from_pulsed_compression([])
    with pytest.raises(ValueError, match="at least 2"):
        MRTISpectrumTracker(k_max_m_inv=10.0, n_modes=1)
    assert isclose(float(mrti_growth_rate(4.0, 9.0)[()]), 6.0)
