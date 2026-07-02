# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MRTI Tests
from __future__ import annotations

from dataclasses import dataclass
import math
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
    """MRTI growth reduces to ``sqrt(k a)`` with no magnetic tension."""
    k_modes: NDArray[np.float64] = np.array([1.0, 4.0, 16.0], dtype=np.float64)
    acceleration = 2.5e6

    gamma = mrti_growth_rate(k_modes, acceleration, B_perp=0.0, rho_kg_m3=1.0e-3)

    np.testing.assert_allclose(gamma, np.sqrt(k_modes * acceleration), rtol=1.0e-12, atol=0.0)


def test_mrti_growth_rate_magnetic_tension_stabilizes_short_modes() -> None:
    """Magnetic tension clips stabilized short-wavelength modes to zero growth."""
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
    """Constant-driver tracking matches the analytical exponential update."""
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
    """Extreme growth remains finite in amplitude space while preserving log evidence."""
    tracker = MRTISpectrumTracker(k_modes_m_inv=[1.0, 4.0], initial_perturbation_m=1.0e-9)

    state = tracker.step(1.0, 1.0e8)

    assert np.all(np.isfinite(state.amplitudes_m))
    assert np.isfinite(state.max_amplitude_m)
    assert np.all(np.isfinite(state.log_amplitudes))
    assert state.max_log_amplitude > np.log(np.finfo(np.float64).max)
    assert state.amplitude_overflow_limited is True
    assert state.saturation_warning is True


def test_spectrum_tracker_records_first_saturation_breach() -> None:
    """The tracker records the first time that amplitudes breach saturation."""
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
    """Finite differences recover a linear radial-speed ramp."""
    time_s: NDArray[np.float64] = np.linspace(0.0, 1.0e-6, 9, dtype=np.float64)
    acceleration = -3.25e11
    speed = 5.0e4 + acceleration * time_s

    estimated = effective_acceleration_from_radius_rate(time_s, speed, smoothing_window=3)

    np.testing.assert_allclose(estimated, acceleration, rtol=1.0e-12, atol=2.0e-4)


def test_pulsed_compression_trajectory_drives_mrti_tracker() -> None:
    """A real pulsed-compression trajectory supplies finite MRTI driver states."""
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

    expected = np.asarray(
        [-state.radial_acceleration_m_s2 for state in states],
        dtype=np.float64,
    )
    assert len(acceleration) == len(states)
    assert len(snapshots) == len(states) - 1
    assert np.all(np.isfinite(acceleration))
    np.testing.assert_allclose(acceleration, expected, rtol=0.0, atol=0.0)
    assert float(np.max(acceleration)) > 0.0
    assert snapshots[-1].t_s == pytest.approx(states[-1].t_s)
    assert snapshots[-1].max_amplitude_m >= snapshots[0].max_amplitude_m


def test_step_interval_constant_drivers_match_frozen_step() -> None:
    """Interval integration reduces exactly to frozen stepping for constant drivers."""
    dt = 2.5e-7
    acceleration = 4.0e6
    field = 8.0e-4

    frozen = MRTISpectrumTracker(k_modes_m_inv=[2.0, 8.0, 20.0], initial_perturbation_m=2.0e-9)
    interval = MRTISpectrumTracker(k_modes_m_inv=[2.0, 8.0, 20.0], initial_perturbation_m=2.0e-9)
    for _ in range(8):
        frozen_state = frozen.step(dt, acceleration, B_perp_t=field)
        interval_state = interval.step_interval(dt, acceleration, acceleration, field, field)

    np.testing.assert_array_equal(interval_state.amplitudes_m, frozen_state.amplitudes_m)
    np.testing.assert_array_equal(interval_state.log_amplitudes, frozen_state.log_amplitudes)
    np.testing.assert_array_equal(
        interval_state.growth_rates_s_inv, frozen_state.growth_rates_s_inv
    )
    assert interval_state.t_s == frozen_state.t_s


def _ramp_a0_s_t() -> tuple[float, float, float]:
    """Return the acceleration ramp parameters used by convergence tests."""
    return 1.0e6, 1.0e12, 1.0e-6


def _ramp_cumulative_log_mode0(n_steps: int, *, use_interval: bool) -> float:
    """Return the cumulative log amplitude for mode zero on a linear ramp."""
    a0, s, total_t = _ramp_a0_s_t()
    tracker = MRTISpectrumTracker(
        k_modes_m_inv=[1.0, 2.0],
        initial_perturbation_m=1.0,
        rho_kg_m3=1.0e-3,
        saturation_threshold_m=1.0e30,
    )
    h = total_t / n_steps
    for n in range(n_steps):
        a_start = a0 + s * (n * h)
        a_end = a0 + s * ((n + 1) * h)
        if use_interval:
            tracker.step_interval(h, a_start, a_end, 0.0, 0.0)
        else:
            tracker.step(h, a_end, 0.0)
    return float(tracker.state().log_amplitudes[0])


def _ramp_analytic_log_mode0() -> float:
    """Return the analytic cumulative growth exponent for mode zero."""
    a0, s, total_t = _ramp_a0_s_t()
    k = 1.0
    return math.sqrt(k) * (2.0 / (3.0 * s)) * (
        math.pow(a0 + s * total_t, 1.5) - math.pow(a0, 1.5)
    )


def test_step_interval_trapezoidal_is_second_order_on_acceleration_ramp() -> None:
    """Trapezoidal interval integration converges as second order on a ramp."""
    analytic = _ramp_analytic_log_mode0()

    err_trap_50 = abs(_ramp_cumulative_log_mode0(50, use_interval=True) - analytic)
    err_trap_100 = abs(_ramp_cumulative_log_mode0(100, use_interval=True) - analytic)
    err_endpoint_50 = abs(_ramp_cumulative_log_mode0(50, use_interval=False) - analytic)

    assert err_trap_100 < err_trap_50
    # Halving dt cuts a second-order error by ~4x; allow a tolerance band.
    assert 3.0 < err_trap_50 / err_trap_100 < 5.0
    # Trapezoidal integration is materially closer to the analytic cumulative
    # exponent than the first-order endpoint-frozen step on the same grid.
    assert err_trap_50 < 0.1 * err_endpoint_50


def test_most_amplified_mode_tracks_cumulative_not_instantaneous_peak() -> None:
    """Most-amplified mode reports cumulative amplitude, not instantaneous gamma."""
    # Interval 1 (constant) drives only the long-wavelength mode: choose a field
    # whose Alfven speed stabilises k = 300 while k = 100 stays unstable.
    field_va_unit = math.sqrt(MRTI_MU_0 * 1.0e-3 * 1.0)  # v_A^2 = B^2/(mu0 rho) = 1.0
    tracker = MRTISpectrumTracker(
        k_modes_m_inv=[100.0, 300.0],
        initial_perturbation_m=1.0,
        rho_kg_m3=1.0e-3,
        saturation_threshold_m=1.0e30,
    )
    tracker.step(1.0, 200.0, B_perp_t=field_va_unit)
    # Interval 2 makes k = 300 the instantaneous fastest mode (B = 0), but over a
    # short dt it cannot overtake the cumulative lead the long-wavelength mode
    # accrued in interval 1.
    state = tracker.step(1.0e-3, 1.0e6, B_perp_t=0.0)

    assert state.fastest_growing_k_m_inv == 300.0
    assert state.most_amplified_k_m_inv == 100.0
    assert state.most_amplified_k_m_inv == float(
        state.k_modes_m_inv[int(np.argmax(state.log_amplitudes))]
    )


@dataclass(frozen=True)
class _StubCompressionState:
    """Minimal pulsed-compression state for MRTI adapter contract tests."""

    t_s: float
    R_s_m: float
    dR_s_dt_m_s: float
    radial_acceleration_m_s2: float
    B_ext_T: float


def _stub_compression_states() -> list[_StubCompressionState]:
    """Return a valid three-sample pulsed-compression trajectory."""
    return [
        _StubCompressionState(0.0, 0.20, -1.0e4, -1.0, 4.0),
        _StubCompressionState(1.0e-7, 0.19, -2.0e4, -4.0, 4.5),
        _StubCompressionState(2.0e-7, 0.18, -3.0e4, -7.0, 5.0),
    ]


def test_track_pulsed_compression_uses_trapezoidal_interval_integration() -> None:
    """Trajectory coupling delegates each interval to trapezoidal tracker updates."""
    states = [
        _StubCompressionState(0.0, 0.20, 0.0, 0.0, 4.0),
        _StubCompressionState(1.0e-7, 0.19, -1.0e5, -1.0e12, 4.5),
        _StubCompressionState(2.0e-7, 0.17, -2.2e5, -1.6e12, 5.2),
        _StubCompressionState(3.0e-7, 0.14, -3.1e5, -1.1e12, 6.0),
    ]
    coupled = track_mrti_from_pulsed_compression(
        states, MRTISpectrumTracker(k_modes_m_inv=[50.0, 150.0], initial_perturbation_m=1.0e-9)
    )

    accelerations = effective_acceleration_from_pulsed_compression(states)
    reference = MRTISpectrumTracker(k_modes_m_inv=[50.0, 150.0], initial_perturbation_m=1.0e-9)
    for index in range(1, len(states)):
        dt_s = states[index].t_s - states[index - 1].t_s
        reference.step_interval(
            dt_s,
            float(accelerations[index - 1]),
            float(accelerations[index]),
            float(states[index - 1].B_ext_T),
            float(states[index].B_ext_T),
        )

    assert len(coupled) == len(states) - 1
    np.testing.assert_array_equal(coupled[-1].amplitudes_m, reference.state().amplitudes_m)
    np.testing.assert_array_equal(
        coupled[-1].growth_rates_s_inv, reference.state().growth_rates_s_inv
    )


def test_mrti_inputs_fail_closed() -> None:
    """Public MRTI inputs reject malformed scalar and coarse-grid contracts."""
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


def test_growth_rate_rejects_non_finite_inputs() -> None:
    """Growth-rate validation rejects non-finite modes and drivers."""
    with pytest.raises(ValueError, match="finite values"):
        mrti_growth_rate([math.nan], 1.0)
    with pytest.raises(ValueError, match="a_eff must be finite"):
        mrti_growth_rate([1.0], math.inf)
    with pytest.raises(ValueError, match="B_perp must be finite"):
        mrti_growth_rate([1.0], 1.0, B_perp=math.nan)


def test_effective_acceleration_from_radius_rate_validates_grid_contracts() -> None:
    """Radius-rate acceleration rejects malformed grids before estimating gradients."""
    two_dimensional: NDArray[np.float64] = np.array([[0.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="one-dimensional"):
        effective_acceleration_from_radius_rate(two_dimensional, two_dimensional)
    with pytest.raises(ValueError, match="identical shape"):
        effective_acceleration_from_radius_rate([0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="at least two samples"):
        effective_acceleration_from_radius_rate([0.0], [1.0])
    with pytest.raises(ValueError, match="cannot exceed"):
        effective_acceleration_from_radius_rate([0.0, 1.0], [1.0, 2.0], smoothing_window=3)


def test_effective_acceleration_from_radius_rate_returns_unsmoothed_gradient() -> None:
    """The default radius-rate adapter returns the raw finite-difference gradient."""
    time_s: NDArray[np.float64] = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    speed_m_s: NDArray[np.float64] = np.array([1.0, 2.0, 5.0], dtype=np.float64)

    acceleration = effective_acceleration_from_radius_rate(time_s, speed_m_s)

    np.testing.assert_allclose(
        acceleration,
        np.gradient(speed_m_s, time_s, edge_order=2),
        rtol=0.0,
        atol=0.0,
    )


def test_effective_acceleration_from_pulsed_compression_smooths_projection() -> None:
    """Pulsed-compression acceleration smoothing is applied before projection."""
    acceleration = effective_acceleration_from_pulsed_compression(
        _stub_compression_states(),
        smoothing_window=3,
    )

    np.testing.assert_allclose(acceleration, [2.0, 4.0, 6.0], rtol=0.0, atol=1.0e-15)


def test_effective_acceleration_from_pulsed_compression_rejects_bad_projection() -> None:
    """Pulsed-compression coupling rejects a zero radial projection sign."""
    with pytest.raises(ValueError, match="non-zero"):
        effective_acceleration_from_pulsed_compression(
            _stub_compression_states(),
            radial_projection_sign=0.0,
        )


def test_effective_acceleration_from_pulsed_compression_validates_states() -> None:
    """Pulsed-compression coupling rejects invalid trajectory samples."""
    states = _stub_compression_states()
    states[1] = _StubCompressionState(1.0e-7, 0.0, -2.0e4, -4.0, 4.5)
    with pytest.raises(ValueError, match="radii must be positive"):
        effective_acceleration_from_pulsed_compression(states)

    states = _stub_compression_states()
    states[1] = _StubCompressionState(0.0, 0.19, -2.0e4, -4.0, 4.5)
    with pytest.raises(ValueError, match="time_s must be finite and strictly increasing"):
        effective_acceleration_from_pulsed_compression(states)

    states = _stub_compression_states()
    states[1] = _StubCompressionState(1.0e-7, 0.19, math.nan, -4.0, 4.5)
    with pytest.raises(ValueError, match="speeds must be finite"):
        effective_acceleration_from_pulsed_compression(states)

    states = _stub_compression_states()
    states[1] = _StubCompressionState(1.0e-7, 0.19, -2.0e4, math.inf, 4.5)
    with pytest.raises(ValueError, match="radial accelerations must be finite"):
        effective_acceleration_from_pulsed_compression(states)

    states = _stub_compression_states()
    states[1] = _StubCompressionState(1.0e-7, 0.19, -2.0e4, -4.0, math.nan)
    with pytest.raises(ValueError, match="fields must be finite"):
        effective_acceleration_from_pulsed_compression(states)


def test_effective_acceleration_from_pulsed_compression_validates_smoothing() -> None:
    """Pulsed-compression smoothing windows must be positive odd in-range values."""
    with pytest.raises(ValueError, match="positive odd"):
        effective_acceleration_from_pulsed_compression(
            _stub_compression_states(),
            smoothing_window=2,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        effective_acceleration_from_pulsed_compression(
            _stub_compression_states(),
            smoothing_window=5,
        )


def test_spectrum_tracker_validates_mode_configuration() -> None:
    """Tracker construction rejects ambiguous or malformed mode spectra."""
    with pytest.raises(ValueError, match="k_max_m_inv is required"):
        MRTISpectrumTracker()
    with pytest.raises(ValueError, match="one-dimensional"):
        MRTISpectrumTracker(k_modes_m_inv=np.array([[1.0, 2.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="at least two MRTI modes"):
        MRTISpectrumTracker(k_modes_m_inv=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        MRTISpectrumTracker(k_modes_m_inv=[-1.0, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        MRTISpectrumTracker(k_modes_m_inv=[2.0, 2.0])


def test_spectrum_tracker_properties_return_defensive_copies() -> None:
    """Mode and amplitude properties expose copies instead of mutable internals."""
    tracker = MRTISpectrumTracker(k_modes_m_inv=[1.0, 2.0], initial_perturbation_m=3.0e-9)

    modes = tracker.k_modes_m_inv
    amplitudes = tracker.amplitudes_m
    modes[0] = 99.0
    amplitudes[0] = 99.0

    np.testing.assert_array_equal(tracker.k_modes_m_inv, np.array([1.0, 2.0], dtype=np.float64))
    np.testing.assert_array_equal(
        tracker.amplitudes_m,
        np.array([3.0e-9, 3.0e-9], dtype=np.float64),
    )


def test_track_mrti_from_pulsed_compression_validates_coupling_inputs() -> None:
    """Trajectory tracking rejects incomplete trajectories and negative field scaling."""
    tracker = MRTISpectrumTracker(k_modes_m_inv=[1.0, 2.0])
    with pytest.raises(ValueError, match="at least two pulsed-compression states"):
        track_mrti_from_pulsed_compression(_stub_compression_states()[:1], tracker)
    with pytest.raises(ValueError, match="b_perp_scale must be non-negative"):
        track_mrti_from_pulsed_compression(
            _stub_compression_states(),
            tracker,
            b_perp_scale=-1.0,
        )
