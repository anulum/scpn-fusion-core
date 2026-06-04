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
    MRTISpectrumTracker,
    effective_acceleration_from_radius_rate,
    mrti_growth_rate,
)
from scpn_fusion.core.mrti import MU_0


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
    raw = k_modes * acceleration - (k_modes * k_modes * b_perp * b_perp) / (MU_0 * density)

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
    assert state.fastest_growing_k_m_inv == 8.0
    assert not state.saturation_warning
    assert state.time_of_breach_s is None


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


def test_mrti_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mrti_growth_rate([-1.0], 1.0)
    with pytest.raises(ValueError, match="positive"):
        mrti_growth_rate([1.0], 1.0, rho_kg_m3=0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        effective_acceleration_from_radius_rate([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="positive odd"):
        effective_acceleration_from_radius_rate([0.0, 1.0], [1.0, 2.0], smoothing_window=2)
    with pytest.raises(ValueError, match="at least 2"):
        MRTISpectrumTracker(k_max_m_inv=10.0, n_modes=1)
    assert isclose(float(mrti_growth_rate(4.0, 9.0)[()]), 6.0)
