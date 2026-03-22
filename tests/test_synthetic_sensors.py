# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Synthetic Sensor Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.diagnostics.synthetic_sensors import SensorSuite


class _MockKernel:
    """Minimal kernel mock for SensorSuite tests."""

    NR = 33
    NZ = 33
    R = np.linspace(3.0, 9.0, NR)
    Z = np.linspace(-3.5, 3.5, NZ)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    RR, ZZ = np.meshgrid(R, Z)
    Psi = np.exp(-((RR - 6.0) ** 2 + ZZ**2) / 4.0)
    B_R = np.zeros((NZ, NR))
    B_Z = np.ones((NZ, NR)) * 0.1
    J = np.exp(-((RR - 6.0) ** 2 + ZZ**2) / 2.0) * 1e6
    J_phi = J


class TestSensorSuiteInit:
    def test_creates_with_kernel(self):
        ss = SensorSuite(_MockKernel(), seed=42)
        assert len(ss.wall_R) == 20
        assert len(ss.bolo_chords) > 0

    def test_seed_and_rng_exclusive(self):
        with pytest.raises(ValueError, match="seed or rng"):
            SensorSuite(_MockKernel(), seed=1, rng=np.random.default_rng(1))


class TestMeasureMagnetics:
    def test_returns_20_measurements(self):
        ss = SensorSuite(_MockKernel(), seed=42)
        meas = ss.measure_magnetics()
        assert len(meas) == 20
        assert np.all(np.isfinite(meas))

    def test_deterministic_with_seed(self):
        m1 = SensorSuite(_MockKernel(), seed=42).measure_magnetics()
        m2 = SensorSuite(_MockKernel(), seed=42).measure_magnetics()
        np.testing.assert_array_equal(m1, m2)


class TestMeasureBField:
    def test_returns_br_bz(self):
        ss = SensorSuite(_MockKernel(), seed=42)
        Br, Bz = ss.measure_b_field()
        assert len(Br) == 20
        assert len(Bz) == 20
        assert np.all(np.isfinite(Br))
        assert np.all(np.isfinite(Bz))


class TestMeasureBolometer:
    def test_returns_signals(self):
        ss = SensorSuite(_MockKernel(), seed=42)
        emission = np.exp(-((ss.kernel.RR - 6.0) ** 2 + ss.kernel.ZZ**2) / 2.0)
        signals = ss.measure_bolometer(emission)
        assert len(signals) == len(ss.bolo_chords)
        assert np.all(np.isfinite(signals))


class TestMeasureInterferometer:
    def test_returns_phases(self):
        ss = SensorSuite(_MockKernel(), seed=42)
        ne_profile = 10.0 * np.exp(-((ss.kernel.RR - 6.0) ** 2 + ss.kernel.ZZ**2) / 3.0)
        phases = ss.measure_interferometer(ne_profile)
        assert len(phases) > 0
        assert np.all(np.isfinite(phases))


class TestNoise:
    def test_noise_with_rng(self):
        ss = SensorSuite(_MockKernel(), seed=42)
        n1 = ss._noise(0.1)
        n2 = ss._noise(0.1)
        assert n1 != n2
        assert np.isfinite(n1)

    def test_noise_zero_scale(self):
        ss = SensorSuite(_MockKernel(), seed=42)
        n = ss._noise(0.0)
        assert n == 0.0

    def test_noise_without_rng(self):
        ss = SensorSuite(_MockKernel())
        n = ss._noise(0.1)
        assert np.isfinite(n)
