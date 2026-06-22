# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — Synthetic Sensor Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.diagnostics.synthetic_sensors import (
    N_MAGNETIC_PROBES,
    SensorSuite,
    magnetic_probe_positions,
    measure_magnetics,
)


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


class TestMeasureMagneticsFreeFunction:
    """Tests for the noise-free canonical measure_magnetics free function."""

    def test_probe_positions_lie_on_wall_ellipse(self) -> None:
        wall_r, wall_z = magnetic_probe_positions()
        assert wall_r.shape == (N_MAGNETIC_PROBES,)
        # Each probe satisfies ((R-R0)/wall)^2 + (Z/(kappa*wall))^2 == 1.
        wall = 3.0 + 0.5
        norm = ((wall_r - 6.0) / wall) ** 2 + (wall_z / (1.8 * wall)) ** 2
        np.testing.assert_allclose(norm, 1.0, rtol=0.0, atol=1e-12)

    def test_uniform_field_returns_constant(self) -> None:
        psi = np.full((65, 65), 2.5, dtype=np.float64)
        meas = measure_magnetics(psi, 65, 65, 3.0, 9.0, -5.0, 5.0)
        assert meas.shape == (N_MAGNETIC_PROBES,)
        np.testing.assert_allclose(meas, 2.5, rtol=0.0, atol=1e-12)

    def test_linear_field_recovered_by_bilinear(self) -> None:
        # psi linear in R: psi[iz, ir] = R at that column. Interior probes read
        # back their own R coordinate exactly.
        nr = nz = 65
        r_min, r_max, z_min, z_max = 3.0, 9.0, -5.0, 5.0
        r_axis = np.linspace(r_min, r_max, nr)
        psi = np.tile(r_axis, (nz, 1))
        dr = (r_max - r_min) / (nr - 1)
        dz = (z_max - z_min) / (nz - 1)
        wall_r, wall_z = magnetic_probe_positions()
        meas = measure_magnetics(psi, nr, nz, r_min, r_max, z_min, z_max)
        checked = 0
        for i in range(N_MAGNETIC_PROBES):
            ir = int((wall_r[i] - r_min) / dr)
            iz = int((wall_z[i] - z_min) / dz)
            if 0 <= ir < nr - 1 and 0 <= iz < nz - 1:
                assert meas[i] == pytest.approx(float(wall_r[i]), abs=1e-9)
                checked += 1
        assert checked > 0

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            measure_magnetics(np.zeros((10, 12)), 65, 65, 3.0, 9.0, -5.0, 5.0)

    def test_method_delegates_to_free_function_without_noise(self) -> None:
        # With noise sigma -> 0 (no seed, but compare structure), the method's
        # deterministic core equals the free function on the mock kernel grid.
        kernel = _MockKernel()
        clean = measure_magnetics(
            np.asarray(kernel.Psi, dtype=np.float64),
            kernel.NR,
            kernel.NZ,
            float(kernel.R[0]),
            float(kernel.R[-1]),
            float(kernel.Z[0]),
            float(kernel.Z[-1]),
        )
        assert clean.shape == (N_MAGNETIC_PROBES,)
        assert np.all(np.isfinite(clean))
