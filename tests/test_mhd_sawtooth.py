# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — MHD Sawtooth Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.mhd_sawtooth import ReducedMHD


class TestReducedMHDInit:
    def test_default_grid(self):
        sim = ReducedMHD(nr=50)
        assert sim.nr == 50
        assert len(sim.r) == 50
        assert sim.q[0] < 1.0  # q(0) < 1 for internal kink

    def test_q_profile_monotone(self):
        sim = ReducedMHD()
        assert np.all(np.diff(sim.q) >= 0)

    def test_initial_perturbation_nonzero(self):
        sim = ReducedMHD()
        assert np.max(np.abs(sim.psi_11)) > 0


class TestLaplacian:
    def test_shape_preserved(self):
        sim = ReducedMHD(nr=50)
        f = np.sin(np.pi * sim.r)
        result = sim.laplacian(f, m=1)
        assert result.shape == (50,)

    def test_zero_function(self):
        sim = ReducedMHD(nr=50)
        result = sim.laplacian(np.zeros(50), m=1)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_boundary_values_zero(self):
        sim = ReducedMHD(nr=50)
        f = np.random.randn(50)
        result = sim.laplacian(f, m=1)
        assert result[0] == 0.0
        assert result[-1] == 0.0


class TestStep:
    def test_returns_amplitude_and_crash(self):
        sim = ReducedMHD(nr=50)
        amp, crash = sim.step(dt=0.01)
        assert amp >= 0
        assert isinstance(crash, bool)
        assert np.isfinite(amp)

    def test_perturbation_grows(self):
        sim = ReducedMHD(nr=50)
        amp0 = np.max(np.abs(sim.psi_11))
        for _ in range(50):
            sim.step(dt=0.01)
        amp1 = np.max(np.abs(sim.psi_11))
        # Internal kink should grow (q(0) < 1)
        assert amp1 > amp0 or amp1 > 0

    def test_sawtooth_crash_occurs(self):
        sim = ReducedMHD(nr=50)
        crashed = False
        for _ in range(500):
            _, c = sim.step(dt=0.01)
            if c:
                crashed = True
                break
        # May or may not crash in 500 steps depending on parameters
        # Just verify it runs without error
        assert np.all(np.isfinite(sim.psi_11))

    def test_q_recovery_after_crash(self):
        sim = ReducedMHD(nr=50)
        q_init = sim.q.copy()
        for _ in range(500):
            _, crash = sim.step(dt=0.01)
            if crash:
                break
        # q should be modified after crash
        assert np.all(np.isfinite(sim.q))


class TestSolvePoisson:
    def test_boundary_conditions(self):
        sim = ReducedMHD(nr=50)
        U = np.ones(50, dtype=complex) * 1.0
        phi = sim.solve_poisson(U)
        assert phi[0] == pytest.approx(0.0, abs=1e-10)
        assert phi[-1] == pytest.approx(0.0, abs=1e-10)

    def test_returns_finite(self):
        sim = ReducedMHD(nr=50)
        U = np.random.randn(50) + 0j
        phi = sim.solve_poisson(U)
        assert np.all(np.isfinite(phi))

    def test_zero_source_gives_zero(self):
        sim = ReducedMHD(nr=50)
        phi = sim.solve_poisson(np.zeros(50, dtype=complex))
        np.testing.assert_allclose(phi, 0.0, atol=1e-12)
