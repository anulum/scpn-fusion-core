# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Hall MHD Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.hall_mhd_discovery import HallMHD


class TestHallMHDInit:
    def test_default_grid(self):
        sim = HallMHD(N=16)
        assert sim.N == 16
        assert sim.phi_k.shape == (16, 16)
        assert sim.psi_k.shape == (16, 16)

    def test_custom_eta_nu(self):
        sim = HallMHD(N=16, eta=1e-3, nu=1e-3)
        assert sim.eta == 1e-3
        assert sim.nu == 1e-3


class TestDynamics:
    def test_dynamics_returns_two_arrays(self):
        sim = HallMHD(N=16)
        dphi, dpsi = sim.dynamics(sim.phi_k, sim.psi_k)
        assert dphi.shape == (16, 16)
        assert dpsi.shape == (16, 16)

    def test_poisson_bracket_antisymmetric(self):
        sim = HallMHD(N=16)
        ab = sim.poisson_bracket(sim.phi_k, sim.psi_k)
        ba = sim.poisson_bracket(sim.psi_k, sim.phi_k)
        np.testing.assert_allclose(np.real(ab + ba), 0.0, atol=1e-10)


class TestStep:
    def test_step_returns_energies(self):
        sim = HallMHD(N=16)
        E_tot, E_zonal = sim.step()
        assert E_tot >= 0
        assert E_zonal >= 0
        assert np.isfinite(E_tot)

    def test_energy_tracked(self):
        sim = HallMHD(N=16)
        for _ in range(10):
            sim.step()
        assert len(sim.energy_history) == 10

    def test_multiple_steps_stable(self):
        sim = HallMHD(N=16)
        for _ in range(50):
            E, _ = sim.step()
        assert np.isfinite(E)


class TestParameterSweep:
    def test_returns_results_dict(self):
        sim = HallMHD(N=8)
        results = sim.parameter_sweep(
            eta_range=(1e-4, 1e-3),
            nu_range=(1e-4, 1e-3),
            n_steps=2,
            sim_steps=20,
        )
        assert "eta" in results
        assert "nu" in results
        assert "growth_rate" in results
        assert len(results["eta"]) == 4  # 2x2 grid


class TestTearingThreshold:
    def test_returns_threshold(self):
        sim = HallMHD(N=8)
        result = sim.find_tearing_threshold(
            eta_range=(1e-5, 1e-2),
            n_bisect=3,
            sim_steps=30,
        )
        assert "threshold_eta" in result
        assert result["threshold_eta"] > 0
        assert result["lo"] <= result["threshold_eta"] <= result["hi"]
