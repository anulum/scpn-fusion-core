# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — Coverage Batch 2
from __future__ import annotations

import numpy as np


class TestELMModel:
    def test_peeling_ballooning_init(self):
        from scpn_fusion.core.elm_model import PeelingBallooningBoundary

        pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.33, a=2.0, R0=6.2)
        assert pb is not None


class TestLHTransition:
    def test_predator_prey_init(self):
        from scpn_fusion.core.lh_transition import PredatorPreyModel

        model = PredatorPreyModel()
        assert model is not None


class TestTearingModeCoupling:
    def test_chirikov_init(self):
        from scpn_fusion.core.tearing_mode_coupling import ChirikovOverlap

        co = ChirikovOverlap()
        assert co is not None


class TestAlfvenEigenmodes:
    def test_fast_particle_drive_init(self):
        from scpn_fusion.core.alfven_eigenmodes import FastParticleDrive

        fpd = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.01)
        assert fpd is not None


class TestMomentumTransport:
    def test_turbulence_suppression(self):
        from scpn_fusion.core.momentum_transport import turbulence_suppression_factor

        omega = np.array([0.0, 0.5, 1.0, 2.0])
        gamma = np.array([1.0, 1.0, 1.0, 1.0])
        f = turbulence_suppression_factor(omega, gamma)
        assert len(f) == 4
        assert f[0] == 1.0


class TestVMECLite:
    def test_init(self):
        from scpn_fusion.core.vmec_lite import VMECLiteSolver

        solver = VMECLiteSolver(n_s=11, m_pol=2, n_tor=1, n_fp=1)
        assert solver is not None


class TestPelletInjection:
    def test_ngs_rate(self):
        from scpn_fusion.core.pellet_injection import ngs_ablation_rate

        rate = ngs_ablation_rate(0.003, 1e20, 5000.0, 2.0)
        assert rate > 0
        assert np.isfinite(rate)
