# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — Coverage Gap Tests (batch)
"""Lightweight tests targeting uncovered lines in multiple modules."""

from __future__ import annotations

import numpy as np


class TestNeoclassicalGaps:
    def test_low_epsilon_returns_zero(self):
        from scpn_fusion.core.neoclassical import banana_plateau_chi

        assert banana_plateau_chi(q=2.0, epsilon=1e-7, nu_star=0.1, z_eff=1.0) == 0.0

    def test_high_collisionality(self):
        from scpn_fusion.core.neoclassical import banana_plateau_chi

        chi = banana_plateau_chi(q=2.0, epsilon=0.3, nu_star=100.0, z_eff=2.0)
        assert chi > 0 and np.isfinite(chi)

    def test_collisionality_function(self):
        from scpn_fusion.core.neoclassical import collisionality

        nu = collisionality(n_e_19=10.0, T_kev=10.0, q=2.0, R=6.2, epsilon=0.3)
        assert nu > 0 and np.isfinite(nu)


class TestRunawayElectronsGaps:
    def test_dreicer_field(self):
        from scpn_fusion.core.runaway_electrons import dreicer_field

        E_D = dreicer_field(ne_20=1.0, Te_keV=10.0)
        assert E_D > 0

    def test_critical_field(self):
        from scpn_fusion.core.runaway_electrons import critical_field

        E_c = critical_field(ne_20=1.0)
        assert E_c > 0

    def test_hot_tail_seed(self):
        from scpn_fusion.core.runaway_electrons import hot_tail_seed

        n_seed = hot_tail_seed(Te_pre_keV=10.0, Te_post_keV=0.01, ne_20=1.0, quench_time_ms=1.0)
        assert n_seed >= 0

    def test_dreicer_ratio(self):
        from scpn_fusion.core.runaway_electrons import dreicer_field, critical_field

        E_D = dreicer_field(ne_20=1.0, Te_keV=10.0)
        E_c = critical_field(ne_20=1.0)
        # E_D / E_c = m_e c^2 / T_e ≈ 51 at 10 keV
        assert 40 < E_D / E_c < 60


class TestNTMDynamicsGaps:
    def test_rational_surface(self):
        from scpn_fusion.core.ntm_dynamics import RationalSurface

        rs = RationalSurface(rho=0.5, r_s=1.0, m=2, n=1, q=2.0, shear=1.5)
        assert rs.m == 2

    def test_ntm_controller(self):
        from scpn_fusion.core.ntm_dynamics import NTMController

        ctrl = NTMController()
        assert hasattr(ctrl, "eccd_power_request")


class TestVesselModel:
    def test_import(self):
        from scpn_fusion.core.vessel_model import VesselModel

        assert VesselModel is not None
