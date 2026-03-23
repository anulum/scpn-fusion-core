# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Coverage Batch 12 (passing only)
from __future__ import annotations

import numpy as np


class TestSlidingModeVertical:
    def test_super_twisting_init(self):
        from scpn_fusion.control.sliding_mode_vertical import SuperTwistingSMC

        smc = SuperTwistingSMC(alpha=5.0, beta=2.0, c=1.0, u_max=10.0)
        assert smc.alpha == 5.0

    def test_step(self):
        from scpn_fusion.control.sliding_mode_vertical import SuperTwistingSMC

        smc = SuperTwistingSMC(alpha=5.0, beta=2.0, c=1.0, u_max=10.0)
        result = smc.step(e=0.1, de_dt=0.01, dt=0.001)
        assert result is not None


class TestFluxBudget:
    def test_init(self):
        from scpn_fusion.control.volt_second_manager import FluxBudget

        fb = FluxBudget(Phi_CS_Vs=100.0, L_plasma_uH=10.0, R_plasma_uOhm=5.0)
        assert fb.Phi_CS_Vs == 100.0
