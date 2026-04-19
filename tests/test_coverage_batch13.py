# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — Coverage Batch 13 (engineering + stellarator)
from __future__ import annotations

import numpy as np
import pytest


class TestStellaratorGeometry:
    def test_w7x_config(self):
        from scpn_fusion.core.stellarator_geometry import w7x_config

        cfg = w7x_config()
        assert cfg.R0 > 0

    def test_iota_profile(self):
        from scpn_fusion.core.stellarator_geometry import iota_profile, w7x_config

        cfg = w7x_config()
        rho = np.linspace(0, 1, 20)
        iota = iota_profile(cfg, rho)
        assert len(iota) == 20


class TestBalanceOfPlant:
    def test_power_plant_model(self):
        from scpn_fusion.engineering.balance_of_plant import PowerPlantModel

        ppm = PowerPlantModel(coolant_type="water")
        assert ppm is not None


class TestThermalHydraulics:
    def test_churchill_friction(self):
        from scpn_fusion.engineering.thermal_hydraulics import churchill_friction_factor

        f = churchill_friction_factor(Re=1e5)
        assert f > 0
        assert np.isfinite(f)

    def test_coolant_loop(self):
        from scpn_fusion.engineering.thermal_hydraulics import CoolantLoop

        cl = CoolantLoop(coolant_type="water")
        assert cl is not None
