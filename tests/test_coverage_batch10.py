# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Coverage Batch 10 (passing only)
from __future__ import annotations

import pytest


class TestConfigSchema:
    def test_dimensions(self):
        from scpn_fusion.core.config_schema import Dimensions

        d = Dimensions(R_min=1.0, R_max=9.0, Z_min=-5.0, Z_max=5.0)
        assert d.R_min == 1.0

    def test_coil(self):
        from scpn_fusion.core.config_schema import Coil

        c = Coil(r=6.2, z=0.0, current=5.0)
        assert c.r == pytest.approx(6.2)


class TestGKSpecies:
    def test_deuterium_ion(self):
        from scpn_fusion.core.gk_species import deuterium_ion

        d = deuterium_ion()
        assert d.mass_amu == pytest.approx(2.0)

    def test_electron(self):
        from scpn_fusion.core.gk_species import electron

        e = electron()
        assert e.charge == -1
