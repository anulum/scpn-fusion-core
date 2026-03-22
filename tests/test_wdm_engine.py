# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — WDM Engine Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.wdm_engine import WholeDeviceModel


class TestThomasFermiPressure:
    """Test the Thomas-Fermi EOS without needing a full solver."""

    def _bare_wdm(self):
        wdm = WholeDeviceModel.__new__(WholeDeviceModel)
        return wdm

    def test_positive_pressure(self):
        wdm = self._bare_wdm()
        P = wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=100.0)
        assert P > 0

    def test_pressure_increases_with_temperature(self):
        wdm = self._bare_wdm()
        P1 = wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=10.0)
        P2 = wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=100.0)
        assert P2 > P1

    def test_pressure_increases_with_density(self):
        wdm = self._bare_wdm()
        P1 = wdm.thomas_fermi_pressure(n_e_m3=1e19, T_eV=50.0)
        P2 = wdm.thomas_fermi_pressure(n_e_m3=1e21, T_eV=50.0)
        assert P2 > P1

    def test_degeneracy_dominates_at_high_density(self):
        wdm = self._bare_wdm()
        # At very high density, degeneracy pressure should dominate
        P_high = wdm.thomas_fermi_pressure(n_e_m3=1e30, T_eV=1.0)
        P_ideal = 1e30 * (1.0 * 1.602e-19)
        assert P_high > P_ideal

    def test_zero_temperature_still_positive(self):
        wdm = self._bare_wdm()
        P = wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=0.0)
        assert P > 0  # degeneracy pressure

    def test_returns_float(self):
        wdm = self._bare_wdm()
        P = wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=50.0)
        assert isinstance(P, (float, np.floating))
