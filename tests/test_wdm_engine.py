# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

    def test_rejects_negative_temperature(self):
        wdm = self._bare_wdm()
        with pytest.raises(ValueError, match="T_eV"):
            wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=-1.0)


class _DummyTransport:
    def __init__(self) -> None:
        self.ne = np.array([0.8, 1.2], dtype=float)
        self.Te = np.array([1.0, 0.9], dtype=float)
        self.n_impurity = np.array([0.0, 0.0], dtype=float)
        self.solve_calls = 0

    def solve_equilibrium(self) -> None:
        self.solve_calls += 1

    def evolve_profiles(self, _dt: float, _p_aux: float) -> tuple[float, float]:
        self.n_impurity += 1e-3
        return 1.0, 1.1

    def inject_impurities(self, amount: float, _dt: float) -> None:
        self.n_impurity += amount

    def map_profiles_to_2d(self) -> None:
        return None


class _DummyPWI:
    def calculate_erosion_rate(self, _flux_wall: float, _t_edge: float) -> dict[str, float]:
        return {"Impurity_Source": 2.0e18}


def _bare_runtime_wdm() -> WholeDeviceModel:
    wdm = WholeDeviceModel.__new__(WholeDeviceModel)
    wdm.transport = _DummyTransport()
    wdm.pwi = _DummyPWI()
    return wdm


def test_run_discharge_returns_structured_history(monkeypatch: pytest.MonkeyPatch) -> None:
    wdm = _bare_runtime_wdm()
    monkeypatch.setattr(wdm, "plot_results", lambda history: None)

    history = wdm.run_discharge(duration_sec=0.03)
    assert len(history) >= 1
    assert {"time", "Te_core", "W_impurity", "P_rad", "status"} <= set(history[0].keys())
    assert history[0]["status"] == "OK"


def test_run_discharge_rejects_invalid_duration() -> None:
    wdm = _bare_runtime_wdm()
    with pytest.raises(ValueError, match="duration_sec"):
        wdm.run_discharge(duration_sec=0.0)


def test_plot_results_rejects_empty_history() -> None:
    wdm = _bare_runtime_wdm()
    with pytest.raises(ValueError, match="history must not be empty"):
        wdm.plot_results([])
