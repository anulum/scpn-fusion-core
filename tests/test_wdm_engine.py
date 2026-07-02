# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Contract tests for the warm dense matter whole-device model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from scpn_fusion.core import wdm_engine as wdm_module
from scpn_fusion.core.wdm_engine import WholeDeviceModel

FloatArray = npt.NDArray[np.float64]
DischargeState = dict[str, float | str]


class TestThomasFermiPressure:
    """Test the Thomas-Fermi EOS without needing a full solver."""

    def _bare_wdm(self) -> WholeDeviceModel:
        """Create an uninitialized WDM instance for pure EOS tests."""
        return WholeDeviceModel.__new__(WholeDeviceModel)

    def test_positive_pressure(self) -> None:
        """Thomas-Fermi pressure is positive for physical density and temperature."""
        wdm = self._bare_wdm()
        P = wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=100.0)
        assert P > 0

    def test_pressure_increases_with_temperature(self) -> None:
        """Ideal pressure contribution increases with electron temperature."""
        wdm = self._bare_wdm()
        P1 = wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=10.0)
        P2 = wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=100.0)
        assert P2 > P1

    def test_pressure_increases_with_density(self) -> None:
        """Total pressure increases with electron density."""
        wdm = self._bare_wdm()
        P1 = wdm.thomas_fermi_pressure(n_e_m3=1e19, T_eV=50.0)
        P2 = wdm.thomas_fermi_pressure(n_e_m3=1e21, T_eV=50.0)
        assert P2 > P1

    def test_degeneracy_dominates_at_high_density(self) -> None:
        """Degeneracy pressure dominates the ideal term at very high density."""
        wdm = self._bare_wdm()
        # At very high density, degeneracy pressure should dominate
        P_high = wdm.thomas_fermi_pressure(n_e_m3=1e30, T_eV=1.0)
        P_ideal = 1e30 * (1.0 * 1.602e-19)
        assert P_high > P_ideal

    def test_zero_temperature_still_positive(self) -> None:
        """Degeneracy pressure keeps the EOS positive at zero temperature."""
        wdm = self._bare_wdm()
        P = wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=0.0)
        assert P > 0  # degeneracy pressure

    def test_returns_float(self) -> None:
        """The EOS returns a plain scalar float for downstream reporting."""
        wdm = self._bare_wdm()
        P = wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=50.0)
        assert isinstance(P, (float, np.floating))

    def test_rejects_negative_temperature(self) -> None:
        """Negative temperature inputs are rejected before pressure evaluation."""
        wdm = self._bare_wdm()
        with pytest.raises(ValueError, match="T_eV"):
            wdm.thomas_fermi_pressure(n_e_m3=1e20, T_eV=-1.0)


class _InitTransport:
    """Minimal constructor collaborator for WDM initialization tests."""

    last_instance: _InitTransport | None = None

    def __init__(self, config_path: str) -> None:
        """Record the configuration path supplied by ``WholeDeviceModel``."""
        self.config_path = config_path
        self.solve_calls = 0
        type(self).last_instance = self

    def solve_equilibrium(self) -> None:
        """Track the initial equilibrium solve call."""
        self.solve_calls += 1


class _InitPWI:
    """Minimal wall-interaction collaborator for WDM initialization tests."""

    last_instance: _InitPWI | None = None

    def __init__(self, material: str) -> None:
        """Record the material requested by ``WholeDeviceModel``."""
        self.material = material
        type(self).last_instance = self


class _DummyTransport:
    """Runtime transport stub for deterministic discharge-loop tests."""

    def __init__(self) -> None:
        """Initialize stable edge profiles and impurity inventory."""
        self.ne: FloatArray = np.array([0.8, 1.2], dtype=np.float64)
        self.Te: FloatArray = np.array([1.0, 0.9], dtype=np.float64)
        self.n_impurity: FloatArray = np.array([0.0, 0.0], dtype=np.float64)
        self.solve_calls = 0

    def solve_equilibrium(self) -> None:
        """Track equilibrium solve requests."""
        self.solve_calls += 1

    def evolve_profiles(self, _dt: float, _p_aux: float) -> tuple[float, float]:
        """Return a stable core temperature while accumulating impurity state."""
        self.n_impurity += 1e-3
        return 1.0, 1.1

    def inject_impurities(self, amount: float, _dt: float) -> None:
        """Apply injected impurity inventory to the whole profile."""
        self.n_impurity += amount

    def map_profiles_to_2d(self) -> None:
        """Expose the transport remapping hook used before equilibrium solves."""
        return None


class _CollapsingTransport(_DummyTransport):
    """Transport stub that forces radiative collapse on the first step."""

    def evolve_profiles(self, _dt: float, _p_aux: float) -> tuple[float, float]:
        """Return a sub-threshold core temperature."""
        self.n_impurity += 1e-3
        return 1.0, 0.4


class _DummyPWI:
    """Wall-interaction stub with deterministic tungsten source."""

    def calculate_erosion_rate(self, _flux_wall: float, _t_edge: float) -> dict[str, float]:
        """Return a fixed gross impurity source."""
        return {"Impurity_Source": 2.0e18}


def _bare_runtime_wdm(
    transport: _DummyTransport | None = None,
    pwi: _DummyPWI | None = None,
) -> WholeDeviceModel:
    """Create a WDM instance with deterministic runtime collaborators."""
    wdm = WholeDeviceModel.__new__(WholeDeviceModel)
    object.__setattr__(wdm, "transport", transport or _DummyTransport())
    object.__setattr__(wdm, "pwi", pwi or _DummyPWI())
    return wdm


def _ignore_plot(_history: list[DischargeState]) -> None:
    """Replace plotting when tests only need discharge-loop contracts."""
    return None


def test_constructor_wires_transport_wall_model_and_initial_equilibrium(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialization builds collaborators and solves equilibrium once."""
    _InitTransport.last_instance = None
    _InitPWI.last_instance = None
    monkeypatch.setattr(wdm_module, "TransportSolver", _InitTransport)
    monkeypatch.setattr(wdm_module, "SputteringPhysics", _InitPWI)

    wdm = WholeDeviceModel("config.json")

    assert isinstance(wdm.transport, _InitTransport)
    assert isinstance(wdm.pwi, _InitPWI)
    assert _InitTransport.last_instance is wdm.transport
    assert wdm.transport.config_path == "config.json"
    assert wdm.transport.solve_calls == 1
    assert _InitPWI.last_instance is wdm.pwi
    assert wdm.pwi.material == "Tungsten"


def test_run_discharge_returns_structured_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stable short discharge returns typed state rows and OK status."""
    wdm = _bare_runtime_wdm()
    monkeypatch.setattr(wdm, "plot_results", _ignore_plot)

    history = wdm.run_discharge(duration_sec=0.03)
    assert len(history) >= 1
    assert {"time", "Te_core", "W_impurity", "P_rad", "status"} <= set(history[0].keys())
    assert history[0]["status"] == "OK"


def test_run_discharge_stops_on_radiative_collapse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sub-threshold core temperature marks collapse and stops the timeline."""
    wdm = _bare_runtime_wdm(transport=_CollapsingTransport())
    monkeypatch.setattr(wdm, "plot_results", _ignore_plot)

    history = wdm.run_discharge(duration_sec=0.03)

    assert len(history) == 1
    assert history[0]["Te_core"] == pytest.approx(0.4)
    assert history[0]["status"] == "COLLAPSE"


def test_run_discharge_rejects_invalid_duration() -> None:
    """Non-positive discharge duration is rejected before simulation."""
    wdm = _bare_runtime_wdm()
    with pytest.raises(ValueError, match="duration_sec"):
        wdm.run_discharge(duration_sec=0.0)


def test_redeposition_rejects_invalid_field() -> None:
    """Prompt redeposition requires a positive magnetic field."""
    wdm = _bare_runtime_wdm()

    with pytest.raises(ValueError, match="B_field_T"):
        wdm.calculate_redeposition_fraction(T_edge_eV=10.0, B_field_T=0.0)


def test_plot_results_exports_summary_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plotting renders both WDM traces and writes the fixed summary filename."""
    wdm = _bare_runtime_wdm()
    saved_paths: list[str] = []

    def _record_savefig(path: str) -> None:
        saved_paths.append(path)

    monkeypatch.setattr(plt, "savefig", _record_savefig)

    try:
        wdm.plot_results(
            [
                {"time": 0.0, "Te_core": 1.1, "W_impurity": 0.0, "P_rad": 0.0, "status": "OK"},
                {
                    "time": 0.01,
                    "Te_core": 1.0,
                    "W_impurity": 1e-3,
                    "P_rad": 0.1,
                    "status": "OK",
                },
            ]
        )
    finally:
        plt.close("all")

    assert saved_paths == ["WDM_Simulation_Result.png"]


def test_plot_results_rejects_empty_history() -> None:
    """Plotting rejects an empty timeline before constructing a dataframe."""
    wdm = _bare_runtime_wdm()
    with pytest.raises(ValueError, match="history must not be empty"):
        wdm.plot_results([])
