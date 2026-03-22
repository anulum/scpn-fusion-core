# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — PWI Erosion Tests
from __future__ import annotations

import pytest

from scpn_fusion.nuclear.pwi_erosion import SputteringPhysics, run_pwi_demo


class TestSputteringPhysicsInit:
    def test_tungsten_defaults(self):
        sp = SputteringPhysics("Tungsten")
        assert sp.E_th == 200.0
        assert sp.atomic_mass == pytest.approx(183.84)

    def test_carbon_fallback(self):
        sp = SputteringPhysics("Carbon")
        assert sp.E_th == 30.0
        assert sp.Q == 0.1

    def test_redeposition_clipped(self):
        sp = SputteringPhysics(redeposition_factor=1.5)
        assert sp.redeposition_factor <= 0.999
        sp2 = SputteringPhysics(redeposition_factor=-0.5)
        assert sp2.redeposition_factor >= 0.0


class TestCalculateYield:
    def test_below_threshold_zero(self):
        sp = SputteringPhysics("Tungsten")
        assert sp.calculate_yield(100.0) == 0.0

    def test_above_threshold_positive(self):
        sp = SputteringPhysics("Tungsten")
        Y = sp.calculate_yield(500.0)
        assert Y > 0

    def test_yield_increases_with_energy(self):
        sp = SputteringPhysics("Tungsten")
        Y1 = sp.calculate_yield(300.0)
        Y2 = sp.calculate_yield(1000.0)
        assert Y2 > Y1

    def test_oblique_angle_increases_yield(self):
        sp = SputteringPhysics("Tungsten")
        Y_normal = sp.calculate_yield(500.0, angle_deg=0.0)
        Y_oblique = sp.calculate_yield(500.0, angle_deg=70.0)
        assert Y_oblique >= Y_normal

    def test_rejects_nan_energy(self):
        sp = SputteringPhysics()
        with pytest.raises(ValueError, match="finite"):
            sp.calculate_yield(float("nan"))

    def test_rejects_nan_angle(self):
        sp = SputteringPhysics()
        with pytest.raises(ValueError, match="finite"):
            sp.calculate_yield(500.0, angle_deg=float("inf"))


class TestCalculateErosionRate:
    def test_returns_expected_keys(self):
        sp = SputteringPhysics()
        result = sp.calculate_erosion_rate(1e24, T_ion_eV=50.0)
        for key in (
            "Yield",
            "E_impact",
            "Net_Flux",
            "Redeposition",
            "Erosion_mm_year",
            "Impurity_Source",
        ):
            assert key in result

    def test_zero_flux_zero_erosion(self):
        sp = SputteringPhysics()
        result = sp.calculate_erosion_rate(0.0, T_ion_eV=50.0)
        assert result["Net_Flux"] == 0.0
        assert result["Erosion_mm_year"] == 0.0

    def test_erosion_increases_with_temperature(self):
        sp = SputteringPhysics()
        r1 = sp.calculate_erosion_rate(1e24, T_ion_eV=10.0)
        r2 = sp.calculate_erosion_rate(1e24, T_ion_eV=100.0)
        assert r2["Erosion_mm_year"] >= r1["Erosion_mm_year"]

    def test_rejects_negative_flux(self):
        sp = SputteringPhysics()
        with pytest.raises(ValueError, match="flux"):
            sp.calculate_erosion_rate(-1e24, T_ion_eV=50.0)

    def test_redeposition_reduces_net_flux(self):
        sp_low = SputteringPhysics(redeposition_factor=0.1)
        sp_high = SputteringPhysics(redeposition_factor=0.95)
        r_low = sp_low.calculate_erosion_rate(1e24, T_ion_eV=50.0)
        r_high = sp_high.calculate_erosion_rate(1e24, T_ion_eV=50.0)
        assert r_high["Net_Flux"] < r_low["Net_Flux"]


class TestRunPWIDemo:
    def test_returns_summary(self, monkeypatch):
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)
        result = run_pwi_demo(num_points=10, save_plot=False, verbose=False)
        assert isinstance(result, dict)
        assert "max_erosion_mm_year" in result

    def test_carbon_material(self, monkeypatch):
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)
        result = run_pwi_demo(material="Carbon", num_points=10, save_plot=False, verbose=False)
        assert result["material"] == "Carbon"
