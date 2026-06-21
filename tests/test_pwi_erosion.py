# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — PWI Erosion Tests
"""Tests for the plasma-wall-interaction sputtering yield and erosion model."""

from __future__ import annotations

import pytest

from scpn_fusion.nuclear.pwi_erosion import SputteringPhysics, run_pwi_demo


class TestSputteringPhysicsInit:
    """Material parameter selection and redeposition clamping."""

    def test_tungsten_defaults(self) -> None:
        """Tungsten selects its threshold energy and atomic mass."""
        sp = SputteringPhysics("Tungsten")
        assert sp.E_th == 200.0
        assert sp.atomic_mass == pytest.approx(183.84)

    def test_carbon_fallback(self) -> None:
        """Carbon selects its lower threshold and yield prefactor."""
        sp = SputteringPhysics("Carbon")
        assert sp.E_th == 30.0
        assert sp.Q == 0.1

    def test_redeposition_clipped(self) -> None:
        """The redeposition factor is clamped into the physical [0, 1) range."""
        sp = SputteringPhysics(redeposition_factor=1.5)
        assert sp.redeposition_factor <= 0.999
        sp2 = SputteringPhysics(redeposition_factor=-0.5)
        assert sp2.redeposition_factor >= 0.0


class TestCalculateYield:
    """Bohdansky-style sputtering-yield behaviour."""

    def test_below_threshold_zero(self) -> None:
        """An impact energy at or below threshold yields nothing."""
        sp = SputteringPhysics("Tungsten")
        assert sp.calculate_yield(100.0) == 0.0

    def test_above_threshold_positive(self) -> None:
        """An impact energy above threshold gives a positive yield."""
        sp = SputteringPhysics("Tungsten")
        Y = sp.calculate_yield(500.0)
        assert Y > 0

    def test_just_above_threshold_clamps_threshold_term(self) -> None:
        """Energy fractionally above threshold keeps the yield finite and non-negative."""
        sp = SputteringPhysics("Tungsten")
        Y = sp.calculate_yield(sp.E_th + 1.0)
        assert Y >= 0.0

    def test_yield_increases_with_energy(self) -> None:
        """Yield rises with impact energy above threshold."""
        sp = SputteringPhysics("Tungsten")
        Y1 = sp.calculate_yield(300.0)
        Y2 = sp.calculate_yield(1000.0)
        assert Y2 > Y1

    def test_oblique_angle_increases_yield(self) -> None:
        """Oblique incidence enhances the sputtering yield."""
        sp = SputteringPhysics("Tungsten")
        Y_normal = sp.calculate_yield(500.0, angle_deg=0.0)
        Y_oblique = sp.calculate_yield(500.0, angle_deg=70.0)
        assert Y_oblique >= Y_normal

    def test_rejects_nan_energy(self) -> None:
        """A non-finite impact energy is rejected."""
        sp = SputteringPhysics()
        with pytest.raises(ValueError, match="finite"):
            sp.calculate_yield(float("nan"))

    def test_rejects_nan_angle(self) -> None:
        """A non-finite incidence angle is rejected."""
        sp = SputteringPhysics()
        with pytest.raises(ValueError, match="finite"):
            sp.calculate_yield(500.0, angle_deg=float("inf"))


class TestCalculateErosionRate:
    """Net erosion and impurity-source accounting."""

    def test_returns_expected_keys(self) -> None:
        """The erosion summary carries every expected metric."""
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

    def test_zero_flux_zero_erosion(self) -> None:
        """Zero incident flux gives zero net flux and erosion."""
        sp = SputteringPhysics()
        result = sp.calculate_erosion_rate(0.0, T_ion_eV=50.0)
        assert result["Net_Flux"] == 0.0
        assert result["Erosion_mm_year"] == 0.0

    def test_erosion_increases_with_temperature(self) -> None:
        """Erosion rises with ion temperature at fixed flux."""
        sp = SputteringPhysics()
        r1 = sp.calculate_erosion_rate(1e24, T_ion_eV=10.0)
        r2 = sp.calculate_erosion_rate(1e24, T_ion_eV=100.0)
        assert r2["Erosion_mm_year"] >= r1["Erosion_mm_year"]

    def test_rejects_negative_flux(self) -> None:
        """A negative incident flux is rejected."""
        sp = SputteringPhysics()
        with pytest.raises(ValueError, match="flux"):
            sp.calculate_erosion_rate(-1e24, T_ion_eV=50.0)

    def test_rejects_negative_temperature(self) -> None:
        """A negative ion temperature is rejected."""
        sp = SputteringPhysics()
        with pytest.raises(ValueError, match="T_ion_eV"):
            sp.calculate_erosion_rate(1e24, T_ion_eV=-5.0)

    def test_redeposition_reduces_net_flux(self) -> None:
        """Higher redeposition lowers the net eroded flux."""
        sp_low = SputteringPhysics(redeposition_factor=0.1)
        sp_high = SputteringPhysics(redeposition_factor=0.95)
        r_low = sp_low.calculate_erosion_rate(1e24, T_ion_eV=50.0)
        r_high = sp_high.calculate_erosion_rate(1e24, T_ion_eV=50.0)
        assert r_high["Net_Flux"] < r_low["Net_Flux"]


class TestRunPWIDemo:
    """End-to-end PWI erosion scan and rendering."""

    def test_returns_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The scan returns a summary dictionary without plotting."""
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)
        result = run_pwi_demo(num_points=10, save_plot=False, verbose=False)
        assert isinstance(result, dict)
        assert "max_erosion_mm_year" in result

    def test_carbon_material(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The scan honours a non-default wall material."""
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)
        result = run_pwi_demo(material="Carbon", num_points=10, save_plot=False, verbose=False)
        assert result["material"] == "Carbon"

    def test_renders_and_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The verbose plotting path renders, saves, and reports success."""
        import matplotlib.pyplot as plt

        saved: list[str] = []
        monkeypatch.setattr(plt, "savefig", lambda path, *a, **kw: saved.append(str(path)))
        monkeypatch.setattr(plt, "close", lambda *a, **kw: None)
        result = run_pwi_demo(num_points=10, save_plot=True, verbose=True)
        assert saved == ["PWI_Erosion_Result.png"]
        assert result["plot_saved"] is True

    def test_records_plot_error_on_render_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rendering failure is caught and reported without aborting the scan."""
        import matplotlib.pyplot as plt

        def _boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("backend down")

        monkeypatch.setattr(plt, "savefig", _boom)
        result = run_pwi_demo(num_points=10, save_plot=True, verbose=False)
        assert result["plot_saved"] is False
        assert isinstance(result["plot_error"], str)

    def test_rejects_invalid_temperature_range(self) -> None:
        """An inverted temperature range is rejected."""
        with pytest.raises(ValueError, match="temp_min_eV"):
            run_pwi_demo(temp_min_eV=100.0, temp_max_eV=10.0, save_plot=False, verbose=False)

    def test_rejects_too_few_points(self) -> None:
        """A scan with fewer than three points is rejected."""
        with pytest.raises(ValueError, match="num_points"):
            run_pwi_demo(num_points=2, save_plot=False, verbose=False)
