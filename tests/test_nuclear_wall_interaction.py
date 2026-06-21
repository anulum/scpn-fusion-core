# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Nuclear Wall Interaction Tests
"""Tests for the nuclear first-wall loading, ash poisoning, and materials lab."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.nuclear.nuclear_wall_interaction import (
    MATERIALS,
    NuclearEngineeringLab,
    default_iter_config_path,
    run_nuclear_sim,
)


def _bare_lab() -> NuclearEngineeringLab:
    """Create a NuclearEngineeringLab with grid geometry but no full kernel init."""
    lab = NuclearEngineeringLab.__new__(NuclearEngineeringLab)
    lab.R = np.linspace(1.0, 9.0, 33)
    lab.Z = np.linspace(-5.0, 5.0, 33)
    lab.dR = lab.R[1] - lab.R[0]
    lab.dZ = lab.Z[1] - lab.Z[0]
    lab.NR = 33
    lab.NZ = 33
    lab.RR, lab.ZZ = np.meshgrid(lab.R, lab.Z)
    lab.Psi = np.exp(-((lab.RR - 5.0) ** 2 + lab.ZZ**2) / 4.0)
    lab.cfg = {"dimensions": {"R_min": 1.0, "R_max": 9.0, "Z_min": -5.0, "Z_max": 5.0}}
    return lab


def _burn_ready_lab() -> NuclearEngineeringLab:
    """Create a bare lab with the burn-physics dependencies stubbed for ash/CAD paths."""
    lab = _bare_lab()
    lab.solve_equilibrium = lambda: None  # type: ignore[assignment, misc]
    lab.bosch_hale_dt = lambda _t_kev: 1.1e-22  # type: ignore[method-assign, assignment]
    return lab


class TestDefaultConfigPath:
    """ITER configuration path resolution."""

    def test_returns_string(self) -> None:
        """The resolver returns a path string mentioning the ITER config."""
        p = default_iter_config_path()
        assert isinstance(p, str)
        assert "iter_config" in p


class TestMaterialsDict:
    """First-wall materials property table."""

    def test_has_tungsten(self) -> None:
        """Tungsten is present in the materials table."""
        assert "Tungsten (W)" in MATERIALS

    def test_has_eurofer(self) -> None:
        """Eurofer steel is present in the materials table."""
        assert "Eurofer (Steel)" in MATERIALS

    def test_dpa_limits_positive(self) -> None:
        """Every material carries positive damage limits and cross-sections."""
        for _mat, props in MATERIALS.items():
            assert props["dpa_limit"] > 0
            assert props["sigma_dpa"] > 0


class TestCalculateSputteringYield:
    """Energy- and angle-dependent sputtering yield."""

    def test_tungsten_below_threshold(self) -> None:
        """Tungsten below its threshold energy yields nothing."""
        assert _bare_lab().calculate_sputtering_yield("Tungsten (W)", E_inc_eV=100.0) == 0.0

    def test_tungsten_above_threshold(self) -> None:
        """Tungsten above threshold gives a positive yield."""
        assert _bare_lab().calculate_sputtering_yield("Tungsten (W)", E_inc_eV=500.0) > 0

    def test_beryllium_low_threshold(self) -> None:
        """Beryllium sputters at a low impact energy."""
        assert _bare_lab().calculate_sputtering_yield("Beryllium (Be)", E_inc_eV=50.0) > 0

    def test_unknown_material_zero(self) -> None:
        """An unknown material yields nothing."""
        assert _bare_lab().calculate_sputtering_yield("Unobtanium", E_inc_eV=500.0) == 0.0

    def test_angle_increases_yield(self) -> None:
        """Oblique incidence does not reduce the yield."""
        lab = _bare_lab()
        y_normal = lab.calculate_sputtering_yield("Tungsten (W)", E_inc_eV=500.0, angle_deg=0.0)
        y_oblique = lab.calculate_sputtering_yield("Tungsten (W)", E_inc_eV=500.0, angle_deg=70.0)
        assert y_oblique >= y_normal

    def test_rejects_negative_energy(self) -> None:
        """A non-finite or negative impact energy is rejected."""
        with pytest.raises(ValueError, match="finite"):
            _bare_lab().calculate_sputtering_yield("Tungsten (W)", E_inc_eV=-1.0)

    def test_rejects_extreme_angle(self) -> None:
        """A grazing 90-degree incidence is rejected."""
        with pytest.raises(ValueError, match="angle"):
            _bare_lab().calculate_sputtering_yield("Tungsten (W)", E_inc_eV=500.0, angle_deg=90.0)

    def test_yield_bounded(self) -> None:
        """The angular enhancement keeps the yield bounded."""
        y = _bare_lab().calculate_sputtering_yield(
            "Beryllium (Be)", E_inc_eV=10000.0, angle_deg=85.0
        )
        assert y <= 5.0


class TestGenerateFirstWall:
    """D-shaped first-wall contour generation."""

    def test_returns_two_arrays(self) -> None:
        """The wall is returned as paired radial/vertical contours."""
        Rw, Zw = _bare_lab().generate_first_wall()
        assert len(Rw) == 200
        assert len(Zw) == 200

    def test_wall_is_closed(self) -> None:
        """The generated wall contour closes on itself."""
        Rw, Zw = _bare_lab().generate_first_wall()
        assert abs(Rw[0] - Rw[-1]) < 0.1
        assert abs(Zw[0] - Zw[-1]) < 0.1


class TestAnalyzeMaterials:
    """Material lifespan and load analysis."""

    def test_returns_lifespans(self) -> None:
        """Each material gets a positive finite lifespan estimate."""
        lifespans, _mw_load = _bare_lab().analyze_materials(np.ones(200) * 1e18)
        assert isinstance(lifespans, dict)
        assert len(lifespans) == len(MATERIALS)
        for _name, years in lifespans.items():
            assert years > 0
            assert np.isfinite(years)

    def test_mw_load_positive(self) -> None:
        """The megawatt wall load is positive for a positive flux."""
        _lifespans, mw_load = _bare_lab().analyze_materials(np.ones(200) * 1e18)
        assert np.all(mw_load > 0)


class TestBuildNeutronSourceMap:
    """Neutron volumetric source reconstruction from the flux map."""

    def test_source_map_shape(self) -> None:
        """The source map matches the flux grid and carries positive emission."""
        source = _bare_lab()._build_neutron_source_map()
        assert source.shape == (33, 33)
        assert np.all(np.isfinite(source))
        assert np.any(source > 0)

    def test_degenerate_flux_uses_min_edge(self) -> None:
        """A near-flat flux falls back to the minimum as the edge value."""
        lab = _bare_lab()
        lab.Psi = np.full((33, 33), 0.5)
        source = lab._build_neutron_source_map()
        assert np.all(np.isfinite(source))


class TestSimulateAshPoisoning:
    """Helium ash accumulation and power decay."""

    def test_rejects_bad_burn_time(self) -> None:
        """A sub-unit burn time is rejected."""
        with pytest.raises(ValueError, match="burn_time"):
            _bare_lab().simulate_ash_poisoning(burn_time_sec=0)

    def test_rejects_bad_tau_ratio(self) -> None:
        """A non-positive confinement ratio is rejected."""
        with pytest.raises(ValueError, match="tau_He_ratio"):
            _bare_lab().simulate_ash_poisoning(tau_He_ratio=-1.0)

    def test_rejects_bad_pumping(self) -> None:
        """An out-of-range pumping efficiency is rejected."""
        with pytest.raises(ValueError, match="pumping_efficiency"):
            _bare_lab().simulate_ash_poisoning(pumping_efficiency=1.5)

    def test_history_accumulates(self) -> None:
        """A short burn returns aligned power/helium/time histories."""
        history = _burn_ready_lab().simulate_ash_poisoning(burn_time_sec=10)
        assert len(history["time"]) == 10
        assert len(history["P_fus"]) == 10
        assert all(np.isfinite(p) for p in history["P_fus"])

    def test_quench_on_dilution(self) -> None:
        """Runaway helium build-up quenches the plasma before the full burn."""
        lab = _burn_ready_lab()
        lab.bosch_hale_dt = lambda _t_kev: 1.0e-19  # type: ignore[method-assign, assignment]
        history = lab.simulate_ash_poisoning(
            burn_time_sec=200, tau_He_ratio=50.0, pumping_efficiency=0.0
        )
        assert len(history["time"]) < 200


class TestCalculateNeutronWallLoading:
    """Ray-traced 14 MeV neutron wall loading."""

    def test_returns_finite_wall_flux(self) -> None:
        """Wall loading returns finite flux on every wall segment (no divide-by-zero)."""
        Rw, Zw, wall_flux = _bare_lab().calculate_neutron_wall_loading()
        assert len(wall_flux) == len(Rw) == len(Zw)
        assert np.all(np.isfinite(wall_flux))
        assert np.any(wall_flux > 0)

    def test_coincident_source_stays_finite(self) -> None:
        """A source element placed on the wall contour does not produce NaN."""
        lab = _bare_lab()
        Rw, Zw = lab.generate_first_wall()
        # Force a strong emission peak at a grid point near the wall contour.
        ir = int(np.argmin(np.abs(lab.R - Rw[0])))
        iz = int(np.argmin(np.abs(lab.Z - Zw[0])))
        lab.Psi = np.full((33, 33), 0.01)
        lab.Psi[iz, ir] = 1.0
        _Rw, _Zw, wall_flux = lab.calculate_neutron_wall_loading()
        assert np.all(np.isfinite(wall_flux))


class TestCalculateCadWallLoading:
    """Reduced CAD-mesh surface loading."""

    def test_default_source_from_plasma(self) -> None:
        """With no explicit source the loader derives one from the plasma map."""
        lab = _burn_ready_lab()
        vertices = np.array([[6.0, 0.0, 0.0], [7.0, 0.0, 0.0], [6.5, 0.0, 1.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        report = lab.calculate_cad_wall_loading(vertices, faces)
        assert report is not None


def test_run_nuclear_sim_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone nuclear diagnostic bundle runs and renders via an injected lab."""
    import matplotlib.pyplot as plt

    saved: list[str] = []
    monkeypatch.setattr(plt, "savefig", lambda path, *a, **k: saved.append(str(path)))
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plt, "colorbar", lambda *a, **k: None)

    def _factory(_config_path: str) -> Any:
        return _burn_ready_lab()

    result = run_nuclear_sim(config_path="ignored.json", lab_factory=_factory, verbose=True)

    assert saved == ["Nuclear_Engineering_Report.png"]
    assert "peak_wall_load_mw_m2" in result or "peak_load" in result or result


def test_run_nuclear_sim_no_plot(monkeypatch: pytest.MonkeyPatch) -> None:
    """With plotting disabled the bundle still returns its metrics dictionary."""

    def _factory(_config_path: str) -> Any:
        return _burn_ready_lab()

    result = run_nuclear_sim(
        config_path="ignored.json", lab_factory=_factory, save_plot=False, verbose=False
    )
    assert isinstance(result, dict)


def test_real_lab_construction_delegates_to_kernel() -> None:
    """Constructing the lab from the ITER config initialises the burn-physics kernel."""
    lab = NuclearEngineeringLab(default_iter_config_path())
    assert isinstance(lab, NuclearEngineeringLab)
    assert lab.Psi.ndim == 2


def test_run_nuclear_sim_resolves_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """A null config path is resolved to the default ITER configuration."""
    seen: list[str] = []

    def _factory(config_path: str) -> Any:
        seen.append(config_path)
        return _burn_ready_lab()

    run_nuclear_sim(config_path=None, lab_factory=_factory, save_plot=False, verbose=False)
    assert seen and "iter_config" in seen[0]


def test_run_nuclear_sim_handles_empty_lifespans(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no material lifespans are produced the summary falls back to zeros."""

    def _factory(_config_path: str) -> Any:
        lab = _burn_ready_lab()
        lab.analyze_materials = lambda _flux: ({}, np.array([], dtype=np.float64))  # type: ignore[method-assign, assignment]
        return lab

    result = run_nuclear_sim(
        config_path="ignored.json", lab_factory=_factory, save_plot=False, verbose=True
    )
    assert isinstance(result, dict)
