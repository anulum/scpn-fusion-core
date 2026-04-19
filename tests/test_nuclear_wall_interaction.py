# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — Nuclear Wall Interaction Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.nuclear.nuclear_wall_interaction import (
    MATERIALS,
    NuclearEngineeringLab,
    default_iter_config_path,
)


def _bare_lab():
    """Create a NuclearEngineeringLab without full kernel init."""
    lab = NuclearEngineeringLab.__new__(NuclearEngineeringLab)
    lab.RR = np.zeros((33, 33))
    lab.ZZ = np.zeros((33, 33))
    lab.Psi = np.zeros((33, 33))
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


class TestDefaultConfigPath:
    def test_returns_string(self):
        p = default_iter_config_path()
        assert isinstance(p, str)
        assert "iter_config" in p


class TestMaterialsDict:
    def test_has_tungsten(self):
        assert "Tungsten (W)" in MATERIALS

    def test_has_eurofer(self):
        assert "Eurofer (Steel)" in MATERIALS

    def test_dpa_limits_positive(self):
        for mat, props in MATERIALS.items():
            assert props["dpa_limit"] > 0
            assert props["sigma_dpa"] > 0


class TestCalculateSputteringYield:
    def test_tungsten_below_threshold(self):
        lab = _bare_lab()
        Y = lab.calculate_sputtering_yield("Tungsten (W)", E_inc_eV=100.0)
        assert Y == 0.0

    def test_tungsten_above_threshold(self):
        lab = _bare_lab()
        Y = lab.calculate_sputtering_yield("Tungsten (W)", E_inc_eV=500.0)
        assert Y > 0

    def test_beryllium_low_threshold(self):
        lab = _bare_lab()
        Y = lab.calculate_sputtering_yield("Beryllium (Be)", E_inc_eV=50.0)
        assert Y > 0

    def test_unknown_material_zero(self):
        lab = _bare_lab()
        Y = lab.calculate_sputtering_yield("Unobtanium", E_inc_eV=500.0)
        assert Y == 0.0

    def test_angle_increases_yield(self):
        lab = _bare_lab()
        Y_normal = lab.calculate_sputtering_yield("Tungsten (W)", E_inc_eV=500.0, angle_deg=0.0)
        Y_oblique = lab.calculate_sputtering_yield("Tungsten (W)", E_inc_eV=500.0, angle_deg=70.0)
        assert Y_oblique >= Y_normal

    def test_rejects_negative_energy(self):
        lab = _bare_lab()
        with pytest.raises(ValueError, match="finite"):
            lab.calculate_sputtering_yield("Tungsten (W)", E_inc_eV=-1.0)

    def test_rejects_extreme_angle(self):
        lab = _bare_lab()
        with pytest.raises(ValueError, match="angle"):
            lab.calculate_sputtering_yield("Tungsten (W)", E_inc_eV=500.0, angle_deg=90.0)

    def test_yield_bounded(self):
        lab = _bare_lab()
        Y = lab.calculate_sputtering_yield("Beryllium (Be)", E_inc_eV=10000.0, angle_deg=85.0)
        assert Y <= 5.0


class TestGenerateFirstWall:
    def test_returns_two_arrays(self):
        lab = _bare_lab()
        Rw, Zw = lab.generate_first_wall()
        assert len(Rw) == 200
        assert len(Zw) == 200

    def test_wall_is_closed(self):
        lab = _bare_lab()
        Rw, Zw = lab.generate_first_wall()
        assert abs(Rw[0] - Rw[-1]) < 0.1
        assert abs(Zw[0] - Zw[-1]) < 0.1


class TestAnalyzeMaterials:
    def test_returns_lifespans(self):
        lab = _bare_lab()
        flux = np.ones(200) * 1e18
        lifespans, mw_load = lab.analyze_materials(flux)
        assert isinstance(lifespans, dict)
        assert len(lifespans) == len(MATERIALS)
        for name, years in lifespans.items():
            assert years > 0
            assert np.isfinite(years)

    def test_mw_load_positive(self):
        lab = _bare_lab()
        flux = np.ones(200) * 1e18
        _, mw_load = lab.analyze_materials(flux)
        assert np.all(mw_load > 0)


class TestBuildNeutronSourceMap:
    def test_source_map_shape(self):
        lab = _bare_lab()
        source = lab._build_neutron_source_map()
        assert source.shape == (33, 33)
        assert np.all(np.isfinite(source))
        assert np.any(source > 0)


class TestSimulateAshPoisoning:
    def test_rejects_bad_burn_time(self):
        lab = _bare_lab()
        with pytest.raises(ValueError, match="burn_time"):
            lab.simulate_ash_poisoning(burn_time_sec=0)

    def test_rejects_bad_tau_ratio(self):
        lab = _bare_lab()
        with pytest.raises(ValueError, match="tau_He_ratio"):
            lab.simulate_ash_poisoning(tau_He_ratio=-1.0)

    def test_rejects_bad_pumping(self):
        lab = _bare_lab()
        with pytest.raises(ValueError, match="pumping_efficiency"):
            lab.simulate_ash_poisoning(pumping_efficiency=1.5)
