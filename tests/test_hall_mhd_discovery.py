from __future__ import annotations

import numpy as np

from scpn_fusion.core.hall_mhd_discovery import HallMHD, spitzer_resistivity


def test_spitzer_resistivity_handles_nonpositive_temperature() -> None:
    assert spitzer_resistivity(0.0) == 1e-4
    assert spitzer_resistivity(-1.0) == 1e-4


def test_hall_mhd_step_returns_finite_energy() -> None:
    sim = HallMHD(N=8, eta=1e-4, nu=1e-4)
    total, zonal = sim.step()
    assert np.isfinite(total)
    assert np.isfinite(zonal)
    assert total >= 0.0


def test_hall_mhd_parameter_sweep_small_grid() -> None:
    sim = HallMHD(N=8)
    result = sim.parameter_sweep((1e-4, 2e-4), (1e-4, 2e-4), n_steps=2, sim_steps=2)
    assert set(result.keys()) == {"eta", "nu", "growth_rate"}
    assert len(result["eta"]) == 4
    assert len(result["nu"]) == 4
    assert len(result["growth_rate"]) == 4


def test_hall_mhd_tearing_threshold_bounds() -> None:
    sim = HallMHD(N=8)
    result = sim.find_tearing_threshold(eta_range=(1e-5, 1e-4), n_bisect=2, sim_steps=2)
    assert result["lo"] <= result["threshold_eta"] <= result["hi"]
