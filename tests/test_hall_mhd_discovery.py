# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the Hall-MHD zonal-flow discovery solver and tearing search."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_fusion.core.hall_mhd_discovery as hall_mhd_discovery
from scpn_fusion.core.hall_mhd_discovery import HallMHD, spitzer_resistivity


def test_spitzer_resistivity_handles_nonpositive_temperature() -> None:
    """Non-positive electron temperature returns the resistivity floor."""
    assert spitzer_resistivity(0.0) == 1e-4
    assert spitzer_resistivity(-1.0) == 1e-4


def test_hall_mhd_step_returns_finite_energy() -> None:
    """One Hall-MHD step returns finite total and zonal energies."""
    sim = HallMHD(N=8, eta=1e-4, nu=1e-4)
    total, zonal = sim.step()
    assert np.isfinite(total)
    assert np.isfinite(zonal)
    assert total >= 0.0


def test_hall_mhd_parameter_sweep_small_grid() -> None:
    """A small eta/nu sweep returns aligned growth-rate columns."""
    sim = HallMHD(N=8)
    result = sim.parameter_sweep((1e-4, 2e-4), (1e-4, 2e-4), n_steps=2, sim_steps=2)
    assert set(result.keys()) == {"eta", "nu", "growth_rate"}
    assert len(result["eta"]) == 4
    assert len(result["nu"]) == 4
    assert len(result["growth_rate"]) == 4


def test_hall_mhd_tearing_threshold_bounds() -> None:
    """The tearing-threshold bisection brackets its estimate."""
    sim = HallMHD(N=8)
    result = sim.find_tearing_threshold(eta_range=(1e-5, 1e-4), n_bisect=2, sim_steps=2)
    assert result["lo"] <= result["threshold_eta"] <= result["hi"]


def test_spitzer_resistivity_positive_temperature() -> None:
    """A positive electron temperature gives a finite positive resistivity."""
    eta = spitzer_resistivity(100.0, Z_eff=1.5)
    assert eta > 0.0
    assert np.isfinite(eta)


def test_parameter_sweep_uses_growth_history_window() -> None:
    """A long enough sweep populates the log-growth estimate from the history window."""
    sim = HallMHD(N=8)
    result = sim.parameter_sweep((1e-4, 2e-4), (1e-4, 2e-4), n_steps=2, sim_steps=15)
    assert len(result["growth_rate"]) == 4
    assert all(np.isfinite(g) for g in result["growth_rate"])


def test_find_tearing_threshold_brackets_with_history_window() -> None:
    """The bisection uses the long-history growth estimate and brackets the threshold."""
    sim = HallMHD(N=8)
    result = sim.find_tearing_threshold(eta_range=(1e-9, 1e-6), n_bisect=4, sim_steps=30)
    assert result["lo"] <= result["threshold_eta"] <= result["hi"]


def test_find_tearing_threshold_lowers_ceiling_when_unstable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A positive growth estimate pulls the upper resistivity bound downward."""

    def _growing_step(self: HallMHD) -> tuple[float, float]:
        energy = getattr(self, "_probe_energy", 1.0) * 1.5
        self._probe_energy = energy  # type: ignore[attr-defined]
        self.energy_history.append(energy)
        return energy, 0.0

    monkeypatch.setattr(HallMHD, "step", _growing_step)
    sim = HallMHD(N=8)
    result = sim.find_tearing_threshold(eta_range=(1e-6, 1e-2), n_bisect=3, sim_steps=25)
    # Every iteration sees growth > 0, so the ceiling collapses toward the floor.
    assert result["hi"] < 1e-2
    assert result["lo"] <= result["threshold_eta"] <= result["hi"]


def test_run_discovery_sim_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The zonal-flow discovery demo runs and renders both diagnostics."""
    import matplotlib.pyplot as plt

    saved: list[str] = []
    monkeypatch.setattr(hall_mhd_discovery, "STEPS", 20)
    monkeypatch.setattr(plt, "savefig", lambda path, *a, **k: saved.append(str(path)))
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    hall_mhd_discovery.run_discovery_sim()

    assert "Hall_MHD_Discovery.png" in saved
    assert "Hall_MHD_Structure.png" in saved
