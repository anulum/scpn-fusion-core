# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Nuclear Wall Interaction Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for nuclear wall interaction runtime hardening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.nuclear.nuclear_wall_interaction import (
    NuclearEngineeringLab,
    default_iter_config_path,
    run_nuclear_sim,
)


class _DummyNuclearLab:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.RR = np.zeros((2, 2), dtype=np.float64)
        self.ZZ = np.zeros((2, 2), dtype=np.float64)
        self.Psi = np.zeros((2, 2), dtype=np.float64)

    def simulate_ash_poisoning(
        self,
        burn_time_sec: int = 1000,
        tau_He_ratio: float = 5.0,
        pumping_efficiency: float = 1.0,
    ) -> dict[str, list[float]]:
        del burn_time_sec, tau_He_ratio
        final_f_he = 0.08 * (1.0 - 0.6 * float(pumping_efficiency))
        return {
            "time": [0.0, 1.0, 2.0],
            "P_fus": [110.0, 109.0, 108.0],
            "f_He": [0.01, 0.03, final_f_he],
            "Q": [],
        }

    def calculate_neutron_wall_loading(self):  # type: ignore[no-untyped-def]
        return (
            np.array([5.0, 5.5, 6.0], dtype=np.float64),
            np.array([-0.2, 0.0, 0.2], dtype=np.float64),
            np.array([1.0e19, 1.4e19, 1.2e19], dtype=np.float64),
        )

    def analyze_materials(self, wall_flux):  # type: ignore[no-untyped-def]
        del wall_flux
        return (
            {"Tungsten (W)": 4.0, "Eurofer (Steel)": 9.0},
            np.array([1.9, 2.4, 2.1], dtype=np.float64),
        )


def test_default_iter_config_path_points_to_repo_config() -> None:
    cfg = Path(default_iter_config_path())
    assert cfg.name == "iter_config.json"
    assert cfg.exists()


def test_run_nuclear_sim_noninteractive_returns_summary() -> None:
    summary = run_nuclear_sim(
        config_path="dummy.json",
        save_plot=False,
        verbose=False,
        lab_factory=_DummyNuclearLab,
    )
    for key in (
        "config_path",
        "good_final_f_he",
        "bad_final_f_he",
        "peak_wall_load_mw_m2",
        "avg_wall_load_mw_m2",
        "min_material_lifespan_years",
        "max_material_lifespan_years",
        "plot_saved",
        "plot_error",
    ):
        assert key in summary
    assert summary["config_path"] == "dummy.json"
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert np.isfinite(summary["peak_wall_load_mw_m2"])
    assert np.isfinite(summary["avg_wall_load_mw_m2"])
    assert summary["min_material_lifespan_years"] <= summary["max_material_lifespan_years"]


def test_pumping_efficiency_reduces_helium_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab = NuclearEngineeringLab(default_iter_config_path())
    monkeypatch.setattr(lab, "solve_equilibrium", lambda: None)
    low = lab.simulate_ash_poisoning(
        burn_time_sec=40,
        tau_He_ratio=8.0,
        pumping_efficiency=0.2,
    )
    high = lab.simulate_ash_poisoning(
        burn_time_sec=40,
        tau_He_ratio=8.0,
        pumping_efficiency=1.0,
    )
    assert high["f_He"][-1] < low["f_He"][-1]
    assert high["P_fus"][-1] >= low["P_fus"][-1]


def test_pumping_efficiency_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab = NuclearEngineeringLab(default_iter_config_path())
    monkeypatch.setattr(lab, "solve_equilibrium", lambda: None)
    with pytest.raises(ValueError, match="pumping_efficiency"):
        lab.simulate_ash_poisoning(
            burn_time_sec=10,
            tau_He_ratio=5.0,
            pumping_efficiency=1.1,
        )
