# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Divertor Thermal Sim Tests
"""Tests for the 2-point divertor transport model and liquid-metal TEMHD exhaust."""

from __future__ import annotations

import pytest

from scpn_fusion.core.divertor_thermal_sim import DivertorLab


def test_2point_target_temperature_decreases_with_radiative_fraction() -> None:
    """Radiative cooling lowers the target temperature at fixed upstream conditions."""
    lab = DivertorLab(P_sol_MW=80.0, R_major=2.1, B_pol=2.5)
    t_u0, t_t0 = lab.solve_2point_transport(expansion_factor=15.0, f_rad=0.0)
    t_u1, t_t1 = lab.solve_2point_transport(expansion_factor=15.0, f_rad=0.8)
    assert t_u0 == pytest.approx(t_u1)
    assert t_t1 < t_t0
    assert 1.0 <= t_t1 <= t_u1


def test_2point_target_temperature_decreases_with_flux_expansion() -> None:
    """Greater flux expansion lowers the target temperature."""
    lab = DivertorLab(P_sol_MW=80.0, R_major=2.1, B_pol=2.5)
    _, t_t_small = lab.solve_2point_transport(expansion_factor=5.0, f_rad=0.3)
    _, t_t_large = lab.solve_2point_transport(expansion_factor=20.0, f_rad=0.3)
    assert t_t_large < t_t_small


def test_2point_transport_rejects_invalid_operating_inputs() -> None:
    """Invalid expansion factor or radiative fraction is rejected."""
    lab = DivertorLab(P_sol_MW=80.0, R_major=2.1, B_pol=2.5)
    with pytest.raises(ValueError, match="expansion_factor"):
        lab.solve_2point_transport(expansion_factor=0.0, f_rad=0.2)
    with pytest.raises(ValueError, match="f_rad"):
        lab.solve_2point_transport(expansion_factor=10.0, f_rad=1.0)


def _lab(p_sol: float = 80.0) -> DivertorLab:
    """Build a compact-pilot divertor lab at the given scrape-off-layer power."""
    return DivertorLab(P_sol_MW=p_sol, R_major=2.1, B_pol=2.5)


def test_calculate_heat_load_returns_positive_flux() -> None:
    """The unmitigated target heat load is a positive solid-surface flux."""
    assert _lab().calculate_heat_load(expansion_factor=10.0) > 0.0


def test_simulate_tungsten_reports_ok_and_melted() -> None:
    """The tungsten monoblock reports OK at a low load and MELTED at a high load."""
    low = _lab()
    low.q_target_solid = 1.0e6
    _, status_low = low.simulate_tungsten()
    assert status_low == "OK"

    high = _lab()
    high.q_target_solid = 5.0e8
    _, status_high = high.simulate_tungsten()
    assert status_high == "MELTED"


def test_simulate_lithium_vapor_shields_and_converges() -> None:
    """Self-consistent lithium vapour shielding returns a finite shielded state."""
    t_surf, q_surf, f_rad = _lab().simulate_lithium_vapor()
    assert t_surf > 0.0
    assert q_surf >= 0.0
    assert 0.0 <= f_rad <= 1.0


def test_simulate_lithium_vapor_rejects_invalid_inputs() -> None:
    """Out-of-range relaxation, iteration, or tolerance inputs are rejected."""
    lab = _lab()
    with pytest.raises(ValueError, match="relaxation"):
        lab.simulate_lithium_vapor(relaxation=1.0)
    with pytest.raises(ValueError, match="max_iter"):
        lab.simulate_lithium_vapor(max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        lab.simulate_lithium_vapor(tol=0.0)


def test_simulate_lithium_vapor_non_convergence_returns_best() -> None:
    """A single-iteration budget falls back to the best residual with a warning."""
    t_surf, q_surf, f_rad = _lab().simulate_lithium_vapor(max_iter=1, tol=1e-9)
    assert t_surf > 0.0
    assert 0.0 <= f_rad <= 1.0


def test_calculate_mhd_pressure_loss_returns_summary() -> None:
    """The Hartmann-corrected pressure-loss model returns a positive summary."""
    out = _lab().calculate_mhd_pressure_loss(flow_velocity_m_s=0.5)
    assert out["pressure_loss_pa"] > 0.0
    assert out["hartmann_number"] > 0.0
    assert out["flow_velocity_m_s"] == pytest.approx(0.5)


def test_estimate_evaporation_rate_increases_with_temperature() -> None:
    """Lithium evaporation rises with surface temperature and falls with flow."""
    lab = _lab()
    cool = lab.estimate_evaporation_rate(400.0, 0.5)
    hot = lab.estimate_evaporation_rate(700.0, 0.5)
    assert hot > cool > 0.0


def test_simulate_temhd_liquid_metal_returns_state() -> None:
    """The combined TEMHD divertor state reports a stability verdict."""
    state = _lab().simulate_temhd_liquid_metal(flow_velocity_m_s=0.6)
    assert isinstance(state["is_stable"], bool)
    flux = state["surface_heat_flux_w_m2"]
    pressure = state["pressure_loss_pa"]
    assert isinstance(flux, float)
    assert isinstance(pressure, float)
    assert flux >= 0.0
    assert pressure > 0.0


def test_run_divertor_sim_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The standalone divertor comparison runs and renders safely."""
    import matplotlib.pyplot as plt
    import scpn_fusion.core.divertor_thermal_sim as divertor_thermal_sim

    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    from scpn_fusion.core.divertor_thermal_sim import run_divertor_sim

    with caplog.at_level("INFO", logger=divertor_thermal_sim.__name__):
        run_divertor_sim()

    assert "Divertor comparison figure saved" in caplog.text


def test_simulate_lithium_vapor_adaptive_relaxation_under_high_flux() -> None:
    """A high incident flux drives the adaptive under-relaxation branch."""
    lab = _lab()
    lab.q_target_solid = 5.0e8
    t_surf, q_surf, f_rad = lab.simulate_lithium_vapor(relaxation=0.1, max_iter=50, tol=1e-6)
    assert t_surf > 0.0
    assert 0.0 <= f_rad <= 1.0
