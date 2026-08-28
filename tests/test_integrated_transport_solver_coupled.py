# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coupled Integrated Transport Runtime Tests
"""Behavioral tests for the public coupled transport step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core import CoupledTransportInputs
from scpn_fusion.core.integrated_transport_solver import TransportSolver

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "validation" / "iter_config.json"


def _inputs(**overrides: float) -> CoupledTransportInputs:
    values = {
        "time_s": 0.0,
        "dt_s": 0.01,
        "major_radius_m": 6.2,
        "minor_radius_m": 2.0,
        "magnetic_field_t": 5.3,
        "effective_charge": 1.5,
        "ion_heat_diffusivity_m2_s": 1.0,
        "electron_heat_diffusivity_m2_s": 1.0,
        "electron_particle_diffusivity_m2_s": 0.3,
        "heat_power_w": 1.0e7,
        "electron_heat_fraction": 0.6,
        "heat_center_rho": 0.3,
        "heat_width_rho": 0.25,
        "particle_rate_s": 1.0e20,
        "particle_center_rho": 0.45,
        "particle_width_rho": 0.25,
        "driven_current_a": 1.0e5,
        "current_center_rho": 0.35,
        "current_width_rho": 0.2,
        "ion_electron_exchange_rate_s": 1.0,
        "ion_temperature_edge_kev": 0.55,
        "electron_temperature_edge_kev": 0.45,
        "electron_density_edge_1e19_m3": 2.05,
        "resistivity_multiplier": 20.0,
    }
    values.update(overrides)
    return CoupledTransportInputs(**values)


def _solver(nr: int = 17) -> TransportSolver:
    solver = TransportSolver(CONFIG, nr=nr)
    solver.Ti = 0.5 + 4.5 * (1.0 - solver.rho**2)
    solver.Te = 0.4 + 3.6 * (1.0 - solver.rho**2)
    solver.ne = 2.0 + 3.0 * (1.0 - solver.rho**2)
    return solver


def test_coupled_step_advances_all_states_and_closes_sources() -> None:
    solver = _solver()
    ti_before = solver.Ti.copy()
    te_before = solver.Te.copy()
    ne_before = solver.ne.copy()

    result = solver.evolve_coupled_transport(_inputs())

    assert result.converged
    assert result.time_s == pytest.approx(0.01)
    assert not np.array_equal(result.ion_temperature_kev, ti_before)
    assert not np.array_equal(result.electron_temperature_kev, te_before)
    assert not np.array_equal(result.electron_density_1e19_m3, ne_before)
    assert np.all(np.isfinite(result.poloidal_flux_wb_per_rad))
    assert result.ion_temperature_kev[-1] == pytest.approx(0.55)
    assert result.electron_temperature_kev[-1] == pytest.approx(0.45)
    assert result.electron_density_1e19_m3[-1] == pytest.approx(2.05)
    budget = result.budget
    assert budget.ion_heat_source_reconstructed_w == pytest.approx(4.0e6, rel=1e-12)
    assert budget.electron_heat_source_reconstructed_w == pytest.approx(6.0e6, rel=1e-12)
    assert budget.particle_source_reconstructed_s == pytest.approx(1.0e20, rel=1e-12)
    assert budget.driven_current_reconstructed_a == pytest.approx(1.0e5, rel=1e-12)
    assert budget.ion_electron_exchange_closure_j == pytest.approx(0.0, abs=1e-6)
    assert budget.ion_temperature_linear_residual_linf < 1e-10
    assert budget.electron_temperature_linear_residual_linf < 1e-10
    assert budget.electron_density_linear_residual_linf < 1e-10
    assert budget.current_linear_residual_linf < 1e-10


def test_exchange_reduces_temperature_separation_without_net_energy() -> None:
    without_exchange = _solver()
    with_exchange = _solver()
    no_exchange_result = without_exchange.evolve_coupled_transport(
        _inputs(ion_electron_exchange_rate_s=0.0)
    )
    exchange_result = with_exchange.evolve_coupled_transport(
        _inputs(ion_electron_exchange_rate_s=5.0)
    )

    no_exchange_gap = np.linalg.norm(
        no_exchange_result.ion_temperature_kev[:-1]
        - no_exchange_result.electron_temperature_kev[:-1]
    )
    exchange_gap = np.linalg.norm(
        exchange_result.ion_temperature_kev[:-1] - exchange_result.electron_temperature_kev[:-1]
    )
    assert exchange_gap < no_exchange_gap
    assert exchange_result.budget.ion_electron_exchange_closure_j == pytest.approx(0.0, abs=1e-6)


def test_coupled_flux_initialization_is_used_by_next_step() -> None:
    solver = _solver()
    initial_flux = 0.2 * (1.0 - solver.rho**2)
    solver.set_coupled_flux_profile(
        initial_flux,
        major_radius_m=6.2,
        minor_radius_m=2.0,
        magnetic_field_t=5.3,
    )

    result = solver.evolve_coupled_transport(_inputs())

    assert result.budget.flux_l2_before == pytest.approx(np.linalg.norm(initial_flux))
    assert not np.array_equal(result.poloidal_flux_wb_per_rad, initial_flux)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dt_s", 0.0),
        ("electron_heat_fraction", 1.1),
        ("electron_heat_fraction", float("nan")),
        ("heat_center_rho", -0.1),
        ("particle_rate_s", -1.0),
        ("resistivity_multiplier", float("nan")),
    ],
)
def test_coupled_inputs_reject_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        _inputs(**{field: value})


def test_coupled_step_rejects_nonuniform_grid() -> None:
    solver = _solver()
    solver.rho[3] += 1.0e-3
    with pytest.raises(ValueError, match="uniform"):
        solver.evolve_coupled_transport(_inputs())
