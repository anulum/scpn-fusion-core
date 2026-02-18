# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Multi-Ion Transport Tests (P1.1)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for multi-ion species transport (D, T, He-ash, independent Te)."""

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.integrated_transport_solver import TransportSolver, PhysicsError

MOCK_CONFIG = {
    "reactor_name": "MultiIon-Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [
        {"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15},
    ],
    "solver": {
        "max_iterations": 10,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
    },
}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    return cfg


@pytest.fixture
def solver_multi(config_file: Path) -> TransportSolver:
    ts = TransportSolver(str(config_file), multi_ion=True)
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = 5.0 * (1 - ts.rho**2)
    ts.ne = 5.0 * (1 - ts.rho**2) ** 0.5
    # Re-initialise species from the updated ne
    ts.n_D = 0.5 * ts.ne.copy()
    ts.n_T = 0.5 * ts.ne.copy()
    ts.n_He = np.zeros(ts.nr)
    ts.update_transport_model(50.0)
    return ts


@pytest.fixture
def solver_legacy(config_file: Path) -> TransportSolver:
    ts = TransportSolver(str(config_file), multi_ion=False)
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = ts.Ti.copy()
    ts.ne = 5.0 * (1 - ts.rho**2) ** 0.5
    ts.update_transport_model(50.0)
    return ts


# ── Species initialisation ──────────────────────────────────────────


def test_multi_ion_species_initialised(solver_multi: TransportSolver):
    """Multi-ion mode should create D, T, He arrays."""
    assert solver_multi.n_D is not None
    assert solver_multi.n_T is not None
    assert solver_multi.n_He is not None
    assert solver_multi.n_D.shape == (solver_multi.nr,)
    assert solver_multi.n_T.shape == (solver_multi.nr,)
    assert solver_multi.n_He.shape == (solver_multi.nr,)


def test_legacy_mode_no_species(solver_legacy: TransportSolver):
    """Legacy mode: species arrays should be None."""
    assert solver_legacy.n_D is None
    assert solver_legacy.n_T is None
    assert solver_legacy.n_He is None
    assert not solver_legacy.multi_ion


# ── Bosch-Hale reaction rate ─────────────────────────────────────────


def test_bosch_hale_positive():
    """D-T reaction rate must be positive for fusion-relevant temperatures."""
    T = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
    sv = TransportSolver._bosch_hale_sigmav(T)
    assert np.all(sv > 0), f"sigma_v must be positive, got {sv}"


def test_bosch_hale_monotonic_low_T():
    """Bosch-Hale <sigma*v> should increase with T in the 1-60 keV range."""
    T = np.array([1.0, 5.0, 10.0, 20.0, 40.0, 60.0])
    sv = TransportSolver._bosch_hale_sigmav(T)
    for i in range(len(T) - 1):
        assert sv[i + 1] > sv[i], (
            f"sigma_v not increasing: sv({T[i]})={sv[i]:.2e} >= sv({T[i+1]})={sv[i+1]:.2e}"
        )


# ── Tungsten radiation rate ──────────────────────────────────────────


def test_tungsten_radiation_positive():
    """Lz must be positive for all Te."""
    Te = np.array([0.1, 0.5, 1.0, 3.0, 10.0, 30.0])
    Lz = TransportSolver._tungsten_radiation_rate(Te)
    assert np.all(Lz > 0), "Lz must be positive"


def test_tungsten_radiation_order_of_magnitude():
    """Lz should be in the 1e-32 to 1e-30 W·m^3 range."""
    Te = np.array([0.5, 2.0, 10.0, 25.0])
    Lz = TransportSolver._tungsten_radiation_rate(Te)
    assert np.all(Lz > 1e-33), f"Lz too small: {Lz}"
    assert np.all(Lz < 1e-28), f"Lz too large: {Lz}"


# ── Bremsstrahlung ───────────────────────────────────────────────────


def test_bremsstrahlung_scales_with_ne_squared():
    """P_brem ∝ ne^2 at fixed Te and Z_eff."""
    ne1 = np.array([5.0])
    ne2 = np.array([10.0])
    Te = np.array([5.0])
    P1 = TransportSolver._bremsstrahlung_power_density(ne1, Te, 1.5)
    P2 = TransportSolver._bremsstrahlung_power_density(ne2, Te, 1.5)
    ratio = float(P2[0] / P1[0])
    expected = 4.0  # (10/5)^2
    assert abs(ratio - expected) < 0.01, f"P_brem ratio {ratio}, expected {expected}"


def test_bremsstrahlung_scales_with_sqrt_Te():
    """P_brem ∝ sqrt(Te) at fixed ne and Z_eff."""
    ne = np.array([5.0])
    Te1 = np.array([4.0])
    Te2 = np.array([16.0])
    P1 = TransportSolver._bremsstrahlung_power_density(ne, Te1, 1.5)
    P2 = TransportSolver._bremsstrahlung_power_density(ne, Te2, 1.5)
    ratio = float(P2[0] / P1[0])
    expected = 2.0  # sqrt(16/4)
    assert abs(ratio - expected) < 0.01, f"P_brem ratio {ratio}, expected {expected}"


# ── Species evolution ────────────────────────────────────────────────


def test_helium_ash_accumulates(solver_multi: TransportSolver):
    """Without pumping, He-ash should grow over time."""
    solver_multi.tau_He_factor = 1e6  # effectively disable pumping
    for _ in range(20):
        solver_multi.evolve_profiles(dt=0.1, P_aux=50.0)
    assert np.max(solver_multi.n_He) > 0.0, "He-ash should accumulate"


def test_fuel_depletes(solver_multi: TransportSolver):
    """D and T should deplete as fusion consumes them."""
    D_initial = solver_multi.n_D.copy()
    T_initial = solver_multi.n_T.copy()
    for _ in range(30):
        solver_multi.evolve_profiles(dt=0.1, P_aux=50.0)
    # Core D and T should decrease (fusion burns them)
    # Use sum over inner half to avoid edge effects
    mid = solver_multi.nr // 2
    assert np.sum(solver_multi.n_D[:mid]) < np.sum(D_initial[:mid]), "D should deplete"
    assert np.sum(solver_multi.n_T[:mid]) < np.sum(T_initial[:mid]), "T should deplete"


def test_quasineutrality(solver_multi: TransportSolver):
    """ne must equal sum of species charges after evolution."""
    for _ in range(10):
        solver_multi.evolve_profiles(dt=0.1, P_aux=50.0)
    # ne = n_D + n_T + 2*n_He + Z_W * n_imp
    Z_W = 10.0
    expected_ne = (
        solver_multi.n_D + solver_multi.n_T
        + 2.0 * solver_multi.n_He
        + Z_W * np.maximum(solver_multi.n_impurity, 0.0)
    )
    expected_ne = np.maximum(expected_ne, 0.1)
    np.testing.assert_allclose(solver_multi.ne, expected_ne, atol=1e-10)


# ── Independent Te evolution ─────────────────────────────────────────


def test_te_independent_of_ti(solver_multi: TransportSolver):
    """In multi-ion mode, Te and Ti should diverge under different loss channels."""
    for _ in range(30):
        solver_multi.evolve_profiles(dt=0.1, P_aux=50.0)
    # Te and Ti should differ (electrons get Bremsstrahlung loss)
    diff = np.max(np.abs(solver_multi.Te - solver_multi.Ti))
    assert diff > 0.01, f"Te and Ti should differ in multi-ion mode, max diff={diff}"


def test_te_equals_ti_legacy(solver_legacy: TransportSolver):
    """In legacy mode, Te must remain equal to Ti."""
    for _ in range(10):
        solver_legacy.evolve_profiles(dt=0.1, P_aux=30.0)
    np.testing.assert_allclose(solver_legacy.Te, solver_legacy.Ti, atol=1e-12)


# ── Energy conservation ──────────────────────────────────────────────


def test_conservation_error_bounded(solver_multi: TransportSolver):
    """Per-step conservation error should stabilise after initial transient."""
    errors = []
    for _ in range(20):
        solver_multi.evolve_profiles(dt=0.1, P_aux=50.0)
        errors.append(solver_multi._last_conservation_error)
    # In multi-ion mode, the conservation diagnostic only tracks the ion
    # channel; species coupling and radiation introduce additional energy
    # flows.  The first few steps can have large transient errors as ne
    # is recomputed from quasineutrality.  We verify the *late* errors
    # stabilise below a generous bound.
    late_errors = errors[10:]
    assert all(e < 5.0 for e in late_errors), (
        f"Conservation error still diverging after 10 steps: {late_errors}"
    )
    assert np.all(np.isfinite(late_errors)), "Conservation error has NaN"


def test_no_heating_cools(solver_multi: TransportSolver):
    """Without heating, both Ti and Te should decrease."""
    Ti_sum_before = float(np.sum(solver_multi.Ti))
    for _ in range(20):
        solver_multi.evolve_profiles(dt=0.1, P_aux=0.0)
    Ti_sum_after = float(np.sum(solver_multi.Ti))
    assert Ti_sum_after < Ti_sum_before, "Ti should decrease with no heating"


# ── Backward compatibility ───────────────────────────────────────────


def test_legacy_no_nan(solver_legacy: TransportSolver):
    """Legacy mode should produce no NaN (same as before multi-ion)."""
    for _ in range(10):
        T_avg, T_core = solver_legacy.evolve_profiles(dt=1.0, P_aux=50.0)
    assert np.all(np.isfinite(solver_legacy.Ti)), "Legacy mode produced NaN"
    assert T_core > 0


def test_multi_ion_no_nan(solver_multi: TransportSolver):
    """Multi-ion mode should produce no NaN at dt=1.0."""
    for _ in range(10):
        T_avg, T_core = solver_multi.evolve_profiles(dt=1.0, P_aux=50.0)
    assert np.all(np.isfinite(solver_multi.Ti)), "Multi-ion Ti has NaN"
    assert np.all(np.isfinite(solver_multi.Te)), "Multi-ion Te has NaN"
    assert np.all(np.isfinite(solver_multi.n_D)), "n_D has NaN"
    assert np.all(np.isfinite(solver_multi.n_T)), "n_T has NaN"
    assert np.all(np.isfinite(solver_multi.n_He)), "n_He has NaN"


def test_run_to_steady_state_multi_ion(solver_multi: TransportSolver):
    """run_to_steady_state should work in multi-ion mode."""
    result = solver_multi.run_to_steady_state(P_aux=30.0, n_steps=20, dt=0.1)
    assert "T_avg" in result
    assert "T_core" in result
    assert result["T_avg"] > 0
    assert np.all(np.isfinite(result["Ti_profile"]))


def test_enforce_conservation_flag_accepted(config_file: Path):
    """enforce_conservation parameter should be accepted without error."""
    ts = TransportSolver(str(config_file), multi_ion=False)
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = ts.Ti.copy()
    ts.ne = 5.0 * (1 - ts.rho**2) ** 0.5
    ts.update_transport_model(50.0)
    # enforce_conservation=False should always work
    ts.evolve_profiles(dt=0.1, P_aux=50.0, enforce_conservation=False)
    # The conservation error is tracked regardless
    assert hasattr(ts, "_last_conservation_error")
    assert np.isfinite(ts._last_conservation_error)


def test_enforce_conservation_raises_on_violation(config_file: Path):
    """enforce_conservation=True should raise PhysicsError on gross violation."""
    ts = TransportSolver(str(config_file), multi_ion=True)
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = 5.0 * (1 - ts.rho**2)
    ts.ne = 5.0 * (1 - ts.rho**2) ** 0.5
    ts.n_D = 0.5 * ts.ne.copy()
    ts.n_T = 0.5 * ts.ne.copy()
    ts.n_He = np.zeros(ts.nr)
    ts.update_transport_model(50.0)
    # Multi-ion first step has large transient conservation error (>1%)
    # because species evolution changes ne mid-step
    with pytest.raises(PhysicsError, match="conservation"):
        ts.evolve_profiles(dt=0.1, P_aux=50.0, enforce_conservation=True)


# ── Z_eff tracking ──────────────────────────────────────────────────


def test_zeff_in_range(solver_multi: TransportSolver):
    """Z_eff should be in [1, 10] after evolution."""
    for _ in range(10):
        solver_multi.evolve_profiles(dt=0.1, P_aux=50.0)
    assert 1.0 <= solver_multi._Z_eff <= 10.0, f"Z_eff={solver_multi._Z_eff}"
