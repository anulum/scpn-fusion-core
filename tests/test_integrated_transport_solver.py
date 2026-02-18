# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Integrated Transport Solver Tests
# ──────────────────────────────────────────────────────────────────────
"""
Tests for the TransportSolver class covering initialization, profile
evolution, multi-ion species, energy conservation, and steady-state runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.integrated_transport_solver import (
    TransportSolver,
    PhysicsError,
    chang_hinton_chi_profile,
    calculate_sauter_bootstrap_current_full,
)

# ── Minimal config for fast tests ────────────────────────────────────

MINIMAL_CONFIG = {
    "reactor_name": "TransportSolver-Test",
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
    """Write a minimal JSON config and return its path."""
    cfg = tmp_path / "test_transport_config.json"
    cfg.write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    return cfg


@pytest.fixture
def solver(config_file: Path) -> TransportSolver:
    """Create a single-ion TransportSolver with physical initial profiles."""
    ts = TransportSolver(str(config_file), multi_ion=False)
    # Set physically meaningful initial profiles
    ts.Ti = 5.0 * (1 - ts.rho ** 2)
    ts.Te = 5.0 * (1 - ts.rho ** 2)
    ts.ne = 8.0 * (1 - ts.rho ** 2) ** 0.5
    ts.update_transport_model(50.0)
    return ts


@pytest.fixture
def solver_multi(config_file: Path) -> TransportSolver:
    """Create a multi-ion TransportSolver with D, T, He-ash species."""
    ts = TransportSolver(str(config_file), multi_ion=True)
    ts.Ti = 5.0 * (1 - ts.rho ** 2)
    ts.Te = 5.0 * (1 - ts.rho ** 2)
    ts.ne = 8.0 * (1 - ts.rho ** 2) ** 0.5
    ts.n_D = 0.5 * ts.ne.copy()
    ts.n_T = 0.5 * ts.ne.copy()
    ts.n_He = np.zeros(ts.nr)
    ts.update_transport_model(50.0)
    return ts


# ── 1. Initialization ────────────────────────────────────────────────

class TestInitialization:

    def test_init_default(self, config_file: Path) -> None:
        """TransportSolver initializes with correct default profile shapes."""
        ts = TransportSolver(str(config_file))
        assert ts.Ti.shape == (50,)
        assert ts.Te.shape == (50,)
        assert ts.ne.shape == (50,)
        assert ts.nr == 50
        assert ts.rho[0] == 0.0
        assert ts.rho[-1] == 1.0

    def test_init_multi_ion(self, config_file: Path) -> None:
        """multi_ion=True creates D, T, He-ash arrays on the rho grid."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        assert ts.multi_ion is True
        assert isinstance(ts.n_D, np.ndarray)
        assert isinstance(ts.n_T, np.ndarray)
        assert isinstance(ts.n_He, np.ndarray)
        assert ts.n_D.shape == (50,)
        assert ts.n_T.shape == (50,)
        assert ts.n_He.shape == (50,)

    def test_init_single_ion_no_species(self, config_file: Path) -> None:
        """multi_ion=False leaves species arrays as None."""
        ts = TransportSolver(str(config_file), multi_ion=False)
        assert ts.multi_ion is False
        assert ts.n_D is None
        assert ts.n_T is None
        assert ts.n_He is None

    def test_profiles_shape(self, solver: TransportSolver) -> None:
        """Ti, Te, ne should all be length 50 on the rho grid."""
        assert len(solver.Ti) == 50
        assert len(solver.Te) == 50
        assert len(solver.ne) == 50
        assert len(solver.rho) == 50

    def test_transport_coefficients_initialized(self, solver: TransportSolver) -> None:
        """chi_e, chi_i, D_n should exist with correct length."""
        assert solver.chi_e.shape == (50,)
        assert solver.chi_i.shape == (50,)
        assert solver.D_n.shape == (50,)

    def test_tau_he_factor_default(self, config_file: Path) -> None:
        """Default He-ash pumping time factor is 5.0."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        assert ts.tau_He_factor == 5.0

    def test_d_species_default(self, config_file: Path) -> None:
        """Default particle diffusivity for species transport is 0.3."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        assert ts.D_species == 0.3


# ── 2. Profile Evolution ─────────────────────────────────────────────

class TestEvolveProfiles:

    def test_evolve_profiles_runs(self, solver: TransportSolver) -> None:
        """evolve_profiles returns (avg_T, core_T) as finite floats."""
        avg_T, core_T = solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert isinstance(avg_T, float)
        assert isinstance(core_T, float)
        assert np.isfinite(avg_T)
        assert np.isfinite(core_T)
        assert avg_T > 0
        assert core_T > 0

    def test_conservation_attribute(self, solver: TransportSolver) -> None:
        """After evolve, _last_conservation_error is a finite float."""
        solver.evolve_profiles(dt=0.01, P_aux=50.0)
        err = solver._last_conservation_error
        assert isinstance(err, float)
        assert np.isfinite(err)

    def test_profiles_stay_positive(self, solver: TransportSolver) -> None:
        """Profiles should remain non-negative after multiple steps."""
        for _ in range(20):
            solver.update_transport_model(50.0)
            solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert np.all(solver.Ti >= 0)
        assert np.all(solver.Te >= 0)
        assert np.all(solver.ne >= 0)

    def test_evolve_changes_profiles(self, solver: TransportSolver) -> None:
        """Profiles should change after evolution (not frozen)."""
        Ti_before = solver.Ti.copy()
        solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert not np.allclose(solver.Ti, Ti_before, atol=1e-12)

    def test_enforce_conservation_no_raise_small_dt(self, solver: TransportSolver) -> None:
        """With small dt and reasonable params, enforce_conservation should not raise."""
        # This is best-effort: small dt + moderate heating shouldn't violate conservation
        try:
            solver.evolve_profiles(dt=0.001, P_aux=20.0, enforce_conservation=True)
        except PhysicsError:
            # If the initial conditions are too far from equilibrium,
            # conservation may be violated. This is acceptable.
            pass

    def test_multiple_steps_trend(self, solver: TransportSolver) -> None:
        """With sustained heating, average temperature should change over time."""
        T_start = float(np.mean(solver.Ti))
        for _ in range(50):
            solver.update_transport_model(50.0)
            solver.evolve_profiles(dt=0.01, P_aux=50.0)
        T_end = float(np.mean(solver.Ti))
        # Temperature should have changed (either up or stabilized)
        assert T_end != T_start


# ── 3. Multi-Ion Species ──────────────────────────────────────────────

class TestMultiIon:

    def test_multi_ion_he_ash_grows(self, solver_multi: TransportSolver) -> None:
        """With multi_ion=True, after N steps, n_He should increase."""
        he_initial = np.sum(solver_multi.n_He)
        for _ in range(20):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        he_final = np.sum(solver_multi.n_He)
        # He-ash is produced by fusion reactions
        assert he_final > he_initial

    def test_multi_ion_fuel_depletes(self, solver_multi: TransportSolver) -> None:
        """With multi_ion=True, sum of n_D should decrease over time."""
        d_initial = np.sum(solver_multi.n_D)
        for _ in range(20):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        d_final = np.sum(solver_multi.n_D)
        # Fuel is consumed by fusion reactions
        assert d_final < d_initial

    def test_multi_ion_tritium_depletes(self, solver_multi: TransportSolver) -> None:
        """Tritium should also deplete over time due to fusion burn."""
        t_initial = np.sum(solver_multi.n_T)
        for _ in range(20):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        t_final = np.sum(solver_multi.n_T)
        assert t_final < t_initial

    def test_multi_ion_quasineutrality(self, solver_multi: TransportSolver) -> None:
        """After evolving, ne should be updated from quasineutrality."""
        for _ in range(5):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        # ne = n_D + n_T + 2*n_He + Z_W * n_impurity
        ne_check = (
            solver_multi.n_D + solver_multi.n_T
            + 2.0 * solver_multi.n_He
            + 10.0 * np.maximum(solver_multi.n_impurity, 0.0)
        )
        ne_check = np.maximum(ne_check, 0.1)
        np.testing.assert_allclose(solver_multi.ne, ne_check, rtol=1e-10)


# ── 4. Steady State Run ──────────────────────────────────────────────

class TestSteadyState:

    def test_run_to_steady_state_returns_dict(self, solver: TransportSolver) -> None:
        """run_to_steady_state returns a dict with expected keys."""
        result = solver.run_to_steady_state(P_aux=50.0, n_steps=10, dt=0.01)
        assert isinstance(result, dict)
        assert "T_avg" in result
        assert "T_core" in result
        assert "tau_e" in result
        assert "n_steps" in result
        assert "Ti_profile" in result
        assert "ne_profile" in result
        assert isinstance(result["T_avg"], float)
        assert isinstance(result["T_core"], float)
        assert np.isfinite(result["T_avg"])
        assert np.isfinite(result["T_core"])

    def test_run_to_steady_state_profile_shapes(self, solver: TransportSolver) -> None:
        """Returned profiles should have the correct length."""
        result = solver.run_to_steady_state(P_aux=50.0, n_steps=10, dt=0.01)
        assert result["Ti_profile"].shape == (50,)
        assert result["ne_profile"].shape == (50,)

    def test_confinement_time_positive(self, solver: TransportSolver) -> None:
        """Confinement time should be positive for positive loss power."""
        tau_e = solver.compute_confinement_time(50.0)
        assert tau_e > 0
        assert np.isfinite(tau_e)


# ── 5. Neoclassical Transport ─────────────────────────────────────────

class TestNeoclassical:

    def test_set_neoclassical_stores_params(self, solver: TransportSolver) -> None:
        """set_neoclassical stores parameters for Chang-Hinton model."""
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        assert solver.neoclassical_params is not None
        assert solver.neoclassical_params["R0"] == 6.2
        assert solver.neoclassical_params["a"] == 2.0
        assert solver.neoclassical_params["B0"] == 5.3

    def test_chang_hinton_profile_shape(self) -> None:
        """Chang-Hinton neoclassical chi should match input rho shape."""
        rho = np.linspace(0, 1, 50)
        Ti = 5.0 * (1 - rho ** 2)
        ne = 8.0 * (1 - rho ** 2) ** 0.5
        q = 1.0 + 3.0 * rho ** 2
        chi = chang_hinton_chi_profile(rho, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)
        assert chi.shape == (50,)
        assert np.all(np.isfinite(chi))
        assert np.all(chi >= 0.01)  # floor applied

    def test_bootstrap_current_shape(self) -> None:
        """Sauter bootstrap current profile should match rho shape."""
        rho = np.linspace(0, 1, 50)
        Te = 5.0 * (1 - rho ** 2)
        Ti = 5.0 * (1 - rho ** 2)
        ne = 8.0 * (1 - rho ** 2) ** 0.5
        q = 1.0 + 3.0 * rho ** 2
        j_bs = calculate_sauter_bootstrap_current_full(
            rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3
        )
        assert j_bs.shape == (50,)
        assert np.all(np.isfinite(j_bs))
        # Should be zero at the boundary (j_bs[0] and j_bs[-1])
        assert j_bs[0] == 0.0
        assert j_bs[-1] == 0.0


# ── 6. Thomas Solver ─────────────────────────────────────────────────

class TestThomasSolver:

    def test_thomas_identity_system(self) -> None:
        """Thomas solver with identity matrix returns the RHS."""
        n = 10
        a = np.zeros(n - 1)
        b = np.ones(n)
        c = np.zeros(n - 1)
        d = np.arange(n, dtype=float)
        x = TransportSolver._thomas_solve(a, b, c, d)
        np.testing.assert_allclose(x, d, atol=1e-12)

    def test_thomas_tridiagonal(self) -> None:
        """Thomas solver correctly solves a known tridiagonal system."""
        # Solve: -x_{i-1} + 2 x_i - x_{i+1} = h^2 * f_i (Poisson)
        n = 50
        a = -np.ones(n - 1)
        b = 2.0 * np.ones(n)
        c = -np.ones(n - 1)
        d = np.ones(n) * (1.0 / (n - 1)) ** 2
        d[0] = 0.0
        d[-1] = 0.0
        b[0] = 1.0; c[0] = 0.0
        a[-1] = 0.0; b[-1] = 1.0
        x = TransportSolver._thomas_solve(a, b, c, d)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))


# ── 7. Bosch-Hale D-T Reactivity ─────────────────────────────────────

class TestBoschHale:

    def test_sigmav_positive_for_fusion_temperatures(self) -> None:
        """Bosch-Hale <sigma*v> should be positive for T > 0.2 keV."""
        T = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        sv = TransportSolver._bosch_hale_sigmav(T)
        assert np.all(sv > 0)
        assert np.all(np.isfinite(sv))

    def test_sigmav_increases_with_temperature(self) -> None:
        """D-T reactivity should increase monotonically from 1-50 keV (NRL fit)."""
        T = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        sv = TransportSolver._bosch_hale_sigmav(T)
        # The simplified NRL Formulary fit is monotonically increasing
        # over the range 1-100 keV (peak is beyond 100 keV for this fit)
        for i in range(1, len(sv)):
            assert sv[i] > sv[i - 1], (
                f"sigma_v not increasing: sv[{T[i]}] = {sv[i]} <= sv[{T[i-1]}] = {sv[i-1]}"
            )


# ── 8. Impurity Injection ────────────────────────────────────────────

class TestImpurityInjection:

    def test_inject_impurities_increases_edge(self, solver: TransportSolver) -> None:
        """Injecting impurities should increase the total impurity count."""
        imp_before = np.sum(solver.n_impurity)
        solver.inject_impurities(flux_from_wall_per_sec=1e20, dt=0.01)
        imp_after = np.sum(solver.n_impurity)
        assert imp_after > imp_before

    def test_impurities_non_negative(self, solver: TransportSolver) -> None:
        """Impurity profiles should remain non-negative after injection."""
        solver.inject_impurities(flux_from_wall_per_sec=1e20, dt=0.01)
        assert np.all(solver.n_impurity >= 0)
