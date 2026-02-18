# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Whole Device Model (WDM) Engine Tests
# ──────────────────────────────────────────────────────────────────────
"""
Tests for the WholeDeviceModel class which couples TransportSolver
with SputteringPhysics for multi-physics discharge simulation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Guard imports — WDM depends on TransportSolver + SputteringPhysics + pandas
try:
    from scpn_fusion.core.wdm_engine import WholeDeviceModel
    from scpn_fusion.core.integrated_transport_solver import TransportSolver
    from scpn_fusion.nuclear.pwi_erosion import SputteringPhysics

    _WDM_AVAILABLE = True
except ImportError as _exc:
    _WDM_AVAILABLE = False
    _import_reason = str(_exc)

pytestmark = pytest.mark.skipif(
    not _WDM_AVAILABLE,
    reason=f"WDM engine not importable: {_import_reason if not _WDM_AVAILABLE else ''}",
)

# ── Minimal test configuration ───────────────────────────────────────

MINIMAL_CONFIG = {
    "reactor_name": "WDM-Test",
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
    cfg = tmp_path / "wdm_test_config.json"
    cfg.write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    return cfg


@pytest.fixture
def wdm(config_file: Path) -> WholeDeviceModel:
    """Create a WholeDeviceModel with the minimal test config."""
    return WholeDeviceModel(str(config_file))


# ── 1. Initialization ────────────────────────────────────────────────

class TestWDMInit:

    def test_init(self, wdm: WholeDeviceModel) -> None:
        """WholeDeviceModel initializes without error and has transport attribute."""
        assert hasattr(wdm, "transport")
        assert hasattr(wdm, "pwi")

    def test_transport_attribute(self, wdm: WholeDeviceModel) -> None:
        """model.transport should be a TransportSolver instance."""
        assert isinstance(wdm.transport, TransportSolver)

    def test_pwi_attribute(self, wdm: WholeDeviceModel) -> None:
        """model.pwi should be a SputteringPhysics instance."""
        assert isinstance(wdm.pwi, SputteringPhysics)

    def test_transport_profiles_initialized(self, wdm: WholeDeviceModel) -> None:
        """The transport solver should have initialized profiles."""
        assert wdm.transport.Ti is not None
        assert wdm.transport.Te is not None
        assert wdm.transport.ne is not None
        assert len(wdm.transport.Ti) == 50

    def test_pwi_material(self, wdm: WholeDeviceModel) -> None:
        """PWI physics should be initialized with Tungsten."""
        assert wdm.pwi.material == "Tungsten"

    def test_equilibrium_solved_at_init(self, wdm: WholeDeviceModel) -> None:
        """The GS equilibrium should have been solved during __init__."""
        # Psi should be non-trivially initialized after solve_equilibrium
        assert wdm.transport.Psi is not None
        assert wdm.transport.Psi.shape[0] > 0
        assert wdm.transport.Psi.shape[1] > 0


# ── 2. Short Discharge Runs ──────────────────────────────────────────

class TestDischargeRun:

    def test_run_short_discharge(self, wdm: WholeDeviceModel) -> None:
        """Run a 0.1 second discharge; verify it completes without error.

        We mock plot_results to avoid matplotlib side effects.
        """
        with patch.object(wdm, "plot_results"):
            wdm.run_discharge(duration_sec=0.1)

    def test_run_very_short_discharge(self, wdm: WholeDeviceModel) -> None:
        """Run a minimal 0.02 second discharge (2 steps)."""
        with patch.object(wdm, "plot_results"):
            wdm.run_discharge(duration_sec=0.02)

    def test_profiles_evolve_during_discharge(self, wdm: WholeDeviceModel) -> None:
        """Ti profile should change during a short discharge."""
        Ti_before = wdm.transport.Ti.copy()
        with patch.object(wdm, "plot_results"):
            wdm.run_discharge(duration_sec=0.05)
        Ti_after = wdm.transport.Ti
        # Profiles should have evolved
        assert not np.allclose(Ti_before, Ti_after, atol=1e-12)

    def test_impurities_accumulate_during_discharge(self, wdm: WholeDeviceModel) -> None:
        """Impurity content should increase during a discharge."""
        imp_before = np.sum(wdm.transport.n_impurity)
        with patch.object(wdm, "plot_results"):
            wdm.run_discharge(duration_sec=0.05)
        imp_after = np.sum(wdm.transport.n_impurity)
        # Edge erosion injects impurities
        assert imp_after >= imp_before


# ── 3. Component Integration ─────────────────────────────────────────

class TestComponentIntegration:

    def test_transport_ne_is_array(self, wdm: WholeDeviceModel) -> None:
        """Transport solver density profile is a numpy array."""
        assert isinstance(wdm.transport.ne, np.ndarray)

    def test_transport_solver_can_evolve(self, wdm: WholeDeviceModel) -> None:
        """Transport solver works independently within WDM."""
        avg_T, core_T = wdm.transport.evolve_profiles(dt=0.01, P_aux=50.0)
        assert np.isfinite(avg_T)
        assert np.isfinite(core_T)

    def test_pwi_can_calculate_yield(self, wdm: WholeDeviceModel) -> None:
        """PWI sputtering physics works independently within WDM."""
        Y = wdm.pwi.calculate_yield(E_ion_eV=500.0, angle_deg=45.0)
        assert isinstance(Y, float)
        assert Y >= 0.0

    def test_pwi_can_calculate_erosion(self, wdm: WholeDeviceModel) -> None:
        """PWI erosion rate calculation works within WDM."""
        result = wdm.pwi.calculate_erosion_rate(
            flux_particles_m2_s=1e24, T_ion_eV=100.0
        )
        assert isinstance(result, dict)
        assert "Impurity_Source" in result
        assert "Erosion_mm_year" in result


# ── 4. Config Path Handling ──────────────────────────────────────────

class TestConfigHandling:

    def test_accepts_string_path(self, config_file: Path) -> None:
        """WholeDeviceModel accepts a string config path."""
        wdm = WholeDeviceModel(str(config_file))
        assert wdm.transport is not None

    def test_uses_iter_config_if_available(self) -> None:
        """If the shipped iter_config.json exists, WDM can use it."""
        iter_config = (
            Path(__file__).resolve().parents[1] / "iter_config.json"
        )
        if not iter_config.exists():
            pytest.skip("iter_config.json not found in repo root")
        wdm = WholeDeviceModel(str(iter_config))
        assert isinstance(wdm.transport, TransportSolver)
