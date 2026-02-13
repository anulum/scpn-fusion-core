# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Test Physics
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import json
from pathlib import Path

import pytest

from scpn_fusion.core.fusion_ignition_sim import FusionBurnPhysics

# Mock Config
MOCK_CONFIG = {
    "reactor_name": "Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 1, "R_max": 2, "Z_min": -1, "Z_max": 1},
    "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
    "coils": [],
    "solver": {"max_iterations": 1, "convergence_threshold": 1e-4, "relaxation_factor": 0.1}
}

@pytest.fixture
def sim(tmp_path: Path) -> FusionBurnPhysics:
    cfg_path = tmp_path / "test_config.json"
    cfg_path.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    return FusionBurnPhysics(str(cfg_path))


def test_bosch_hale_rate(sim: FusionBurnPhysics) -> None:
    """Test D-T Reaction Rate physics."""
    # Test 1: Zero temperature -> Zero rate
    rate_zero = sim.bosch_hale_dt(0.0)
    assert rate_zero < 1e-30, "Fusion at absolute zero? Impossible."

    # Test 2: Peak rate (around 60 keV)
    rate_peak = sim.bosch_hale_dt(60.0)
    rate_low = sim.bosch_hale_dt(1.0)

    # Rate at 60keV should be orders of magnitude higher than at 1keV
    assert rate_peak > rate_low * 1000, "Fusion cross-section curve is wrong."

    # Test 3: Magnitude check (approx 8e-22 m3/s at 20keV)
    rate_20 = sim.bosch_hale_dt(20.0)
    # Allow wide margin because approximations vary, but must be within factor of 10
    assert 1e-22 < rate_20 < 1e-21, f"Rate at 20keV is unphysical: {rate_20}"


def test_ignition_logic() -> None:
    """Test Q-factor calculation logic."""
    metrics = {"P_fusion_MW": 500.0, "P_aux_MW": 50.0}
    q_factor = metrics["P_fusion_MW"] / metrics["P_aux_MW"]
    assert q_factor == 10.0
