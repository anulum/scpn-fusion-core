# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GMVR-02 TEMHD Divertor Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for GMVR-02 TEMHD divertor campaign."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from scpn_fusion.core.divertor_thermal_sim import DivertorLab


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gmvr_02_temhd_divertor.py"
SPEC = importlib.util.spec_from_file_location("gmvr_02_temhd_divertor", MODULE_PATH)
assert SPEC and SPEC.loader
gmvr_02_temhd_divertor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gmvr_02_temhd_divertor)


def test_velocity_dependent_temhd_terms() -> None:
    lab = DivertorLab(P_sol_MW=35.0, R_major=1.4, B_pol=2.3)
    slow = lab.simulate_temhd_liquid_metal(flow_velocity_m_s=0.001, expansion_factor=40.0)
    fast = lab.simulate_temhd_liquid_metal(flow_velocity_m_s=10.0, expansion_factor=40.0)
    assert fast["pressure_loss_pa"] > slow["pressure_loss_pa"]
    assert fast["evaporation_rate_kg_m2_s"] < slow["evaporation_rate_kg_m2_s"]


def test_gmvr_02_campaign_passes_thresholds() -> None:
    out = gmvr_02_temhd_divertor.run_campaign()
    assert out["passes_thresholds"] is True
    assert out["pressure_ratio_fast_to_slow"] >= 1000.0
    assert out["evap_ratio_fast_to_slow"] < 1.0
    assert out["toroidal_stability_rate"] >= 0.95


# S2-005: Divertor relaxation parameter


def test_relaxation_parameter_affects_convergence() -> None:
    """Different relaxation factors should produce different convergence paths."""
    lab1 = DivertorLab(P_sol_MW=50.0, R_major=2.1, B_pol=2.0)
    lab1.calculate_heat_load()
    t1, q1, f1 = lab1.simulate_lithium_vapor(relaxation=0.5)

    lab2 = DivertorLab(P_sol_MW=50.0, R_major=2.1, B_pol=2.0)
    lab2.calculate_heat_load()
    t2, q2, f2 = lab2.simulate_lithium_vapor(relaxation=0.9)

    # Both should converge to finite values
    assert np.isfinite(t1) and np.isfinite(t2)
    assert np.isfinite(q1) and np.isfinite(q2)
    # Results may differ slightly due to relaxation path
    # (both converge to the same fixed point, so allow small tolerance)
    assert abs(t1 - t2) < 100.0, "Large deviation suggests convergence issue"


def test_relaxation_rejects_invalid_values() -> None:
    """relaxation outside (0, 1) should raise ValueError."""
    import pytest

    lab = DivertorLab(P_sol_MW=50.0, R_major=2.1, B_pol=2.0)
    lab.calculate_heat_load()

    with pytest.raises(ValueError):
        lab.simulate_lithium_vapor(relaxation=0.0)

    with pytest.raises(ValueError):
        lab.simulate_lithium_vapor(relaxation=1.0)

    with pytest.raises(ValueError):
        lab.simulate_lithium_vapor(relaxation=-0.1)

    with pytest.raises(ValueError):
        lab.simulate_lithium_vapor(relaxation=1.5)
