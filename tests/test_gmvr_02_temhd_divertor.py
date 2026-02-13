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
