# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Contract Tests
# ----------------------------------------------------------------------
"""Tests for reusable disruption contracts."""

from __future__ import annotations

import numpy as np

from scpn_fusion.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_fusion.control.disruption_contracts import run_disruption_episode
from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.nuclear.blanket_neutronics import BreedingBlanket


def test_run_disruption_episode_contract_smoke() -> None:
    rng = np.random.default_rng(42)
    agent = FusionAIAgent(epsilon=0.05)
    explorer = GlobalDesignExplorer("dummy")
    base_tbr = float(
        BreedingBlanket(thickness_cm=260.0, li6_enrichment=1.0)
        .calculate_volumetric_tbr(
            major_radius_m=6.2,
            minor_radius_m=2.0,
            elongation=1.7,
            radial_cells=8,
            poloidal_cells=16,
            toroidal_cells=12,
        )
        .tbr
    )
    out = run_disruption_episode(
        rng=rng,
        rl_agent=agent,
        base_tbr=base_tbr,
        explorer=explorer,
    )
    assert 0.0 <= float(out["risk_before"]) <= 1.0
    assert 0.0 <= float(out["risk_after"]) <= 1.0
    assert float(out["q_proxy"]) > 0.0
    assert float(out["tbr_proxy"]) > 0.0
    assert float(out["halo_current_ma"]) >= 0.0
    assert float(out["runaway_beam_ma"]) >= 0.0
