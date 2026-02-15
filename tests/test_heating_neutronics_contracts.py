# ----------------------------------------------------------------------
# SCPN Fusion Core -- Heating/Neutronics Contract Tests
# ----------------------------------------------------------------------
"""Tests for reusable heating/neutronics contracts."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.core.heating_neutronics_contracts import (
    quick_candidate,
    refine_candidate_tbr,
)
from scpn_fusion.nuclear.blanket_neutronics import BreedingBlanket


def test_quick_and_refine_candidate_contract_smoke() -> None:
    rng = np.random.default_rng(7)
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
            incident_flux=1e14,
        )
        .tbr
    )
    quick = quick_candidate(
        rng=rng,
        idx=0,
        base_tbr=base_tbr,
        explorer=explorer,
    )
    refined = refine_candidate_tbr(quick)
    assert float(refined["q_proxy"]) > 0.0
    assert float(refined["q_aries_at_proxy"]) > 0.0
    assert float(refined["tbr_final"]) > 0.0
    assert float(refined["tbr_mc"]) > 0.0
    assert 0.0 <= float(refined["neutron_leakage_rate"]) <= 1.0
