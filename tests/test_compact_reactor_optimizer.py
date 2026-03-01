from __future__ import annotations

import pytest

from scpn_fusion.core.compact_reactor_optimizer import CompactReactorArchitect


def test_plasma_physics_model_returns_positive_outputs() -> None:
    architect = CompactReactorArchitect()
    p_fus, ip, vol = architect.plasma_physics_model(R=1.5, a=0.5, B0=10.0)
    assert p_fus > 0.0
    assert ip > 0.0
    assert vol > 0.0


def test_radial_build_constraints_fail_for_impossible_post_radius() -> None:
    architect = CompactReactorArchitect()
    ok, b_coil = architect.radial_build_constraints(R=0.2, a=0.1, B0=10.0)
    assert ok is False
    assert b_coil == 0


def test_calculate_economics_returns_finite_values() -> None:
    architect = CompactReactorArchitect()
    design = {
        "R": 1.7,
        "a": 0.6,
        "B_coil": 20.0,
        "P_fus": 350.0,
        "Vol": 20.0,
    }
    coe, capex = architect.calculate_economics(design)
    assert coe > 0.0
    assert capex > 0.0
    assert coe == pytest.approx(float(coe))
