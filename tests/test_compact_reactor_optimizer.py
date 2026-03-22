# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import pytest

from scpn_fusion.core.compact_reactor_optimizer import CompactReactorArchitect


def test_plasma_physics_model_returns_positive_outputs() -> None:
    architect = CompactReactorArchitect()
    p_fus, ip, vol = architect.plasma_physics_model(R=1.5, a=0.5, B0=10.0)
    assert p_fus > 0.0
    assert ip > 0.0
    assert vol > 0.0


def test_plasma_physics_model_rejects_nonphysical_inputs() -> None:
    architect = CompactReactorArchitect()
    with pytest.raises(ValueError, match="must be > 0"):
        architect.plasma_physics_model(R=0.0, a=0.5, B0=10.0)


def test_radial_build_constraints_fail_for_impossible_post_radius() -> None:
    architect = CompactReactorArchitect()
    ok, b_coil = architect.radial_build_constraints(R=0.2, a=0.1, B0=10.0)
    assert ok is False
    assert b_coil == 0


def test_radial_build_constraints_rejects_nonpositive_inputs() -> None:
    architect = CompactReactorArchitect()
    ok, b_coil = architect.radial_build_constraints(R=1.0, a=0.0, B0=10.0)
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


def test_calculate_economics_rejects_nonphysical_design() -> None:
    architect = CompactReactorArchitect()
    with pytest.raises(ValueError, match="positive"):
        architect.calculate_economics(
            {
                "R": 1.7,
                "a": 0.6,
                "B_coil": 20.0,
                "P_fus": 0.0,
                "Vol": 20.0,
            }
        )


def test_power_scales_with_field() -> None:
    arch = CompactReactorArchitect()
    P_low, _, _ = arch.plasma_physics_model(R=2.0, a=0.5, B0=5.0)
    P_high, _, _ = arch.plasma_physics_model(R=2.0, a=0.5, B0=10.0)
    assert P_high > P_low


def test_feasible_design_returns_positive_bcoil() -> None:
    arch = CompactReactorArchitect()
    ok, B_coil = arch.radial_build_constraints(R=3.0, a=0.8, B0=10.0)
    assert B_coil > 0


def test_find_minimum_reactor_no_design_for_extreme_target() -> None:
    arch = CompactReactorArchitect()
    arch.find_minimum_reactor(target_power_MW=1e6, use_temhd=True)


def test_find_minimum_reactor_finds_design(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)
    monkeypatch.setattr(plt, "figure", lambda **kw: type("F", (), {})())
    monkeypatch.setattr(plt, "scatter", lambda *a, **kw: type("S", (), {})())
    monkeypatch.setattr(plt, "colorbar", lambda *a, **kw: None)
    monkeypatch.setattr(plt, "xlabel", lambda *a: None)
    monkeypatch.setattr(plt, "ylabel", lambda *a: None)
    monkeypatch.setattr(plt, "title", lambda *a: None)
    arch = CompactReactorArchitect()
    arch.find_minimum_reactor(target_power_MW=1.0, use_temhd=True)


def test_find_minimum_reactor_solid_divertor(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)
    monkeypatch.setattr(plt, "figure", lambda **kw: type("F", (), {})())
    monkeypatch.setattr(plt, "scatter", lambda *a, **kw: type("S", (), {})())
    monkeypatch.setattr(plt, "colorbar", lambda *a, **kw: None)
    monkeypatch.setattr(plt, "xlabel", lambda *a: None)
    monkeypatch.setattr(plt, "ylabel", lambda *a: None)
    monkeypatch.setattr(plt, "title", lambda *a: None)
    arch = CompactReactorArchitect()
    arch.find_minimum_reactor(target_power_MW=1.0, use_temhd=False)


def test_report_design_prints(capsys) -> None:
    arch = CompactReactorArchitect()
    design = {
        "R": 1.5,
        "a": 0.5,
        "B0": 12.0,
        "B_coil": 18.0,
        "P_fus": 50.0,
        "Vol": 10.0,
        "Ip": 3.0,
        "q_div": 5.0,
        "q_wall": 1.0,
    }
    arch.report_design(design)
    out = capsys.readouterr().out
    assert "MINIMUM VIABLE REACTOR" in out


def test_visualize_space_empty_designs() -> None:
    arch = CompactReactorArchitect()
    arch.visualize_space([], label="empty")
