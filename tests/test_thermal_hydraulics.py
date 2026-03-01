from __future__ import annotations

import pytest

from scpn_fusion.engineering.thermal_hydraulics import CoolantLoop, churchill_friction_factor


def test_churchill_friction_factor_is_positive() -> None:
    val = churchill_friction_factor(1.0e5)
    assert val > 0.0


def test_churchill_friction_factor_rejects_nonpositive_re() -> None:
    with pytest.raises(ValueError):
        churchill_friction_factor(0.0)


def test_calculate_pumping_power_returns_expected_keys() -> None:
    loop = CoolantLoop("water")
    result = loop.calculate_pumping_power(Q_thermal_MW=250.0)
    for key in ("mdot_kg_s", "velocity_m_s", "Re", "dP_Pa", "P_pump_MW"):
        assert key in result
    assert result["mdot_kg_s"] > 0.0
    assert result["P_pump_MW"] >= 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"Q_thermal_MW": -1.0}, "Q_thermal_MW"),
        ({"Q_thermal_MW": 100.0, "delta_T": 0.0}, "delta_T"),
        ({"Q_thermal_MW": 100.0, "L": 0.0}, "L"),
        ({"Q_thermal_MW": 100.0, "D": 0.0}, "D"),
    ],
)
def test_calculate_pumping_power_rejects_invalid_inputs(kwargs: dict, match: str) -> None:
    loop = CoolantLoop("water")
    with pytest.raises(ValueError, match=match):
        loop.calculate_pumping_power(**kwargs)
