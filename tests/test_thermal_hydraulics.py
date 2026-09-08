# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Public coolant flow, pumping and input-domain regressions."""

from __future__ import annotations

from typing import Any

import pytest

from scpn_fusion.engineering.thermal_hydraulics import CoolantLoop, churchill_friction_factor


def test_churchill_friction_factor_is_positive() -> None:
    """Check a turbulent reference regime has positive friction."""
    val = churchill_friction_factor(1.0e5)
    assert val > 0.0


def test_churchill_friction_factor_rejects_nonpositive_re() -> None:
    """Reject the undefined zero-Reynolds friction request."""
    with pytest.raises(ValueError):
        churchill_friction_factor(0.0)


def test_calculate_pumping_power_returns_expected_keys() -> None:
    """Check positive-load public flow and pumping diagnostics."""
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
def test_calculate_pumping_power_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    """Reject negative heat and nonpositive channel geometry."""
    loop = CoolantLoop("water")
    with pytest.raises(ValueError, match=match):
        loop.calculate_pumping_power(**kwargs)


def test_churchill_friction_factor_low_reynolds_limit() -> None:
    """Retain the existing documented low-Reynolds correlation floor."""
    # Re below the 1e-3 floor returns the clamped laminar value 64 / 1e-3.
    assert churchill_friction_factor(Re=5e-4) == pytest.approx(64.0 / 1e-3)


@pytest.mark.parametrize("coolant", ["water", "helium", "lipb"])
def test_parallel_channels_conserve_total_flow_and_electrical_power(coolant: str) -> None:
    """An equal-flow bank matches the sum of independently evaluated channels."""
    loop = CoolantLoop(coolant)
    bank = loop.calculate_pumping_power(500.0, parallel_channels=100)
    channel = loop.calculate_pumping_power(5.0)
    assert bank["mdot_kg_s"] == pytest.approx(100 * channel["mdot_kg_s"])
    assert bank["channel_mass_flow_kg_s"] == pytest.approx(channel["mdot_kg_s"])
    for key in ("velocity_m_s", "Re", "dP_Pa"):
        assert bank[key] == pytest.approx(channel[key])
    assert bank["P_pump_MW"] == pytest.approx(100 * channel["P_pump_MW"])
    assert bank["parallel_channels"] == 100


def test_zero_load_has_zero_flow_and_power() -> None:
    """A configured idle coolant bank needs no heat-driven mass flow."""
    result = CoolantLoop().calculate_pumping_power(0.0, parallel_channels=12)
    assert result["parallel_channels"] == 12
    assert all(value == 0 for key, value in result.items() if key != "parallel_channels")


@pytest.mark.parametrize("channels", [0, -1, 1.5, True])
def test_invalid_parallel_channel_counts_are_rejected(channels: int) -> None:
    """Reject unusable channel counts even when the thermal load is zero."""
    with pytest.raises(ValueError, match="parallel_channels"):
        CoolantLoop().calculate_pumping_power(0.0, parallel_channels=channels)


@pytest.mark.parametrize("name", ["Q_thermal_MW", "delta_T", "L", "D"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_cooling_inputs_are_rejected(name: str, value: float) -> None:
    """Nonfinite design values cannot produce plausible pumping metrics."""
    inputs: dict[str, Any] = {"Q_thermal_MW": 10.0, name: value}
    with pytest.raises(ValueError, match=name):
        CoolantLoop().calculate_pumping_power(**inputs)


def test_unknown_coolant_is_not_silently_replaced_with_water() -> None:
    """Misspelled material identities must fail instead of changing the design."""
    with pytest.raises(ValueError, match="Unknown coolant"):
        CoolantLoop("heliumm")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Q_thermal_MW": 1e303},
        {"Q_thermal_MW": 1.0, "D": 1e-200},
        {"Q_thermal_MW": 1e120},
        {"Q_thermal_MW": 1e-300, "L": 1e-300},
    ],
)
def test_unrepresentable_cooling_results_are_rejected(kwargs: dict[str, float]) -> None:
    """Finite inputs must never return infinite/underflowed pumping diagnostics."""
    with pytest.raises(ValueError):
        CoolantLoop().calculate_pumping_power(
            kwargs["Q_thermal_MW"], L=kwargs.get("L", 100.0), D=kwargs.get("D", 0.05)
        )
