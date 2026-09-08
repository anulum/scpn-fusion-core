# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Public plant power accounting and configured cooling regressions."""

from __future__ import annotations

import pytest

from scpn_fusion.engineering.balance_of_plant import PowerPlantModel


def test_calculate_plant_performance_returns_consistent_metrics() -> None:
    """Check thermal, electrical and gain metrics for a positive load."""
    model = PowerPlantModel(coolant_type="water")
    metrics = model.calculate_plant_performance(P_fusion_MW=500.0, P_aux_absorbed_MW=50.0)

    expected_keys = {
        "P_fusion",
        "P_thermal",
        "P_gross",
        "P_recirc",
        "P_net",
        "Q_plasma",
        "Q_eng",
        "breakdown",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["P_fusion"] == pytest.approx(500.0)
    assert metrics["Q_plasma"] == pytest.approx(10.0)
    assert metrics["P_thermal"] > 0.0
    assert metrics["P_gross"] > 0.0
    assert metrics["P_recirc"] > 0.0
    assert isinstance(metrics["breakdown"], dict)


def test_calculate_plant_performance_handles_zero_aux_power() -> None:
    """Preserve the zero-auxiliary gain convention with fusion heating."""
    model = PowerPlantModel(coolant_type="helium")
    metrics = model.calculate_plant_performance(P_fusion_MW=300.0, P_aux_absorbed_MW=0.0)
    assert metrics["Q_plasma"] == 0
    assert metrics["P_thermal"] > 0.0


def test_plot_sankey_diagram_returns_figure() -> None:
    """Render actual plant results through the public plotting API."""
    model = PowerPlantModel(coolant_type="water")
    metrics = model.calculate_plant_performance(P_fusion_MW=400.0, P_aux_absorbed_MW=40.0)
    fig = model.plot_sankey_diagram(metrics)
    assert fig is not None
    assert len(fig.axes) == 1


def test_plant_cooling_configuration_reaches_real_hydraulics() -> None:
    """Configured paths change pump loads without changing conversion accounting."""
    model = PowerPlantModel()
    result = model.calculate_plant_performance(
        500.0,
        50.0,
        coolant_parallel_channels=100,
        coolant_length_m=20.0,
        coolant_diameter_m=0.1,
        coolant_temperature_rise_k=80.0,
    )
    channel = model.coolant.calculate_pumping_power(
        result["P_thermal"] / 100,
        delta_T=80.0,
        L=20.0,
        D=0.1,
    )
    assert result["hydraulics"]["parallel_channels"] == 100
    assert result["breakdown"]["Pumps"] == pytest.approx(100 * channel["P_pump_MW"])
    assert result["P_net"] == pytest.approx(result["P_gross"] - result["P_recirc"])
    assert result["P_recirc"] == pytest.approx(sum(result["breakdown"].values()))


def test_zero_power_plant_preserves_standby_loads() -> None:
    """No fusion heat means zero pumping, but fixed house loads still consume power."""
    result = PowerPlantModel().calculate_plant_performance(0.0, 0.0)
    assert result["breakdown"]["Pumps"] == 0
    assert result["P_net"] == -45.0
    assert result["Q_plasma"] == result["Q_eng"] == 0


@pytest.mark.parametrize(
    "fusion, auxiliary", [(-1.0, 0.0), (0.0, -1.0), (float("nan"), 1.0), (1.0, float("inf"))]
)
def test_invalid_plant_powers_are_rejected(fusion: float, auxiliary: float) -> None:
    """Negative or nonfinite loads must not enter thermal/electrical accounting."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        PowerPlantModel().calculate_plant_performance(fusion, auxiliary)


def test_zero_load_with_no_standby_does_not_divide_by_zero() -> None:
    """Explicitly absent house loads retain the documented zero gain convention."""
    model = PowerPlantModel()
    model.P_cryo = model.P_bop_misc = 0.0
    result = model.calculate_plant_performance(0.0, 0.0)
    assert result["P_recirc"] == result["P_net"] == result["Q_eng"] == 0.0


def test_unrepresentable_cooling_cannot_become_plant_output() -> None:
    """The plant propagates hydraulic domain refusal instead of an infinite net loss."""
    with pytest.raises(ValueError, match="representable domain"):
        PowerPlantModel().calculate_plant_performance(1.0, 0.0, coolant_diameter_m=1e-200)
