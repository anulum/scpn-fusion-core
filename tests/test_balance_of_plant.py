from __future__ import annotations

import pytest

from scpn_fusion.engineering.balance_of_plant import PowerPlantModel


def test_calculate_plant_performance_returns_consistent_metrics() -> None:
    model = PowerPlantModel(coolant_type="water")
    metrics = model.calculate_plant_performance(P_fusion_MW=500.0, P_aux_absorbed_MW=50.0)

    expected_keys = {"P_fusion", "P_thermal", "P_gross", "P_recirc", "P_net", "Q_plasma", "Q_eng", "breakdown"}
    assert expected_keys.issubset(metrics.keys())
    assert metrics["P_fusion"] == pytest.approx(500.0)
    assert metrics["Q_plasma"] == pytest.approx(10.0)
    assert metrics["P_thermal"] > 0.0
    assert metrics["P_gross"] > 0.0
    assert metrics["P_recirc"] > 0.0
    assert isinstance(metrics["breakdown"], dict)


def test_calculate_plant_performance_handles_zero_aux_power() -> None:
    model = PowerPlantModel(coolant_type="helium")
    metrics = model.calculate_plant_performance(P_fusion_MW=300.0, P_aux_absorbed_MW=0.0)
    assert metrics["Q_plasma"] == 0
    assert metrics["P_thermal"] > 0.0


def test_plot_sankey_diagram_returns_figure() -> None:
    model = PowerPlantModel(coolant_type="water")
    metrics = model.calculate_plant_performance(P_fusion_MW=400.0, P_aux_absorbed_MW=40.0)
    fig = model.plot_sankey_diagram(metrics)
    assert fig is not None
    assert len(fig.axes) == 1
