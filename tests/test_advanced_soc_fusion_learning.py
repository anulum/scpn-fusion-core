# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Advanced SOC Fusion Learning Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for advanced_soc_fusion_learning runtime paths."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.advanced_soc_fusion_learning import (
    CoupledSandpileReactor,
    FusionAIAgent,
    run_advanced_learning_sim,
)


def test_run_advanced_learning_sim_returns_finite_summary_without_plot() -> None:
    summary = run_advanced_learning_sim(
        size=24,
        time_steps=160,
        seed=123,
        epsilon=0.08,
        noise_probability=0.0,
        save_plot=False,
        verbose=False,
    )
    for key in (
        "seed",
        "steps",
        "final_core_temp",
        "final_flow",
        "final_external_shear",
        "mean_turbulence",
        "mean_flow",
        "mean_core_temp",
        "max_external_shear",
        "mean_total_shear",
        "total_reward",
        "q_table_mean",
        "q_table_max_abs",
        "plot_saved",
    ):
        assert key in summary
    assert summary["seed"] == 123
    assert summary["steps"] == 160
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert 0.0 <= summary["final_external_shear"] <= 1.0
    assert 0.0 <= summary["max_external_shear"] <= 1.0
    assert np.isfinite(summary["mean_core_temp"])
    assert np.isfinite(summary["total_reward"])
    assert np.isfinite(summary["q_table_max_abs"])


def test_run_advanced_learning_sim_is_deterministic_for_fixed_seed() -> None:
    kwargs = dict(
        size=20,
        time_steps=120,
        seed=77,
        epsilon=0.0,
        noise_probability=0.02,
        save_plot=False,
        verbose=False,
    )
    a = run_advanced_learning_sim(**kwargs)
    b = run_advanced_learning_sim(**kwargs)
    for key in (
        "final_core_temp",
        "final_flow",
        "final_external_shear",
        "mean_turbulence",
        "mean_flow",
        "mean_core_temp",
        "mean_total_shear",
        "total_reward",
        "q_table_mean",
        "q_table_max_abs",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_agent_learn_updates_q_table_and_reward() -> None:
    agent = FusionAIAgent(alpha=0.2, gamma=0.9, epsilon=0.0)
    before = float(agent.q_table[0, 0, 1])
    updated = agent.learn((0, 0), 1, (1, 1), reward=2.5)
    after = float(agent.q_table[0, 0, 1])
    assert after != before
    assert updated == after
    assert agent.total_reward == pytest.approx(2.5, rel=0.0, abs=0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"time_steps": 0}, "time_steps"),
        ({"shear_step": -0.1}, "shear_step"),
        ({"shear_step": float("nan")}, "shear_step"),
        ({"noise_probability": -0.1}, "noise_probability"),
        ({"noise_probability": 1.1}, "noise_probability"),
    ],
)
def test_run_advanced_learning_sim_rejects_invalid_runtime_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    params: dict[str, float | int | bool] = {
        "size": 16,
        "time_steps": 32,
        "seed": 5,
        "save_plot": False,
        "verbose": False,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        run_advanced_learning_sim(**params)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"size": 4}, "size"),
        ({"flow_generation": -0.1}, "flow_generation"),
        ({"flow_damping": 1.0}, "flow_damping"),
        ({"shear_efficiency": -0.1}, "shear_efficiency"),
        ({"max_sub_steps": 0}, "max_sub_steps"),
    ],
)
def test_reactor_rejects_invalid_constructor_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        CoupledSandpileReactor(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": -0.1}, "alpha"),
        ({"gamma": 1.1}, "gamma"),
        ({"epsilon": float("nan")}, "epsilon"),
        ({"n_states_turb": 0}, "n_states_turb"),
        ({"n_states_flow": 0}, "n_states_flow"),
        ({"n_actions": 0}, "n_actions"),
    ],
)
def test_agent_rejects_invalid_constructor_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        FusionAIAgent(**kwargs)
