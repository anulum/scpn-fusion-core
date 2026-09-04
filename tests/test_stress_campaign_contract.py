# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stress-campaign contract tests
"""Tests for scenario identity and honest lane-result semantics."""

from __future__ import annotations

import pytest

from validation.stress_campaign_contract import (
    ControllerMetrics,
    EpisodeFailure,
    EpisodeResult,
    StressScenario,
)


def _episode(scenario: StressScenario, index: int = 0) -> EpisodeResult:
    """Build one deterministic successful episode for aggregate tests."""
    return EpisodeResult(
        episode_index=index,
        seed=scenario.episode_seed(index),
        scenario_digest=scenario.digest,
        evaluation_contract_digest="test-contract",
        disturbance_trace_digest=f"trace-{index}",
        realized_measurement_noise_rms_m=scenario.measurement_noise_std_m,
        mean_abs_r_error_m=0.1,
        mean_abs_z_error_m=0.02,
        tracking_reward_m=-0.12,
        control_policy_latency_us=[14.0, 15.0, 16.0],
        simulation_wall_time_us=100.0,
        disrupted=False,
        t_disruption_s=30.0,
        magnetic_actuator_absolute_current_offset_integral_ma_s=0.2,
        mean_abs_coil_current_offset_tracking_error_ma=0.001,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"measurement_noise_std_m": -0.1}, "measurement_noise_std_m"),
        ({"measurement_noise_std_m": float("nan")}, "measurement_noise_std_m"),
        ({"actuator_delay_s": -0.1}, "actuator_delay_s"),
        ({"master_seed": -1}, "master_seed"),
        ({"master_seed": 2**64}, "master_seed"),
    ],
)
def test_scenario_rejects_nonphysical_or_nonreplayable_values(
    kwargs: dict[str, float | int], message: str
) -> None:
    """The shared scenario fails closed on invalid units and seed range."""
    with pytest.raises(ValueError, match=message):
        StressScenario(**kwargs)


def test_episode_seed_is_order_independent_and_common_across_lanes() -> None:
    """Seed derivation is stable and gives every lane the same episode disturbance."""
    scenario = StressScenario(master_seed=1942)
    expected = scenario.episode_seed(7)

    assert scenario.episode_seed(7) == expected
    assert scenario.episode_seed(8) != expected
    assert StressScenario(master_seed=1942).digest == scenario.digest


def test_zero_success_lane_has_no_pseudo_measurements() -> None:
    """A failed lane reports null aggregates rather than attractive zeroes."""
    scenario = StressScenario(master_seed=3)
    metrics = ControllerMetrics(
        name="PID",
        requested_episodes=1,
        scenario_digest=scenario.digest,
        evaluation_contract_digest="test-contract",
        policy_implementation="test.pid",
    )
    metrics.failures.append(
        EpisodeFailure(
            episode_index=0,
            seed=3,
            exception_type="RuntimeError",
            message="failed",
            backend="PID",
            stage="episode_runner",
            traceback_text="RuntimeError: failed",
        )
    )
    metrics.finalize(30.0)

    assert metrics.status == "failed"
    assert metrics.n_episodes == 0
    assert metrics.failed_episodes == 1
    assert metrics.mean_tracking_reward_m is None
    assert metrics.p95_control_policy_latency_us is None
    assert not metrics.comparable


def test_complete_lane_is_comparable_only_for_exact_scenario() -> None:
    """Completion requires every requested episode under one scenario digest."""
    scenario = StressScenario(master_seed=11)
    metrics = ControllerMetrics(
        name="PID",
        requested_episodes=1,
        scenario_digest=scenario.digest,
        evaluation_contract_digest="test-contract",
        policy_implementation="test.pid",
    )
    metrics.episodes.append(_episode(scenario))
    metrics.finalize(30.0)

    assert metrics.status == "complete"
    assert metrics.comparable
    assert metrics.mean_tracking_reward_m == pytest.approx(-0.12)
    assert metrics.mean_abs_z_error_m == pytest.approx(0.02)
    assert metrics.mean_def == pytest.approx(1.0)

    metrics.episodes[0] = EpisodeResult(
        **{**metrics.episodes[0].to_dict(), "scenario_digest": "different"}
    )
    assert not metrics.comparable


def test_episode_result_rejects_nonfinite_scientific_output() -> None:
    """A runner cannot inject NaN into checkpoints or aggregate evidence."""
    scenario = StressScenario(master_seed=21)
    data = _episode(scenario).to_dict()
    data["mean_abs_r_error_m"] = float("nan")
    with pytest.raises(ValueError, match="nonnegative metrics"):
        EpisodeResult.from_dict(data)
