# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real stress-campaign policy integration tests
"""Real controller-policy integration on the common tokamak plant contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_fusion._data_paths import default_iter_config_path
from validation import stress_test_campaign as campaign
from validation.stress_campaign_contract import StressScenario


def test_real_policy_lanes_share_exact_plant_contract_and_disturbance_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """All admissible policies run once per step against identical observations."""
    if campaign.PyRustIsoFluxController is None:
        pytest.skip("Rust PID policy extension is not built")
    monkeypatch.setenv(campaign.HINF_RESEARCH_ENV, "1")
    controllers = ["PID", "H-infinity", "LQR", "MPC", "LIF-NEF-SNN", "Rust-PID"]
    results = campaign.run_campaign(
        n_episodes=1,
        shot_duration=1,
        surrogate=True,
        controllers=controllers,
        measurement_noise_std_m=0.01,
        actuator_delay_ms=50.0,
        seed=20260904,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    assert campaign.campaign_is_complete_and_comparable(results)
    assert campaign.campaign_promotion_status(results) == (
        False,
        "evaluation_scope:wiring_only",
    )
    assert {lane.status for lane in results.values()} == {"complete"}
    assert len({lane.evaluation_contract_digest for lane in results.values()}) == 1
    episodes = [lane.episodes[0] for lane in results.values()]
    assert len({episode.disturbance_trace_digest for episode in episodes}) == 1
    assert all(len(episode.control_policy_latency_us) == 100 for episode in episodes)
    assert all(episode.mean_abs_z_error_m >= 0.0 for episode in episodes)
    assert all(
        episode.magnetic_actuator_absolute_current_offset_integral_ma_s >= 0.0
        for episode in episodes
    )
    python_pid = results["PID"].episodes[0]
    rust_pid = results["Rust-PID"].episodes[0]
    assert rust_pid.mean_abs_r_error_m == pytest.approx(python_pid.mean_abs_r_error_m)
    assert rust_pid.mean_abs_z_error_m == pytest.approx(python_pid.mean_abs_z_error_m)
    assert rust_pid.magnetic_actuator_absolute_current_offset_integral_ma_s == pytest.approx(
        python_pid.magnetic_actuator_absolute_current_offset_integral_ma_s
    )


def test_uncalibrated_nmpc_is_present_but_fail_closed(tmp_path: Path) -> None:
    """The random-MLP NMPC lane is never omitted or promoted as evidence."""
    results = campaign.run_campaign(
        n_episodes=1,
        shot_duration=1,
        surrogate=True,
        controllers=["NMPC-JAX"],
        seed=7,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    lane = results["NMPC-JAX"]
    assert lane.status == "unavailable"
    assert lane.reason == "uncalibrated_dynamics_model:no_trained_artifact_or_held_out_gate"
    assert lane.episodes == []
    assert not lane.comparable


def test_evaluation_contract_uses_config_targets_and_ma_action_units() -> None:
    """Targets and current units come from the governed common-plant contract."""
    scenario = StressScenario(measurement_noise_std_m=0.01, actuator_delay_s=0.05, master_seed=3)
    contract = campaign.build_evaluation_contract(
        default_iter_config_path(), surrogate=False, scenario=scenario
    )
    payload = contract["payload"]

    assert payload["axis_target_m"] == {"R": 5.9, "Z": -2.1}
    assert payload["action"] == "full_ordered_coil_current_offset_setpoint_ma"
    assert payload["evidence_scope"] == "controller_comparison"
    assert payload["actuator"]["offset_limit_ma"] == pytest.approx(0.05)
    assert payload["actuator"]["pure_command_delay_s"] == pytest.approx(0.05)
