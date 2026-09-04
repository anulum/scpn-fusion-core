# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stress-campaign orchestration tests
"""Focused public-workflow tests for the controller stress campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import validation.stress_test_campaign as campaign
from validation.stress_campaign_contract import (
    RESULT_SCHEMA,
    CampaignResults,
    ControllerMetrics,
    EpisodeResult,
    StressScenario,
)


def _successful_episode(
    *,
    scenario: StressScenario,
    episode_index: int,
    episode_seed: int,
    evaluation_contract_digest: str = "test-contract",
) -> EpisodeResult:
    """Return a deterministic episode for orchestration-boundary tests."""
    return EpisodeResult(
        episode_index=episode_index,
        seed=episode_seed,
        scenario_digest=scenario.digest,
        evaluation_contract_digest=evaluation_contract_digest,
        disturbance_trace_digest=f"trace-{episode_index}",
        realized_measurement_noise_rms_m=scenario.measurement_noise_std_m,
        mean_abs_r_error_m=0.02 + episode_index * 0.001,
        mean_abs_z_error_m=0.01,
        tracking_reward_m=-0.03 - episode_index * 0.001,
        control_policy_latency_us=[10.0 + episode_index],
        simulation_wall_time_us=100.0,
        disrupted=False,
        t_disruption_s=1.0,
        magnetic_actuator_absolute_current_offset_integral_ma_s=0.01,
        mean_abs_coil_current_offset_tracking_error_ma=0.001,
    )


def _recording_runner(calls: list[dict[str, Any]]):
    """Build a protocol-conforming runner that records the received scenario."""

    def runner(
        config_path: str | Path,
        shot_duration: int = 30,
        surrogate: bool = False,
        *,
        scenario: StressScenario,
        episode_index: int,
        episode_seed: int,
        evaluation_contract_digest: str,
    ) -> EpisodeResult:
        calls.append(
            {
                "config_path": Path(config_path),
                "shot_duration": shot_duration,
                "surrogate": surrogate,
                "scenario": scenario,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "evaluation_contract_digest": evaluation_contract_digest,
            }
        )
        return _successful_episode(
            scenario=scenario,
            episode_index=episode_index,
            episode_seed=episode_seed,
            evaluation_contract_digest=evaluation_contract_digest,
        )

    return runner


def test_campaign_applies_one_exact_scenario_to_every_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner receives explicit disturbance units and stable per-episode seeds."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(campaign, "CONTROLLERS", {"PID": _recording_runner(calls)})

    results = campaign.run_campaign(
        n_episodes=3,
        shot_duration=1,
        controllers=["PID"],
        measurement_noise_std_m=0.015,
        actuator_delay_ms=50.0,
        seed=91,
        checkpoint_dir=tmp_path,
    )

    lane = results["PID"]
    assert lane.status == "complete"
    assert lane.comparable
    assert [call["episode_index"] for call in calls] == [0, 1, 2]
    assert len({call["episode_seed"] for call in calls}) == 3
    assert all(call["scenario"].digest == results.scenario.digest for call in calls)
    assert results.scenario.measurement_noise_std_m == pytest.approx(0.015)
    assert results.scenario.actuator_delay_s == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_episodes": 1.5}, "n_episodes"),
        ({"n_episodes": True}, "n_episodes"),
        ({"shot_duration": 1.5}, "shot_duration"),
        ({"controllers": []}, "at least one"),
        ({"controllers": ["PID", "PID"]}, "duplicate"),
    ],
)
def test_campaign_rejects_ambiguous_counts_and_controller_sets(
    kwargs: dict[str, Any], message: str, tmp_path: Path
) -> None:
    """The public API never truncates counts or deduplicates requested lanes."""
    with pytest.raises(ValueError, match=message):
        campaign.run_campaign(seed=1, checkpoint_dir=tmp_path, **kwargs)


def test_resume_requires_explicit_seed_and_existing_identity(tmp_path: Path) -> None:
    """Recovery cannot silently create a fresh campaign after operator drift."""
    with pytest.raises(ValueError, match="exact recorded --seed"):
        campaign.run_campaign(
            n_episodes=1,
            shot_duration=1,
            controllers=["PID"],
            checkpoint_dir=tmp_path / "missing",
            resume=True,
        )

    with pytest.raises(FileNotFoundError, match="identity manifest"):
        campaign.run_campaign(
            n_episodes=1,
            shot_duration=1,
            controllers=["PID"],
            seed=3,
            checkpoint_dir=tmp_path / "missing",
            resume=True,
        )


def test_failed_lane_serializes_null_metrics_and_failure_details(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A backend exception cannot become a zero-error pseudo-measurement."""

    def failing_runner(
        config_path: str | Path,
        shot_duration: int = 30,
        surrogate: bool = False,
        *,
        scenario: StressScenario,
        episode_index: int,
        episode_seed: int,
        evaluation_contract_digest: str,
    ) -> EpisodeResult:
        raise RuntimeError(f"sensor frame missing at episode {episode_index}")

    monkeypatch.setattr(campaign, "CONTROLLERS", {"PID": failing_runner})
    results = campaign.run_campaign(
        n_episodes=1,
        shot_duration=1,
        controllers=["PID"],
        seed=12,
        checkpoint_dir=tmp_path / "checkpoint",
    )
    output = tmp_path / "result.json"
    provenance = campaign.collect_provenance(
        n_episodes=1,
        shot_duration=1,
        seed=results.scenario.master_seed,
        controllers=["PID"],
        timestamp_utc="2026-09-04T12:00:00+00:00",
        scenario=results.scenario,
        campaign_identity=results.campaign_identity,
    )
    campaign.save_results_json(results, output, provenance)
    payload = json.loads(output.read_text(encoding="utf-8"))

    lane = payload["controllers"]["PID"]
    assert lane["status"] == "failed"
    assert lane["n_episodes"] == 0
    assert lane["mean_tracking_reward_m"] is None
    assert lane["p95_control_policy_latency_us"] is None
    assert lane["failures"][0]["exception_type"] == "RuntimeError"
    assert lane["failures"][0]["stage"] == "episode_runner"
    assert "RuntimeError" in lane["failures"][0]["traceback_text"]


def test_result_writer_rejects_unbound_or_unprovenanced_input(tmp_path: Path) -> None:
    """A v3 scientific report cannot carry null campaign identity or provenance."""
    with pytest.raises(TypeError, match="CampaignResults"):
        campaign.save_results_json({}, tmp_path / "plain.json", {"host": "test"})  # type: ignore[arg-type]

    scenario = StressScenario(master_seed=4)
    results = CampaignResults(scenario=scenario, campaign_identity={"digest": "test"})
    with pytest.raises(ValueError, match="provenance"):
        campaign.save_results_json(results, tmp_path / "missing.json", {})


def test_resume_continues_after_last_atomic_episode_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interrupted campaign resumes without replaying a completed episode."""
    first_calls: list[int] = []

    def interrupted_runner(
        config_path: str | Path,
        shot_duration: int = 30,
        surrogate: bool = False,
        *,
        scenario: StressScenario,
        episode_index: int,
        episode_seed: int,
        evaluation_contract_digest: str,
    ) -> EpisodeResult:
        first_calls.append(episode_index)
        if episode_index == 1:
            raise SystemExit("operator interruption")
        return _successful_episode(
            scenario=scenario,
            episode_index=episode_index,
            episode_seed=episode_seed,
            evaluation_contract_digest=evaluation_contract_digest,
        )

    checkpoint_dir = tmp_path / "checkpoint"
    monkeypatch.setattr(campaign, "CONTROLLERS", {"PID": interrupted_runner})
    with pytest.raises(SystemExit, match="operator interruption"):
        campaign.run_campaign(
            n_episodes=2,
            shot_duration=1,
            controllers=["PID"],
            seed=44,
            checkpoint_dir=checkpoint_dir,
        )
    assert first_calls == [0, 1]

    resumed_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(campaign, "CONTROLLERS", {"PID": _recording_runner(resumed_calls)})
    resumed = campaign.run_campaign(
        n_episodes=2,
        shot_duration=1,
        controllers=["PID"],
        seed=44,
        checkpoint_dir=checkpoint_dir,
        resume=True,
    )

    assert [call["episode_index"] for call in resumed_calls] == [1]
    assert [episode.episode_index for episode in resumed["PID"].episodes] == [0, 1]
    assert resumed["PID"].status == "complete"


def test_hinf_gate_is_an_explicit_unavailable_lane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A disabled research controller emits status/reason and no metrics."""
    monkeypatch.delenv(campaign.HINF_RESEARCH_ENV, raising=False)
    results = campaign.run_campaign(
        n_episodes=2,
        shot_duration=1,
        controllers=["H-infinity"],
        seed=8,
        checkpoint_dir=tmp_path,
    )

    lane = results["H-infinity"]
    assert lane.status == "unavailable"
    assert lane.reason == f"research_gate_disabled:{campaign.HINF_RESEARCH_ENV}"
    assert lane.n_episodes == 0
    assert lane.mean_tracking_reward_m is None
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["remaining_episodes"] == 0
    assert progress["unavailable_episodes"] == 2


def test_summary_and_graduation_reject_incomplete_comparison() -> None:
    """Operator and promotion outputs expose incomplete lanes as N/A/ineligible."""
    scenario = StressScenario(master_seed=5)
    pid = ControllerMetrics(
        "PID", 1, scenario.digest, "test-contract", "test.pid", status="failed", reason="failure"
    )
    hinf = ControllerMetrics(
        "H-infinity",
        1,
        scenario.digest,
        "test-contract",
        "test.hinf",
        status="unavailable",
        reason="gate",
    )
    results = {"PID": pid, "H-infinity": hinf}

    table = campaign.generate_summary_table(results)
    graduation = campaign.derive_hinf_graduation_status(results)
    assert "N/A" in table
    assert graduation["eligible_for_default_lane"] is False
    assert graduation["reason"] == "pid_or_hinf_lane_incomplete_or_incomparable"


def test_campaign_rejects_cross_lane_disturbance_trace_drift() -> None:
    """A completed lane set is incomparable when one trace digest diverges."""
    scenario = StressScenario(master_seed=17)
    identity = {
        "digest": "test",
        "payload": {
            "evaluation_contract": {"payload": {"evidence_scope": "controller_comparison"}}
        },
    }
    results = CampaignResults(scenario=scenario, campaign_identity=identity)
    for name in ("PID", "LQR"):
        lane = ControllerMetrics(name, 1, scenario.digest, "test-contract", f"test.{name}")
        lane.episodes.append(
            _successful_episode(
                scenario=scenario,
                episode_index=0,
                episode_seed=scenario.episode_seed(0),
            )
        )
        lane.finalize(1.0)
        results[name] = lane

    results["LQR"].episodes[0] = EpisodeResult(
        **{
            **results["LQR"].episodes[0].to_dict(),
            "disturbance_trace_digest": "different-trace",
        }
    )

    assert not campaign.campaign_is_complete_and_comparable(results)
    assert campaign.campaign_promotion_status(results) == (
        False,
        "campaign_incomplete_or_incomparable",
    )


def test_cli_parser_and_main_forward_full_scenario_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Public CLI flags reach the campaign without global RNG mutation."""
    captured: dict[str, Any] = {}
    scenario = StressScenario(0.01, 0.025, 99)

    def recording_campaign(**kwargs: Any) -> CampaignResults:
        captured.update(kwargs)
        results = CampaignResults(scenario=scenario, campaign_identity={"digest": "test"})
        lane = ControllerMetrics("PID", 1, scenario.digest, "test-contract", "test.pid")
        lane.episodes.append(
            _successful_episode(
                scenario=scenario,
                episode_index=0,
                episode_seed=scenario.episode_seed(0),
            )
        )
        lane.n_episodes = 1
        lane.finalize(1.0)
        results["PID"] = lane
        return results

    monkeypatch.setattr(campaign, "run_campaign", recording_campaign)
    result = campaign.main(
        [
            "--episodes",
            "2",
            "--controllers",
            "PID",
            "--measurement-noise-std-m",
            "0.01",
            "--actuator-delay-ms",
            "25",
            "--seed",
            "99",
            "--checkpoint-dir",
            str(tmp_path),
            "--resume",
        ]
    )

    assert isinstance(result, CampaignResults)
    assert captured["controllers"] == ["PID"]
    assert captured["measurement_noise_std_m"] == pytest.approx(0.01)
    assert captured["actuator_delay_ms"] == pytest.approx(25.0)
    assert captured["seed"] == 99
    assert captured["resume"] is True


def test_cli_exits_nonzero_for_failed_scientific_campaign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automation cannot mistake caught episode failures for campaign success."""
    scenario = StressScenario(master_seed=99)
    results = CampaignResults(scenario=scenario, campaign_identity={"digest": "test"})
    results["PID"] = ControllerMetrics(
        "PID",
        1,
        scenario.digest,
        "test-contract",
        "test.pid",
        status="failed",
        reason="episode_failure",
    )
    monkeypatch.setattr(campaign, "run_campaign", lambda **kwargs: results)

    with pytest.raises(SystemExit) as exc:
        campaign.main(["--episodes", "1", "--controllers", "PID", "--seed", "99"])
    assert exc.value.code == 2


def test_report_contains_schema_scenario_identity_and_complete_episode_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The public JSON report is self-describing and recovery-grade."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(campaign, "CONTROLLERS", {"PID": _recording_runner(calls)})
    results = campaign.run_campaign(
        n_episodes=1,
        shot_duration=1,
        controllers=["PID"],
        seed=101,
        checkpoint_dir=tmp_path / "checkpoint",
    )
    provenance = campaign.collect_provenance(
        n_episodes=1,
        shot_duration=1,
        seed=results.scenario.master_seed,
        controllers=["PID"],
        timestamp_utc="2026-09-04T12:00:00+00:00",
        scenario=results.scenario,
        campaign_identity=results.campaign_identity,
    )
    output = tmp_path / "report.json"
    campaign.save_results_json(results, output, provenance)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema"] == RESULT_SCHEMA
    assert payload["scenario_digest"] == results.scenario.digest
    assert payload["campaign_identity"]["digest"] == results.campaign_identity["digest"]
    assert payload["campaign_status"] == "complete"
    assert payload["campaign_complete"] is True
    assert payload["promotion_eligible"] is True
    assert payload["promotion_ineligibility_reason"] is None
    assert payload["controllers"]["PID"]["status"] == "complete"
    assert payload["controllers"]["PID"]["episodes"][0]["seed"] == calls[0]["episode_seed"]
    assert provenance["schema"] == "scpn-fusion-core.stress-test-campaign-provenance.v3"
