# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stress-campaign recovery tests
"""Tests for atomic checkpoints, identity binding, and progress records."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from validation.stress_campaign_contract import (
    ControllerMetrics,
    EpisodeResult,
    StressScenario,
    digest_json,
)
from validation.stress_campaign_recovery import (
    PROGRESS_SCHEMA,
    RecoveryStore,
    atomic_write_json,
    build_campaign_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _identity(config_path: Path, scenario: StressScenario) -> dict[str, object]:
    """Build a real source-bound identity with a temporary config input."""
    return build_campaign_identity(
        repo_root=REPO_ROOT,
        config_path=config_path,
        scenario=scenario,
        controllers=["PID"],
        requested_episodes=2,
        shot_duration_s=30,
        surrogate=False,
        software_versions={"numpy": "test"},
        execution_options={"hinf_research_enabled": False},
        evaluation_contract={"digest": "test-contract", "payload": {"plant": "test"}},
        controller_implementations={"PID": "test.pid"},
    )


def _metrics(scenario: StressScenario) -> ControllerMetrics:
    """Build a checkpointable lane with one completed episode."""
    metrics = ControllerMetrics(
        name="PID",
        requested_episodes=2,
        scenario_digest=scenario.digest,
        evaluation_contract_digest="test-contract",
        policy_implementation="test.pid",
        status="running",
    )
    metrics.episodes.append(
        EpisodeResult(
            episode_index=0,
            seed=scenario.episode_seed(0),
            scenario_digest=scenario.digest,
            evaluation_contract_digest="test-contract",
            disturbance_trace_digest="trace-0",
            realized_measurement_noise_rms_m=scenario.measurement_noise_std_m,
            mean_abs_r_error_m=0.1,
            mean_abs_z_error_m=0.02,
            tracking_reward_m=-0.12,
            control_policy_latency_us=[12.0],
            simulation_wall_time_us=100.0,
            disrupted=False,
            t_disruption_s=30.0,
            magnetic_actuator_absolute_current_offset_integral_ma_s=0.2,
            mean_abs_coil_current_offset_tracking_error_ma=0.001,
        )
    )
    metrics.n_episodes = 1
    return metrics


def test_atomic_write_replaces_complete_json_without_residue(tmp_path: Path) -> None:
    """Atomic publication leaves one parseable destination and no temp files."""
    destination = tmp_path / "progress.json"
    atomic_write_json(destination, {"generation": 1})
    atomic_write_json(destination, {"generation": 2})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"generation": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_round_trip_and_corruption_rejection(tmp_path: Path) -> None:
    """A lane round-trips exactly and fails closed after content tampering."""
    config = tmp_path / "config.json"
    config.write_text('{"machine":"ITER"}\n', encoding="utf-8")
    scenario = StressScenario(master_seed=7)
    store = RecoveryStore(tmp_path / "recovery", _identity(config, scenario))
    store.write_lane(_metrics(scenario))

    restored = store.load_lane("PID")
    assert restored is not None
    assert restored.to_dict() == _metrics(scenario).to_dict()

    checkpoint = store.checkpoint_path("PID")
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    data["metrics"]["episodes"][0]["tracking_reward_m"] = 999.0
    checkpoint.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        store.load_lane("PID")


def test_resume_rejects_config_or_environment_identity_drift(tmp_path: Path) -> None:
    """Changing an input or environment prevents checkpoint reuse."""
    config = tmp_path / "config.json"
    config.write_text('{"machine":"ITER"}\n', encoding="utf-8")
    scenario = StressScenario(master_seed=9)
    original = RecoveryStore(tmp_path / "recovery", _identity(config, scenario))
    original.write_lane(_metrics(scenario))

    config.write_text('{"machine":"SPARC"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        RecoveryStore(tmp_path / "recovery", _identity(config, scenario), require_existing=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "LQR", "embedded controller"),
        ("requested_episodes", 1, "requested episode count"),
        ("scenario_digest", "wrong", "scenario digest"),
        ("policy_implementation", "wrong", "policy implementation"),
    ],
)
def test_resume_rejects_embedded_lane_identity_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """Outer checkpoint identity cannot conceal mismatched embedded metrics."""
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    scenario = StressScenario(master_seed=11)
    store = RecoveryStore(tmp_path / "recovery", _identity(config, scenario))
    store.write_lane(_metrics(scenario))
    checkpoint = store.checkpoint_path("PID")
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    data["metrics"][field] = value
    body = {key: item for key, item in data.items() if key != "content_digest"}
    data["content_digest"] = digest_json(body)
    checkpoint.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        store.load_lane("PID")


def test_progress_reports_success_failure_remaining_and_eta(tmp_path: Path) -> None:
    """Operator progress exposes lane-level counts before final result creation."""
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    scenario = StressScenario(master_seed=13)
    store = RecoveryStore(tmp_path / "recovery", _identity(config, scenario))
    lane = _metrics(scenario)
    store.write_progress(
        lanes={"PID": lane},
        active_controller="PID",
        active_episode_index=1,
        active_episode_seed=scenario.episode_seed(1),
        started_monotonic=time.monotonic() - 1.0,
        attempted_at_start=0,
    )

    progress = json.loads((store.directory / "progress.json").read_text(encoding="utf-8"))
    assert progress["schema"] == PROGRESS_SCHEMA
    assert progress["completed_episodes"] == 1
    assert progress["failed_episodes"] == 0
    assert progress["remaining_episodes"] == 1
    assert progress["eta_s"] is None
    assert progress["eta_method"] is None
    assert progress["active_controller"] == "PID"
    assert progress["active_episode_index"] == 1
    assert progress["active_episode_seed"] == scenario.episode_seed(1)


def test_progress_rejects_incomplete_active_episode_identity(tmp_path: Path) -> None:
    """An operator must never see an ambiguous active-episode heartbeat."""
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    scenario = StressScenario(master_seed=17)
    store = RecoveryStore(tmp_path / "recovery", _identity(config, scenario))
    lane = _metrics(scenario)

    with pytest.raises(ValueError, match="reported together"):
        store.write_progress(
            lanes={"PID": lane},
            active_controller="PID",
            active_episode_index=1,
            active_episode_seed=None,
            started_monotonic=time.monotonic(),
            attempted_at_start=0,
        )


def test_active_episode_heartbeat_refreshes_stale_detectable_progress(tmp_path: Path) -> None:
    """A long runner publishes timestamped liveness without changing episode counts."""
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    scenario = StressScenario(master_seed=19)
    store = RecoveryStore(tmp_path / "recovery", _identity(config, scenario))
    lane = _metrics(scenario)
    with store.progress_heartbeat(
        lanes={"PID": lane},
        active_controller="PID",
        active_episode_index=1,
        active_episode_seed=scenario.episode_seed(1),
        started_monotonic=time.monotonic(),
        attempted_at_start=1,
        interval_s=0.01,
    ):
        time.sleep(0.03)

    progress = json.loads((store.directory / "progress.json").read_text(encoding="utf-8"))
    assert progress["snapshot_kind"] == "heartbeat"
    assert progress["active_controller"] == "PID"
    assert progress["active_episode_index"] == 1
    assert progress["updated_at_utc"].endswith("+00:00")
    assert progress["writer_run_id"] == store.writer_run_id
    assert progress["writer_pid"] > 0

    with pytest.raises(ValueError, match="requires an active controller"):
        store.write_progress(
            lanes={"PID": lane},
            active_controller=None,
            active_episode_index=1,
            active_episode_seed=scenario.episode_seed(1),
            started_monotonic=time.monotonic(),
            attempted_at_start=0,
        )
