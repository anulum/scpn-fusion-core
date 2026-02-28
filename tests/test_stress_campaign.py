# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Stress-Test Campaign Tests (P1.4)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for the controller stress-test campaign and RESULTS.md wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Ensure validation/ is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))


# ── Campaign data structures ─────────────────────────────────────────


def test_episode_result_dataclass():
    """EpisodeResult should store all required fields."""
    from validation.stress_test_campaign import EpisodeResult
    ep = EpisodeResult(
        mean_abs_r_error=0.02,
        mean_abs_z_error=0.03,
        reward=-0.05,
        latency_us=50.0,
        disrupted=False,
        t_disruption=30.0,
        energy_efficiency=0.95,
    )
    assert ep.mean_abs_r_error == 0.02
    assert ep.disrupted is False
    assert ep.energy_efficiency == 0.95


def test_controller_metrics_dataclass():
    """ControllerMetrics should have correct defaults."""
    from validation.stress_test_campaign import ControllerMetrics
    m = ControllerMetrics(name="test")
    assert m.name == "test"
    assert m.n_episodes == 0
    assert m.mean_reward == 0.0
    assert m.disruption_rate == 0.0
    assert isinstance(m.episodes, list)


# ── Summary table generation ─────────────────────────────────────────


def test_generate_summary_table_format():
    """generate_summary_table should produce a markdown table."""
    from validation.stress_test_campaign import (
        ControllerMetrics,
        generate_summary_table,
    )
    results = {
        "PID": ControllerMetrics(
            name="PID", n_episodes=10,
            mean_reward=-0.1, std_reward=0.05,
            mean_r_error=0.03,
            p50_latency_us=40.0, p95_latency_us=60.0, p99_latency_us=80.0,
            disruption_rate=0.1, mean_def=0.9,
            mean_energy_efficiency=0.85,
        ),
        "H-infinity": ControllerMetrics(
            name="H-infinity", n_episodes=10,
            mean_reward=-0.08, std_reward=0.04,
            mean_r_error=0.02,
            p50_latency_us=45.0, p95_latency_us=70.0, p99_latency_us=100.0,
            disruption_rate=0.05, mean_def=0.95,
            mean_energy_efficiency=0.88,
        ),
    }
    table = generate_summary_table(results)
    assert "| PID" in table
    assert "| H-infinity" in table
    assert "Controller" in table
    assert "Mean Reward" in table
    # Should be multi-line
    lines = table.strip().split("\n")
    assert len(lines) >= 3  # header + separator + 2 data rows


# ── run_controller_campaign wiring ───────────────────────────────────


def test_run_controller_campaign_returns_dict():
    """run_controller_campaign should return a dict with expected keys."""
    from validation.stress_test_campaign import (
        ControllerMetrics,
        EpisodeResult,
    )

    # Mock the campaign runner
    mock_results = {
        "PID": ControllerMetrics(
            name="PID", n_episodes=5,
            mean_reward=-0.1, std_reward=0.05,
            mean_r_error=0.03,
            p50_latency_us=40.0, p95_latency_us=60.0, p99_latency_us=80.0,
            disruption_rate=0.1, mean_def=0.9,
            mean_energy_efficiency=0.85,
        ),
    }

    with patch("validation.stress_test_campaign.run_campaign", return_value=mock_results):
        from validation.collect_results import run_controller_campaign
        result = run_controller_campaign(quick=True)

    assert result is not None
    assert "n_episodes" in result
    assert "controllers" in result
    assert "markdown_table" in result
    assert "PID" in result["controllers"]


def test_campaign_controller_fields():
    """Each controller in campaign result should have all metric fields."""
    from validation.stress_test_campaign import ControllerMetrics

    mock_results = {
        "PID": ControllerMetrics(
            name="PID", n_episodes=5,
            mean_reward=-0.1, std_reward=0.05,
            mean_r_error=0.03,
            p50_latency_us=40.0, p95_latency_us=60.0, p99_latency_us=80.0,
            disruption_rate=0.1, mean_def=0.9,
            mean_energy_efficiency=0.85,
        ),
    }

    with patch("validation.stress_test_campaign.run_campaign", return_value=mock_results):
        from validation.collect_results import run_controller_campaign
        result = run_controller_campaign(quick=True)

    pid_data = result["controllers"]["PID"]
    expected_keys = [
        "n_episodes", "mean_reward", "std_reward", "mean_r_error",
        "p50_latency_us", "p95_latency_us", "p99_latency_us",
        "disruption_rate", "mean_def", "mean_energy_efficiency",
    ]
    for key in expected_keys:
        assert key in pid_data, f"Missing key: {key}"


# ── RESULTS.md generation with campaign ──────────────────────────────


def test_generate_results_md_includes_campaign():
    """generate_results_md should include campaign table when provided."""
    from validation.collect_results import generate_results_md
    campaign = {
        "n_episodes": 5,
        "controllers": {
            "PID": {"mean_reward": -0.1, "disruption_rate": 0.1},
        },
        "markdown_table": "| PID | 5 | ... |",
    }
    results = {
        "hil": None,
        "disruption": None,
        "q10": None,
        "tbr": None,
        "ecrh": None,
        "fb3d": None,
        "surrogates": None,
        "neural_eq": None,
        "fokker_planck": None,
        "spi_ablation": None,
        "campaign": campaign,
    }
    md = generate_results_md(
        hw="Test HW",
        results=results,
        elapsed_s=10.0,
    )
    assert "Controller Performance" in md
    assert "PID" in md


def test_generate_results_md_without_campaign():
    """generate_results_md without campaign should not fail."""
    from validation.collect_results import generate_results_md
    results = {
        "hil": None,
        "disruption": None,
        "q10": None,
        "tbr": None,
        "ecrh": None,
        "fb3d": None,
        "surrogates": None,
        "neural_eq": None,
        "fokker_planck": None,
        "spi_ablation": None,
        "campaign": None,
    }
    md = generate_results_md(
        hw="Test HW",
        results=results,
        elapsed_s=10.0,
    )
    assert "SCPN Fusion Core" in md
    assert "Controller Performance" not in md


# ── Controller registry ──────────────────────────────────────────────


def test_controllers_registry_has_pid_and_hinf():
    """CONTROLLERS registry should always have PID and H-infinity."""
    from validation.stress_test_campaign import CONTROLLERS
    assert "PID" in CONTROLLERS
    assert "H-infinity" in CONTROLLERS


def test_controllers_registry_tracks_optional_mpc_lane():
    """MPC lane should only be exposed when the optional dependency is present."""
    import validation.stress_test_campaign as mod

    if mod._mpc_available:
        assert "MPC" in mod.CONTROLLERS
    else:
        assert "MPC" not in mod.CONTROLLERS


def test_mpc_episode_requires_optional_dependency(monkeypatch):
    """MPC episode runner should fail cleanly when optional deps are absent."""
    import validation.stress_test_campaign as mod

    monkeypatch.setattr(mod, "_mpc_available", False)
    monkeypatch.setattr(mod, "ModelPredictiveController", None, raising=False)
    monkeypatch.setattr(mod, "NeuralSurrogate", None, raising=False)
    with pytest.raises(RuntimeError, match="MPC controller is unavailable"):
        mod._run_mpc_episode(config_path="unused", shot_duration=30)


def test_mpc_episode_with_mocked_dependencies(monkeypatch):
    """MPC runner should emit a finite EpisodeResult when dependencies are mocked."""
    import validation.stress_test_campaign as mod

    class FakeKernel:
        def __init__(self):
            self.cfg = {"coils": [{"current": 0.0}, {"current": 0.0}]}
            self.Psi = np.array([[0.0, 1.0], [0.2, 0.3]], dtype=np.float64)
            self.R = np.array([5.8, 6.1], dtype=np.float64)
            self.Z = np.array([-0.2, 0.1], dtype=np.float64)

        def find_x_point(self, psi):
            return np.array([5.0, -3.5], dtype=np.float64), 0.0

    class FakeIsoFluxController:
        def __init__(self, config_path, verbose=False, kernel_factory=None, control_dt_s=0.05):
            self.kernel = FakeKernel()
            self.pid_R = {"Kp": 2.0}
            self.pid_step = lambda pid, err: float(err)

        def run_shot(self, shot_duration: int, save_plot: bool = False):
            _ = self.pid_step(self.pid_R, 0.1)
            return {
                "steps": float(shot_duration),
                "mean_abs_r_error": 0.12,
                "mean_abs_z_error": 0.09,
                "mean_abs_radial_actuator_lag": 0.2,
            }

    class FakeSurrogate:
        def __init__(self, n_coils, n_state, verbose=False):
            self.n_coils = n_coils
            self.n_state = n_state

        def train_on_kernel(self, kernel, perturbation: float = 1.0):
            return None

    class FakeMPC:
        def __init__(
            self,
            surrogate_model,
            target_state,
            prediction_horizon=6,
            learning_rate=0.25,
            iterations=8,
            action_limit=2.0,
        ):
            self.target_state = target_state

        def plan_trajectory(self, state):
            return np.array([0.4, 0.0], dtype=np.float64)

    monkeypatch.setattr(mod, "_mpc_available", True)
    monkeypatch.setattr(mod, "IsoFluxController", FakeIsoFluxController)
    monkeypatch.setattr(mod, "NeuralSurrogate", FakeSurrogate, raising=False)
    monkeypatch.setattr(mod, "ModelPredictiveController", FakeMPC, raising=False)

    episode = mod._run_mpc_episode(config_path="unused", shot_duration=30, surrogate=False)
    assert np.isfinite(episode.reward)
    assert episode.t_disruption == 30.0
    assert episode.energy_efficiency == pytest.approx(1.0 / 1.2, abs=1e-12)


def test_rust_pid_episode_non_disrupted_uses_full_duration_for_def(monkeypatch):
    """Non-disrupted Rust episodes must report full-shot DEF support."""
    import validation.stress_test_campaign as mod

    class FakeReport:
        steps = 100
        duration_s = 30.0
        mean_abs_r_error = 0.02
        mean_abs_z_error = 0.03
        disrupted = False
        pf_constraint_events = 0
        heating_constraint_events = 0
        vessel_contact_events = 0

    class FakeSim:
        def __init__(self, target_r: float, target_z: float, control_hz: float) -> None:
            pass

        def run_shot(self, shot_duration: float) -> FakeReport:
            return FakeReport()

    monkeypatch.setattr(mod, "_rust_flight_sim_available", True)
    monkeypatch.setattr(mod, "PyRustFlightSim", FakeSim, raising=False)
    episode = mod._run_rust_pid_episode(config_path="unused", shot_duration=30)
    assert episode.disrupted is False
    assert episode.t_disruption == 30.0


def test_rust_pid_episode_clamps_disruption_time_to_shot_duration(monkeypatch):
    """Disrupted Rust episodes should never report t_disruption above shot length."""
    import validation.stress_test_campaign as mod

    class FakeReport:
        steps = 120
        duration_s = 45.0
        mean_abs_r_error = 0.8
        mean_abs_z_error = 0.7
        disrupted = True
        pf_constraint_events = 0
        heating_constraint_events = 0
        vessel_contact_events = 0

    class FakeSim:
        def __init__(self, target_r: float, target_z: float, control_hz: float) -> None:
            pass

        def run_shot(self, shot_duration: float) -> FakeReport:
            return FakeReport()

    monkeypatch.setattr(mod, "_rust_flight_sim_available", True)
    monkeypatch.setattr(mod, "PyRustFlightSim", FakeSim, raising=False)
    episode = mod._run_rust_pid_episode(config_path="unused", shot_duration=30)
    assert episode.disrupted is True
    assert episode.t_disruption == 30.0


def test_rust_pid_episode_requires_rust_binding(monkeypatch):
    """Calling Rust lane without binding should fail with explicit error."""
    import validation.stress_test_campaign as mod

    monkeypatch.setattr(mod, "_rust_flight_sim_available", False)
    monkeypatch.setattr(mod, "PyRustFlightSim", None, raising=False)
    with pytest.raises(RuntimeError, match="Rust flight simulator is unavailable"):
        mod._run_rust_pid_episode(config_path="unused", shot_duration=30)


def test_rust_pid_episode_energy_efficiency_tracks_constraint_events(monkeypatch):
    """Rust lane should derive efficiency from constraint-event burden."""
    import validation.stress_test_campaign as mod

    class FakeReport:
        steps = 100
        duration_s = 30.0
        mean_abs_r_error = 0.05
        mean_abs_z_error = 0.04
        disrupted = False
        pf_constraint_events = 5
        heating_constraint_events = 10
        vessel_contact_events = 0

    class FakeSim:
        def __init__(self, target_r: float, target_z: float, control_hz: float) -> None:
            pass

        def run_shot(self, shot_duration: float) -> FakeReport:
            return FakeReport()

    monkeypatch.setattr(mod, "_rust_flight_sim_available", True)
    monkeypatch.setattr(mod, "PyRustFlightSim", FakeSim, raising=False)
    episode = mod._run_rust_pid_episode(config_path="unused", shot_duration=30)
    assert episode.energy_efficiency == pytest.approx(0.85, abs=1e-12)
