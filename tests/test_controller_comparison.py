# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Controller Comparison Hardening Tests
"""Regression tests for validation/controller_comparison.py hardening."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure validation/ is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))


def test_hinf_episode_uses_independent_axis_controllers(monkeypatch) -> None:
    """The H-infinity episode runs one controller per axis, each primed and reset once."""
    import validation.controller_comparison as mod

    created: list[Any] = []

    class FakeHinf:
        """H-infinity stand-in that records every step and reset."""

        def __init__(self) -> None:
            """Start with a feed-through gain of 2 and empty records."""
            self._Fd = np.array([[2.0]], dtype=np.float64)
            self.steps: list[tuple[float, float]] = []
            self.reset_calls = 0

        def step(self, err: float, dt: float) -> float:
            """Record the step and return ten times the error."""
            self.steps.append((float(err), float(dt)))
            return float(err) * 10.0

        def reset(self) -> None:
            """Count the reset."""
            self.reset_calls += 1

    class FakeIsoFluxController:
        """IsoFlux stand-in that drives one radial and one vertical step per shot."""

        def __init__(
            self,
            config_path: str,
            verbose: bool = False,
            control_dt_s: float = 0.05,
        ) -> None:
            """Store the PID gains and an identity loop step."""
            self.pid_R = {"Kp": 2.0}
            self.pid_Z = {"Kp": 5.0}
            self.pid_step = lambda pid, err: float(err)

        def run_shot(self, shot_duration: int, save_plot: bool = False) -> dict[str, float]:
            """Step each loop once and return fixed errors."""
            _ = self.pid_step(self.pid_R, 0.2)
            _ = self.pid_step(self.pid_Z, -0.1)
            return {
                "steps": float(shot_duration),
                "mean_abs_r_error": 0.1,
                "mean_abs_z_error": 0.1,
                "mean_abs_radial_actuator_lag": 0.0,
            }

    def fake_get_hinf() -> FakeHinf:
        """Build and remember one stand-in controller."""
        ctrl = FakeHinf()
        created.append(ctrl)
        return ctrl

    monkeypatch.setattr(mod, "IsoFluxController", FakeIsoFluxController)
    monkeypatch.setattr(mod, "get_radial_robust_controller", fake_get_hinf)
    monkeypatch.setenv(mod.HINF_RESEARCH_ENV, "1")

    episode = mod._run_hinf_episode("unused.json", shot_duration=8)

    assert len(created) == 2
    assert all(ctrl.reset_calls == 1 for ctrl in created)
    assert created[0].steps[0] == (0.0, 0.05)
    assert created[1].steps[0] == (0.0, 0.05)
    assert any(abs(err - 0.2) < 1e-12 for err, _ in created[0].steps[1:])
    assert any(abs(err + 0.1) < 1e-12 for err, _ in created[1].steps[1:])
    assert episode.t_disruption == 8.0


def test_hinf_episode_requires_explicit_research_gate(monkeypatch) -> None:
    """The H-infinity lane refuses to run without its research gate."""
    import validation.controller_comparison as mod

    monkeypatch.delenv(mod.HINF_RESEARCH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="disabled by default"):
        mod._run_hinf_episode("unused.json", shot_duration=8)


def test_controller_registry_contains_baseline_controllers() -> None:
    """The registry always offers PID and H-infinity."""
    import validation.controller_comparison as mod

    registry = mod._build_controller_registry()
    assert "PID" in registry
    assert "H-infinity" in registry


def test_run_comparison_aggregates_metrics(monkeypatch) -> None:
    """Aggregates over successful episodes match the per-episode values."""
    import validation.controller_comparison as mod

    def run_stable(config_path: Any, shot_duration: int) -> mod.EpisodeResult:
        """Return a stable episode."""
        return mod.EpisodeResult(
            mean_abs_r_error=0.1,
            mean_abs_z_error=0.2,
            reward=-0.3,
            latency_us=12.0,
            disrupted=False,
            t_disruption=float(shot_duration),
            energy_efficiency=0.9,
        )

    def run_disrupted(config_path: Any, shot_duration: int) -> mod.EpisodeResult:
        """Return a disrupted episode at half the shot."""
        return mod.EpisodeResult(
            mean_abs_r_error=0.8,
            mean_abs_z_error=0.8,
            reward=-1.6,
            latency_us=30.0,
            disrupted=True,
            t_disruption=float(shot_duration) * 0.5,
            energy_efficiency=0.7,
        )

    monkeypatch.setattr(
        mod,
        "CONTROLLERS",
        {"StableCtrl": run_stable, "DisruptedCtrl": run_disrupted},
    )

    results = mod.run_comparison(
        n_episodes=4,
        shot_duration=20,
        config_path="unused.json",
    )

    assert results["StableCtrl"].n_episodes == 4
    assert results["StableCtrl"].disruption_rate == 0.0
    assert results["StableCtrl"].mean_def == 1.0
    assert results["DisruptedCtrl"].n_episodes == 4
    assert results["DisruptedCtrl"].disruption_rate == 1.0
    assert results["DisruptedCtrl"].mean_def == 0.5


def test_failed_episodes_are_counted_and_saved(monkeypatch, tmp_path: Path) -> None:
    """A failed episode is counted, named and written out instead of silently dropped."""
    import json

    import validation.controller_comparison as mod

    calls: list[int] = []

    def run_flaky(config_path: Any, shot_duration: int) -> mod.EpisodeResult:
        """Fail on the second of three episodes."""
        calls.append(len(calls))
        if len(calls) == 2:
            raise FloatingPointError("state became nonfinite")
        return mod.EpisodeResult(
            mean_abs_r_error=0.1,
            mean_abs_z_error=0.2,
            reward=-0.3,
            latency_us=12.0,
            disrupted=False,
            t_disruption=float(shot_duration),
            energy_efficiency=0.9,
        )

    monkeypatch.setattr(mod, "CONTROLLERS", {"FlakyCtrl": run_flaky})

    results = mod.run_comparison(n_episodes=3, shot_duration=20, config_path="unused.json")
    output = tmp_path / "comparison.json"
    mod.save_results_json(results, output)

    metrics = results["FlakyCtrl"]
    assert metrics.n_episodes == 2
    assert metrics.failed_episodes == 1
    assert metrics.failures == ["episode 1: FloatingPointError: state became nonfinite"]
    saved = json.loads(output.read_text(encoding="utf-8"))["controllers"]["FlakyCtrl"]
    assert saved["failed_episodes"] == 1
    assert saved["failures"] == metrics.failures
    table = mod.generate_comparison_table(results)
    assert "| Failed |" in table
    assert "|        2 |      1 |" in table
    assert " & 2 & 1 & " in mod.generate_latex_table(results)


def test_run_comparison_reports_progress_through_its_logger(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Progress is logged at INFO and a failed episode at WARNING; nothing is printed."""
    import logging

    import validation.controller_comparison as mod

    def run_failing(config_path: Any, shot_duration: int) -> mod.EpisodeResult:
        """Fail every episode."""
        raise RuntimeError("plant unavailable")

    monkeypatch.setattr(mod, "CONTROLLERS", {"FailingCtrl": run_failing})

    with caplog.at_level(logging.INFO, logger=mod.logger.name):
        mod.run_comparison(n_episodes=1, shot_duration=20, config_path="unused.json")

    assert capsys.readouterr().out == ""
    messages = [(record.levelno, record.getMessage()) for record in caplog.records]
    assert (logging.INFO, "=== 4-Way Controller Comparison ===") in messages
    assert (logging.INFO, "--- Running FailingCtrl (1 episodes) ---") in messages
    assert (logging.WARNING, "  Episode 0 failed: plant unavailable") in messages
