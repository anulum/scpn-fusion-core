# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Controller Comparison Hardening Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for validation/controller_comparison.py hardening."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure validation/ is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))


def test_hinf_episode_uses_independent_axis_controllers(monkeypatch) -> None:
    import validation.controller_comparison as mod

    created: list[Any] = []

    class FakeHinf:
        def __init__(self) -> None:
            self._Fd = np.array([[2.0]], dtype=np.float64)
            self.steps: list[tuple[float, float]] = []
            self.reset_calls = 0

        def step(self, err: float, dt: float) -> float:
            self.steps.append((float(err), float(dt)))
            return float(err) * 10.0

        def reset(self) -> None:
            self.reset_calls += 1

    class FakeIsoFluxController:
        def __init__(
            self,
            config_path: str,
            verbose: bool = False,
            control_dt_s: float = 0.05,
        ) -> None:
            self.pid_R = {"Kp": 2.0}
            self.pid_Z = {"Kp": 5.0}
            self.pid_step = lambda pid, err: float(err)

        def run_shot(self, shot_duration: int, save_plot: bool = False) -> dict[str, float]:
            _ = self.pid_step(self.pid_R, 0.2)
            _ = self.pid_step(self.pid_Z, -0.1)
            return {
                "steps": float(shot_duration),
                "mean_abs_r_error": 0.1,
                "mean_abs_z_error": 0.1,
                "mean_abs_radial_actuator_lag": 0.0,
            }

    def fake_get_hinf() -> FakeHinf:
        ctrl = FakeHinf()
        created.append(ctrl)
        return ctrl

    monkeypatch.setattr(mod, "IsoFluxController", FakeIsoFluxController)
    monkeypatch.setattr(mod, "get_radial_robust_controller", fake_get_hinf)

    episode = mod._run_hinf_episode("unused.json", shot_duration=8)

    assert len(created) == 2
    assert all(ctrl.reset_calls == 1 for ctrl in created)
    assert created[0].steps[0] == (0.0, 0.05)
    assert created[1].steps[0] == (0.0, 0.05)
    assert any(abs(err - 0.2) < 1e-12 for err, _ in created[0].steps[1:])
    assert any(abs(err + 0.1) < 1e-12 for err, _ in created[1].steps[1:])
    assert episode.t_disruption == 8.0


def test_controller_registry_contains_baseline_controllers() -> None:
    import validation.controller_comparison as mod

    registry = mod._build_controller_registry()
    assert "PID" in registry
    assert "H-infinity" in registry


def test_run_comparison_aggregates_metrics(monkeypatch) -> None:
    import validation.controller_comparison as mod

    def run_stable(config_path: Any, shot_duration: int) -> mod.EpisodeResult:
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
