# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Full Validation Pipeline Tests
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for validation/full_validation_pipeline.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "full_validation_pipeline.py"
SPEC = importlib.util.spec_from_file_location("full_validation_pipeline", MODULE_PATH)
assert SPEC and SPEC.loader
full_validation_pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = full_validation_pipeline
SPEC.loader.exec_module(full_validation_pipeline)


def test_campaign_runs_100_plus_disruption_scenarios() -> None:
    report = full_validation_pipeline.run_campaign(
        seed=21,
        scenario_count=120,
        controllers=("mpc", "rl"),
        prefer_live_archives=False,
    )
    assert report["disruption_scenarios_evaluated"] == 120
    assert set(report["controller_metrics"].keys()) == {"mpc", "rl"}
    assert report["controller_metrics"]["mpc"]["mean_psi_rmse_pct"] < 5.0
    assert report["controller_metrics"]["mpc"]["mean_tau_rmse_pct"] < 5.0
    assert report["passes_target"] is True


def test_rewrite_gate_triggers_for_large_error_scales() -> None:
    report = full_validation_pipeline.run_campaign(
        seed=22,
        scenario_count=120,
        controllers=("mpc", "rl"),
        prefer_live_archives=False,
        mpc_error_scale=4.2,
        rl_error_scale=4.8,
        high_beta_penalty_scale=4.0,
        synthetic_high_beta_fraction=0.9,
    )
    assert report["rewrite_required"] is True
    assert report["fno_retrain_plan"]["recommended"] is True
    assert any("retrain FNO" in item for item in report["recommendations"])


def test_high_beta_pivot_triggers_for_divergent_lane() -> None:
    report = full_validation_pipeline.run_campaign(
        seed=23,
        scenario_count=120,
        controllers=("rl",),
        prefer_live_archives=False,
        rl_error_scale=5.5,
        high_beta_penalty_scale=7.0,
        synthetic_high_beta_fraction=1.0,
        high_beta_threshold=2.0,
    )
    assert report["pivot_to_hybrid_2d"] is True
    assert report["controller_metrics"]["rl"]["high_beta_divergence_rate"] >= 0.25


def test_render_markdown_contains_key_sections() -> None:
    report = full_validation_pipeline.run_campaign(
        seed=24,
        scenario_count=100,
        controllers=("mpc",),
        prefer_live_archives=False,
    )
    text = full_validation_pipeline.render_markdown(report)
    assert "Full Empirical Validation Pipeline" in text
    assert "Controller Metrics" in text
    assert "Rewrite reduced-order model" in text


def test_campaign_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(8080)
    state = np.random.get_state()
    _ = full_validation_pipeline.run_campaign(
        seed=25,
        scenario_count=100,
        controllers=("mpc", "rl"),
        prefer_live_archives=False,
    )
    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected
