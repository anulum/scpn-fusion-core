# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 2 Benchmark Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for Task 2 surrogate + latency benchmark validation lane."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "task2_pretrained_surrogates_benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "task2_pretrained_surrogates_benchmark", MODULE_PATH
)
assert SPEC and SPEC.loader
task2_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task2_bench)


def test_task2_campaign_passes_thresholds() -> None:
    out = task2_bench.run_campaign(
        seed=42,
        force_retrain=False,
        latency_backend="auto",
        latency_trials=48,
        latency_grid_size=56,
        latency_fault_runs=10,
    )
    assert out["passes_thresholds"] is True
    assert out["disruption_auc"]["tm1_proxy"]["auc"] >= 0.95
    assert out["disruption_auc"]["tokamaknet_proxy"]["auc"] >= 0.95
    assert out["equilibrium_latency"]["p95_ms_est"] < 1.0
    assert out["equilibrium_latency"]["fault_p95_ms_est"] < 1.0
    assert out["equilibrium_latency"]["fault_runs"] == 10.0
    assert isinstance(out["wall_latency_advisory_pass"], bool)
    assert out["thresholds"]["max_equilibrium_p95_ms_wall"] == 10.0
    assert (
        out["thresholds"]["max_equilibrium_p95_ms_wall"]
        == out["thresholds"]["max_equilibrium_p95_ms_wall_advisory"]
    )
    assert out["disruption_auc_publication"]["published"] is True
    assert out["surrogate_coverage"]["coverage_percent"] > 0.0
    assert len(out["surrogate_coverage"]["requires_user_training"]) >= 1
    assert len(out["consumer_latency_profiles"]["profiles"]) >= 3


def test_task2_auc_benchmark_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="profile"):
        task2_bench.run_disruption_auc_benchmark(profile="unknown", seed=1)
    with pytest.raises(ValueError, match="samples"):
        task2_bench.run_disruption_auc_benchmark(profile="tm1", seed=1, samples=0)
    with pytest.raises(ValueError, match="window"):
        task2_bench.run_disruption_auc_benchmark(profile="tokamaknet", seed=1, window=8)
    with pytest.raises(ValueError, match="label_flip_rate"):
        task2_bench.run_disruption_auc_benchmark(
            profile="tm1", seed=1, label_flip_rate=1.1
        )


def test_task2_markdown_contains_required_sections() -> None:
    report = task2_bench.generate_report(
        seed=42,
        force_retrain=False,
        latency_backend="auto",
        latency_trials=32,
        latency_grid_size=48,
        latency_fault_runs=6,
    )
    text = task2_bench.render_markdown(report)
    assert "# Task 2 Surrogate + Benchmark Report" in text
    assert "Disruption Predictor AUC" in text
    assert "Equilibrium Latency (10x Fault Runs)" in text
    assert "Consumer Hardware Latency Profiles" in text
    assert "Wall-latency advisory pass" in text
    assert "advisory `<= " in text
