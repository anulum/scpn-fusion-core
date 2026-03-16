# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Benchmark Fairness Payload Tests
"""Regression tests for timing/accuracy/conditions benchmark pairing."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure validation/ and src/ are importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))


def test_disturbance_benchmark_json_contains_fairness_triplets() -> None:
    import validation.benchmark_disturbance_rejection as mod

    metrics = [
        mod.ScenarioMetrics(
            controller="PID",
            scenario="VDE",
            ise=1.23,
            settling_time_s=0.45,
            peak_overshoot=0.12,
            control_effort=3.4,
            wall_clock_s=0.067,
            stable=True,
        )
    ]

    payload = mod.generate_json_results(metrics)
    assert payload["fairness_schema_version"] == "1.0"
    assert payload["fairness"]
    entry = payload["fairness"][0]
    assert "timing" in entry
    assert "accuracy" in entry
    assert "conditions" in entry
    assert "wall_clock_s" in entry["timing"]
    assert "ise" in entry["accuracy"]
    assert "dt_s" in entry["conditions"]


def test_solver_benchmark_summary_contains_fairness_triplets() -> None:
    import validation.benchmark_solvers as mod

    results = [
        {
            "method": "Rust sor",
            "shots": 10,
            "converged": 10,
            "mean_ms": 1.2,
            "median_ms": 1.1,
            "p95_ms": 1.6,
            "min_ms": 1.0,
            "max_ms": 2.0,
            "mean_residual": 1.0e-4,
            "times_ms": [1.0, 1.2, 1.4],
        }
    ]

    summary = mod.build_json_summary(results, config_path="iter_config.json")
    assert len(summary) == 1
    assert "times_ms" not in summary[0]
    fairness = summary[0]["fairness"]
    assert "timing" in fairness
    assert "accuracy" in fairness
    assert "conditions" in fairness
    assert fairness["conditions"]["config_path"] == "iter_config.json"
