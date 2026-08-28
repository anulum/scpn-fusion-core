# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX-Hybrid Realtime Control Benchmark Tests
"""Real-surface tests for the TORAX-hybrid realtime control benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scpn_fusion.control.torax_hybrid_loop import (
    CONTROL_ARTIFACT_NAME,
    run_nstxu_torax_hybrid_campaign,
)

pytestmark = pytest.mark.experimental


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = ROOT / "validation" / "torax_hybrid_realtime_control_benchmark.py"
VALIDATION_SPEC = importlib.util.spec_from_file_location(
    "torax_hybrid_realtime_control_benchmark", VALIDATION_PATH
)
assert VALIDATION_SPEC and VALIDATION_SPEC.loader
benchmark_cli = importlib.util.module_from_spec(VALIDATION_SPEC)
sys.modules[VALIDATION_SPEC.name] = benchmark_cli
VALIDATION_SPEC.loader.exec_module(benchmark_cli)

RUNTIME_PATH = ROOT / "run_realtime_simulation.py"
RUNTIME_SPEC = importlib.util.spec_from_file_location("run_realtime_simulation", RUNTIME_PATH)
assert RUNTIME_SPEC and RUNTIME_SPEC.loader
run_realtime_simulation = importlib.util.module_from_spec(RUNTIME_SPEC)
sys.modules[RUNTIME_SPEC.name] = run_realtime_simulation
RUNTIME_SPEC.loader.exec_module(run_realtime_simulation)


def test_campaign_meets_thresholds_smoke() -> None:
    out = run_nstxu_torax_hybrid_campaign(seed=42, episodes=8, steps_per_episode=160)
    repeated = run_nstxu_torax_hybrid_campaign(seed=42, episodes=8, steps_per_episode=160)
    assert out == repeated
    assert out.control_artifact_name == CONTROL_ARTIFACT_NAME
    assert out.disruption_avoidance_rate >= 0.90
    assert out.torax_parity_pct >= 95.0
    assert out.p95_loop_latency_ms <= 1.0
    assert out.passes_thresholds is True


def test_validation_report_contains_threshold_pass() -> None:
    report = benchmark_cli.generate_report(seed=7, episodes=6, steps_per_episode=140)
    benchmark = benchmark_cli.validate_report(report)
    text = benchmark_cli.render_markdown(report)
    assert report["schema_version"] == 2
    assert report["report_kind"] == "torax_hybrid_realtime_control_benchmark"
    assert benchmark["control_artifact_name"] == CONTROL_ARTIFACT_NAME
    assert benchmark["passes_thresholds"] is True
    assert "TORAX-Hybrid Realtime Control Benchmark" in text
    assert "TORAX parity" in text
    assert "Threshold pass: `YES`" in text


@pytest.mark.parametrize(
    "threshold_override",
    [
        {"min_disruption_avoidance_rate": 1.1},
        {"min_torax_parity_pct": 101.0},
        {"max_p95_loop_latency_ms": -1.0},
    ],
)
def test_report_threshold_failures_are_explicit(
    threshold_override: dict[str, float],
) -> None:
    report = benchmark_cli.generate_report(
        seed=9,
        episodes=3,
        steps_per_episode=96,
        **threshold_override,
    )
    assert benchmark_cli.validate_report(report)["passes_thresholds"] is False
    assert "Threshold pass: `NO`" in benchmark_cli.render_markdown(report)


def test_run_realtime_hybrid_smoke_function(capsys: pytest.CaptureFixture[str]) -> None:
    summary = run_realtime_simulation.run_torax_hybrid_smoke(
        seed=10, episodes=5, steps_per_episode=120
    )
    assert summary["passes_thresholds"] is True
    assert summary["torax_parity_pct"] >= 95.0
    output = capsys.readouterr().out
    assert "[TORAX-HYBRID]" in output
    assert "realtime control benchmark" in output.lower()


@pytest.mark.parametrize(
    ("episodes", "steps_per_episode", "match"),
    [
        (0, 220, "episodes"),
        (16, 16, "steps_per_episode"),
    ],
)
def test_campaign_rejects_invalid_runtime_inputs(
    episodes: int, steps_per_episode: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        run_nstxu_torax_hybrid_campaign(
            seed=42,
            episodes=episodes,
            steps_per_episode=steps_per_episode,
        )


def test_campaign_registers_disruption_on_sustained_high_risk() -> None:
    def sustained_high_risk(_history: list[float], _observations: dict[str, float]) -> float:
        return 0.99

    out = run_nstxu_torax_hybrid_campaign(
        seed=1,
        episodes=1,
        steps_per_episode=32,
        risk_predictor=sustained_high_risk,
    )
    assert out.disruption_avoidance_rate == 0.0
    assert out.passes_thresholds is False


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"generated_at_utc": None}, "generated_at_utc must be"),
        ({"runtime_seconds": -1.0}, "runtime_seconds must be"),
        ({"runtime_seconds": "slow"}, "runtime_seconds must be"),
        ({"torax_hybrid_realtime_control_benchmark": []}, "must be an object"),
    ],
)
def test_report_contract_rejects_invalid_current_payloads(
    change: dict[str, Any], message: str
) -> None:
    report = benchmark_cli.generate_report(seed=11, episodes=2, steps_per_episode=64)
    report.update(change)
    with pytest.raises(ValueError, match=message):
        benchmark_cli.validate_report(report)


def test_report_contract_rejects_obsolete_coded_payload_exactly() -> None:
    stale_report = {
        "generated_at_utc": "2026-08-26T00:00:00+00:00",
        "runtime_seconds": 1.0,
        "gai_02": {"passes_thresholds": True},
    }
    with pytest.raises(ValueError, match="current descriptive contract"):
        benchmark_cli.validate_report(stale_report)


def test_cli_writes_current_contract_and_enforces_strict_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_json = tmp_path / "benchmark.json"
    output_md = tmp_path / "benchmark.md"
    common_args = [
        "--seed",
        "13",
        "--episodes",
        "2",
        "--steps-per-episode",
        "64",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]

    assert benchmark_cli.main(common_args) == 0
    assert benchmark_cli.main([*common_args, "--strict"]) == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    benchmark_cli.validate_report(written)
    assert output_md.read_text(encoding="utf-8").startswith(
        "# TORAX-Hybrid Realtime Control Benchmark"
    )
    assert "benchmark complete" in capsys.readouterr().out

    assert (
        benchmark_cli.main(
            [
                *common_args,
                "--strict",
                "--min-disruption-avoidance-rate",
                "1.1",
            ]
        )
        == 2
    )
