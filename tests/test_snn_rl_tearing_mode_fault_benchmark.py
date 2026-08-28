# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — SNN/RL Tearing-Mode Fault Benchmark Tests
"""Public-surface tests for the SNN/RL tearing-mode fault benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "snn_rl_tearing_mode_fault_benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "snn_rl_tearing_mode_fault_benchmark",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
benchmark_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_cli
SPEC.loader.exec_module(benchmark_cli)


def _current_report(payload: Any = None) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "report_kind": "snn_rl_tearing_mode_fault_benchmark",
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
        "runtime_seconds": 0.1,
        "snn_rl_tearing_mode_fault_benchmark": (
            {"passes_thresholds": True} if payload is None else payload
        ),
    }


def test_benchmark_is_deterministic_for_seed() -> None:
    first = benchmark_cli.run_benchmark(seed=17, episodes=3, window=32)
    second = benchmark_cli.run_benchmark(seed=17, episodes=3, window=32)

    assert first == second


def test_benchmark_meets_default_thresholds() -> None:
    result = benchmark_cli.run_benchmark(seed=42, episodes=4, window=40)

    assert result["decision_agreement"] >= 0.95
    assert result["mean_abs_delta"] <= 0.08
    assert result["stochastic_float_equivalence_error"] <= 0.05
    assert result["stochastic_float_equivalence_error_pct"] <= 5.0
    assert result["oracle_sc_mean_abs_delta"] <= 0.05
    assert result["oracle_sc_firing_mean_abs_delta"] <= 0.05
    assert result["recovery_ms_p95"] <= 1.0
    assert all(result["gate_results"].values())
    assert result["passes_thresholds"] is True


def test_controller_has_descriptive_artifact_and_adaptive_margin() -> None:
    controller = benchmark_cli.build_tearing_mode_fault_controller()

    assert controller.artifact.meta.name == "snn_rl_tearing_mode_fault_controller"
    assert getattr(controller, "_sc_binary_margin", 0.0) > 0.0


@pytest.mark.parametrize(
    "threshold",
    [
        {"max_mean_abs_delta": 0.0},
        {"max_stochastic_float_equivalence_error": 0.0},
        {"max_oracle_sc_mean_abs_delta": 0.0},
        {"max_oracle_sc_firing_delta": 0.0},
        {"recovery_epsilon": 1e-16, "max_recovery_ms_p95": 0.0},
    ],
)
def test_benchmark_can_fail_public_metric_gates(threshold: dict[str, float]) -> None:
    result = benchmark_cli.run_benchmark(
        seed=42,
        episodes=2,
        window=40,
        **threshold,
    )

    assert result["passes_thresholds"] is False
    assert not all(result["gate_results"].values())


def test_benchmark_records_unrecovered_fault_window() -> None:
    result = benchmark_cli.run_benchmark(
        seed=0,
        episodes=1,
        window=16,
        recovery_epsilon=1e-16,
        recovery_window_steps=1,
        max_recovery_ms_p95=0.0,
    )

    assert result["recovery_steps_p95"] == 2.0
    assert result["gate_results"]["recovery_ms_p95"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"episodes": 0}, "episodes"),
        ({"window": 8}, "window"),
        ({"recovery_window_steps": 0}, "recovery_window_steps"),
        ({"recovery_epsilon": 0.0}, "recovery_epsilon"),
        ({"recovery_epsilon": float("nan")}, "recovery_epsilon"),
        ({"dt_ms": 0.0}, "dt_ms"),
        ({"dt_ms": float("inf")}, "dt_ms"),
        ({"min_decision_agreement": float("nan")}, "min_decision_agreement"),
        ({"min_decision_agreement": -0.1}, "min_decision_agreement"),
        ({"min_decision_agreement": 1.1}, "min_decision_agreement"),
        ({"max_mean_abs_delta": -0.1}, "max_mean_abs_delta"),
        ({"max_mean_abs_delta": 1.1}, "max_mean_abs_delta"),
        (
            {"max_stochastic_float_equivalence_error": -0.1},
            "max_stochastic_float_equivalence_error",
        ),
        ({"max_oracle_sc_mean_abs_delta": 1.1}, "max_oracle_sc_mean_abs_delta"),
        ({"max_oracle_sc_firing_delta": -0.1}, "max_oracle_sc_firing_delta"),
        ({"max_recovery_ms_p95": -0.1}, "max_recovery_ms_p95"),
        ({"max_recovery_ms_p95": 1_001.0}, "max_recovery_ms_p95"),
    ],
)
def test_benchmark_rejects_invalid_public_inputs(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark_cli.run_benchmark(**kwargs)


def test_report_contract_and_markdown_are_descriptive() -> None:
    report = benchmark_cli.generate_report(seed=5, episodes=2, window=24)
    payload = benchmark_cli.validate_report(report)
    markdown = benchmark_cli.render_markdown(report)

    assert report["schema_version"] == 2
    assert report["report_kind"] == "snn_rl_tearing_mode_fault_benchmark"
    assert payload["passes_thresholds"] is True
    assert markdown.startswith("# SNN/RL Tearing-Mode Fault Benchmark")
    assert "SNN/RL decision agreement" in markdown
    assert "TORAX" not in markdown
    assert "Threshold pass: `YES`" in markdown


def test_markdown_reports_failed_public_threshold() -> None:
    report = benchmark_cli.generate_report(
        seed=5,
        episodes=2,
        window=24,
        max_mean_abs_delta=0.0,
    )

    assert "Threshold pass: `NO`" in benchmark_cli.render_markdown(report)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": None}, "generated_at_utc must be"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"runtime_seconds": "slow"}, "runtime_seconds must be"),
        ({"runtime_seconds": float("nan")}, "runtime_seconds must be"),
        ({"runtime_seconds": -0.1}, "runtime_seconds must be"),
        ({"snn_rl_tearing_mode_fault_benchmark": []}, "must be an object"),
        ({"extra": True}, "current descriptive contract"),
    ],
)
def test_report_contract_rejects_invalid_current_payloads(
    change: dict[str, Any], message: str
) -> None:
    report = _current_report()
    report.update(change)

    with pytest.raises(ValueError, match=message):
        benchmark_cli.validate_report(report)


def test_report_contract_rejects_obsolete_coded_payload_exactly() -> None:
    stale_report = {
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
        "runtime_seconds": 0.1,
        "gneu_01": {"passes_thresholds": True},
    }

    with pytest.raises(ValueError, match="current descriptive contract"):
        benchmark_cli.validate_report(stale_report)


def test_parse_args_uses_descriptive_default_artifact_names() -> None:
    args = benchmark_cli.parse_args([])

    assert args.output_json.endswith("snn_rl_tearing_mode_fault_benchmark.json")
    assert args.output_md.endswith("snn_rl_tearing_mode_fault_benchmark.md")


def test_cli_writes_current_contract_and_enforces_strict_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_json = tmp_path / "fault-benchmark.json"
    output_md = tmp_path / "fault-benchmark.md"
    common_args = [
        "--seed",
        "3",
        "--episodes",
        "2",
        "--window",
        "24",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]

    assert benchmark_cli.main([*common_args, "--strict"]) == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    benchmark_cli.validate_report(written)
    assert output_md.read_text(encoding="utf-8").startswith("# SNN/RL Tearing-Mode Fault Benchmark")
    assert "tearing-mode fault benchmark complete" in capsys.readouterr().out

    failed_args = [*common_args, "--max-mean-abs-delta", "0.0"]
    assert benchmark_cli.main([*failed_args, "--strict"]) == 2
    assert benchmark_cli.main(failed_args) == 0


def test_script_entry_point_runs_real_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_json = tmp_path / "script-report.json"
    output_md = tmp_path / "script-report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--episodes",
            "2",
            "--window",
            "24",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--strict",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert output_json.is_file()
    assert output_md.is_file()
