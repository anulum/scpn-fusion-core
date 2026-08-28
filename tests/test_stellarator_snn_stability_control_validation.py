# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stellarator SNN Stability-Control Validation Tests
"""Public-surface tests for stellarator SNN stability-control validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "stellarator_snn_stability_control_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "stellarator_snn_stability_control_validation",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
validation_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation_cli
SPEC.loader.exec_module(validation_cli)


def _current_report(payload: Any = None) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "report_kind": "stellarator_snn_stability_control_validation",
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
        "stellarator_snn_stability_control_validation": (
            {"passes_thresholds": True} if payload is None else payload
        ),
    }


def test_campaign_passes_default_thresholds() -> None:
    result = validation_cli.run_campaign(iterations=6, parity_samples=120)

    assert result["passes_thresholds"] is True
    assert result["field_periods"] == 5
    assert result["final_instability_metric"] <= 0.025
    assert result["improvement_pct"] >= 30.0
    assert result["synthetic_reference_parity_pct"] >= 95.0
    assert 0.0 <= result["synthetic_reference_parity_pct"] <= 100.0


def test_campaign_is_deterministic_except_for_runtime() -> None:
    first = validation_cli.run_campaign(iterations=3, parity_samples=40)
    second = validation_cli.run_campaign(iterations=3, parity_samples=40)
    first.pop("runtime_seconds")
    second.pop("runtime_seconds")

    assert first == second


@pytest.mark.parametrize(
    "threshold",
    [
        {"max_final_instability_metric": 0.0},
        {"min_improvement_pct": 100.0},
        {"min_synthetic_reference_parity_pct": 100.0},
    ],
)
def test_campaign_can_fail_each_public_metric_gate(threshold: dict[str, float]) -> None:
    result = validation_cli.run_campaign(iterations=6, parity_samples=80, **threshold)

    assert result["passes_thresholds"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"iterations": True}, "iterations"),
        ({"iterations": 0}, "iterations"),
        ({"iterations": 1}, "iterations must be >= 2"),
        ({"parity_samples": True}, "parity_samples"),
        ({"parity_samples": 0}, "parity_samples"),
        (
            {"max_final_instability_metric": float("nan")},
            "max_final_instability_metric must be finite",
        ),
        ({"max_final_instability_metric": -0.1}, "max_final_instability_metric"),
        ({"min_improvement_pct": float("inf")}, "min_improvement_pct must be finite"),
        ({"min_improvement_pct": -0.1}, "min_improvement_pct"),
        ({"min_improvement_pct": 100.1}, "min_improvement_pct"),
        ({"min_synthetic_reference_parity_pct": -0.1}, "min_synthetic_reference_parity_pct"),
        ({"min_synthetic_reference_parity_pct": 100.1}, "min_synthetic_reference_parity_pct"),
    ],
)
def test_campaign_rejects_invalid_public_inputs(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validation_cli.run_campaign(**kwargs)


def test_report_contract_and_markdown_are_descriptive() -> None:
    report = validation_cli.generate_report(iterations=6, parity_samples=40)
    payload = validation_cli.validate_report(report)
    markdown = validation_cli.render_markdown(report)

    assert report["schema_version"] == 2
    assert report["report_kind"] == "stellarator_snn_stability_control_validation"
    assert payload["passes_thresholds"] is True
    assert markdown.startswith("# Stellarator SNN Stability-Control Validation")
    assert "## Reduced Stability Control" in markdown
    assert "## In-Repository Synthetic-Reference Parity" in markdown
    assert "Overall pass: `YES`" in markdown


def test_markdown_reports_failed_public_threshold() -> None:
    report = validation_cli.generate_report(
        iterations=3,
        parity_samples=40,
        max_final_instability_metric=0.0,
    )

    assert "Overall pass: `NO`" in validation_cli.render_markdown(report)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": None}, "generated_at_utc must be"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"stellarator_snn_stability_control_validation": []}, "must be an object"),
        ({"extra": True}, "current descriptive contract"),
    ],
)
def test_report_contract_rejects_invalid_current_payloads(
    change: dict[str, Any], message: str
) -> None:
    report = _current_report()
    report.update(change)

    with pytest.raises(ValueError, match=message):
        validation_cli.validate_report(report)


def test_report_contract_rejects_obsolete_coded_payload_exactly() -> None:
    stale_report = {
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
        "gmvr_03": {"passes_thresholds": True},
    }

    with pytest.raises(ValueError, match="current descriptive contract"):
        validation_cli.validate_report(stale_report)


def test_parse_args_uses_descriptive_default_artifact_names() -> None:
    args = validation_cli.parse_args([])

    assert args.output_json.endswith("stellarator_snn_stability_control_validation.json")
    assert args.output_md.endswith("stellarator_snn_stability_control_validation.md")


def test_cli_writes_current_contract_and_enforces_strict_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_json = tmp_path / "stellarator-validation.json"
    output_md = tmp_path / "stellarator-validation.md"
    common_args = [
        "--iterations",
        "6",
        "--parity-samples",
        "40",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]

    assert validation_cli.main([*common_args, "--strict"]) == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    validation_cli.validate_report(written)
    assert output_md.read_text(encoding="utf-8").startswith(
        "# Stellarator SNN Stability-Control Validation"
    )
    assert "stability-control validation complete" in capsys.readouterr().out

    assert (
        validation_cli.main([*common_args, "--strict", "--max-final-instability-metric", "0.0"])
        == 2
    )
    assert validation_cli.main([*common_args, "--max-final-instability-metric", "0.0"]) == 0


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
            "--iterations",
            "6",
            "--parity-samples",
            "40",
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
