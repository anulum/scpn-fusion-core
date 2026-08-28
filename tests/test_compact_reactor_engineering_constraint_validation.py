# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Compact-Reactor Engineering-Constraint Validation Tests
"""Public-surface tests for compact-reactor engineering-constraint validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "compact_reactor_engineering_constraint_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "compact_reactor_engineering_constraint_validation",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
validation_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation_cli
SPEC.loader.exec_module(validation_cli)


def test_campaign_is_deterministic_for_seed_and_uses_real_scanner() -> None:
    first = validation_cli.run_campaign(seed=42, scan_samples=160)
    second = validation_cli.run_campaign(seed=42, scan_samples=160)

    assert first["evaluated_designs"] == 160
    assert second["evaluated_designs"] == 160
    assert first["feasible_designs"] == second["feasible_designs"]
    assert first["best_design"] == second["best_design"]


def test_campaign_passes_declared_default_constraints() -> None:
    result = validation_cli.run_campaign(seed=42, scan_samples=200)

    assert result["passes_thresholds"] is True
    assert result["feasible_designs"] >= result["thresholds"]["min_feasible_designs"]
    best = result["best_design"]
    assert best is not None
    assert 1.2 <= best["radius_m"] <= 1.5
    assert best["fusion_gain"] > 5.0
    assert best["divertor_flux_mw_m2"] <= 45.0
    assert best["zeff_proxy"] <= 0.4
    assert best["hts_peak_field_t"] <= 21.0
    assert best["cost_proxy"] > 0.0


def test_campaign_can_fail_through_public_fusion_gain_threshold() -> None:
    result = validation_cli.run_campaign(
        seed=42,
        scan_samples=80,
        min_fusion_gain=1.0e12,
    )

    assert result["passes_thresholds"] is False
    assert result["feasible_designs"] == 0
    assert result["best_design"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": True}, "seed"),
        ({"scan_samples": True}, "scan_samples"),
        ({"scan_samples": 0}, "scan_samples"),
        ({"radius_min_m": float("nan")}, "radius_min_m must be finite"),
        ({"radius_min_m": 0.0}, "radius_min_m must be > 0"),
        ({"radius_max_m": float("inf")}, "radius_max_m must be finite"),
        ({"radius_min_m": 1.5, "radius_max_m": 1.5}, "radius_max_m"),
        ({"min_fusion_gain": 0.0}, "min_fusion_gain"),
        ({"min_feasible_designs": True}, "min_feasible_designs"),
        ({"min_feasible_designs": 0}, "min_feasible_designs"),
        ({"divertor_flux_cap_mw_m2": float("nan")}, "divertor_flux_cap_mw_m2"),
        ({"zeff_cap": 1.1}, "zeff_cap must be <= 1.0"),
        ({"hts_peak_cap_t": 0.0}, "hts_peak_cap_t"),
    ],
)
def test_campaign_rejects_invalid_public_inputs(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validation_cli.run_campaign(**kwargs)


def test_report_contract_and_markdown_are_descriptive() -> None:
    report = validation_cli.generate_report(seed=42, scan_samples=160)
    payload = validation_cli.validate_report(report)
    markdown = validation_cli.render_markdown(report)

    assert report["schema_version"] == 2
    assert report["report_kind"] == "compact_reactor_engineering_constraint_validation"
    assert payload["passes_thresholds"] is True
    assert markdown.startswith("# Compact-Reactor Engineering-Constraint Validation")
    assert "## Acceptance Thresholds" in markdown
    assert "## Best Feasible Synthetic Design" in markdown
    assert "Overall pass: `YES`" in markdown


def test_markdown_reports_no_best_design_for_failed_campaign() -> None:
    report = validation_cli.generate_report(
        seed=42,
        scan_samples=80,
        min_fusion_gain=1.0e12,
    )

    markdown = validation_cli.render_markdown(report)

    assert "None found in the configured scan" in markdown
    assert "Overall pass: `NO`" in markdown


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": None}, "generated_at_utc must be"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"compact_reactor_engineering_constraint_validation": []}, "must be an object"),
    ],
)
def test_report_contract_rejects_invalid_current_payloads(
    change: dict[str, Any], message: str
) -> None:
    report = validation_cli.generate_report(seed=42, scan_samples=80)
    report.update(change)

    with pytest.raises(ValueError, match=message):
        validation_cli.validate_report(report)


def test_report_contract_rejects_obsolete_coded_payload_exactly() -> None:
    stale_report = {
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
        "gmvr_01": {"passes_thresholds": True},
    }

    with pytest.raises(ValueError, match="current descriptive contract"):
        validation_cli.validate_report(stale_report)


def test_report_contract_rejects_extra_keys() -> None:
    report = validation_cli.generate_report(seed=42, scan_samples=80)
    report["extra"] = True

    with pytest.raises(ValueError, match="current descriptive contract"):
        validation_cli.validate_report(report)


def test_parse_args_uses_descriptive_default_artifact_names() -> None:
    args = validation_cli.parse_args([])

    assert args.output_json.endswith("compact_reactor_engineering_constraint_validation.json")
    assert args.output_md.endswith("compact_reactor_engineering_constraint_validation.md")


def test_cli_writes_current_contract_and_enforces_strict_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_json = tmp_path / "compact-reactor-validation.json"
    output_md = tmp_path / "compact-reactor-validation.md"
    common_args = [
        "--seed",
        "42",
        "--scan-samples",
        "120",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]

    assert validation_cli.main([*common_args, "--strict"]) == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    validation_cli.validate_report(written)
    assert output_md.read_text(encoding="utf-8").startswith(
        "# Compact-Reactor Engineering-Constraint Validation"
    )
    assert "engineering-constraint validation complete" in capsys.readouterr().out

    assert (
        validation_cli.main(
            [
                *common_args,
                "--strict",
                "--min-fusion-gain",
                "1000000000000",
            ]
        )
        == 2
    )

    assert (
        validation_cli.main(
            [
                *common_args,
                "--min-fusion-gain",
                "1000000000000",
            ]
        )
        == 0
    )


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
            "--scan-samples",
            "100",
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
