# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Liquid-Metal TEMHD Divertor Validation Tests
"""Public-surface tests for liquid-metal TEMHD divertor validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any, cast

import numpy as np
import pytest

from scpn_fusion.core.divertor_thermal_sim import DivertorLab


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "liquid_metal_temhd_divertor_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "liquid_metal_temhd_divertor_validation",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
validation_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation_cli
SPEC.loader.exec_module(validation_cli)


def _current_report(payload: Any = None) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "report_kind": "liquid_metal_temhd_divertor_validation",
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
        "liquid_metal_temhd_divertor_validation": (
            {"passes_thresholds": True} if payload is None else payload
        ),
    }


def test_velocity_dependent_temhd_terms_use_real_divertor_lab() -> None:
    lab = DivertorLab(P_sol_MW=35.0, R_major=1.4, B_pol=2.3)
    slow = lab.simulate_temhd_liquid_metal(flow_velocity_m_s=0.001, expansion_factor=40.0)
    fast = lab.simulate_temhd_liquid_metal(flow_velocity_m_s=10.0, expansion_factor=40.0)

    assert cast(float, fast["pressure_loss_pa"]) > cast(float, slow["pressure_loss_pa"])
    assert cast(float, fast["evaporation_rate_kg_m2_s"]) < cast(
        float, slow["evaporation_rate_kg_m2_s"]
    )
    assert slow["is_stable"] is True
    assert fast["is_stable"] is True


def test_campaign_passes_default_thresholds() -> None:
    result = validation_cli.run_campaign()

    assert result["passes_thresholds"] is True
    assert result["pressure_ratio_fast_to_slow"] >= 1000.0
    assert result["evaporation_ratio_fast_to_slow"] < 1.0
    assert result["toroidal_samples"] == 36
    assert result["toroidal_stability_rate"] >= 0.95
    assert result["toroidal_stability_index_min"] <= result["toroidal_stability_index_max"]


def test_campaign_is_deterministic_except_for_runtime() -> None:
    first = validation_cli.run_campaign(toroidal_samples=12)
    second = validation_cli.run_campaign(toroidal_samples=12)
    first.pop("runtime_seconds")
    second.pop("runtime_seconds")

    assert first == second


@pytest.mark.parametrize(
    "threshold",
    [
        {"min_pressure_ratio_fast_to_slow": 1.0e12},
        {"max_evap_ratio_fast_to_slow": 0.0},
        {"max_toroidal_stability_index": 0.1},
    ],
)
def test_campaign_can_fail_each_public_metric_gate(threshold: dict[str, float]) -> None:
    result = validation_cli.run_campaign(**threshold)

    assert result["passes_thresholds"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"slow_flow_velocity_m_s": float("nan")}, "slow_flow_velocity_m_s must be finite"),
        ({"slow_flow_velocity_m_s": 0.0}, "slow_flow_velocity_m_s must be > 0"),
        ({"fast_flow_velocity_m_s": float("inf")}, "fast_flow_velocity_m_s must be finite"),
        (
            {"slow_flow_velocity_m_s": 1.0, "fast_flow_velocity_m_s": 1.0},
            "fast_flow_velocity_m_s must exceed",
        ),
        ({"expansion_factor": 0.0}, "expansion_factor"),
        ({"toroidal_samples": True}, "toroidal_samples"),
        ({"toroidal_samples": 0}, "toroidal_samples"),
        (
            {"min_pressure_ratio_fast_to_slow": float("nan")},
            "min_pressure_ratio_fast_to_slow must be finite",
        ),
        ({"min_pressure_ratio_fast_to_slow": 0.0}, "min_pressure_ratio_fast_to_slow"),
        (
            {"max_evap_ratio_fast_to_slow": float("inf")},
            "max_evap_ratio_fast_to_slow must be finite",
        ),
        ({"max_evap_ratio_fast_to_slow": -0.1}, "max_evap_ratio_fast_to_slow"),
        ({"max_toroidal_stability_index": 0.0}, "max_toroidal_stability_index"),
        (
            {"min_toroidal_stability_rate": float("nan")},
            "min_toroidal_stability_rate must be finite",
        ),
        ({"min_toroidal_stability_rate": -0.1}, "min_toroidal_stability_rate"),
        ({"min_toroidal_stability_rate": 1.1}, "min_toroidal_stability_rate"),
    ],
)
def test_campaign_rejects_invalid_public_inputs(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validation_cli.run_campaign(**kwargs)


def test_report_contract_and_markdown_are_descriptive() -> None:
    report = validation_cli.generate_report(toroidal_samples=12)
    payload = validation_cli.validate_report(report)
    markdown = validation_cli.render_markdown(report)

    assert report["schema_version"] == 2
    assert report["report_kind"] == "liquid_metal_temhd_divertor_validation"
    assert payload["passes_thresholds"] is True
    assert markdown.startswith("# Liquid-Metal TEMHD Divertor Validation")
    assert "## Slow and Fast Flow" in markdown
    assert "## Synthetic 3D Toroidal Sweep" in markdown
    assert "Overall pass: `YES`" in markdown


def test_markdown_reports_failed_public_threshold() -> None:
    report = validation_cli.generate_report(
        toroidal_samples=12,
        min_pressure_ratio_fast_to_slow=1.0e12,
    )

    assert "Overall pass: `NO`" in validation_cli.render_markdown(report)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": None}, "generated_at_utc must be"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"liquid_metal_temhd_divertor_validation": []}, "must be an object"),
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
        "gmvr_02": {"passes_thresholds": True},
    }

    with pytest.raises(ValueError, match="current descriptive contract"):
        validation_cli.validate_report(stale_report)


def test_parse_args_uses_descriptive_default_artifact_names() -> None:
    args = validation_cli.parse_args([])

    assert args.output_json.endswith("liquid_metal_temhd_divertor_validation.json")
    assert args.output_md.endswith("liquid_metal_temhd_divertor_validation.md")


def test_cli_writes_current_contract_and_enforces_strict_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_json = tmp_path / "temhd-validation.json"
    output_md = tmp_path / "temhd-validation.md"
    common_args = [
        "--toroidal-samples",
        "12",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]

    assert validation_cli.main([*common_args, "--strict"]) == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    validation_cli.validate_report(written)
    assert output_md.read_text(encoding="utf-8").startswith(
        "# Liquid-Metal TEMHD Divertor Validation"
    )
    assert "TEMHD divertor validation complete" in capsys.readouterr().out

    assert (
        validation_cli.main(
            [
                *common_args,
                "--strict",
                "--min-pressure-ratio-fast-to-slow",
                "1000000000000",
            ]
        )
        == 2
    )
    assert (
        validation_cli.main(
            [
                *common_args,
                "--min-pressure-ratio-fast-to-slow",
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
            "--toroidal-samples",
            "12",
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


# Divertor relaxation-parameter convergence and validation


def test_relaxation_parameter_affects_convergence() -> None:
    """Different relaxation factors should produce different convergence paths."""
    lab1 = DivertorLab(P_sol_MW=50.0, R_major=2.1, B_pol=2.0)
    lab1.calculate_heat_load()
    t1, q1, _ = lab1.simulate_lithium_vapor(relaxation=0.5)

    lab2 = DivertorLab(P_sol_MW=50.0, R_major=2.1, B_pol=2.0)
    lab2.calculate_heat_load()
    t2, q2, _ = lab2.simulate_lithium_vapor(relaxation=0.9)

    assert np.isfinite(t1) and np.isfinite(t2)
    assert np.isfinite(q1) and np.isfinite(q2)
    assert abs(t1 - t2) < 100.0, "Large deviation suggests convergence issue"


def test_relaxation_rejects_invalid_values() -> None:
    """Relaxation outside (0, 1) raises ValueError."""
    lab = DivertorLab(P_sol_MW=50.0, R_major=2.1, B_pol=2.0)
    lab.calculate_heat_load()

    with pytest.raises(ValueError):
        lab.simulate_lithium_vapor(relaxation=0.0)
    with pytest.raises(ValueError):
        lab.simulate_lithium_vapor(relaxation=1.0)
    with pytest.raises(ValueError):
        lab.simulate_lithium_vapor(relaxation=-0.1)
    with pytest.raises(ValueError):
        lab.simulate_lithium_vapor(relaxation=1.5)
