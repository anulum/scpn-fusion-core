# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Ice-Pellet Density-Control Validation Tests
"""Public-surface tests for ice-pellet density control and validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest

from scpn_fusion.control.fueling_mode import (
    IcePelletFuelingController,
    build_ice_pellet_fueling_controller,
    run_fueling_mode,
    simulate_iter_density_control,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "ice_pellet_density_control_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "ice_pellet_density_control_validation",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
validation_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation_cli
SPEC.loader.exec_module(validation_cli)


def _current_report(payload: Any = None) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "report_kind": "ice_pellet_density_control_validation",
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
        "runtime_seconds": 0.1,
        "ice_pellet_density_control_validation": (
            {"passes_thresholds": True} if payload is None else payload
        ),
    }


def test_density_control_is_deterministic_and_meets_target() -> None:
    first = simulate_iter_density_control(steps=3000)
    second = simulate_iter_density_control(steps=3000)

    assert first == second
    assert first.final_abs_error <= 1e-3
    assert len(first.history_density) == 3000
    assert len(first.history_command) == 3000


def test_public_controller_builder_is_descriptive() -> None:
    controller = build_ice_pellet_fueling_controller()

    assert controller.artifact.meta.name == "ice_pellet_fueling_controller"
    assert getattr(controller, "_sc_binary_margin", 0.0) > 0.0


def test_controller_step_returns_bounded_command_and_error() -> None:
    controller = IcePelletFuelingController(target_density=1.0)
    positive_command, positive_error = controller.step(0.0, 0, 1.0)
    negative_command, negative_error = controller.step(2.0, 1, 1.0)

    assert -2.0 <= positive_command <= 2.0
    assert positive_error == 1.0
    assert -2.0 <= negative_command <= 2.0
    assert negative_error == -1.0


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan")])
def test_controller_rejects_invalid_target_density(target: float) -> None:
    with pytest.raises(ValueError, match="target_density"):
        IcePelletFuelingController(target)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 7}, "steps"),
        ({"target_density": 0.0}, "target_density"),
        ({"target_density": float("nan")}, "target_density"),
        ({"initial_density": -0.1}, "initial_density"),
        ({"initial_density": float("nan")}, "initial_density"),
        ({"dt_s": 0.0}, "dt_s"),
        ({"dt_s": float("inf")}, "dt_s"),
        ({"dt_s": 1e-6}, "dt_s"),
    ],
)
def test_simulation_rejects_invalid_inputs(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_iter_density_control(**kwargs)


def test_large_timestep_keeps_density_non_negative() -> None:
    result = simulate_iter_density_control(steps=8, dt_s=2.0)

    assert min(result.history_density) >= 0.0


def test_fueling_mode_summary_uses_real_traces() -> None:
    summary = run_fueling_mode(steps=24, dt_s=0.01)

    assert summary["steps"] == 24
    assert summary["max_abs_command"] > 0.0
    assert summary["min_density"] >= 0.0
    assert summary["max_density"] >= summary["min_density"]
    assert isinstance(summary["passes_thresholds"], bool)


def test_validation_passes_and_can_fail_public_threshold() -> None:
    passed = validation_cli.run_validation(steps=3000)
    failed = validation_cli.run_validation(steps=3000, max_final_abs_error=0.0)

    assert passed["passes_thresholds"] is True
    assert failed["passes_thresholds"] is False


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), -0.1],
)
def test_validation_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="max_final_abs_error"):
        validation_cli.run_validation(max_final_abs_error=threshold)


def test_report_and_markdown_use_descriptive_schema() -> None:
    report = validation_cli.generate_report(steps=3000)
    payload = validation_cli.validate_report(report)
    markdown = validation_cli.render_markdown(report)

    assert report["schema_version"] == 2
    assert report["report_kind"] == "ice_pellet_density_control_validation"
    assert payload["passes_thresholds"] is True
    assert markdown.startswith("# Ice-Pellet Density-Control Validation")
    assert "Overall pass: `YES`" in markdown


def test_markdown_reports_failed_threshold() -> None:
    report = validation_cli.generate_report(steps=3000, max_final_abs_error=0.0)

    assert "Overall pass: `NO`" in validation_cli.render_markdown(report)


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
        ({"ice_pellet_density_control_validation": []}, "must be an object"),
        ({"extra": True}, "current descriptive contract"),
    ],
)
def test_report_contract_rejects_invalid_payloads(change: dict[str, Any], message: str) -> None:
    report = _current_report()
    report.update(change)

    with pytest.raises(ValueError, match=message):
        validation_cli.validate_report(report)


def test_report_contract_rejects_obsolete_coded_payload_exactly() -> None:
    stale_report = {
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
        "runtime_seconds": 0.1,
        "gneu_03": {"passes_thresholds": True},
    }

    with pytest.raises(ValueError, match="current descriptive contract"):
        validation_cli.validate_report(stale_report)


def test_cli_defaults_and_strict_pass_fail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    defaults = validation_cli.parse_args([])
    assert defaults.output_json.endswith("ice_pellet_density_control_validation.json")
    assert defaults.output_md.endswith("ice_pellet_density_control_validation.md")

    output_json = tmp_path / "density.json"
    output_md = tmp_path / "density.md"
    common = [
        "--steps",
        "3000",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    assert validation_cli.main([*common, "--strict"]) == 0
    validation_cli.validate_report(json.loads(output_json.read_text(encoding="utf-8")))
    assert "density-control validation complete" in capsys.readouterr().out

    failed = [*common, "--max-final-abs-error", "0.0"]
    assert validation_cli.main([*failed, "--strict"]) == 2
    assert validation_cli.main(failed) == 0


def test_script_entry_point_runs_real_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_json = tmp_path / "script.json"
    output_md = tmp_path / "script.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--steps",
            "3000",
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
