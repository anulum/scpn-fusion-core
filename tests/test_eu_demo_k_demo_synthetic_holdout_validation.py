# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — EU-DEMO/K-DEMO Synthetic Holdout Validation Tests
"""Real-file and CLI tests for EU-DEMO/K-DEMO synthetic holdout validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "validation" / "reference_data" / "blind"
MODULE_PATH = ROOT / "validation" / "eu_demo_k_demo_synthetic_holdout_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "eu_demo_k_demo_synthetic_holdout_validation",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
validation_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation_cli
SPEC.loader.exec_module(validation_cli)


def _write_reference_pair(directory: Path, shots: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, machine in zip(validation_cli.BLIND_REFERENCE_FILES, ("EU-DEMO", "K-DEMO")):
        payload = {"machine": machine, "shots": shots}
        (directory / filename).write_text(json.dumps(payload), encoding="utf-8")


def _extreme_reference_row() -> dict[str, Any]:
    return {
        "shot": "SYNTHETIC-EXTREME",
        "I_p_MA": 1.0,
        "B_t_T": 1.0,
        "n_e_1e19": 1.0,
        "P_loss_MW": 1.0,
        "R_m": 1.0,
        "a_m": 0.2,
        "kappa": 1.0,
        "A_eff_amu": 2.0,
        "tau_E_s": 1.0e6,
        "beta_N": 1.0e6,
        "core_edge_match": 1.0e6,
    }


def test_reference_loader_contains_expected_synthetic_machines() -> None:
    rows = validation_cli.load_blind_references(REFERENCE_DIR)

    assert {row["machine"] for row in rows} == {"EU-DEMO", "K-DEMO"}
    assert len(rows) == 10
    assert {row["shot"] for row in rows} >= {"EU-4101", "KD-5101"}


def test_dashboard_function_loader_uses_real_module_and_rejects_unknown_format(
    tmp_path: Path,
) -> None:
    confinement_model, metric = validation_cli.load_rmse_dashboard_functions(
        ROOT / "validation" / "rmse_dashboard.py"
    )

    assert callable(confinement_model)
    assert metric([1.0], [1.0]) == 0.0
    with pytest.raises(RuntimeError, match="Unable to load RMSE dashboard"):
        validation_cli.load_rmse_dashboard_functions(tmp_path / "no_extension")


def test_reference_loader_fails_when_a_required_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="eu_demo_reference.json"):
        validation_cli.load_blind_references(tmp_path)


def test_reference_loader_fails_when_both_files_have_no_rows(tmp_path: Path) -> None:
    _write_reference_pair(tmp_path, [])

    with pytest.raises(ValueError, match="No blind reference rows"):
        validation_cli.load_blind_references(tmp_path)


def test_public_proxy_functions_return_finite_domain_values() -> None:
    row = validation_cli.load_blind_references(REFERENCE_DIR)[0]
    beta_n = validation_cli.estimate_beta_n_proxy(row, tau_pred_s=5.0)
    core_edge = validation_cli.estimate_core_edge_match_proxy(5.0, beta_n)

    assert beta_n > 0.0
    assert 0.83 <= core_edge <= 0.97
    assert validation_cli.estimate_core_edge_match_proxy(-1.0e6, -1.0e6) == pytest.approx(0.83)
    assert validation_cli.estimate_core_edge_match_proxy(1.0e6, 1.0e6) == pytest.approx(0.97)


def test_campaign_passes_default_thresholds() -> None:
    result = validation_cli.run_campaign()

    assert result["passes_thresholds"] is True
    assert result["sample_count"] == 10
    assert result["aggregate"]["parity_pct"] >= result["thresholds"]["min_parity_pct"]
    assert [machine["machine"] for machine in result["machines"]] == ["EU-DEMO", "K-DEMO"]
    assert all(machine["passes_thresholds"] for machine in result["machines"])


@pytest.mark.parametrize(
    "threshold",
    [
        {"max_tau_rmse_s": 0.0},
        {"max_beta_rmse": 0.0},
        {"max_core_edge_rmse": 0.0},
        {"min_parity_pct": 100.0},
    ],
)
def test_campaign_can_fail_each_public_acceptance_gate(threshold: dict[str, float]) -> None:
    result = validation_cli.run_campaign(reference_dir=REFERENCE_DIR, **threshold)

    assert result["passes_thresholds"] is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_tau_rmse_s": -0.1}, "max_tau_rmse_s"),
        ({"max_tau_rmse_s": float("nan")}, "max_tau_rmse_s"),
        ({"max_beta_rmse": -0.1}, "max_beta_rmse"),
        ({"max_beta_rmse": float("inf")}, "max_beta_rmse"),
        ({"max_core_edge_rmse": -0.1}, "max_core_edge_rmse"),
        ({"max_core_edge_rmse": float("nan")}, "max_core_edge_rmse"),
        ({"min_parity_pct": -0.1}, "min_parity_pct"),
        ({"min_parity_pct": 100.1}, "min_parity_pct"),
        ({"min_parity_pct": float("inf")}, "min_parity_pct"),
    ],
)
def test_campaign_rejects_invalid_thresholds(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validation_cli.run_campaign(reference_dir=REFERENCE_DIR, **kwargs)


def test_campaign_parity_score_floors_at_zero_for_extreme_reference(tmp_path: Path) -> None:
    _write_reference_pair(tmp_path, [_extreme_reference_row()])

    result = validation_cli.run_campaign(reference_dir=tmp_path)

    assert result["aggregate"]["parity_pct"] == 0.0
    assert result["passes_thresholds"] is False


def test_report_contract_and_markdown_are_descriptive() -> None:
    report = validation_cli.generate_report(reference_dir=REFERENCE_DIR)
    payload = validation_cli.validate_report(report)
    markdown = validation_cli.render_markdown(report)

    assert report["schema_version"] == 2
    assert report["report_kind"] == "eu_demo_k_demo_synthetic_holdout_validation"
    assert payload["passes_thresholds"] is True
    assert markdown.startswith("# EU-DEMO/K-DEMO Synthetic Holdout Validation")
    assert "## Aggregate Metrics" in markdown
    assert "## EU-DEMO" in markdown
    assert "## K-DEMO" in markdown
    assert "Overall pass" in markdown


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"generated_at_utc": None}, "generated_at_utc must be"),
        ({"eu_demo_k_demo_synthetic_holdout_validation": []}, "must be an object"),
    ],
)
def test_report_contract_rejects_invalid_current_payloads(
    change: dict[str, Any], message: str
) -> None:
    report = validation_cli.generate_report(reference_dir=REFERENCE_DIR)
    report.update(change)
    with pytest.raises(ValueError, match=message):
        validation_cli.validate_report(report)


def test_report_contract_rejects_obsolete_coded_payload_exactly() -> None:
    stale_report = {
        "generated_at_utc": "2026-08-26T00:00:00+00:00",
        "gdep_03": {"passes_thresholds": True},
    }
    with pytest.raises(ValueError, match="current descriptive contract"):
        validation_cli.validate_report(stale_report)


def test_cli_writes_current_contract_and_enforces_strict_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_json = tmp_path / "holdout-validation.json"
    output_md = tmp_path / "holdout-validation.md"
    common_args = [
        "--reference-dir",
        str(REFERENCE_DIR),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]

    parsed = validation_cli.parse_args(common_args)
    assert parsed.reference_dir == str(REFERENCE_DIR)
    assert validation_cli.main([*common_args, "--strict"]) == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    validation_cli.validate_report(written)
    assert output_md.read_text(encoding="utf-8").startswith(
        "# EU-DEMO/K-DEMO Synthetic Holdout Validation"
    )
    assert "synthetic holdout validation complete" in capsys.readouterr().out

    assert (
        validation_cli.main(
            [
                *common_args,
                "--strict",
                "--max-tau-rmse-s",
                "0.0",
            ]
        )
        == 2
    )


def test_script_entry_point_runs_the_real_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_json = tmp_path / "script-report.json"
    output_md = tmp_path / "script-report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--reference-dir",
            str(REFERENCE_DIR),
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
