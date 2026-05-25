# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/check_freegs_strict_artifact.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_freegs_strict_artifact.py"
SPEC = importlib.util.spec_from_file_location("check_freegs_strict_artifact", MODULE_PATH)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_evaluate_passes_for_strict_freegs_contract() -> None:
    report = {
        "mode": "freegs",
        "require_freegs_backend": True,
        "runtime_fallback_allowed": False,
        "freegs_runtime_fallback_cases": 0,
        "unconverged_case_count": 0,
        "all_cases_converged": True,
        "cases": [
            {
                "reference_backend": "freegs",
                "passes": True,
                "freegs_fallback": False,
                "psi_nrmse": 0.01,
                "psi_nrmse_normalized": 0.01,
                "q_profile_nrmse": 0.01,
                "axis_error_m": 0.01,
                "separatrix_nrmse": 0.01,
                "flux_area_rel_error": 0.01,
                "invariant_pass_fraction": 1.0,
            },
            {
                "reference_backend": "freegs",
                "passes": True,
                "freegs_fallback": False,
                "psi_nrmse": 0.02,
                "psi_nrmse_normalized": 0.02,
                "q_profile_nrmse": 0.02,
                "axis_error_m": 0.02,
                "separatrix_nrmse": 0.02,
                "flux_area_rel_error": 0.02,
                "invariant_pass_fraction": 1.0,
            },
        ],
    }
    summary = checker.evaluate(report)
    assert summary["overall_pass"] is True
    assert summary["failed_checks"] == []


def test_evaluate_fails_when_runtime_fallback_detected() -> None:
    report = {
        "mode": "freegs",
        "require_freegs_backend": True,
        "runtime_fallback_allowed": False,
        "freegs_runtime_fallback_cases": 1,
        "cases": [
            {"reference_backend": "solovev_fallback", "passes": True, "freegs_fallback": True},
        ],
    }
    summary = checker.evaluate(report)
    assert summary["overall_pass"] is False
    assert "runtime_fallback_case_count_zero" in summary["failed_checks"]
    assert "all_reference_backends_freegs" in summary["failed_checks"]


def test_evaluate_fails_when_any_case_is_unconverged() -> None:
    report = {
        "mode": "freegs",
        "require_freegs_backend": True,
        "runtime_fallback_allowed": False,
        "freegs_runtime_fallback_cases": 0,
        "unconverged_case_count": 1,
        "all_cases_converged": False,
        "cases": [
            {"reference_backend": "freegs", "passes": False, "freegs_fallback": False},
        ],
    }
    summary = checker.evaluate(report)
    assert summary["overall_pass"] is False
    assert "all_cases_converged" in summary["failed_checks"]
    assert "unconverged_case_count_zero" in summary["failed_checks"]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_evaluate_fails_when_strict_metric_is_nonfinite(bad_value: float) -> None:
    report = {
        "mode": "freegs",
        "require_freegs_backend": True,
        "runtime_fallback_allowed": False,
        "freegs_runtime_fallback_cases": 0,
        "unconverged_case_count": 0,
        "all_cases_converged": True,
        "cases": [
            {
                "reference_backend": "freegs",
                "passes": True,
                "freegs_fallback": False,
                "psi_nrmse": bad_value,
                "psi_nrmse_normalized": 0.01,
                "q_profile_nrmse": 0.01,
                "axis_error_m": 0.01,
                "separatrix_nrmse": 0.01,
                "flux_area_rel_error": 0.01,
                "invariant_pass_fraction": 1.0,
            }
        ],
    }

    summary = checker.evaluate(report)

    assert summary["overall_pass"] is False
    assert "all_required_metrics_finite" in summary["failed_checks"]


def test_evaluate_fails_when_strict_metric_is_missing() -> None:
    report = {
        "mode": "freegs",
        "require_freegs_backend": True,
        "runtime_fallback_allowed": False,
        "freegs_runtime_fallback_cases": 0,
        "unconverged_case_count": 0,
        "all_cases_converged": True,
        "cases": [
            {
                "reference_backend": "freegs",
                "passes": True,
                "freegs_fallback": False,
                "psi_nrmse": 0.01,
            }
        ],
    }

    summary = checker.evaluate(report)

    assert summary["overall_pass"] is False
    assert "all_required_metrics_present" in summary["failed_checks"]


def test_main_writes_summary_json(tmp_path: Path) -> None:
    report_path = tmp_path / "freegs.json"
    summary_path = tmp_path / "summary.json"
    report_path.write_text(
        json.dumps(
            {
                "mode": "freegs",
                "require_freegs_backend": True,
                "runtime_fallback_allowed": False,
                "freegs_runtime_fallback_cases": 0,
                "unconverged_case_count": 0,
                "all_cases_converged": True,
                "cases": [
                    {
                        "reference_backend": "freegs",
                        "passes": True,
                        "freegs_fallback": False,
                        "psi_nrmse": 0.01,
                        "psi_nrmse_normalized": 0.01,
                        "q_profile_nrmse": 0.01,
                        "axis_error_m": 0.01,
                        "separatrix_nrmse": 0.01,
                        "flux_area_rel_error": 0.01,
                        "invariant_pass_fraction": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rc = checker.main(
        [
            "--report",
            str(report_path),
            "--summary-json",
            str(summary_path),
        ]
    )
    assert rc == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True
