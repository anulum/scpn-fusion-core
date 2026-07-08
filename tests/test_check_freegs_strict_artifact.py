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
import runpy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_freegs_strict_artifact.py"
SPEC = importlib.util.spec_from_file_location("tools.check_freegs_strict_artifact", MODULE_PATH)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def _strict_report(**overrides: object) -> dict[str, object]:
    """Return a minimal passing FreeGS strict artifact payload."""
    report: dict[str, object] = {
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
    report.update(overrides)
    return report


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON payload to ``path``."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_passes_for_strict_freegs_contract() -> None:
    """Evaluator passes for a strict FreeGS artifact with finite case metrics."""
    report = _strict_report()

    summary = checker.evaluate(report)

    assert summary["overall_pass"] is True
    assert summary["failed_checks"] == []


def test_evaluate_fails_when_runtime_fallback_detected() -> None:
    """Evaluator reports runtime and case-level fallback violations."""
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
    assert "no_case_level_fallback" in summary["failed_checks"]


def test_evaluate_fails_when_any_case_is_unconverged() -> None:
    """Evaluator reports unconverged and failed case contracts."""
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
    assert "all_cases_pass" in summary["failed_checks"]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_evaluate_fails_when_strict_metric_is_nonfinite(bad_value: float) -> None:
    """Evaluator reports non-finite strict metrics."""
    report = _strict_report()
    cases = report["cases"]
    assert isinstance(cases, list)
    case = cases[0]
    assert isinstance(case, dict)
    case["psi_nrmse"] = bad_value

    summary = checker.evaluate(report)

    assert summary["overall_pass"] is False
    assert "all_required_metrics_finite" in summary["failed_checks"]


def test_evaluate_fails_when_strict_metric_is_missing() -> None:
    """Evaluator reports missing strict metrics."""
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


def test_evaluate_fails_for_non_freegs_mode_and_case_errors() -> None:
    """Evaluator reports mode, strict-request, fallback, and case-error violations."""
    report = _strict_report(
        mode="solovev",
        require_freegs_backend=False,
        runtime_fallback_allowed=True,
        cases=[
            {
                "reference_backend": "freegs",
                "passes": True,
                "freegs_fallback": False,
                "error": "solver failed",
                "psi_nrmse": 0.01,
                "psi_nrmse_normalized": 0.01,
                "q_profile_nrmse": 0.01,
                "axis_error_m": 0.01,
                "separatrix_nrmse": 0.01,
                "flux_area_rel_error": 0.01,
                "invariant_pass_fraction": 1.0,
            }
        ],
    )

    summary = checker.evaluate(report)

    assert summary["overall_pass"] is False
    assert "strict_requested" in summary["failed_checks"]
    assert "mode_is_freegs" in summary["failed_checks"]
    assert "runtime_fallback_disallowed" in summary["failed_checks"]
    assert "no_case_level_errors" in summary["failed_checks"]


def test_main_returns_failure_for_zero_case_artifact(tmp_path: Path) -> None:
    """Main fails closed when the report has zero usable cases."""
    report_path = tmp_path / "freegs.json"
    summary_path = tmp_path / "summary.json"
    _write_json(report_path, {"cases": []})

    rc = checker.main(["--report", str(report_path), "--summary-json", str(summary_path)])

    assert rc == 1
    assert not summary_path.exists()


def test_main_returns_failure_for_non_object_json(tmp_path: Path) -> None:
    """Main fails closed when the report JSON is not an object."""
    report_path = tmp_path / "freegs.json"
    summary_path = tmp_path / "summary.json"
    _write_json(report_path, [])

    rc = checker.main(["--report", str(report_path), "--summary-json", str(summary_path)])

    assert rc == 1
    assert not summary_path.exists()


def test_main_resolves_repo_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main resolves report and summary paths relative to ``REPO_ROOT``."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_json(artifacts / "freegs.json", _strict_report())
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)

    rc = checker.main(["--report", "artifacts/freegs.json", "--summary-json", "artifacts/summary.json"])

    assert rc == 0
    assert (artifacts / "summary.json").exists()


def test_main_writes_summary_json(tmp_path: Path) -> None:
    """Main writes a passing summary JSON for a valid strict artifact."""
    report_path = tmp_path / "freegs.json"
    summary_path = tmp_path / "summary.json"
    _write_json(report_path, _strict_report())

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


def test_main_writes_failure_summary_json(tmp_path: Path) -> None:
    """Main writes a failure summary JSON for contract violations."""
    report_path = tmp_path / "freegs.json"
    summary_path = tmp_path / "summary.json"
    _write_json(report_path, _strict_report(mode="fallback"))

    rc = checker.main(["--report", str(report_path), "--summary-json", str(summary_path)])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert rc == 1
    assert payload["overall_pass"] is False
    assert payload["failed_checks"] == ["mode_is_freegs"]


def test_script_entrypoint_exits_with_main_return_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executable entrypoint delegates through ``main`` and exits with its code."""
    report_path = tmp_path / "freegs.json"
    summary_path = tmp_path / "summary.json"
    _write_json(report_path, _strict_report())
    monkeypatch.setattr(
        sys,
        "argv",
        [str(MODULE_PATH), "--report", str(report_path), "--summary-json", str(summary_path)],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exc_info.value.code == 0
