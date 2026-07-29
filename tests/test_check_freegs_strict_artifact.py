# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/check_freegs_strict_artifact.py."""

from __future__ import annotations

import importlib.util
import importlib
import json
import runpy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_freegs_strict_artifact.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "freegs-strict.yml"
SPEC = importlib.util.spec_from_file_location("tools.check_freegs_strict_artifact", MODULE_PATH)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_freegs_strict_workflow_runs_the_public_same_case_lane() -> None:
    """The scheduled workflow must run fresh evidence and reject a blocked contract."""
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'cron: "23 4 * * 2"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "validation/benchmark_vs_freegs.py --strict-backend" not in workflow
    assert "tools/check_freegs_strict_artifact.py" in workflow
    assert "--run-public-example" in workflow
    assert "--report artifacts/freegs_public_example_strict_live.json" in workflow
    assert "--summary-json artifacts/freegs_strict_guard_summary.json" in workflow
    assert "validation/benchmark_free_boundary_strict_parity.py" in workflow
    assert workflow.count("--strict") == 1
    assert "if-no-files-found: error" in workflow
    assert "artifacts/freegs_public_example_strict_live.json" in workflow
    assert "artifacts/freegs_strict_guard_summary.json" in workflow
    assert "validation/reports/free_boundary_strict_parity_benchmark.json" in workflow


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


def _public_example_report(**overrides: object) -> dict[str, object]:
    """Return a minimal passing fresh public-example reconstruction payload."""
    report: dict[str, object] = {
        "schema": "freegs-public-example-reconstruction-report.v1",
        "status": "accepted_public_freegs_same_case_free_boundary_parity",
        "report_generation_mode": "external_backend_reconstruction",
        "freegs_backend_available": True,
        "freegs_version": "0.8.2",
        "case_count": 1,
        "external_nonlinear_output_ready": True,
        "strict_free_boundary_parity_evidence": {
            "accepted_full_fidelity": True,
            "blocking_requirements": [],
            "failed_threshold_check_count": 0,
            "grid_convergence_ready": True,
            "strict_threshold_acceptance_ready": True,
            "cases": [
                {
                    "external_nonlinear_output_ready": True,
                    "native_same_case_profile_source_ready": True,
                    "strict_threshold_acceptance_ready": True,
                }
            ],
        },
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


def test_evaluate_passes_for_fresh_public_example_contract() -> None:
    """Evaluator accepts fresh FreeGS same-case evidence only when every gate passes."""
    summary = checker.evaluate(_public_example_report())

    assert summary["overall_pass"] is True
    assert summary["failed_checks"] == []
    assert summary["case_count"] == 1
    assert summary["report_schema"] == "freegs-public-example-reconstruction-report.v1"
    assert summary["report_status"] == "accepted_public_freegs_same_case_free_boundary_parity"
    assert summary["freegs_version"] == "0.8.2"
    assert summary["blocking_requirements"] == []


def test_evaluate_fails_closed_for_incomplete_public_example_contract() -> None:
    """Malformed strict evidence cannot inherit acceptance from top-level metadata."""
    summary = checker.evaluate(
        _public_example_report(
            case_count=2,
            external_nonlinear_output_ready=False,
            strict_free_boundary_parity_evidence="invalid",
        )
    )

    assert summary["overall_pass"] is False
    assert set(summary["failed_checks"]) == {
        "case_count_matches",
        "external_nonlinear_output_ready",
        "strict_parity_accepted",
        "grid_convergence_ready",
        "strict_threshold_acceptance_ready",
        "failed_threshold_check_count_zero",
    }


def test_evaluate_rejects_tracked_public_example_fallback() -> None:
    """A cached accepted report cannot substitute for a fresh scheduled solve."""
    summary = checker.evaluate(
        _public_example_report(report_generation_mode="tracked_report_fallback")
    )

    assert summary["overall_pass"] is False
    assert summary["failed_checks"] == ["fresh_external_backend_reconstruction"]


def test_evaluate_rejects_public_example_case_and_blocker_drift() -> None:
    """Every strict case must be ready and the blocker collection must be empty."""
    strict = {
        "accepted_full_fidelity": True,
        "blocking_requirements": "missing reference",
        "failed_threshold_check_count": 0,
        "grid_convergence_ready": True,
        "strict_threshold_acceptance_ready": True,
        "cases": [
            {
                "external_nonlinear_output_ready": False,
                "native_same_case_profile_source_ready": False,
                "strict_threshold_acceptance_ready": False,
            },
            "invalid-case",
        ],
    }

    summary = checker.evaluate(_public_example_report(strict_free_boundary_parity_evidence=strict))

    assert summary["overall_pass"] is False
    assert summary["blocking_requirements"] == ["missing reference"]
    assert "blocking_requirements_empty" in summary["failed_checks"]
    assert "all_cases_external_ready" in summary["failed_checks"]
    assert "all_cases_native_same_case_ready" in summary["failed_checks"]
    assert "all_cases_strict_threshold_ready" in summary["failed_checks"]


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


def test_main_resolves_repo_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Main resolves report and summary paths relative to ``REPO_ROOT``."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_json(artifacts / "freegs.json", _strict_report())
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)

    rc = checker.main(
        ["--report", "artifacts/freegs.json", "--summary-json", "artifacts/summary.json"]
    )

    assert rc == 0
    assert (artifacts / "summary.json").exists()


def test_public_example_runner_delegates_without_canonical_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live runner must keep the benchmark's canonical write mode disabled."""
    benchmark = importlib.import_module("validation.benchmark_freegs_public_example_reconstruction")
    writes: list[bool] = []

    def fake_run_benchmark(*, write: bool = True) -> dict[str, object]:
        writes.append(write)
        return _public_example_report()

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)

    report = checker._run_public_example_report()

    assert report == _public_example_report()
    assert writes == [False]


def test_main_runs_and_writes_fresh_public_example_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live mode writes ephemeral evidence and its passing guard summary."""
    report_path = tmp_path / "freegs.json"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(checker, "_run_public_example_report", _public_example_report)

    rc = checker.main(
        [
            "--run-public-example",
            "--report",
            str(report_path),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert rc == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == _public_example_report()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["overall_pass"] is True


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
