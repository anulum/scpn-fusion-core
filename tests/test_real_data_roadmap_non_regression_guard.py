# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Regression tests for tools/real_data_roadmap_non_regression_guard.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "real_data_roadmap_non_regression_guard.py"
SPEC = importlib.util.spec_from_file_location(
    "real_data_roadmap_non_regression_guard",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
real_data_roadmap_non_regression_guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(real_data_roadmap_non_regression_guard)


def test_non_regression_passes_when_metrics_hold_or_improve() -> None:
    """Accept current metrics that equal or exceed every baseline floor."""
    progress = {
        "roadmap_version": "v4.0",
        "metrics": [
            {"metric": "equilibrium_files_total", "current": 19},
            {"metric": "transport_shots_total", "current": 60},
        ],
        "d3d_raw_ingestion_ready": False,
    }
    baseline = {
        "metrics": {
            "equilibrium_files_total": 18,
            "transport_shots_total": 53,
        },
        "d3d_raw_ingestion_ready": False,
    }
    summary = real_data_roadmap_non_regression_guard.evaluate(
        progress=progress,
        baseline=baseline,
    )
    assert summary["overall_pass"] is True
    assert summary["regressions"] == []


def test_non_regression_fails_on_metric_drop() -> None:
    """Report a regression when a current metric falls below its floor."""
    progress = {
        "roadmap_version": "v4.0",
        "metrics": [
            {"metric": "equilibrium_files_total", "current": 17},
        ],
        "d3d_raw_ingestion_ready": False,
    }
    baseline = {
        "metrics": {"equilibrium_files_total": 18},
        "d3d_raw_ingestion_ready": False,
    }
    summary = real_data_roadmap_non_regression_guard.evaluate(
        progress=progress,
        baseline=baseline,
    )
    assert summary["overall_pass"] is False
    assert "equilibrium_files_total" in summary["regressions"]


def test_evaluate_rejects_invalid_baseline_and_d3d_regression() -> None:
    """Reject a malformed baseline and fail closed on lost DIII-D readiness."""
    with pytest.raises(ValueError, match="missing 'metrics' object"):
        real_data_roadmap_non_regression_guard.evaluate(
            progress={"metrics": []},
            baseline={"metrics": []},
        )

    summary = real_data_roadmap_non_regression_guard.evaluate(
        progress={"metrics": "invalid", "d3d_raw_ingestion_ready": False},
        baseline={"metrics": {}, "d3d_raw_ingestion_ready": True},
    )
    assert summary["regressions"] == ["d3d_raw_ingestion_ready"]
    assert summary["d3d_raw_ingestion"]["passes"] is False


def test_evaluate_ignores_malformed_progress_rows() -> None:
    """Ignore rows that lack the metric mapping contract."""
    summary = real_data_roadmap_non_regression_guard.evaluate(
        progress={
            "metrics": [None, {"metric": None}, {"metric": "shots"}],
        },
        baseline={"metrics": {"shots": 0}},
    )
    assert summary["metric_checks"]["shots"] == {
        "current": 0,
        "baseline": 0,
        "passes": True,
    }


def test_main_writes_passing_summary_from_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the passing CLI boundary with repository-relative paths."""
    monkeypatch.setattr(real_data_roadmap_non_regression_guard, "REPO_ROOT", tmp_path)
    (tmp_path / "progress.json").write_text(
        json.dumps({"metrics": [{"metric": "shots", "current": 2}]}),
        encoding="utf-8",
    )
    (tmp_path / "baseline.json").write_text(
        json.dumps({"metrics": {"shots": 1}}),
        encoding="utf-8",
    )

    result = real_data_roadmap_non_regression_guard.main(
        [
            "--progress-json",
            "progress.json",
            "--baseline-json",
            "baseline.json",
            "--summary-json",
            "reports/summary.json",
        ]
    )

    assert result == 0
    summary = json.loads((tmp_path / "reports" / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall_pass"] is True
    assert "pass=True regressions=0" in capsys.readouterr().out


def test_main_reports_regression_from_absolute_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the failing CLI boundary with absolute paths."""
    progress = tmp_path / "progress.json"
    baseline = tmp_path / "baseline.json"
    summary_path = tmp_path / "summary.json"
    progress.write_text(
        json.dumps({"metrics": [{"metric": "shots", "current": 0}]}),
        encoding="utf-8",
    )
    baseline.write_text(json.dumps({"metrics": {"shots": 1}}), encoding="utf-8")

    result = real_data_roadmap_non_regression_guard.main(
        [
            "--progress-json",
            str(progress),
            "--baseline-json",
            str(baseline),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert result == 1
    assert json.loads(summary_path.read_text(encoding="utf-8"))["regressions"] == ["shots"]
    output = capsys.readouterr().out
    assert "pass=False regressions=1" in output
    assert "Regressed metrics: shots" in output


def test_main_rejects_non_object_json(tmp_path: Path) -> None:
    """Reject a progress payload that is not a JSON object."""
    progress = tmp_path / "progress.json"
    baseline = tmp_path / "baseline.json"
    progress.write_text("[]\n", encoding="utf-8")
    baseline.write_text('{"metrics": {}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object payload"):
        real_data_roadmap_non_regression_guard.main(
            [
                "--progress-json",
                str(progress),
                "--baseline-json",
                str(baseline),
                "--summary-json",
                str(tmp_path / "summary.json"),
            ]
        )
