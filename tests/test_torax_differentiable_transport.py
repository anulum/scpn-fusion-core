# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Differentiable Transport Evidence Tests
"""Custody and threshold tests for tracked differentiable evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from validation import benchmark_torax_differentiable_transport as benchmark


def test_tracked_differentiable_report_is_current() -> None:
    assert benchmark.check_report() == []
    report = json.loads(benchmark.REPORT_JSON.read_text(encoding="utf-8"))
    assert report["passes_thresholds"] is True
    assert all(report["gates"].values())
    assert report["performance_superiority_claimed"] is False
    assert report["general_transport_differentiability_claimed"] is False


def test_check_rejects_mutated_scientific_projection(tmp_path: Path) -> None:
    report = json.loads(benchmark.REPORT_JSON.read_text(encoding="utf-8"))
    report["gradient_metrics"]["maximum_relative_error"] = 1.0
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    report_json.write_text(json.dumps(report), encoding="utf-8")
    report_md.write_text(benchmark.render_markdown(report), encoding="utf-8")
    errors = benchmark.check_report(report_json=report_json, report_md=report_md)
    assert "differentiable transport scientific projection is stale" in errors


def test_cli_check_reports_success(capsys: pytest.CaptureFixture[str]) -> None:
    assert benchmark.main(["--check"]) == 0
    assert capsys.readouterr().err == ""
