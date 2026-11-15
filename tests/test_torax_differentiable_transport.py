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
from collections.abc import Callable
from pathlib import Path
from typing import cast

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


def test_build_report_runs_in_float64_even_when_jax_starts_in_float32() -> None:
    """The benchmark switches JAX to float64 itself, so the replay gate holds either way."""
    import jax

    read = cast(Callable[[str], object], jax.config.read)
    update = cast(Callable[[str, bool], None], jax.config.update)
    previous = bool(read("jax_enable_x64"))
    update("jax_enable_x64", False)
    try:
        report = benchmark.build_report()
        assert bool(read("jax_enable_x64")) is True
    finally:
        update("jax_enable_x64", previous)

    assert report["gates"]["production_forward_replay"] is True
    assert report["forward_replay"]["maximum_absolute_error"] < 1.0e-12
