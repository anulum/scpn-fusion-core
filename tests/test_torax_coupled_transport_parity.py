# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coupled TORAX Parity Tests
"""Production-boundary tests for coupled TORAX/native evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from validation import benchmark_torax_coupled_transport_parity as benchmark


def _payload() -> dict[str, Any]:
    return json.loads(benchmark.REFERENCE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_reference_authenticates_pinned_deck_and_deterministic_replay() -> None:
    reference = benchmark.load_reference()
    assert reference["provenance"]["torax_version"] == "1.4.3"
    assert reference["determinism"]["byte_identical_scientific_projection"] is True
    assert reference["primary"]["sim_error"] == "SimError.NO_ERROR"
    assert reference["refined"]["sim_error"] == "SimError.NO_ERROR"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("schema", "wrong"), "unexpected schema"),
        (("provenance.torax_version", "0.0.0"), "pinned TORAX 1.4.3"),
        (("provenance.deck_sha256", "0" * 64), "deck source digest is stale"),
        (
            ("determinism.byte_identical_scientific_projection", False),
            "lacks deterministic replay",
        ),
        (("primary.sim_error", "SimError.NUMERICAL_ERROR"), "did not complete cleanly"),
    ],
)
def test_reference_rejects_mutated_custody(
    tmp_path: Path,
    mutation: tuple[str, object],
    message: str,
) -> None:
    payload = _payload()
    key, value = mutation
    cursor: dict[str, Any] = payload
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        benchmark.load_reference(path)


def test_build_report_exercises_four_state_pareto_gate() -> None:
    report = benchmark.build_report()

    assert report["passes_thresholds"] is True
    assert all(report["gates"].values())
    assert set(report["state_metrics"]) == {
        "ion_temperature_kev",
        "electron_temperature_kev",
        "electron_density_m3",
        "poloidal_flux_wb_per_rad",
    }
    assert report["source_budget_metrics"]["torax_max_relative_error"] < 1e-12
    assert report["source_budget_metrics"]["native_max_relative_error"] < 1e-12
    assert report["conservation_metrics"]["native_max_linear_residual_linf"] < 1e-10
    assert report["determinism"]["native_byte_identical"] is True
    assert report["determinism"]["torax_byte_identical"] is True
    assert report["performance_superiority_claimed"] is False
    assert report["physics_equivalence_claimed"] is False


def test_tracked_report_and_markdown_are_current() -> None:
    assert benchmark.check_report() == []
    report = json.loads(benchmark.REPORT_JSON.read_text(encoding="utf-8"))
    markdown = benchmark.render_markdown(report)
    assert "Overall pass: `True`" in markdown
    assert "Performance superiority claimed: `False`" in markdown
    assert "Physics equivalence claimed: `False`" in markdown
    assert "`source_budgets`: `True`" in markdown


def test_check_report_rejects_mutated_projection_and_runtime(tmp_path: Path) -> None:
    report = json.loads(benchmark.REPORT_JSON.read_text(encoding="utf-8"))
    report["state_metrics"]["ion_temperature_kev"] = 9.0
    report["runtime_seconds"]["native_warm"] = 0.0
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    report_json.write_text(json.dumps(report), encoding="utf-8")
    report_md.write_text(benchmark.render_markdown(report), encoding="utf-8")

    errors = benchmark.check_report(report_json=report_json, report_md=report_md)

    assert "coupled parity report scientific projection digest is stale" in errors
    assert "coupled parity report runtime evidence is incomplete" in errors


def test_cli_check_reports_success(capsys: pytest.CaptureFixture[str]) -> None:
    assert benchmark.main(["--check"]) == 0
    assert capsys.readouterr().err == ""
