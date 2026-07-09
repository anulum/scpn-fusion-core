# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — External Parity Score Tests
"""Tests for the integrated TORAX and FreeGS/FreeGSNKE parity score report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from validation import external_parity_score as score


def test_external_parity_score_aggregates_tracked_reports() -> None:
    """The current tracked artifacts produce a fail-closed scored report."""
    report = score.build_report()

    assert report["schema"] == score.SCHEMA
    assert report["status"] == "blocked_external_parity_score"
    assert report["acceptance_passed"] is False
    assert report["reproducibility_score"] == 1.0
    assert report["parity_score"] == 0.6

    lanes = {lane["lane"]: lane for lane in report["lanes"]}
    assert set(lanes) == {"torax_transport", "freegsnke_free_boundary"}
    assert lanes["torax_transport"]["status"] == "blocked_same_physics_thresholds"
    assert lanes["torax_transport"]["parity_score"] == 0.2
    assert lanes["torax_transport"]["blocked_requirements"] == [
        "native_transport_model",
        "sources_and_boundary_conditions",
        "time_integration_contract",
    ]
    assert lanes["freegsnke_free_boundary"]["status"] == "accepted_external_parity"
    assert lanes["freegsnke_free_boundary"]["parity_score"] == 1.0
    assert len(report["source_reports"]) == 6
    assert all(source["file_sha256"] for source in report["source_reports"])


def test_write_and_check_report_roundtrip(tmp_path: Path) -> None:
    """Generated JSON and Markdown reports pass the drift checker."""
    json_report = tmp_path / "external_parity_score.json"
    md_report = tmp_path / "external_parity_score.md"

    written = score.write_report(json_report_path=json_report, md_report_path=md_report)
    errors = score.check_report(json_report_path=json_report, md_report_path=md_report)

    assert errors == []
    assert json.loads(json_report.read_text(encoding="utf-8")) == written
    assert "# External Parity Score" in md_report.read_text(encoding="utf-8")


def test_check_report_detects_stale_json_and_markdown(tmp_path: Path) -> None:
    """The checker reports stale or missing tracked score artifacts."""
    json_report = tmp_path / "external_parity_score.json"
    md_report = tmp_path / "external_parity_score.md"
    json_report.write_text(json.dumps({"schema": score.SCHEMA}) + "\n", encoding="utf-8")
    md_report.write_text("stale\n", encoding="utf-8")

    errors = score.check_report(json_report_path=json_report, md_report_path=md_report)
    missing_errors = score.check_report(
        json_report_path=tmp_path / "missing.json",
        md_report_path=tmp_path / "missing.md",
    )

    assert "tracked external parity score JSON report is stale" in errors
    assert "tracked external parity score Markdown report is stale" in errors
    assert missing_errors == [
        f"missing external parity score JSON report: {tmp_path / 'missing.json'}",
        f"missing external parity score Markdown report: {tmp_path / 'missing.md'}",
    ]


def test_load_report_fails_closed_on_wrong_shape_or_schema(tmp_path: Path) -> None:
    """Source report loading rejects non-object JSON and schema drift."""
    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    wrong_schema = tmp_path / "wrong.json"
    wrong_schema.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        score._load_report(non_object, expected_schema=score.SCHEMA)
    with pytest.raises(ValueError, match="schema mismatch"):
        score._load_report(wrong_schema, expected_schema=score.SCHEMA)


def test_score_helpers_cover_empty_and_targeted_matrix_cases() -> None:
    """Scoring helpers handle empty input and explicit matrix status lookup."""
    matrix: list[dict[str, Any]] = [{"component": "radial_grid", "status": "shared_ready"}]

    assert score._score([]) == 0.0
    assert score._score([{"ready": True}, {"ready": False}, {"ready": "yes"}]) == 0.333333
    assert score._matrix_has_status(matrix, "radial_grid", "shared_ready") is True
    assert score._matrix_has_status(matrix, "radial_grid", "blocked") is False
    assert score._matrix_has_status(matrix, "missing", "shared_ready") is False


def test_markdown_renderer_lists_lanes_and_sources() -> None:
    """Markdown rendering exposes lane scores and source checksums."""
    report = score.build_report()
    rendered = score.render_markdown(report)

    assert "`blocked_external_parity_score`" in rendered
    assert "torax_transport" in rendered
    assert "FreeGS/FreeGSNKE" in rendered
    assert "validation/reports/torax_real_parity.json" in rendered


def test_main_writes_checks_and_strict_fails_on_blocked_score(tmp_path: Path) -> None:
    """The CLI writes normally, checks drift, and fails only in strict mode."""
    json_report = tmp_path / "score.json"
    md_report = tmp_path / "score.md"

    assert score.main(["--report-json", str(json_report), "--report-md", str(md_report)]) == 0
    assert (
        score.main(["--report-json", str(json_report), "--report-md", str(md_report), "--check"])
        == 0
    )
    assert (
        score.main(
            [
                "--report-json",
                str(json_report),
                "--report-md",
                str(md_report),
                "--strict",
            ]
        )
        == 1
    )
    json_report.write_text(json.dumps({"schema": score.SCHEMA}) + "\n", encoding="utf-8")
    assert (
        score.main(["--report-json", str(json_report), "--report-md", str(md_report), "--check"])
        == 1
    )
