# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coverage Delta Report Tool Tests
"""Contract tests for the coverage delta report generator tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.coverage_delta_report import (
    _delta_row,
    _display_path,
    _fmt_delta,
    _fmt_pct,
    _load_json,
    _render_section,
    _resolve,
    _to_float_map,
    build_report,
    render_markdown,
)
from tools.coverage_delta_report import REPO_ROOT, main

_SUMMARY: dict[str, Any] = {
    "line_rate_pct": 95.0,
    "branch_rate_pct": 90.0,
    "domain_line_rate_pct": {"core": 96.0},
    "domain_branch_rate_pct": {"core": 89.0},
    "file_line_rate_pct": {"a.py": 80.0},
}
_THRESHOLDS: dict[str, Any] = {
    "global_min_line_rate": 90.0,
    "global_min_branch_rate": 92.0,
    "domain_min_line_rate": {"core": 95.0},
    "domain_min_branch_rate": {"core": 88.0},
    "file_min_line_rate": {"a.py": 85.0, "b.py": 70.0},
    "file_min_branch_rate": {"a.py": 60.0},
}


class TestPathHelpers:
    """Path resolution and display helpers."""

    def test_resolve_relative_is_repo_anchored(self) -> None:
        """A relative path resolves under the repository root."""
        assert _resolve("artifacts/x.json") == REPO_ROOT / "artifacts" / "x.json"

    def test_resolve_absolute_is_unchanged(self, tmp_path: Path) -> None:
        """An already-absolute path resolves to itself."""
        assert _resolve(str(tmp_path)) == tmp_path

    def test_display_path_inside_repo_is_relative(self) -> None:
        """A path under the repo root is displayed relative to it."""
        assert _display_path(REPO_ROOT / "artifacts" / "x.json") == "artifacts/x.json"

    def test_display_path_outside_repo_is_absolute(self, tmp_path: Path) -> None:
        """A path outside the repo root is displayed as an absolute posix path."""
        assert _display_path(tmp_path) == tmp_path.as_posix()


class TestJsonLoaders:
    """JSON loading and float-map coercion helpers."""

    def test_load_json_missing_file(self, tmp_path: Path) -> None:
        """A missing JSON path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="thing not found"):
            _load_json(tmp_path / "absent.json", label="thing")

    def test_load_json_rejects_non_object(self, tmp_path: Path) -> None:
        """A JSON payload that is not an object is rejected."""
        path = tmp_path / "list.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            _load_json(path, label="thing")

    def test_load_json_valid(self, tmp_path: Path) -> None:
        """A JSON object payload loads into a dict."""
        path = tmp_path / "ok.json"
        path.write_text('{"a": 1}', encoding="utf-8")
        assert _load_json(path, label="thing") == {"a": 1}

    def test_to_float_map_none_is_empty(self) -> None:
        """A ``None`` payload maps to an empty dict."""
        assert _to_float_map(None, label="x") == {}

    def test_to_float_map_valid(self) -> None:
        """Numeric values are coerced to floats."""
        assert _to_float_map({"a": 1, "b": 2.5}, label="x") == {"a": 1.0, "b": 2.5}

    def test_to_float_map_rejects_non_dict(self) -> None:
        """A non-dict payload is rejected."""
        with pytest.raises(ValueError, match="must be a JSON object"):
            _to_float_map([1, 2], label="x")

    def test_to_float_map_rejects_non_string_key(self) -> None:
        """A non-string key is rejected."""
        with pytest.raises(ValueError, match="keys must be strings"):
            _to_float_map({1: 2.0}, label="x")

    def test_to_float_map_rejects_non_numeric_value(self) -> None:
        """A non-numeric value is rejected."""
        with pytest.raises(ValueError, match="must be numeric"):
            _to_float_map({"a": "nope"}, label="x")


class TestDeltaRowAndFormatters:
    """Per-check delta rows and cell formatters."""

    def test_delta_row_missing_observation(self) -> None:
        """A missing observation yields a failing, flagged row."""
        row = _delta_row(name="c", target=90.0, observed=None)
        assert row["missing_observation"] is True
        assert row["passes"] is False
        assert row["delta"] is None

    def test_delta_row_pass_and_fail(self) -> None:
        """An observation at or above target passes; below target fails."""
        assert _delta_row(name="c", target=90.0, observed=95.0)["passes"] is True
        assert _delta_row(name="c", target=90.0, observed=80.0)["passes"] is False

    def test_fmt_pct(self) -> None:
        """Percentage formatting handles values and the missing case."""
        assert _fmt_pct(None) == "n/a"
        assert _fmt_pct(95.0) == "95.00%"

    def test_fmt_delta_sign(self) -> None:
        """Delta formatting signs positives and handles the missing case."""
        assert _fmt_delta(None) == "n/a"
        assert _fmt_delta(1.5) == "+1.50pp"
        assert _fmt_delta(-2.0) == "-2.00pp"

    def test_render_section_statuses(self) -> None:
        """A section renders PASS, FAIL, and MISSING statuses."""
        rows = [
            _delta_row(name="p", target=90.0, observed=95.0),
            _delta_row(name="f", target=90.0, observed=80.0),
            _delta_row(name="m", target=90.0, observed=None),
        ]
        text = "\n".join(_render_section("Section", rows))
        assert "PASS" in text and "FAIL" in text and "MISSING" in text


class TestBuildReport:
    """The end-to-end delta report builder."""

    def test_reports_pass_fail_and_missing(self) -> None:
        """The report aggregates passing, failing, and missing checks."""
        report = build_report(summary=_SUMMARY, thresholds=_THRESHOLDS)
        assert report["overall_pass"] is False  # a.py 80 < 85; b.py + a.py branch missing
        assert report["failing_count"] >= 1
        assert report["missing_count"] >= 1
        assert isinstance(report["generated_at"], str)
        assert report["worst_delta_check"] is not None

    def test_all_pass_report(self) -> None:
        """A summary meeting every threshold reports an overall pass."""
        summary = {"line_rate_pct": 99.0}
        thresholds = {"global_min_line_rate": 90.0}
        report = build_report(summary=summary, thresholds=thresholds)
        assert report["overall_pass"] is True
        assert report["worst_delta_check"]["name"] == "global_line_rate_pct"


class TestRenderMarkdown:
    """Markdown rendering of a delta report."""

    def test_render_includes_worst_and_sections(self) -> None:
        """The markdown includes the worst-delta block and section tables."""
        report = build_report(summary=_SUMMARY, thresholds=_THRESHOLDS)
        text = render_markdown(report)
        assert "# Coverage Delta Report" in text
        assert "## Worst Delta" in text
        assert "## Global Coverage Deltas" in text

    def test_render_without_observations_omits_worst(self) -> None:
        """With no observed rows the worst-delta block is omitted."""
        report = build_report(summary={}, thresholds={"domain_min_line_rate": {"x": 90.0}})
        text = render_markdown(report)
        assert "## Worst Delta" not in text


class TestMain:
    """The command-line entry point."""

    def _write(
        self, tmp_path: Path, summary: dict[str, Any], thresholds: dict[str, Any]
    ) -> tuple[Path, Path]:
        """Write summary and threshold JSON files and return their paths."""
        s = tmp_path / "summary.json"
        t = tmp_path / "thresholds.json"
        s.write_text(json.dumps(summary), encoding="utf-8")
        t.write_text(json.dumps(thresholds), encoding="utf-8")
        return s, t

    def test_main_writes_reports_and_returns_zero(self, tmp_path: Path) -> None:
        """A non-strict run writes both reports and returns zero even on failures."""
        s, t = self._write(tmp_path, _SUMMARY, _THRESHOLDS)
        out_json = tmp_path / "report.json"
        out_md = tmp_path / "report.md"
        rc = main(
            [
                "--coverage-summary",
                str(s),
                "--thresholds",
                str(t),
                "--output-json",
                str(out_json),
                "--output-md",
                str(out_md),
            ]
        )
        assert rc == 0
        assert out_json.exists() and out_md.exists()

    def test_main_strict_returns_one_on_failure(self, tmp_path: Path) -> None:
        """A strict run returns one when a threshold is not met."""
        s, t = self._write(tmp_path, _SUMMARY, _THRESHOLDS)
        rc = main(
            [
                "--coverage-summary",
                str(s),
                "--thresholds",
                str(t),
                "--output-json",
                str(tmp_path / "r.json"),
                "--output-md",
                str(tmp_path / "r.md"),
                "--strict",
            ]
        )
        assert rc == 1

    def test_main_strict_returns_zero_on_pass(self, tmp_path: Path) -> None:
        """A strict run returns zero when every threshold is met."""
        s, t = self._write(tmp_path, {"line_rate_pct": 99.0}, {"global_min_line_rate": 90.0})
        rc = main(
            [
                "--coverage-summary",
                str(s),
                "--thresholds",
                str(t),
                "--output-json",
                str(tmp_path / "r.json"),
                "--output-md",
                str(tmp_path / "r.md"),
                "--strict",
            ]
        )
        assert rc == 0
