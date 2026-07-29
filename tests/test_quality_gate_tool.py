# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Behavioural contract tests for ``tools/coverage_guard.py``."""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "coverage_guard.py"
SPEC = importlib.util.spec_from_file_location("coverage_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def _write_branch_aware_quality_xml(path: Path) -> None:
    xml = """<?xml version="1.0" ?>
<coverage line-rate="0.90" branch-rate="0.75" lines-covered="9" lines-valid="10" branches-covered="3" branches-valid="4">
  <packages>
    <package name="scpn_fusion.control" line-rate="0.90">
      <classes>
        <class name="control" filename="src/scpn_fusion/control/example.py" line-rate="0.90" branch-rate="0.75">
          <lines>
            <line number="1" hits="1" branch="true" condition-coverage="100% (2/2)" />
            <line number="2" hits="1" branch="true" condition-coverage="50% (1/2)" />
            <line number="3" hits="1" />
            <line number="4" hits="1" />
            <line number="5" hits="1" />
            <line number="6" hits="1" />
            <line number="7" hits="1" />
            <line number="8" hits="1" />
            <line number="9" hits="1" />
            <line number="10" hits="0" />
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""
    path.write_text(xml, encoding="utf-8")


def _write_line_only_quality_xml(path: Path) -> None:
    xml = """<?xml version="1.0" ?>
<coverage line-rate="0.90" branch-rate="0.00" lines-covered="9" lines-valid="10" branches-covered="0" branches-valid="0">
  <packages>
    <package name="scpn_fusion.control" line-rate="0.90">
      <classes>
        <class name="control" filename="src/scpn_fusion/control/example.py" line-rate="0.90" branch-rate="0.00">
          <lines>
            <line number="1" hits="1" />
            <line number="2" hits="1" />
            <line number="3" hits="1" />
            <line number="4" hits="1" />
            <line number="5" hits="1" />
            <line number="6" hits="1" />
            <line number="7" hits="1" />
            <line number="8" hits="1" />
            <line number="9" hits="1" />
            <line number="10" hits="0" />
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""
    path.write_text(xml, encoding="utf-8")


def test_main_accepts_branch_aware_report_above_all_thresholds(tmp_path: Path) -> None:
    """The gate accepts global, domain, and file thresholds that are met."""
    quality_xml = tmp_path / "quality.xml"
    thresholds = tmp_path / "thresholds.json"
    _write_branch_aware_quality_xml(quality_xml)
    thresholds.write_text(
        json.dumps(
            {
                "global_min_line_rate": 80.0,
                "global_min_branch_rate": 70.0,
                "domain_min_line_rate": {"control": 80.0},
                "domain_min_branch_rate": {"control": 70.0},
                "file_min_line_rate": {"src/scpn_fusion/control/example.py": 80.0},
                "file_min_branch_rate": {"src/scpn_fusion/control/example.py": 70.0},
            }
        ),
        encoding="utf-8",
    )

    rc = guard.main(["--coverage-xml", str(quality_xml), "--thresholds", str(thresholds)])

    assert rc == 0


def test_main_rejects_report_below_global_domain_and_file_thresholds(tmp_path: Path) -> None:
    """The gate fails when configured thresholds exceed measured evidence."""
    quality_xml = tmp_path / "quality.xml"
    thresholds = tmp_path / "thresholds.json"
    _write_branch_aware_quality_xml(quality_xml)
    thresholds.write_text(
        json.dumps(
            {
                "global_min_line_rate": 95.0,
                "global_min_branch_rate": 80.0,
                "domain_min_line_rate": {"control": 95.0},
                "domain_min_branch_rate": {"control": 80.0},
                "file_min_line_rate": {"src/scpn_fusion/control/example.py": 95.0},
                "file_min_branch_rate": {"src/scpn_fusion/control/example.py": 80.0},
            }
        ),
        encoding="utf-8",
    )

    rc = guard.main(["--coverage-xml", str(quality_xml), "--thresholds", str(thresholds)])

    assert rc == 1


def test_main_rejects_missing_branch_evidence_when_branch_threshold_is_required(
    tmp_path: Path,
) -> None:
    """A branch threshold without branch evidence is a failing gate condition."""
    quality_xml = tmp_path / "quality.xml"
    thresholds = tmp_path / "thresholds.json"
    _write_line_only_quality_xml(quality_xml)
    thresholds.write_text(
        json.dumps(
            {
                "global_min_line_rate": 80.0,
                "global_min_branch_rate": 1.0,
                "domain_min_line_rate": {"control": 80.0},
                "file_min_line_rate": {"src/scpn_fusion/control/example.py": 80.0},
            }
        ),
        encoding="utf-8",
    )

    rc = guard.main(["--coverage-xml", str(quality_xml), "--thresholds", str(thresholds)])

    assert rc == 1


@pytest.mark.parametrize(
    "loader_name",
    ["coverage", "thresholds"],
)
def test_loaders_reject_missing_inputs(tmp_path: Path, loader_name: str) -> None:
    """Fail explicitly when a required coverage or threshold input is absent."""
    missing = tmp_path / "missing"
    loader = guard.load_coverage if loader_name == "coverage" else guard.load_thresholds

    with pytest.raises(FileNotFoundError, match="not found"):
        loader(missing)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must be a JSON object"),
        ({}, "must define global_min_line_rate"),
        ({"global_min_line_rate": True}, "must be numeric"),
        ({"global_min_line_rate": []}, "must be numeric"),
        ({"global_min_line_rate": "invalid"}, "must be numeric"),
        ({"global_min_line_rate": "nan"}, "must be finite"),
        ({"global_min_line_rate": -1}, "must be in \\[0, 100\\]"),
        ({"global_min_line_rate": 101}, "must be in \\[0, 100\\]"),
        (
            {"global_min_line_rate": 80, "domain_min_line_rate": []},
            "domain_min_line_rate must be a JSON object",
        ),
        (
            {"global_min_line_rate": 80, "file_min_branch_rate": {"sample.py": None}},
            "file_min_branch_rate\\[sample.py\\] must be numeric",
        ),
    ],
)
def test_load_thresholds_rejects_malformed_configuration(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    """Reject invalid roots, shapes, scalar types, and percentage ranges."""
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        guard.load_thresholds(thresholds)


def test_load_thresholds_accepts_numeric_strings(tmp_path: Path) -> None:
    """Preserve support for finite in-range numeric strings in JSON."""
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "global_min_line_rate": "80",
                "global_min_branch_rate": "70.5",
                "domain_min_line_rate": {"core": "75"},
            }
        ),
        encoding="utf-8",
    )

    assert guard.load_thresholds(thresholds)["global_min_line_rate"] == "80"


@pytest.mark.parametrize(
    ("condition_coverage", "expected"),
    [
        ("not available", None),
        ("0% (0/0)", None),
        ("200% (2/1)", None),
        ("50% (1/2)", (1, 2)),
    ],
)
def test_condition_coverage_parser_rejects_invalid_counters(
    condition_coverage: str,
    expected: tuple[int, int] | None,
) -> None:
    """Accept only positive, internally consistent branch counters."""
    assert guard._parse_condition_coverage(condition_coverage) == expected


def test_load_coverage_normalizes_paths_and_empty_classes(tmp_path: Path) -> None:
    """Normalize Windows paths and retain zero-line classes in the other domain."""
    coverage_xml = tmp_path / "coverage.xml"
    coverage_xml.write_text(
        """<?xml version="1.0" ?>
<coverage line-rate="0.50" lines-covered="1" lines-valid="2" branches-covered="0" branches-valid="0">
  <packages><package name="mixed"><classes>
    <class name="blank" filename="" line-rate="0.0"><lines /></class>
    <class name="empty" filename="standalone.py" line-rate="0.0"><lines /></class>
    <class name="core" filename="src\\scpn_fusion\\core\\sample.py" line-rate="0.5">
      <lines>
        <line number="1" hits="1" branch="true" condition-coverage="not available" />
        <line number="2" hits="0" branch="true" condition-coverage="200% (2/1)" />
      </lines>
    </class>
  </classes></package></packages>
</coverage>
""",
        encoding="utf-8",
    )

    summary = guard.load_coverage(coverage_xml)

    assert summary.branch_rate_pct is None
    assert summary.file_line_rate_pct == {
        "standalone.py": 0.0,
        "src/scpn_fusion/core/sample.py": 50.0,
    }
    assert summary.file_branch_rate_pct == {}
    assert summary.domain_line_rate_pct == {"other": 0.0, "core": 50.0}


def test_main_reports_every_missing_configured_surface(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report absent line and branch evidence for both domains and files."""
    quality_xml = tmp_path / "quality.xml"
    thresholds = tmp_path / "thresholds.json"
    _write_branch_aware_quality_xml(quality_xml)
    thresholds.write_text(
        json.dumps(
            {
                "global_min_line_rate": 80,
                "domain_min_line_rate": {"missing": 1},
                "file_min_line_rate": {"missing.py": 1},
                "domain_min_branch_rate": {"missing": 1},
                "file_min_branch_rate": {"missing.py": 1},
            }
        ),
        encoding="utf-8",
    )

    rc = guard.main(["--coverage-xml", str(quality_xml), "--thresholds", str(thresholds)])

    output = capsys.readouterr().out
    assert rc == 1
    assert "Domain 'missing' missing from coverage report." in output
    assert "File 'missing.py' missing from coverage report." in output
    assert "Domain 'missing' missing branch coverage data." in output
    assert "File 'missing.py' missing branch coverage data." in output


def test_main_writes_summary_for_relative_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Resolve relative CLI paths and emit the downstream summary contract."""
    quality_xml = tmp_path / "quality.xml"
    thresholds = tmp_path / "thresholds.json"
    summary_path = tmp_path / "nested" / "summary.json"
    _write_branch_aware_quality_xml(quality_xml)
    thresholds.write_text(json.dumps({"global_min_line_rate": 80}), encoding="utf-8")
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    rc = guard.main(
        [
            "--coverage-xml",
            "quality.xml",
            "--thresholds",
            "thresholds.json",
            "--summary-json",
            "nested/summary.json",
        ]
    )

    output = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "Coverage branch rate: 75.00% (3/4)" in output
    assert "Domain control" in output
    assert payload["line_rate_pct"] == 90.0
    assert payload["domain_branch_rate_pct"] == {"control": 75.0}


def test_evaluate_rejects_malformed_optional_mapping(tmp_path: Path) -> None:
    """Keep direct evaluator use fail-closed for malformed optional mappings."""
    quality_xml = tmp_path / "quality.xml"
    _write_branch_aware_quality_xml(quality_xml)
    summary = guard.load_coverage(quality_xml)

    with pytest.raises(ValueError, match="domain_min_line_rate must be a JSON object"):
        guard.evaluate(
            summary,
            {"global_min_line_rate": 80, "domain_min_line_rate": None},
        )


def test_direct_script_entry_point_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the actual script guard with a passing temporary report."""
    quality_xml = tmp_path / "quality.xml"
    thresholds = tmp_path / "thresholds.json"
    _write_line_only_quality_xml(quality_xml)
    thresholds.write_text(json.dumps({"global_min_line_rate": 80}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--coverage-xml",
            str(quality_xml),
            "--thresholds",
            str(thresholds),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exit_info.value.code == 0
    assert "Coverage branch rate: n/a" in capsys.readouterr().out
