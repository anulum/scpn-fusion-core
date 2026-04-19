# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/coverage_delta_report.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "coverage_delta_report.py"
SPEC = importlib.util.spec_from_file_location("coverage_delta_report", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_main_writes_reports(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    thresholds = tmp_path / "thresholds.json"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    _write_json(
        summary,
        {
            "line_rate_pct": 90.0,
            "branch_rate_pct": 70.0,
            "domain_line_rate_pct": {"control": 91.0},
            "domain_branch_rate_pct": {"control": 71.0},
            "file_line_rate_pct": {"src/scpn_fusion/control/example.py": 92.0},
            "file_branch_rate_pct": {"src/scpn_fusion/control/example.py": 72.0},
        },
    )
    _write_json(
        thresholds,
        {
            "global_min_line_rate": 80.0,
            "global_min_branch_rate": 60.0,
            "domain_min_line_rate": {"control": 80.0},
            "domain_min_branch_rate": {"control": 60.0},
            "file_min_line_rate": {"src/scpn_fusion/control/example.py": 80.0},
            "file_min_branch_rate": {"src/scpn_fusion/control/example.py": 60.0},
        },
    )

    rc = mod.main(
        [
            "--coverage-summary",
            str(summary),
            "--thresholds",
            str(thresholds),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--strict",
        ]
    )
    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["overall_pass"] is True
    assert report["failing_count"] == 0
    assert report["missing_count"] == 0


def test_main_strict_fails_on_negative_delta(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    thresholds = tmp_path / "thresholds.json"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    _write_json(
        summary,
        {
            "line_rate_pct": 70.0,
            "branch_rate_pct": None,
            "domain_line_rate_pct": {},
            "domain_branch_rate_pct": {},
            "file_line_rate_pct": {},
            "file_branch_rate_pct": {},
        },
    )
    _write_json(
        thresholds,
        {
            "global_min_line_rate": 80.0,
            "global_min_branch_rate": 10.0,
        },
    )

    rc = mod.main(
        [
            "--coverage-summary",
            str(summary),
            "--thresholds",
            str(thresholds),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--strict",
        ]
    )
    assert rc == 1
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["overall_pass"] is False
    assert report["failing_count"] == 1
    assert report["missing_count"] == 1
