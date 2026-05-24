# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Quality Gate Tool Tests
"""Behavioural contract tests for ``tools/coverage_guard.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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
