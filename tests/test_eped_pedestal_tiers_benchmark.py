# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — EPED Pedestal Tier Benchmark Tests
"""Tests for the pedestal tier benchmark against digitised EPED1 references."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_eped_pedestal_tiers.py"
REFERENCE = (
    ROOT / "validation" / "reference_data" / "eped" / "eped1_snyder_apsdpp_diiid_ip_scan.json"
)
REPORT = ROOT / "validation" / "reports" / "eped_pedestal_tiers_benchmark.json"


@pytest.fixture(scope="module")
def bench() -> Any:
    spec = importlib.util.spec_from_file_location("benchmark_eped_pedestal_tiers", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report(bench: Any) -> dict[str, Any]:
    reference = bench.load_reference(REFERENCE)
    return dict(bench.build_report(reference))


class TestDigitisedReference:
    """The committed digitised reference is complete and provenanced."""

    def test_loads_and_validates(self, bench: Any) -> None:
        reference = bench.load_reference(REFERENCE)
        assert len(reference["diiid_ip_scan"]["cases"]) == 3

    def test_source_provenance_present(self) -> None:
        payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
        assert payload["source"]["url"].startswith("https://fusion.gat.com/")
        assert payload["digitisation_uncertainty"]["p_ped_kPa"] > 0.0
        assert any("103016" in c for c in payload["source"]["citations"])

    def test_unpublished_inputs_are_declared_null(self) -> None:
        payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
        machine = payload["diiid_ip_scan"]["machine_inputs"]
        assert machine["R0_m"] is None
        assert machine["a_m"] is None
        assert machine["n_ped_1e19"] is None

    def test_rejects_foreign_schema(self, bench: Any, tmp_path: Path) -> None:
        payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
        payload["schema"] = "other.schema"
        target = tmp_path / "ref.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="unexpected digitised EPED1 reference schema"):
            bench.load_reference(target)

    def test_rejects_missing_cases(self, bench: Any, tmp_path: Path) -> None:
        payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
        payload["diiid_ip_scan"]["cases"] = []
        target = tmp_path / "ref.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty diiid_ip_scan.cases"):
            bench.load_reference(target)


class TestReportInvariants:
    """The benchmark documents divergence and never claims parity."""

    def test_status_is_fail_closed(self, report: dict[str, Any]) -> None:
        assert report["status"] == "documented_divergence_no_parity_claim"

    def test_blockers_named(self, report: dict[str, Any]) -> None:
        assert any("miller_ballooning" in blocker for blocker in report["blockers"])
        assert any("not_published" in blocker for blocker in report["blockers"])

    def test_every_case_has_both_tier_rows(self, report: dict[str, Any]) -> None:
        assert len(report["rows"]) == 3
        for row in report["rows"]:
            assert row["fast_tier"]["height_ratio_to_eped1"] > 0.0
            assert row["pb_kbm_tier"]["height_ratio_to_eped1"] > 0.0
            assert len(row["fast_tier"]["predictions"]) == 3
            assert len(row["pb_kbm_tier"]["predictions"]) == 3

    def test_fast_tier_underprediction_is_documented(self, report: dict[str, Any]) -> None:
        """The width-height fast tier under-predicts the published rows."""
        assert report["summary"]["fast_tier_underpredicts_all_cases"] is True
        assert report["summary"]["fast_tier_height_ratio_max"] < 1.0

    def test_pb_kbm_collapse_is_documented(self, report: dict[str, Any]) -> None:
        """The s-alpha first-stability collapse is recorded per case."""
        assert report["summary"]["pb_kbm_collapsed_all_cases"] is True
        for row in report["rows"]:
            assert row["pb_kbm_tier"]["verdict"] == "collapsed_salpha_first_stability"

    def test_assumptions_declared(self, report: dict[str, Any]) -> None:
        assert report["assumed_geometry"] == {"R0_m": 1.67, "a_m": 0.67}
        assert (
            "not published" in report["assumption_note"].lower()
            or "NOT published" in report["assumption_note"]
        )

    def test_markdown_renders_blockers(self, bench: Any, report: dict[str, Any]) -> None:
        rendered = bench._render_markdown(report)
        assert "documented_divergence_no_parity_claim" in rendered
        for blocker in report["blockers"]:
            assert blocker in rendered


class TestCliAndCommittedReport:
    """CLI writes the tracked report; the committed report stays honest."""

    def test_main_writes_report(self, bench: Any, tmp_path: Path) -> None:
        output = tmp_path / "report.json"
        rc = bench.main(["--output-json", str(output)])
        assert rc == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["schema"] == bench.REPORT_SCHEMA
        assert output.with_suffix(".md").exists()

    def test_committed_report_schema_and_status(self) -> None:
        payload = json.loads(REPORT.read_text(encoding="utf-8"))
        assert payload["schema"] == "scpn-fusion-core.eped-pedestal-tiers-benchmark.v1"
        assert payload["status"] == "documented_divergence_no_parity_claim"
        assert payload["summary"]["pb_kbm_collapsed_all_cases"] is True
