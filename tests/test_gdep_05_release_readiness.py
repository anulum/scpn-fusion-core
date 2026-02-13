# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GDEP-05 Release Readiness Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for GDEP-05 release-readiness validation lane."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gdep_05_release_readiness.py"
SPEC = importlib.util.spec_from_file_location("gdep_05_release_readiness", MODULE_PATH)
assert SPEC and SPEC.loader
gdep_05_release_readiness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gdep_05_release_readiness)


def test_tracker_parser_extracts_known_statuses() -> None:
    statuses = gdep_05_release_readiness.parse_tracker_statuses(
        ROOT / "docs" / "PHASE2_ADVANCED_RFC_TRACKER.md"
    )
    assert statuses.get("GDEP-05") == "Done"
    assert statuses.get("GDEP-03") == "Done"


def test_phase3_parser_extracts_s2_s3_s4_statuses() -> None:
    statuses = gdep_05_release_readiness.parse_phase3_active_statuses(
        ROOT / "docs" / "PHASE3_EXECUTION_REGISTRY.md"
    )
    assert statuses.get("S2-001") == "Completed"
    assert statuses.get("S2-008") in {"Completed", "In progress"}
    assert statuses.get("S3-001") == "Completed"
    assert statuses.get("S3-006") in {"Completed", "In progress"}
    assert statuses.get("S4-001") == "Completed"
    assert statuses.get("S4-004") in {"Completed", "In progress"}


def test_gdep_05_campaign_passes_thresholds() -> None:
    out = gdep_05_release_readiness.run_campaign()
    assert out["passes_thresholds"] is True
    assert out["done_count"] == out["required_done_count"]
    assert out["changelog_phrase_present"] is True
    assert out["phase3_queue_parse_ok"] is True
    assert out["s2_queue_health"]["parse_ok"] is True
    assert out["s2_queue_health"]["completed_count"] >= 8
    assert out["s2_queue_health"]["in_progress_count"] >= 0
    assert out["s3_queue_health"]["parse_ok"] is True
    assert out["s3_queue_health"]["completed_count"] >= 5
    assert out["s3_queue_health"]["in_progress_count"] >= 0
    assert out["s4_queue_health"]["parse_ok"] is True
    assert out["s4_queue_health"]["completed_count"] >= 3
    assert out["s4_queue_health"]["in_progress_count"] >= 0


def test_render_markdown_contains_sections() -> None:
    report = gdep_05_release_readiness.generate_report()
    text = gdep_05_release_readiness.render_markdown(report)
    assert "# GDEP-05 Release Readiness" in text
    assert "Tracker Coverage" in text
    assert "Phase 3 Queue" in text
    assert "Active S3 tasks" in text
    assert "Active S4 tasks" in text
    assert "Overall pass" in text
