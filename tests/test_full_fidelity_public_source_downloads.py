# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Tests for public full-fidelity source download provenance."""

from __future__ import annotations

import json
from pathlib import Path

from validation.full_fidelity_end_to_end_campaign import run_campaign

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "validation" / "reports" / "full_fidelity_public_source_downloads.json"
REGISTRY = ROOT / "validation" / "reference_data" / "full_fidelity_public_sources.json"


def test_public_source_download_report_is_fail_closed_provenance_only() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["schema"] == "full-fidelity-public-source-downloads.v1"
    assert report["gitignored_cache"] is True
    assert report["cache_root"] == "data/external/full_fidelity_public_sources"
    assert report["all_reachable_downloads_completed"] is True
    assert "not parity artifacts" in report["note"]

    items = {item["name"]: item for item in report["items"]}
    expected = {
        "aurora",
        "aurora_docs",
        "cgyro_docs",
        "dream",
        "dream_docs",
        "freegs",
        "freegs_docs",
        "freegsnke",
        "gacode",
        "gacode_cgyro_docs",
        "genecode_landing",
        "genecode_main",
        "gs2",
        "gs2_docs",
        "gs2_user_manual",
    }
    assert expected.issubset(items)
    assert all(item["tracked_in_repo"] is False for item in items.values())
    assert all(item["status"] == "downloaded" for item in items.values())


def test_public_source_registry_keeps_cached_sources_separate_from_artifacts() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert (
        registry["download_report"]
        == "validation/reports/full_fidelity_public_source_downloads.json"
    )
    assert registry["raw_cache_gitignored"] is True
    assert registry["raw_cache_root"] == "data/external/full_fidelity_public_sources"

    for source in registry["sources"]:
        assert source["local_artifact_status"] == "not_ingested"
        assert source["raw_cache_items"]


def test_integrated_campaign_exposes_downloaded_but_not_validated_state() -> None:
    report = run_campaign()

    assert report["public_sources_cached"] is True
    assert report["reference_parity_ready"] is False
    assert report["status"] == "not_full_fidelity"
    assert (
        report["public_source_download_report"]
        == "validation/reports/full_fidelity_public_source_downloads.json"
    )
