# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Paper Provenance Manifest Tests
"""Regression tests for public FRC paper acquisition provenance."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPO_ROOT
    / "validation"
    / "reference_data"
    / "full_fidelity_public_artifacts"
    / "frc_reference_papers_manifest.json"
)


def _papers_by_key() -> dict[str, dict[str, object]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {str(paper["key"]): paper for paper in manifest["papers"]}


def test_frc_reference_manifest_preserves_open_blockers() -> None:
    papers = _papers_by_key()

    steinhauer = papers["steinhauer_2011_review_field_reversed_configurations"]
    slough = papers["slough_2011_merging_compression_frc_plasmoids"]

    assert steinhauer["download_status"] == "blocked_by_publisher_http_403"
    assert steinhauer["local_artifacts"] == []
    assert "Steinhauer" in str(steinhauer["notes"])

    assert slough["download_status"] == "publisher_served_landing_or_javascript_not_pdf"
    assert "No machine-readable Figure 5 trajectory" in str(slough["notes"])
    assert "B.11 remains blocked" in str(slough["notes"])


def test_romero_2018_topology_pdf_is_metadata_only_public_evidence() -> None:
    papers = _papers_by_key()

    romero = papers["romero_2018_frc_topology_alfvenic_transients"]

    assert romero["doi"] == "10.1038/s41467-018-03110-5"
    assert romero["download_status"] == "internal_open_access_pdf_metadata_recorded"
    assert romero["redistribution_license"] == "CC-BY-4.0"
    assert romero["local_artifacts"] == []
    assert romero["internal_pdf_sha256"] == (
        "63073de674e0ff09d4c4a150975653e2d943dce3615679634e112e5244569abc"
    )
    assert romero["internal_pdf_size_bytes"] == 4_588_529
    assert "not Steinhauer rotating-BVP closure" in str(romero["claim_boundary"])
    assert "not Slough 2011 Fig. 5 trajectory data" in str(romero["claim_boundary"])
