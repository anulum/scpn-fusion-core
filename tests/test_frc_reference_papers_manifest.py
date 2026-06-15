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

    # Steinhauer Fig. 3 rotating-BVP closure is still blocked behind the AIP
    # HTTP 403. The only local artifacts are publisher-probe HTML landing pages
    # that record the block — never a redistributable PDF or digitised figure.
    assert steinhauer["download_status"] == "blocked_by_publisher_http_403"
    steinhauer_artifacts = steinhauer["local_artifacts"]
    assert steinhauer_artifacts, "expected the 403 publisher-probe provenance"
    assert all(
        artifact["content_type_detected"] == "HTML document" for artifact in steinhauer_artifacts
    ), "Steinhauer artifacts must stay publisher-probe HTML, never reference data"
    assert "Steinhauer" in str(steinhauer["notes"])

    # Slough 2011 Fig. 5 trajectory is still blocked: the publisher served a
    # landing / bot-validation page, not a PDF, so no machine-readable figure
    # was acquired. The status wording drifts across probe runs, so assert the
    # invariant ("not a PDF") rather than the exact landing-page reason.
    assert str(slough["download_status"]).endswith("not_pdf")
    assert "No machine-readable Figure 5 trajectory" in str(slough["notes"])
    assert "B.11 remains blocked" in str(slough["notes"])


def test_romero_2018_topology_pdf_is_open_access_public_evidence_only() -> None:
    papers = _papers_by_key()

    romero = papers["romero_2018_frc_topology_alfvenic_transients"]

    assert romero["doi"] == "10.1038/s41467-018-03110-5"
    # The open-access CC-BY PDF was downloaded and its provenance recorded.
    assert romero["download_status"] == "downloaded_open_access_pdf"
    assert romero["redistribution_license"] == "CC-BY-4.0"

    # The single local artifact is the open-access PDF, and its checksum/size
    # match the dedicated internal_pdf provenance fields.
    artifacts = romero["local_artifacts"]
    assert len(artifacts) == 1
    pdf = artifacts[0]
    assert pdf["content_type_detected"] == "PDF document"
    assert pdf["sha256"] == romero["internal_pdf_sha256"]
    assert pdf["size_bytes"] == romero["internal_pdf_size_bytes"]

    assert romero["internal_pdf_sha256"] == (
        "63073de674e0ff09d4c4a150975653e2d943dce3615679634e112e5244569abc"
    )
    assert romero["internal_pdf_size_bytes"] == 4_588_529

    # Fail-closed claim boundary: topology-inference evidence only, NOT the
    # Steinhauer rotating-BVP closure nor the Slough 2011 Fig. 5 trajectory.
    assert "not Steinhauer rotating-BVP closure" in str(romero["claim_boundary"])
    assert "not Slough 2011 Fig. 5 trajectory data" in str(romero["claim_boundary"])
