# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — full reproduction evidence tests
"""Tests for one-command full reproduction evidence reporting."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import scpn_fusion.repro as repro
import validation.full_fidelity_end_to_end_campaign as campaign_mod


def _ledger_fixture() -> dict[str, Any]:
    return {
        "schema": "full-fidelity-validation-ledger.v1",
        "campaign_report": "validation/reports/full_fidelity_end_to_end_campaign.json",
        "campaign_schema": "full-fidelity-end-to-end-campaign.v1",
        "ledger_publication_ready": False,
        "blocked_lane_count": 1,
        "accepted_full_fidelity_lane_count": 0,
        "source_reports": [
            {"path": "validation/reports/full_fidelity_end_to_end_campaign.json"},
            {"path": "validation/reports/full_fidelity_acceptance_benchmark.json"},
        ],
        "lanes": [
            {
                "lane": "gene_cgyro_gs2_nonlinear_gk_parity",
                "blocked_for_public_full_fidelity": True,
            },
            {
                "lane": "free_boundary_equilibrium_strict_parity",
                "blocked_for_public_full_fidelity": False,
            },
        ],
    }


def _campaign_fixture() -> dict[str, Any]:
    return {
        "schema": "full-fidelity-end-to-end-campaign.v1",
        "status": "not_full_fidelity",
        "acceptance_passed": False,
    }


def test_run_full_reproduction_writes_checksummed_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The reproduction command writes JSON and Markdown evidence with source checksums."""
    campaign_json = tmp_path / "campaign.json"
    campaign_json.write_text(
        json.dumps(
            {
                "schema": "full-fidelity-end-to-end-campaign.v1",
                "status": "not_full_fidelity",
            }
        ),
        encoding="utf-8",
    )
    acceptance_json = tmp_path / "acceptance.json"
    acceptance_json.write_text(
        json.dumps(
            {
                "schema": "full-fidelity-acceptance.v1",
                "status": "not_full_fidelity",
            }
        ),
        encoding="utf-8",
    )
    ledger_json = tmp_path / "ledger.json"
    ledger_json.write_text(
        json.dumps(
            {
                "schema": "full-fidelity-validation-ledger.v1",
                "status": "not_full_fidelity",
            }
        ),
        encoding="utf-8",
    )
    ledger_md = tmp_path / "ledger.md"
    ledger_md.write_text("# Ledger\n", encoding="utf-8")
    campaign_md = tmp_path / "campaign.md"
    campaign_md.write_text("# Campaign\n", encoding="utf-8")

    written_campaigns: list[dict[str, Any]] = []
    ledger = _ledger_fixture()
    ledger["source_reports"] = [
        {"path": campaign_json.relative_to(repro.REPO_ROOT).as_posix()}
        if campaign_json.is_relative_to(repro.REPO_ROOT)
        else {"path": str(campaign_json)},
        {"path": acceptance_json.relative_to(repro.REPO_ROOT).as_posix()}
        if acceptance_json.is_relative_to(repro.REPO_ROOT)
        else {"path": str(acceptance_json)},
    ]

    def fake_run_campaign() -> dict[str, Any]:
        return _campaign_fixture()

    def fake_write_reports(report: dict[str, Any]) -> None:
        written_campaigns.append(report)

    monkeypatch.setattr(campaign_mod, "run_campaign", fake_run_campaign)
    monkeypatch.setattr(campaign_mod, "write_reports", fake_write_reports)
    monkeypatch.setattr(campaign_mod, "build_public_ledger", lambda report: ledger)
    monkeypatch.setattr(campaign_mod, "LEDGER_JSON_REPORT", ledger_json)
    monkeypatch.setattr(campaign_mod, "LEDGER_MD_REPORT", ledger_md)
    monkeypatch.setattr(campaign_mod, "MD_REPORT", campaign_md)
    monkeypatch.setattr(repro, "_current_commit", lambda: "abc123")

    json_output = tmp_path / "evidence.json"
    markdown_output = tmp_path / "evidence.md"
    report = repro.run_full_reproduction(
        json_output=json_output,
        markdown_output=markdown_output,
    )

    persisted = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    campaign_digest = hashlib.sha256(campaign_json.read_bytes()).hexdigest()

    assert written_campaigns == [_campaign_fixture()]
    assert persisted == report
    assert report["schema"] == "scpn-fusion-full-reproduction-evidence.v1"
    assert report["source_commit"] == "abc123"
    assert report["full_fidelity_ready"] is False
    assert report["blocked_lanes"] == ["gene_cgyro_gs2_nonlinear_gk_parity"]
    assert report["missing_artifact_count"] == 0
    assert any(artifact["sha256"] == campaign_digest for artifact in report["artifacts"])
    assert report["evidence_payload_sha256"] == repro._canonical_json_sha256(
        {key: value for key, value in report.items() if key != "evidence_payload_sha256"}
    )
    assert "Full Reproduction Evidence" in markdown
    assert "scpn-fusion repro --full" in markdown


def test_artifact_record_handles_missing_invalid_and_non_object_json(tmp_path: Path) -> None:
    missing = repro._artifact_record(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{invalid", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    text = tmp_path / "report.md"
    text.write_text("ok", encoding="utf-8")

    invalid_record = repro._artifact_record(invalid)
    scalar_record = repro._artifact_record(scalar)
    text_record = repro._artifact_record(text)

    assert missing["exists"] is False
    assert missing["sha256"] is None
    assert invalid_record["status"] == "invalid_json"
    assert scalar_record["status"] == "non_object_json"
    assert text_record["schema"] is None
    assert text_record["bytes"] == 2


def test_current_commit_returns_unknown_when_git_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (args, kwargs)
        return subprocess.CompletedProcess(["git"], 1, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert repro._current_commit() == "unknown"


def test_current_commit_returns_git_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (args, kwargs)
        return subprocess.CompletedProcess(["git"], 0, stdout="def456\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert repro._current_commit() == "def456"


def test_source_artifacts_rejects_missing_path_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ledger = _ledger_fixture()
    ledger["source_reports"] = [{"path": 3}]
    ledger_json = tmp_path / "ledger.json"
    ledger_json.write_text("{}", encoding="utf-8")
    ledger_md = tmp_path / "ledger.md"
    ledger_md.write_text("# Ledger\n", encoding="utf-8")
    campaign_md = tmp_path / "campaign.md"
    campaign_md.write_text("# Campaign\n", encoding="utf-8")

    monkeypatch.setattr(campaign_mod, "LEDGER_JSON_REPORT", ledger_json)
    monkeypatch.setattr(campaign_mod, "LEDGER_MD_REPORT", ledger_md)
    monkeypatch.setattr(campaign_mod, "MD_REPORT", campaign_md)
    artifacts = repro._source_artifacts(ledger)

    assert [artifact["path"] for artifact in artifacts] == [
        str(ledger_json),
        str(ledger_md),
        str(campaign_md),
    ]


def test_display_path_uses_repo_relative_and_absolute_paths(tmp_path: Path) -> None:
    inside = repro.REPO_ROOT / "validation" / "reports" / "example.json"
    outside = tmp_path / "outside.json"

    assert repro._display_path(inside) == "validation/reports/example.json"
    assert repro._display_path(outside) == str(outside.resolve())


def test_markdown_renderer_lists_artifacts() -> None:
    report = {
        "schema": "scpn-fusion-full-reproduction-evidence.v1",
        "status": "not_full_fidelity",
        "acceptance_passed": False,
        "full_fidelity_ready": False,
        "blocked_lane_count": 1,
        "accepted_full_fidelity_lane_count": 0,
        "evidence_payload_sha256": "abc",
        "artifacts": [
            {
                "path": "validation/reports/example.json",
                "exists": True,
                "sha256": "def",
                "schema": "example.v1",
                "status": "not_full_fidelity",
            }
        ],
        "claim_boundary": "fail closed",
    }

    rendered = repro.render_reproduction_markdown(report)

    assert "`validation/reports/example.json`" in rendered
    assert "`example.v1`" in rendered
    assert "fail closed" in rendered
