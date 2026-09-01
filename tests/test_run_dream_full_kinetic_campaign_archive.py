# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Campaign Attempt-Archive Tests
"""Crash-journal integrity contracts for DREAM campaign attempt archives."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from tools.run_dream_full_kinetic_campaign import (
    ROOT,
    execute_campaign,
    prepare_campaign,
    reconcile_campaign,
)
from validation.dream_full_kinetic_execution_custody import (
    atomic_write_json,
    read_json_object,
    sha256_file,
)


@pytest.fixture
def durable_campaign_root() -> Generator[Path, None, None]:
    parent = ROOT / "data/external"
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="dream-archive-test-", dir=parent))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _failed_campaign(campaign_root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    manifest = prepare_campaign(
        campaign_root=campaign_root,
        run_id=run_id,
        dreami=Path("/usr/bin/false"),
    )
    campaign_dir = Path(manifest["campaign_dir"])
    executable = Path("/usr/bin/false")
    manifest["dream"]["dreami_path"] = str(executable)
    manifest["dream"]["dreami_sha256"] = sha256_file(executable)
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    failed = execute_campaign(campaign_dir, heartbeat_seconds=0.01)
    assert failed["status"] == "failed"
    return campaign_dir, failed


def _archive_payload(
    attempt_dir: Path,
    member: dict[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn-fusion.dream-attempt-archive.v1",
        "status": "collecting",
        "attempt_id": attempt_dir.name,
        "resolution": attempt_dir.parents[1].name,
        "member_state_before_reconciliation": member,
        "artifacts": {},
    }
    payload.update(overrides)
    return payload


def test_reconcile_rejects_symlinked_attempt_directory(durable_campaign_root: Path) -> None:
    campaign_dir, _failed = _failed_campaign(durable_campaign_root, "attempt-directory-link")
    attempts = campaign_dir / "veryfine/attempts"
    attempts.mkdir()
    target = durable_campaign_root / "attempt-target"
    target.mkdir()
    (attempts / "attempt-0001").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="attempt custody object is not a directory"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_symlinked_archive_manifest(durable_campaign_root: Path) -> None:
    campaign_dir, failed = _failed_campaign(durable_campaign_root, "archive-link")
    attempt_dir = campaign_dir / "veryfine/attempts/attempt-0001"
    attempt_dir.mkdir(parents=True)
    receipt = Path(failed["resolutions"]["veryfine"]["receipt_path"])
    (attempt_dir / "archive.json").symlink_to(receipt)
    with pytest.raises(ValueError, match="attempt archive is a symlink"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_attempt_archive_identity_drift(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, failed = _failed_campaign(durable_campaign_root, "archive-identity")
    member = failed["resolutions"]["veryfine"]
    attempt_dir = campaign_dir / "veryfine/attempts/attempt-0001"
    attempt_dir.mkdir(parents=True)
    atomic_write_json(
        attempt_dir / "archive.json",
        _archive_payload(attempt_dir, member, attempt_id="attempt-9999"),
    )
    with pytest.raises(ValueError, match="attempt archive identity mismatch"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_attempt_archive_resolution_drift(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, failed = _failed_campaign(durable_campaign_root, "archive-resolution")
    member = failed["resolutions"]["veryfine"]
    attempt_dir = campaign_dir / "veryfine/attempts/attempt-0001"
    attempt_dir.mkdir(parents=True)
    atomic_write_json(
        attempt_dir / "archive.json",
        _archive_payload(attempt_dir, member, resolution="superfine"),
    )
    with pytest.raises(ValueError, match="attempt archive resolution mismatch"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_attempt_archive_status_drift(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, failed = _failed_campaign(durable_campaign_root, "archive-status")
    member = failed["resolutions"]["veryfine"]
    attempt_dir = campaign_dir / "veryfine/attempts/attempt-0001"
    attempt_dir.mkdir(parents=True)
    atomic_write_json(
        attempt_dir / "archive.json",
        _archive_payload(attempt_dir, member, status="unknown"),
    )
    with pytest.raises(ValueError, match="attempt archive status is invalid"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_non_object_artifact_registry(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, failed = _failed_campaign(durable_campaign_root, "archive-artifacts")
    member = failed["resolutions"]["veryfine"]
    attempt_dir = campaign_dir / "veryfine/attempts/attempt-0001"
    attempt_dir.mkdir(parents=True)
    atomic_write_json(
        attempt_dir / "archive.json",
        _archive_payload(attempt_dir, member, artifacts=[]),
    )
    with pytest.raises(ValueError, match="archive artifacts are not an object"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_live_and_archived_copy_collision(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, failed = _failed_campaign(durable_campaign_root, "archive-collision")
    member = failed["resolutions"]["veryfine"]
    attempt_dir = campaign_dir / "veryfine/attempts/attempt-0001"
    attempt_dir.mkdir(parents=True)
    atomic_write_json(attempt_dir / "archive.json", _archive_payload(attempt_dir, member))
    shutil.copy2(Path(member["process_log_path"]), attempt_dir / "process.log")
    with pytest.raises(FileExistsError, match="both live and archived process_log_path"):
        reconcile_campaign(campaign_dir)


def test_reconcile_does_not_duplicate_registered_collecting_attempt(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, failed = _failed_campaign(durable_campaign_root, "registered-collecting")
    member = failed["resolutions"]["veryfine"]
    attempt_dir = campaign_dir / "veryfine/attempts/attempt-0001"
    attempt_dir.mkdir(parents=True)
    archive_path = attempt_dir / "archive.json"
    atomic_write_json(archive_path, _archive_payload(attempt_dir, member))
    member["attempts"] = [{"attempt_id": attempt_dir.name, "archive_path": str(archive_path)}]
    atomic_write_json(campaign_dir / "campaign.json", failed)

    reconciled = reconcile_campaign(campaign_dir)

    attempts = reconciled["resolutions"]["veryfine"]["attempts"]
    assert len(attempts) == 1
    assert read_json_object(archive_path)["status"] == "completed"
