# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Full-Kinetic Campaign Runner Tests
"""Real DREAM-settings and process-boundary tests for the campaign runner."""

from __future__ import annotations

import importlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from tools.run_dream_full_kinetic_campaign import (
    DEFAULT_DREAM_ROOT,
    ROOT,
    campaign_launch_command,
    campaign_status,
    execute_campaign,
    main,
    prepare_campaign,
    reconcile_campaign,
)
from validation.dream_full_kinetic_execution_custody import (
    atomic_write_json,
    read_json_object,
    sha256_file,
)

h5py = importlib.import_module("h5py")


def _install_output_provider(campaign_dir: Path, *, exit_status: int = 14) -> Path:
    provider = campaign_dir / "dream-output-provider"
    provider.write_text(
        "\n".join(
            (
                f"#!{ROOT / '.venv/bin/python'}",
                "import sys",
                "import time",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(ROOT)!r})",
                "import h5py",
                "from tests.test_dream_full_kinetic_execution_custody import _complete_output",
                "from validation.dream_full_kinetic_execution_custody import frozen_output_contract",
                "settings_path = Path(sys.argv[1])",
                "with h5py.File(settings_path, 'r') as handle:",
                "    raw = handle['output/filename'][()].reshape(-1).tolist()",
                "output_path = Path(b''.join(raw).decode())",
                "contract = frozen_output_contract(settings_path.parent.name)",
                "_complete_output(output_path, contract)",
                "time.sleep(0.04)",
                f"raise SystemExit({exit_status})",
                "",
            )
        ),
        encoding="utf-8",
    )
    provider.chmod(0o700)
    return provider


def _install_sleeping_provider(campaign_dir: Path) -> Path:
    provider = campaign_dir / "sleeping-dream-provider"
    provider.write_text(
        "\n".join((f"#!{ROOT / '.venv/bin/python'}", "import time", "time.sleep(30)", "")),
        encoding="utf-8",
    )
    provider.chmod(0o700)
    return provider


def _bind_executable(campaign_dir: Path, executable: Path) -> None:
    manifest_path = campaign_dir / "campaign.json"
    stored = read_json_object(manifest_path)
    stored["dream"]["dreami_path"] = str(executable)
    stored["dream"]["dreami_sha256"] = sha256_file(executable)
    atomic_write_json(manifest_path, stored)


def _prepare_failed_campaign(campaign_root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    manifest = prepare_campaign(campaign_root=campaign_root, run_id=run_id)
    campaign_dir = Path(manifest["campaign_dir"])
    _bind_executable(campaign_dir, Path("/usr/bin/false"))
    failed = execute_campaign(campaign_dir, heartbeat_seconds=0.01)
    assert failed["status"] == "failed"
    return campaign_dir, failed


def _write_collecting_archive(
    attempt_dir: Path,
    member: dict[str, Any],
    **overrides: Any,
) -> Path:
    attempt_dir.mkdir(parents=True)
    payload: dict[str, Any] = {
        "schema": "scpn-fusion.dream-attempt-archive.v1",
        "status": "collecting",
        "attempt_id": attempt_dir.name,
        "resolution": attempt_dir.parents[1].name,
        "member_state_before_reconciliation": member,
        "artifacts": {},
    }
    payload.update(overrides)
    archive_path = attempt_dir / "archive.json"
    atomic_write_json(archive_path, payload)
    return archive_path


@pytest.fixture
def durable_campaign_root() -> Generator[Path, None, None]:
    parent = ROOT / "data/external"
    path = Path(tempfile.mkdtemp(prefix="dream-custody-test-", dir=parent))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def test_prepare_freezes_real_settings_without_starting_solver(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(
        campaign_root=durable_campaign_root,
        run_id="settings-only",
    )
    campaign_dir = Path(manifest["campaign_dir"])

    assert manifest["status"] == "prepared"
    assert manifest["resolution_order"] == ["veryfine", "superfine"]
    assert campaign_status(campaign_dir)["resolutions"] == {
        "veryfine": "prepared",
        "superfine": "prepared",
    }
    for resolution, expected in (("veryfine", (12, 48, 120)), ("superfine", (14, 56, 140))):
        member = manifest["resolutions"][resolution]
        assert not Path(member["partial_output_path"]).exists()
        with h5py.File(member["settings_path"], "r") as handle:
            assert int(handle["radialgrid/nr"][0]) == expected[0]
            assert int(handle["runawaygrid/nxi"][0]) == expected[1]
            assert int(handle["runawaygrid/np"][0]) == expected[2]
            output = b"".join(handle["output/filename"][()].reshape(-1).tolist()).decode()
            assert output == member["partial_output_path"]
    with pytest.raises(FileExistsError, match="campaign already exists"):
        prepare_campaign(campaign_root=durable_campaign_root, run_id="settings-only")


def test_execute_reaps_real_failure_and_stops_before_superfine(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="reaping-failure")
    campaign_dir = Path(manifest["campaign_dir"])
    failing_executable = Path("/usr/bin/false")
    _bind_executable(campaign_dir, failing_executable)

    result = execute_campaign(campaign_dir, heartbeat_seconds=0.01)

    assert result["status"] == "failed"
    assert result["resolutions"]["veryfine"]["status"] == "failed"
    assert result["resolutions"]["superfine"]["status"] == "prepared"
    receipt = read_json_object(Path(result["resolutions"]["veryfine"]["receipt_path"]))
    assert receipt["exit_status"] == 1
    assert receipt["observed_signal"] is None
    assert receipt["output_promoted"] is False
    assert campaign_status(campaign_dir)["status"] == "failed"


def test_failed_attempt_is_archived_before_safe_retry(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="recover-failure")
    campaign_dir = Path(manifest["campaign_dir"])
    _bind_executable(campaign_dir, Path("/usr/bin/false"))
    failed = execute_campaign(campaign_dir, heartbeat_seconds=0.01)
    assert failed["status"] == "failed"

    with pytest.raises(ValueError, match="not executable from status"):
        execute_campaign(campaign_dir, heartbeat_seconds=0.01)
    reconciled = reconcile_campaign(campaign_dir)
    member = reconciled["resolutions"]["veryfine"]
    archive = read_json_object(Path(member["attempts"][0]["archive_path"]))
    assert reconciled["status"] == "prepared"
    assert member["status"] == "prepared"
    assert archive["status"] == "completed"
    assert archive["artifacts"]["process_log_path"]["present"] is True
    assert archive["artifacts"]["receipt_path"]["present"] is True
    assert not Path(member["process_log_path"]).exists()
    assert not Path(member["receipt_path"]).exists()

    _bind_executable(campaign_dir, Path("/usr/bin/true"))
    retried = execute_campaign(campaign_dir, heartbeat_seconds=0.01)
    retry_receipt = read_json_object(Path(retried["resolutions"]["veryfine"]["receipt_path"]))
    assert retried["status"] == "failed"
    assert retry_receipt["exit_status"] == 0


def test_reconcile_resumes_an_interrupted_attempt_archive_journal(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="resume-archive")
    campaign_dir = Path(manifest["campaign_dir"])
    _bind_executable(campaign_dir, Path("/usr/bin/false"))
    failed = execute_campaign(campaign_dir, heartbeat_seconds=0.01)
    member = failed["resolutions"]["veryfine"]
    attempt_dir = campaign_dir / "veryfine/attempts/attempt-0001"
    attempt_dir.mkdir(parents=True)
    archive_path = attempt_dir / "archive.json"
    atomic_write_json(
        archive_path,
        {
            "schema": "scpn-fusion.dream-attempt-archive.v1",
            "status": "collecting",
            "attempt_id": "attempt-0001",
            "resolution": "veryfine",
            "member_state_before_reconciliation": member,
            "artifacts": {},
        },
    )
    os.replace(Path(member["process_log_path"]), attempt_dir / "process.log")

    reconciled = reconcile_campaign(campaign_dir)

    registered = reconciled["resolutions"]["veryfine"]["attempts"][0]
    archive = read_json_object(Path(registered["archive_path"]))
    assert archive["status"] == "completed"
    assert archive["artifacts"]["process_log_path"]["present"] is True
    assert archive["artifacts"]["receipt_path"]["present"] is True


def test_reconcile_rejects_symlinked_attempt_custody_root(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, _failed = _prepare_failed_campaign(durable_campaign_root, "attempt-root-link")
    target = durable_campaign_root / "attempt-target"
    target.mkdir()
    (campaign_dir / "veryfine/attempts").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="attempt custody root"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_orphaned_nonempty_attempt_directory(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, _failed = _prepare_failed_campaign(durable_campaign_root, "orphan-attempt")
    attempt_dir = campaign_dir / "veryfine/attempts/attempt-0001"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "orphan").write_text("unregistered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks its archive manifest"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_attempt_archive_schema_drift(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, failed = _prepare_failed_campaign(durable_campaign_root, "archive-schema")
    member = failed["resolutions"]["veryfine"]
    _write_collecting_archive(
        campaign_dir / "veryfine/attempts/attempt-0001",
        member,
        schema="wrong",
    )
    with pytest.raises(ValueError, match="attempt archive schema mismatch"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_multiple_unregistered_attempt_archives(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, _failed = _prepare_failed_campaign(durable_campaign_root, "multiple-attempts")
    (campaign_dir / "veryfine/attempts/attempt-0001").mkdir(parents=True)
    (campaign_dir / "veryfine/attempts/attempt-0002").mkdir()
    with pytest.raises(ValueError, match="multiple unregistered"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_archived_artifact_digest_drift(
    durable_campaign_root: Path,
) -> None:
    campaign_dir, failed = _prepare_failed_campaign(durable_campaign_root, "archive-digest")
    member = failed["resolutions"]["veryfine"]
    attempt_dir = campaign_dir / "veryfine/attempts/attempt-0001"
    archive_path = _write_collecting_archive(attempt_dir, member)
    log_target = attempt_dir / "process.log"
    os.replace(Path(member["process_log_path"]), log_target)
    archive = read_json_object(archive_path)
    archive["artifacts"]["process_log_path"] = {
        "present": True,
        "path": str(log_target),
        "bytes": log_target.stat().st_size,
        "sha256": "0" * 64,
    }
    atomic_write_json(archive_path, archive)
    with pytest.raises(ValueError, match="archived process_log_path custody drift"):
        reconcile_campaign(campaign_dir)


def test_reconcile_rejects_prepared_campaign(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="prepared-reconcile")
    with pytest.raises(ValueError, match="not reconcilable from status"):
        reconcile_campaign(Path(manifest["campaign_dir"]))


def test_launch_command_rejects_terminal_campaign(durable_campaign_root: Path) -> None:
    campaign_dir, _failed = _prepare_failed_campaign(durable_campaign_root, "failed-launch")
    with pytest.raises(ValueError, match="not launchable from status"):
        campaign_launch_command(campaign_dir)


def test_status_treats_missing_same_boot_pid_as_stale(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="missing-pid")
    campaign_dir = Path(manifest["campaign_dir"])
    current_boot = campaign_status(campaign_dir)["current_boot_id"]
    manifest.update(
        status="running",
        execution_boot_id=current_boot,
        supervisor_pid=2**31,
        supervisor_start_ticks=1,
    )
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    status = campaign_status(campaign_dir)
    assert status["active_running_process"] is False
    assert status["stale_running_state"] is True


def test_supervisor_signal_is_forwarded_and_recorded_as_interruption(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="signal-forwarding")
    campaign_dir = Path(manifest["campaign_dir"])
    _bind_executable(campaign_dir, _install_sleeping_provider(campaign_dir))
    process = subprocess.Popen(  # nosec B603
        [
            sys.executable,
            "-m",
            "tools.run_dream_full_kinetic_campaign",
            "execute",
            "--campaign-dir",
            str(campaign_dir),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        state = read_json_object(campaign_dir / "campaign.json")
        if state["resolutions"]["veryfine"]["status"] == "running":
            break
        if process.poll() is not None:
            pytest.fail(f"supervisor exited before signal: {process.communicate()!r}")
        time.sleep(0.01)
    else:
        process.kill()
        pytest.fail("supervisor did not publish a running child before timeout")

    os.kill(process.pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)
    terminal = read_json_object(campaign_dir / "campaign.json")
    receipt = read_json_object(Path(terminal["resolutions"]["veryfine"]["receipt_path"]))
    assert process.returncode == 0, (stdout, stderr)
    assert terminal["status"] == "interrupted"
    assert receipt["supervisor_signal"] == signal.SIGTERM
    assert receipt["observed_signal"] == signal.SIGTERM


def test_execute_promotes_both_complete_outputs_sequentially(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="complete-family")
    campaign_dir = Path(manifest["campaign_dir"])
    provider = _install_output_provider(campaign_dir)
    _bind_executable(campaign_dir, provider)

    result = execute_campaign(campaign_dir, heartbeat_seconds=0.01)

    assert result["status"] == "completed"
    for resolution in ("veryfine", "superfine"):
        member = result["resolutions"][resolution]
        assert member["status"] == "completed"
        assert Path(member["final_output_path"]).is_file()
        assert not Path(member["partial_output_path"]).exists()
        receipt = read_json_object(Path(member["receipt_path"]))
        assert receipt["exit_status"] == 14
        assert receipt["output_promoted"] is True
        assert receipt["output"]["validated"] is True
        assert receipt["output"]["sha256"] == member["output_sha256"]

    manifest_path = campaign_dir / "campaign.json"
    interrupted_terminal_commit = read_json_object(manifest_path)
    interrupted_terminal_commit["status"] = "running"
    interrupted_terminal_commit["execution_boot_id"] = "pre-reboot-boot-id"
    interrupted_terminal_commit.pop("completed_at_utc")
    atomic_write_json(manifest_path, interrupted_terminal_commit)
    reconciled = reconcile_campaign(campaign_dir)
    assert reconciled["status"] == "prepared"
    assert all(
        reconciled["resolutions"][resolution]["status"] == "completed"
        for resolution in ("veryfine", "superfine")
    )
    assert execute_campaign(campaign_dir, heartbeat_seconds=0.01)["status"] == "completed"


def test_execute_accepts_clean_exit_but_rejects_missing_output(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="missing-output")
    campaign_dir = Path(manifest["campaign_dir"])
    _bind_executable(campaign_dir, Path("/usr/bin/true"))

    result = execute_campaign(campaign_dir, heartbeat_seconds=0.01)

    receipt = read_json_object(Path(result["resolutions"]["veryfine"]["receipt_path"]))
    assert result["status"] == "failed"
    assert receipt["exit_status"] == 0
    assert "without a complete-output candidate" in receipt["failure_reason"]


def test_execute_rejects_nonpositive_heartbeat(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="heartbeat-guard")
    with pytest.raises(ValueError, match="heartbeat interval"):
        execute_campaign(Path(manifest["campaign_dir"]), heartbeat_seconds=0.0)


def test_execute_rejects_unreconciled_output_candidate(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="existing-output")
    campaign_dir = Path(manifest["campaign_dir"])
    Path(manifest["resolutions"]["veryfine"]["partial_output_path"]).touch()
    with pytest.raises(FileExistsError, match="unreconciled output"):
        execute_campaign(campaign_dir, heartbeat_seconds=0.01)


def test_prepare_rejects_invalid_identity_and_unpinned_checkout(
    durable_campaign_root: Path,
) -> None:
    with pytest.raises(ValueError, match="run-id"):
        prepare_campaign(campaign_root=durable_campaign_root, run_id="INVALID ID")
    with pytest.raises(ValueError, match="pinned"):
        prepare_campaign(
            campaign_root=durable_campaign_root,
            run_id="wrong-source",
            dream_root=ROOT,
            dreami=DEFAULT_DREAM_ROOT / "build/iface/dreami",
        )


def test_campaign_rejects_schema_drift(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="schema-drift")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["schema"] = "wrong"
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="schema mismatch"):
        campaign_status(campaign_dir)


def test_campaign_rejects_custody_schema_drift(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="custody-schema")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["custody_schema"] = "wrong"
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="custody schema mismatch"):
        campaign_status(campaign_dir)


def test_campaign_rejects_invalid_stored_run_identity(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="run-identity")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["run_id"] = "INVALID ID"
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="run identity is invalid"):
        campaign_status(campaign_dir)


def test_campaign_rejects_detached_stored_run_identity(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="run-directory")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["run_id"] = "different-valid-id"
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="run identity is detached"):
        campaign_status(campaign_dir)


def test_campaign_rejects_invalid_campaign_status(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="campaign-status")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["status"] = "unknown"
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="campaign status is invalid"):
        campaign_status(campaign_dir)


def test_campaign_rejects_detached_manifest(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="detached")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["campaign_dir"] = str(durable_campaign_root)
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="detached"):
        campaign_status(campaign_dir)


def test_campaign_rejects_resolution_order_drift(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="order-drift")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["resolution_order"] = ["superfine", "veryfine"]
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="resolution order"):
        campaign_status(campaign_dir)


def test_campaign_rejects_dreami_digest_drift(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="binary-drift")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["dream"]["dreami_sha256"] = "0" * 64
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="dreami digest drift"):
        execute_campaign(campaign_dir)


def test_campaign_rejects_member_path_escape(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="path-escape")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["resolutions"]["veryfine"]["process_log_path"] = str(
        durable_campaign_root / "escaped.log"
    )
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="escapes its frozen custody path"):
        campaign_status(campaign_dir)


def test_campaign_rejects_resolution_membership_drift(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="member-drift")
    campaign_dir = Path(manifest["campaign_dir"])
    del manifest["resolutions"]["superfine"]
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="membership mismatch"):
        campaign_status(campaign_dir)


def test_campaign_rejects_non_object_member(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="member-type")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["resolutions"]["veryfine"] = []
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="member is not an object"):
        campaign_status(campaign_dir)


def test_campaign_rejects_invalid_member_status(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="member-status")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["resolutions"]["veryfine"]["status"] = "unknown"
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="member status is invalid"):
        campaign_status(campaign_dir)


def test_campaign_rejects_invalid_attempt_registry(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="attempt-type")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["resolutions"]["veryfine"]["attempts"] = ["not-an-object"]
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="attempt registry"):
        campaign_status(campaign_dir)


def test_status_uses_boot_and_process_identity_before_reconciliation(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="process-identity")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest_path = campaign_dir / "campaign.json"
    status = campaign_status(campaign_dir)
    process_stat = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="ascii")
    start_ticks = int(process_stat.rpartition(")")[2].split()[19])
    manifest.update(
        status="running",
        execution_boot_id=status["current_boot_id"],
        supervisor_pid=os.getpid(),
        supervisor_start_ticks=start_ticks,
    )
    atomic_write_json(manifest_path, manifest)

    active = campaign_status(campaign_dir)
    assert active["active_running_process"] is True
    assert active["stale_running_state"] is False
    with pytest.raises(RuntimeError, match="still has a live"):
        reconcile_campaign(campaign_dir)

    manifest["supervisor_start_ticks"] = start_ticks + 1
    atomic_write_json(manifest_path, manifest)
    stale = campaign_status(campaign_dir)
    assert stale["active_running_process"] is False
    assert stale["stale_running_state"] is True
    assert stale["stale_running_after_reboot"] is False
    assert reconcile_campaign(campaign_dir)["status"] == "prepared"


def test_status_marks_running_state_stale_after_reboot(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="reboot-stale")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest.update(status="running", execution_boot_id="pre-reboot-boot-id")
    atomic_write_json(campaign_dir / "campaign.json", manifest)

    stale = campaign_status(campaign_dir)

    assert stale["stale_running_state"] is True
    assert stale["stale_running_after_reboot"] is True
    assert stale["active_running_process"] is False
    assert reconcile_campaign(campaign_dir)["status"] == "prepared"


def test_prepare_cli_publishes_machine_readable_manifest(
    durable_campaign_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "prepare",
                "--campaign-root",
                str(durable_campaign_root),
                "--run-id",
                "prepare-cli",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "prepared"
    assert payload["run_id"] == "prepare-cli"


def test_status_cli_reports_frozen_member_order(
    durable_campaign_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="status-cli")
    assert main(["status", "--campaign-dir", manifest["campaign_dir"]]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload["resolutions"]) == ["superfine", "veryfine"]
    assert payload["status"] == "prepared"


def test_launch_dry_run_cli_does_not_submit_systemd_unit(
    durable_campaign_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="launch-preview")
    campaign_dir = Path(manifest["campaign_dir"])
    expected = campaign_launch_command(campaign_dir)

    assert main(["launch", "--campaign-dir", str(campaign_dir), "--dry-run"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"command": expected, "submitted": False}
    assert expected[:2] == ["/usr/bin/systemd-run", "--user"]
    assert expected[-2:] == ["--campaign-dir", str(campaign_dir)]
    assert campaign_status(campaign_dir)["status"] == "prepared"


def test_execute_cli_reaps_backend_and_prints_terminal_state(
    durable_campaign_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="execute-cli")
    campaign_dir = Path(manifest["campaign_dir"])
    _bind_executable(campaign_dir, Path("/usr/bin/true"))

    assert main(["execute", "--campaign-dir", str(campaign_dir)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["resolutions"]["veryfine"]["status"] == "failed"


def test_reconcile_cli_archives_terminal_attempt_and_prints_prepared_state(
    durable_campaign_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="reconcile-cli")
    campaign_dir = Path(manifest["campaign_dir"])
    _bind_executable(campaign_dir, Path("/usr/bin/false"))
    assert execute_campaign(campaign_dir, heartbeat_seconds=0.01)["status"] == "failed"

    assert main(["reconcile", "--campaign-dir", str(campaign_dir)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "prepared"
    assert payload["resolutions"]["veryfine"]["attempts"][0]["attempt_id"] == "attempt-0001"


def test_prepare_rejects_non_executable_backend(durable_campaign_root: Path) -> None:
    backend = durable_campaign_root / "not-executable"
    backend.write_text("not executable\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an executable"):
        prepare_campaign(
            campaign_root=durable_campaign_root,
            run_id="non-executable",
            dreami=backend,
        )


def test_execute_rejects_lost_backend_execute_permission(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="lost-execute-bit")
    campaign_dir = Path(manifest["campaign_dir"])
    provider = _install_sleeping_provider(campaign_dir)
    _bind_executable(campaign_dir, provider)
    provider.chmod(0o600)
    with pytest.raises(ValueError, match="lost executable permission"):
        execute_campaign(campaign_dir)


def test_execute_rejects_dream_source_revision_drift(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="source-revision")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["dream"]["commit"] = "0" * 40
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="source revision drift"):
        execute_campaign(campaign_dir)


def test_execute_rejects_campaign_source_digest_drift(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="source-digest")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["source_digests"] = {}
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="campaign source digest drift"):
        execute_campaign(campaign_dir)


def test_execute_rejects_settings_digest_drift(durable_campaign_root: Path) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="settings-digest")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["resolutions"]["veryfine"]["settings_sha256"] = "0" * 64
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="settings digest drift"):
        execute_campaign(campaign_dir)


def test_execute_rejects_deck_manifest_digest_drift(
    durable_campaign_root: Path,
) -> None:
    manifest = prepare_campaign(campaign_root=durable_campaign_root, run_id="deck-digest")
    campaign_dir = Path(manifest["campaign_dir"])
    manifest["resolutions"]["veryfine"]["deck_manifest_sha256"] = "0" * 64
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    with pytest.raises(ValueError, match="deck-manifest digest drift"):
        execute_campaign(campaign_dir)
