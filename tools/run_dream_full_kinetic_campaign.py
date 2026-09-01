# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DREAM Full-Kinetic Campaign Runner
"""Prepare, supervise and recover the frozen DREAM veryfine/superfine campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal

# Required for fixed-argv solver and systemd supervision; no shell is used.
import subprocess  # nosec B404
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any, Final

from validation.dream_full_kinetic_execution_custody import (
    CUSTODY_SCHEMA,
    atomic_write_json,
    frozen_output_contract,
    inspect_dream_output,
    read_json_object,
    sha256_file,
    validate_durable_root,
)
from validation.reference_data.dream.full_kinetic_radial_parity_deck import (
    DREAM_COMMIT,
    build_settings,
    deck_manifest,
)


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MONOREPO_ROOT: Final[Path] = ROOT.parents[1]
DEFAULT_CAMPAIGN_ROOT: Final[Path] = (
    MONOREPO_ROOT / ".coordination/evidence/SCPN-FUSION-CORE/dream_full_kinetic_runs"
)
DEFAULT_DREAM_ROOT: Final[Path] = ROOT / "data/external/full_fidelity_public_sources/repos/dream"
CAMPAIGN_SCHEMA: Final[str] = "scpn-fusion.dream-full-kinetic-campaign.v1"
RESOLUTION_ORDER: Final[tuple[str, str]] = ("veryfine", "superfine")
ACCEPTED_EXIT_STATUSES: Final[frozenset[int]] = frozenset({0, 14})
RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
CAMPAIGN_STATUSES: Final[frozenset[str]] = frozenset(
    {"prepared", "running", "failed", "interrupted", "completed"}
)
MEMBER_FILENAMES: Final[dict[str, str]] = {
    "settings_path": "settings.h5",
    "deck_manifest_path": "deck_manifest.json",
    "partial_output_path": "output.partial.h5",
    "final_output_path": "output.h5",
    "process_log_path": "process.log",
    "receipt_path": "execution_receipt.json",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def _process_start_ticks(pid: int) -> int | None:
    """Read Linux process identity field 22 without confusing PID reuse."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        _prefix, separator, suffix = stat.rpartition(")")
        if not separator:
            return None
        return int(suffix.split()[19])
    except (FileNotFoundError, IndexError, PermissionError, ValueError):
        return None


def _process_identity_is_live(pid: object, start_ticks: object) -> bool:
    return (
        isinstance(pid, int)
        and isinstance(start_ticks, int)
        and _process_start_ticks(pid) == start_ticks
    )


def _git_head(repo: Path) -> str:
    result = subprocess.run(  # nosec B603
        ["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _campaign_path(campaign_root: Path, run_id: str) -> Path:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run-id must use lowercase letters, digits, dot, dash or underscore")
    root = validate_durable_root(campaign_root)
    return root / run_id


def _source_digests() -> dict[str, str]:
    paths = (
        Path("tools/run_dream_full_kinetic_campaign.py"),
        Path("validation/dream_full_kinetic_execution_custody.py"),
        Path("validation/dream_full_kinetic_reference.py"),
        Path("validation/reference_data/dream/full_kinetic_radial_parity_deck.py"),
        Path("validation/reference_data/dream/full_kinetic_radial_parity_lock.json"),
    )
    return {str(path): sha256_file(ROOT / path) for path in paths}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_campaign(
    *,
    campaign_root: Path,
    run_id: str,
    dream_root: Path = DEFAULT_DREAM_ROOT,
    dreami: Path | None = None,
) -> dict[str, Any]:
    """Freeze both durable settings files without launching the solver."""

    campaign_dir = _campaign_path(campaign_root, run_id)
    if campaign_dir.exists():
        raise FileExistsError(f"campaign already exists: {campaign_dir}")
    dream_root = dream_root.resolve(strict=True)
    if _git_head(dream_root) != DREAM_COMMIT:
        raise ValueError(f"DREAM checkout must be pinned to {DREAM_COMMIT}")
    executable = (dreami or dream_root / "build/iface/dreami").resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"dreami is not an executable regular file: {executable}")

    campaign_dir.mkdir(parents=True, exist_ok=False)
    resolutions: dict[str, dict[str, Any]] = {}
    for resolution in RESOLUTION_ORDER:
        member_dir = campaign_dir / resolution
        member_dir.mkdir()
        settings_path = member_dir / "settings.h5"
        partial_output = member_dir / "output.partial.h5"
        final_output = member_dir / "output.h5"
        manifest_path = member_dir / "deck_manifest.json"
        settings = build_settings(
            dream_root=dream_root,
            resolution=resolution,
            output=partial_output,
        )
        settings.save(str(settings_path))
        atomic_write_json(
            manifest_path,
            deck_manifest(
                resolution=resolution,
                settings_path=settings_path,
                output_path=partial_output,
            ),
        )
        resolutions[resolution] = {
            "status": "prepared",
            "settings_path": str(settings_path),
            "settings_sha256": sha256_file(settings_path),
            "deck_manifest_path": str(manifest_path),
            "deck_manifest_sha256": sha256_file(manifest_path),
            "partial_output_path": str(partial_output),
            "final_output_path": str(final_output),
            "process_log_path": str(member_dir / "process.log"),
            "receipt_path": str(member_dir / "execution_receipt.json"),
        }

    manifest: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "custody_schema": CUSTODY_SCHEMA,
        "run_id": run_id,
        "campaign_dir": str(campaign_dir),
        "status": "prepared",
        "created_at_utc": _utc_now(),
        "prepared_boot_id": _boot_id(),
        "resolution_order": list(RESOLUTION_ORDER),
        "repository_head": _git_head(ROOT),
        "source_digests": _source_digests(),
        "dream": {
            "root": str(dream_root),
            "commit": DREAM_COMMIT,
            "dreami_path": str(executable),
            "dreami_sha256": sha256_file(executable),
        },
        "resolutions": resolutions,
    }
    atomic_write_json(campaign_dir / "campaign.json", manifest)
    _fsync_directory(campaign_dir)
    return manifest


def _load_campaign(campaign_dir: Path) -> tuple[Path, dict[str, Any]]:
    resolved = validate_durable_root(campaign_dir).resolve(strict=True)
    manifest = read_json_object(resolved / "campaign.json")
    if manifest.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError("campaign schema mismatch")
    if manifest.get("custody_schema") != CUSTODY_SCHEMA:
        raise ValueError("campaign custody schema mismatch")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("campaign run identity is invalid")
    if resolved.name != run_id:
        raise ValueError("campaign run identity is detached from its directory")
    if manifest.get("status") not in CAMPAIGN_STATUSES:
        raise ValueError("campaign status is invalid")
    if Path(str(manifest.get("campaign_dir"))).resolve() != resolved:
        raise ValueError("campaign manifest is detached from its directory")
    if manifest.get("resolution_order") != list(RESOLUTION_ORDER):
        raise ValueError("campaign resolution order is not veryfine then superfine")
    members = manifest.get("resolutions")
    if not isinstance(members, dict) or set(members) != set(RESOLUTION_ORDER):
        raise ValueError("campaign resolution membership mismatch")
    for resolution in RESOLUTION_ORDER:
        member = members[resolution]
        if not isinstance(member, dict):
            raise ValueError(f"{resolution} campaign member is not an object")
        if member.get("status") not in CAMPAIGN_STATUSES:
            raise ValueError(f"{resolution} campaign member status is invalid")
        attempts = member.get("attempts", [])
        if not isinstance(attempts, list) or not all(isinstance(item, dict) for item in attempts):
            raise ValueError(f"{resolution} attempt registry is not a list of objects")
        for field, filename in MEMBER_FILENAMES.items():
            expected = resolved / resolution / filename
            if Path(str(member.get(field))).resolve(strict=False) != expected:
                raise ValueError(f"{resolution} {field} escapes its frozen custody path")
    return resolved, manifest


def _verify_prepared_custody(manifest: dict[str, Any]) -> None:
    dream = manifest["dream"]
    dreami = Path(dream["dreami_path"])
    if not os.access(dreami, os.X_OK):
        raise ValueError("dreami lost executable permission")
    if sha256_file(dreami) != dream["dreami_sha256"]:
        raise ValueError("dreami digest drift")
    if _git_head(Path(dream["root"])) != dream["commit"] or dream["commit"] != DREAM_COMMIT:
        raise ValueError("DREAM source revision drift")
    if _source_digests() != manifest["source_digests"]:
        raise ValueError("campaign source digest drift; prepare a new run before execution")
    for resolution in RESOLUTION_ORDER:
        member = manifest["resolutions"][resolution]
        if sha256_file(Path(member["settings_path"])) != member["settings_sha256"]:
            raise ValueError(f"{resolution} settings digest drift")
        if sha256_file(Path(member["deck_manifest_path"])) != member["deck_manifest_sha256"]:
            raise ValueError(f"{resolution} deck-manifest digest drift")


def _record_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = _utc_now()
    atomic_write_json(path, manifest)


def _campaign_has_live_process(manifest: dict[str, Any]) -> bool:
    if manifest.get("execution_boot_id") != _boot_id():
        return False
    if _process_identity_is_live(
        manifest.get("supervisor_pid"), manifest.get("supervisor_start_ticks")
    ):
        return True
    return any(
        member.get("status") == "running"
        and _process_identity_is_live(member.get("child_pid"), member.get("child_start_ticks"))
        for member in manifest["resolutions"].values()
    )


def _next_attempt_dir(member_dir: Path, member: dict[str, Any]) -> Path:
    attempts_dir = member_dir / "attempts"
    attempts_dir.mkdir(exist_ok=True)
    if attempts_dir.is_symlink() or not attempts_dir.is_dir():
        raise ValueError(f"attempt custody root is not a regular directory: {attempts_dir}")
    registered = {
        str(item.get("archive_path"))
        for item in member.get("attempts", [])
        if isinstance(item, dict)
    }
    pending: list[Path] = []
    for candidate in sorted(attempts_dir.glob("attempt-*")):
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(f"attempt custody object is not a directory: {candidate}")
        archive_path = candidate / "archive.json"
        if archive_path.is_symlink():
            raise ValueError(f"attempt archive is a symlink: {archive_path}")
        if not archive_path.exists():
            if any(candidate.iterdir()):
                raise ValueError(f"attempt directory lacks its archive manifest: {candidate}")
            pending.append(candidate)
            continue
        archive = read_json_object(archive_path)
        if archive.get("schema") != "scpn-fusion.dream-attempt-archive.v1":
            raise ValueError(f"attempt archive schema mismatch: {archive_path}")
        if archive.get("attempt_id") != candidate.name:
            raise ValueError(f"attempt archive identity mismatch: {archive_path}")
        if archive.get("resolution") != member_dir.name:
            raise ValueError(f"attempt archive resolution mismatch: {archive_path}")
        status = archive.get("status")
        if status not in {"collecting", "completed"}:
            raise ValueError(f"attempt archive status is invalid: {archive_path}")
        if status == "collecting" or (
            status == "completed" and str(archive_path) not in registered
        ):
            pending.append(candidate)
    if len(pending) > 1:
        raise ValueError(f"multiple unregistered attempt archives exist under {attempts_dir}")
    if pending:
        return pending[0]
    indexes = [
        int(candidate.name.removeprefix("attempt-"))
        for candidate in attempts_dir.glob("attempt-[0-9][0-9][0-9][0-9]")
        if candidate.name.removeprefix("attempt-").isdigit()
    ]
    attempt_dir = attempts_dir / f"attempt-{max(indexes, default=0) + 1:04d}"
    attempt_dir.mkdir()
    _fsync_directory(attempts_dir)
    return attempt_dir


def _archive_member_attempt(
    campaign_dir: Path,
    resolution: str,
    member: dict[str, Any],
) -> dict[str, Any]:
    """Move incomplete attempt artefacts into a resumable, checksummed archive."""

    member_dir = campaign_dir / resolution
    attempt_dir = _next_attempt_dir(member_dir, member)
    archive_path = attempt_dir / "archive.json"
    if archive_path.exists():
        archive = read_json_object(archive_path)
    else:
        archive = {
            "schema": "scpn-fusion.dream-attempt-archive.v1",
            "status": "collecting",
            "attempt_id": attempt_dir.name,
            "resolution": resolution,
            "created_at_utc": _utc_now(),
            "member_state_before_reconciliation": dict(member),
            "artifacts": {},
        }
        atomic_write_json(archive_path, archive)

    if not isinstance(archive.get("artifacts"), dict):
        raise ValueError(f"attempt archive artifacts are not an object: {archive_path}")

    artifact_fields = ("partial_output_path", "process_log_path", "receipt_path")
    for field in artifact_fields:
        source = Path(member[field])
        target = attempt_dir / source.name
        if source.exists() and target.exists():
            raise FileExistsError(f"both live and archived {field} exist for {resolution}")
        if source.exists():
            source_digest = sha256_file(source)
            source_bytes = source.stat().st_size
            os.replace(source, target)
            _fsync_directory(source.parent)
            _fsync_directory(attempt_dir)
            if sha256_file(target) != source_digest:
                raise ValueError(f"{resolution} {field} digest changed during archival")
            archive["artifacts"][field] = {
                "present": True,
                "path": str(target),
                "bytes": source_bytes,
                "sha256": source_digest,
            }
        elif target.exists():
            digest = sha256_file(target)
            size = target.stat().st_size
            recorded = archive["artifacts"].get(field)
            if isinstance(recorded, dict) and recorded.get("present") is True:
                if recorded.get("sha256") != digest or recorded.get("bytes") != size:
                    raise ValueError(f"{resolution} archived {field} custody drift")
            archive["artifacts"][field] = {
                "present": True,
                "path": str(target),
                "bytes": size,
                "sha256": digest,
            }
        else:
            archive["artifacts"][field] = {"present": False}
        atomic_write_json(archive_path, archive)

    archive["status"] = "completed"
    archive["completed_at_utc"] = _utc_now()
    atomic_write_json(archive_path, archive)
    return {
        "attempt_id": attempt_dir.name,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
    }


def reconcile_campaign(campaign_dir: Path) -> dict[str, Any]:
    """Archive a terminal or stale attempt and restore a safe prepared state."""

    resolved, manifest = _load_campaign(campaign_dir)
    _verify_prepared_custody(manifest)
    previous_status = manifest.get("status")
    if previous_status == "running" and _campaign_has_live_process(manifest):
        raise RuntimeError("campaign still has a live supervisor or DREAM child")
    if previous_status not in {"running", "failed", "interrupted"}:
        raise ValueError(f"campaign is not reconcilable from status {previous_status!r}")

    reconciled_members: list[dict[str, Any]] = []
    runtime_fields = {
        "child_pid",
        "child_process_group",
        "child_start_ticks",
        "execution_boot_id",
        "started_at_utc",
        "heartbeat_at_utc",
        "heartbeat_monotonic",
        "elapsed_wall_seconds",
        "completed_at_utc",
        "output_sha256",
        "receipt_sha256",
    }
    for resolution in RESOLUTION_ORDER:
        member = manifest["resolutions"][resolution]
        if member["status"] == "completed":
            inspect_dream_output(
                Path(member["final_output_path"]), frozen_output_contract(resolution)
            )
            continue
        has_attempt = any(
            Path(member[field]).exists()
            for field in ("partial_output_path", "process_log_path", "receipt_path")
        ) or member["status"] in {"running", "failed", "interrupted"}
        if has_attempt:
            archived = _archive_member_attempt(resolved, resolution, member)
            attempts = member.setdefault("attempts", [])
            if not any(item.get("archive_path") == archived["archive_path"] for item in attempts):
                attempts.append(archived)
            reconciled_members.append({"resolution": resolution, **archived})
        for field in runtime_fields:
            member.pop(field, None)
        member["status"] = "prepared"
        member["reconciled_at_utc"] = _utc_now()

    for field in (
        "supervisor_pid",
        "supervisor_start_ticks",
        "execution_boot_id",
        "started_at_utc",
        "completed_at_utc",
    ):
        manifest.pop(field, None)
    manifest["status"] = "prepared"
    manifest.setdefault("reconciliations", []).append(
        {
            "at_utc": _utc_now(),
            "boot_id": _boot_id(),
            "previous_status": previous_status,
            "members": reconciled_members,
        }
    )
    _record_manifest(resolved / "campaign.json", manifest)
    return manifest


def execute_campaign(campaign_dir: Path, *, heartbeat_seconds: float = 60.0) -> dict[str, Any]:
    """Run and reap veryfine then superfine, promoting only validated outputs."""

    if heartbeat_seconds <= 0.0:
        raise ValueError("heartbeat interval must be positive")
    resolved, manifest = _load_campaign(campaign_dir)
    _verify_prepared_custody(manifest)
    manifest_path = resolved / "campaign.json"
    if manifest["status"] != "prepared":
        raise ValueError(f"campaign is not executable from status {manifest['status']!r}")

    interrupted_signal: int | None = None
    child: subprocess.Popen[bytes] | None = None

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        nonlocal interrupted_signal
        interrupted_signal = signum
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    previous_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        manifest["status"] = "running"
        manifest["execution_boot_id"] = _boot_id()
        manifest["supervisor_pid"] = os.getpid()
        manifest["supervisor_start_ticks"] = _process_start_ticks(os.getpid())
        manifest["started_at_utc"] = _utc_now()
        _record_manifest(manifest_path, manifest)
        for resolution in RESOLUTION_ORDER:
            member = manifest["resolutions"][resolution]
            if member["status"] == "completed":
                inspect_dream_output(
                    Path(member["final_output_path"]), frozen_output_contract(resolution)
                )
                continue
            partial_output = Path(member["partial_output_path"])
            final_output = Path(member["final_output_path"])
            if final_output.exists() or partial_output.exists():
                raise FileExistsError(f"unreconciled output exists for {resolution}")
            log_path = Path(member["process_log_path"])
            started_monotonic = time.monotonic()
            with log_path.open("xb", buffering=0) as log_stream:
                child = subprocess.Popen(  # nosec B603
                    [manifest["dream"]["dreami_path"], member["settings_path"]],
                    cwd=partial_output.parent,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                member.update(
                    {
                        "status": "running",
                        "started_at_utc": _utc_now(),
                        "child_pid": child.pid,
                        "child_process_group": child.pid,
                        "child_start_ticks": _process_start_ticks(child.pid),
                        "execution_boot_id": _boot_id(),
                    }
                )
                _record_manifest(manifest_path, manifest)
                while child.poll() is None:
                    time.sleep(min(heartbeat_seconds, 1.0))
                    if time.monotonic() - float(member.get("heartbeat_monotonic", 0.0)) >= (
                        heartbeat_seconds
                    ):
                        member["elapsed_wall_seconds"] = time.monotonic() - started_monotonic
                        member["heartbeat_at_utc"] = _utc_now()
                        member["heartbeat_monotonic"] = time.monotonic()
                        _record_manifest(manifest_path, manifest)
                exit_status = child.wait()
                os.fsync(log_stream.fileno())
            elapsed = time.monotonic() - started_monotonic
            receipt: dict[str, Any] = {
                "schema": "scpn-fusion.dream-execution-receipt.v1",
                "resolution": resolution,
                "command": [manifest["dream"]["dreami_path"], member["settings_path"]],
                "child_pid": child.pid,
                "exit_status": exit_status,
                "observed_signal": -exit_status if exit_status < 0 else None,
                "supervisor_signal": interrupted_signal,
                "elapsed_wall_seconds": elapsed,
                "process_log_path": str(log_path),
                "process_log_sha256": sha256_file(log_path),
                "terminal_at_utc": _utc_now(),
                "output_promoted": False,
            }
            if interrupted_signal is not None:
                member["status"] = "interrupted"
                manifest["status"] = "interrupted"
            elif exit_status not in ACCEPTED_EXIT_STATUSES:
                member["status"] = "failed"
                manifest["status"] = "failed"
            elif not partial_output.is_file():
                receipt["failure_reason"] = "solver exited without a complete-output candidate"
                member["status"] = "failed"
                manifest["status"] = "failed"
            else:
                inspection = inspect_dream_output(
                    partial_output, frozen_output_contract(resolution)
                )
                os.replace(partial_output, final_output)
                _fsync_directory(final_output.parent)
                inspection["path"] = str(final_output.resolve())
                if sha256_file(final_output) != inspection["sha256"]:
                    raise ValueError("output digest changed during atomic promotion")
                receipt["output_promoted"] = True
                receipt["output"] = inspection
                member["status"] = "completed"
                member["completed_at_utc"] = _utc_now()
                member["output_sha256"] = inspection["sha256"]
            atomic_write_json(Path(member["receipt_path"]), receipt)
            member["receipt_sha256"] = sha256_file(Path(member["receipt_path"]))
            _record_manifest(manifest_path, manifest)
            if member["status"] != "completed":
                break
        if all(manifest["resolutions"][name]["status"] == "completed" for name in RESOLUTION_ORDER):
            manifest["status"] = "completed"
            manifest["completed_at_utc"] = _utc_now()
            _record_manifest(manifest_path, manifest)
        return manifest
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def campaign_launch_command(campaign_dir: Path) -> list[str]:
    """Build the exact custody-checked user-systemd launch command."""

    resolved, manifest = _load_campaign(campaign_dir)
    _verify_prepared_custody(manifest)
    if manifest["status"] != "prepared":
        raise ValueError(f"campaign is not launchable from status {manifest['status']!r}")
    unit_suffix = re.sub(r"[^a-zA-Z0-9_.-]", "-", manifest["run_id"])
    unit = f"scpn-dream-full-kinetic-{unit_suffix}"
    command = [
        "/usr/bin/systemd-run",
        "--user",
        f"--unit={unit}",
        "--collect",
        "--property=Type=exec",
        "--property=KillMode=mixed",
        "--property=TimeoutStopSec=120",
        f"--working-directory={ROOT}",
        sys.executable,
        "-m",
        "tools.run_dream_full_kinetic_campaign",
        "execute",
        "--campaign-dir",
        str(resolved),
    ]
    return command


def launch_campaign(campaign_dir: Path) -> list[str]:
    """Launch the durable supervisor as a user-systemd service."""

    command = campaign_launch_command(campaign_dir)
    subprocess.run(command, check=True, timeout=30)  # nosec B603
    return command


def campaign_status(campaign_dir: Path) -> dict[str, Any]:
    """Return durable campaign state with explicit stale-running detection."""

    resolved, manifest = _load_campaign(campaign_dir)
    current_boot = _boot_id()
    active = manifest.get("status") == "running" and _campaign_has_live_process(manifest)
    stale = manifest.get("status") == "running" and not active
    stale_after_reboot = stale and manifest.get("execution_boot_id") != current_boot
    return {
        "campaign_dir": str(resolved),
        "status": manifest["status"],
        "stale_running_state": stale,
        "stale_running_after_reboot": stale_after_reboot,
        "active_running_process": active,
        "current_boot_id": current_boot,
        "execution_boot_id": manifest.get("execution_boot_id"),
        "resolutions": {name: manifest["resolutions"][name]["status"] for name in RESOLUTION_ORDER},
    }


def main(argv: list[str] | None = None) -> int:
    """Dispatch the prepare, launch, execute, reconcile and status lifecycle commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze durable settings without running DREAM")
    prepare.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--dream-root", type=Path, default=DEFAULT_DREAM_ROOT)
    prepare.add_argument("--dreami", type=Path)
    launch = commands.add_parser("launch")
    launch.add_argument("--campaign-dir", type=Path, required=True)
    launch.add_argument("--dry-run", action="store_true")
    for name in ("execute", "reconcile", "status"):
        command = commands.add_parser(name)
        command.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_campaign(
            campaign_root=args.campaign_root,
            run_id=args.run_id,
            dream_root=args.dream_root,
            dreami=args.dreami,
        )
    elif args.command == "launch":
        if args.dry_run:
            result = {
                "command": campaign_launch_command(args.campaign_dir),
                "submitted": False,
            }
        else:
            result = {"command": launch_campaign(args.campaign_dir), "submitted": True}
    elif args.command == "execute":
        result = execute_campaign(args.campaign_dir)
    elif args.command == "reconcile":
        result = reconcile_campaign(args.campaign_dir)
    else:
        result = campaign_status(args.campaign_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
