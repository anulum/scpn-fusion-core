# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Commit trailer verifier tests
"""Tests for the commit-message trailer checker."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools import check_commit_trailers

REQUIRED_AUTHORSHIP_LINE = check_commit_trailers.REQUIRED_AUTHORSHIP_LINE
VALID_SEAT_TRAILER = "Seat: 14753"


def _message(*lines: str) -> str:
    """Build a commit message with a trailing newline."""
    return "\n".join(lines) + "\n"


def _message_file(tmp_path: Path, text: str) -> Path:
    """Write a temporary commit-message file."""
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(text, encoding="utf-8")
    return path


def test_commit_msg_hook_accepts_valid_trailer_block(tmp_path: Path) -> None:
    """A valid local agent commit message passes hook validation."""
    path = _message_file(
        tmp_path,
        _message(
            "test: add trailer guard",
            "",
            VALID_SEAT_TRAILER,
            "",
            REQUIRED_AUTHORSHIP_LINE,
        ),
    )

    assert check_commit_trailers.main([str(path)]) == 0


def test_commit_msg_hook_rejects_missing_authorship_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hook rejects messages without the required authorship line."""
    path = _message_file(tmp_path, _message("test: add trailer guard", "", VALID_SEAT_TRAILER))

    assert check_commit_trailers.main([str(path)]) == 1
    assert f"missing `{REQUIRED_AUTHORSHIP_LINE}` authorship line" in capsys.readouterr().err


def test_commit_msg_hook_rejects_duplicate_authorship_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exactly one authorship line is accepted."""
    path = _message_file(
        tmp_path,
        _message(
            "test: add trailer guard",
            "",
            VALID_SEAT_TRAILER,
            "",
            REQUIRED_AUTHORSHIP_LINE,
            REQUIRED_AUTHORSHIP_LINE,
        ),
    )

    assert check_commit_trailers.main([str(path)]) == 1
    assert "expected exactly one authorship line" in capsys.readouterr().err


def test_commit_msg_hook_rejects_missing_seat_trailer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hook rejects messages without the required seat trailer."""
    path = _message_file(
        tmp_path, _message("test: add trailer guard", "", REQUIRED_AUTHORSHIP_LINE)
    )

    assert check_commit_trailers.main([str(path)]) == 1
    assert "missing `Seat: <seat-id>` trailer" in capsys.readouterr().err


def test_commit_msg_hook_rejects_duplicate_seat_trailers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exactly one seat trailer is accepted."""
    path = _message_file(
        tmp_path,
        _message(
            "test: add trailer guard",
            "",
            "Seat: 14753",
            "Seat: rf01",
            "",
            REQUIRED_AUTHORSHIP_LINE,
        ),
    )

    assert check_commit_trailers.main([str(path)]) == 1
    assert "expected exactly one `Seat: <seat-id>` trailer" in capsys.readouterr().err


@pytest.mark.parametrize("seat_line", ["Seat:", "Seat: codex-14753", "Seat: claude-rf01"])
def test_commit_msg_hook_rejects_invalid_or_vendor_prefixed_seats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], seat_line: str
) -> None:
    """Seat IDs must be syntactically valid and vendor-neutral."""
    path = _message_file(
        tmp_path,
        _message("test: add trailer guard", "", seat_line, "", REQUIRED_AUTHORSHIP_LINE),
    )

    assert check_commit_trailers.main([str(path)]) == 1
    err = capsys.readouterr().err
    assert (
        "invalid `Seat: <seat-id>` trailer" in err
        or "vendor-prefixed `Seat:` trailer is forbidden" in err
    )


def test_commit_msg_hook_rejects_split_trailer_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The seat trailer must stay in the authorship trailer block."""
    path = _message_file(
        tmp_path,
        _message(
            "test: add trailer guard",
            "",
            VALID_SEAT_TRAILER,
            "",
            "Body text splits the trailer block.",
            "",
            REQUIRED_AUTHORSHIP_LINE,
        ),
    )

    assert check_commit_trailers.main([str(path)]) == 1
    assert "`Seat:` trailer must immediately precede the authorship line" in capsys.readouterr().err


def test_commit_msg_hook_rejects_public_subject_puffery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Public commit subjects cannot carry self-applied quality claims."""
    path = _message_file(
        tmp_path,
        _message(
            "test: add robust trailer guard",
            "",
            VALID_SEAT_TRAILER,
            "",
            REQUIRED_AUTHORSHIP_LINE,
        ),
    )

    assert check_commit_trailers.main([str(path)]) == 1
    assert "banned public subject term(s): robust" in capsys.readouterr().err


def test_banned_terms_are_not_matched_inside_body() -> None:
    """Body prose may cite a removed term while the subject stays clean."""
    message = _message(
        "test: add trailer guard",
        "",
        "This removes robust wording from a public page.",
        "",
        VALID_SEAT_TRAILER,
        "",
        REQUIRED_AUTHORSHIP_LINE,
    )

    assert check_commit_trailers.message_violations(message) == []


def test_missing_arguments_return_invocation_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Running without hook path or audit range fails closed."""
    assert check_commit_trailers.main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_help_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """The command exposes argparse help for operators."""
    with pytest.raises(SystemExit) as exc_info:
        check_commit_trailers.main(["--help"])
    assert exc_info.value.code == 0
    assert "Validate SCPN Fusion Core commit trailers" in capsys.readouterr().out


def test_audit_range_passes_clean_fake_history(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Audit mode validates each commit message in the requested range."""
    monkeypatch.setattr(check_commit_trailers, "_resolve_git_executable", lambda: sys.executable)
    clean_sha = "abc1234567890"

    def fake_run_git(_git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
        if args == ("rev-list", "HEAD"):
            return subprocess.CompletedProcess(args, 0, stdout=f"{clean_sha}\n")
        if args == ("log", "-1", "--format=%B", clean_sha):
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=_message(
                    "test: add trailer guard",
                    "",
                    VALID_SEAT_TRAILER,
                    "",
                    REQUIRED_AUTHORSHIP_LINE,
                ),
            )
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(check_commit_trailers, "_run_git", fake_run_git)

    assert check_commit_trailers.main(["--audit-range", "HEAD"]) == 0
    assert "Trailer audit passed." in capsys.readouterr().out


def test_audit_range_reports_policy_violations(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Audit mode reports commit-message violations with short SHAs."""
    monkeypatch.setattr(check_commit_trailers, "_resolve_git_executable", lambda: sys.executable)
    failing_sha = "bad9999000000"

    def fake_run_git(_git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
        if args == ("rev-list", "HEAD"):
            return subprocess.CompletedProcess(args, 0, stdout=f"{failing_sha}\n")
        if args == ("log", "-1", "--format=%B", failing_sha):
            return subprocess.CompletedProcess(args, 0, stdout=_message("test: add trailer guard"))
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(check_commit_trailers, "_run_git", fake_run_git)

    assert check_commit_trailers.main(["--audit-range", "HEAD"]) == 1
    out = capsys.readouterr().out
    assert "Trailer audit failed:" in out
    assert failing_sha[:12] in out


def test_audit_range_fails_closed_without_git(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Audit mode refuses to proceed when git cannot be resolved."""
    monkeypatch.setattr(check_commit_trailers, "_resolve_git_executable", lambda: None)

    assert check_commit_trailers.main(["--audit-range", "HEAD"]) == 2
    assert "git executable unavailable" in capsys.readouterr().err


def test_audit_range_reports_rev_list_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bad revision range returns the audit infrastructure error code."""
    monkeypatch.setattr(check_commit_trailers, "_resolve_git_executable", lambda: sys.executable)

    def fake_run_git(_git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(128, ["git", *args], stderr="bad revision")

    monkeypatch.setattr(check_commit_trailers, "_run_git", fake_run_git)

    assert check_commit_trailers.main(["--audit-range", "bad..range"]) == 2
    assert "git rev-list failed: bad revision" in capsys.readouterr().err


def test_resolve_git_executable_rejects_non_executable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The git resolver rejects non-executable PATH hits."""
    candidate = tmp_path / "git"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    candidate.chmod(0o644)
    monkeypatch.setattr("tools.check_commit_trailers.shutil.which", lambda _name: str(candidate))

    assert check_commit_trailers._resolve_git_executable() is None


def test_resolve_git_executable_rejects_missing_and_unresolvable_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The git resolver fails closed for missing or stale PATH hits."""
    missing = tmp_path / "missing-git"
    monkeypatch.setattr("tools.check_commit_trailers.shutil.which", lambda _name: None)
    assert check_commit_trailers._resolve_git_executable() is None

    monkeypatch.setattr("tools.check_commit_trailers.shutil.which", lambda _name: str(missing))
    assert check_commit_trailers._resolve_git_executable() is None


def test_resolve_git_executable_and_run_git_admit_current_git() -> None:
    """The resolver returns a git binary that can run a harmless command."""
    git_executable = check_commit_trailers._resolve_git_executable()

    assert git_executable is not None
    result = check_commit_trailers._run_git(git_executable, "--version")
    assert result.stdout.startswith("git version ")
