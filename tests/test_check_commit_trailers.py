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
LEGACY_COAUTHOR_LINE = "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"


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


@pytest.mark.parametrize(
    "legacy_line",
    [LEGACY_COAUTHOR_LINE, "  co-authored-by: Other Author <other@example.test>"],
)
def test_commit_msg_hook_rejects_legacy_coauthor_trailer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    legacy_line: str,
) -> None:
    """The superseded co-author trailer is forbidden for new commits."""
    path = _message_file(
        tmp_path,
        _message(
            "test: add trailer guard",
            "",
            VALID_SEAT_TRAILER,
            "",
            REQUIRED_AUTHORSHIP_LINE,
            legacy_line,
        ),
    )

    assert check_commit_trailers.main([str(path)]) == 1
    assert "legacy `Co-Authored-By:` trailer is forbidden" in capsys.readouterr().err


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
        """Answer the git calls the checker makes with a fixed fake history."""
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
        """Answer the git calls the checker makes with a fixed fake history."""
        if args == ("rev-list", "HEAD"):
            return subprocess.CompletedProcess(args, 0, stdout=f"{failing_sha}\n")
        if args == ("log", "-1", "--format=%B", failing_sha):
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=_message(
                    "test: add trailer guard",
                    "",
                    VALID_SEAT_TRAILER,
                    "",
                    REQUIRED_AUTHORSHIP_LINE,
                    LEGACY_COAUTHOR_LINE,
                ),
            )
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(check_commit_trailers, "_run_git", fake_run_git)

    assert check_commit_trailers.main(["--audit-range", "HEAD"]) == 1
    out = capsys.readouterr().out
    assert "Trailer audit failed:" in out
    assert failing_sha[:12] in out
    assert "legacy `Co-Authored-By:` trailer is forbidden" in out


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
        """Answer the git calls the checker makes with a fixed fake history."""
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


def _valid_body(*body: str) -> str:
    """Build an otherwise valid message whose body holds ``body``."""
    return _message(
        "fix: describe the change", "", *body, "", VALID_SEAT_TRAILER, REQUIRED_AUTHORSHIP_LINE
    )


def test_body_seat_id_and_vendor_names_are_rejected() -> None:
    """A seat id or a vendor name anywhere in the message is a public-surface violation."""
    violations = check_commit_trailers.message_violations(
        _valid_body("Reviewed by codex-01a1beef; logs in runs/claude09bd_case/."),
        private_patterns=[],
    )

    assert "agent seat id in the message: codex-01a1beef" in violations
    assert "vendor or model name in the message: claude, codex" in violations


def test_words_merely_containing_a_vendor_name_pass() -> None:
    """Letters around a name make another word, which is not a vendor reference."""
    assert (
        check_commit_trailers.message_violations(
            _valid_body("The codexes and Groking tables are unaffected."), private_patterns=[]
        )
        == []
    )


def test_private_patterns_reject_planning_codes(tmp_path: Path) -> None:
    """Codes matched by the private pattern file are refused; other text passes."""
    patterns_file = tmp_path / "patterns.txt"
    patterns_file.write_text("# private\n\n\\bTASK-[0-9]+\\b\n", encoding="utf-8")
    patterns = check_commit_trailers.load_private_patterns(patterns_file)

    leaked = check_commit_trailers.message_violations(_valid_body("Closes TASK-7."), patterns)
    clean = check_commit_trailers.message_violations(_valid_body("Closes the task."), patterns)

    assert leaked == ["internal planning code in the message: TASK-7"]
    assert clean == []


def test_private_pattern_misconfiguration_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A configured but missing file, or an invalid line, stops the check with exit 2."""
    message_file = _message_file(tmp_path, _valid_body("A clean body."))
    monkeypatch.setenv(check_commit_trailers.PRIVATE_PATTERNS_ENV, str(tmp_path / "missing.txt"))
    assert check_commit_trailers.main([str(message_file)]) == 2
    assert "private commit pattern file not found" in capsys.readouterr().err

    broken = tmp_path / "broken.txt"
    broken.write_text("valid\n(unclosed\n", encoding="utf-8")
    monkeypatch.setenv(check_commit_trailers.PRIVATE_PATTERNS_ENV, str(broken))
    assert check_commit_trailers.main([str(message_file)]) == 2
    assert "invalid private commit pattern on line 2" in capsys.readouterr().err


def test_private_patterns_path_searches_the_coordination_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without the variable, the file is found above the checkout, or there is none."""
    monkeypatch.delenv(check_commit_trailers.PRIVATE_PATTERNS_ENV, raising=False)
    outside = Path("/nonexistent-checkout-root/repo/tools/check_commit_trailers.py")
    monkeypatch.setattr(check_commit_trailers, "__file__", str(outside))
    assert check_commit_trailers.private_patterns_path() is None

    # The nearest coordination tree wins, even when another one exists further up.
    module_file = tmp_path / "repositories" / "repo" / "tools" / "check_commit_trailers.py"
    monkeypatch.setattr(check_commit_trailers, "__file__", str(module_file))
    expected = tmp_path / check_commit_trailers.PRIVATE_PATTERNS_RELATIVE
    expected.parent.mkdir(parents=True)
    expected.write_text("\\bTASK-[0-9]+\\b\n", encoding="utf-8")
    assert check_commit_trailers.private_patterns_path() == expected
    assert check_commit_trailers.load_private_patterns(None) == []


def test_pre_push_audits_each_published_range(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Updates check remote..local, new refs check unpublished commits, deletions nothing."""
    import io

    monkeypatch.setattr(check_commit_trailers, "_resolve_git_executable", lambda: sys.executable)
    monkeypatch.setattr(check_commit_trailers, "private_patterns_path", lambda: None)
    zero = check_commit_trailers.ZERO_SHA
    local, remote, fresh = "a" * 40, "b" * 40, "c" * 40
    rev_lists: list[tuple[str, ...]] = []

    def fake_run_git(_git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
        """Answer the git calls the checker makes with a fixed fake history."""
        if args[0] == "rev-list":
            rev_lists.append(args[1:])
            sha = "1111111111111" if args[1] == f"{remote}..{local}" else "2222222222222"
            return subprocess.CompletedProcess(args, 0, stdout=f"{sha}\n")
        if args[:3] == ("log", "-1", "--format=%B"):
            body = "Leaks codex-0123abcd." if args[3] == "2222222222222" else "Clean."
            return subprocess.CompletedProcess(args, 0, stdout=_valid_body(body))
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(check_commit_trailers, "_run_git", fake_run_git)
    stdin = io.StringIO(
        f"refs/heads/main {local} refs/heads/main {remote}\n"
        f"refs/heads/new {fresh} refs/heads/new {zero}\n"
        f"refs/heads/gone {zero} refs/heads/gone {remote}\n"
        "malformed line\n"
    )
    monkeypatch.setattr(sys, "stdin", stdin)

    assert check_commit_trailers.main(["--pre-push"]) == 1
    assert rev_lists == [(f"{remote}..{local}",), (fresh, "--not", "--remotes")]
    out = capsys.readouterr().out
    assert "agent seat id in the message: codex-0123abcd" in out
    assert "Trailer audit passed." in out
