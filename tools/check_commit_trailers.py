#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Commit trailer verifier
"""Validate commit-message trailers for local agent commits.

The primary entrypoint is the pre-commit ``commit-msg`` hook. It receives the
pending commit-message file path, rejects messages without the required
``Seat:`` trailer and authorship line, and rejects commit subjects containing
outward-facing self-praise terms.

The optional ``--audit-range`` mode checks an explicit Git revision range. It is
intended for targeted operator audits, not as a historical rewrite demand.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path

REQUIRED_AUTHORSHIP_LINE = "Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)"
SEAT_TRAILER_PREFIX_RE = re.compile(r"^\s*Seat:")
SEAT_TRAILER_RE = re.compile(r"^Seat:\s+([A-Za-z0-9][A-Za-z0-9_-]{0,63})\s*$")
FORBIDDEN_SEAT_PREFIXES = (
    "claude-",
    "codex-",
    "gemini-",
    "grok-",
    "kimi-",
    "openai-",
)
PUBLIC_SUBJECT_BANNED_TERMS = (
    "world-class",
    "best-in-class",
    "state-of-the-art",
    "cutting-edge",
    "revolutionary",
    "groundbreaking",
    "unrivalled",
    "elite",
    "comprehensive",
    "robust",
    "leveraging",
)
PUBLIC_SUBJECT_BANNED_RE = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in PUBLIC_SUBJECT_BANNED_TERMS) + r")\b",
    re.IGNORECASE,
)


def _subject_line(message: str) -> str:
    """Return the first non-empty commit-message line."""
    return next((line.strip() for line in message.splitlines() if line.strip()), "")


def _authorship_line_indices(lines: list[str]) -> list[int]:
    """Return indexes containing the exact required authorship line."""
    return [index for index, line in enumerate(lines) if line.strip() == REQUIRED_AUTHORSHIP_LINE]


def _seat_trailer_indices(lines: list[str]) -> list[int]:
    """Return indexes of lines that look like ``Seat:`` trailers."""
    return [index for index, line in enumerate(lines) if SEAT_TRAILER_PREFIX_RE.match(line)]


def _seat_trailer_violations(lines: list[str], authorship_indices: list[int]) -> list[str]:
    """Return validation errors for the forward-only ``Seat:`` trailer."""
    seat_indices = _seat_trailer_indices(lines)
    if not seat_indices:
        return ["missing `Seat: <seat-id>` trailer"]
    if len(seat_indices) != 1:
        return ["expected exactly one `Seat: <seat-id>` trailer"]

    violations: list[str] = []
    seat_index = seat_indices[0]
    seat_line = lines[seat_index].strip()
    seat_match = SEAT_TRAILER_RE.match(seat_line)
    if seat_match is None:
        return ["invalid `Seat: <seat-id>` trailer"]

    seat_id = seat_match.group(1).lower()
    if any(seat_id.startswith(prefix) for prefix in FORBIDDEN_SEAT_PREFIXES):
        violations.append("vendor-prefixed `Seat:` trailer is forbidden")

    if len(authorship_indices) == 1:
        authorship_index = authorship_indices[0]
        between = lines[seat_index + 1 : authorship_index]
        if seat_index >= authorship_index or any(line.strip() for line in between):
            violations.append("`Seat:` trailer must immediately precede the authorship line")
    return violations


def message_violations(message: str) -> list[str]:
    """Return commit-message policy violations.

    Parameters
    ----------
    message:
        Full commit message text.

    Returns
    -------
    list[str]
        Empty when the message satisfies the local agent commit policy.
    """
    lines = message.splitlines()
    violations: list[str] = []
    authorship_indices = _authorship_line_indices(lines)
    if not authorship_indices:
        violations.append(f"missing `{REQUIRED_AUTHORSHIP_LINE}` authorship line")
    elif len(authorship_indices) != 1:
        violations.append("expected exactly one authorship line")
    violations.extend(_seat_trailer_violations(lines, authorship_indices))

    banned_terms = sorted(
        {
            match.group(1).lower()
            for match in PUBLIC_SUBJECT_BANNED_RE.finditer(_subject_line(message))
        }
    )
    if banned_terms:
        violations.append(f"banned public subject term(s): {', '.join(banned_terms)}")
    return violations


def _commit_msg_hook(path: Path) -> int:
    """Validate a commit-message file and return a hook-compatible exit code."""
    violations = message_violations(path.read_text(encoding="utf-8"))
    if not violations:
        return 0

    print("Commit message rejected:", file=sys.stderr)
    for violation in violations:
        print(f"  - {violation}", file=sys.stderr)
    print(file=sys.stderr)
    print("Required trailer block:", file=sys.stderr)
    print("  Seat: <seat-id>", file=sys.stderr)
    print(file=sys.stderr)
    print(f"  {REQUIRED_AUTHORSHIP_LINE}", file=sys.stderr)
    return 1


def _resolve_git_executable() -> str | None:
    """Return an executable absolute ``git`` path when it is available."""
    located = shutil.which("git")
    if located is None:
        return None
    try:
        resolved = Path(located).resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        return None
    return str(resolved)


def _run_git(git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run an admitted ``git`` command without shell expansion."""
    return subprocess.run(  # nosec B603
        [git_executable, *args],
        capture_output=True,
        check=True,
        shell=False,
        text=True,
    )


def _audit_range(range_spec: str) -> int:
    """Validate commit messages in an explicit Git revision range."""
    git_executable = _resolve_git_executable()
    if git_executable is None:
        print("git executable unavailable", file=sys.stderr)
        return 2
    try:
        rev_list = _run_git(git_executable, "rev-list", range_spec)
    except subprocess.CalledProcessError as exc:
        print(f"git rev-list failed: {exc.stderr}", file=sys.stderr)
        return 2

    shas = [line.strip() for line in rev_list.stdout.splitlines() if line.strip()]
    failures: list[str] = []
    for sha in shas:
        message = _run_git(git_executable, "log", "-1", "--format=%B", sha).stdout
        violations = message_violations(message)
        if violations:
            failures.append(f"{sha[:12]}: {'; '.join(violations)}")

    print(f"Audited {len(shas)} commit(s) in {range_spec}")
    if not failures:
        print("Trailer audit passed.")
        return 0
    print("Trailer audit failed:")
    for failure in failures:
        print(f"  - {failure}")
    return 1


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for hook and audit modes."""
    parser = argparse.ArgumentParser(description="Validate SCPN Fusion Core commit trailers.")
    parser.add_argument(
        "message_file",
        nargs="?",
        type=Path,
        help="commit message file supplied by the commit-msg hook",
    )
    parser.add_argument(
        "--audit-range",
        metavar="REVISION_RANGE",
        help="audit an explicit Git revision range instead of one hook message",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the commit-trailer checker.

    Parameters
    ----------
    argv:
        Optional argument vector without the executable name.

    Returns
    -------
    int
        ``0`` when validation passes, ``1`` for policy violations, and ``2`` for
        invocation or Git infrastructure errors.
    """
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.audit_range is not None:
        return _audit_range(args.audit_range)
    if args.message_file is None:
        _parser().print_usage(sys.stderr)
        return 2
    return _commit_msg_hook(args.message_file)


if __name__ == "__main__":
    raise SystemExit(main())
