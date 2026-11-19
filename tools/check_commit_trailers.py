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
outward-facing self-praise terms. It also rejects the superseded
``Co-Authored-By:`` trailer in forward-only commits.

A pushed commit message is public, so the whole message is also checked for
agent seat ids and vendor or model names. Internal planning codes are matched
against private patterns read from a file outside the repository (see
``private_patterns_path``). That file is never committed, because a list of the
codes would itself publish them; where it is absent, as in a public clone or in
CI, the planning-code check is skipped.

The optional ``--audit-range`` mode checks an explicit Git revision range. It is
intended for targeted operator audits, not as a historical rewrite demand. The
``--pre-push`` mode reads the ``pre-push`` hook's standard input and checks every
commit that the push would publish.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

REQUIRED_AUTHORSHIP_LINE = "Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)"
LEGACY_COAUTHOR_PREFIX_RE = re.compile(r"^\s*Co-Authored-By:", re.IGNORECASE)
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
SEAT_ID_IN_TEXT_RE = re.compile(
    r"\b(?:claude|codex|gemini|grok|kimi|openai|qwen)-[0-9a-f]{4,}\b", re.IGNORECASE
)
# Letters only on either side: a name inside a path such as ``<vendor>09bd_run`` still counts.
VENDOR_NAME_RE = re.compile(
    r"(?<![a-z])(?:anthropic|openai|chatgpt|claude|codex|gemini|grok|kimi|qwen)(?![a-z])",
    re.IGNORECASE,
)
PRIVATE_PATTERNS_ENV = "SCPN_PRIVATE_COMMIT_PATTERNS"
PRIVATE_PATTERNS_RELATIVE = (
    Path(".coordination")
    / "planning"
    / "SCPN-REACTOR-SYSTEMS"
    / "commit_message_private_patterns.txt"
)
ZERO_SHA = "0" * 40
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


def private_patterns_path() -> Path | None:
    """Return the private planning-code pattern file, or ``None`` where there is none.

    ``SCPN_PRIVATE_COMMIT_PATTERNS`` names the file explicitly. Otherwise the
    coordination tree is searched upwards from this checkout, where it exists only
    on the maintainers' hosts.
    """
    configured = os.environ.get(PRIVATE_PATTERNS_ENV)
    if configured:
        return Path(configured)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / PRIVATE_PATTERNS_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def load_private_patterns(path: Path | None) -> list[re.Pattern[str]]:
    """Compile one regular expression per non-blank, non-comment line of ``path``.

    Raises
    ------
    ValueError
        If a configured file is missing or a line is not a valid expression, so a
        broken private configuration never silently disables the check.
    """
    if path is None:
        return []
    if not path.is_file():
        raise ValueError(f"private commit pattern file not found: {path}")
    patterns: list[re.Pattern[str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            patterns.append(re.compile(line))
        except re.error as exc:
            raise ValueError(f"invalid private commit pattern on line {number}: {exc}") from exc
    return patterns


def _public_surface_violations(
    lines: list[str], private_patterns: Sequence[re.Pattern[str]]
) -> list[str]:
    """Return seat ids, vendor names and private planning codes found in the message."""
    text = "\n".join(line for line in lines if line.strip() != REQUIRED_AUTHORSHIP_LINE)
    violations: list[str] = []
    seat_ids = sorted({match.group(0) for match in SEAT_ID_IN_TEXT_RE.finditer(text)})
    if seat_ids:
        violations.append(f"agent seat id in the message: {', '.join(seat_ids)}")
    vendors = sorted({match.group(0).lower() for match in VENDOR_NAME_RE.finditer(text)})
    if vendors:
        violations.append(f"vendor or model name in the message: {', '.join(vendors)}")
    codes = sorted(
        {match.group(0) for pattern in private_patterns for match in pattern.finditer(text)}
    )
    if codes:
        violations.append(f"internal planning code in the message: {', '.join(codes)}")
    return violations


def message_violations(
    message: str, private_patterns: Sequence[re.Pattern[str]] | None = None
) -> list[str]:
    """Return commit-message policy violations.

    Parameters
    ----------
    message:
        Full commit message text.
    private_patterns:
        Compiled planning-code patterns; ``None`` loads them from
        :func:`private_patterns_path`.

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
    if any(LEGACY_COAUTHOR_PREFIX_RE.match(line) for line in lines):
        violations.append("legacy `Co-Authored-By:` trailer is forbidden")

    banned_terms = sorted(
        {
            match.group(1).lower()
            for match in PUBLIC_SUBJECT_BANNED_RE.finditer(_subject_line(message))
        }
    )
    if banned_terms:
        violations.append(f"banned public subject term(s): {', '.join(banned_terms)}")
    if private_patterns is None:
        private_patterns = load_private_patterns(private_patterns_path())
    violations.extend(_public_surface_violations(lines, private_patterns))
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


def _audit_revisions(rev_args: list[str], label: str) -> int:
    """Validate the messages of the commits that ``git rev-list <rev_args>`` lists."""
    git_executable = _resolve_git_executable()
    if git_executable is None:
        print("git executable unavailable", file=sys.stderr)
        return 2
    try:
        rev_list = _run_git(git_executable, "rev-list", *rev_args)
    except subprocess.CalledProcessError as exc:
        print(f"git rev-list failed: {exc.stderr}", file=sys.stderr)
        return 2

    private_patterns = load_private_patterns(private_patterns_path())
    shas = [line.strip() for line in rev_list.stdout.splitlines() if line.strip()]
    failures: list[str] = []
    for sha in shas:
        message = _run_git(git_executable, "log", "-1", "--format=%B", sha).stdout
        violations = message_violations(message, private_patterns)
        if violations:
            failures.append(f"{sha[:12]}: {'; '.join(violations)}")

    print(f"Audited {len(shas)} commit(s) in {label}")
    if not failures:
        print("Trailer audit passed.")
        return 0
    print("Trailer audit failed:")
    for failure in failures:
        print(f"  - {failure}")
    return 1


def _audit_range(range_spec: str) -> int:
    """Validate commit messages in an explicit Git revision range."""
    return _audit_revisions([range_spec], range_spec)


def _pre_push(stream: TextIO) -> int:
    """Check every commit a push would publish, from the ``pre-push`` hook's input.

    Each input line is ``<local ref> <local sha> <remote ref> <remote sha>``. A
    deletion publishes nothing. An update checks ``<remote sha>..<local sha>``; a
    new remote ref checks the commits that no remote has yet.
    """
    worst = 0
    for raw in stream:
        fields = raw.split()
        if len(fields) != 4:
            continue
        _local_ref, local_sha, _remote_ref, remote_sha = fields
        if local_sha == ZERO_SHA:
            continue
        if remote_sha == ZERO_SHA:
            result = _audit_revisions(
                [local_sha, "--not", "--remotes"], f"{local_sha[:12]} (new ref)"
            )
        else:
            result = _audit_range(f"{remote_sha}..{local_sha}")
        worst = max(worst, result)
    return worst


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
    parser.add_argument(
        "--pre-push",
        action="store_true",
        help="read the pre-push hook's standard input and audit every commit being pushed",
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
        invocation, Git infrastructure or private-pattern configuration errors.
    """
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.pre_push:
            return _pre_push(sys.stdin)
        if args.audit_range is not None:
            return _audit_range(args.audit_range)
        if args.message_file is None:
            _parser().print_usage(sys.stderr)
            return 2
        return _commit_msg_hook(args.message_file)
    except ValueError as exc:
        print(f"commit-message check misconfigured: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
