# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Guard against brittle GitHub App installation token format assumptions."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TRACKED_TEXT_GLOBS = (
    ".github/workflows/",
    "tools/",
    "src/",
)


def _tracked_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    out: list[Path] = []
    for rel in proc.stdout.splitlines():
        if not rel:
            continue
        if not rel.endswith((".py", ".yml", ".yaml", ".sh")):
            continue
        if not rel.startswith(TRACKED_TEXT_GLOBS):
            continue
        out.append(REPO_ROOT / rel)
    return out


SUSPECT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"len\s*\(\s*[A-Za-z_][A-Za-z0-9_]*token[A-Za-z0-9_]*\s*\)\s*([=!<>]=?)\s*40\b"),
        "Token length hardcoded to 40 chars.",
    ),
    (
        re.compile(r"ghs_\[A-Za-z0-9\]\{[0-9,\s]+\}"),
        "Regex only allows legacy ghs_ token charset/length; include dot/underscore and variable length.",
    ),
)


def evaluate() -> list[str]:
    """Scan tracked repo files for brittle GitHub token format assumptions.

    Returns
    -------
    list[str]
        A list of findings in ``file:line: message :: code`` format.
        Empty when the repository avoids token-length/format assumptions.
    """
    findings: list[str] = []
    for file_path in _tracked_files():
        rel = file_path.relative_to(REPO_ROOT).as_posix()
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern, message in SUSPECT_PATTERNS:
                if pattern.search(line):
                    findings.append(f"{rel}:{lineno}: {message} :: {line.strip()}")
    return findings


def main() -> int:
    """Run the token format guard and return a CI-compatible exit code.

    Returns
    -------
    int
        ``0`` when no brittle token checks are found, ``1`` when suspicious
        patterns are detected.
    """
    findings = evaluate()
    if findings:
        print("GitHub token format guard failed:")
        for item in findings:
            print(f" - {item}")
        print(
            "Use format-agnostic handling for GitHub App installation tokens "
            "(ghs_* may be JWT-like and ~520 chars)."
        )
        return 1
    print("GitHub token format guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
