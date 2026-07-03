# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core SPDX header guard tests

"""Regression guard for mandatory GOTM headers on tracked source/config files."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER_MARKERS = (
    "SPDX-License-Identifier: AGPL-3.0-or-later",
    "Commercial license available",
    "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
    "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
    "ORCID: 0009-0009-3560-0851",
    "Contact: www.anulum.li | protoscience@anulum.li",
)
SOURCE_HEADER_SUFFIXES = frozenset({".py", ".rs", ".sh", ".toml", ".yaml", ".yml"})
SOURCE_HEADER_FILENAMES = frozenset({"Makefile"})


def _tracked_source_config_files() -> tuple[Path, ...]:
    """Return tracked source/config files that must carry the GOTM header."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        rel_path = Path(line.strip())
        if not rel_path.name:
            continue
        if rel_path.suffix in SOURCE_HEADER_SUFFIXES or rel_path.name in SOURCE_HEADER_FILENAMES:
            paths.append(rel_path)
    return tuple(paths)


def test_tracked_source_config_files_have_gotm_header() -> None:
    """Keep tracked source and config files under the mandatory GOTM header."""
    missing: list[str] = []
    for rel_path in _tracked_source_config_files():
        text = (ROOT / rel_path).read_text(encoding="utf-8", errors="ignore")
        header_window = "\n".join(text.splitlines()[:9])
        if not all(marker in header_window for marker in HEADER_MARKERS):
            missing.append(rel_path.as_posix())

    assert missing == []
