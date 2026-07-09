# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Subsystem Logging Contract Tests
"""Regression guards for subsystem logging contracts."""

from __future__ import annotations

import ast
from pathlib import Path

TARGET_DIRS = (
    Path("src/scpn_fusion/io"),
    Path("src/scpn_fusion/diagnostics"),
    Path("src/scpn_fusion/engineering"),
    Path("src/scpn_fusion/nuclear"),
    Path("src/scpn_fusion/hpc"),
)


def _print_call_lines(path: Path) -> list[int]:
    """Return line numbers of direct builtin print calls in a Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            lines.append(int(node.lineno))
    return lines


def test_subsystem_sources_do_not_call_print() -> None:
    """Subsystem source directories use logging instead of direct stdout prints."""
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for target_dir in TARGET_DIRS:
        for path in sorted((repo_root / target_dir).glob("*.py")):
            for line in _print_call_lines(path):
                offenders.append(f"{path.relative_to(repo_root)}:{line}")

    assert offenders == []
