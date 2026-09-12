# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — all-definition native documentation gate
"""Require docstrings on every module, class and function in explicit Python files."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
from pathlib import Path


def missing_docstrings(paths: Sequence[Path]) -> list[str]:
    """Find undocumented definitions, including private and nested objects.

    Parameters
    ----------
    paths : sequence of pathlib.Path
        Explicit maintained Python files; test files use the same policy.

    Returns
    -------
    list of str
        File, line and definition name for each missing or blank docstring.

    Raises
    ------
    ValueError
        The supplied file scope is empty.
    OSError
        A requested file cannot be read.
    SyntaxError
        A requested file is not valid Python.
    """
    if not paths:
        raise ValueError("The documentation scope must contain at least one file")
    findings = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not (ast.get_docstring(node) or "").strip():
                    line = getattr(node, "lineno", 1)
                    name = getattr(node, "name", "<module>")
                    findings.append(f"{path}:{line}: missing docstring: {name}")
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    """Check explicit files and return a failing status for missing documentation.

    Parameters
    ----------
    argv : sequence of str or None
        File arguments, or process arguments when omitted.

    Returns
    -------
    int
        Zero when all definitions are documented; one when findings exist.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    findings = missing_docstrings(args.files)
    for finding in findings:
        print(finding)
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
