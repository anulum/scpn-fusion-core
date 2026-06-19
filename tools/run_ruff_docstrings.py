#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Enforce NumPy-convention docstring rules on the strict type cohort.

The strict mypy cohort in ``pyproject.toml`` (``[tool.mypy].files``) is the
single source of truth for which modules have been brought to production typing
and documentation grade. This runner applies ruff's pydocstyle (``D``) rules
under the NumPy convention to exactly those ``src/`` modules, so docstring
enforcement grows automatically as files graduate into the cohort without a
second hand-maintained allowlist.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
DOCSTRING_CONVENTION = "numpy"


def cohort_source_files(pyproject: Path = PYPROJECT) -> list[str]:
    """Return the ``src/`` Python files in the strict mypy cohort.

    Parameters
    ----------
    pyproject : Path
        Location of the ``pyproject.toml`` holding ``[tool.mypy].files``.

    Returns
    -------
    list of str
        Repo-relative POSIX paths of cohort files under ``src/`` ending in
        ``.py``, sorted and de-duplicated. Test, tool, and other non-``src``
        cohort entries are excluded because docstring enforcement here targets
        the library surface only.

    Raises
    ------
    SystemExit
        If the cohort list is missing or empty, which would silently disable
        the gate.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    try:
        files = data["tool"]["mypy"]["files"]
    except (KeyError, TypeError) as exc:  # pragma: no cover - config contract
        raise SystemExit(f"[run_ruff_docstrings] missing [tool.mypy].files in {pyproject}") from exc
    cohort = sorted({f for f in files if f.startswith("src/") and f.endswith(".py")})
    if not cohort:
        raise SystemExit("[run_ruff_docstrings] empty src cohort; refusing to run a no-op gate.")
    return cohort


def build_command(files: Sequence[str], extra_args: Sequence[str]) -> list[str]:
    """Assemble the ruff invocation for the docstring gate.

    Parameters
    ----------
    files : sequence of str
        Cohort source paths to check.
    extra_args : sequence of str
        Additional ruff arguments passed through verbatim (for example
        ``--fix`` or ``--statistics``).

    Returns
    -------
    list of str
        The full ``ruff check`` command, selecting only ``D`` rules under the
        NumPy convention so the global lint ``select`` is not widened.
    """
    return [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--no-cache",
        "--select",
        "D",
        "--config",
        f'lint.pydocstyle.convention="{DOCSTRING_CONVENTION}"',
        *extra_args,
        *files,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cohort docstring gate and propagate ruff's exit code.

    Parameters
    ----------
    argv : sequence of str or None
        Extra ruff arguments; defaults to ``sys.argv`` tail when ``None``.

    Returns
    -------
    int
        ``0`` when every cohort module passes the NumPy docstring rules,
        otherwise ruff's non-zero exit code.
    """
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--list-cohort", action="store_true")
    args, extra_args = parser.parse_known_args(sys.argv[1:] if argv is None else list(argv))

    files = cohort_source_files()
    if args.list_cohort:
        print("\n".join(files))
        return 0

    cmd = build_command(files, extra_args)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=os.environ.copy(), check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
