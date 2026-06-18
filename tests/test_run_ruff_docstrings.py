# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Cohort Docstring Runner Tests
"""Behavioural contract tests for ``tools/run_ruff_docstrings.py``."""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.run_ruff_docstrings as runner


def _write_pyproject(tmp_path: Path, files: list[str]) -> Path:
    """Write a minimal pyproject with a ``[tool.mypy].files`` cohort."""
    body = "[tool.mypy]\nstrict = true\nfiles = [\n"
    body += "".join(f'    "{f}",\n' for f in files)
    body += "]\n"
    path = tmp_path / "pyproject.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_cohort_source_files_keeps_only_src_python(tmp_path: Path) -> None:
    """Non-``src`` cohort entries and non-Python entries are excluded and sorted."""
    pyproject = _write_pyproject(
        tmp_path,
        [
            "src/scpn_fusion/core/b.py",
            "src/scpn_fusion/core/a.py",
            "tests/test_hpc_bridge.py",
            "src/scpn_fusion/io/notes.txt",
            "src/scpn_fusion/core/a.py",
        ],
    )

    assert runner.cohort_source_files(pyproject) == [
        "src/scpn_fusion/core/a.py",
        "src/scpn_fusion/core/b.py",
    ]


def test_cohort_source_files_rejects_empty_cohort(tmp_path: Path) -> None:
    """An empty src cohort raises rather than running a no-op gate."""
    pyproject = _write_pyproject(tmp_path, ["tests/test_only.py"])

    with pytest.raises(SystemExit):
        runner.cohort_source_files(pyproject)


def test_cohort_source_files_requires_files_table(tmp_path: Path) -> None:
    """A pyproject without ``[tool.mypy].files`` raises a clear error."""
    path = tmp_path / "pyproject.toml"
    path.write_text("[tool.mypy]\nstrict = true\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        runner.cohort_source_files(path)


def test_build_command_selects_only_numpy_docstring_rules() -> None:
    """The ruff command restricts checks to D rules under the NumPy convention."""
    cmd = runner.build_command(["src/scpn_fusion/core/a.py"], ["--statistics"])

    assert cmd[1:4] == ["-m", "ruff", "check"]
    assert "--select" in cmd
    assert cmd[cmd.index("--select") + 1] == "D"
    assert 'lint.pydocstyle.convention="numpy"' in cmd
    assert cmd[-2:] == ["--statistics", "src/scpn_fusion/core/a.py"]


def test_repository_cohort_is_non_empty() -> None:
    """The live repository cohort resolves to a non-empty src file list."""
    cohort = runner.cohort_source_files()

    assert cohort
    assert all(f.startswith("src/") and f.endswith(".py") for f in cohort)
