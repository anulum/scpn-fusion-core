# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Docstring Guard Tool Tests
"""Behavioural contract tests for ``tools/check_docstring_coverage.py``."""

from __future__ import annotations

import json
from pathlib import Path

import tools.check_docstring_coverage as guard


def test_collect_docstring_issues_reports_public_api_gaps(tmp_path: Path) -> None:
    """Missing module, class, function, and method docs are separate findings."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    module = src / "sample.py"
    module.write_text(
        "\n".join(
            [
                "class PublicClass:",
                "    def public_method(self):",
                "        return 1",
                "    def _private_method(self):",
                "        return 2",
                "def public_function():",
                "    return 3",
                "def _private_function():",
                "    return 4",
            ]
        ),
        encoding="utf-8",
    )

    issues = guard.collect_docstring_issues(tmp_path, ("src",))

    assert [issue.kind for issue in issues] == ["module", "class", "method", "function"]
    assert [issue.qualname for issue in issues] == [
        "src.pkg.sample",
        "PublicClass",
        "PublicClass.public_method",
        "public_function",
    ]


def test_collect_docstring_issues_accepts_documented_public_api(tmp_path: Path) -> None:
    """Documented public module, class, method, and function produce no findings."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "documented.py").write_text(
        '\n"""Module docs."""\n\n'
        'class PublicClass:\n    """Class docs."""\n\n'
        '    def public_method(self):\n        """Method docs."""\n        return 1\n\n'
        'def public_function():\n    """Function docs."""\n    return 2\n',
        encoding="utf-8",
    )

    assert guard.collect_docstring_issues(tmp_path, ("src",)) == []


def test_main_fails_when_issue_count_exceeds_baseline(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """The CLI exits non-zero when current findings exceed the baseline."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "sample.py").write_text("def public_function():\n    return 1\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"total_issues": 0}), encoding="utf-8")
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    rc = guard.main(["--baseline", str(baseline), "--roots", "src"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "exceeds baseline" in captured.err


def test_main_passes_when_issue_count_matches_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI accepts repositories that do not exceed the stored baseline."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "sample.py").write_text('\n"""Module docs."""\n', encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"total_issues": 0}), encoding="utf-8")
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    assert guard.main(["--baseline", str(baseline), "--roots", "src", "--json"]) == 0
