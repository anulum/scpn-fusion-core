# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Behavioural contract tests for ``tools/check_docstring_coverage.py``."""

from __future__ import annotations

import ast
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_collect_docstring_issues_skips_overload_stubs(tmp_path: Path) -> None:
    """``typing.overload`` signature stubs are excluded, matching ruff D418."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "overloaded.py").write_text(
        '"""Module docs."""\n\n'
        "from typing import overload\n\n"
        "@overload\n"
        "def widen(value: int) -> int: ...\n\n"
        "@overload\n"
        "def widen(value: str) -> str: ...\n\n"
        'def widen(value):\n    """Widen a value while preserving its type."""\n    return value\n',
        encoding="utf-8",
    )

    assert guard.collect_docstring_issues(tmp_path, ("src",)) == []


def test_collect_docstring_issues_skips_attribute_overload_stubs(tmp_path: Path) -> None:
    """Attribute-style ``typing.overload`` decorators are also excluded."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "attr_overloaded.py").write_text(
        '"""Module docs."""\n\n'
        "import typing\n\n"
        "@typing.overload\n"
        "def widen(value: int) -> int: ...\n\n"
        'def widen(value):\n    """Widen a value while preserving its type."""\n    return value\n',
        encoding="utf-8",
    )

    assert guard.collect_docstring_issues(tmp_path, ("src",)) == []


def test_main_fails_when_issue_count_exceeds_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI accepts repositories that do not exceed the stored baseline."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "sample.py").write_text('\n"""Module docs."""\n', encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"total_issues": 0}), encoding="utf-8")
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    assert guard.main(["--baseline", str(baseline), "--roots", "src"]) == 0


def test_git_file_discovery_handles_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse non-empty Git output and fall back when the command is unavailable."""

    def completed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="src/a.py\n\n", stderr="")

    monkeypatch.setattr(subprocess, "run", completed)
    assert guard._run_git_ls_files(tmp_path) == [tmp_path / "src/a.py"]

    def unavailable(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("git unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable)
    assert guard._run_git_ls_files(tmp_path) is None


def test_iter_python_files_filters_scope_and_uses_rglob_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclude out-of-root, test, and unselected paths in both discovery modes."""
    source = tmp_path / "src" / "a.py"
    excluded = tmp_path / "tests" / "b.py"
    unrelated = tmp_path / "other" / "c.py"
    outside = tmp_path.parent / "outside.py"
    for path in (source, excluded, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Documented."""\n', encoding="utf-8")

    monkeypatch.setattr(
        guard,
        "_run_git_ls_files",
        lambda root: [source, excluded, unrelated, outside],
    )
    assert guard.iter_python_files(tmp_path, ("src",)) == [source]

    monkeypatch.setattr(guard, "_run_git_ls_files", lambda root: None)
    assert guard.iter_python_files(tmp_path, ("src",)) == [source]


def test_non_definition_ast_node_has_no_docstring() -> None:
    """Treat AST nodes that cannot own docstrings as undocumented."""
    assert guard._has_docstring(ast.Constant(value="text")) is False

    decorated = ast.parse("@decorator\ndef public_function(): ...\n").body[0]
    assert isinstance(decorated, ast.FunctionDef)
    assert guard._is_overload(decorated) is False


@pytest.mark.parametrize(
    "payload",
    [[], {}, {"total_issues": "0"}, {"total_issues": True}],
)
def test_invalid_baselines_fail_closed(tmp_path: Path, payload: object) -> None:
    """Reject non-object, missing, non-integer, and boolean baseline counts."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="Invalid docstring baseline"):
        guard._load_baseline(baseline)


def test_missing_baseline_fails_closed(tmp_path: Path) -> None:
    """Report a missing baseline separately from malformed baseline content."""
    with pytest.raises(SystemExit, match="Missing docstring baseline"):
        guard._load_baseline(tmp_path / "missing.json")


def test_main_writes_deterministic_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write and immediately consume the deterministic baseline contract."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "sample.py").write_text('"""Module docs."""\n', encoding="utf-8")
    baseline = tmp_path / "nested" / "baseline.json"
    baseline.parent.mkdir()
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    rc = guard.main(
        [
            "--baseline",
            str(baseline),
            "--roots",
            "src",
            "--write-baseline",
            "--json",
        ]
    )

    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload == {
        "by_kind": {},
        "files_with_issues": 0,
        "schema_version": "docstring-coverage-baseline.v1",
        "scope": ["src", "validation", "tools"],
        "total_issues": 0,
    }


def test_main_json_regression_and_negative_list_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emit JSON regressions and bound negative human-readable finding limits."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "sample.py").write_text("def public_function():\n    return 1\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"total_issues": 0}), encoding="utf-8")
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    assert guard.main(["--baseline", str(baseline), "--roots", "src", "--json"]) == 1
    json_output = json.loads(capsys.readouterr().out)
    assert json_output["passes"] is False

    baseline.write_text(json.dumps({"total_issues": 2}), encoding="utf-8")
    assert guard.main(["--baseline", str(baseline), "--roots", "src", "--max-list", "-1"]) == 0
    human_output = capsys.readouterr().out
    assert "First findings:" in human_output
    assert "src/pkg/sample.py:" not in human_output


def test_direct_script_entry_point_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the real script entry point on its documented own source."""
    module_path = Path(guard.__file__)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module_path),
            "--baseline",
            str(guard.DEFAULT_BASELINE),
            "--roots",
            "tools/check_docstring_coverage.py",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(module_path), run_name="__main__")

    assert exit_info.value.code == 0
    assert json.loads(capsys.readouterr().out)["passes"] is True
