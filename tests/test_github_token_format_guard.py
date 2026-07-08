# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/github_token_format_guard.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "github_token_format_guard.py"
SPEC = importlib.util.spec_from_file_location("tools.github_token_format_guard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def test_flags_len_40_token_assumption() -> None:
    line = "if len(github_token) == 40:"
    for pattern, _ in guard.SUSPECT_PATTERNS:
        if pattern.search(line):
            break
    else:  # pragma: no cover
        raise AssertionError("Expected token-length hardcoding pattern to match.")


def test_flags_legacy_ghs_regex_assumption() -> None:
    line = r'pattern = r"ghs_[A-Za-z0-9]{40}"'
    for pattern, _ in guard.SUSPECT_PATTERNS:
        if pattern.search(line):
            break
    else:  # pragma: no cover
        raise AssertionError("Expected legacy ghs_ regex pattern to match.")


def test_does_not_flag_opaque_token_pass_through() -> None:
    line = 'headers = {"Authorization": f"Bearer {token}"}'
    assert all(not pattern.search(line) for pattern, _ in guard.SUSPECT_PATTERNS)


def test_tracked_files_filters_repo_text_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracked-file discovery keeps only supported text files under guarded roots."""
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            stdout="\n".join(
                [
                    ".github/workflows/ci.yml",
                    "",
                    "docs/not-guarded.py",
                    "src/module.py",
                    "tools/script.sh",
                    "tools/readme.txt",
                ]
            )
        )

    monkeypatch.setattr(guard.subprocess, "run", fake_run)

    assert [path.relative_to(tmp_path).as_posix() for path in guard._tracked_files()] == [
        ".github/workflows/ci.yml",
        "src/module.py",
        "tools/script.sh",
    ]


def test_evaluate_reports_findings_and_skips_non_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation reports suspect token checks and ignores non-UTF-8 files."""
    src_file = tmp_path / "src" / "bad.py"
    binary_file = tmp_path / "tools" / "binary.py"
    src_file.parent.mkdir()
    binary_file.parent.mkdir()
    src_file.write_text("if len(github_token) == 40:\n    pass\n", encoding="utf-8")
    binary_file.write_bytes(b"\xff")

    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(guard, "_tracked_files", lambda: [src_file, binary_file])

    findings = guard.evaluate()
    assert len(findings) == 1
    assert "src/bad.py:1" in findings[0]


def test_main_reports_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Main returns CI-compatible status codes for clean and failing scans."""
    monkeypatch.setattr(guard, "evaluate", lambda: [])
    assert guard.main() == 0
    assert "passed" in capsys.readouterr().out

    monkeypatch.setattr(guard, "evaluate", lambda: ["tools/x.py:1: bad :: code"])
    assert guard.main() == 1
    assert "failed" in capsys.readouterr().out
