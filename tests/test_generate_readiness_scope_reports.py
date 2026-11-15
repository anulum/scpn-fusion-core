# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Internal readiness Scope Reports Tests
"""Production-contract tests for the split readiness-scope report generator."""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_readiness_scope_reports.py"
SPEC = importlib.util.spec_from_file_location("generate_readiness_scope_reports", MODULE_PATH)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _stub_scope_reports(*, top_limit: int, full_limit: int) -> tuple[str, str, str]:
    assert top_limit > 0
    assert full_limit > 0
    return (
        "# source\n- Generated at: `2026-07-29T00:00:00Z`\n",
        "# docs\n- Generated at: `2026-07-29T00:00:00Z`\n",
        json.dumps({"generated_at": "2026-07-29T00:00:00Z", "snapshots": []}) + "\n",
    )


def _output_args(tmp_path: Path) -> tuple[list[str], Path, Path, Path]:
    source_out = tmp_path / "source.md"
    docs_out = tmp_path / "docs.md"
    summary_out = tmp_path / "summary.json"
    return (
        [
            "--source-output",
            str(source_out),
            "--docs-output",
            str(docs_out),
            "--summary-json",
            str(summary_out),
            "--top-limit",
            "8",
            "--full-limit",
            "12",
        ],
        source_out,
        docs_out,
        summary_out,
    )


def test_main_writes_both_scope_reports_and_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI writes the source and docs reports and a summary of both scopes."""
    args, source_out, docs_out, summary_out = _output_args(tmp_path)

    assert tool.main(args) == 0

    assert "# Internal readiness Register" in source_out.read_text(encoding="utf-8")
    assert "# Internal readiness Register" in docs_out.read_text(encoding="utf-8")
    payload = json.loads(summary_out.read_text(encoding="utf-8"))
    assert payload["generator"] == "tools/generate_readiness_scope_reports.py"
    assert {snap["scope"] for snap in payload["snapshots"]} == {"source", "docs_claims"}
    # Paths outside the repository are reported as they are.
    assert f"- {source_out.as_posix()}" in capsys.readouterr().out


def test_main_generates_and_check_passes(tmp_path: Path) -> None:
    source_out = tmp_path / "source.md"
    docs_out = tmp_path / "docs.md"
    summary_out = tmp_path / "summary.json"
    rc_gen = tool.main(
        [
            "--source-output",
            str(source_out),
            "--docs-output",
            str(docs_out),
            "--summary-json",
            str(summary_out),
            "--top-limit",
            "8",
            "--full-limit",
            "12",
        ]
    )
    assert rc_gen == 0
    rc_check = tool.main(
        [
            "--source-output",
            str(source_out),
            "--docs-output",
            str(docs_out),
            "--summary-json",
            str(summary_out),
            "--top-limit",
            "8",
            "--full-limit",
            "12",
            "--check",
        ]
    )
    assert rc_check == 0


def test_main_check_fails_on_missing_outputs(tmp_path: Path) -> None:
    rc = tool.main(
        [
            "--source-output",
            str(tmp_path / "missing_source.md"),
            "--docs-output",
            str(tmp_path / "missing_docs.md"),
            "--summary-json",
            str(tmp_path / "missing_summary.json"),
            "--check",
        ]
    )
    assert rc == 1


@pytest.mark.parametrize("missing_loader", [False, True])
def test_main_refuses_to_run_without_the_readiness_register_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, missing_loader: bool
) -> None:
    """Reject both a missing dynamic-import spec and a spec without a loader."""
    spec = ModuleSpec("generate_readiness_register", loader=None) if missing_loader else None
    monkeypatch.setattr(tool.importlib.util, "spec_from_file_location", lambda *_args: spec)

    with pytest.raises(RuntimeError, match="Failed to load"):
        tool.main(_output_args(tmp_path)[0])


def test_check_names_missing_repository_outputs_relative_to_the_repository(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Relative output paths resolve under the repository and are reported that way."""
    names = [
        f"docs/internal/missing_scope_check_{part}" for part in ("src.md", "docs.md", "s.json")
    ]
    argv = ["--source-output", names[0], "--docs-output", names[1], "--summary-json", names[2]]

    assert tool.main([*argv, "--top-limit", "8", "--full-limit", "12", "--check"]) == 1

    out = capsys.readouterr().out
    assert all(f"- {name}" in out for name in names)
    assert not any((ROOT / name).exists() for name in names)


def test_check_ignores_only_the_generation_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later run differs from the tracked reports only in its timestamps, which check ignores."""
    monkeypatch.setattr(tool, "_build_scope_reports", _stub_scope_reports)
    args, source_out, _docs_out, summary_out = _output_args(tmp_path)
    assert tool.main(args) == 0
    source_out.write_text(
        source_out.read_text(encoding="utf-8").replace(
            "2026-07-29T00:00:00Z", "2030-01-01T00:00:00Z"
        ),
        encoding="utf-8",
    )
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    summary["generated_at"] = "2030-01-01T00:00:00Z"
    summary_out.write_text(json.dumps(summary), encoding="utf-8")

    assert tool.main([*args, "--check"]) == 0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]", "summary payload must be a JSON object"),
        ('{"generated_at": 7}', "generated_at must be a string when present"),
    ],
)
def test_check_rejects_a_malformed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: str, message: str
) -> None:
    """Fail closed when the summary root or its timestamp has the wrong type."""
    monkeypatch.setattr(tool, "_build_scope_reports", _stub_scope_reports)
    args, _source_out, _docs_out, summary_out = _output_args(tmp_path)
    assert tool.main(args) == 0
    summary_out.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        tool.main([*args, "--check"])


@pytest.mark.parametrize(
    "invalid_limit",
    ["--top-limit", "--full-limit"],
)
def test_main_rejects_nonpositive_limits(invalid_limit: str, tmp_path: Path) -> None:
    """Reject nonpositive truncation limits before generating any output."""
    args, source_out, docs_out, summary_out = _output_args(tmp_path)
    args[args.index(invalid_limit) + 1] = "0"

    with pytest.raises(ValueError, match=f"{invalid_limit} must be >= 1"):
        tool.main(args)
    assert not any(path.exists() for path in (source_out, docs_out, summary_out))


def test_main_check_reports_each_output_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report source, docs, and summary drift through the public check boundary."""
    monkeypatch.setattr(tool, "_build_scope_reports", _stub_scope_reports)
    args, source_out, docs_out, summary_out = _output_args(tmp_path)
    assert tool.main(args) == 0

    source_out.write_text("drift\n", encoding="utf-8")
    assert tool.main([*args, "--check"]) == 1
    assert "Scope report drift detected" in capsys.readouterr().out
    source_out.write_text(_stub_scope_reports(top_limit=8, full_limit=12)[0], encoding="utf-8")

    docs_out.write_text("drift\n", encoding="utf-8")
    assert tool.main([*args, "--check"]) == 1
    assert "Scope report drift detected" in capsys.readouterr().out
    docs_out.write_text(_stub_scope_reports(top_limit=8, full_limit=12)[1], encoding="utf-8")

    summary_out.write_text('{"generated_at": "old", "snapshots": ["drift"]}\n', encoding="utf-8")
    assert tool.main([*args, "--check"]) == 1
    assert "Scope summary drift detected" in capsys.readouterr().out
