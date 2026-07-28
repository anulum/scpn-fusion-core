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


def test_build_scope_reports_returns_nonempty_payloads() -> None:
    source_md, docs_md, summary_json = tool._build_scope_reports(top_limit=10, full_limit=20)  # noqa: SLF001
    assert "# Internal readiness Register" in source_md
    assert "# Internal readiness Register" in docs_md
    payload = json.loads(summary_json)
    assert payload["generator"] == "tools/generate_readiness_scope_reports.py"
    scopes = {snap["scope"] for snap in payload["snapshots"]}
    assert scopes == {"source", "docs_claims"}


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
def test_load_readiness_module_rejects_invalid_spec(
    monkeypatch: pytest.MonkeyPatch, *, missing_loader: bool
) -> None:
    """Reject both a missing dynamic-import spec and a spec without a loader."""
    spec = ModuleSpec("generate_readiness_register", loader=None) if missing_loader else None
    monkeypatch.setattr(tool.importlib.util, "spec_from_file_location", lambda *_args: spec)

    with pytest.raises(RuntimeError, match="Failed to load"):
        tool._load_readiness_module()  # noqa: SLF001


def test_path_and_normalization_helpers_cover_dynamic_fields(tmp_path: Path) -> None:
    """Resolve relative paths and normalize only timestamp-bearing fields."""
    relative = Path("docs/internal/example.md")
    resolved = tool._resolve(relative.as_posix())  # noqa: SLF001

    assert resolved == ROOT / relative
    assert tool._display_path(resolved) == relative.as_posix()  # noqa: SLF001
    assert tool._display_path(tmp_path / "outside.md") == (tmp_path / "outside.md").as_posix()  # noqa: SLF001
    assert (
        tool._normalize_for_check(  # noqa: SLF001
            "heading\n- Generated at: `old`\ntail\n"
        )
        == "heading\n- Generated at: `<dynamic>`\ntail"
    )
    assert tool._normalize_json_for_check("{}") == {"generated_at": "<dynamic>"}  # noqa: SLF001


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]", "summary payload must be a JSON object"),
        ('{"generated_at": 7}', "generated_at must be a string when present"),
    ],
)
def test_normalize_json_rejects_malformed_contracts(payload: str, message: str) -> None:
    """Fail closed when the summary root or dynamic timestamp has the wrong type."""
    with pytest.raises(ValueError, match=message):
        tool._normalize_json_for_check(payload)  # noqa: SLF001


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
