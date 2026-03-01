# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Underdeveloped Scope Reports Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_underdeveloped_scope_reports.py"
SPEC = importlib.util.spec_from_file_location("generate_underdeveloped_scope_reports", MODULE_PATH)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def test_build_scope_reports_returns_nonempty_payloads() -> None:
    source_md, docs_md, summary_json = tool._build_scope_reports(top_limit=10, full_limit=20)  # noqa: SLF001
    assert "# Underdeveloped Register" in source_md
    assert "# Underdeveloped Register" in docs_md
    payload = json.loads(summary_json)
    assert payload["generator"] == "tools/generate_underdeveloped_scope_reports.py"
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
