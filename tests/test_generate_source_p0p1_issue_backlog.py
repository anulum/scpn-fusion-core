"""Tests for tools/generate_source_p0p1_issue_backlog.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_source_p0p1_issue_backlog.py"
SPEC = importlib.util.spec_from_file_location("generate_source_p0p1_issue_backlog", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_collect_source_issues_filters_to_source_p0p1() -> None:
    issues = mod.collect_source_issues(ROOT)
    assert len(issues) > 0
    assert all(issue.file_path.startswith("src/scpn_fusion/") for issue in issues)
    assert all(issue.priority in {"P0", "P1"} for issue in issues)


def test_render_markdown_contains_expected_sections() -> None:
    issues = mod.collect_source_issues(ROOT)
    rendered = mod.render_markdown(issues)
    assert "# Source P0/P1 Issue Backlog" in rendered
    assert "## Auto-generated Issue Seeds" in rendered
    assert "Acceptance Checklist" in rendered


def test_main_writes_markdown_and_json(tmp_path: Path) -> None:
    md_path = tmp_path / "issues.md"
    json_path = tmp_path / "issues.json"
    rc = mod.main(
        [
            "--output-md",
            str(md_path),
            "--output-json",
            str(json_path),
        ]
    )
    assert rc == 0
    assert md_path.exists()
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(payload.get("issues"), list)
    assert payload.get("issue_count", 0) > 0

