# ----------------------------------------------------------------------
# SCPN Fusion Core -- Underdeveloped Register Tests
# ----------------------------------------------------------------------
"""Tests for tools/generate_underdeveloped_register.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_underdeveloped_register.py"
SPEC = importlib.util.spec_from_file_location("generate_underdeveloped_register", MODULE_PATH)
assert SPEC and SPEC.loader
underdev = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = underdev
SPEC.loader.exec_module(underdev)


def test_collect_entries_returns_actionable_findings() -> None:
    entries = underdev.collect_entries(ROOT)
    assert len(entries) > 0
    assert any(entry.path.startswith("src/scpn_fusion/") for entry in entries)
    assert any(entry.marker in {"DEPRECATED", "EXPERIMENTAL", "SIMPLIFIED"} for entry in entries)
    assert all(entry.path != "docs/CLAIMS_EVIDENCE_MAP.md" for entry in entries)
    assert all(entry.path != "docs/SOURCE_P0P1_ISSUE_BACKLOG.md" for entry in entries)
    assert all(entry.path != "docs/SOURCE_P0P1_ISSUE_BACKLOG.json" for entry in entries)


def test_render_markdown_contains_priority_sections() -> None:
    entries = underdev.collect_entries(ROOT)
    rendered = underdev.render_markdown(entries=entries, top_limit=10, full_limit=20)
    assert "# Underdeveloped Register" in rendered
    assert "## Source-Centric Priority Backlog" in rendered
    assert "## Top Priority Backlog" in rendered
    assert "## Full Register" in rendered


def test_source_scope_filters_to_src_paths() -> None:
    entries = underdev.collect_entries(ROOT)
    scoped = underdev._filter_entries_by_scope(entries, scope="source")
    assert len(scoped) > 0
    assert all(entry.path.startswith("src/scpn_fusion/") for entry in scoped)
