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
    rule_markers = {rule.marker for rule in underdev.MARKER_RULES}
    assert all(entry.marker in rule_markers for entry in entries)
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


def test_fallback_metadata_lines_are_suppressed() -> None:
    assert underdev._is_marker_suppressed(
        rel_path="src/scpn_fusion/control/disruption_predictor.py",
        marker="FALLBACK",
        line='"fallback": True,',
        file_text="",
    )


def test_fallback_comment_lines_are_suppressed() -> None:
    assert underdev._is_marker_suppressed(
        rel_path="src/scpn_fusion/core/fusion_kernel.py",
        marker="FALLBACK",
        line="# Python fallback path",
        file_text="",
    )


def test_analytic_solver_fallback_line_is_suppressed_when_guarded() -> None:
    assert underdev._is_marker_suppressed(
        rel_path="src/scpn_fusion/control/analytic_solver.py",
        marker="FALLBACK",
        line='fallback = repo_root / "validation" / "iter_validated_config.json"',
        file_text=(
            'allow_validation_fallback=True\n'
            '"config_source": "validation_fallback_default"\n'
            '"fallback_used": True\n'
        ),
    )


def test_fallback_assignment_metadata_line_is_suppressed() -> None:
    assert underdev._is_marker_suppressed(
        rel_path="src/scpn_fusion/control/disruption_predictor.py",
        marker="FALLBACK",
        line='info["fallback"] = False',
        file_text="",
    )
