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
    assert isinstance(scoped, list)
    assert all(entry.path.startswith("src/scpn_fusion/") for entry in scoped)


def test_docs_claims_scope_filters_to_docs_domain() -> None:
    entries = underdev.collect_entries(ROOT)
    scoped = underdev._filter_entries_by_scope(entries, scope="docs_claims")
    assert isinstance(scoped, list)
    assert all(entry.domain == "docs_claims" for entry in scoped)


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


def test_deprecated_guard_lines_are_suppressed() -> None:
    assert underdev._is_marker_suppressed(
        rel_path="tools/deprecated_default_lane_guard.py",
        marker="DEPRECATED",
        line='print("Deprecated default lane guard failed.")',
        file_text="",
    )


def test_collect_source_heuristics_detects_monolith_and_test_gap(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src" / "scpn_fusion" / "core"
    tests_dir = repo_root / "tests"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    huge_module = src_dir / "synthetic_huge_module.py"
    huge_module.write_text(
        "\n".join("value = 1" for _ in range(underdev.SOURCE_MONOLITH_LOC_WARN + 5)),
        encoding="utf-8",
    )
    # Include at least one test file so test corpus generation path is exercised.
    (tests_dir / "test_dummy.py").write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

    entries = underdev._collect_source_heuristic_entries(repo_root)
    markers = {entry.marker for entry in entries}
    assert "MONOLITH" in markers
    assert "TEST_GAP" in markers


def test_narrative_docs_claims_receive_priority_penalty() -> None:
    penalty = underdev._score_context_penalty(
        rel_path="docs/DOE_ARPA_E_CONVERGENCE_PITCH.md",
        marker="EXPERIMENTAL",
        line="- Experimental data access remains in-progress.",
    )
    assert penalty >= 20


def test_release_claim_surfaces_do_not_receive_narrative_penalty() -> None:
    penalty = underdev._score_context_penalty(
        rel_path="docs/HONEST_SCOPE.md",
        marker="EXPERIMENTAL",
        line="| 5D gyrokinetic turbulence | Deliberately reduced-order for real-time control |",
    )
    assert penalty == 0


def test_main_check_mode_passes_when_output_is_current(tmp_path: Path) -> None:
    output_path = tmp_path / "UNDERDEVELOPED_REGISTER.md"
    rc_generate = underdev.main(["--output", str(output_path)])
    assert rc_generate == 0
    rc_check = underdev.main(["--output", str(output_path), "--check"])
    assert rc_check == 0


def test_main_check_mode_fails_when_output_is_missing(tmp_path: Path) -> None:
    output_path = tmp_path / "missing_underdeveloped_register.md"
    rc_check = underdev.main(["--output", str(output_path), "--check"])
    assert rc_check == 1


def test_main_check_mode_fails_on_drift(tmp_path: Path) -> None:
    output_path = tmp_path / "UNDERDEVELOPED_REGISTER.md"
    output_path.write_text("# stale\n", encoding="utf-8")
    rc_check = underdev.main(["--output", str(output_path), "--check"])
    assert rc_check == 1
