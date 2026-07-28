# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for tools/generate_source_p0p1_readiness.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_source_p0p1_readiness.py"
SPEC = importlib.util.spec_from_file_location("generate_source_p0p1_readiness", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def _issue(
    *,
    file_path: str = "src/scpn_fusion/core/example.py",
    priority: str = "P0",
    score: int = 100,
    markers: tuple[str, ...] = ("MONOLITH",),
) -> Any:
    return mod.SourceIssue(
        file_path=file_path,
        domain="core",
        owner="Core WG",
        priority=priority,
        score=score,
        markers=markers,
        lines=(10, 20),
        proposed_actions=("Tighten the production contract.",),
    )


def test_collect_source_issues_filters_to_source_p0p1() -> None:
    issues = mod.collect_source_issues(ROOT)
    assert isinstance(issues, list)
    assert all(issue.file_path.startswith("src/scpn_fusion/") for issue in issues)
    assert all(issue.priority in {"P0", "P1"} for issue in issues)


def test_render_markdown_contains_expected_sections() -> None:
    issues = mod.collect_source_issues(ROOT)
    rendered = mod.render_markdown(issues)
    assert "# Source P0/P1 Issue Readiness" in rendered
    assert "## Auto-generated Issue Seeds" in rendered
    if issues:
        assert "Readiness Criteria" in rendered
        assert "Closure Metrics" in rendered


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
    assert isinstance(payload.get("issue_count"), int)
    if payload["issues"]:
        issue0 = payload["issues"][0]
        assert isinstance(issue0.get("closure_metrics"), list)
        assert isinstance(issue0.get("issue_slug"), str)
        assert isinstance(issue0.get("github_body"), str)


def test_coverage_thresholds_and_marker_closure_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve file/domain thresholds and every marker-specific closure rule."""
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(mod, "COVERAGE_THRESHOLDS_PATH", missing)
    assert mod._load_coverage_thresholds() == {}  # noqa: SLF001

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(mod, "COVERAGE_THRESHOLDS_PATH", invalid)
    assert mod._load_coverage_thresholds() == {}  # noqa: SLF001

    issue = _issue(markers=("MONOLITH", "DEPRECATED", "NOT_VALIDATED", "FALLBACK"))
    assert mod._module_domain_from_path("short.py") == "other"  # noqa: SLF001
    assert (
        mod._coverage_line_target(  # noqa: SLF001
            issue, {"file_min_line_rate": {issue.file_path: 99}}
        )
        == 99.0
    )
    assert (
        mod._coverage_line_target(  # noqa: SLF001
            issue,
            {
                "file_min_line_rate": {issue.file_path: "invalid"},
                "domain_min_line_rate": {"core": 97.5},
            },
        )
        == 97.5
    )
    assert (
        mod._coverage_line_target(  # noqa: SLF001
            issue, {"file_min_line_rate": [], "domain_min_line_rate": []}
        )
        is None
    )

    closure = mod._closure_metrics(  # noqa: SLF001
        issue, {"file_min_line_rate": {issue.file_path: 99}}
    )
    assert len(closure) == 6
    assert any("Deprecated-default-lane" in item for item in closure)
    assert any("Fallback budget" in item for item in closure)


@pytest.mark.parametrize("missing_loader", [False, True])
def test_load_readiness_module_rejects_invalid_spec(
    monkeypatch: pytest.MonkeyPatch, *, missing_loader: bool
) -> None:
    """Reject both a missing import spec and a spec without a loader."""
    spec = ModuleSpec("generate_readiness_register", loader=None) if missing_loader else None
    monkeypatch.setattr(mod.importlib.util, "spec_from_file_location", lambda *_args: spec)

    with pytest.raises(RuntimeError, match="Failed to load"):
        mod._load_readiness_module()  # noqa: SLF001


def test_collect_source_issues_supports_legacy_unscoped_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback-filter a legacy register and preserve grouped deterministic ordering."""
    entries = [
        SimpleNamespace(
            path="README.md",
            score=100,
            domain="docs",
            owner="Docs WG",
            marker="EXPERIMENTAL",
            line=1,
            proposed_action="Ignore non-source row.",
        ),
        SimpleNamespace(
            path="src/scpn_fusion/core/low.py",
            score=70,
            domain="core",
            owner="Core WG",
            marker="SIMPLIFIED",
            line=2,
            proposed_action="Ignore P2 row.",
        ),
        SimpleNamespace(
            path="src/scpn_fusion/io/p1.py",
            score=90,
            domain="io",
            owner="IO WG",
            marker="FALLBACK",
            line=30,
            proposed_action="Action B.",
        ),
        SimpleNamespace(
            path="src/scpn_fusion/io/p1.py",
            score=85,
            domain="io",
            owner="IO WG",
            marker="DEPRECATED",
            line=20,
            proposed_action="Action A.",
        ),
        SimpleNamespace(
            path="src/scpn_fusion/core/p0.py",
            score=96,
            domain="core",
            owner="Core WG",
            marker="MONOLITH",
            line=10,
            proposed_action="Split responsibility.",
        ),
    ]
    readiness = SimpleNamespace(collect_entries=lambda _root: entries)
    monkeypatch.setattr(mod, "_load_readiness_module", lambda: readiness)

    issues = mod.collect_source_issues(ROOT)
    assert [issue.file_path for issue in issues] == [
        "src/scpn_fusion/core/p0.py",
        "src/scpn_fusion/io/p1.py",
    ]
    assert issues[1].markers == ("DEPRECATED", "FALLBACK")
    assert issues[1].lines == (20, 30)
    assert issues[1].proposed_actions == ("Action A.", "Action B.")


def test_renderers_cover_all_acceptance_and_empty_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render all marker criteria, GitHub fields, thresholds, and empty reports."""
    issue = _issue(
        markers=("DEPRECATED", "NOT_VALIDATED", "SIMPLIFIED", "FALLBACK", "EXPERIMENTAL")
    )
    monkeypatch.setattr(
        mod,
        "_load_coverage_thresholds",
        lambda: {"domain_min_line_rate": {"core": 98.0}},
    )

    markdown = mod.render_markdown([issue])
    payload = json.loads(mod.render_json([issue]))
    assert "Remove deprecated runtime-default path" in markdown
    assert "Record fallback telemetry" in markdown
    assert "File line coverage in release lane is >= 98.0%" in markdown
    assert payload["issues"][0]["issue_slug"] == "p0-core-example"
    assert len(payload["issues"][0]["readiness_criteria"]) == 7
    assert mod.render_markdown([]).endswith("\n")
    assert json.loads(mod.render_json([]))["issues"] == []


def test_main_check_covers_missing_drift_and_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise every check-mode outcome through the production CLI boundary."""
    issue = _issue()
    monkeypatch.setattr(mod, "collect_source_issues", lambda _root: [issue])
    monkeypatch.setattr(mod, "_load_coverage_thresholds", lambda: {})
    md_path = tmp_path / "nested" / "issues.md"
    json_path = tmp_path / "nested" / "issues.json"
    args = ["--output-md", str(md_path), "--output-json", str(json_path)]

    assert mod.main([*args, "--check"]) == 1
    assert "Readiness outputs missing" in capsys.readouterr().out
    assert mod.main(args) == 0
    assert mod.main([*args, "--check"]) == 0
    assert "is up to date" in capsys.readouterr().out

    md_path.write_text("drift\n", encoding="utf-8")
    assert mod.main([*args, "--check"]) == 1
    assert "Markdown readiness drift" in capsys.readouterr().out
    assert mod.main(args) == 0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["issues"] = []
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    assert mod.main([*args, "--check"]) == 1
    assert "JSON readiness drift" in capsys.readouterr().out
    assert mod.main(args) == 0

    json_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="readiness payload must be a JSON object"):
        mod.main([*args, "--check"])

    payload = json.loads(mod.render_json([issue]))
    payload["generated_at"] = 7
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="generated_at must be a string"):
        mod.main([*args, "--check"])


def test_output_and_normalization_helpers_cover_relative_and_external_paths(
    tmp_path: Path,
) -> None:
    """Resolve output paths and normalize only dynamic generated-at fields."""
    relative = Path("docs/internal/example.md")
    resolved = mod._resolve_output(relative.as_posix())  # noqa: SLF001
    assert resolved == ROOT / relative
    assert mod._display_path(resolved) == relative.as_posix()  # noqa: SLF001
    assert mod._display_path(tmp_path / "external.md") == (tmp_path / "external.md").as_posix()  # noqa: SLF001
    assert (
        mod._normalize_for_check(  # noqa: SLF001
            "heading\n- Generated at: `old`\ntail\n"
        )
        == "heading\n- Generated at: `<dynamic>`\ntail"
    )
    assert mod._normalize_json_for_check("{}") == {"generated_at": "<dynamic>"}  # noqa: SLF001
