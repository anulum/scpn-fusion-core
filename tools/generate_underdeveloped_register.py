#!/usr/bin/env python
"""Generate an actionable underdeveloped/simplified register from repo markers."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "UNDERDEVELOPED_REGISTER.md"

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".rst",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
}
EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".hypothesis",
    "__pycache__",
    "artifacts",
    "validation/reports",
    "docs/notebooks",
    "tests",
    "scpn-fusion-rs/target",
}
INCLUDED_ROOTS = (
    "src",
    "docs",
    "validation",
    "tools",
    "README.md",
    "RESULTS.md",
    "VALIDATION.md",
    "CHANGELOG.md",
)
EXCLUDED_SUFFIXES = {".html"}
EXCLUDED_PATHS = {
    "tools/generate_underdeveloped_register.py",
    "validation/claims_manifest.json",
    "docs/V3_6_MILESTONE_BOARD.md",
    "docs/CLAIMS_EVIDENCE_MAP.md",
    "docs/SOURCE_P0P1_ISSUE_BACKLOG.md",
    "docs/SOURCE_P0P1_ISSUE_BACKLOG.json",
    "docs/UNDERDEVELOPED_SOURCE_REGISTER.md",
    "docs/UNDERDEVELOPED_DOCS_CLAIMS_REGISTER.md",
    "docs/UNDERDEVELOPED_SCOPE_SUMMARY.json",
}

# Release-critical claim surfaces should remain high-signal in the queue.
RELEASE_CLAIM_SURFACES = {
    "README.md",
    "RESULTS.md",
    "VALIDATION.md",
    "docs/HONEST_SCOPE.md",
    "docs/BENCHMARKS.md",
    "docs/VALIDATION_GATE_MATRIX.md",
    "docs/V3_9_3_RELEASE_CHECKLIST.md",
    "docs/RELEASE_ACCEPTANCE_CHECKLIST.md",
    "docs/competitive_analysis.md",
    "docs/sphinx/userguide/validation.rst",
}

# Lower-priority narrative/planning docs are still tracked, but should not
# dominate the P0/P1 hardening queue ahead of implementation-facing risks.
NARRATIVE_DOC_PREFIXES = (
    "docs/promotions/",
    "docs/rfc/",
    "docs/PHASE3_EXECUTION_REGISTRY.md",
    "docs/DEEP_AUDIT_AND_SOTA_PLAN_",
    "docs/HARDENING_30_DAY_EXECUTION_PLAN.md",
    "docs/DOE_ARPA_E_CONVERGENCE_PITCH.md",
    "docs/PACKET_",
    "docs/session_logs/",
)


@dataclass(frozen=True)
class MarkerRule:
    marker: str
    pattern: re.Pattern[str]
    base_score: int
    proposed_action: str


@dataclass(frozen=True)
class RegisterEntry:
    path: str
    line: int
    marker: str
    snippet: str
    domain: str
    owner: str
    score: int
    proposed_action: str


MARKER_RULES: tuple[MarkerRule, ...] = (
    MarkerRule(
        marker="DEPRECATED",
        pattern=re.compile(r"\bdeprecated\b", flags=re.IGNORECASE),
        base_score=95,
        proposed_action="Replace default path or remove lane before next major release.",
    ),
    MarkerRule(
        marker="EXPERIMENTAL",
        pattern=re.compile(r"\bexperimental\b", flags=re.IGNORECASE),
        base_score=88,
        proposed_action="Gate behind explicit flag and define validation exit criteria.",
    ),
    MarkerRule(
        marker="NOT_VALIDATED",
        pattern=re.compile(r"\bnot validated\b", flags=re.IGNORECASE),
        base_score=86,
        proposed_action="Add real-data validation campaign and publish error bars.",
    ),
    MarkerRule(
        marker="SIMPLIFIED",
        pattern=re.compile(r"\bsimplified\b", flags=re.IGNORECASE),
        base_score=74,
        proposed_action="Upgrade with higher-fidelity closure or tighten domain contract.",
    ),
    MarkerRule(
        marker="FALLBACK",
        pattern=re.compile(r"\bfallback\b", flags=re.IGNORECASE),
        base_score=65,
        proposed_action="Measure fallback hit-rate and retire fallback from default lane.",
    ),
    MarkerRule(
        marker="PLANNED",
        pattern=re.compile(r"\bplanned\b", flags=re.IGNORECASE),
        base_score=55,
        proposed_action="Convert roadmap note into scheduled milestone task + owner.",
    ),
    MarkerRule(
        marker="MONOLITH",
        pattern=re.compile(r"$^"),
        base_score=90,
        proposed_action="Split module into focused subcomponents and lock interface contracts.",
    ),
    MarkerRule(
        marker="FALLBACK_DENSITY",
        pattern=re.compile(r"$^"),
        base_score=84,
        proposed_action="Reduce fallback concentration and enforce strict-backend parity checks.",
    ),
    MarkerRule(
        marker="TEST_GAP",
        pattern=re.compile(r"$^"),
        base_score=88,
        proposed_action="Add direct module tests and eliminate allowlist-only linkage.",
    ),
)

SOURCE_MONOLITH_LOC_WARN = 500
SOURCE_MONOLITH_LOC_CRITICAL = 800
SOURCE_FALLBACK_DENSITY_WARN = 6
SOURCE_FALLBACK_DENSITY_CRITICAL = 12
SOURCE_MIN_LOC_FOR_FALLBACK_DENSITY = 180
SOURCE_TEST_GAP_LOC_THRESHOLD = 350
_MARKER_RULE_BY_NAME = {rule.marker: rule for rule in MARKER_RULES}


def _is_marker_suppressed(
    *,
    rel_path: str,
    marker: str,
    line: str,
    file_text: str,
) -> bool:
    """Suppress known false positives where hardening guardrails already exist."""
    lowered_line = line.lower()
    normalized_line = " ".join(line.strip().split()).lower()
    rel_is_release_claim_surface = rel_path in RELEASE_CLAIM_SURFACES

    if marker == "DEPRECATED":
        if "deprecated-default-lane guard" in normalized_line:
            return True
        if "deprecated-default-lane-guard" in normalized_line:
            return True

    if marker == "EXPERIMENTAL" and rel_is_release_claim_surface:
        # Commands/docs that explicitly *gate* experimental lanes should not be
        # treated as unresolved underdevelopment debt.
        if "--experimental" in lowered_line:
            return True
        if "@pytest.mark.experimental" in lowered_line:
            return True
        if "pytest -m experimental" in lowered_line:
            return True
        if "experimental marker contract" in lowered_line:
            return True
        if "experimental-only" in lowered_line:
            return True
        if "python-research-gate" in lowered_line:
            return True
        if "experimental/research" in lowered_line:
            return True
        if "not experimental" in lowered_line:
            return True
        if "experimental tests" in lowered_line and "exclude" in lowered_line:
            return True

    if rel_path in {
        "tools/deprecated_default_lane_guard.py",
        "validation/benchmark_deprecated_mode_exclusion.py",
    } and marker in {"DEPRECATED", "EXPERIMENTAL"}:
        # Guard/benchmark internals should not self-trigger top-priority backlog noise.
        return True
    if rel_path == "tools/run_python_preflight.py" and marker == "DEPRECATED":
        if "deprecated default lane guard" in lowered_line:
            return True
        if "--skip-deprecated-default-lane-guard" in lowered_line:
            return True
    if rel_path == "tools/generate_source_p0p1_issue_backlog.py" and marker in {
        "DEPRECATED",
        "EXPERIMENTAL",
        "FALLBACK",
        "SIMPLIFIED",
    }:
        if "if \"" in line and "\" in markers" in line:
            return True
        if "acceptance" in lowered_line and "checklist" in lowered_line:
            return True
    if rel_path == "tools/fallback_budget_guard.py" and marker == "FALLBACK":
        if "fallback budget summary" in lowered_line:
            return True
        if "fallback budget guard failed" in lowered_line:
            return True
        if "fallback budget guard passed" in lowered_line:
            return True
    if rel_path == "src/scpn_fusion/cli.py" and marker == "EXPERIMENTAL":
        # The launcher has explicit opt-in + acknowledgement gating for experimental modes.
        return (
            "EXPERIMENTAL_ACK_TOKEN" in file_text
            and "experimental acknowledgement missing" in file_text
            and "--experimental-ack" in file_text
        )
    if marker == "EXPERIMENTAL" and rel_path in {
        "validation/full_validation_pipeline.py",
        "validation/run_experimental_validation.py",
        "validation/validate_against_sparc.py",
    }:
        # Experimental validation entrypoints are explicitly locked behind flag+ack.
        return (
            "require_experimental_opt_in" in file_text
            and "--experimental-ack" in file_text
            and "SCPN_EXPERIMENTAL_ACK" in file_text
        )
    if marker == "EXPERIMENTAL" and rel_path in {
        "validation/stress_test_campaign.py",
        "validation/controller_comparison.py",
    }:
        if "experimental policy path" in lowered_line:
            return True
    if marker == "FALLBACK":
        stripped = line.strip()
        # Pure comments/docstring headers often document an already-hardened fallback lane.
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            return True
        if rel_path == "src/scpn_fusion/control/analytic_solver.py":
            # Default-path fallback is now explicit and policy-controlled.
            return (
                "allow_validation_fallback" in file_text
                and '"config_source"' in file_text
                and '"fallback_used"' in file_text
            )
        lowered = line.lower()
        # Metadata keys and explicit fallback-control fields are observability, not risk markers.
        if re.search(r"[\"']fallback[\"']\s*:", lowered):
            return True
        if re.search(r"\[[\"']fallback[\"']\]\s*=", lowered):
            return True
        if 'mode"] = "fallback"' in lowered or "mode'] = 'fallback'" in lowered:
            return True
        if "fallback disabled" in lowered:
            return True
        if "fallback_used" in lowered or "fallback_reason" in lowered:
            return True
        if "allow_fallback" in lowered or "allow_numpy_fallback" in lowered:
            return True
    return False


def _count_nontrivial_loc(text: str) -> int:
    count = 0
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        count += 1
    return count


def _fallback_density_signal_count(text: str) -> int:
    """Count fallback-risk signals while ignoring observability/config noise."""
    count = 0
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lowered = stripped.lower()
        if "fallback" not in lowered:
            continue
        if "record_fallback_event(" in lowered:
            count += 1
            continue
        if re.search(r"\bif\s+not\s+[a-z_]*fallback\b", lowered):
            count += 1
            continue
        if re.search(r"\bif\s+[a-z_]*fallback\b", lowered):
            count += 1
            continue
        if re.search(r"\bexcept\b.*\bfallback\b", lowered):
            count += 1
            continue
    return count


def _has_direct_test_linkage(*, rel_path: str, test_corpus: str) -> bool:
    module_rel = rel_path.removeprefix("src/").removesuffix(".py")
    import_path = module_rel.replace("/", ".")
    stem = Path(rel_path).stem
    return (
        (import_path in test_corpus)
        or (f"test_{stem}" in test_corpus)
        or (f"from {import_path}" in test_corpus)
        or (f"import {import_path}" in test_corpus)
    )


def _collect_source_heuristic_entries(repo_root: Path) -> list[RegisterEntry]:
    entries: list[RegisterEntry] = []
    tests_root = repo_root / "tests"
    test_corpus_parts: list[str] = []
    for test_file in sorted(tests_root.rglob("test_*.py")):
        test_corpus_parts.append(test_file.read_text(encoding="utf-8", errors="ignore"))
    test_corpus = "\n".join(test_corpus_parts)

    for path in sorted((repo_root / "src" / "scpn_fusion").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel_path = _normalize(path.relative_to(repo_root))
        domain = _domain_for(rel_path)
        if domain not in SOURCE_DOMAINS:
            continue
        owner = DOMAIN_OWNER[domain]
        bonus = DOMAIN_BONUS.get(domain, 0)
        text = path.read_text(encoding="utf-8", errors="ignore")
        loc = _count_nontrivial_loc(text)
        fallback_mentions = _fallback_density_signal_count(text)
        has_linkage = _has_direct_test_linkage(rel_path=rel_path, test_corpus=test_corpus)

        if loc >= SOURCE_MONOLITH_LOC_WARN:
            base = SOURCE_MONOLITH_LOC_CRITICAL if loc >= SOURCE_MONOLITH_LOC_CRITICAL else SOURCE_MONOLITH_LOC_WARN
            marker = "MONOLITH"
            rule = _MARKER_RULE_BY_NAME[marker]
            score = int(rule.base_score + bonus + (8 if base == SOURCE_MONOLITH_LOC_CRITICAL else 0))
            entries.append(
                RegisterEntry(
                    path=rel_path,
                    line=1,
                    marker=marker,
                    snippet=f"module LOC={loc} exceeds monolith threshold ({SOURCE_MONOLITH_LOC_WARN}+).",
                    domain=domain,
                    owner=owner,
                    score=score,
                    proposed_action=rule.proposed_action,
                )
            )

        if loc >= SOURCE_MIN_LOC_FOR_FALLBACK_DENSITY and fallback_mentions >= SOURCE_FALLBACK_DENSITY_WARN:
            is_critical = fallback_mentions >= SOURCE_FALLBACK_DENSITY_CRITICAL
            marker = "FALLBACK_DENSITY"
            rule = _MARKER_RULE_BY_NAME[marker]
            score = int(rule.base_score + bonus + (6 if is_critical else 0))
            entries.append(
                RegisterEntry(
                    path=rel_path,
                    line=1,
                    marker=marker,
                    snippet=(
                        f"fallback risk signals={fallback_mentions} across LOC={loc}; "
                        "high fallback concentration in runtime code paths."
                    ),
                    domain=domain,
                    owner=owner,
                    score=score,
                    proposed_action=rule.proposed_action,
                )
            )

        if loc >= SOURCE_TEST_GAP_LOC_THRESHOLD and not has_linkage:
            marker = "TEST_GAP"
            rule = _MARKER_RULE_BY_NAME[marker]
            score = int(rule.base_score + bonus)
            entries.append(
                RegisterEntry(
                    path=rel_path,
                    line=1,
                    marker=marker,
                    snippet="large source module without direct test import/stem linkage.",
                    domain=domain,
                    owner=owner,
                    score=score,
                    proposed_action=rule.proposed_action,
                )
            )
    return entries


def _score_context_penalty(*, rel_path: str, marker: str, line: str) -> int:
    """Return score penalty for governance/audit marker contexts.

    These contexts are still tracked in the register, but they should not
    dominate the top P0/P1 queue intended for hardening implementation gaps.
    """
    lowered = line.lower()
    penalty = 0

    # Non-release narrative docs should not crowd out release-critical triage.
    if rel_path.startswith("docs/") and rel_path not in RELEASE_CLAIM_SURFACES:
        if rel_path.startswith(NARRATIVE_DOC_PREFIXES):
            if marker in {"DEPRECATED", "EXPERIMENTAL", "NOT_VALIDATED"}:
                penalty += 40
            elif marker in {"SIMPLIFIED", "FALLBACK", "PLANNED"}:
                penalty += 28
        elif marker in {"EXPERIMENTAL", "PLANNED", "FALLBACK"}:
            penalty += 18
        elif marker in {"DEPRECATED", "NOT_VALIDATED"}:
            penalty += 12

    if rel_path == "tools/generate_source_p0p1_issue_backlog.py":
        if marker in {"DEPRECATED", "EXPERIMENTAL"}:
            penalty += 26
    if rel_path == "validation/benchmark_deprecated_mode_exclusion.py":
        if marker in {"DEPRECATED", "EXPERIMENTAL"}:
            penalty += 26
    if rel_path == "tools/run_python_preflight.py" and marker == "EXPERIMENTAL":
        if "experimental-only" in lowered or '-m", "experimental"' in lowered:
            penalty += 18
    if rel_path == "validation/collect_results.py" and marker == "DEPRECATED":
        if "deprecated" in lowered and "fno" in lowered:
            penalty += 14
    if rel_path in {"CHANGELOG.md", "docs/sphinx/changelog.rst"}:
        if marker in {"DEPRECATED", "EXPERIMENTAL"}:
            penalty += 18
    return penalty


DOMAIN_OWNER = {
    "control": "Control WG",
    "core_physics": "Core Physics WG",
    "nuclear": "Nuclear WG",
    "diagnostics_io": "Diagnostics/IO WG",
    "compiler_runtime": "Runtime WG",
    "docs_claims": "Docs WG",
    "validation": "Validation WG",
    "other": "Architecture WG",
}
DOMAIN_BONUS = {
    "control": 12,
    "core_physics": 11,
    "nuclear": 10,
    "diagnostics_io": 9,
    "compiler_runtime": 10,
    "validation": 8,
    "docs_claims": -18,
    "other": 0,
}

SOURCE_DOMAINS = {
    "control",
    "core_physics",
    "nuclear",
    "diagnostics_io",
    "compiler_runtime",
    "validation",
}


def _normalize(path: Path) -> str:
    rel_path = path
    if path.is_absolute():
        try:
            rel_path = path.relative_to(REPO_ROOT)
        except ValueError:
            rel_path = path
    return rel_path.as_posix().lstrip("./")


def _is_excluded(path: Path) -> bool:
    posix = _normalize(path)
    if posix in EXCLUDED_PATHS:
        return True
    if any(posix == item or posix.startswith(f"{item}/") for item in EXCLUDED_DIR_NAMES):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return False


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    return path.name in {"README.md", "RESULTS.md", "VALIDATION.md", "CHANGELOG.md"}


def _domain_for(path: str) -> str:
    if path.startswith("src/scpn_fusion/control/"):
        return "control"
    if path.startswith("src/scpn_fusion/core/"):
        return "core_physics"
    if path.startswith("src/scpn_fusion/nuclear/"):
        return "nuclear"
    if path.startswith("src/scpn_fusion/scpn/") or path.startswith("src/scpn_fusion/hpc/"):
        return "compiler_runtime"
    if path.startswith("src/scpn_fusion/diagnostics/") or path.startswith("src/scpn_fusion/io/"):
        return "diagnostics_io"
    if path.startswith("validation/") or path.startswith("tools/"):
        return "validation"
    if path.startswith("README.md") or path.startswith("RESULTS.md") or path.startswith("docs/"):
        return "docs_claims"
    return "other"


def _priority(score: int) -> str:
    if score >= 95:
        return "P0"
    if score >= 82:
        return "P1"
    if score >= 68:
        return "P2"
    return "P3"


def _clean_snippet(line: str) -> str:
    collapsed = " ".join(line.strip().split())
    if len(collapsed) > 140:
        return f"{collapsed[:137]}..."
    return collapsed.replace("|", "\\|")


def _iter_candidate_files(repo_root: Path) -> Iterable[Path]:
    for item in INCLUDED_ROOTS:
        root = repo_root / item
        if root.is_file():
            if not _is_excluded(root) and _is_text_file(root):
                yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if _is_excluded(path):
                continue
            if not _is_text_file(path):
                continue
            yield path


def collect_entries(repo_root: Path) -> list[RegisterEntry]:
    entries: list[RegisterEntry] = []
    seen: set[tuple[str, int, str]] = set()

    for path in _iter_candidate_files(repo_root):
        rel = _normalize(path.relative_to(repo_root))
        domain = _domain_for(rel)
        owner = DOMAIN_OWNER[domain]
        bonus = DOMAIN_BONUS.get(domain, 0)
        text = path.read_text(encoding="utf-8", errors="ignore")
        file_hits = 0

        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            for rule in MARKER_RULES:
                if not rule.pattern.search(line):
                    continue
                if _is_marker_suppressed(
                    rel_path=rel,
                    marker=rule.marker,
                    line=line,
                    file_text=text,
                ):
                    continue
                key = (rel, lineno, rule.marker)
                if key in seen:
                    continue
                seen.add(key)
                file_hits += 1
                if file_hits >= 20:
                    break
                score = int(rule.base_score + bonus - _score_context_penalty(
                    rel_path=rel,
                    marker=rule.marker,
                    line=line,
                ))
                entries.append(
                    RegisterEntry(
                        path=rel,
                        line=lineno,
                        marker=rule.marker,
                        snippet=_clean_snippet(line),
                        domain=domain,
                        owner=owner,
                        score=score,
                        proposed_action=rule.proposed_action,
                    )
                )
            if file_hits >= 20:
                break
    for entry in _collect_source_heuristic_entries(repo_root):
        key = (entry.path, entry.line, entry.marker)
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)
    entries.sort(key=lambda e: (-e.score, e.domain, e.path, e.line))
    return entries


def _filter_entries_by_scope(entries: list[RegisterEntry], *, scope: str) -> list[RegisterEntry]:
    if scope == "full":
        return list(entries)
    if scope == "source":
        return [entry for entry in entries if entry.path.startswith("src/scpn_fusion/")]
    if scope == "docs_claims":
        return [entry for entry in entries if entry.domain == "docs_claims"]
    raise ValueError(f"Unsupported scope: {scope}")


def _render_counts(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| Key | Count |", "|---|---:|"]
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    return lines


def render_markdown(
    *,
    entries: list[RegisterEntry],
    top_limit: int,
    full_limit: int,
    scope: str = "full",
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    marker_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for entry in entries:
        marker_counts[entry.marker] = marker_counts.get(entry.marker, 0) + 1
        domain_counts[entry.domain] = domain_counts.get(entry.domain, 0) + 1

    source_entries = [entry for entry in entries if entry.domain in SOURCE_DOMAINS]
    docs_entries = [entry for entry in entries if entry.domain == "docs_claims"]
    source_p0p1 = [entry for entry in source_entries if _priority(entry.score) in {"P0", "P1"}]
    source_backlog = sorted(source_p0p1, key=lambda e: (-e.score, e.domain, e.path, e.line))
    if scope == "source":
        source_scope_note = "source-only (`src/scpn_fusion/**`) markers"
    elif scope == "docs_claims":
        source_scope_note = "docs-claims-only markers"
    else:
        source_scope_note = "production code + docs claims markers (tests/reports/html excluded)"

    lines: list[str] = [
        "# Underdeveloped Register",
        "",
        f"- Generated at: `{now}`",
        "- Generator: `tools/generate_underdeveloped_register.py`",
        f"- Scope: {source_scope_note}",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total flagged entries | {len(entries)} |",
        f"| P0 + P1 entries | {sum(1 for e in entries if _priority(e.score) in {'P0', 'P1'})} |",
        f"| Source-domain entries | {len(source_entries)} |",
        f"| Source-domain P0 + P1 entries | {len(source_backlog)} |",
        f"| Docs-claims entries | {len(docs_entries)} |",
        f"| Domains affected | {len(domain_counts)} |",
        "",
    ]
    lines.extend(_render_counts("Marker Distribution", marker_counts))
    lines.extend(_render_counts("Domain Distribution", domain_counts))

    if scope == "full":
        lines.extend(
            [
                f"## Source-Centric Priority Backlog (Top {min(top_limit, len(source_backlog))})",
                "",
                "_Filtered to implementation domains to reduce docs/claims noise during hardening triage._",
                "",
                "| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |",
                "|---|---:|---|---|---|---|---|---|",
            ]
        )
        for entry in source_backlog[:top_limit]:
            lines.append(
                "| "
                f"{_priority(entry.score)} | {entry.score} | `{entry.domain}` | `{entry.marker}` | "
                f"`{entry.path}:{entry.line}` | {entry.owner} | {entry.proposed_action} | {entry.snippet} |"
            )
        lines.append("")

    lines.extend(
        [
            f"## Top Priority Backlog (Top {min(top_limit, len(entries))})",
            "",
            "| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |",
            "|---|---:|---|---|---|---|---|---|",
        ]
    )
    for entry in entries[:top_limit]:
        lines.append(
            "| "
            f"{_priority(entry.score)} | {entry.score} | `{entry.domain}` | `{entry.marker}` | "
            f"`{entry.path}:{entry.line}` | {entry.owner} | {entry.proposed_action} | {entry.snippet} |"
        )
    lines.append("")

    lines.extend(
        [
            f"## Full Register (Top {min(full_limit, len(entries))})",
            "",
            "| Priority | Domain | Marker | Location | Snippet |",
            "|---|---|---|---|---|",
        ]
    )
    for entry in entries[:full_limit]:
        lines.append(
            f"| {_priority(entry.score)} | `{entry.domain}` | `{entry.marker}` | "
            f"`{entry.path}:{entry.line}` | {entry.snippet} |"
        )
    lines.append("")
    return "\n".join(lines)


def _resolve_output(path_value: str) -> Path:
    output_path = Path(path_value)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    return output_path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_for_check(content: str) -> str:
    # Generated timestamps change every run; compare everything else.
    lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("- Generated at: `"):
            lines.append("- Generated at: `<dynamic>`")
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output markdown path (default: UNDERDEVELOPED_REGISTER.md at repo root).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: fail if output differs from generated content.",
    )
    parser.add_argument(
        "--top-limit",
        type=int,
        default=80,
        help="Number of top entries in the priority table.",
    )
    parser.add_argument(
        "--full-limit",
        type=int,
        default=250,
        help="Number of entries to include in the full register section.",
    )
    parser.add_argument(
        "--scope",
        choices=("full", "source", "docs_claims"),
        default="full",
        help=(
            "Report scope: full (default), source (src/scpn_fusion only), "
            "or docs_claims."
        ),
    )
    args = parser.parse_args(argv)

    if args.top_limit < 1:
        raise ValueError("--top-limit must be >= 1.")
    if args.full_limit < 1:
        raise ValueError("--full-limit must be >= 1.")

    entries = collect_entries(REPO_ROOT)
    entries = _filter_entries_by_scope(entries, scope=str(args.scope))
    report = render_markdown(
        entries=entries,
        top_limit=int(args.top_limit),
        full_limit=int(args.full_limit),
        scope=str(args.scope),
    )
    output_path = _resolve_output(str(args.output))

    if args.check:
        if not output_path.exists():
            print(
                "Underdeveloped register output missing. "
                "Run without --check to generate:\n"
                f"- {_display_path(output_path)}"
            )
            return 1
        current = output_path.read_text(encoding="utf-8")
        if _normalize_for_check(current) != _normalize_for_check(report):
            print(f"Underdeveloped register drift detected: {_display_path(output_path)}")
            return 1
        print(
            f"Underdeveloped register is up to date ({len(entries)} entries, "
            f"scope={args.scope})."
        )
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(
        f"Generated underdeveloped register with {len(entries)} entries "
        f"(scope={args.scope}): {_display_path(output_path)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
