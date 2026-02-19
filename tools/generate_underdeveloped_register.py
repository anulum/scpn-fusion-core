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
}


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
)

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
    "control": 10,
    "core_physics": 9,
    "nuclear": 8,
    "diagnostics_io": 7,
    "compiler_runtime": 8,
    "validation": 6,
    "docs_claims": -6,
    "other": 0,
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
                key = (rel, lineno, rule.marker)
                if key in seen:
                    continue
                seen.add(key)
                file_hits += 1
                if file_hits > 20:
                    break
                score = int(rule.base_score + bonus)
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
            if file_hits > 20:
                break
    entries.sort(key=lambda e: (-e.score, e.domain, e.path, e.line))
    return entries


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
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    marker_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for entry in entries:
        marker_counts[entry.marker] = marker_counts.get(entry.marker, 0) + 1
        domain_counts[entry.domain] = domain_counts.get(entry.domain, 0) + 1

    lines: list[str] = [
        "# Underdeveloped Register",
        "",
        f"- Generated at: `{now}`",
        "- Generator: `tools/generate_underdeveloped_register.py`",
        "- Scope: production code + docs claims markers (tests/reports/html excluded)",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total flagged entries | {len(entries)} |",
        f"| P0 + P1 entries | {sum(1 for e in entries if _priority(e.score) in {'P0', 'P1'})} |",
        f"| Domains affected | {len(domain_counts)} |",
        "",
    ]
    lines.extend(_render_counts("Marker Distribution", marker_counts))
    lines.extend(_render_counts("Domain Distribution", domain_counts))

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output markdown path (default: UNDERDEVELOPED_REGISTER.md at repo root).",
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
    args = parser.parse_args(argv)

    if args.top_limit < 1:
        raise ValueError("--top-limit must be >= 1.")
    if args.full_limit < 1:
        raise ValueError("--full-limit must be >= 1.")

    entries = collect_entries(REPO_ROOT)
    report = render_markdown(
        entries=entries,
        top_limit=int(args.top_limit),
        full_limit=int(args.full_limit),
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.write_text(report, encoding="utf-8")
    print(
        f"Generated underdeveloped register with {len(entries)} entries: "
        f"{output_path.relative_to(REPO_ROOT).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
