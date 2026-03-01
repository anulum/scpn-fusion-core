#!/usr/bin/env python
"""Generate source-only P0/P1 issue backlog from underdeveloped markers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
UNDERDEV_MODULE_PATH = REPO_ROOT / "tools" / "generate_underdeveloped_register.py"
DEFAULT_OUTPUT_MD = REPO_ROOT / "docs" / "SOURCE_P0P1_ISSUE_BACKLOG.md"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "docs" / "SOURCE_P0P1_ISSUE_BACKLOG.json"


@dataclass(frozen=True)
class SourceIssue:
    file_path: str
    domain: str
    owner: str
    priority: str
    score: int
    markers: tuple[str, ...]
    lines: tuple[int, ...]
    proposed_actions: tuple[str, ...]


def _priority(score: int) -> str:
    if score >= 95:
        return "P0"
    if score >= 82:
        return "P1"
    if score >= 68:
        return "P2"
    return "P3"


def _load_underdeveloped_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "generate_underdeveloped_register",
        UNDERDEV_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tools/generate_underdeveloped_register.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect_source_issues(repo_root: Path) -> list[SourceIssue]:
    underdev = _load_underdeveloped_module()
    entries = underdev.collect_entries(repo_root)
    if hasattr(underdev, "_filter_entries_by_scope"):
        entries = underdev._filter_entries_by_scope(entries, scope="source")

    grouped: dict[str, list[Any]] = {}
    for entry in entries:
        if not entry.path.startswith("src/scpn_fusion/"):
            continue
        if _priority(int(entry.score)) not in {"P0", "P1"}:
            continue
        grouped.setdefault(entry.path, []).append(entry)

    issues: list[SourceIssue] = []
    for file_path, file_entries in grouped.items():
        score = max(int(item.score) for item in file_entries)
        priority = _priority(score)
        domain = str(file_entries[0].domain)
        owner = str(file_entries[0].owner)
        markers = tuple(sorted({str(item.marker) for item in file_entries}))
        lines = tuple(sorted({int(item.line) for item in file_entries}))
        actions = tuple(sorted({str(item.proposed_action) for item in file_entries}))
        issues.append(
            SourceIssue(
                file_path=file_path,
                domain=domain,
                owner=owner,
                priority=priority,
                score=score,
                markers=markers,
                lines=lines,
                proposed_actions=actions,
            )
        )

    def _sort_key(issue: SourceIssue) -> tuple[int, int, str]:
        pri_rank = 0 if issue.priority == "P0" else 1
        return (pri_rank, -issue.score, issue.file_path)

    return sorted(issues, key=_sort_key)


def _acceptance_items(markers: tuple[str, ...]) -> list[str]:
    items: list[str] = [
        "Add or tighten regression tests for this module path and update coverage baselines.",
        "Update claim/evidence references if behavior or metrics change.",
    ]
    if "DEPRECATED" in markers:
        items.append("Remove deprecated runtime-default path or replace with validated default lane.")
    if "NOT_VALIDATED" in markers:
        items.append("Publish real-data validation artifact and link it from RESULTS/claims map.")
    if "SIMPLIFIED" in markers:
        items.append("Document model-domain limits and tighten contract checks for out-of-domain inputs.")
    if "FALLBACK" in markers:
        items.append("Record fallback telemetry and enforce strict-backend parity where applicable.")
    if "EXPERIMENTAL" in markers:
        items.append("Ensure release lane remains experimental-excluded unless explicitly opted in.")
    # Keep deterministic order with de-duplication.
    unique: dict[str, None] = {}
    for item in items:
        unique.setdefault(item, None)
    return list(unique.keys())


def render_markdown(issues: list[SourceIssue]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    p0_count = sum(1 for issue in issues if issue.priority == "P0")
    p1_count = sum(1 for issue in issues if issue.priority == "P1")
    marker_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for issue in issues:
        domain_counts[issue.domain] = domain_counts.get(issue.domain, 0) + 1
        for marker in issue.markers:
            marker_counts[marker] = marker_counts.get(marker, 0) + 1

    lines: list[str] = [
        "# Source P0/P1 Issue Backlog",
        "",
        f"- Generated at: `{now}`",
        "- Generator: `tools/generate_source_p0p1_issue_backlog.py`",
        "- Scope: source files only (`src/scpn_fusion/**`) with P0/P1 severity",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Source issue seeds | {len(issues)} |",
        f"| P0 seeds | {p0_count} |",
        f"| P1 seeds | {p1_count} |",
        f"| Domains represented | {len(domain_counts)} |",
        "",
        "## Marker Distribution",
        "",
        "| Marker | Count |",
        "|---|---:|",
    ]

    for marker, count in sorted(marker_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{marker}` | {count} |")

    lines.extend(
        [
            "",
            "## Domain Distribution",
            "",
            "| Domain | Count |",
            "|---|---:|",
        ]
    )
    for domain, count in sorted(domain_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{domain}` | {count} |")

    lines.extend(
        [
            "",
            "## Auto-generated Issue Seeds",
            "",
            "_Each section below is ready to open as a GitHub issue with owner hints and closure criteria._",
            "",
        ]
    )

    for idx, issue in enumerate(issues, start=1):
        marker_str = ", ".join(f"`{marker}`" for marker in issue.markers)
        line_str = ", ".join(str(line) for line in issue.lines)
        actions = "\n".join(f"- {item}" for item in issue.proposed_actions)
        acceptance = "\n".join(f"- [ ] {item}" for item in _acceptance_items(issue.markers))
        labels = ", ".join(
            [
                "`hardening`",
                "`underdeveloped`",
                f"`{issue.priority.lower()}`",
                f"`{issue.domain}`",
            ]
        )

        lines.extend(
            [
                f"### {idx}. [{issue.priority}] Harden `{issue.file_path}`",
                "",
                f"- **Labels**: {labels}",
                f"- **Owner Hint**: {issue.owner}",
                f"- **Priority Score**: `{issue.score}`",
                f"- **Markers**: {marker_str}",
                f"- **Trigger Lines**: `{line_str}`",
                "",
                "**Proposed Actions**",
                actions,
                "",
                "**Acceptance Checklist**",
                acceptance,
                "",
            ]
        )

    return "\n".join(lines)


def render_json(issues: list[SourceIssue]) -> str:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/generate_source_p0p1_issue_backlog.py",
        "issue_count": len(issues),
        "issues": [
            {
                "title": f"[{issue.priority}] Harden {issue.file_path}",
                "file_path": issue.file_path,
                "domain": issue.domain,
                "owner_hint": issue.owner,
                "priority": issue.priority,
                "score": issue.score,
                "markers": list(issue.markers),
                "trigger_lines": list(issue.lines),
                "labels": [
                    "hardening",
                    "underdeveloped",
                    issue.priority.lower(),
                    issue.domain,
                ],
                "proposed_actions": list(issue.proposed_actions),
                "acceptance_checklist": _acceptance_items(issue.markers),
            }
            for issue in issues
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _resolve_output(path_value: str) -> Path:
    output = Path(path_value)
    if not output.is_absolute():
        output = REPO_ROOT / output
    return output


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _normalize_for_check(content: str) -> str:
    # Generated timestamps drift every run; compare everything else.
    lines = []
    for line in content.splitlines():
        if line.startswith("- Generated at: `"):
            lines.append("- Generated at: `<dynamic>`")
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT_MD),
        help="Markdown output path.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="JSON output path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: fail if outputs differ from generated content.",
    )
    args = parser.parse_args(argv)

    issues = collect_source_issues(REPO_ROOT)
    md_content = render_markdown(issues)
    json_content = render_json(issues)

    md_path = _resolve_output(args.output_md)
    json_path = _resolve_output(args.output_json)

    if args.check:
        if not md_path.exists() or not json_path.exists():
            print(
                f"Backlog outputs missing. Run without --check to generate:\n"
                f"- {_display_path(md_path)}\n"
                f"- {_display_path(json_path)}",
            )
            return 1
        current_md = md_path.read_text(encoding="utf-8")
        current_json = json_path.read_text(encoding="utf-8")
        if _normalize_for_check(current_md) != _normalize_for_check(md_content):
            print(f"Markdown backlog drift detected: {_display_path(md_path)}")
            return 1
        current_payload = json.loads(current_json)
        new_payload = json.loads(json_content)
        for payload in (current_payload, new_payload):
            generated_at = payload.get("generated_at")
            if generated_at is not None and not isinstance(generated_at, str):
                raise ValueError("generated_at must be a string when present.")
        current_payload["generated_at"] = "<dynamic>"
        new_payload["generated_at"] = "<dynamic>"
        if current_payload != new_payload:
            print(f"JSON backlog drift detected: {_display_path(json_path)}")
            return 1
        print(f"Source P0/P1 backlog is up to date ({len(issues)} issue seeds).")
        return 0

    _write(md_path, md_content)
    _write(json_path, json_content)
    print(
        "Wrote source P0/P1 issue backlog:\n"
        f"- {_display_path(md_path)}\n"
        f"- {_display_path(json_path)}\n"
        f"issue seeds: {len(issues)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
