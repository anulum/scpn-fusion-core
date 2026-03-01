#!/usr/bin/env python
"""Generate split underdeveloped reports for source and docs-claims scopes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
UNDERDEV_MODULE_PATH = REPO_ROOT / "tools" / "generate_underdeveloped_register.py"
DEFAULT_SOURCE_MD = REPO_ROOT / "docs" / "UNDERDEVELOPED_SOURCE_REGISTER.md"
DEFAULT_DOCS_MD = REPO_ROOT / "docs" / "UNDERDEVELOPED_DOCS_CLAIMS_REGISTER.md"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "docs" / "UNDERDEVELOPED_SCOPE_SUMMARY.json"


@dataclass(frozen=True)
class ScopeSnapshot:
    scope: str
    total_entries: int
    p0_p1_entries: int
    marker_counts: dict[str, int]


def _load_underdev_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_underdeveloped_register", UNDERDEV_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tools/generate_underdeveloped_register.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_for_check(content: str) -> str:
    lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("- Generated at: `"):
            lines.append("- Generated at: `<dynamic>`")
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _collect_snapshot(entries: list[Any], *, scope: str, module: Any) -> ScopeSnapshot:
    marker_counts: dict[str, int] = {}
    p0p1 = 0
    for entry in entries:
        marker = str(entry.marker)
        marker_counts[marker] = marker_counts.get(marker, 0) + 1
        if module._priority(int(entry.score)) in {"P0", "P1"}:  # noqa: SLF001
            p0p1 += 1
    return ScopeSnapshot(
        scope=scope,
        total_entries=len(entries),
        p0_p1_entries=p0p1,
        marker_counts=marker_counts,
    )


def _build_scope_reports(*, top_limit: int, full_limit: int) -> tuple[str, str, str]:
    module = _load_underdev_module()
    entries = module.collect_entries(REPO_ROOT)
    source_entries = module._filter_entries_by_scope(entries, scope="source")  # noqa: SLF001
    docs_entries = module._filter_entries_by_scope(entries, scope="docs_claims")  # noqa: SLF001

    source_md = module.render_markdown(
        entries=source_entries,
        top_limit=top_limit,
        full_limit=full_limit,
        scope="source",
    )
    docs_md = module.render_markdown(
        entries=docs_entries,
        top_limit=top_limit,
        full_limit=full_limit,
        scope="docs_claims",
    )

    source_snapshot = _collect_snapshot(source_entries, scope="source", module=module)
    docs_snapshot = _collect_snapshot(docs_entries, scope="docs_claims", module=module)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/generate_underdeveloped_scope_reports.py",
        "snapshots": [asdict(source_snapshot), asdict(docs_snapshot)],
    }
    summary_json = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    return source_md, docs_md, summary_json


def _normalize_json_for_check(payload_text: str) -> dict[str, Any]:
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("summary payload must be a JSON object")
    generated_at = payload.get("generated_at")
    if generated_at is not None and not isinstance(generated_at, str):
        raise ValueError("generated_at must be a string when present")
    payload["generated_at"] = "<dynamic>"
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", default=str(DEFAULT_SOURCE_MD))
    parser.add_argument("--docs-output", default=str(DEFAULT_DOCS_MD))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--top-limit", type=int, default=80)
    parser.add_argument("--full-limit", type=int, default=250)
    parser.add_argument("--check", action="store_true", help="Fail if outputs differ from generated content.")
    args = parser.parse_args(argv)

    if int(args.top_limit) < 1:
        raise ValueError("--top-limit must be >= 1")
    if int(args.full_limit) < 1:
        raise ValueError("--full-limit must be >= 1")

    source_out = _resolve(args.source_output)
    docs_out = _resolve(args.docs_output)
    summary_out = _resolve(args.summary_json)
    source_md, docs_md, summary_json = _build_scope_reports(
        top_limit=int(args.top_limit),
        full_limit=int(args.full_limit),
    )

    if args.check:
        missing = [p for p in (source_out, docs_out, summary_out) if not p.exists()]
        if missing:
            print("Underdeveloped scope outputs missing:")
            for path in missing:
                print(f"- {_display_path(path)}")
            return 1
        if _normalize_for_check(source_out.read_text(encoding="utf-8")) != _normalize_for_check(source_md):
            print(f"Scope report drift detected: {_display_path(source_out)}")
            return 1
        if _normalize_for_check(docs_out.read_text(encoding="utf-8")) != _normalize_for_check(docs_md):
            print(f"Scope report drift detected: {_display_path(docs_out)}")
            return 1
        current_json = _normalize_json_for_check(summary_out.read_text(encoding="utf-8"))
        new_json = _normalize_json_for_check(summary_json)
        if current_json != new_json:
            print(f"Scope summary drift detected: {_display_path(summary_out)}")
            return 1
        print("Underdeveloped scope reports are up to date.")
        return 0

    for path, content in (
        (source_out, source_md),
        (docs_out, docs_md),
        (summary_out, summary_json),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(
        "Wrote underdeveloped scope reports:\n"
        f"- {_display_path(source_out)}\n"
        f"- {_display_path(docs_out)}\n"
        f"- {_display_path(summary_out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
