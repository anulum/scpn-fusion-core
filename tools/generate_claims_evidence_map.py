#!/usr/bin/env python
"""Generate docs/CLAIMS_EVIDENCE_MAP.md from validation/claims_manifest.json."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "validation" / "claims_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "CLAIMS_EVIDENCE_MAP.md"


@dataclass(frozen=True)
class EvidencePattern:
    file: str
    pattern: str


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    source_file: str
    source_pattern: str
    evidence_files: tuple[str, ...]
    evidence_patterns: tuple[EvidencePattern, ...]


def _require_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _require_str_list(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings.")
    out: list[str] = []
    for i, item in enumerate(value):
        out.append(_require_str(f"{name}[{i}]", item))
    return tuple(out)


def _parse_evidence_patterns(value: Any) -> tuple[EvidencePattern, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("evidence_patterns must be a list.")
    out: list[EvidencePattern] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"evidence_patterns[{i}] must be an object.")
        out.append(
            EvidencePattern(
                file=_require_str(f"evidence_patterns[{i}].file", item.get("file")),
                pattern=_require_str(
                    f"evidence_patterns[{i}].pattern", item.get("pattern")
                ),
            )
        )
    return tuple(out)


def load_manifest(path: Path) -> tuple[ClaimSpec, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Claims manifest must be a JSON object.")
    claims_raw = raw.get("claims")
    if not isinstance(claims_raw, list) or len(claims_raw) == 0:
        raise ValueError("Claims manifest must contain non-empty 'claims' list.")

    out: list[ClaimSpec] = []
    seen_ids: set[str] = set()
    for i, claim in enumerate(claims_raw):
        if not isinstance(claim, dict):
            raise ValueError(f"claims[{i}] must be an object.")
        claim_id = _require_str(f"claims[{i}].id", claim.get("id"))
        if claim_id in seen_ids:
            raise ValueError(f"Duplicate claim id: {claim_id}")
        seen_ids.add(claim_id)
        out.append(
            ClaimSpec(
                claim_id=claim_id,
                source_file=_require_str(
                    f"claims[{i}].source_file", claim.get("source_file")
                ),
                source_pattern=_require_str(
                    f"claims[{i}].source_pattern", claim.get("source_pattern")
                ),
                evidence_files=_require_str_list(
                    f"claims[{i}].evidence_files",
                    claim.get("evidence_files", []),
                ),
                evidence_patterns=_parse_evidence_patterns(
                    claim.get("evidence_patterns", [])
                ),
            )
        )
    return tuple(out)


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("`", "\\`")


def render_markdown(claims: tuple[ClaimSpec, ...], manifest_path: str) -> str:
    lines: list[str] = [
        "# Claims Evidence Map",
        "",
        "Auto-generated map from `validation/claims_manifest.json` linking headline claims",
        "to concrete evidence files and patterns.",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Claims tracked: `{len(claims)}`",
        "",
        "## Summary",
        "",
        "| Claim ID | Source | Evidence Files | Pattern Checks |",
        "|---|---|---:|---:|",
    ]
    for claim in claims:
        lines.append(
            "| "
            f"`{_escape_cell(claim.claim_id)}` | "
            f"`{_escape_cell(claim.source_file)}` | "
            f"{len(claim.evidence_files)} | "
            f"{len(claim.evidence_patterns)} |"
        )

    lines.extend(["", "## Claim Details", ""])
    for claim in claims:
        lines.extend(
            [
                f"### `{claim.claim_id}`",
                "",
                f"- Source file: `{claim.source_file}`",
                f"- Source pattern: `{claim.source_pattern}`",
                "",
                "Evidence files:",
            ]
        )
        if claim.evidence_files:
            for evidence_file in claim.evidence_files:
                lines.append(f"- `{evidence_file}`")
        else:
            lines.append("- (none)")

        lines.extend(["", "Evidence pattern checks:"])
        if claim.evidence_patterns:
            lines.extend(
                [
                    "",
                    "| File | Pattern |",
                    "|---|---|",
                ]
            )
            for pattern in claim.evidence_patterns:
                lines.append(
                    f"| `{_escape_cell(pattern.file)}` | "
                    f"`{_escape_cell(pattern.pattern)}` |"
                )
        else:
            lines.append("- (none)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to claims manifest JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Markdown output path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if output is stale instead of writing it.",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    claims = load_manifest(manifest_path)
    try:
        rel_manifest = manifest_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_manifest = manifest_path.as_posix()
    rendered = render_markdown(claims, manifest_path=rel_manifest)

    if args.check:
        if not output_path.exists():
            print(f"Claims evidence map missing: {output_path}")
            return 1
        existing = output_path.read_text(encoding="utf-8")
        if existing != rendered:
            print(
                "Claims evidence map is stale. "
                "Run tools/generate_claims_evidence_map.py to refresh."
            )
            return 1
        print(f"Claims evidence map is up to date: {output_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote claims evidence map: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
