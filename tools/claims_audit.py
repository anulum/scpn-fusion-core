#!/usr/bin/env python
"""Audit high-level project claims against explicit evidence artifacts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "validation" / "claims_manifest.json"


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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def run_audit(claims: tuple[ClaimSpec, ...], repo_root: Path) -> list[str]:
    errors: list[str] = []
    for claim in claims:
        source = repo_root / claim.source_file
        if not source.exists():
            errors.append(
                f"[{claim.claim_id}] source file missing: {claim.source_file}"
            )
            continue
        source_text = _read_text(source)
        if re.search(claim.source_pattern, source_text, flags=re.MULTILINE) is None:
            errors.append(
                f"[{claim.claim_id}] source pattern not found in {claim.source_file}: "
                f"{claim.source_pattern}"
            )
        for evidence_file in claim.evidence_files:
            evidence = repo_root / evidence_file
            if not evidence.exists():
                errors.append(
                    f"[{claim.claim_id}] evidence file missing: {evidence_file}"
                )

        for pattern_check in claim.evidence_patterns:
            evidence = repo_root / pattern_check.file
            if not evidence.exists():
                errors.append(
                    f"[{claim.claim_id}] evidence pattern file missing: "
                    f"{pattern_check.file}"
                )
                continue
            evidence_text = _read_text(evidence)
            if (
                re.search(pattern_check.pattern, evidence_text, flags=re.MULTILINE)
                is None
            ):
                errors.append(
                    f"[{claim.claim_id}] evidence pattern not found in "
                    f"{pattern_check.file}: {pattern_check.pattern}"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to claims manifest JSON (default: validation/claims_manifest.json).",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Claims manifest not found: {manifest_path}")

    claims = load_manifest(manifest_path)
    errors = run_audit(claims, REPO_ROOT)
    if errors:
        print(f"Claims audit FAILED ({len(errors)} issue(s))")
        for error in errors:
            print(f" - {error}")
        return 1

    print(
        f"Claims audit passed for {len(claims)} claims "
        f"using {manifest_path.relative_to(REPO_ROOT).as_posix()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
