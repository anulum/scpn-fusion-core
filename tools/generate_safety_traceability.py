#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate docs/SAFETY_TRACEABILITY_MATRIX.md from validation/safety_traceability.json.

Fail-closed traceability for public control-safety claims: every referenced
hazard, public-claim pattern, implementation symbol, test, Lean theorem, and
evidence artifact must exist in the repository at generation time, so the
``--check`` mode also breaks when linked code or proofs are removed.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "validation" / "safety_traceability.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "SAFETY_TRACEABILITY_MATRIX.md"
MANIFEST_SCHEMA = "scpn-fusion-core.safety-traceability.v1"

_SYMBOL_PATTERNS: dict[str, str] = {
    ".py": r"(?m)^\s*(?:def|class)\s+{symbol}\b",
    ".rs": r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|mod|trait)\s+{symbol}\b",
    ".lean": r"(?m)^\s*theorem\s+{symbol}\b",
}


@dataclass(frozen=True)
class Anchor:
    """Repository anchor: a file, optionally narrowed to a named symbol."""

    file: str
    symbol: str | None

    @property
    def label(self) -> str:
        """Return the ``file::symbol`` (or bare file) display form."""
        return self.file if self.symbol is None else f"{self.file}::{self.symbol}"


@dataclass(frozen=True)
class Hazard:
    """Declared hazard mitigated by one or more safety requirements."""

    hazard_id: str
    description: str


@dataclass(frozen=True)
class PublicClaim:
    """Public control-safety claim location: source file plus regex pattern."""

    file: str
    pattern: str


@dataclass(frozen=True)
class SafetyRequirement:
    """One traceability row: requirement linked to hazards, code, tests, proofs."""

    requirement_id: str
    statement: str
    hazard_ids: tuple[str, ...]
    public_claim: PublicClaim
    iec61508_alignment: str
    implementation: tuple[Anchor, ...]
    tests: tuple[Anchor, ...]
    proofs: tuple[Anchor, ...]
    evidence: tuple[Anchor, ...]
    gaps: tuple[str, ...]


@dataclass(frozen=True)
class TraceabilityManifest:
    """Validated manifest content: language boundary, hazards, requirements."""

    language_boundary: str
    hazards: tuple[Hazard, ...]
    requirements: tuple[SafetyRequirement, ...]


def _require_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _parse_anchor(name: str, value: Any) -> Anchor:
    text = _require_str(name, value)
    if "::" in text:
        file_part, _, symbol = text.partition("::")
        return Anchor(file=_require_str(name, file_part), symbol=_require_str(name, symbol))
    return Anchor(file=text, symbol=None)


def _parse_anchor_list(name: str, value: Any, *, minimum: int) -> tuple[Anchor, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    if len(value) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} entries.")
    return tuple(_parse_anchor(f"{name}[{i}]", item) for i, item in enumerate(value))


def _parse_str_list(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    return tuple(_require_str(f"{name}[{i}]", item) for i, item in enumerate(value))


def _parse_hazards(value: Any) -> tuple[Hazard, ...]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError("hazards must be a non-empty list.")
    out: list[Hazard] = []
    seen: set[str] = set()
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"hazards[{i}] must be an object.")
        hazard_id = _require_str(f"hazards[{i}].id", item.get("id"))
        if hazard_id in seen:
            raise ValueError(f"Duplicate hazard id: {hazard_id}")
        seen.add(hazard_id)
        out.append(
            Hazard(
                hazard_id=hazard_id,
                description=_require_str(f"hazards[{i}].description", item.get("description")),
            )
        )
    return tuple(out)


def _parse_requirement(index: int, item: Any) -> SafetyRequirement:
    if not isinstance(item, dict):
        raise ValueError(f"requirements[{index}] must be an object.")
    name = f"requirements[{index}]"
    claim_raw = item.get("public_claim")
    if not isinstance(claim_raw, dict):
        raise ValueError(f"{name}.public_claim must be an object.")
    return SafetyRequirement(
        requirement_id=_require_str(f"{name}.id", item.get("id")),
        statement=_require_str(f"{name}.statement", item.get("statement")),
        hazard_ids=_parse_str_list(f"{name}.hazard_ids", item.get("hazard_ids")),
        public_claim=PublicClaim(
            file=_require_str(f"{name}.public_claim.file", claim_raw.get("file")),
            pattern=_require_str(f"{name}.public_claim.pattern", claim_raw.get("pattern")),
        ),
        iec61508_alignment=_require_str(
            f"{name}.iec61508_alignment", item.get("iec61508_alignment")
        ),
        implementation=_parse_anchor_list(
            f"{name}.implementation", item.get("implementation"), minimum=1
        ),
        tests=_parse_anchor_list(f"{name}.tests", item.get("tests"), minimum=1),
        proofs=_parse_anchor_list(f"{name}.proofs", item.get("proofs", []), minimum=0),
        evidence=_parse_anchor_list(f"{name}.evidence", item.get("evidence", []), minimum=0),
        gaps=_parse_str_list(f"{name}.gaps", item.get("gaps", [])),
    )


def load_manifest(path: Path) -> TraceabilityManifest:
    """Load and validate the safety traceability manifest.

    Args:
        path: Filesystem path to ``safety_traceability.json``.

    Returns:
        Parsed manifest with unique hazard and requirement identifiers.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Safety traceability manifest must be a JSON object.")
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unexpected manifest schema: {raw.get('schema')!r}")
    hazards = _parse_hazards(raw.get("hazards"))
    requirements_raw = raw.get("requirements")
    if not isinstance(requirements_raw, list) or len(requirements_raw) == 0:
        raise ValueError("requirements must be a non-empty list.")
    requirements: list[SafetyRequirement] = []
    seen: set[str] = set()
    for i, item in enumerate(requirements_raw):
        requirement = _parse_requirement(i, item)
        if requirement.requirement_id in seen:
            raise ValueError(f"Duplicate requirement id: {requirement.requirement_id}")
        seen.add(requirement.requirement_id)
        requirements.append(requirement)
    return TraceabilityManifest(
        language_boundary=_require_str("language_boundary", raw.get("language_boundary")),
        hazards=hazards,
        requirements=tuple(requirements),
    )


def _verify_anchor(root: Path, anchor: Anchor, context: str) -> list[str]:
    errors: list[str] = []
    target = root / anchor.file
    if not target.is_file():
        return [f"{context}: file not found: {anchor.file}"]
    if anchor.symbol is None:
        return errors
    template = _SYMBOL_PATTERNS.get(target.suffix)
    if template is None:
        return [f"{context}: symbol anchors unsupported for {target.suffix!r}: {anchor.label}"]
    pattern = template.format(symbol=re.escape(anchor.symbol))
    if re.search(pattern, target.read_text(encoding="utf-8")) is None:
        errors.append(f"{context}: symbol not found: {anchor.label}")
    return errors


def verify_manifest(manifest: TraceabilityManifest, root: Path) -> list[str]:
    """Verify every anchor in the manifest against the repository tree.

    Args:
        manifest: Parsed traceability manifest.
        root: Repository root against which anchors are resolved.

    Returns:
        List of human-readable errors; empty when fully traceable.
    """
    errors: list[str] = []
    known_hazards = {hazard.hazard_id for hazard in manifest.hazards}
    referenced_hazards: set[str] = set()
    for requirement in manifest.requirements:
        rid = requirement.requirement_id
        if len(requirement.hazard_ids) == 0:
            errors.append(f"{rid}: must reference at least one hazard.")
        for hazard_id in requirement.hazard_ids:
            if hazard_id not in known_hazards:
                errors.append(f"{rid}: unknown hazard id: {hazard_id}")
            referenced_hazards.add(hazard_id)
        claim_path = root / requirement.public_claim.file
        if not claim_path.is_file():
            errors.append(f"{rid}: public claim file not found: {requirement.public_claim.file}")
        elif (
            re.search(requirement.public_claim.pattern, claim_path.read_text(encoding="utf-8"))
            is None
        ):
            errors.append(
                f"{rid}: public claim pattern not found in "
                f"{requirement.public_claim.file}: {requirement.public_claim.pattern!r}"
            )
        for kind, anchors in (
            ("implementation", requirement.implementation),
            ("test", requirement.tests),
            ("proof", requirement.proofs),
            ("evidence", requirement.evidence),
        ):
            for anchor in anchors:
                errors.extend(_verify_anchor(root, anchor, f"{rid} {kind}"))
    for hazard in manifest.hazards:
        if hazard.hazard_id not in referenced_hazards:
            errors.append(f"hazard {hazard.hazard_id} is not referenced by any requirement.")
    return errors


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _anchor_lines(title: str, anchors: tuple[Anchor, ...]) -> list[str]:
    lines = ["", f"{title}:", ""]
    if anchors:
        lines.extend(f"- `{anchor.label}`" for anchor in anchors)
    else:
        lines.append("- (none)")
    return lines


def render_markdown(manifest: TraceabilityManifest, manifest_rel: str) -> str:
    """Render the safety traceability matrix as Markdown.

    Args:
        manifest: Validated traceability manifest.
        manifest_rel: Manifest path string shown in the generated header.

    Returns:
        Rendered Markdown document as a single string.
    """
    lines: list[str] = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Generated by tools/generate_safety_traceability.py; do not edit by hand. -->",
        "",
        "# Safety Traceability Matrix",
        "",
        "Auto-generated from the safety traceability manifest. Do not edit by hand;",
        "run `tools/generate_safety_traceability.py` to refresh.",
        "",
        f"> **Language boundary.** {manifest.language_boundary}",
        "",
        f"- Manifest: `{manifest_rel}`",
        f"- Requirements tracked: `{len(manifest.requirements)}`",
        f"- Hazards tracked: `{len(manifest.hazards)}`",
        "",
        "Every implementation symbol, test, Lean theorem, evidence artifact, and",
        "public-claim pattern below is verified to exist at generation time; the",
        "CI drift check fails when any linked entity disappears.",
        "",
        "## Hazard log",
        "",
        "| Hazard ID | Description |",
        "|---|---|",
    ]
    for hazard in manifest.hazards:
        lines.append(f"| `{hazard.hazard_id}` | {_escape_cell(hazard.description)} |")

    lines.extend(
        [
            "",
            "## Requirements summary",
            "",
            "| Requirement | Hazards | Impl | Tests | Proofs | Evidence | Gaps |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for req in manifest.requirements:
        hazard_cell = ", ".join(f"`{hid}`" for hid in req.hazard_ids)
        lines.append(
            f"| `{req.requirement_id}` | {hazard_cell} | {len(req.implementation)} | "
            f"{len(req.tests)} | {len(req.proofs)} | {len(req.evidence)} | {len(req.gaps)} |"
        )

    lines.extend(["", "## Requirement details", ""])
    for req in manifest.requirements:
        lines.extend(
            [
                f"### `{req.requirement_id}`",
                "",
                f"{req.statement}",
                "",
                f"- Hazards: {', '.join(f'`{hid}`' for hid in req.hazard_ids)}",
                f"- Public claim: `{req.public_claim.file}` — pattern "
                f"`{_escape_cell(req.public_claim.pattern)}`",
                f"- IEC 61508 alignment: {_escape_cell(req.iec61508_alignment)}",
            ]
        )
        lines.extend(_anchor_lines("Implementation", req.implementation))
        lines.extend(_anchor_lines("Tests", req.tests))
        lines.extend(_anchor_lines("Formal proofs", req.proofs))
        lines.extend(_anchor_lines("Evidence artifacts", req.evidence))
        lines.extend(["", "Declared gaps:", ""])
        if req.gaps:
            lines.extend(f"- {gap}" for gap in req.gaps)
        else:
            lines.append("- (none declared)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Render or drift-check the safety traceability matrix.

    Verification runs in both modes, so a removed test, theorem, or symbol
    fails the check even when the rendered Markdown itself is unchanged.

    Args:
        argv: Optional CLI args. If omitted, reads from process arguments.

    Returns:
        0 on success, 1 when verification fails or the output is stale.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to safety traceability manifest JSON.",
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

    manifest = load_manifest(manifest_path)
    errors = verify_manifest(manifest, REPO_ROOT)
    if errors:
        for error in errors:
            print(f"TRACEABILITY ERROR: {error}")
        return 1

    try:
        manifest_rel = manifest_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        manifest_rel = manifest_path.as_posix()
    rendered = render_markdown(manifest, manifest_rel)

    if args.check:
        if not output_path.exists():
            print(f"Safety traceability matrix missing: {output_path}")
            return 1
        if output_path.read_text(encoding="utf-8") != rendered:
            print(
                "Safety traceability matrix is stale. "
                "Run tools/generate_safety_traceability.py to refresh."
            )
            return 1
        print(f"Safety traceability matrix is up to date: {output_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote safety traceability matrix: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
