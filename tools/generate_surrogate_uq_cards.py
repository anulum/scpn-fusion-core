#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate docs/SURROGATE_UQ_CARDS.md from validation/surrogate_uq_cards.json.

Per-surrogate UQ cards (SOTA-6 / master-plan T-4): training provenance,
calibration evidence, OOD mechanism and thresholds, fallback behaviour, and
retraining provenance for every surrogate lane on a public surface. Every
anchor (``path`` or ``path::symbol``) is verified against the repository
tree, so ``--check`` fails when a linked model, guard, tool, or evidence
artifact disappears — the guard keeps the cards in sync with the code.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "validation" / "surrogate_uq_cards.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "SURROGATE_UQ_CARDS.md"
MANIFEST_SCHEMA = "scpn-fusion-core.surrogate-uq-cards.v1"

VALID_STATUSES = ("promoted", "deprecated_scoped", "fail_closed_stub", "retrain_blocked")

_PY_SYMBOL_PATTERN = r"(?m)^(?:\s*(?:def|class)\s+{symbol}\b|{symbol}\s*=)"


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
class EvidenceSection:
    """One card section: prose summary plus verified anchors."""

    summary: str
    anchors: tuple[Anchor, ...]


@dataclass(frozen=True)
class UqCard:
    """Per-surrogate UQ card with provenance, calibration, OOD, and fallback."""

    card_id: str
    description: str
    status: str
    model_artifacts: tuple[Anchor, ...]
    training_provenance: EvidenceSection
    calibration: EvidenceSection
    ood_mechanism: str
    ood_thresholds: dict[str, float]
    ood_anchors: tuple[Anchor, ...]
    fallback: EvidenceSection
    retraining: EvidenceSection
    gaps: tuple[str, ...]


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


def _parse_anchor_list(name: str, value: Any) -> tuple[Anchor, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    return tuple(_parse_anchor(f"{name}[{i}]", item) for i, item in enumerate(value))


def _parse_section(name: str, value: Any) -> EvidenceSection:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return EvidenceSection(
        summary=_require_str(f"{name}.summary", value.get("summary")),
        anchors=_parse_anchor_list(f"{name}.anchors", value.get("anchors", [])),
    )


def _parse_thresholds(name: str, value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    out: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{name}.{key} must be a number.")
        out[str(key)] = float(item)
    return out


def _parse_card(index: int, item: Any) -> UqCard:
    if not isinstance(item, dict):
        raise ValueError(f"cards[{index}] must be an object.")
    name = f"cards[{index}]"
    status = _require_str(f"{name}.status", item.get("status"))
    if status not in VALID_STATUSES:
        raise ValueError(f"{name}.status must be one of {VALID_STATUSES}, got {status!r}")
    ood_raw = item.get("ood")
    if not isinstance(ood_raw, dict):
        raise ValueError(f"{name}.ood must be an object.")
    model_artifacts = _parse_anchor_list(f"{name}.model_artifacts", item.get("model_artifacts"))
    if len(model_artifacts) == 0:
        raise ValueError(f"{name}.model_artifacts must not be empty.")
    gaps_raw = item.get("gaps", [])
    if not isinstance(gaps_raw, list):
        raise ValueError(f"{name}.gaps must be a list.")
    return UqCard(
        card_id=_require_str(f"{name}.id", item.get("id")),
        description=_require_str(f"{name}.description", item.get("description")),
        status=status,
        model_artifacts=model_artifacts,
        training_provenance=_parse_section(
            f"{name}.training_provenance", item.get("training_provenance")
        ),
        calibration=_parse_section(f"{name}.calibration", item.get("calibration")),
        ood_mechanism=_require_str(f"{name}.ood.mechanism", ood_raw.get("mechanism")),
        ood_thresholds=_parse_thresholds(f"{name}.ood.thresholds", ood_raw.get("thresholds", {})),
        ood_anchors=_parse_anchor_list(f"{name}.ood.anchors", ood_raw.get("anchors", [])),
        fallback=_parse_section(f"{name}.fallback", item.get("fallback")),
        retraining=_parse_section(f"{name}.retraining", item.get("retraining")),
        gaps=tuple(_require_str(f"{name}.gaps[{i}]", gap) for i, gap in enumerate(gaps_raw)),
    )


def load_manifest(path: Path) -> tuple[str, tuple[UqCard, ...]]:
    """Load and validate the surrogate UQ-card manifest.

    Args:
        path: Filesystem path to ``surrogate_uq_cards.json``.

    Returns:
        Tuple of the scope note and the parsed cards.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Surrogate UQ-card manifest must be a JSON object.")
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unexpected manifest schema: {raw.get('schema')!r}")
    cards_raw = raw.get("cards")
    if not isinstance(cards_raw, list) or len(cards_raw) == 0:
        raise ValueError("cards must be a non-empty list.")
    cards: list[UqCard] = []
    seen: set[str] = set()
    for i, item in enumerate(cards_raw):
        card = _parse_card(i, item)
        if card.card_id in seen:
            raise ValueError(f"Duplicate card id: {card.card_id}")
        seen.add(card.card_id)
        cards.append(card)
    return _require_str("scope_note", raw.get("scope_note")), tuple(cards)


def _verify_anchor(root: Path, anchor: Anchor, context: str) -> list[str]:
    target = root / anchor.file
    if not target.is_file():
        return [f"{context}: file not found: {anchor.file}"]
    if anchor.symbol is None:
        return []
    if target.suffix != ".py":
        return [f"{context}: symbol anchors unsupported for {target.suffix!r}: {anchor.label}"]
    pattern = _PY_SYMBOL_PATTERN.format(symbol=re.escape(anchor.symbol))
    if re.search(pattern, target.read_text(encoding="utf-8")) is None:
        return [f"{context}: symbol not found: {anchor.label}"]
    return []


def verify_cards(cards: tuple[UqCard, ...], root: Path) -> list[str]:
    """Verify every card anchor against the repository tree.

    Args:
        cards: Parsed UQ cards.
        root: Repository root against which anchors are resolved.

    Returns:
        List of human-readable errors; empty when fully anchored.
    """
    errors: list[str] = []
    for card in cards:
        sections: tuple[tuple[str, tuple[Anchor, ...]], ...] = (
            ("model_artifacts", card.model_artifacts),
            ("training_provenance", card.training_provenance.anchors),
            ("calibration", card.calibration.anchors),
            ("ood", card.ood_anchors),
            ("fallback", card.fallback.anchors),
            ("retraining", card.retraining.anchors),
        )
        for section_name, anchors in sections:
            for anchor in anchors:
                errors.extend(_verify_anchor(root, anchor, f"{card.card_id} {section_name}"))
        if card.status != "promoted" and len(card.gaps) == 0:
            errors.append(f"{card.card_id}: non-promoted card must declare its gaps.")
    return errors


def _anchor_lines(title: str, anchors: tuple[Anchor, ...]) -> list[str]:
    lines = ["", f"{title}:", ""]
    if anchors:
        lines.extend(f"- `{anchor.label}`" for anchor in anchors)
    else:
        lines.append("- (none)")
    return lines


def render_markdown(scope_note: str, cards: tuple[UqCard, ...], manifest_rel: str) -> str:
    """Render the surrogate UQ cards as Markdown.

    Args:
        scope_note: Manifest scope note shown in the header.
        cards: Validated UQ cards.
        manifest_rel: Manifest path string shown in the generated header.

    Returns:
        Rendered Markdown document as a single string.
    """
    lines: list[str] = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Generated by tools/generate_surrogate_uq_cards.py; do not edit by hand. -->",
        "",
        "# Surrogate UQ Cards",
        "",
        "Auto-generated from the surrogate UQ-card manifest. Do not edit by hand;",
        "run `tools/generate_surrogate_uq_cards.py` to refresh.",
        "",
        f"> {scope_note}",
        "",
        f"- Manifest: `{manifest_rel}`",
        f"- Cards: `{len(cards)}`",
        "",
        "## Status summary",
        "",
        "| Surrogate | Status | OOD mechanism | Gaps |",
        "|---|---|---|---:|",
    ]
    for card in cards:
        ood_short = card.ood_mechanism.split(";")[0].split(".")[0]
        lines.append(f"| `{card.card_id}` | `{card.status}` | {ood_short} | {len(card.gaps)} |")
    lines.extend(["", "## Cards", ""])
    for card in cards:
        lines.extend(
            [
                f"### `{card.card_id}`",
                "",
                card.description,
                "",
                f"- Status: `{card.status}`",
            ]
        )
        lines.extend(_anchor_lines("Model artifacts", card.model_artifacts))
        for title, section in (
            ("Training provenance", card.training_provenance),
            ("Calibration", card.calibration),
        ):
            lines.extend(["", f"{title}: {section.summary}"])
            lines.extend(_anchor_lines(f"{title} anchors", section.anchors))
        lines.extend(["", f"OOD mechanism: {card.ood_mechanism}"])
        if card.ood_thresholds:
            lines.extend(["", "OOD thresholds:", ""])
            lines.extend(
                f"- `{key}` = {value}" for key, value in sorted(card.ood_thresholds.items())
            )
        lines.extend(_anchor_lines("OOD anchors", card.ood_anchors))
        for title, section in (
            ("Fallback behaviour", card.fallback),
            ("Retraining", card.retraining),
        ):
            lines.extend(["", f"{title}: {section.summary}"])
            lines.extend(_anchor_lines(f"{title} anchors", section.anchors))
        lines.extend(["", "Declared gaps:", ""])
        if card.gaps:
            lines.extend(f"- {gap}" for gap in card.gaps)
        else:
            lines.append("- (none declared)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Render or drift-check the surrogate UQ cards.

    Verification runs in both modes, so a removed model file, guard, or
    evidence artifact fails the check even when the Markdown is unchanged.

    Args:
        argv: Optional CLI args. If omitted, reads from process arguments.

    Returns:
        0 on success, 1 when verification fails or the output is stale.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to surrogate UQ-card manifest JSON.",
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

    scope_note, cards = load_manifest(manifest_path)
    errors = verify_cards(cards, REPO_ROOT)
    if errors:
        for error in errors:
            print(f"UQ CARD ERROR: {error}")
        return 1

    try:
        manifest_rel = manifest_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        manifest_rel = manifest_path.as_posix()
    rendered = render_markdown(scope_note, cards, manifest_rel)

    if args.check:
        if not output_path.exists():
            print(f"Surrogate UQ cards missing: {output_path}")
            return 1
        if output_path.read_text(encoding="utf-8") != rendered:
            print(
                "Surrogate UQ cards are stale. Run tools/generate_surrogate_uq_cards.py to refresh."
            )
            return 1
        print(f"Surrogate UQ cards are up to date: {output_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote surrogate UQ cards: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
