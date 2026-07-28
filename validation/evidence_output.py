# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — explicit benchmark evidence output policy
"""Keep routine benchmark runs from overwriting tracked evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvidenceOutputPaths:
    """Resolved JSON and Markdown destinations for one benchmark run."""

    json: Path
    markdown: Path


def add_evidence_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared explicit-evidence output switches to ``parser``."""
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="JSON output path (default: gitignored artifacts/_local_*.json).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Markdown output path (default: gitignored artifacts/_local_*.md).",
    )
    parser.add_argument(
        "--commit-evidence",
        action="store_true",
        help="Write canonical tracked evidence defaults instead of local outputs.",
    )


def _absolute(path: Path, *, root: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    return candidate.resolve(strict=False)


def _is_protected_evidence_path(path: Path, *, root: Path) -> bool:
    try:
        relative = path.relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    parts = relative.parts
    if len(parts) >= 2 and parts[:2] == ("validation", "reports"):
        return True
    if parts and parts[0] == "artifacts":
        is_ignored_local_file = (
            len(parts) == 2 and path.name.startswith("_local_") and path.suffix in {".json", ".md"}
        )
        return not is_ignored_local_file
    return False


def resolve_evidence_outputs(
    *,
    root: Path,
    canonical_json: Path,
    canonical_markdown: Path,
    requested_json: Path | None,
    requested_markdown: Path | None,
    commit_evidence: bool,
) -> EvidenceOutputPaths:
    """Resolve benchmark outputs and reject implicit writes to evidence roots.

    Explicit caller paths outside ``artifacts/`` and ``validation/reports/``
    remain supported for tests and one-off exports. Inside those repository
    evidence roots, routine runs may write only direct ``_local_*`` files.
    ``--commit-evidence`` is therefore required for every canonical or other
    non-local evidence destination.
    """
    root = root.resolve(strict=False)
    canonical_json = _absolute(canonical_json, root=root)
    canonical_markdown = _absolute(canonical_markdown, root=root)
    local_json = root / "artifacts" / f"_local_{canonical_json.name}"
    local_markdown = root / "artifacts" / f"_local_{canonical_markdown.name}"

    default_json = canonical_json if commit_evidence else local_json
    default_markdown = canonical_markdown if commit_evidence else local_markdown

    output_json = _absolute(
        requested_json if requested_json is not None else default_json,
        root=root,
    )
    output_markdown = _absolute(
        requested_markdown if requested_markdown is not None else default_markdown,
        root=root,
    )

    if not commit_evidence:
        protected = [
            path
            for path in (output_json, output_markdown)
            if _is_protected_evidence_path(path, root=root)
        ]
        if protected:
            labels = ", ".join(path.relative_to(root).as_posix() for path in protected)
            raise ValueError(
                f"refusing tracked evidence output without --commit-evidence: {labels}"
            )
    return EvidenceOutputPaths(json=output_json, markdown=output_markdown)
