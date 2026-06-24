# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Honest validated-coverage measurement (fleet WS-6)
"""Measure FUSION's honest validated-coverage from the claims-to-evidence ledger.

Fleet workstream WS-6 (coverage frontier): honesty without coverage is theatre, so each
studio reports its *honest-validated answer-rate* — the fraction of claims that are
reference-validated against real evidence, never a promotion translation laundered into
``validated`` (the CEO LOCK-4 ruling, 2026-06-24).

FUSION's ``validation/claims_manifest.json`` is a *claim-to-evidence linkage* manifest, not
a graded eight-state ClaimStatus ledger. This module therefore maps each claim conservatively
to a coverage band from the *kind of evidence it carries*, applying the honesty floor:
``reference-validated`` is granted ONLY when the claim's evidence includes a real-shot
validation artefact or an external-reference-code parity artefact — never on an internal
benchmark, a surrogate's own test metric, or a scope declaration. A scope-boundary claim is
counted as ``boundary`` (the honesty model working: it states what is *not* covered), never as
a gap or a validated result.

The number this produces is a *conservative floor*: the true graded coverage requires a real
ClaimStatus grading in the ledger, which is the flagged follow-on (see :data:`LEDGER_GAP`).
This module does not inflate; it under-counts before it over-counts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_MANIFEST = REPO_ROOT / "validation" / "claims_manifest.json"

#: The standing gap this measurement floors against — the honest follow-on.
LEDGER_GAP = (
    "FUSION's claims_manifest.json is claim-to-evidence linkage, not a graded ClaimStatus "
    "ledger; this coverage is a conservative floor classified from evidence kind. A rigorous "
    "validated-coverage requires real per-claim ClaimStatus grading in the ledger (follow-on)."
)

#: Evidence-file markers that count as reference-validation (real-shot or external-code parity).
#: Conservative: only experimental real-shot validation and external reference-code parity qualify.
_REFERENCE_VALIDATION_MARKERS = (
    "real_shot_validation",  # validation against real experimental discharges
    "freegs_benchmark",  # parity against the external FreeGS reference solver
)

#: Evidence-file markers that are internal metrics/benchmarks (strong but not reference-validated).
_INTERNAL_METRIC_MARKERS = (
    ".metrics.json",  # a model's own held-out test metric (e.g. a surrogate)
    "stress_test_campaign",  # internal simulation campaign
    "threshold_sweep",  # internal sweep over a small evaluation set
    "scpn_end_to_end_latency",  # internal latency benchmark
)


class CoverageBand(StrEnum):
    """A conservative coverage band mapped from a claim's evidence kind.

    The bands floor onto the fleet ClaimStatus lattice without laundering: a promotion or an
    internal benchmark never becomes ``reference_validated``.
    """

    REFERENCE_VALIDATED = "reference-validated"
    BOUNDED_MODEL = "bounded-model"
    BOUNDARY = "boundary"
    PRODUCER_ASSERTED = "producer-asserted"
    VALIDATION_GAP = "validation-gap"


@dataclass(frozen=True)
class ClaimCoverage:
    """One claim's id and its conservatively-assigned coverage band."""

    claim_id: str
    band: CoverageBand
    rationale: str


def _is_scope_boundary(claim: dict[str, object]) -> bool:
    """Return whether a claim is a scope-boundary declaration (states what is not covered)."""
    claim_id = str(claim.get("id", ""))
    pattern = str(claim.get("source_pattern", ""))
    return (
        "scope_boundary" in claim_id or "scope-boundary" in claim_id or "scope" in pattern.lower()
    )


def _is_administrative(claim: dict[str, object]) -> bool:
    """Return whether a claim is administrative/provenance, not a physics-performance claim."""
    claim_id = str(claim.get("id", ""))
    return claim_id.startswith("security_") or "supported_release" in claim_id


def _evidence_strings(claim: dict[str, object]) -> list[str]:
    """Return the claim's evidence file/pattern strings."""
    files = claim.get("evidence_files") or []
    patterns = claim.get("evidence_patterns") or []
    out: list[str] = []
    if isinstance(files, list):
        out.extend(str(f) for f in files)
    if isinstance(patterns, list):
        out.extend(str(p) for p in patterns)
    return out


def classify_claim(claim: dict[str, object]) -> ClaimCoverage:
    """Map one claim to a conservative coverage band from its evidence kind.

    Parameters
    ----------
    claim
        A claim entry from ``claims_manifest.json`` (``id`` + ``evidence_files``/patterns).

    Returns
    -------
    ClaimCoverage
        The claim id, its conservative band, and a one-line rationale.
    """
    claim_id = str(claim.get("id", "unknown"))
    if _is_scope_boundary(claim):
        return ClaimCoverage(claim_id, CoverageBand.BOUNDARY, "scope-boundary declaration")
    if _is_administrative(claim):
        return ClaimCoverage(claim_id, CoverageBand.PRODUCER_ASSERTED, "administrative/provenance")

    evidence = _evidence_strings(claim)
    if not evidence:
        return ClaimCoverage(claim_id, CoverageBand.VALIDATION_GAP, "no backing evidence")

    blob = " ".join(evidence)
    if any(marker in blob for marker in _REFERENCE_VALIDATION_MARKERS):
        return ClaimCoverage(
            claim_id,
            CoverageBand.REFERENCE_VALIDATED,
            "real-shot or external-reference-code parity evidence",
        )
    if any(marker in blob for marker in _INTERNAL_METRIC_MARKERS):
        return ClaimCoverage(
            claim_id, CoverageBand.BOUNDED_MODEL, "internal benchmark / own test metric"
        )
    # Evidence present but neither reference-validation nor a recognised internal metric:
    # floor to bounded-model rather than launder it up to reference-validated.
    return ClaimCoverage(
        claim_id, CoverageBand.BOUNDED_MODEL, "evidence present, not reference-grade"
    )


def measure_coverage(manifest_path: Path = CLAIMS_MANIFEST) -> dict[str, object]:
    """Measure FUSION's honest validated-coverage from the claims ledger.

    Parameters
    ----------
    manifest_path
        Path to ``claims_manifest.json``.

    Returns
    -------
    dict
        ``{total, distribution, honest_validated_coverage, per_claim, ledger_gap}`` — the band
        counts, the reference-validated fraction, the per-claim classification, and the
        standing ledger-grading gap.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    claims = data.get("claims", [])
    classified = [classify_claim(c) for c in claims]
    distribution = Counter(c.band.value for c in classified)
    total = len(classified)
    validated = distribution.get(CoverageBand.REFERENCE_VALIDATED.value, 0)
    return {
        "total": total,
        "distribution": dict(distribution),
        "honest_validated_coverage": (validated / total) if total else 0.0,
        "per_claim": [(c.claim_id, c.band.value, c.rationale) for c in classified],
        "ledger_gap": LEDGER_GAP,
    }


def main() -> int:
    """Print FUSION's honest validated-coverage report (WS-6)."""
    result = measure_coverage()
    distribution = cast(dict[str, int], result["distribution"])
    coverage = cast(float, result["honest_validated_coverage"])
    print("FUSION honest validated-coverage (WS-6) — conservative floor")
    print(f"  total claims: {result['total']}")
    for band, count in sorted(distribution.items()):
        print(f"    {band:22s} {count}")
    print(f"  honest-validated coverage: {coverage * 100.0:.1f}% (reference-validated / total)")
    print(f"  ledger gap: {result['ledger_gap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
