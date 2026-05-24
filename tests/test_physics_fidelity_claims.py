# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Physics fidelity public-claims contract
# ----------------------------------------------------------------------
"""Guard reduced-order physics surfaces against full-fidelity public overclaims."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
HONEST_SCOPE = ROOT / "docs" / "HONEST_SCOPE.md"


def _table_cell_for_capability(readme: str, capability: str) -> str:
    pattern = re.compile(rf"^\| {re.escape(capability)} \| (?P<cell>[^|]+) \|", re.MULTILINE)
    match = pattern.search(readme)
    assert match is not None, f"README competitive table is missing {capability!r}"
    return match.group("cell").strip()


def test_reduced_order_competitive_claims_disclose_actual_fidelity() -> None:
    """README competitive cells must state reduced-order scope for non-parity physics."""
    readme = README.read_text(encoding="utf-8")

    required_boundaries = {
        "Free-boundary GS solve": ("Reduced", "FreeGS parity currently **FAIL**"),
        "Native GK eigenvalue solver": ("Linear eigenvalue only", "no nonlinear 5D"),
        "Free-boundary tracking": ("not EFIT/LiUQE-grade", "inverse reconstruction"),
        "Disruption chain (TQ+CQ+RE+halo)": ("Reduced chain", "0D runaway rates"),
        "ELM model + RMP suppression": ("Peeling-ballooning proxy", "no nonlinear MHD"),
        "Runaway electron dynamics": ("0D rates only", "no DREAM-level kinetic"),
        "Impurity transport (neoclassical)": ("Reduced neoclassical", "no STRAHL/JINTRAC"),
    }

    for capability, phrases in required_boundaries.items():
        cell = _table_cell_for_capability(readme, capability)
        assert cell != "**Y**"
        for phrase in phrases:
            assert phrase in cell


def test_honest_scope_names_first_principles_comparators() -> None:
    """Scope documentation must identify the external solvers this repo does not replace."""
    honest_scope = " ".join(HONEST_SCOPE.read_text(encoding="utf-8").replace("**", "").split())

    required_phrases = [
        "not a replacement for TRANSP, JINTRAC, GENE, or any",
        "full nonlinear gyrokinetic (GENE, GS2, CGYRO solve 5D Vlasov-Maxwell)",
        "Full EFIT/LiUQE-quality profile and boundary reconstruction",
        "Full Hirshman-Sigmar multi-species collisional operator",
        "Kinetic runaway distribution (CODE/DREAM-level Fokker-Planck)",
        "Nonlinear MHD ELM simulation (JOREK, BOUT++)",
    ]

    for phrase in required_phrases:
        assert phrase in honest_scope
