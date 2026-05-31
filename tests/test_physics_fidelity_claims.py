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

from validation.benchmark_full_fidelity_acceptance import run_benchmark


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PHYSICS_METHODS = ROOT / "docs" / "PHYSICS_METHODS_COMPLETE.md"


def _table_cell_for_capability(readme: str, capability: str) -> str:
    pattern = re.compile(rf"^\| {re.escape(capability)} \| (?P<cell>[^|]+) \|", re.MULTILINE)
    match = pattern.search(readme)
    assert match is not None, f"README competitive table is missing {capability!r}"
    return match.group("cell").strip()


def test_reduced_order_competitive_claims_disclose_actual_fidelity() -> None:
    """README competitive cells must state reduced-order scope for non-parity physics."""
    readme = README.read_text(encoding="utf-8")

    required_boundaries = {
        "Free-boundary GS solve": (
            "Public GEQDSK operator-source gate passes",
            "profile-source/free-boundary reconstruction gate remains open",
            "not EFIT-grade",
        ),
        "Native GK solver": (
            "Linear eigenvalue plus nonlinear 5D operator/invariant benchmarks",
            "not GENE/CGYRO-class",
        ),
        "Free-boundary tracking": ("not EFIT/LiUQE-grade", "inverse reconstruction"),
        "Disruption chain (TQ+CQ+RE+halo)": ("Reduced chain", "0D runaway rates"),
        "ELM model + RMP suppression": ("Peeling-ballooning proxy", "no nonlinear MHD"),
        "Runaway electron dynamics": (
            "DREAM-style fluid balance",
            "no multidimensional DREAM kinetic-distribution parity",
        ),
        "Impurity transport (neoclassical)": (
            "Trace radial transport",
            "no STRAHL/JINTRAC collisional-operator parity",
        ),
    }

    for capability, phrases in required_boundaries.items():
        cell = _table_cell_for_capability(readme, capability)
        assert cell != "**Y**"
        for phrase in phrases:
            assert phrase in cell


def test_public_scope_names_first_principles_comparators() -> None:
    """Public scope documentation must identify external solvers not replaced here."""
    public_scope = " ".join(
        (README.read_text(encoding="utf-8") + "\n" + PHYSICS_METHODS.read_text(encoding="utf-8"))
        .replace("**", "")
        .split()
    )

    required_phrases = [
        "not a replacement for TRANSP, JINTRAC, or GENE",
        "not GENE/CGYRO-class production turbulence",
        "not yet EFIT-grade",
        "Hirshman-Sigmar-style",
        "does not claim STRAHL/JINTRAC collisional-operator parity",
        "does not claim parity with DREAM's kinetic momentum-space distribution solver",
        "no nonlinear MHD ELM simulation",
    ]

    for phrase in required_phrases:
        assert phrase in public_scope


def test_full_fidelity_acceptance_contract_fails_closed_until_reference_parity() -> None:
    """Full-order claims require explicit public reference parity evidence."""
    report = run_benchmark()

    assert report["schema"] == "full-fidelity-acceptance.v1"
    assert report["gate_mode"] == "diagnostic_fail_closed"
    assert report["acceptance_passed"] is False

    surfaces = {surface["surface"]: surface for surface in report["surfaces"]}
    assert set(surfaces) == {
        "native_nonlinear_gyrokinetics",
        "runaway_electrons",
        "impurity_transport",
    }
    assert surfaces["native_nonlinear_gyrokinetics"]["required_reference_equivalence"] == (
        "GENE/CGYRO/GS2 full nonlinear 5D Vlasov-Maxwell"
    )
    assert surfaces["runaway_electrons"]["required_reference_equivalence"] == (
        "DREAM kinetic/fluid runaway electron solver"
    )
    assert surfaces["impurity_transport"]["required_reference_equivalence"] == (
        "Aurora/STRAHL collisional-operator impurity transport"
    )

    for surface in surfaces.values():
        assert surface["status"] == "not_full_fidelity"
        assert surface["acceptance_passed"] is False
        assert surface["missing_requirements"]
