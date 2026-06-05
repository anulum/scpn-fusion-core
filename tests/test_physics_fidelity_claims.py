# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Physics fidelity public-claims contract
# ----------------------------------------------------------------------
"""Guard reduced-order physics surfaces against full-fidelity public overclaims."""

from __future__ import annotations

import re
from pathlib import Path

from validation.benchmark_full_fidelity_acceptance import (
    evaluate_artifact_thresholds,
    run_benchmark,
)


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
            "multidimensional artifact-export contract",
            "no public DREAM kinetic-distribution parity",
        ),
        "Impurity transport (neoclassical)": (
            "Trace radial transport",
            "charge-state artifact/source-sink contract",
            "no public Aurora/STRAHL collisional-operator parity",
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
        "does not claim Aurora/STRAHL/JINTRAC collisional-operator parity",
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
    assert (
        report["reference_manifest"]
        == "validation/reference_data/full_fidelity_reference_cases.json"
    )

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

    for surface_name, surface in surfaces.items():
        assert surface["status"] == "not_full_fidelity"
        assert surface["acceptance_passed"] is False
        assert surface["missing_requirements"]
        if surface_name == "impurity_transport":
            assert surface["reference_cases"]["ready"] is True
            assert surface["native_same_case_comparison"]["comparison_ready"] is True
            assert surface["native_same_case_comparison"]["threshold_checks_ready"] is True
            assert surface["native_same_case_comparison"]["thresholds_passed"] is True
            assert (
                surface["native_same_case_comparison"]["status"]
                == "accepted_native_aurora_effective_transport_closure_thresholds"
            )
            assert surface["implemented_dimensions"][
                "aurora_strahl_same_case_transport_comparison"
            ] is True
            assert surface["implemented_dimensions"][
                "aurora_strahl_same_case_transport_threshold_checks"
            ] is True
            assert surface["implemented_dimensions"][
                "aurora_strahl_same_case_transport_thresholds_passed"
            ] is True
            assert surface["implemented_dimensions"][
                "aurora_effective_source_recycling_closure"
            ] is True
            assert surface["implemented_dimensions"][
                "charge_state_resolved_radial_transport_operator"
            ] is True
            assert surface["implemented_dimensions"][
                "aurora_strahl_same_case_source_sink_matrix_parity"
            ] is True
            assert surface["reference_cases"]["missing_cases"] == []
            case = surface["reference_cases"]["ready_cases"][0]
            assert case["artifact_exists"] is True
            assert case["provenance_ready"] is True
            assert case["redistribution_license_ready"] is True
            assert case["sha256_ready"] is True
            assert case["observable_keys_ready"] is True
            assert case["coordinate_keys_ready"] is True
        else:
            assert surface["reference_cases"]["ready"] is False
            assert surface["reference_cases"]["missing_cases"]
            case = surface["reference_cases"]["missing_cases"][0]
            assert case["artifact_exists"] is False
            assert case["provenance_ready"] is False
            assert case["redistribution_license_ready"] is False
            assert case["sha256_ready"] is False
            assert case["observable_keys_ready"] is False
            assert case["observable_keys_missing"]
            assert case["coordinate_keys_ready"] is False
            assert case["coordinate_keys_missing"]
        assert case["observables_declared"] is True
        assert case["observable_contracts_ready"] is True
        assert case["observable_axis_contracts_ready"] is True
        assert case["observable_axis_contracts_invalid"] == []
        assert case["observable_payload_invalid"] == []
        assert case["coordinates_declared"] is True
        assert case["coordinate_payload_invalid"] == []
        assert case["threshold_values_ready"] is True
        assert case["threshold_contracts_ready"] is True
        assert case["threshold_invalid"] == []
        assert case["threshold_contracts_missing"] == []


def test_full_fidelity_artifact_threshold_evaluator_passes_matching_payloads() -> None:
    """Reference artifact comparisons must compute thresholded quantitative metrics."""
    reference = {
        "observables": {
            "ion_heat_flux_spectrum": [[[1.0, 2.0], [3.0, 4.0]]],
            "electromagnetic_apar_energy": [1.0, 2.0, 3.0],
        }
    }
    candidate = {
        "observables": {
            "ion_heat_flux_spectrum": [[[1.01, 2.01], [2.99, 3.99]]],
            "electromagnetic_apar_energy": [1.0, 2.02, 2.98],
        }
    }
    thresholds = {
        "ion_heat_flux_relative_l2_max": 0.01,
        "field_energy_relative_error_max": 0.02,
    }
    threshold_contracts = {
        "ion_heat_flux_relative_l2_max": {
            "comparator": "<=",
            "metric": "relative_l2",
            "observable": "ion_heat_flux_spectrum",
        },
        "field_energy_relative_error_max": {
            "comparator": "<=",
            "metric": "relative_error",
            "observable": "electromagnetic_apar_energy",
        },
    }

    report = evaluate_artifact_thresholds(candidate, reference, thresholds, threshold_contracts)

    assert report["ready"] is True
    assert report["passed"] is True
    assert {check["threshold"] for check in report["checks"]} == set(thresholds)
    assert all(check["valid"] and check["passed"] for check in report["checks"])


def test_full_fidelity_artifact_threshold_evaluator_fails_closed_on_missing_observable() -> None:
    """Missing candidate/reference observables must be invalid, not silently skipped."""
    reference = {"observables": {"ion_heat_flux_spectrum": [1.0, 2.0]}}
    candidate = {"observables": {}}
    thresholds = {"ion_heat_flux_relative_l2_max": 0.1}
    threshold_contracts = {
        "ion_heat_flux_relative_l2_max": {
            "comparator": "<=",
            "metric": "relative_l2",
            "observable": "ion_heat_flux_spectrum",
        }
    }

    report = evaluate_artifact_thresholds(candidate, reference, thresholds, threshold_contracts)

    assert report["ready"] is False
    assert report["passed"] is False
    assert report["checks"][0]["valid"] is False
    assert report["checks"][0]["reason"] == "missing_candidate_observable"
