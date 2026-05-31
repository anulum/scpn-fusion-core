"""Full-fidelity acceptance contract for native high-order physics surfaces.

This diagnostic is intentionally stricter than the current native solver
implementations. It records whether native nonlinear gyrokinetics, runaway
electrons, and impurity transport satisfy end-to-end equivalence contracts
against GENE/CGYRO/GS2, DREAM, and Aurora/STRAHL-style production references.
It is a fail-closed acceptance benchmark: absence of public reference parity is
reported as an unmet requirement rather than silently downgraded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core._gk_nonlinear_types import NonlinearGKConfig  # noqa: E402

REPORT_DIR = ROOT / "validation" / "reports"
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "full_fidelity_reference_cases.json"
JSON_REPORT = REPORT_DIR / "full_fidelity_acceptance_benchmark.json"
MD_REPORT = REPORT_DIR / "full_fidelity_acceptance_benchmark.md"


def _load_reference_cases() -> dict[str, Any]:
    """Load and validate the full-fidelity public reference manifest."""
    manifest = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    if manifest.get("schema") != "full-fidelity-reference-cases.v1":
        raise ValueError("full-fidelity reference manifest schema mismatch")
    surfaces = manifest.get("surfaces")
    if not isinstance(surfaces, dict):
        raise ValueError("full-fidelity reference manifest must define surfaces")
    return manifest


def _reference_readiness(surface: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Return fail-closed public reference readiness for one physics surface."""
    surface_manifest = manifest["surfaces"].get(surface)
    if not isinstance(surface_manifest, dict):
        raise ValueError(f"missing reference manifest surface: {surface}")
    cases = surface_manifest.get("required_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"surface {surface} must define at least one required reference case")

    ready_cases = []
    missing_cases = []
    for case in cases:
        artifact = case.get("artifact_path")
        artifact_exists = bool(artifact) and (ROOT / str(artifact)).exists()
        thresholds = case.get("thresholds")
        threshold_ready = isinstance(thresholds, dict) and bool(thresholds)
        status_ready = case.get("status") == "available"
        case_ready = bool(status_ready and artifact_exists and threshold_ready)
        row = {
            "case_id": case.get("case_id"),
            "reference_family": case.get("reference_family"),
            "status": case.get("status"),
            "artifact_path": artifact,
            "artifact_exists": artifact_exists,
            "threshold_ready": threshold_ready,
            "ready": case_ready,
        }
        if case_ready:
            ready_cases.append(row)
        else:
            missing_cases.append(row)

    return {
        "manifest": str(REFERENCE_CASES.relative_to(ROOT)),
        "required_equivalence": surface_manifest.get("required_equivalence"),
        "ready": len(missing_cases) == 0,
        "ready_cases": ready_cases,
        "missing_cases": missing_cases,
    }


def _nonlinear_gk_contract(reference_cases: dict[str, Any]) -> dict[str, Any]:
    cfg = NonlinearGKConfig(
        n_kx=4,
        n_ky=4,
        n_theta=8,
        n_vpar=8,
        n_mu=6,
        n_species=2,
        kinetic_electrons=True,
        electromagnetic=True,
        nonlinear=True,
        collisions=True,
        collision_model="sugama",
    )
    implemented_dimensions = {
        "five_dimensional_delta_f_state": cfg.n_kx > 1
        and cfg.n_ky > 1
        and cfg.n_theta > 1
        and cfg.n_vpar > 1
        and cfg.n_mu > 1,
        "nonlinear_exb_operator": cfg.nonlinear,
        "kinetic_electron_surface": cfg.kinetic_electrons,
        "electromagnetic_a_parallel_surface": cfg.electromagnetic,
        "moment_conserving_collision_contract": cfg.collisions and cfg.collision_model == "sugama",
    }
    readiness = _reference_readiness("native_nonlinear_gyrokinetics", reference_cases)
    missing_requirements = [
        "public nonlinear GENE/CGYRO/GS2 benchmark deck parity",
        "production-scale radial/toroidal domain decomposition and convergence evidence",
        "Maxwell field solve parity beyond compact A_parallel contract",
        "validated flux spectra, zonal-flow, and saturation parity against production GK outputs",
    ]
    return {
        "surface": "native_nonlinear_gyrokinetics",
        "required_reference_equivalence": "GENE/CGYRO/GS2 full nonlinear 5D Vlasov-Maxwell",
        "implemented_dimensions": implemented_dimensions,
        "reference_cases": readiness,
        "missing_requirements": missing_requirements,
        "acceptance_passed": False,
        "status": "not_full_fidelity",
    }


def _runaway_contract(reference_cases: dict[str, Any]) -> dict[str, Any]:
    implemented_dimensions = {
        "dreicer_source": True,
        "avalanche_source": True,
        "hot_tail_seed": True,
        "fluid_density_balance": True,
        "one_dimensional_momentum_fokker_planck_contract": True,
    }
    readiness = _reference_readiness("runaway_electrons", reference_cases)
    missing_requirements = [
        "multidimensional DREAM kinetic distribution parity",
        "coupled radial-momentum-pitch kinetic grid with DREAM reference cases",
        "synchrotron, bremsstrahlung, partial-screening, and transport parity gates",
        "public DREAM deck ingestion and distribution-function RMSE thresholds",
    ]
    return {
        "surface": "runaway_electrons",
        "required_reference_equivalence": "DREAM kinetic/fluid runaway electron solver",
        "implemented_dimensions": implemented_dimensions,
        "reference_cases": readiness,
        "missing_requirements": missing_requirements,
        "acceptance_passed": False,
        "status": "not_full_fidelity",
    }


def _impurity_contract(reference_cases: dict[str, Any]) -> dict[str, Any]:
    implemented_dimensions = {
        "trace_radial_transport": True,
        "edge_source_particle_conservation": True,
        "neoclassical_pinch_contract": True,
        "radiated_power_monotonicity": True,
    }
    readiness = _reference_readiness("impurity_transport", reference_cases)
    missing_requirements = [
        "charge-state-resolved collisional-radiative operator parity",
        "ADAS-backed ionisation/recombination/radiation coefficient ingestion",
        "Aurora/STRAHL public case ingestion and density/radiation RMSE gates",
        "multi-species source/sink matrix conservation across charge states",
    ]
    return {
        "surface": "impurity_transport",
        "required_reference_equivalence": "Aurora/STRAHL collisional-operator impurity transport",
        "implemented_dimensions": implemented_dimensions,
        "reference_cases": readiness,
        "missing_requirements": missing_requirements,
        "acceptance_passed": False,
        "status": "not_full_fidelity",
    }


def run_benchmark() -> dict[str, Any]:
    """Return full-fidelity acceptance status for native physics surfaces."""
    reference_cases = _load_reference_cases()
    surfaces = [
        _nonlinear_gk_contract(reference_cases),
        _runaway_contract(reference_cases),
        _impurity_contract(reference_cases),
    ]
    return {
        "benchmark": "full_fidelity_acceptance",
        "schema": "full-fidelity-acceptance.v1",
        "description": (
            "Fail-closed acceptance contract for native nonlinear GK, runaway electron, "
            "and impurity transport equivalence against production community solvers."
        ),
        "gate_mode": "diagnostic_fail_closed",
        "reference_manifest": str(REFERENCE_CASES.relative_to(ROOT)),
        "surfaces": surfaces,
        "acceptance_passed": all(surface["acceptance_passed"] for surface in surfaces),
    }


def write_reports(report: dict[str, Any]) -> None:
    """Write JSON and Markdown full-fidelity acceptance reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Full-Fidelity Acceptance Benchmark",
        "",
        "This report is a fail-closed diagnostic for native full-fidelity claims.",
        "A surface passes only after public reference parity against the named production solver family is demonstrated.",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Gate mode: `{report['gate_mode']}`",
        f"- Reference manifest: `{report['reference_manifest']}`",
        f"- Acceptance passed: `{report['acceptance_passed']}`",
        "",
        "| Surface | Required reference equivalence | Status | Reference cases ready | Implemented dimensions | Missing requirements |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for surface in report["surfaces"]:
        implemented = ", ".join(
            name for name, present in surface["implemented_dimensions"].items() if present
        )
        missing = "<br>".join(surface["missing_requirements"])
        lines.append(
            "| {surface} | {reference} | {status} | {ready} | {implemented} | {missing} |".format(
                surface=surface["surface"],
                reference=surface["required_reference_equivalence"],
                status=surface["status"],
                ready=surface["reference_cases"]["ready"],
                implemented=implemented,
                missing=missing,
            )
        )
    lines.append("")
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Execute the full-fidelity acceptance diagnostic."""
    report = run_benchmark()
    write_reports(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
