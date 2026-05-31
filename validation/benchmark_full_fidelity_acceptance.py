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
JSON_REPORT = REPORT_DIR / "full_fidelity_acceptance_benchmark.json"
MD_REPORT = REPORT_DIR / "full_fidelity_acceptance_benchmark.md"


def _nonlinear_gk_contract() -> dict[str, Any]:
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
        "missing_requirements": missing_requirements,
        "acceptance_passed": False,
        "status": "not_full_fidelity",
    }


def _runaway_contract() -> dict[str, Any]:
    implemented_dimensions = {
        "dreicer_source": True,
        "avalanche_source": True,
        "hot_tail_seed": True,
        "fluid_density_balance": True,
        "one_dimensional_momentum_fokker_planck_contract": True,
    }
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
        "missing_requirements": missing_requirements,
        "acceptance_passed": False,
        "status": "not_full_fidelity",
    }


def _impurity_contract() -> dict[str, Any]:
    implemented_dimensions = {
        "trace_radial_transport": True,
        "edge_source_particle_conservation": True,
        "neoclassical_pinch_contract": True,
        "radiated_power_monotonicity": True,
    }
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
        "missing_requirements": missing_requirements,
        "acceptance_passed": False,
        "status": "not_full_fidelity",
    }


def run_benchmark() -> dict[str, Any]:
    """Return full-fidelity acceptance status for native physics surfaces."""
    surfaces = [_nonlinear_gk_contract(), _runaway_contract(), _impurity_contract()]
    return {
        "benchmark": "full_fidelity_acceptance",
        "schema": "full-fidelity-acceptance.v1",
        "description": (
            "Fail-closed acceptance contract for native nonlinear GK, runaway electron, "
            "and impurity transport equivalence against production community solvers."
        ),
        "gate_mode": "diagnostic_fail_closed",
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
        f"- Acceptance passed: `{report['acceptance_passed']}`",
        "",
        "| Surface | Required reference equivalence | Status | Implemented dimensions | Missing requirements |",
        "| --- | --- | --- | --- | --- |",
    ]
    for surface in report["surfaces"]:
        implemented = ", ".join(
            name for name, present in surface["implemented_dimensions"].items() if present
        )
        missing = "<br>".join(surface["missing_requirements"])
        lines.append(
            "| {surface} | {reference} | {status} | {implemented} | {missing} |".format(
                surface=surface["surface"],
                reference=surface["required_reference_equivalence"],
                status=surface["status"],
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
