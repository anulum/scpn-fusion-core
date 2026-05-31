"""Full-fidelity acceptance contract for native high-order physics surfaces.

This diagnostic is intentionally stricter than the current native solver
implementations. It records whether native nonlinear gyrokinetics, runaway
electrons, and impurity transport satisfy end-to-end equivalence contracts
against GENE/CGYRO/GS2, DREAM, and Aurora/STRAHL-style production references.
It is a fail-closed acceptance benchmark: absence of public reference parity is
reported as an unmet requirement rather than silently downgraded.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core._gk_nonlinear_types import NonlinearGKConfig  # noqa: E402

REPORT_DIR = ROOT / "validation" / "reports"
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "full_fidelity_reference_cases.json"
ARTIFACT_SCHEMA = ROOT / "validation" / "reference_data" / "full_fidelity_artifact_schema.json"
JSON_REPORT = REPORT_DIR / "full_fidelity_acceptance_benchmark.json"
MD_REPORT = REPORT_DIR / "full_fidelity_acceptance_benchmark.md"


def _load_artifact_schema() -> dict[str, Any]:
    """Load the strict public reference artefact schema."""
    schema = json.loads(ARTIFACT_SCHEMA.read_text(encoding="utf-8"))
    if schema.get("schema") != "full-fidelity-artifact-schema.v1":
        raise ValueError("full-fidelity artefact schema mismatch")
    return schema


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a reference artefact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference_cases() -> dict[str, Any]:
    """Load and validate the full-fidelity public reference manifest."""
    manifest = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    if manifest.get("schema") != "full-fidelity-reference-cases.v1":
        raise ValueError("full-fidelity reference manifest schema mismatch")
    surfaces = manifest.get("surfaces")
    if not isinstance(surfaces, dict):
        raise ValueError("full-fidelity reference manifest must define surfaces")
    return manifest


def _observable_readiness(path: Path | None, observables: Any, contracts: Any) -> dict[str, Any]:
    """Return required observable presence and payload validity for JSON/NPZ artefacts."""
    required = [str(name) for name in observables] if isinstance(observables, list) else []
    if path is None or not path.exists() or not required:
        return {
            "ready": False,
            "present": [],
            "missing": required,
            "invalid": [],
        }

    data: dict[str, Any]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("observables", payload) if isinstance(payload, dict) else {}
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            data = {name: payload[name] for name in payload.files}
    else:
        data = {}

    keys = set(data) if isinstance(data, dict) else set()
    present = [name for name in required if name in keys]
    missing = [name for name in required if name not in keys]
    contract_map = contracts if isinstance(contracts, dict) else {}
    invalid = []
    for name in present:
        contract = (
            contract_map.get(name, {}) if isinstance(contract_map.get(name, {}), dict) else {}
        )
        try:
            array = np.asarray(data[name], dtype=float)
        except (TypeError, ValueError):
            invalid.append({"observable": name, "reason": "not_numeric"})
            continue
        min_rank = int(contract.get("min_rank", 0))
        if bool(contract.get("non_empty", True)) and array.size == 0:
            invalid.append({"observable": name, "reason": "empty"})
        elif array.ndim < min_rank:
            invalid.append(
                {
                    "observable": name,
                    "reason": "rank_below_minimum",
                    "rank": int(array.ndim),
                    "min_rank": min_rank,
                }
            )
        elif bool(contract.get("finite", True)) and not bool(np.all(np.isfinite(array))):
            invalid.append({"observable": name, "reason": "non_finite"})

    return {
        "ready": not missing and not invalid,
        "present": present,
        "missing": missing,
        "invalid": invalid,
    }


def _threshold_readiness(thresholds: Any, contracts: Any, observables: Any) -> dict[str, Any]:
    """Return quantitative parity-threshold value and comparator readiness."""
    if not isinstance(thresholds, dict) or not thresholds:
        return {
            "ready": False,
            "values_ready": False,
            "contracts_ready": False,
            "invalid": [],
            "missing_contracts": [],
        }

    contract_map = contracts if isinstance(contracts, dict) else {}
    observable_names = set(observables) if isinstance(observables, list) else set()
    invalid = []
    missing_contracts = []
    allowed_comparators = {"<=", ">="}
    allowed_metrics = {"absolute_error", "relative_error", "relative_l2"}
    for name, value in thresholds.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            invalid.append({"threshold": name, "reason": "not_numeric"})
            continue
        if not bool(np.isfinite(scalar)):
            invalid.append({"threshold": name, "reason": "non_finite"})
        elif scalar < 0.0:
            invalid.append({"threshold": name, "reason": "negative"})

        contract = contract_map.get(name)
        if not isinstance(contract, dict):
            missing_contracts.append(str(name))
            continue
        comparator = contract.get("comparator")
        if comparator not in allowed_comparators:
            invalid.append({"threshold": name, "reason": "unsupported_comparator"})
        metric = contract.get("metric")
        if not metric:
            invalid.append({"threshold": name, "reason": "missing_metric"})
        elif metric not in allowed_metrics:
            invalid.append({"threshold": name, "reason": "unsupported_metric"})
        observable = contract.get("observable")
        if not observable:
            invalid.append({"threshold": name, "reason": "missing_observable"})
        elif observable not in observable_names:
            invalid.append({"threshold": name, "reason": "observable_not_required"})

    values_ready = not invalid
    contracts_ready = not missing_contracts
    return {
        "ready": values_ready and contracts_ready,
        "values_ready": values_ready,
        "contracts_ready": contracts_ready,
        "invalid": invalid,
        "missing_contracts": missing_contracts,
    }


def _coordinate_readiness(path: Path | None, contracts: Any) -> dict[str, Any]:
    """Return coordinate/grid axis presence and payload validity for JSON/NPZ artefacts."""
    contract_map = contracts if isinstance(contracts, dict) else {}
    required = list(contract_map)
    if path is None or not path.exists() or not required:
        return {
            "ready": False,
            "present": [],
            "missing": required,
            "invalid": [],
        }

    data: dict[str, Any]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            coordinates = payload.get("coordinates", {})
            data = coordinates if isinstance(coordinates, dict) else {}
        else:
            data = {}
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            data = {name: payload[name] for name in payload.files}
    else:
        data = {}

    keys = set(data)
    present = [name for name in required if name in keys]
    missing = [name for name in required if name not in keys]
    invalid = []
    for name in present:
        contract = contract_map[name]
        try:
            axis = np.asarray(data[name], dtype=float)
        except (TypeError, ValueError):
            invalid.append({"coordinate": name, "reason": "not_numeric"})
            continue
        min_length = int(contract.get("min_length", 1))
        if axis.size < min_length:
            invalid.append(
                {
                    "coordinate": name,
                    "reason": "below_min_length",
                    "length": int(axis.size),
                    "min_length": min_length,
                }
            )
        elif axis.ndim != 1:
            invalid.append({"coordinate": name, "reason": "not_one_dimensional"})
        elif not bool(np.all(np.isfinite(axis))):
            invalid.append({"coordinate": name, "reason": "non_finite"})
        elif bool(contract.get("strictly_increasing", False)) and not bool(
            np.all(np.diff(axis) > 0.0)
        ):
            invalid.append({"coordinate": name, "reason": "not_strictly_increasing"})
        if not contract.get("units"):
            invalid.append({"coordinate": name, "reason": "missing_units"})

    return {
        "ready": not missing and not invalid,
        "present": present,
        "missing": missing,
        "invalid": invalid,
    }


def _observable_axis_contract_readiness(
    observables: Any, observable_contracts: Any, coordinate_contracts: Any
) -> dict[str, Any]:
    """Return observable-to-coordinate axis contract readiness."""
    required_observables = (
        [str(name) for name in observables] if isinstance(observables, list) else []
    )
    observable_contract_map = observable_contracts if isinstance(observable_contracts, dict) else {}
    coordinate_names = (
        set(coordinate_contracts) if isinstance(coordinate_contracts, dict) else set()
    )
    invalid = []
    for name in required_observables:
        contract = observable_contract_map.get(name)
        if not isinstance(contract, dict):
            invalid.append({"observable": name, "reason": "missing_observable_contract"})
            continue
        if not contract.get("units"):
            invalid.append({"observable": name, "reason": "missing_units"})
        axes = contract.get("axes")
        if not isinstance(axes, list) or not axes:
            invalid.append({"observable": name, "reason": "missing_axes"})
            continue
        axis_names = [str(axis) for axis in axes]
        min_rank = int(contract.get("min_rank", 0))
        if len(axis_names) < min_rank:
            invalid.append(
                {
                    "observable": name,
                    "reason": "axes_below_min_rank",
                    "axis_count": len(axis_names),
                    "min_rank": min_rank,
                }
            )
        missing_axes = [axis for axis in axis_names if axis not in coordinate_names]
        if missing_axes:
            invalid.append(
                {
                    "observable": name,
                    "reason": "axis_not_declared",
                    "missing_axes": missing_axes,
                }
            )

    return {
        "ready": not invalid and bool(required_observables),
        "invalid": invalid,
    }


def _reference_readiness(
    surface: str, manifest: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    """Return fail-closed public reference readiness for one physics surface."""
    surface_manifest = manifest["surfaces"].get(surface)
    if not isinstance(surface_manifest, dict):
        raise ValueError(f"missing reference manifest surface: {surface}")
    cases = surface_manifest.get("required_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"surface {surface} must define at least one required reference case")

    required_fields = tuple(schema["required_case_fields"])
    allowed_statuses = set(schema["allowed_statuses"])
    supported_suffixes = set(schema["supported_artifact_formats"])
    ready_cases = []
    missing_cases = []
    for case in cases:
        missing_fields = [field for field in required_fields if field not in case]
        artifact = case.get("artifact_path")
        artifact_path = ROOT / str(artifact) if artifact else None
        artifact_exists = bool(artifact_path and artifact_path.exists())
        suffix_ready = bool(artifact_path and artifact_path.suffix in supported_suffixes)
        thresholds = case.get("thresholds")
        threshold_ready = isinstance(thresholds, dict) and bool(thresholds)
        observables = case.get("required_observables")
        observable_contracts = case.get("observable_contracts")
        observables_declared = isinstance(observables, list) and bool(observables)
        contracts_ready = isinstance(observable_contracts, dict) and all(
            name in observable_contracts for name in (observables or [])
        )
        observable_report = _observable_readiness(artifact_path, observables, observable_contracts)
        coordinate_contracts = case.get("coordinate_contracts")
        coordinates_declared = isinstance(coordinate_contracts, dict) and bool(coordinate_contracts)
        coordinate_report = _coordinate_readiness(artifact_path, coordinate_contracts)
        observable_axis_report = _observable_axis_contract_readiness(
            observables, observable_contracts, coordinate_contracts
        )
        threshold_contracts = case.get("threshold_contracts")
        threshold_report = _threshold_readiness(thresholds, threshold_contracts, observables)
        status = case.get("status")
        status_ready = status == "available"
        status_known = status in allowed_statuses
        provenance_ready = bool(case.get("provenance_url"))
        license_ready = bool(case.get("redistribution_license"))
        expected_sha = case.get("sha256")
        actual_sha = (
            _sha256(artifact_path) if artifact_exists and artifact_path is not None else None
        )
        sha_ready = bool(expected_sha and actual_sha == expected_sha)
        case_ready = bool(
            not missing_fields
            and status_known
            and status_ready
            and artifact_exists
            and suffix_ready
            and provenance_ready
            and license_ready
            and sha_ready
            and observables_declared
            and contracts_ready
            and observable_axis_report["ready"]
            and observable_report["ready"]
            and coordinates_declared
            and coordinate_report["ready"]
            and threshold_ready
            and threshold_report["ready"]
        )
        row = {
            "case_id": case.get("case_id"),
            "reference_family": case.get("reference_family"),
            "status": status,
            "status_known": status_known,
            "artifact_path": artifact,
            "artifact_exists": artifact_exists,
            "artifact_format_ready": suffix_ready,
            "provenance_ready": provenance_ready,
            "redistribution_license_ready": license_ready,
            "sha256_expected": expected_sha,
            "sha256_actual": actual_sha,
            "sha256_ready": sha_ready,
            "observables_declared": observables_declared,
            "observable_contracts_ready": contracts_ready,
            "observable_axis_contracts_ready": observable_axis_report["ready"],
            "observable_axis_contracts_invalid": observable_axis_report["invalid"],
            "observable_keys_ready": observable_report["ready"],
            "observable_keys_present": observable_report["present"],
            "observable_keys_missing": observable_report["missing"],
            "observable_payload_invalid": observable_report["invalid"],
            "coordinates_declared": coordinates_declared,
            "coordinate_keys_ready": coordinate_report["ready"],
            "coordinate_keys_present": coordinate_report["present"],
            "coordinate_keys_missing": coordinate_report["missing"],
            "coordinate_payload_invalid": coordinate_report["invalid"],
            "threshold_ready": threshold_ready,
            "threshold_values_ready": threshold_report["values_ready"],
            "threshold_contracts_ready": threshold_report["contracts_ready"],
            "threshold_invalid": threshold_report["invalid"],
            "threshold_contracts_missing": threshold_report["missing_contracts"],
            "missing_fields": missing_fields,
            "ready": case_ready,
        }
        if case_ready:
            ready_cases.append(row)
        else:
            missing_cases.append(row)

    return {
        "manifest": str(REFERENCE_CASES.relative_to(ROOT)),
        "artifact_schema": str(ARTIFACT_SCHEMA.relative_to(ROOT)),
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
        "explicit_5d_phase_space_contract": True,
        "electromagnetic_b_parallel_surface": cfg.electromagnetic,
        "electromagnetic_b_parallel_hamiltonian_coupling": cfg.electromagnetic,
        "electromagnetic_field_energy_accounting": cfg.electromagnetic,
        "electromagnetic_energy_history_export": cfg.electromagnetic,
        "heat_flux_spectrum_history_export": True,
        "zonal_flow_energy_history_export": True,
        "saturation_window_diagnostics_export": True,
        "five_dimensional_delta_f_state": cfg.n_kx > 1
        and cfg.n_ky > 1
        and cfg.n_theta > 1
        and cfg.n_vpar > 1
        and cfg.n_mu > 1,
        "named_conservative_exb_term": True,
        "nonlinear_invariant_history_export": cfg.nonlinear,
        "jax_run_history_parity": True,
        "nonlinear_exb_operator": cfg.nonlinear,
        "kinetic_electron_surface": cfg.kinetic_electrons,
        "electromagnetic_a_parallel_surface": cfg.electromagnetic,
        "moment_conserving_collision_contract": cfg.collisions and cfg.collision_model == "sugama",
    }
    schema = _load_artifact_schema()
    readiness = _reference_readiness("native_nonlinear_gyrokinetics", reference_cases, schema)
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
    schema = _load_artifact_schema()
    readiness = _reference_readiness("runaway_electrons", reference_cases, schema)
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
    schema = _load_artifact_schema()
    readiness = _reference_readiness("impurity_transport", reference_cases, schema)
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
        "artifact_schema": str(ARTIFACT_SCHEMA.relative_to(ROOT)),
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
        f"- Artefact schema: `{report['artifact_schema']}`",
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
