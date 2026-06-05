#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Benchmark trace impurity transport contracts.

This validates Aurora/STRAHL-style transport invariants available in the native
trace impurity surface: positivity, edge-source particle conservation,
neoclassical inward pinch sign, and monotonic radiated power. It does not claim
collisional-operator parity with Aurora, STRAHL, or JINTRAC.
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.impurity_transport import (  # noqa: E402
    AuroraParityCase,
    AuroraParityImpuritySolver,
    ImpuritySpecies,
    ImpurityTransportSolver,
    build_aurora_strahl_charge_state_artifact,
    neoclassical_impurity_pinch,
    total_radiated_power,
)

REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "impurity_transport_contract_benchmark.json"
MD_REPORT = REPORT_DIR / "impurity_transport_contract_benchmark.md"
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "full_fidelity_reference_cases.json"
FloatArray: TypeAlias = NDArray[np.float64]


def _inventory(n_z: FloatArray, rho: FloatArray, R0: float, a: float) -> float:
    vol_element = 4.0 * np.pi**2 * R0 * a**2 * rho
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(n_z * vol_element, rho))


def _observable_finiteness(
    payload: dict[str, Any], required_observables: list[str]
) -> dict[str, bool]:
    observables = payload["observables"]
    return {
        name: bool(
            (values := np.asarray(observables.get(name, []), dtype=np.float64)).size > 0
            and np.all(np.isfinite(values))
        )
        for name in required_observables
    }


def _source_sink_budget_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    observables = payload["observables"]
    source_sink = np.asarray(observables["source_sink_matrix_t_r_z_z"], dtype=np.float64)
    total_density = np.asarray(observables["total_impurity_density_r_t"], dtype=np.float64)
    ionisation = np.asarray(observables["ionisation_source_matrix"], dtype=np.float64)
    recombination = np.asarray(observables["recombination_sink_matrix"], dtype=np.float64)
    line_radiation = np.asarray(observables["line_radiation_power_t_r_z"], dtype=np.float64)
    inventory = np.asarray(observables["total_impurity_inventory_t"], dtype=np.float64)
    budget_arrays = {
        "source_sink_matrix_t_r_z_z": source_sink,
        "ionisation_source_matrix": ionisation,
        "recombination_sink_matrix": recombination,
        "line_radiation_power_t_r_z": line_radiation,
        "total_impurity_inventory_t": inventory,
    }
    diagonal = np.diagonal(source_sink, axis1=2, axis2=3)
    offdiagonal = source_sink.copy()
    charge_count = source_sink.shape[2] if source_sink.ndim == 4 else 0
    for charge_idx in range(charge_count):
        offdiagonal[:, :, charge_idx, charge_idx] = 0.0
    inventory_baseline = max(float(abs(inventory[0])) if inventory.size else 0.0, 1.0)
    inventory_relative_change = (
        float(np.max(np.abs(inventory - inventory[0]) / inventory_baseline))
        if inventory.size
        else np.inf
    )
    radial_density_baseline = (
        np.maximum(np.abs(total_density[0]), 1.0)
        if total_density.ndim == 2 and total_density.shape[0] > 0
        else np.asarray([], dtype=float)
    )
    max_radial_density_change = (
        float(np.max(np.abs(total_density - total_density[0]) / radial_density_baseline))
        if radial_density_baseline.size
        else np.inf
    )
    source_sink_scale = max(float(np.max(np.abs(source_sink))) if source_sink.size else 0.0, 1.0)
    source_sink_tolerance = 1.0e-12 * source_sink_scale + 1.0e-6
    return {
        "schema": "native-impurity-source-sink-budget-evidence.v1",
        "status": "native_artifact_source_sink_budget_only_not_aurora_strahl_operator_parity",
        "budget_terms": list(budget_arrays),
        "time_count": int(source_sink.shape[0]) if source_sink.ndim >= 1 else 0,
        "radius_count": int(source_sink.shape[1]) if source_sink.ndim >= 2 else 0,
        "charge_state_count": int(charge_count),
        "term_shapes": {
            name: [int(axis) for axis in values.shape] for name, values in budget_arrays.items()
        },
        "term_finiteness": {
            name: bool(values.size > 0 and np.all(np.isfinite(values)))
            for name, values in budget_arrays.items()
        },
        "all_budget_terms_finite": bool(
            all(
                values.size > 0 and np.all(np.isfinite(values)) for values in budget_arrays.values()
            )
        ),
        "ionisation_recombination_nonnegative": bool(
            ionisation.size > 0
            and recombination.size > 0
            and np.all(ionisation >= 0.0)
            and np.all(recombination >= 0.0)
        ),
        "source_sink_transfer_conservative": bool(
            source_sink.ndim == 4
            and np.all(np.abs(np.sum(source_sink, axis=3)) <= source_sink_tolerance)
        ),
        "source_sink_row_sum_abs_max": float(np.max(np.abs(np.sum(source_sink, axis=3)))),
        "source_sink_diagonal_nonpositive": bool(diagonal.size > 0 and np.all(diagonal <= 0.0)),
        "source_sink_offdiagonal_nonnegative": bool(
            offdiagonal.size > 0 and np.all(offdiagonal >= 0.0)
        ),
        "radial_total_density_conserved": bool(max_radial_density_change <= 1.0e-12),
        "max_radial_total_density_relative_change": max_radial_density_change,
        "line_radiation_nonnegative": bool(
            line_radiation.size > 0 and np.all(line_radiation >= 0.0)
        ),
        "inventory_relative_change_max": inventory_relative_change,
        "max_ionisation_source_m3_s": float(np.max(ionisation)),
        "max_recombination_sink_m3_s": float(np.max(recombination)),
        "max_line_radiation_power_w": float(np.max(line_radiation)),
        "aurora_strahl_same_case_budget_ready": False,
        "blocking_requirements": [
            "same-case Aurora or STRAHL source-sink matrix output",
            "same-case Aurora or STRAHL charge-state density history",
            "same-case Aurora or STRAHL line-radiation power output",
        ],
    }


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a tracked reference artefact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_l2(candidate: FloatArray, reference: FloatArray) -> float:
    """Return scale-stable relative L2 mismatch for same-case artefacts."""
    return float(
        np.linalg.norm((candidate - reference).ravel())
        / max(float(np.linalg.norm(reference.ravel())), 1.0e-30)
    )


def _relative_error(candidate: FloatArray, reference: FloatArray) -> float:
    """Return scale-stable max relative error for same-case artefacts."""
    return float(
        np.max(np.abs(candidate - reference)) / max(float(np.max(np.abs(reference))), 1.0e-30)
    )


def _load_impurity_reference_case() -> dict[str, Any]:
    """Load the accepted impurity reference-case row from the public manifest."""
    manifest = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != "full-fidelity-reference-cases.v1":
        raise ValueError("full-fidelity reference manifest schema mismatch")
    surfaces = manifest.get("surfaces")
    if not isinstance(surfaces, dict):
        raise ValueError("full-fidelity reference manifest must define surfaces")
    impurity = surfaces.get("impurity_transport")
    if not isinstance(impurity, dict):
        raise ValueError("full-fidelity reference manifest must define impurity_transport")
    cases = impurity.get("required_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("impurity_transport must define at least one required case")
    case = cases[0]
    if not isinstance(case, dict):
        raise ValueError("impurity reference case must be a JSON object")
    return case


def _load_reference_npz(case: dict[str, Any]) -> tuple[Path | None, dict[str, FloatArray], list[str]]:
    """Load a manifest-declared Aurora/STRAHL reference NPZ fail-closed."""
    artifact = case.get("artifact_path")
    if not isinstance(artifact, str) or not artifact:
        return None, {}, ["missing_artifact_path"]
    artifact_path = ROOT / artifact
    if not artifact_path.exists():
        return artifact_path, {}, ["artifact_path_missing"]
    expected_sha = case.get("sha256")
    if not isinstance(expected_sha, str) or _sha256(artifact_path) != expected_sha:
        return artifact_path, {}, ["artifact_sha256_mismatch"]
    arrays: dict[str, FloatArray] = {}
    required = case.get("required_observables")
    coordinate_contracts = case.get("coordinate_contracts")
    required_names = [str(name) for name in required] if isinstance(required, list) else []
    coordinate_names = (
        [str(name) for name in coordinate_contracts]
        if isinstance(coordinate_contracts, dict)
        else []
    )
    optional_case_keys = [
        "convection_m_s_r_z",
        "diffusion_m2_s_r_z",
        "effective_source_m3_s_t_r_z",
        "electron_density_t_r_m3",
        "electron_temperature_t_r_ev",
        "ionisation_coeff_m3_s_t_r_z",
        "line_radiation_coeff_w_m3_t_r_z",
        "recombination_coeff_m3_s_t_r_z",
    ]
    with np.load(artifact_path, allow_pickle=False) as payload:
        missing = [
            name for name in (*coordinate_names, *required_names) if name not in payload.files
        ]
        if missing:
            return artifact_path, {}, [f"missing_payload_keys:{','.join(sorted(missing))}"]
        for name in (
            *coordinate_names,
            *required_names,
            "line_radiation_power_t_r_z",
            *optional_case_keys,
        ):
            if name in payload.files and name not in arrays:
                array = np.asarray(payload[name], dtype=np.float64)
                if array.size == 0 or not np.all(np.isfinite(array)):
                    return artifact_path, {}, [f"invalid_payload_key:{name}"]
                arrays[name] = array
    return artifact_path, arrays, []


def _native_same_case_payload(reference: dict[str, FloatArray]) -> dict[str, Any]:
    """Run the native charge-state CR contract on the accepted Aurora axes."""
    charge_state = reference["charge_state"]
    radius_m = reference["radius_m"]
    time_s = reference["time_s"]
    charge_density = reference["charge_state_density_r_t"]
    if charge_density.ndim != 3:
        raise ValueError("reference charge_state_density_r_t must be time x radius x charge")
    if charge_density.shape != (time_s.size, radius_m.size, charge_state.size):
        raise ValueError("reference density shape does not match time/radius/charge axes")

    radius_norm = radius_m / max(float(radius_m[-1]), 1.0e-30)
    ne_t_r = reference.get("electron_density_t_r_m3")
    if ne_t_r is None:
        ne_profile = 1.0e20 * (1.0 - 0.4 * radius_norm**2) + 4.0e19
        ne_t_r = np.tile(ne_profile, (time_s.size, 1))
    te_t_r = reference.get("electron_temperature_t_r_ev")
    if te_t_r is None:
        te_profile = 5.0e3 * (1.0 - radius_norm**2) ** 1.5 + 100.0
        te_t_r = np.tile(te_profile, (time_s.size, 1))
    diffusion = reference.get("diffusion_m2_s_r_z")
    if diffusion is None:
        diffusion = np.tile(
            np.ones(radius_m.size, dtype=np.float64)[:, np.newaxis],
            (1, charge_state.size),
        )
    convection = reference.get("convection_m_s_r_z")
    if convection is None:
        convection = np.tile((-10.0 * radius_norm**5)[:, np.newaxis], (1, charge_state.size))
    parity_case = AuroraParityCase(
        element="Ar",
        charge_states=charge_state,
        radius_m=radius_m,
        time_s=time_s,
        ne_t_r=ne_t_r,
        Te_t_r=te_t_r,
        initial_charge_state_density_rz=np.maximum(charge_density[0], 0.0),
        diffusion_m2_s_r_z=diffusion,
        convection_m_s_r_z=convection,
        major_radius_m=1.7,
        ionisation_m3_s_t_r_z=reference.get("ionisation_coeff_m3_s_t_r_z"),
        recombination_m3_s_t_r_z=reference.get("recombination_coeff_m3_s_t_r_z"),
        line_radiation_w_m3_t_r_z=reference.get("line_radiation_coeff_w_m3_t_r_z"),
        effective_source_m3_s_t_r_z=reference.get("effective_source_m3_s_t_r_z"),
    )
    return dict(AuroraParityImpuritySolver(parity_case).solve().to_dict())


def _native_observables_as_top_level(payload: dict[str, Any]) -> dict[str, FloatArray]:
    """Return native observable arrays keyed like the public NPZ reference."""
    observables = payload["observables"]
    coordinates = payload["coordinates"]
    return {
        "charge_state": np.asarray(coordinates["charge_state"], dtype=np.float64),
        "radius_m": np.asarray(coordinates["radius_m"], dtype=np.float64),
        "time_s": np.asarray(coordinates["time_s"], dtype=np.float64),
        "charge_state_density_r_t": np.asarray(
            observables["charge_state_density_r_t"], dtype=np.float64
        ),
        "total_impurity_density_r_t": np.asarray(
            observables["total_impurity_density_r_t"], dtype=np.float64
        ),
        "line_radiation_power_t": np.asarray(
            observables["line_radiation_power_t"], dtype=np.float64
        ),
        "line_radiation_power_t_r_z": np.asarray(
            observables["line_radiation_power_t_r_z"], dtype=np.float64
        ),
        "source_sink_matrix_t_r_z_z": np.asarray(
            observables["source_sink_matrix_t_r_z_z"], dtype=np.float64
        ),
        "ionisation_source_matrix": np.asarray(
            observables["ionisation_source_matrix"], dtype=np.float64
        ),
        "recombination_sink_matrix": np.asarray(
            observables["recombination_sink_matrix"], dtype=np.float64
        ),
    }


def _same_case_threshold_checks(
    candidate: dict[str, FloatArray], reference: dict[str, FloatArray], case: dict[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate declared manifest thresholds on native-vs-Aurora arrays."""
    thresholds = case.get("thresholds")
    contracts = case.get("threshold_contracts")
    if not isinstance(thresholds, dict) or not isinstance(contracts, dict):
        return [
            {
                "threshold": "manifest_thresholds",
                "valid": False,
                "passed": False,
                "reason": "missing_threshold_contracts",
            }
        ]
    checks: list[dict[str, Any]] = []
    for threshold_name, raw_limit in thresholds.items():
        contract = contracts.get(threshold_name)
        check: dict[str, Any] = {
            "threshold": str(threshold_name),
            "valid": False,
            "passed": False,
        }
        if not isinstance(contract, dict):
            check["reason"] = "missing_threshold_contract"
            checks.append(check)
            continue
        observable = contract.get("observable")
        metric = contract.get("metric")
        comparator = contract.get("comparator")
        check.update({"observable": observable, "metric": metric, "comparator": comparator})
        if not isinstance(observable, str) or observable not in candidate or observable not in reference:
            check["reason"] = "missing_observable"
            checks.append(check)
            continue
        candidate_array = candidate[observable]
        reference_array = reference[observable]
        if candidate_array.shape != reference_array.shape:
            check.update(
                {
                    "reason": "observable_shape_mismatch",
                    "candidate_shape": [int(axis) for axis in candidate_array.shape],
                    "reference_shape": [int(axis) for axis in reference_array.shape],
                }
            )
            checks.append(check)
            continue
        limit = float(raw_limit)
        if metric == "relative_l2":
            value = _relative_l2(candidate_array, reference_array)
        elif metric == "relative_error":
            value = _relative_error(candidate_array, reference_array)
        elif metric == "absolute_error":
            value = float(np.max(np.abs(candidate_array - reference_array)))
        else:
            check["reason"] = "unsupported_metric"
            checks.append(check)
            continue
        if comparator != "<=":
            check["reason"] = "unsupported_comparator"
            checks.append(check)
            continue
        check.update(
            {
                "limit": limit,
                "value": value,
                "valid": bool(np.isfinite(value) and np.isfinite(limit) and limit >= 0.0),
                "passed": bool(value <= limit),
            }
        )
        checks.append(check)
    return checks


def _aurora_same_case_comparison() -> dict[str, Any]:
    """Compare native impurity output against the accepted Aurora reference."""
    case = _load_impurity_reference_case()
    artifact_path, reference, blockers = _load_reference_npz(case)
    if blockers:
        return {
            "schema": "aurora-strahl-native-same-case-comparison.v1",
            "status": "blocked_reference_artifact_unavailable",
            "artifact_path": str(artifact_path.relative_to(ROOT)) if artifact_path else None,
            "comparison_ready": False,
            "threshold_checks_ready": False,
            "thresholds_passed": False,
            "checks": [],
            "blocking_requirements": blockers,
        }
    native_payload = _native_same_case_payload(reference)
    candidate = _native_observables_as_top_level(native_payload)
    checks = _same_case_threshold_checks(candidate, reference, case)
    checks_ready = bool(checks) and all(bool(check["valid"]) for check in checks)
    thresholds_passed = checks_ready and all(bool(check["passed"]) for check in checks)
    density = candidate["charge_state_density_r_t"]
    total_density = candidate["total_impurity_density_r_t"]
    density_closure = bool(np.allclose(total_density, np.sum(density, axis=2), rtol=1.0e-10))
    effective_closure_ready = "effective_source_m3_s_t_r_z" in reference
    source_sink_matrix_ready = "source_sink_matrix_t_r_z_z" in reference
    source_sink_matrix_passed = any(
        check.get("threshold") == "source_sink_matrix_relative_l2_max"
        and bool(check.get("valid"))
        and bool(check.get("passed"))
        for check in checks
    )
    blocking_requirements: list[str] = []
    if not thresholds_passed:
        blocking_requirements.insert(
            0,
            "native Aurora same-case thresholds are outside accepted limits",
        )
    status = (
        "accepted_native_aurora_effective_transport_closure_thresholds"
        if thresholds_passed and effective_closure_ready
        else "accepted_native_aurora_same_case_thresholds"
        if thresholds_passed
        else "blocked_native_aurora_same_case_threshold_mismatch"
    )
    return {
        "schema": "aurora-strahl-native-same-case-comparison.v1",
        "status": status,
        "artifact_path": str(artifact_path.relative_to(ROOT)) if artifact_path else None,
        "case_id": case.get("case_id"),
        "reference_family": case.get("reference_family"),
        "comparison_ready": True,
        "threshold_checks_ready": checks_ready,
        "thresholds_passed": thresholds_passed,
        "native_density_shape": [int(axis) for axis in density.shape],
        "reference_density_shape": [int(axis) for axis in reference["charge_state_density_r_t"].shape],
        "native_total_density_closure": density_closure,
        "native_coordinate_match": bool(
            np.array_equal(candidate["time_s"], reference["time_s"])
            and np.array_equal(candidate["radius_m"], reference["radius_m"])
            and np.array_equal(candidate["charge_state"], reference["charge_state"])
        ),
        "external_coefficient_tables_ready": bool(
            "ionisation_coeff_m3_s_t_r_z" in reference
            and "recombination_coeff_m3_s_t_r_z" in reference
            and "line_radiation_coeff_w_m3_t_r_z" in reference
        ),
        "aurora_case_profiles_ready": bool(
            "electron_density_t_r_m3" in reference
            and "electron_temperature_t_r_ev" in reference
            and "diffusion_m2_s_r_z" in reference
            and "convection_m_s_r_z" in reference
        ),
        "effective_source_recycling_closure_ready": effective_closure_ready,
        "source_sink_matrix_parity_ready": bool(
            source_sink_matrix_ready and source_sink_matrix_passed
        ),
        "checks": checks,
        "blocking_requirements": blocking_requirements,
    }


def _native_impurity_transport_evidence(
    payload: dict[str, Any],
    artifact_validation: dict[str, Any],
    required_observables: list[str],
    same_case_comparison: dict[str, Any],
) -> dict[str, Any]:
    observables = payload["observables"]
    density = np.asarray(observables["charge_state_density_r_t"], dtype=np.float64)
    source_sink = np.asarray(observables["source_sink_matrix_t_r_z_z"], dtype=np.float64)
    line_radiation = np.asarray(observables["line_radiation_power_t_r_z"], dtype=np.float64)
    row_sum_abs_max = float(np.max(np.abs(np.sum(source_sink, axis=3))))
    radial_operator_ready = bool(
        same_case_comparison["comparison_ready"]
        and same_case_comparison.get("aurora_case_profiles_ready", False)
    )
    effective_closure_ready = bool(
        same_case_comparison.get("effective_source_recycling_closure_ready", False)
    )
    external_coefficients_ready = bool(
        same_case_comparison.get("external_coefficient_tables_ready", False)
    )
    source_sink_parity_ready = bool(
        same_case_comparison.get("source_sink_matrix_parity_ready", False)
    )
    operator_status = (
        "accepted_native_effective_transport_source_sink_closure"
        if radial_operator_ready
        and effective_closure_ready
        and external_coefficients_ready
        and source_sink_parity_ready
        and same_case_comparison["thresholds_passed"]
        else "blocked_native_charge_state_contract_not_full_aurora_strahl_transport_operator"
    )
    source_sink_budget_evidence = _source_sink_budget_evidence(payload)
    if source_sink_parity_ready:
        source_sink_budget_evidence = {
            **source_sink_budget_evidence,
            "aurora_strahl_same_case_budget_ready": True,
            "blocking_requirements": [],
            "status": "accepted_native_same_case_source_sink_budget_parity",
        }

    return {
        "schema": "native-impurity-transport-operator-evidence.v1",
        "operator_evidence_status": operator_status,
        "density_axes": ["time_s", "radius_m", "charge_state"],
        "density_shape": [int(value) for value in density.shape],
        "source_sink_shape": [int(value) for value in source_sink.shape],
        "line_radiation_shape": [int(value) for value in line_radiation.shape],
        "native_artifact_ready": bool(artifact_validation["passed"]),
        "charge_state_density_closure": bool(artifact_validation["density_closure"]),
        "source_sink_conservative": bool(artifact_validation["source_sink_conservative"]),
        "source_sink_row_sum_abs_max": row_sum_abs_max,
        "inventory_conserved": bool(artifact_validation["inventory_conserved"]),
        "charge_state_radial_transport_operator_ready": radial_operator_ready,
        "aurora_strahl_same_case_comparison_ready": bool(
            same_case_comparison["comparison_ready"]
        ),
        "aurora_strahl_same_case_threshold_ready": bool(
            same_case_comparison["threshold_checks_ready"]
        ),
        "aurora_strahl_same_case_threshold_passed": bool(
            same_case_comparison["thresholds_passed"]
        ),
        "operator_terms_present": {
            "trace_radial_transport": True,
            "edge_source_particle_conservation": True,
            "neoclassical_pinch": True,
            "charge_state_source_sink_matrix": True,
            "line_radiation_power": True,
            "total_impurity_inventory_closure": True,
            "charge_state_resolved_radial_transport": radial_operator_ready,
            "external_adas_transport_coefficients": bool(
                external_coefficients_ready
            ),
            "aurora_effective_source_recycling_closure": bool(
                effective_closure_ready
            ),
            "same_case_aurora_strahl_transport_output": bool(
                same_case_comparison["comparison_ready"]
            ),
            "time_resolved_same_case_source_sink_matrix_parity": source_sink_parity_ready,
            "aurora_strahl_collisional_operator_parity": False,
        },
        "observable_finiteness": _observable_finiteness(payload, required_observables),
        "source_sink_budget_evidence": source_sink_budget_evidence,
        "same_case_aurora_strahl_comparison": same_case_comparison,
        "blocking_requirements": [
            "independent mechanistic Aurora/STRAHL recycling validation beyond effective closure replay",
        ],
    }


def run_benchmark() -> dict[str, Any]:
    """Run impurity transport contract checks and return invariant summary."""
    rho = np.linspace(0.0, 1.0, 80, dtype=np.float64)
    R0 = 6.2
    a = 2.0
    dt = 0.2
    source_rate = 1.0e16
    ne = 1.0e20 * (1.0 - 0.2 * rho**2)
    Te = 1500.0 * (1.0 - 0.3 * rho**2)
    Ti = 5000.0 * (1.0 - 0.6 * rho**2)
    q = 1.0 + rho
    eps = 0.2 + 0.2 * rho

    solver = ImpurityTransportSolver(
        rho,
        R0,
        a,
        [ImpuritySpecies("W", 74, 183.8, source_rate=source_rate)],
    )
    pinch = neoclassical_impurity_pinch(74, ne, Te, Ti, q, rho, R0, a, eps)
    result = solver.step(dt, ne, Te, Ti, D_anom=0.0, V_pinch={"W": np.zeros_like(rho)})
    n_w = np.asarray(result["W"], dtype=np.float64)

    expected_particles = source_rate * (4.0 * np.pi**2 * R0 * a) * dt
    actual_particles = _inventory(n_w, rho, R0, a)
    conservation_error = abs(actual_particles - expected_particles) / expected_particles

    low_rad = total_radiated_power(ne, {"W": ne * 1.0e-5}, Te, rho, R0, a)
    high_rad = total_radiated_power(ne, {"W": ne * 1.0e-4}, Te, rho, R0, a)
    radius_m = rho * a
    time_s = np.array([0.0, 1.0e-5, 2.0e-5], dtype=float)
    charge_states = np.array([0, 1, 2, 3], dtype=float)
    ne_t_r = np.tile(ne, (time_s.size, 1))
    Te_t_r = np.tile(Te, (time_s.size, 1))
    density_r_z = np.zeros((rho.size, charge_states.size), dtype=float)
    density_r_z[:, 1] = 1.0e15 * (1.0 - 0.1 * rho)
    cr_artifact = build_aurora_strahl_charge_state_artifact(
        element="Ar",
        charge_states=charge_states,
        radius_m=radius_m,
        time_s=time_s,
        ne_t_r=ne_t_r,
        Te_t_r=Te_t_r,
        initial_charge_state_density_rz=density_r_z,
        major_radius_m=R0,
    )
    radial_budget_diagnostic = AuroraParityImpuritySolver(
        AuroraParityCase(
            element="Ar",
            charge_states=charge_states,
            radius_m=radius_m,
            time_s=time_s,
            ne_t_r=ne_t_r,
            Te_t_r=Te_t_r,
            initial_charge_state_density_rz=density_r_z,
            diffusion_m2_s_r_z=np.full(
                (radius_m.size, charge_states.size),
                0.2,
                dtype=np.float64,
            ),
            convection_m_s_r_z=np.zeros(
                (radius_m.size, charge_states.size),
                dtype=np.float64,
            ),
            major_radius_m=R0,
        )
    ).radial_transport_budget_diagnostic(density_r_z, float(time_s[1] - time_s[0]))
    cr_payload = cr_artifact.to_dict()
    artifact_validation = cr_artifact.validate_contract()
    charge_density = np.asarray(cr_payload["observables"]["charge_state_density_r_t"])
    total_density = np.asarray(cr_payload["observables"]["total_impurity_density_r_t"])
    source_sink = np.asarray(cr_payload["observables"]["source_sink_matrix_t_r_z_z"])
    line_power_t_r_z = np.asarray(cr_payload["observables"]["line_radiation_power_t_r_z"])
    required_observables = [
        "charge_state_density_r_t",
        "total_impurity_density_r_t",
        "line_radiation_power_t",
        "line_radiation_power_t_r_z",
        "source_sink_matrix_t_r_z_z",
        "total_impurity_inventory_t",
    ]
    canonical_payload = json.dumps(cr_payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    same_case_comparison = _aurora_same_case_comparison()
    native_impurity_transport_evidence = _native_impurity_transport_evidence(
        cr_payload,
        artifact_validation,
        required_observables,
        same_case_comparison,
    )
    source_sink_budget_evidence = native_impurity_transport_evidence["source_sink_budget_evidence"]

    invariants = {
        "positivity": bool(np.all(n_w >= 0.0) and np.all(np.isfinite(n_w))),
        "edge_source_conservation": bool(conservation_error <= 2.0e-2),
        "inward_pinch_midradius": bool(pinch[len(rho) // 2] < 0.0),
        "radiation_monotonicity": bool(high_rad > low_rad > 0.0),
        "charge_state_artifact_contract": bool(
            cr_payload["schema"] == "aurora-strahl-charge-state-artifact.v1"
            and charge_density.shape == (time_s.size, rho.size, charge_states.size)
            and total_density.shape == (time_s.size, rho.size)
            and np.all(np.isfinite(charge_density))
        ),
        "charge_state_density_closure": bool(
            np.allclose(total_density, np.sum(charge_density, axis=2), rtol=1.0e-13)
        ),
        "charge_state_particle_conservation": bool(
            cr_artifact.conservation["relative_inventory_error"] <= 1.0e-12
        ),
        "source_sink_matrix_conservative": bool(artifact_validation["source_sink_conservative"]),
        "line_radiation_power_finite": bool(
            np.all(np.isfinite(line_power_t_r_z)) and np.all(line_power_t_r_z >= 0.0)
        ),
        "native_impurity_transport_evidence_fail_closed": bool(
            native_impurity_transport_evidence["native_artifact_ready"]
            and native_impurity_transport_evidence[
                "charge_state_radial_transport_operator_ready"
            ]
            and native_impurity_transport_evidence["aurora_strahl_same_case_threshold_ready"]
            and bool(native_impurity_transport_evidence["blocking_requirements"])
        ),
        "native_source_sink_budget_evidence_fail_closed": bool(
            source_sink_budget_evidence["all_budget_terms_finite"]
            and source_sink_budget_evidence["ionisation_recombination_nonnegative"]
            and source_sink_budget_evidence["source_sink_transfer_conservative"]
            and source_sink_budget_evidence["radial_total_density_conserved"]
            and source_sink_budget_evidence["line_radiation_nonnegative"]
            and source_sink_budget_evidence["aurora_strahl_same_case_budget_ready"]
        ),
        "charge_state_radial_density_conservation": bool(
            source_sink_budget_evidence["radial_total_density_conserved"]
        ),
        "charge_state_radial_transport_operator_budget": bool(
            radial_budget_diagnostic["passed"]
        ),
    }

    return {
        "benchmark": "impurity_transport_contract",
        "description": "Trace impurity transport contract; not Aurora/STRAHL/JINTRAC collisional-operator parity.",
        "metrics": {
            "actual_particles": actual_particles,
            "expected_particles": expected_particles,
            "relative_conservation_error": conservation_error,
            "midradius_pinch_m_s": float(pinch[len(rho) // 2]),
            "low_radiated_power_mw": low_rad,
            "high_radiated_power_mw": high_rad,
            "edge_density_m3": float(n_w[-1]),
            "charge_state_inventory_error": cr_artifact.conservation["relative_inventory_error"],
            "charge_state_count": int(charge_states.size),
            "radial_transport_inventory_relative_error": float(
                radial_budget_diagnostic["relative_inventory_error"]
            ),
        },
        "thresholds": {
            "max_relative_conservation_error": 2.0e-2,
            "max_charge_state_inventory_error": 1.0e-12,
        },
        "artifact_contract": {
            "artifact_sha256": hashlib.sha256(canonical_payload).hexdigest(),
            "contract_validation": artifact_validation,
            "schema": cr_payload["schema"],
            "coordinates": list(cr_payload["coordinates"].keys()),
            "observables": list(cr_payload["observables"].keys()),
            "observable_shapes": artifact_validation["observable_shapes"],
            "parity_status": cr_payload["provenance"]["parity_status"],
            "required_aurora_strahl_observables": required_observables,
            "same_case_aurora_strahl_comparison_ready": bool(
                same_case_comparison["comparison_ready"]
            ),
        },
        "native_impurity_transport_evidence": native_impurity_transport_evidence,
        "invariants": invariants,
        "passed": all(invariants.values()),
    }


def write_reports(results: dict[str, Any]) -> None:
    """Write JSON and markdown artifacts for impurity transport benchmark."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics = results["metrics"]
    thresholds = results["thresholds"]
    invariants = results["invariants"]
    evidence = results["native_impurity_transport_evidence"]
    budget = evidence["source_sink_budget_evidence"]
    same_case = evidence["same_case_aurora_strahl_comparison"]
    lines = [
        "# Impurity Transport Contract Benchmark",
        "",
        "This benchmark validates native trace impurity transport contracts.",
        "It does not claim Aurora/STRAHL/JINTRAC collisional-operator parity.",
        "",
        "## Metrics",
        "",
        f"- Actual particles: {metrics['actual_particles']:.6e}",
        f"- Expected particles: {metrics['expected_particles']:.6e}",
        (
            "- Relative conservation error: "
            f"{metrics['relative_conservation_error']:.6e} "
            f"(threshold {thresholds['max_relative_conservation_error']:.2e})"
        ),
        f"- Midradius pinch: {metrics['midradius_pinch_m_s']:.6e} m/s",
        f"- Low radiated power: {metrics['low_radiated_power_mw']:.6e} MW",
        f"- High radiated power: {metrics['high_radiated_power_mw']:.6e} MW",
        f"- Edge density: {metrics['edge_density_m3']:.6e} m^-3",
        f"- Charge-state count: {metrics['charge_state_count']}",
        (
            "- Charge-state inventory error: "
            f"{metrics['charge_state_inventory_error']:.6e} "
            f"(threshold {thresholds['max_charge_state_inventory_error']:.2e})"
        ),
        "",
        "## Aurora/STRAHL-style artifact contract",
        "",
        f"- Schema: `{results['artifact_contract']['schema']}`",
        f"- Coordinates: {', '.join(results['artifact_contract']['coordinates'])}",
        f"- Observables: {', '.join(results['artifact_contract']['observables'])}",
        f"- Parity status: `{results['artifact_contract']['parity_status']}`",
        f"- SHA-256: `{results['artifact_contract']['artifact_sha256']}`",
        f"- Contract validation passed: `{results['artifact_contract']['contract_validation']['passed']}`",
        f"- Same-case Aurora/STRAHL comparison ready: `{results['artifact_contract']['same_case_aurora_strahl_comparison_ready']}`",
        f"- Observable shapes: `{json.dumps(results['artifact_contract']['observable_shapes'], sort_keys=True)}`",
        (
            "- Required Aurora/STRAHL observables: "
            f"`{', '.join(results['artifact_contract']['required_aurora_strahl_observables'])}`"
        ),
        "",
        "## Native impurity transport operator evidence",
        "",
        f"- Schema: `{evidence['schema']}`",
        f"- Status: `{evidence['operator_evidence_status']}`",
        f"- Native artifact ready: `{evidence['native_artifact_ready']}`",
        (
            "- Charge-state radial transport operator ready: "
            f"`{evidence['charge_state_radial_transport_operator_ready']}`"
        ),
        (
            "- Aurora/STRAHL same-case comparison ready: "
            f"`{evidence['aurora_strahl_same_case_comparison_ready']}`"
        ),
        (
            "- Aurora/STRAHL same-case threshold checks ready: "
            f"`{evidence['aurora_strahl_same_case_threshold_ready']}`"
        ),
        (
            "- Aurora/STRAHL same-case thresholds passed: "
            f"`{evidence['aurora_strahl_same_case_threshold_passed']}`"
        ),
        f"- Density axes: `{', '.join(evidence['density_axes'])}`",
        f"- Density shape: `{json.dumps(evidence['density_shape'])}`",
        f"- Source-sink shape: `{json.dumps(evidence['source_sink_shape'])}`",
        f"- Line-radiation shape: `{json.dumps(evidence['line_radiation_shape'])}`",
        (
            "- Operator terms present: "
            f"`{json.dumps(evidence['operator_terms_present'], sort_keys=True)}`"
        ),
        (
            "- Observable finiteness: "
            f"`{json.dumps(evidence['observable_finiteness'], sort_keys=True)}`"
        ),
        "",
        "## Native same-case Aurora comparison",
        "",
        f"- Schema: `{same_case['schema']}`",
        f"- Status: `{same_case['status']}`",
        f"- Artefact path: `{same_case['artifact_path']}`",
        f"- Comparison ready: `{same_case['comparison_ready']}`",
        f"- Threshold checks ready: `{same_case['threshold_checks_ready']}`",
        f"- Thresholds passed: `{same_case['thresholds_passed']}`",
        f"- Native coordinate match: `{same_case.get('native_coordinate_match', False)}`",
        f"- Native total-density closure: `{same_case.get('native_total_density_closure', False)}`",
        f"- Checks: `{json.dumps(same_case['checks'], sort_keys=True)}`",
        f"- Blocking requirements: `{'; '.join(same_case['blocking_requirements'])}`",
        "",
        "## Native source/sink budget evidence",
        "",
        f"- Schema: `{budget['schema']}`",
        f"- Status: `{budget['status']}`",
        f"- Budget terms: `{', '.join(budget['budget_terms'])}`",
        f"- Time count: `{budget['time_count']}`",
        f"- Radius count: `{budget['radius_count']}`",
        f"- Charge-state count: `{budget['charge_state_count']}`",
        f"- All budget terms finite: `{budget['all_budget_terms_finite']}`",
        (
            "- Ionisation/recombination non-negative: "
            f"`{budget['ionisation_recombination_nonnegative']}`"
        ),
        (f"- Source/sink transfer conservative: `{budget['source_sink_transfer_conservative']}`"),
        f"- Radial total-density conserved: `{budget['radial_total_density_conserved']}`",
        (
            "- Max radial total-density relative change: "
            f"`{budget['max_radial_total_density_relative_change']:.6e}`"
        ),
        f"- Line radiation non-negative: `{budget['line_radiation_nonnegative']}`",
        (
            "- Aurora/STRAHL same-case budget ready: "
            f"`{budget['aurora_strahl_same_case_budget_ready']}`"
        ),
        f"- Blocking requirements: `{'; '.join(evidence['blocking_requirements'])}`",
        "",
        "## Invariants",
        "",
    ]
    for name, passed in invariants.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", f"Overall: {'PASS' if results['passed'] else 'FAIL'}", ""])
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Execute impurity transport contract benchmark and print JSON report."""
    results = run_benchmark()
    write_reports(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
