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
    ImpuritySpecies,
    ImpurityTransportSolver,
    build_aurora_strahl_charge_state_artifact,
    neoclassical_impurity_pinch,
    total_radiated_power,
)

REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "impurity_transport_contract_benchmark.json"
MD_REPORT = REPORT_DIR / "impurity_transport_contract_benchmark.md"
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
            "external ADAS coefficient provenance for transport parity",
        ],
    }


def _native_impurity_transport_evidence(
    payload: dict[str, Any],
    artifact_validation: dict[str, Any],
    required_observables: list[str],
) -> dict[str, Any]:
    observables = payload["observables"]
    density = np.asarray(observables["charge_state_density_r_t"], dtype=np.float64)
    source_sink = np.asarray(observables["source_sink_matrix_t_r_z_z"], dtype=np.float64)
    line_radiation = np.asarray(observables["line_radiation_power_t_r_z"], dtype=np.float64)
    row_sum_abs_max = float(np.max(np.abs(np.sum(source_sink, axis=3))))

    return {
        "schema": "native-impurity-transport-operator-evidence.v1",
        "operator_evidence_status": (
            "blocked_native_charge_state_contract_not_full_aurora_strahl_transport_operator"
        ),
        "density_axes": ["time_s", "radius_m", "charge_state"],
        "density_shape": [int(value) for value in density.shape],
        "source_sink_shape": [int(value) for value in source_sink.shape],
        "line_radiation_shape": [int(value) for value in line_radiation.shape],
        "native_artifact_ready": bool(artifact_validation["passed"]),
        "charge_state_density_closure": bool(artifact_validation["density_closure"]),
        "source_sink_conservative": bool(artifact_validation["source_sink_conservative"]),
        "source_sink_row_sum_abs_max": row_sum_abs_max,
        "inventory_conserved": bool(artifact_validation["inventory_conserved"]),
        "charge_state_radial_transport_operator_ready": False,
        "aurora_strahl_same_case_threshold_ready": False,
        "operator_terms_present": {
            "trace_radial_transport": True,
            "edge_source_particle_conservation": True,
            "neoclassical_pinch": True,
            "charge_state_source_sink_matrix": True,
            "line_radiation_power": True,
            "total_impurity_inventory_closure": True,
            "charge_state_resolved_radial_transport": False,
            "external_adas_transport_coefficients": False,
            "same_case_aurora_strahl_transport_output": False,
            "aurora_strahl_collisional_operator_parity": False,
        },
        "observable_finiteness": _observable_finiteness(payload, required_observables),
        "source_sink_budget_evidence": _source_sink_budget_evidence(payload),
        "blocking_requirements": [
            "public Aurora or STRAHL radial transport output",
            "charge-state-resolved radial transport operator on evolved density",
            "external ADAS coefficient ingestion for transport parity",
            "same-case line-radiation output from Aurora or STRAHL",
            "same-case ionisation/recombination source-sink matrix output",
            "native same-case solver-output comparison",
            "distribution, radiation, and inventory threshold comparison against Aurora/STRAHL",
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
    native_impurity_transport_evidence = _native_impurity_transport_evidence(
        cr_payload,
        artifact_validation,
        required_observables,
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
            and not native_impurity_transport_evidence[
                "charge_state_radial_transport_operator_ready"
            ]
            and not native_impurity_transport_evidence["aurora_strahl_same_case_threshold_ready"]
            and bool(native_impurity_transport_evidence["blocking_requirements"])
        ),
        "native_source_sink_budget_evidence_fail_closed": bool(
            source_sink_budget_evidence["all_budget_terms_finite"]
            and source_sink_budget_evidence["ionisation_recombination_nonnegative"]
            and source_sink_budget_evidence["source_sink_transfer_conservative"]
            and source_sink_budget_evidence["radial_total_density_conserved"]
            and source_sink_budget_evidence["line_radiation_nonnegative"]
            and not source_sink_budget_evidence["aurora_strahl_same_case_budget_ready"]
            and bool(source_sink_budget_evidence["blocking_requirements"])
        ),
        "charge_state_radial_density_conservation": bool(
            source_sink_budget_evidence["radial_total_density_conserved"]
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
            "same_case_aurora_strahl_comparison_ready": False,
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
            "- Aurora/STRAHL same-case thresholds ready: "
            f"`{evidence['aurora_strahl_same_case_threshold_ready']}`"
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
