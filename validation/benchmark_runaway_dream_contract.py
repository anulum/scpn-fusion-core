#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Benchmark DREAM-style runaway-electron contracts.

This validates scalar density-balance invariants compatible with DREAM fluid
runs: subcritical avalanche suppression, supercritical avalanche growth,
mitigation loss accounting, and density-cap enforcement. It does not claim
parity with DREAM's kinetic momentum-space distribution solver.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, TypedDict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_fusion.core.runaway_electrons import (  # noqa: E402
    RunawayEvolution,
    RunawayParams,
    critical_field,
    dream_fluid_density_balance,
)
from scpn_fusion.control.fokker_planck_re import FokkerPlanckSolver  # noqa: E402

REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "runaway_dream_contract_benchmark.json"
MD_REPORT = REPORT_DIR / "runaway_dream_contract_benchmark.md"


class _CaseResult(TypedDict):
    case: str
    dreicer_source_m3_s: float
    avalanche_source_m3_s: float
    loss_source_m3_s: float
    total_source_m3_s: float
    runaway_fraction: float
    growth_time_s: float | None
    wall_time_s: float


class _RunawayBenchmarkResult(TypedDict):
    benchmark: str
    description: str
    cases: list[_CaseResult]
    native_kinetic_artifact: dict[str, Any]
    native_kinetic_operator_evidence: dict[str, Any]
    timing: dict[str, float]
    invariants: dict[str, bool]
    passed: bool


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _case_result(name: str, params: RunawayParams, n_re: float, loss_time_s: float) -> _CaseResult:
    start = time.perf_counter()
    balance = dream_fluid_density_balance(params, n_re, loss_time_s=loss_time_s)
    wall_time_s = time.perf_counter() - start
    return {
        "case": name,
        "dreicer_source_m3_s": balance.dreicer_source,
        "avalanche_source_m3_s": balance.avalanche_source,
        "loss_source_m3_s": balance.loss_source,
        "total_source_m3_s": balance.total_source,
        "runaway_fraction": balance.runaway_fraction,
        "growth_time_s": _json_float(balance.growth_time_s),
        "wall_time_s": wall_time_s,
    }


def _finite_nonnegative_observables(
    payload: dict[str, Any], required_observables: list[str]
) -> tuple[dict[str, bool], dict[str, bool]]:
    observables = payload["observables"]
    finiteness: dict[str, bool] = {}
    nonnegativity: dict[str, bool] = {}
    for name in required_observables:
        values = np.asarray(observables.get(name, []), dtype=np.float64)
        finiteness[name] = bool(values.size > 0 and np.all(np.isfinite(values)))
        nonnegativity[name] = bool(values.size > 0 and np.all(values >= 0.0))
    return finiteness, nonnegativity


def _relative_inventory_change(distribution: np.ndarray[Any, np.dtype[np.float64]]) -> float | None:
    if distribution.ndim != 4 or distribution.shape[0] < 2:
        return None
    inventory_t_radius = np.sum(distribution, axis=(2, 3))
    baseline = np.maximum(np.abs(inventory_t_radius[0]), 1.0)
    relative_change = float(np.max(np.abs(inventory_t_radius - inventory_t_radius[0]) / baseline))
    return relative_change if np.isfinite(relative_change) else None


def _native_kinetic_operator_evidence(
    payload: dict[str, Any],
    validation: dict[str, Any],
    required_observables: list[str],
) -> dict[str, Any]:
    observables = payload["observables"]
    distribution = np.asarray(observables["f_p_xi_t"], dtype=np.float64)
    observable_finiteness, observable_nonnegativity = _finite_nonnegative_observables(
        payload, required_observables
    )

    return {
        "schema": "native-runaway-kinetic-operator-evidence.v1",
        "operator_evidence_status": "blocked_native_projection_artifact_not_full_dream_operator",
        "distribution_axes": ["time_s", "radius_m", "momentum_mec", "pitch_cosine"],
        "distribution_shape": [int(value) for value in distribution.shape],
        "native_artifact_ready": bool(validation["passed"]),
        "radius_pitch_are_evolved_operator_axes": False,
        "full_momentum_pitch_radius_operator_ready": False,
        "dream_same_case_threshold_ready": False,
        "operator_terms_present": {
            "momentum_advection_drag": True,
            "momentum_diffusion": True,
            "dreicer_source": True,
            "avalanche_growth": True,
            "synchrotron_radiation_reaction": True,
            "full_pitch_angle_scattering_operator": False,
            "full_radial_transport_operator": False,
            "partial_screening_dream_operator": False,
            "bremsstrahlung_radiation_loss_operator": False,
            "coupled_momentum_pitch_radius_operator": False,
        },
        "observable_finiteness": observable_finiteness,
        "observable_nonnegativity": observable_nonnegativity,
        "unweighted_inventory_relative_change_max": _relative_inventory_change(distribution),
        "blocking_requirements": [
            "compiled DREAM iface/dreami same-case output",
            "native coupled momentum-pitch-radius Fokker-Planck operator",
            "radial transport operator on evolved radius grid",
            "full pitch-angle scattering operator on evolved pitch grid",
            "DREAM partial-screening operator parity",
            "DREAM bremsstrahlung and synchrotron loss parity",
            "distribution, current, and growth-rate threshold comparison against DREAM",
        ],
    }


def _native_kinetic_artifact_gate() -> dict[str, Any]:
    solver = FokkerPlanckSolver(np_grid=48, p_max=12.0)
    solver.f[12] = 2.5e10
    artifact = solver.run_dream_kinetic_artifact(
        n_steps=4,
        dt=1.0e-6,
        e_field=10.0,
        n_e=5.0e19,
        t_e_ev=2500.0,
        z_eff=2.0,
        radius_m=[0.0, 0.35, 0.70, 1.05],
        pitch_cosine=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    payload = artifact.to_dict()
    validation = artifact.validate_contract()
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    required_observables = [
        "f_p_xi_t",
        "runaway_current_t",
        "avalanche_growth_rate_t",
        "synchrotron_loss_power_t",
        "partial_screening_drag_t",
        "bremsstrahlung_loss_power_t",
    ]
    operator_evidence = _native_kinetic_operator_evidence(payload, validation, required_observables)
    return {
        "artifact_sha256": hashlib.sha256(canonical_payload).hexdigest(),
        "contract_validation": validation,
        "coordinate_lengths": validation["coordinate_lengths"],
        "observable_shapes": validation["observable_shapes"],
        "parity_status": "native_contract_only_not_dream_parity",
        "required_dream_observables": required_observables,
        "same_case_dream_comparison_ready": False,
        "schema": payload["schema"],
        "kinetic_operator_evidence": operator_evidence,
    }


def run_benchmark(repeats: int = 25) -> _RunawayBenchmarkResult:
    """Run DREAM-style scalar balance and native kinetic artifact contracts."""
    params_subcritical = RunawayParams(
        ne_20=1.0,
        Te_keV=0.04,
        E_par=0.5 * critical_field(1.0),
        Z_eff=2.0,
        B0=5.0,
        R0=6.0,
    )
    params_supercritical = RunawayParams(
        ne_20=1.0,
        Te_keV=0.04,
        E_par=8.0,
        Z_eff=2.0,
        B0=5.0,
        R0=6.0,
    )

    cases = [
        _case_result("subcritical_no_avalanche", params_subcritical, 1.0e12, np.inf),
        _case_result("supercritical_growth", params_supercritical, 2.0e12, np.inf),
        _case_result("mitigated_loss_accounting", params_supercritical, 2.0e12, 0.2),
    ]

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        dream_fluid_density_balance(params_supercritical, 2.0e12, loss_time_s=0.2)
        timings.append(time.perf_counter() - start)

    evo = RunawayEvolution(params_supercritical)
    capped_density = evo.step(1.0, 9.0e13, 50.0, max_runaway_fraction=1.0e-6)
    native_kinetic_artifact = _native_kinetic_artifact_gate()
    native_kinetic_operator_evidence = native_kinetic_artifact["kinetic_operator_evidence"]

    invariants = {
        "subcritical_avalanche_zero": bool(cases[0]["avalanche_source_m3_s"] == 0.0),
        "supercritical_avalanche_positive": bool(cases[1]["avalanche_source_m3_s"] > 0.0),
        "loss_accounting_exact": bool(
            np.isclose(
                cases[2]["total_source_m3_s"],
                cases[2]["dreicer_source_m3_s"]
                + cases[2]["avalanche_source_m3_s"]
                - cases[2]["loss_source_m3_s"],
            )
        ),
        "density_cap_enforced": bool(np.isclose(capped_density, 1.0e14)),
        "native_kinetic_artifact_contract": bool(
            native_kinetic_artifact["contract_validation"]["passed"]
        ),
        "native_kinetic_operator_evidence_fail_closed": bool(
            native_kinetic_operator_evidence["native_artifact_ready"]
            and not native_kinetic_operator_evidence["full_momentum_pitch_radius_operator_ready"]
            and not native_kinetic_operator_evidence["dream_same_case_threshold_ready"]
            and bool(native_kinetic_operator_evidence["blocking_requirements"])
        ),
    }

    return {
        "benchmark": "runaway_dream_contract",
        "description": (
            "DREAM-style fluid density-balance plus native momentum-pitch-radius "
            "artifact contract; not kinetic DREAM parity."
        ),
        "cases": cases,
        "native_kinetic_artifact": native_kinetic_artifact,
        "native_kinetic_operator_evidence": native_kinetic_operator_evidence,
        "timing": {
            "repeats": repeats,
            "median_balance_wall_time_s": median(timings),
            "min_balance_wall_time_s": min(timings),
            "max_balance_wall_time_s": max(timings),
        },
        "invariants": invariants,
        "passed": all(invariants.values()),
    }


def write_reports(results: _RunawayBenchmarkResult) -> None:
    """Write JSON and markdown reports for the runaway contract benchmark."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cases = results["cases"]
    timing = results["timing"]
    invariants = results["invariants"]
    artifact = results["native_kinetic_artifact"]
    operator_evidence = results["native_kinetic_operator_evidence"]
    lines = [
        "# Runaway DREAM-Style Contract Benchmark",
        "",
        "This benchmark validates scalar runaway-density balance contracts compatible with DREAM fluid runs plus a native DREAM-style kinetic artifact contract.",
        "It does not claim parity with DREAM's kinetic momentum-space distribution solver.",
        "",
        "## Timing",
        "",
        f"- Repeats: {timing['repeats']}",
        f"- Median balance wall time: {timing['median_balance_wall_time_s']:.6e} s",
        f"- Minimum balance wall time: {timing['min_balance_wall_time_s']:.6e} s",
        f"- Maximum balance wall time: {timing['max_balance_wall_time_s']:.6e} s",
        "",
        "## Cases",
        "",
        "| Case | Dreicer source [m^-3 s^-1] | Avalanche source [m^-3 s^-1] | Loss source [m^-3 s^-1] | Total source [m^-3 s^-1] | Runaway fraction | Growth time [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        growth_time = case["growth_time_s"]
        growth_time_text = "inf" if growth_time is None else f"{growth_time:.6e}"
        lines.append(
            f"| {case['case']} | {case['dreicer_source_m3_s']:.6e} | "
            f"{case['avalanche_source_m3_s']:.6e} | {case['loss_source_m3_s']:.6e} | "
            f"{case['total_source_m3_s']:.6e} | {case['runaway_fraction']:.6e} | "
            f"{growth_time_text} |"
        )
    lines.extend(["", "## Invariants", ""])
    for name, passed in invariants.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Native kinetic artifact contract",
            "",
            f"- Schema: `{artifact['schema']}`",
            f"- SHA-256: `{artifact['artifact_sha256']}`",
            f"- Parity status: `{artifact['parity_status']}`",
            f"- Same-case DREAM comparison ready: `{artifact['same_case_dream_comparison_ready']}`",
            f"- Contract validation passed: `{artifact['contract_validation']['passed']}`",
            f"- Coordinate lengths: `{json.dumps(artifact['coordinate_lengths'], sort_keys=True)}`",
            f"- Observable shapes: `{json.dumps(artifact['observable_shapes'], sort_keys=True)}`",
            f"- Required DREAM observables: `{', '.join(artifact['required_dream_observables'])}`",
            "",
            "## Native kinetic operator evidence",
            "",
            f"- Schema: `{operator_evidence['schema']}`",
            f"- Status: `{operator_evidence['operator_evidence_status']}`",
            f"- Native artifact ready: `{operator_evidence['native_artifact_ready']}`",
            (
                "- Full momentum-pitch-radius operator ready: "
                f"`{operator_evidence['full_momentum_pitch_radius_operator_ready']}`"
            ),
            (
                "- DREAM same-case thresholds ready: "
                f"`{operator_evidence['dream_same_case_threshold_ready']}`"
            ),
            (
                "- Evolved radius/pitch operator axes: "
                f"`{operator_evidence['radius_pitch_are_evolved_operator_axes']}`"
            ),
            (f"- Distribution axes: `{', '.join(operator_evidence['distribution_axes'])}`"),
            (f"- Distribution shape: `{json.dumps(operator_evidence['distribution_shape'])}`"),
            (
                "- Operator terms present: "
                f"`{json.dumps(operator_evidence['operator_terms_present'], sort_keys=True)}`"
            ),
            (
                "- Observable finiteness: "
                f"`{json.dumps(operator_evidence['observable_finiteness'], sort_keys=True)}`"
            ),
            (
                "- Observable non-negativity: "
                f"`{json.dumps(operator_evidence['observable_nonnegativity'], sort_keys=True)}`"
            ),
            (f"- Blocking requirements: `{'; '.join(operator_evidence['blocking_requirements'])}`"),
        ]
    )
    lines.extend(["", f"Overall: {'PASS' if results['passed'] else 'FAIL'}", ""])
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Execute benchmark, persist outputs, and return contract pass code."""
    results = run_benchmark()
    write_reports(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
