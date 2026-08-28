#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Differentiable Coupled Transport Benchmark
"""Certify four-state gradients and one bounded source-control trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT_JSON = ROOT / "validation" / "reports" / "torax_differentiable_transport.json"
REPORT_MD = ROOT / "validation" / "reports" / "torax_differentiable_transport.md"
REFERENCE_PATH = (
    ROOT / "validation" / "reference_data" / "torax" / "torax_coupled_transport_reference.json"
)
RUNTIME_PATH = SRC / "scpn_fusion" / "core" / "integrated_transport_solver_differentiable.py"
NUMERICS_PATH = (
    SRC / "scpn_fusion" / "core" / "_integrated_transport_solver_differentiable_numerics.py"
)
CONTRACTS_PATH = (
    SRC / "scpn_fusion" / "core" / "integrated_transport_solver_differentiable_contracts.py"
)
SCHEMA = "scpn-fusion-core.differentiable-coupled-transport.v1"
TARGET_CONTROLS = (1.25, 0.80, 1.30)
INITIAL_CONTROLS = (1.0, 1.0, 1.0)
FINITE_DIFFERENCE_STEP = 1.0e-4
PERTURBATION = (5.0e-3, -5.0e-3, 5.0e-3)
THRESHOLDS = {
    "forward_maximum_absolute_error": 1.0e-12,
    "gradient_relative_error": 1.0e-2,
    "minimum_gradient_magnitude": 1.0e-12,
    "perturbation_linearisation_relative_error": 5.0e-2,
    "optimised_over_initial_objective": 2.0e-2,
}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from scpn_fusion.core.integrated_transport_solver import TransportSolver  # noqa: E402
from scpn_fusion.core.integrated_transport_solver_differentiable import (  # noqa: E402
    CoupledTransportControls,
    CoupledTransportTarget,
    scaled_inputs,
)
from scpn_fusion.core.integrated_transport_solver_coupled_contracts import (  # noqa: E402
    CoupledTransportInputs,
)
from validation import benchmark_torax_coupled_transport_parity as coupled  # noqa: E402

FloatArray = npt.NDArray[np.float64]
State = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


def _checksum(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _profile(case: Mapping[str, Any], name: str, index: int) -> FloatArray:
    profiles = cast(Mapping[str, Sequence[Sequence[float]]], case["comparison_profiles"])
    values = np.asarray(profiles[name][index], dtype=np.float64)
    if values.ndim != 1 or values.size < 4 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid reference profile: {name}[{index}]")
    return values


def load_reference() -> dict[str, Any]:
    """Load the authenticated pinned coupled TORAX reference."""
    return coupled.load_reference(REFERENCE_PATH)


def _input_history() -> list[CoupledTransportInputs]:
    return [
        coupled._native_inputs(time_s=0.0, dt_s=0.01),
        coupled._native_inputs(time_s=0.01, dt_s=0.01),
    ]


def _solver(case: Mapping[str, Any]) -> TransportSolver:
    rho = np.asarray(case["comparison_rho_norm"], dtype=np.float64)
    solver = TransportSolver(coupled.CONFIG_PATH, nr=int(rho.size))
    solver.rho = rho.copy()
    solver.nr = int(rho.size)
    solver.drho = float(rho[1] - rho[0])
    solver.Ti = _profile(case, "ion_temperature_kev", 0)
    solver.Te = _profile(case, "electron_temperature_kev", 0)
    solver.ne = _profile(case, "electron_density_m3", 0) / 1.0e19
    solver.set_coupled_flux_profile(
        _profile(case, "poloidal_flux_wb_per_rad", 0),
        major_radius_m=6.2,
        minor_radius_m=2.0,
        magnetic_field_t=5.3,
    )
    return solver


def production_rollout(case: Mapping[str, Any], controls: CoupledTransportControls) -> State:
    """Replay source controls through the public stateful production path."""
    solver = _solver(case)
    result = None
    for inputs in _input_history():
        result = solver.evolve_coupled_transport(scaled_inputs(inputs, controls))
    if result is None:
        raise RuntimeError("production rollout produced no steps")
    return (
        result.ion_temperature_kev,
        result.electron_temperature_kev,
        result.electron_density_1e19_m3,
        result.poloidal_flux_wb_per_rad,
    )


def _objective(states: State, target: CoupledTransportTarget) -> float:
    target_states = (
        target.ion_temperature_kev,
        target.electron_temperature_kev,
        target.electron_density_1e19_m3,
        target.poloidal_flux_wb_per_rad,
    )
    total = 0.0
    for observed, desired, weight in zip(states, target_states, target.state_weights, strict=True):
        scale = max(float(np.linalg.norm(desired)) / np.sqrt(desired.size), 1.0e-12)
        total += weight * float(np.mean(((observed - desired) / scale) ** 2))
    return float(total / sum(target.state_weights))


def _target(case: Mapping[str, Any]) -> CoupledTransportTarget:
    target_state = production_rollout(case, CoupledTransportControls(*TARGET_CONTROLS))
    return CoupledTransportTarget(*target_state)


def _finite_difference(case: Mapping[str, Any], target: CoupledTransportTarget) -> FloatArray:
    values = np.asarray(INITIAL_CONTROLS, dtype=np.float64)
    gradient = np.empty(3, dtype=np.float64)
    for index in range(3):
        plus = values.copy()
        minus = values.copy()
        plus[index] += FINITE_DIFFERENCE_STEP
        minus[index] -= FINITE_DIFFERENCE_STEP
        plus_objective = _objective(
            production_rollout(case, CoupledTransportControls.from_array(plus)), target
        )
        minus_objective = _objective(
            production_rollout(case, CoupledTransportControls.from_array(minus)), target
        )
        gradient[index] = (plus_objective - minus_objective) / (2.0 * FINITE_DIFFERENCE_STEP)
    return gradient


def _torax_baseline_state(case: Mapping[str, Any]) -> State:
    return (
        _profile(case, "ion_temperature_kev", -1),
        _profile(case, "electron_temperature_kev", -1),
        _profile(case, "electron_density_m3", -1) / 1.0e19,
        _profile(case, "poloidal_flux_wb_per_rad", -1),
    )


def _scientific_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "reference_projection_sha256",
        "source_provenance",
        "target_contract",
        "forward_replay",
        "gradient_metrics",
        "perturbation_replay",
        "optimisation",
        "determinism",
        "gates",
        "passes_thresholds",
        "claim_boundary",
    )
    return {key: report[key] for key in keys}


def build_report() -> dict[str, Any]:
    """Run production replay, gradient checks, and deterministic optimisation."""
    reference = load_reference()
    case = cast(Mapping[str, Any], reference["primary"])
    target = _target(case)
    inputs_history = _input_history()
    initial = CoupledTransportControls(*INITIAL_CONTROLS)
    solver = _solver(case)
    differentiation_start = time.perf_counter()
    differentiated = solver.differentiate_coupled_transport(inputs_history, initial, target)
    differentiation_seconds = time.perf_counter() - differentiation_start
    production_initial = production_rollout(case, initial)
    differentiated_state = (
        differentiated.ion_temperature_kev,
        differentiated.electron_temperature_kev,
        differentiated.electron_density_1e19_m3,
        differentiated.poloidal_flux_wb_per_rad,
    )
    state_errors = {
        name: float(np.max(np.abs(observed - expected)))
        for name, observed, expected in zip(
            ("ion_temperature", "electron_temperature", "electron_density", "poloidal_flux"),
            differentiated_state,
            production_initial,
            strict=True,
        )
    }
    finite_difference = _finite_difference(case, target)
    autodiff = differentiated.gradient
    relative_errors = np.abs(autodiff - finite_difference) / np.maximum(
        np.maximum(np.abs(autodiff), np.abs(finite_difference)), 1.0e-14
    )
    perturbation = np.asarray(PERTURBATION, dtype=np.float64)
    perturbed_controls = CoupledTransportControls.from_array(initial.as_array() + perturbation)
    perturbed_objective = _objective(production_rollout(case, perturbed_controls), target)
    actual_change = perturbed_objective - differentiated.objective
    predicted_change = float(np.dot(autodiff, perturbation))
    linearisation_error = abs(actual_change - predicted_change) / max(
        abs(actual_change), abs(predicted_change), 1.0e-14
    )

    optimisation_start = time.perf_counter()
    optimisation = _solver(case).optimise_coupled_transport(
        inputs_history,
        initial,
        target,
        iterations=12,
        learning_rate=0.08,
    )
    optimisation_seconds = time.perf_counter() - optimisation_start
    replay = _solver(case).optimise_coupled_transport(
        inputs_history,
        initial,
        target,
        iterations=12,
        learning_rate=0.08,
    )
    torax_objective = _objective(_torax_baseline_state(case), target)
    torax_runtime = cast(Mapping[str, Any], reference["runtime_seconds"])
    torax_cold_seconds = float(torax_runtime["cold_primary"])

    gates = {
        "production_forward_replay": max(state_errors.values())
        <= THRESHOLDS["forward_maximum_absolute_error"],
        "finite_full_chain_gradients": bool(
            np.all(np.isfinite(autodiff))
            and np.all(np.abs(autodiff) >= THRESHOLDS["minimum_gradient_magnitude"])
        ),
        "central_finite_difference": float(np.max(relative_errors))
        <= THRESHOLDS["gradient_relative_error"],
        "perturbation_replay": linearisation_error
        <= THRESHOLDS["perturbation_linearisation_relative_error"],
        "deterministic_optimisation": bool(
            np.array_equal(optimisation.objective_history, replay.objective_history)
            and np.array_equal(optimisation.control_history, replay.control_history)
        ),
        "optimisation_quality": optimisation.final_objective
        <= optimisation.initial_objective * THRESHOLDS["optimised_over_initial_objective"]
        and optimisation.final_objective <= torax_objective,
        "same_case_cost": optimisation_seconds <= torax_cold_seconds,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "differentiable_model_intersection_evaluated",
        "reference_projection_sha256": cast(Mapping[str, Any], reference["determinism"])[
            "primary_projection_sha256"
        ],
        "source_provenance": {
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime_sha256": hashlib.sha256(RUNTIME_PATH.read_bytes()).hexdigest(),
            "numerics_sha256": hashlib.sha256(NUMERICS_PATH.read_bytes()).hexdigest(),
            "contracts_sha256": hashlib.sha256(CONTRACTS_PATH.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
        },
        "target_contract": {
            "kind": "production perturbation recovery",
            "target_controls": list(TARGET_CONTROLS),
            "initial_controls": list(INITIAL_CONTROLS),
            "control_order": ["heat_power_scale", "particle_rate_scale", "driven_current_scale"],
            "finite_difference_step": FINITE_DIFFERENCE_STEP,
            "thresholds": THRESHOLDS,
        },
        "forward_replay": {
            "state_maximum_absolute_errors": state_errors,
            "maximum_absolute_error": max(state_errors.values()),
        },
        "gradient_metrics": {
            "autodiff": autodiff.tolist(),
            "central_finite_difference": finite_difference.tolist(),
            "relative_error": relative_errors.tolist(),
            "maximum_relative_error": float(np.max(relative_errors)),
        },
        "perturbation_replay": {
            "control_delta": perturbation.tolist(),
            "actual_objective_change": actual_change,
            "linear_prediction": predicted_change,
            "relative_error": linearisation_error,
        },
        "optimisation": {
            "iterations": optimisation.iterations,
            "initial_objective": optimisation.initial_objective,
            "final_objective": optimisation.final_objective,
            "final_over_initial": optimisation.final_objective
            / max(optimisation.initial_objective, 1.0e-30),
            "torax_nominal_baseline_objective": torax_objective,
            "final_controls": optimisation.final_controls.as_array().tolist(),
            "objective_history": optimisation.objective_history.tolist(),
            "control_history": optimisation.control_history.tolist(),
            "final_gradient": optimisation.final_gradient.tolist(),
        },
        "determinism": {
            "objective_history_byte_identical": bool(
                np.array_equal(optimisation.objective_history, replay.objective_history)
            ),
            "control_history_byte_identical": bool(
                np.array_equal(optimisation.control_history, replay.control_history)
            ),
        },
        "runtime_seconds": {
            "differentiation_cold": differentiation_seconds,
            "native_optimisation_cold": optimisation_seconds,
            "torax_nominal_cold": torax_cold_seconds,
        },
        "gates": gates,
        "passes_thresholds": all(gates.values()),
        "performance_superiority_claimed": False,
        "general_transport_differentiability_claimed": False,
        "claim_boundary": (
            "The evidence covers only the frozen circular prescribed-coefficient coupled model "
            "intersection and a synthetic production-perturbation recovery target. The TORAX "
            "row is its pinned nominal cold baseline, not a TORAX optimiser comparison. Loaded-host "
            "timings establish cost completeness for this case only, not portable performance "
            "superiority or differentiability of other production transport models."
        ),
    }
    report["scientific_projection_sha256"] = _checksum(_scientific_projection(report))
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    gradient = cast(Mapping[str, Any], report["gradient_metrics"])
    optimisation = cast(Mapping[str, Any], report["optimisation"])
    gates = cast(Mapping[str, Any], report["gates"])
    lines = [
        "# Differentiable Coupled Transport",
        "",
        f"Status: `{report['status']}`",
        f"Overall pass: `{report['passes_thresholds']}`",
        f"Performance superiority claimed: `{report['performance_superiority_claimed']}`",
        f"General transport differentiability claimed: `{report['general_transport_differentiability_claimed']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "## Gradient and optimisation evidence",
        "",
        f"- Maximum AD/central-FD relative error: `{float(gradient['maximum_relative_error']):.12g}`",
        f"- Initial objective: `{float(optimisation['initial_objective']):.12g}`",
        f"- Final objective: `{float(optimisation['final_objective']):.12g}`",
        f"- TORAX nominal baseline objective: `{float(optimisation['torax_nominal_baseline_objective']):.12g}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- `{name}`: `{gates[name]}`" for name in sorted(gates))
    lines.append("")
    return "\n".join(lines)


def write_report() -> dict[str, Any]:
    report = build_report()
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    return report


def check_report(*, report_json: Path = REPORT_JSON, report_md: Path = REPORT_MD) -> list[str]:
    errors: list[str] = []
    try:
        reference = coupled.load_reference(REFERENCE_PATH)
        observed = cast(dict[str, Any], json.loads(report_json.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]
    if observed.get("schema") != SCHEMA:
        errors.append("differentiable transport report has an unexpected schema")
    reference_projection = cast(Mapping[str, Any], reference["determinism"])[
        "primary_projection_sha256"
    ]
    if observed.get("reference_projection_sha256") != reference_projection:
        errors.append("differentiable transport reference projection is stale")
    expected_provenance = {
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(RUNTIME_PATH.read_bytes()).hexdigest(),
        "numerics_sha256": hashlib.sha256(NUMERICS_PATH.read_bytes()).hexdigest(),
        "contracts_sha256": hashlib.sha256(CONTRACTS_PATH.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
    }
    if observed.get("source_provenance") != expected_provenance:
        errors.append("differentiable transport source provenance is stale")
    if observed.get("scientific_projection_sha256") != _checksum(_scientific_projection(observed)):
        errors.append("differentiable transport scientific projection is stale")
    if not bool(observed.get("passes_thresholds", False)):
        errors.append("differentiable transport frozen gates do not pass")
    runtime = observed.get("runtime_seconds")
    if not isinstance(runtime, Mapping) or not all(
        np.isfinite(float(value)) and float(value) > 0.0 for value in runtime.values()
    ):
        errors.append("differentiable transport runtime evidence is incomplete")
    if not report_md.exists() or report_md.read_text(encoding="utf-8") != render_markdown(observed):
        errors.append("differentiable transport Markdown report is stale")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        errors = check_report()
        for error in errors:
            print(f"DIFFERENTIABLE TRANSPORT ERROR: {error}", file=sys.stderr)
        return 1 if errors else 0
    report = write_report()
    print(json.dumps(report["gates"], indent=2, sort_keys=True))
    return 1 if args.strict and not bool(report["passes_thresholds"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
