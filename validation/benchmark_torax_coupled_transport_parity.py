#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Coupled TORAX Transport Parity Benchmark
"""Verify the frozen four-state TORAX/native model-intersection contract."""

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
DECK_PATH = (
    ROOT
    / "validation"
    / "reference_data"
    / "torax"
    / "coupled_transport_model_intersection_deck.py"
)
REFERENCE_PATH = (
    ROOT / "validation" / "reference_data" / "torax" / "torax_coupled_transport_reference.json"
)
CONFIG_PATH = ROOT / "validation" / "iter_config.json"
REPORT_JSON = ROOT / "validation" / "reports" / "torax_coupled_transport_parity.json"
REPORT_MD = ROOT / "validation" / "reports" / "torax_coupled_transport_parity.md"
REFERENCE_RUNNER_PATH = ROOT / "validation" / "torax_coupled_reference_runner.py"
COUPLED_RUNTIME_PATH = SRC / "scpn_fusion" / "core" / "integrated_transport_solver_coupled.py"
COUPLED_CONTRACTS_PATH = (
    SRC / "scpn_fusion" / "core" / "integrated_transport_solver_coupled_contracts.py"
)
COUPLED_NUMERICS_PATH = (
    SRC / "scpn_fusion" / "core" / "_integrated_transport_solver_coupled_numerics.py"
)
CURRENT_DIFFUSION_PATH = SRC / "scpn_fusion" / "core" / "current_diffusion.py"
SCHEMA = "scpn-fusion-core.torax-coupled-transport-parity.v1"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from scpn_fusion.core.integrated_transport_solver import TransportSolver  # noqa: E402
from scpn_fusion.core.integrated_transport_solver_coupled import (  # noqa: E402
    CoupledTransportInputs,
    CoupledTransportStepResult,
)
from validation.reference_data.torax.coupled_transport_model_intersection_deck import (  # noqa: E402
    MODEL_INTERSECTION,
)

FloatArray = npt.NDArray[np.float64]


def _checksum(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _relative_l2(observed: FloatArray, reference: FloatArray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1.0e-30)
    return float(np.linalg.norm(observed - reference) / denominator)


def _finite_vector(raw: object, *, name: str) -> FloatArray:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{name} must be a sequence")
    values = np.asarray([float(value) for value in raw], dtype=np.float64)
    if values.ndim != 1 or values.size < 4 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional profile")
    return values


def _numeric_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def load_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Load and authenticate the dedicated-environment TORAX artifact."""
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema") != "scpn-fusion-core.torax-coupled-transport-reference.v1":
        raise ValueError("TORAX coupled reference has an unexpected schema")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("code") != "TORAX":
        raise ValueError("TORAX coupled reference is missing provenance")
    if provenance.get("torax_version") != "1.4.3":
        raise ValueError("TORAX coupled reference must use pinned TORAX 1.4.3")
    if provenance.get("deck_sha256") != hashlib.sha256(DECK_PATH.read_bytes()).hexdigest():
        raise ValueError("TORAX coupled reference deck source digest is stale")
    if provenance.get("deck_payload_sha256") != _checksum(MODEL_INTERSECTION):
        raise ValueError("TORAX coupled reference deck payload digest is stale")
    if (
        provenance.get("runner_sha256")
        != hashlib.sha256(REFERENCE_RUNNER_PATH.read_bytes()).hexdigest()
    ):
        raise ValueError("TORAX coupled reference runner digest is stale")
    determinism = payload.get("determinism")
    if not isinstance(determinism, Mapping) or not bool(
        determinism.get("byte_identical_scientific_projection", False)
    ):
        raise ValueError("TORAX coupled reference lacks deterministic replay")
    for case_name in ("primary", "refined"):
        case = payload.get(case_name)
        if not isinstance(case, Mapping) or case.get("sim_error") != "SimError.NO_ERROR":
            raise ValueError(f"TORAX {case_name} case did not complete cleanly")
    return payload


def _profile_at(reference_case: Mapping[str, Any], name: str, index: int) -> FloatArray:
    profiles = cast(Mapping[str, Any], reference_case["comparison_profiles"])
    snapshots = cast(Sequence[object], profiles[name])
    return _finite_vector(snapshots[index], name=f"{name}[{index}]")


def _edge_value(initial: float, final: float, fraction: float) -> float:
    return initial + fraction * (final - initial)


def _native_inputs(*, time_s: float, dt_s: float) -> CoupledTransportInputs:
    geometry = cast(Mapping[str, Any], MODEL_INTERSECTION["geometry"])
    time_config = cast(Mapping[str, Any], MODEL_INTERSECTION["time"])
    profiles = cast(Mapping[str, Any], MODEL_INTERSECTION["profiles"])
    transport = cast(Mapping[str, Any], MODEL_INTERSECTION["transport"])
    sources = cast(Mapping[str, Any], MODEL_INTERSECTION["sources"])
    end_time = time_s + dt_s
    fraction = (end_time - float(time_config["initial_s"])) / (
        float(time_config["final_s"]) - float(time_config["initial_s"])
    )
    return CoupledTransportInputs(
        time_s=time_s,
        dt_s=dt_s,
        major_radius_m=float(geometry["major_radius_m"]),
        minor_radius_m=float(geometry["minor_radius_m"]),
        magnetic_field_t=float(geometry["magnetic_field_t"]),
        effective_charge=float(profiles["effective_charge"]),
        ion_heat_diffusivity_m2_s=float(transport["ion_heat_diffusivity_m2_s"]),
        electron_heat_diffusivity_m2_s=float(transport["electron_heat_diffusivity_m2_s"]),
        electron_particle_diffusivity_m2_s=float(transport["electron_particle_diffusivity_m2_s"]),
        heat_power_w=float(sources["heat_power_w"]),
        electron_heat_fraction=float(sources["electron_heat_fraction"]),
        heat_center_rho=float(sources["heat_center_rho"]),
        heat_width_rho=float(sources["heat_width_rho"]),
        particle_rate_s=float(sources["particle_rate_s"]),
        particle_center_rho=float(sources["particle_center_rho"]),
        particle_width_rho=float(sources["particle_width_rho"]),
        driven_current_a=float(sources["driven_current_a"]),
        current_center_rho=float(sources["current_center_rho"]),
        current_width_rho=float(sources["current_width_rho"]),
        ion_electron_exchange_rate_s=float(transport["native_exchange_rate_s"]),
        ion_temperature_edge_kev=_edge_value(
            float(profiles["ion_temperature_edge_initial_kev"]),
            float(profiles["ion_temperature_edge_final_kev"]),
            fraction,
        ),
        electron_temperature_edge_kev=_edge_value(
            float(profiles["electron_temperature_edge_initial_kev"]),
            float(profiles["electron_temperature_edge_final_kev"]),
            fraction,
        ),
        electron_density_edge_1e19_m3=_edge_value(
            float(profiles["electron_density_edge_initial_m3"]) / 1.0e19,
            float(profiles["electron_density_edge_final_m3"]) / 1.0e19,
            fraction,
        ),
        resistivity_multiplier=float(transport["resistivity_multiplier"]),
    )


def _run_native(reference_case: Mapping[str, Any], *, dt_s: float) -> tuple[dict[str, Any], float]:
    geometry = cast(Mapping[str, Any], MODEL_INTERSECTION["geometry"])
    time_config = cast(Mapping[str, Any], MODEL_INTERSECTION["time"])
    rho = _finite_vector(reference_case["comparison_rho_norm"], name="comparison_rho_norm")
    solver = TransportSolver(CONFIG_PATH, nr=int(rho.size))
    solver.rho = rho.copy()
    solver.nr = int(rho.size)
    solver.drho = float(np.diff(rho)[0])
    solver.Ti = _profile_at(reference_case, "ion_temperature_kev", 0)
    solver.Te = _profile_at(reference_case, "electron_temperature_kev", 0)
    solver.ne = _profile_at(reference_case, "electron_density_m3", 0) / 1.0e19
    solver.chi_i = np.ones_like(rho)
    solver.chi_e = np.ones_like(rho)
    solver.D_n = np.ones_like(rho)
    solver.n_impurity = np.zeros_like(rho)
    solver.set_coupled_flux_profile(
        _profile_at(reference_case, "poloidal_flux_wb_per_rad", 0),
        major_radius_m=float(geometry["major_radius_m"]),
        minor_radius_m=float(geometry["minor_radius_m"]),
        magnetic_field_t=float(geometry["magnetic_field_t"]),
    )
    final_time = float(time_config["final_s"])
    step_results: list[CoupledTransportStepResult] = []
    current_time = float(time_config["initial_s"])
    start = time.perf_counter()
    while current_time < final_time - 1.0e-14:
        accepted_dt = min(dt_s, final_time - current_time)
        result = solver.evolve_coupled_transport(
            _native_inputs(time_s=current_time, dt_s=accepted_dt)
        )
        step_results.append(result)
        current_time = result.time_s
    elapsed = time.perf_counter() - start
    if not step_results:
        raise RuntimeError("native coupled transport produced no steps")
    final = step_results[-1]
    budgets = [vars(result.budget) for result in step_results]
    return (
        {
            "dt_s": dt_s,
            "time_s": [float(time_config["initial_s"])]
            + [result.time_s for result in step_results],
            "comparison_rho_norm": rho.tolist(),
            "final_profiles": {
                "ion_temperature_kev": final.ion_temperature_kev.tolist(),
                "electron_temperature_kev": final.electron_temperature_kev.tolist(),
                "electron_density_m3": (final.electron_density_1e19_m3 * 1.0e19).tolist(),
                "poloidal_flux_wb_per_rad": final.poloidal_flux_wb_per_rad.tolist(),
            },
            "step_budgets": budgets,
            "all_steps_converged": all(result.converged for result in step_results),
        },
        elapsed,
    )


def _max_relative_source_error(
    source_totals: Mapping[str, Any],
    expected: Mapping[str, float],
) -> float:
    errors: list[float] = []
    for name, target in expected.items():
        values = cast(Sequence[object], source_totals[name])
        errors.extend(
            abs(_numeric_float(value, name=f"{name} source total") - target)
            / max(abs(target), 1.0e-30)
            for value in values
        )
    return max(errors, default=0.0)


def _native_source_error(native: Mapping[str, Any], expected: Mapping[str, float]) -> float:
    errors: list[float] = []
    for budget_raw in cast(Sequence[object], native["step_budgets"]):
        budget = cast(Mapping[str, Any], budget_raw)
        reconstructed = {
            "ion_heat_w": float(budget["ion_heat_source_reconstructed_w"]),
            "electron_heat_w": float(budget["electron_heat_source_reconstructed_w"]),
            "particles_s": float(budget["particle_source_reconstructed_s"]),
            "driven_current_a": float(budget["driven_current_reconstructed_a"]),
        }
        errors.extend(
            abs(reconstructed[name] - target) / max(abs(target), 1.0e-30)
            for name, target in expected.items()
        )
    return max(errors, default=0.0)


def _max_native_residual(native: Mapping[str, Any]) -> float:
    keys = (
        "ion_temperature_linear_residual_linf",
        "electron_temperature_linear_residual_linf",
        "electron_density_linear_residual_linf",
        "current_linear_residual_linf",
    )
    return max(
        float(cast(Mapping[str, Any], budget)[key])
        for budget in cast(Sequence[object], native["step_budgets"])
        for key in keys
    )


def _max_exchange_relative_closure(native: Mapping[str, Any]) -> float:
    return max(
        abs(float(cast(Mapping[str, Any], budget)["ion_electron_exchange_closure_j"]))
        / max(abs(float(cast(Mapping[str, Any], budget)["thermal_energy_before_j"])), 1.0)
        for budget in cast(Sequence[object], native["step_budgets"])
    )


def _final_profile_metrics(
    native: Mapping[str, Any],
    torax_case: Mapping[str, Any],
) -> dict[str, float]:
    native_profiles = cast(Mapping[str, Any], native["final_profiles"])
    return {
        name: _relative_l2(
            _finite_vector(native_profiles[name], name=f"native.{name}"),
            _profile_at(torax_case, name, -1),
        )
        for name in (
            "ion_temperature_kev",
            "electron_temperature_kev",
            "electron_density_m3",
            "poloidal_flux_wb_per_rad",
        )
    }


def _refinement_metrics(primary: Mapping[str, Any], refined: Mapping[str, Any]) -> dict[str, float]:
    primary_profiles = cast(Mapping[str, Any], primary["final_profiles"])
    refined_profiles = cast(Mapping[str, Any], refined["final_profiles"])
    return {
        name: _relative_l2(
            _finite_vector(primary_profiles[name], name=f"primary.{name}"),
            _finite_vector(refined_profiles[name], name=f"refined.{name}"),
        )
        for name in primary_profiles
    }


def _torax_refinement_metrics(
    primary: Mapping[str, Any], refined: Mapping[str, Any]
) -> dict[str, float]:
    return {
        name: _relative_l2(_profile_at(primary, name, -1), _profile_at(refined, name, -1))
        for name in (
            "ion_temperature_kev",
            "electron_temperature_kev",
            "electron_density_m3",
            "poloidal_flux_wb_per_rad",
        )
    }


def _scientific_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "model_intersection_sha256",
            "reference_projection_sha256",
            "source_provenance",
            "state_metrics",
            "source_budget_metrics",
            "conservation_metrics",
            "refinement_metrics",
            "determinism",
            "gates",
            "passes_thresholds",
            "claim_boundary",
        )
    }


def build_report(reference_path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Run native primary/refined/replay cases and evaluate the frozen gate."""
    reference = load_reference(reference_path)
    primary_reference = cast(Mapping[str, Any], reference["primary"])
    refined_reference = cast(Mapping[str, Any], reference["refined"])
    time_config = cast(Mapping[str, Any], MODEL_INTERSECTION["time"])
    thresholds = cast(Mapping[str, Any], MODEL_INTERSECTION["thresholds"])
    sources = cast(Mapping[str, Any], MODEL_INTERSECTION["sources"])
    primary_native, native_cold = _run_native(
        primary_reference, dt_s=float(time_config["primary_dt_s"])
    )
    replay_native, native_warm = _run_native(
        primary_reference, dt_s=float(time_config["primary_dt_s"])
    )
    refined_native, native_refined = _run_native(
        primary_reference, dt_s=float(time_config["refined_dt_s"])
    )
    state_metrics = _final_profile_metrics(primary_native, primary_reference)
    native_refinement = _refinement_metrics(primary_native, refined_native)
    torax_refinement = _torax_refinement_metrics(primary_reference, refined_reference)
    expected_sources = {
        "ion_heat_w": float(sources["heat_power_w"])
        * (1.0 - float(sources["electron_heat_fraction"])),
        "electron_heat_w": float(sources["heat_power_w"])
        * float(sources["electron_heat_fraction"]),
        "particles_s": float(sources["particle_rate_s"]),
        "driven_current_a": float(sources["driven_current_a"]),
    }
    torax_source_error = _max_relative_source_error(
        cast(Mapping[str, Any], primary_reference["source_totals"]), expected_sources
    )
    native_source_error = _native_source_error(primary_native, expected_sources)
    max_native_residual = _max_native_residual(primary_native)
    max_exchange_closure = _max_exchange_relative_closure(primary_native)
    native_projection_sha = _checksum(primary_native)
    replay_projection_sha = _checksum(replay_native)
    torax_runtime = cast(Mapping[str, Any], reference["runtime_seconds"])
    warm_values = [float(torax_runtime["warm_primary"]), native_warm]
    warm_cost_ratio = max(warm_values) / max(min(warm_values), 1.0e-12)

    gates = {
        "accuracy": all(
            (
                state_metrics["ion_temperature_kev"]
                <= float(thresholds["ion_temperature_final_relative_l2"]),
                state_metrics["electron_temperature_kev"]
                <= float(thresholds["electron_temperature_final_relative_l2"]),
                state_metrics["electron_density_m3"]
                <= float(thresholds["electron_density_final_relative_l2"]),
                state_metrics["poloidal_flux_wb_per_rad"]
                <= float(thresholds["poloidal_flux_final_relative_l2"]),
            )
        ),
        "source_budgets": max(torax_source_error, native_source_error)
        <= float(thresholds["source_total_relative_error"]),
        "conservation": max_native_residual <= float(thresholds["native_linear_residual_linf"])
        and max_exchange_closure <= float(thresholds["native_exchange_relative_closure"]),
        "nonlinear_convergence": bool(primary_native["all_steps_converged"])
        and bool(refined_native["all_steps_converged"])
        and primary_reference["sim_error"] == "SimError.NO_ERROR"
        and refined_reference["sim_error"] == "SimError.NO_ERROR",
        "robustness": max(native_refinement.values())
        <= float(thresholds["native_refinement_relative_l2"])
        and max(torax_refinement.values()) <= float(thresholds["torax_refinement_relative_l2"])
        and native_projection_sha == replay_projection_sha
        and bool(
            cast(Mapping[str, Any], reference["determinism"])[
                "byte_identical_scientific_projection"
            ]
        ),
        "cost_completeness": all(
            np.isfinite(value) and value > 0.0
            for value in (
                float(torax_runtime["cold_primary"]),
                float(torax_runtime["warm_primary"]),
                native_cold,
                native_warm,
            )
        )
        and warm_cost_ratio <= float(thresholds["maximum_warm_cost_ratio"]),
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "coupled_model_intersection_evaluated",
        "model_intersection_sha256": _checksum(MODEL_INTERSECTION),
        "reference_artifact": str(reference_path.relative_to(ROOT)),
        "reference_projection_sha256": cast(Mapping[str, Any], reference["determinism"])[
            "primary_projection_sha256"
        ],
        "source_provenance": {
            "benchmark_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
            "coupled_runtime_sha256": hashlib.sha256(COUPLED_RUNTIME_PATH.read_bytes()).hexdigest(),
            "coupled_contracts_sha256": hashlib.sha256(
                COUPLED_CONTRACTS_PATH.read_bytes()
            ).hexdigest(),
            "coupled_numerics_sha256": hashlib.sha256(
                COUPLED_NUMERICS_PATH.read_bytes()
            ).hexdigest(),
            "current_diffusion_sha256": hashlib.sha256(
                CURRENT_DIFFUSION_PATH.read_bytes()
            ).hexdigest(),
            "deck_sha256": hashlib.sha256(DECK_PATH.read_bytes()).hexdigest(),
            "reference_artifact_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "reference_runner_sha256": hashlib.sha256(
                REFERENCE_RUNNER_PATH.read_bytes()
            ).hexdigest(),
        },
        "state_metrics": state_metrics,
        "source_budget_metrics": {
            "torax_max_relative_error": torax_source_error,
            "native_max_relative_error": native_source_error,
        },
        "conservation_metrics": {
            "native_max_linear_residual_linf": max_native_residual,
            "native_max_exchange_relative_closure": max_exchange_closure,
        },
        "refinement_metrics": {
            "native_primary_vs_refined": native_refinement,
            "torax_primary_vs_refined": torax_refinement,
        },
        "determinism": {
            "native_primary_projection_sha256": native_projection_sha,
            "native_replay_projection_sha256": replay_projection_sha,
            "native_byte_identical": native_projection_sha == replay_projection_sha,
            "torax_byte_identical": cast(Mapping[str, Any], reference["determinism"])[
                "byte_identical_scientific_projection"
            ],
        },
        "runtime_seconds": {
            "torax_cold": float(torax_runtime["cold_primary"]),
            "torax_warm": float(torax_runtime["warm_primary"]),
            "native_cold": native_cold,
            "native_warm": native_warm,
            "native_refined": native_refined,
            "warm_max_over_min_ratio": warm_cost_ratio,
        },
        "gates": gates,
        "passes_thresholds": all(gates.values()),
        "performance_superiority_claimed": False,
        "physics_equivalence_claimed": False,
        "claim_boundary": (
            "The report admits parity only for the frozen circular, prescribed-coefficient "
            "model intersection. It does not claim full TORAX physics equivalence, experimental "
            "validation, hardware-neutral performance, or superiority."
        ),
        "native_primary": primary_native,
        "native_refined": refined_native,
    }
    report["scientific_projection_sha256"] = _checksum(_scientific_projection(report))
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the coupled parity result and its explicit claim boundary."""
    state_metrics = cast(Mapping[str, Any], report["state_metrics"])
    source_metrics = cast(Mapping[str, Any], report["source_budget_metrics"])
    conservation = cast(Mapping[str, Any], report["conservation_metrics"])
    gates = cast(Mapping[str, Any], report["gates"])
    lines = [
        "# Coupled TORAX Transport Parity",
        "",
        f"Status: `{report['status']}`",
        f"Overall pass: `{report['passes_thresholds']}`",
        f"Performance superiority claimed: `{report['performance_superiority_claimed']}`",
        f"Physics equivalence claimed: `{report['physics_equivalence_claimed']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "## Final-state relative L2",
        "",
    ]
    for name in sorted(state_metrics):
        lines.append(f"- `{name}`: `{float(state_metrics[name]):.12g}`")
    lines.extend(
        [
            "",
            "## Source and conservation budgets",
            "",
            f"- TORAX maximum source-total relative error: `{float(source_metrics['torax_max_relative_error']):.12g}`",
            f"- Native maximum source-total relative error: `{float(source_metrics['native_max_relative_error']):.12g}`",
            f"- Native maximum linear residual: `{float(conservation['native_max_linear_residual_linf']):.12g}`",
            f"- Native maximum exchange relative closure: `{float(conservation['native_max_exchange_relative_closure']):.12g}`",
            "",
            "## Pareto gates",
            "",
        ]
    )
    for name in sorted(gates):
        lines.append(f"- `{name}`: `{gates[name]}`")
    lines.append("")
    return "\n".join(lines)


def write_report(*, report_json: Path = REPORT_JSON, report_md: Path = REPORT_MD) -> dict[str, Any]:
    """Execute the benchmark and write tracked JSON and Markdown evidence."""
    report = build_report()
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    return report


def check_report(*, report_json: Path = REPORT_JSON, report_md: Path = REPORT_MD) -> list[str]:
    """Validate tracked evidence without comparing nondeterministic timings."""
    errors: list[str] = []
    try:
        reference = load_reference()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]
    if not report_json.exists():
        return [f"missing coupled parity report: {report_json}"]
    observed = cast(dict[str, Any], json.loads(report_json.read_text(encoding="utf-8")))
    if observed.get("schema") != SCHEMA:
        errors.append("coupled parity report has an unexpected schema")
    if observed.get("model_intersection_sha256") != _checksum(MODEL_INTERSECTION):
        errors.append("coupled parity report model-intersection digest is stale")
    reference_sha = cast(Mapping[str, Any], reference["determinism"])["primary_projection_sha256"]
    if observed.get("reference_projection_sha256") != reference_sha:
        errors.append("coupled parity report reference projection is stale")
    expected_source_provenance = {
        "benchmark_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "coupled_runtime_sha256": hashlib.sha256(COUPLED_RUNTIME_PATH.read_bytes()).hexdigest(),
        "coupled_contracts_sha256": hashlib.sha256(COUPLED_CONTRACTS_PATH.read_bytes()).hexdigest(),
        "coupled_numerics_sha256": hashlib.sha256(COUPLED_NUMERICS_PATH.read_bytes()).hexdigest(),
        "current_diffusion_sha256": hashlib.sha256(CURRENT_DIFFUSION_PATH.read_bytes()).hexdigest(),
        "deck_sha256": hashlib.sha256(DECK_PATH.read_bytes()).hexdigest(),
        "reference_artifact_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
        "reference_runner_sha256": hashlib.sha256(REFERENCE_RUNNER_PATH.read_bytes()).hexdigest(),
    }
    if observed.get("source_provenance") != expected_source_provenance:
        errors.append("coupled parity report source provenance is stale")
    expected_projection_sha = _checksum(_scientific_projection(observed))
    if observed.get("scientific_projection_sha256") != expected_projection_sha:
        errors.append("coupled parity report scientific projection digest is stale")
    if not bool(observed.get("passes_thresholds", False)):
        errors.append("coupled parity report does not pass its frozen thresholds")
    runtime = observed.get("runtime_seconds")
    if not isinstance(runtime, Mapping) or not all(
        np.isfinite(float(value)) and float(value) > 0.0 for value in runtime.values()
    ):
        errors.append("coupled parity report runtime evidence is incomplete")
    expected_md = render_markdown(observed)
    if not report_md.exists():
        errors.append(f"missing coupled parity Markdown report: {report_md}")
    elif report_md.read_text(encoding="utf-8") != expected_md:
        errors.append("coupled parity Markdown report is stale")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run or check the coupled TORAX/native parity evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        errors = check_report(report_json=args.report_json, report_md=args.report_md)
        for error in errors:
            print(f"COUPLED TORAX PARITY ERROR: {error}", file=sys.stderr)
        return 1 if errors else 0
    report = write_report(report_json=args.report_json, report_md=args.report_md)
    print(json.dumps(report["gates"], indent=2, sort_keys=True))
    if args.strict and not bool(report["passes_thresholds"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
