#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime Contract Benchmark
"""Generate and verify the real TORAX runtime request, result, and sidecar."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.integrations.torax import (  # noqa: E402
    TORAX_OUTCOME_SCHEMA,
    TORAX_REQUEST_SCHEMA,
    ToraxClock,
    ToraxConfigBinding,
    ToraxGeometry,
    ToraxProjection,
    ToraxRunRequest,
    ToraxRuntimeClient,
    ToraxSignal,
    build_review_envelope,
    review_envelope_sha256,
    review_envelope_to_bytes,
)
from scpn_fusion.integrations.torax.serialization import (  # noqa: E402
    canonical_sha256,
    file_sha256,
    load_json_object,
    write_bytes_atomic,
    write_json_atomic,
)
from validation.reference_data.torax.coupled_transport_model_intersection_deck import (  # noqa: E402
    CONFIG,
    MODEL_INTERSECTION,
)

DECK_PATH = ROOT / "validation/reference_data/torax/coupled_transport_model_intersection_deck.py"
REQUEST_PATH = ROOT / "validation/reference_data/torax/torax_runtime_request_v1.json"
RESULT_PATH = ROOT / "validation/reference_data/torax/torax_runtime_result_v1.json"
SIDECAR_PATH = ROOT / "validation/reference_data/torax/torax_runtime_primary_v1.nc"
REVIEW_PATH = ROOT / "validation/reference_data/torax/torax_runtime_review_envelope_v1.json"
REPORT_JSON = ROOT / "validation/reports/torax_runtime_contract.json"
REPORT_MD = ROOT / "validation/reports/torax_runtime_contract.md"
SOURCE_COMMIT = "d5a7edbc1af114b940dd94cc8cceeba9164591d0"
REPORT_SCHEMA = "scpn-fusion-core.torax-runtime-contract-report.v1"


def build_request(
    *,
    dt_s: float,
    request_id: str,
    event_id: str,
    timeout_s: float = 180.0,
    max_steps: int = 100,
) -> ToraxRunRequest:
    """Build a complete typed request for the frozen FCE-10 model intersection."""
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and > 0")
    config = cast(dict[str, Any], json.loads(json.dumps(copy.deepcopy(CONFIG))))
    config["numerics"]["fixed_dt"] = dt_s
    geometry_raw = cast(Mapping[str, Any], MODEL_INTERSECTION["geometry"])
    time_raw = cast(Mapping[str, Any], MODEL_INTERSECTION["time"])
    profiles_raw = cast(Mapping[str, Any], MODEL_INTERSECTION["profiles"])
    sources_raw = cast(Mapping[str, Any], MODEL_INTERSECTION["sources"])
    transport_raw = cast(Mapping[str, Any], MODEL_INTERSECTION["transport"])
    initial_ns = _seconds_to_ns(float(time_raw["initial_s"]))
    final_ns = _seconds_to_ns(float(time_raw["final_s"]))
    frame = "axisymmetric_circular_toroidal_flux_rho_norm"
    bindings: list[ToraxConfigBinding] = []

    def bind(
        name: str,
        path: tuple[str, ...],
        unit: str,
        interpretation: str = "scalar",
    ) -> str:
        value: object = config
        for part in path:
            value = cast(Mapping[str, object], value)[part]
        bindings.append(
            ToraxConfigBinding(
                name=name,
                config_path=path,
                unit=unit,
                interpretation=interpretation,
                value=value,
            )
        )
        return name

    bind("clock.initial_s", ("numerics", "t_initial"), "s")
    bind("clock.final_s", ("numerics", "t_final"), "s")
    bind("geometry.major_radius_m", ("geometry", "R_major"), "m")
    bind("geometry.minor_radius_m", ("geometry", "a_minor"), "m")
    bind("geometry.magnetic_field_t", ("geometry", "B_0"), "T")

    state_specs = (
        (
            "ion_temperature",
            "keV",
            ("profile_conditions", "T_i"),
            (
                float(profiles_raw["ion_temperature_core_kev"]),
                float(profiles_raw["ion_temperature_edge_initial_kev"]),
            ),
        ),
        (
            "electron_temperature",
            "keV",
            ("profile_conditions", "T_e"),
            (
                float(profiles_raw["electron_temperature_core_kev"]),
                float(profiles_raw["electron_temperature_edge_initial_kev"]),
            ),
        ),
        (
            "electron_density",
            "m^-3",
            ("profile_conditions", "n_e"),
            (
                float(profiles_raw["electron_density_core_m3"]),
                float(profiles_raw["electron_density_edge_initial_m3"]),
            ),
        ),
    )
    initial_state = tuple(
        ToraxSignal(
            name=name,
            role="initial_state",
            unit=unit,
            frame=frame,
            time_ns=(initial_ns,),
            coordinate_name="rho_norm",
            coordinate_unit="1",
            coordinate=(0.0, 1.0),
            values=(values,),
            binding_name=bind(f"initial_state.{name}", path, unit, "time_radial_map"),
            calibration="prescribed_frozen_deck",
            provenance=str(DECK_PATH.relative_to(ROOT)),
            uncertainty_kind="none_prescribed",
            uncertainty=None,
            application_semantics="initial_condition",
            model_delay_ns=0,
            saturation_minimum=None,
            saturation_maximum=None,
            maximum_slew_per_s=None,
            hardware_limits_status="not_applicable",
        )
        for name, unit, path, values in state_specs
    )

    control_specs = (
        (
            "plasma_current",
            "control",
            "A",
            ("profile_conditions", "Ip"),
            (initial_ns,),
            ((float(profiles_raw["plasma_current_a"]),),),
        ),
        (
            "ion_temperature_edge",
            "control",
            "keV",
            ("profile_conditions", "T_i_right_bc"),
            (initial_ns, final_ns),
            (
                (float(profiles_raw["ion_temperature_edge_initial_kev"]),),
                (float(profiles_raw["ion_temperature_edge_final_kev"]),),
            ),
        ),
        (
            "electron_temperature_edge",
            "control",
            "keV",
            ("profile_conditions", "T_e_right_bc"),
            (initial_ns, final_ns),
            (
                (float(profiles_raw["electron_temperature_edge_initial_kev"]),),
                (float(profiles_raw["electron_temperature_edge_final_kev"]),),
            ),
        ),
        (
            "electron_density_edge",
            "control",
            "m^-3",
            ("profile_conditions", "n_e_right_bc"),
            (initial_ns, final_ns),
            (
                (float(profiles_raw["electron_density_edge_initial_m3"]),),
                (float(profiles_raw["electron_density_edge_final_m3"]),),
            ),
        ),
        (
            "total_heat_power",
            "control",
            "W",
            ("sources", "generic_heat", "P_total"),
            (initial_ns,),
            ((float(sources_raw["heat_power_w"]),),),
        ),
        (
            "electron_heat_fraction",
            "parameter",
            "1",
            ("sources", "generic_heat", "electron_heat_fraction"),
            (initial_ns,),
            ((float(sources_raw["electron_heat_fraction"]),),),
        ),
        (
            "heat_source_location",
            "parameter",
            "rho_norm",
            ("sources", "generic_heat", "gaussian_location"),
            (initial_ns,),
            ((float(sources_raw["heat_center_rho"]),),),
        ),
        (
            "heat_source_width",
            "parameter",
            "rho_norm",
            ("sources", "generic_heat", "gaussian_width"),
            (initial_ns,),
            ((float(sources_raw["heat_width_rho"]),),),
        ),
        (
            "particle_source",
            "control",
            "s^-1",
            ("sources", "generic_particle", "S_total"),
            (initial_ns,),
            ((float(sources_raw["particle_rate_s"]),),),
        ),
        (
            "particle_source_location",
            "parameter",
            "rho_norm",
            ("sources", "generic_particle", "deposition_location"),
            (initial_ns,),
            ((float(sources_raw["particle_center_rho"]),),),
        ),
        (
            "particle_source_width",
            "parameter",
            "rho_norm",
            ("sources", "generic_particle", "particle_width"),
            (initial_ns,),
            ((float(sources_raw["particle_width_rho"]),),),
        ),
        (
            "driven_current",
            "control",
            "A",
            ("sources", "generic_current", "I_generic"),
            (initial_ns,),
            ((float(sources_raw["driven_current_a"]),),),
        ),
        (
            "current_source_location",
            "parameter",
            "rho_norm",
            ("sources", "generic_current", "gaussian_location"),
            (initial_ns,),
            ((float(sources_raw["current_center_rho"]),),),
        ),
        (
            "current_source_width",
            "parameter",
            "rho_norm",
            ("sources", "generic_current", "gaussian_width"),
            (initial_ns,),
            ((float(sources_raw["current_width_rho"]),),),
        ),
        (
            "ion_heat_diffusivity",
            "parameter",
            "m^2/s",
            ("transport", "chi_i"),
            (initial_ns,),
            ((float(transport_raw["ion_heat_diffusivity_m2_s"]),),),
        ),
        (
            "electron_heat_diffusivity",
            "parameter",
            "m^2/s",
            ("transport", "chi_e"),
            (initial_ns,),
            ((float(transport_raw["electron_heat_diffusivity_m2_s"]),),),
        ),
        (
            "electron_particle_diffusivity",
            "parameter",
            "m^2/s",
            ("transport", "D_e"),
            (initial_ns,),
            ((float(transport_raw["electron_particle_diffusivity_m2_s"]),),),
        ),
        (
            "electron_particle_convection",
            "parameter",
            "m/s",
            ("transport", "V_e"),
            (initial_ns,),
            ((float(transport_raw["electron_particle_convection_m_s"]),),),
        ),
        ("fixed_timestep", "parameter", "s", ("numerics", "fixed_dt"), (initial_ns,), ((dt_s,),)),
    )
    controls = tuple(
        ToraxSignal(
            name=name,
            role=role,
            unit=unit,
            frame=frame,
            time_ns=times,
            coordinate_name="",
            coordinate_unit="",
            coordinate=(),
            values=values,
            binding_name=bind(
                f"inputs.{name}",
                path,
                unit,
                "time_scalar_map" if len(times) > 1 else "scalar",
            ),
            calibration="prescribed_frozen_deck",
            provenance=str(DECK_PATH.relative_to(ROOT)),
            uncertainty_kind="none_prescribed",
            uncertainty=None,
            application_semantics=(
                "prescribed_source" if role in {"control", "disturbance"} else "model_parameter"
            ),
            model_delay_ns=0,
            saturation_minimum=None,
            saturation_maximum=None,
            maximum_slew_per_s=None,
            hardware_limits_status=(
                "not_declared_no_actuation_authority"
                if role in {"control", "disturbance"}
                else "not_applicable"
            ),
        )
        for name, role, unit, path, times, values in control_specs
    )
    config_sha256 = canonical_sha256(config)
    return ToraxRunRequest(
        request_id=request_id,
        event_id=event_id,
        model_id="torax-1.4.3-real-runtime",
        scenario_id="coupled-transport-model-intersection-v1",
        reactor_family="magnetic_confinement_tokamak",
        reactor_id="circular_iter_scale_comparison",
        configuration_id="fce10_constant_transport_prescribed_sources",
        clock=ToraxClock(
            domain="simulation_monotonic",
            epoch="scenario_start",
            initial_ns=initial_ns,
            final_ns=final_ns,
            timeout_s=timeout_s,
            max_steps=max_steps,
            reset_policy="fresh_process_no_hidden_state",
        ),
        geometry=ToraxGeometry(
            kind="axisymmetric_circular",
            frame=frame,
            major_radius_m=float(geometry_raw["major_radius_m"]),
            minor_radius_m=float(geometry_raw["minor_radius_m"]),
            magnetic_field_t=float(geometry_raw["magnetic_field_t"]),
        ),
        initial_state=initial_state,
        controls=controls,
        models=MappingProxyType(
            {
                "transport": "constant",
                "solver": "linear",
                "time_step_calculator": "fixed",
                "geometry": "circular",
                "pedestal": "disabled",
                "bootstrap_current_multiplier": 0.0,
            }
        ),
        torax_config=MappingProxyType(config),
        bindings=tuple(bindings),
        custody=MappingProxyType(
            {
                "caller": "validation.benchmark_torax_runtime_contract",
                "created_at_utc": "2026-08-30T00:00:00+00:00",
                "source_repo_commit": SOURCE_COMMIT,
                "config_sha256": config_sha256,
                "deck_path": str(DECK_PATH.relative_to(ROOT)),
                "deck_sha256": file_sha256(DECK_PATH),
            }
        ),
    )


def build_report(
    *,
    torax_python: Path,
    work_directory: Path,
    write_canonical: bool,
    source_revision: str | None = None,
    runtime_timeout_s: float = 360.0,
) -> dict[str, object]:
    """Run primary, repeat, and refined public requests and evaluate every gate."""
    work_directory.mkdir(parents=True, exist_ok=True)
    primary_request = build_request(
        dt_s=float(cast(Mapping[str, Any], MODEL_INTERSECTION["time"])["primary_dt_s"]),
        request_id="torax-runtime-primary-v1",
        event_id="fce10-primary-0001",
        timeout_s=runtime_timeout_s,
    )
    repeat_request = build_request(
        dt_s=float(cast(Mapping[str, Any], MODEL_INTERSECTION["time"])["primary_dt_s"]),
        request_id="torax-runtime-primary-v1",
        event_id="fce10-primary-0001",
        timeout_s=runtime_timeout_s,
    )
    refined_request = build_request(
        dt_s=float(cast(Mapping[str, Any], MODEL_INTERSECTION["time"])["refined_dt_s"]),
        request_id="torax-runtime-refined-v1",
        event_id="fce10-refined-0001",
        timeout_s=runtime_timeout_s,
    )
    request_path = (
        REQUEST_PATH.relative_to(ROOT)
        if write_canonical
        else work_directory / "primary.request.json"
    )
    result_path = (
        RESULT_PATH.relative_to(ROOT) if write_canonical else work_directory / "primary.result.json"
    )
    sidecar_path = (
        SIDECAR_PATH.relative_to(ROOT) if write_canonical else work_directory / "primary.nc"
    )
    client = ToraxRuntimeClient(torax_python, working_directory=ROOT)
    primary = client.run(
        primary_request,
        request_path=request_path,
        result_path=result_path,
        sidecar_path=sidecar_path,
    )
    repeat = client.run(
        repeat_request,
        request_path=work_directory / "repeat.request.json",
        result_path=work_directory / "repeat.result.json",
        sidecar_path=work_directory / "repeat.nc",
    )
    refined = client.run(
        refined_request,
        request_path=work_directory / "refined.request.json",
        result_path=work_directory / "refined.result.json",
        sidecar_path=work_directory / "refined.nc",
    )
    for name, outcome in (("primary", primary), ("repeat", repeat), ("refined", refined)):
        if not outcome.success:
            raise RuntimeError(
                f"{name} TORAX runtime failed: {outcome.failure_code}: {outcome.failure_message}"
            )
    assert primary.projection is not None
    assert repeat.projection is not None
    assert refined.projection is not None
    refinement_metrics = _refinement_metrics(primary.projection, refined.projection)
    refinement = {
        name: cast(float, metric["relative_l2"])
        for name, metric in refinement_metrics["profiles"].items()
    }
    all_refinement_relative_l2 = tuple(
        cast(float, metric["relative_l2"])
        for category in refinement_metrics.values()
        for metric in category.values()
    )
    effective_sidecar_path = ROOT / sidecar_path if not sidecar_path.is_absolute() else sidecar_path
    manifest_path = effective_sidecar_path.with_suffix(
        effective_sidecar_path.suffix + ".manifest.json"
    )
    primary_manifest = load_json_object(manifest_path)
    repeat_manifest = load_json_object(work_directory / "repeat.nc.manifest.json")
    manifest_names = {
        str(variable["name"])
        for group in cast(Sequence[Mapping[str, object]], primary_manifest["groups"])
        for variable in cast(Sequence[Mapping[str, object]], group["variables"])
    }
    forbidden_typed = {"q95", "li3", "beta_N", "W_thermal_total", "regime", "phase"}
    typed_text = json.dumps(primary.projection.to_dict(), sort_keys=True)
    runtime_source_paths = (
        ROOT / "src/scpn_fusion/integrations/torax/contracts.py",
        ROOT / "src/scpn_fusion/integrations/torax/serialization.py",
        ROOT / "src/scpn_fusion/integrations/torax/projection.py",
        ROOT / "src/scpn_fusion/integrations/torax/worker.py",
        ROOT / "src/scpn_fusion/integrations/torax/client.py",
        ROOT / "src/scpn_fusion/integrations/torax/review.py",
        ROOT / "src/scpn_fusion/integrations/torax/__main__.py",
    )
    runtime_source_files = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in runtime_source_paths
    }
    runtime_source_sha256 = canonical_sha256(runtime_source_files)
    primary_dt_ns = _seconds_to_ns(
        float(cast(Mapping[str, Any], MODEL_INTERSECTION["time"])["primary_dt_s"])
    )
    refined_dt_ns = _seconds_to_ns(
        float(cast(Mapping[str, Any], MODEL_INTERSECTION["time"])["refined_dt_s"])
    )
    producer_revision = _repository_head() if source_revision is None else source_revision
    review_envelope = build_review_envelope(
        request=primary_request,
        refined_request=refined_request,
        primary=primary,
        refined=refined,
        refinement_metrics=refinement_metrics,
        primary_dt_ns=primary_dt_ns,
        refined_dt_ns=refined_dt_ns,
        source_revision=producer_revision,
        runtime_source_sha256=runtime_source_sha256,
        artifact_content_sha256=cast(str, primary_manifest["content_sha256"]),
        manifest_inventory_sha256=cast(str, primary_manifest["inventory_sha256"]),
    )
    repeat_review_envelope = build_review_envelope(
        request=primary_request,
        refined_request=refined_request,
        primary=repeat,
        refined=refined,
        refinement_metrics=refinement_metrics,
        primary_dt_ns=primary_dt_ns,
        refined_dt_ns=refined_dt_ns,
        source_revision=producer_revision,
        runtime_source_sha256=runtime_source_sha256,
        artifact_content_sha256=cast(str, repeat_manifest["content_sha256"]),
        manifest_inventory_sha256=cast(str, repeat_manifest["inventory_sha256"]),
    )
    gates = {
        "schemas_exact": primary_request.schema == TORAX_REQUEST_SCHEMA
        and primary.schema == TORAX_OUTCOME_SCHEMA,
        "all_runs_complete": primary.complete and repeat.complete and refined.complete,
        "deterministic_projection": primary.projection.scientific_sha256
        == repeat.projection.scientific_sha256,
        "deterministic_complete_sidecar_content": primary_manifest["content_sha256"]
        == repeat_manifest["content_sha256"]
        and primary_manifest["inventory_sha256"] == repeat_manifest["inventory_sha256"],
        "deterministic_review_envelope": review_envelope_to_bytes(review_envelope)
        == review_envelope_to_bytes(repeat_review_envelope),
        "refinement_converged": max(all_refinement_relative_l2)
        <= float(
            cast(Mapping[str, Any], MODEL_INTERSECTION["thresholds"])[
                "torax_refinement_relative_l2"
            ]
        ),
        "complete_inventory": int(primary_manifest["group_count"]) == 4
        and int(primary_manifest["variable_count"]) >= 180,
        "backend_scalars_retained": {"q95", "li3", "beta_N", "W_thermal_total"} <= manifest_names,
        "inferred_scalars_omitted_from_typed_projection": all(
            name not in typed_text for name in forbidden_typed
        ),
        "clock_exact": primary.projection.time_ns[-1] == primary_request.clock.final_ns,
        "sidecar_custody_verified": primary.artifact is not None
        and primary.artifact.sidecar_sha256 == file_sha256(effective_sidecar_path)
        and primary.artifact.manifest_sha256 == file_sha256(manifest_path),
        "source_totals_finite": all(
            np.isfinite(value)
            for values in primary.projection.source_totals.values()
            for value in values
        ),
        "state_budgets_finite": all(
            np.isfinite(value) for row in primary.projection.state_budgets for value in row.values()
        ),
    }
    source_paths = (*runtime_source_paths, Path(__file__))
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "passes_thresholds": all(gates.values()),
        "gates": gates,
        "request": {
            "schema": primary_request.schema,
            "sha256": canonical_sha256(primary_request.to_dict()),
            "path": str(request_path),
        },
        "outcome": {
            "schema": primary.schema,
            "scientific_sha256": primary.projection.scientific_sha256,
            "path": str(result_path),
        },
        "artifact": None if primary.artifact is None else primary.artifact.to_dict(),
        "review_envelope": {
            "schema": review_envelope.schema,
            "sha256": review_envelope_sha256(review_envelope),
            "path": str(REVIEW_PATH.relative_to(ROOT)),
            "source_revision": review_envelope.source_revision,
            "model_intersection_schema": review_envelope.model_intersection_schema,
        },
        "manifest": {
            "group_count": primary_manifest["group_count"],
            "variable_count": primary_manifest["variable_count"],
            "content_sha256": primary_manifest["content_sha256"],
            "inventory_sha256": primary_manifest["inventory_sha256"],
            "all_variable_names": sorted(manifest_names),
        },
        "refinement_relative_l2": refinement,
        "uncertainty": {
            "kind": "numerical_refinement",
            "comparison": "fixed_dt 0.01 s versus 0.005 s",
            "observables": refinement_metrics,
        },
        "provenance": {
            "repository_head_before_commit": _repository_head(),
            "source_files": {
                str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths
            },
            "runtime_source_sha256": runtime_source_sha256,
            "deck_sha256": file_sha256(DECK_PATH),
            "torax_version": primary.provenance.torax_version,
            "torax_license": primary.provenance.torax_license,
            "runtime_backend": primary.provenance.runtime_backend,
            "precision": primary.provenance.precision,
        },
        "claim_boundary": {
            "real_torax_execution": True,
            "complete_backend_output_retained": True,
            "typed_plant_truth": [
                "T_i",
                "T_e",
                "n_e",
                "psi",
                "source_totals",
                "state_budgets",
                "solver_status",
            ],
            "excluded_inferences": sorted(forbidden_typed),
            "actuation_authority": False,
            "experimental_validation": False,
            "portable_performance_claim": False,
        },
    }
    if not report["passes_thresholds"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"TORAX runtime contract gates failed: {failed}")
    if write_canonical:
        write_bytes_atomic(REVIEW_PATH, review_envelope_to_bytes(review_envelope))
        write_json_atomic(REPORT_JSON, report)
        REPORT_MD.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _relative_l2(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    numerator = float(np.linalg.norm(left_array - right_array))
    denominator = max(float(np.linalg.norm(right_array)), 1e-30)
    return numerator / denominator


def _refinement_metrics(
    primary: ToraxProjection,
    refined: ToraxProjection,
) -> dict[str, dict[str, dict[str, object]]]:
    """Measure every typed observable on the primary run's simulation times."""
    if primary.rho_norm != refined.rho_norm:
        raise ValueError("primary and refined projections use different radial grids")
    refined_time_index = {time_ns: index for index, time_ns in enumerate(refined.time_ns)}
    try:
        matching_refined_indices = tuple(refined_time_index[time_ns] for time_ns in primary.time_ns)
    except KeyError as error:
        raise ValueError("refined projection does not contain every primary sample time") from error

    def metrics(
        left: Sequence[float],
        right: Sequence[float],
        unit: str,
    ) -> dict[str, object]:
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        if left_array.shape != right_array.shape or left_array.size == 0:
            raise ValueError("refinement arrays must be non-empty and shape-identical")
        difference = left_array - right_array
        denominator = max(float(np.linalg.norm(right_array)), 1e-30)
        return {
            "absolute_rms": float(np.sqrt(np.mean(np.square(difference)))),
            "relative_l2": float(np.linalg.norm(difference)) / denominator,
            "unit": unit,
        }

    profiles: dict[str, dict[str, object]] = {}
    for name, rows in primary.profiles.items():
        primary_values = tuple(value for row in rows for value in row)
        refined_values = tuple(
            value for index in matching_refined_indices for value in refined.profiles[name][index]
        )
        profiles[name] = metrics(
            primary_values,
            refined_values,
            primary.profile_units[name],
        )

    source_totals: dict[str, dict[str, object]] = {}
    for name, values in primary.source_totals.items():
        source_totals[name] = metrics(
            values,
            tuple(refined.source_totals[name][index] for index in matching_refined_indices),
            primary.source_units[name],
        )

    state_budgets: dict[str, dict[str, object]] = {}
    for name in sorted(primary.budget_units):
        state_budgets[name] = metrics(
            tuple(row[name] for row in primary.state_budgets),
            tuple(refined.state_budgets[index][name] for index in matching_refined_indices),
            primary.budget_units[name],
        )
    return {
        "profiles": profiles,
        "source_totals": source_totals,
        "state_budgets": state_budgets,
    }


def _seconds_to_ns(value: float) -> int:
    scaled = value * 1_000_000_000.0
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-3:
        raise ValueError("time is not exactly representable in integer nanoseconds")
    return rounded


def _repository_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _render_markdown(report: Mapping[str, object]) -> str:
    gates = cast(Mapping[str, bool], report["gates"])
    refinement = cast(Mapping[str, float], report["refinement_relative_l2"])
    lines = [
        "# Real TORAX runtime contract",
        "",
        f"Overall gate: **{'PASS' if report['passes_thresholds'] else 'FAIL'}**.",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in gates.items())
    lines.extend(["", "## Fixed-timestep refinement", ""])
    lines.extend(f"- `{name}` relative L2: `{value:.12g}`" for name, value in refinement.items())
    lines.extend(
        [
            "",
            "The public runtime executed real TORAX 1.4.3 through the isolated CLI.",
            "The typed projection contains only Ti, Te, ne, poloidal flux, source totals,",
            "state budgets, and numerical status. The checksummed NetCDF DataTree sidecar",
            "retains every backend variable. No actuation, experimental-validation, full-physics",
            "equivalence, or portable-performance claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real contract gate and optionally publish canonical fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torax-python", type=Path, default=ROOT / ".venv-torax/bin/python")
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--write-canonical", action="store_true")
    parser.add_argument(
        "--source-revision",
        help="exact 40-hex producer code commit; defaults to the current repository HEAD",
    )
    parser.add_argument(
        "--runtime-timeout-s",
        type=float,
        default=360.0,
        help="bounded timeout for each isolated TORAX subprocess (default: 360 s)",
    )
    arguments = parser.parse_args(argv)
    if arguments.work_directory is None:
        with tempfile.TemporaryDirectory(prefix="scpn-torax-runtime-") as temporary:
            report = build_report(
                torax_python=arguments.torax_python,
                work_directory=Path(temporary),
                write_canonical=arguments.write_canonical,
                source_revision=arguments.source_revision,
                runtime_timeout_s=arguments.runtime_timeout_s,
            )
    else:
        report = build_report(
            torax_python=arguments.torax_python,
            work_directory=arguments.work_directory,
            write_canonical=arguments.write_canonical,
            source_revision=arguments.source_revision,
            runtime_timeout_s=arguments.runtime_timeout_s,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_report", "build_request", "main"]
