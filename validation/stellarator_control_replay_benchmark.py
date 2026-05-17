# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stellarator Control Replay Benchmark
"""Deterministic geometry-neutral stellarator control replay benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scpn_fusion.control.stellarator_control_contracts import (
    ActuatorChannel,
    ActuatorSet,
    ControlObjective,
    DiagnosticChannel,
    DiagnosticFrame,
    MagneticConfiguration,
    ReplayScenario,
)
from scpn_fusion.core.stellarator_geometry import (
    StellaratorConfig,
    effective_ripple,
    iss04_scaling,
    stellarator_flux_surface,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "stellarator-control-replay-benchmark.v1"
CONFIG_SCHEMA_VERSION = "stellarator-control-replay-config.v1"
DEFAULT_CONFIG_PATH = (
    ROOT / "validation" / "reference_data" / "stellarator_control_replay_public_config.json"
)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "max_final_fieldline_spread": 0.024,
    "min_improvement_fraction": 0.28,
    "max_abs_current_A": 1200.0,
    "max_uncertainty_width": 0.018,
    "max_p95_latency_us": 1000.0,
}
PRIVATE_DATA_MARKERS = ("proprietary", "confidential", "private")
TRACE_REQUIRED_KEYS = (
    "step",
    "time_s",
    "fieldline_spread",
    "target_fieldline_spread",
    "requested_current_A",
    "applied_current_A",
    "uncertainty_sigma",
    "latency_us",
    "fault_active",
)


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16]


def _finite(name: str, value: Any) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(value_f):
        raise ValueError(f"{name} must be finite.")
    return value_f


def _finite_positive(name: str, value: Any) -> float:
    value_f = _finite(name, value)
    if value_f <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return value_f


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object.")
    return value


def _require_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array.")
    return value


def _normalise_thresholds(thresholds: Mapping[str, Any] | None) -> dict[str, float]:
    values = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    missing = sorted(set(DEFAULT_THRESHOLDS) - set(values))
    if missing:
        raise ValueError("missing threshold keys: " + ", ".join(missing))
    return {key: _finite(f"thresholds.{key}", values[key]) for key in DEFAULT_THRESHOLDS}


def _normalise_fault_schedule(
    fault_entries: list[Any],
    *,
    steps: int,
    actuator_names: set[str],
) -> dict[int, dict[str, str]]:
    faults: dict[int, dict[str, str]] = {}
    for index, entry in enumerate(fault_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"fault_schedule[{index}] must be an object.")
        step = int(entry.get("step", -1))
        if step < 0 or step >= steps:
            raise ValueError("fault schedule step out of replay bounds.")
        channel = str(entry.get("channel", "")).strip()
        if channel not in actuator_names:
            raise ValueError(f"unknown actuator channel in fault_schedule: {channel}")
        mode = str(entry.get("mode", "")).strip()
        if mode not in {"stuck"}:
            raise ValueError(f"unsupported actuator fault mode: {mode}")
        faults.setdefault(step, {})[channel] = mode
    return faults


def _contract_path(parent: str, key: str) -> str:
    return key if not parent else f"{parent}.{key}"


def _validate_schema_contract(
    payload: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    label: str,
) -> None:
    """Enforce the required/no-extra-fields subset of local JSON-schema contracts."""
    if schema.get("type") == "object":
        if not isinstance(payload, Mapping):
            return
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return
        for key in schema.get("required", []):
            if key not in payload:
                raise ValueError(f"missing required {label} key: {_contract_path(path, str(key))}")
        if schema.get("additionalProperties") is False:
            for key in payload:
                if key not in properties:
                    raise ValueError(f"unexpected {label} key: {_contract_path(path, str(key))}")
        for key, child_schema in properties.items():
            if key in payload and isinstance(child_schema, Mapping):
                _validate_schema_contract(
                    payload[key],
                    child_schema,
                    path=_contract_path(path, str(key)),
                    label=label,
                )
    elif schema.get("type") == "array":
        if not isinstance(payload, list):
            return
        item_schema = schema.get("items", {})
        if not isinstance(item_schema, Mapping):
            return
        for index, item in enumerate(payload):
            _validate_schema_contract(
                item,
                item_schema,
                path=f"{path}[{index}]",
                label=label,
            )


def _default_config_payload(*, steps: int, seed: int) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "name": "public_w7x_like_stellarator_replay",
        "seed": int(seed),
        "steps": int(steps),
        "dt_s": 0.001,
        "primary_control_actuator": "helical_trim_A",
        "magnetic_configuration": {
            "name": "public_w7x_like_reduced_order",
            "device_class": "stellarator",
            "field_periods": 5,
            "coordinate_system": "boozer_vmec_like",
            "reference": "public synthetic W7-X-like reduced-order fixture",
        },
        "stellarator_parameters": {
            "N_fp": 5,
            "R0": 5.5,
            "a": 0.53,
            "B0": 2.5,
            "iota_0": 0.87,
            "iota_a": 1.0,
            "mirror_ratio": 0.07,
            "helical_excursion": 0.05,
            "name": "public_w7x_like",
        },
        "actuators": [
            {
                "name": "helical_trim_A",
                "unit": "A",
                "min_value": -1200.0,
                "max_value": 1200.0,
                "slew_rate_per_s": 4.0e5,
                "latency_steps": 2,
                "failure_mode": "stuck_supported",
            }
        ],
        "control_objective": {
            "target_metrics": {"fieldline_spread": 0.015},
            "weights": {"fieldline_spread": 1.0, "actuator_margin": 0.15},
            "constraints": {"max_abs_current_A": 1200.0, "max_fieldline_spread": 0.024},
        },
        "diagnostics": {
            "flux_surface_s": 0.72,
            "n_theta": 32,
            "n_phi": 40,
            "fieldline_spread_sigma": 0.0025,
            "effective_ripple_sigma": 0.0008,
        },
        "fault_schedule": [
            {"step": max(2, int(steps) // 2), "channel": "helical_trim_A", "mode": "stuck"}
        ],
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "data_provenance": {
            "geometry": "public_synthetic",
            "equilibrium": "reduced_order_vmec_like",
            "diagnostics": "deterministic_seeded_synthetic",
            "external_company_data": "none",
        },
    }


def _scenario_from_config(
    payload: Mapping[str, Any],
) -> tuple[ReplayScenario, StellaratorConfig, dict[str, float | int | str]]:
    magnetic = _require_mapping(payload, "magnetic_configuration")
    stellarator = _require_mapping(payload, "stellarator_parameters")
    diagnostics = _require_mapping(payload, "diagnostics")
    objective_payload = _require_mapping(payload, "control_objective")
    actuator_payloads = _require_list(payload, "actuators")

    config = StellaratorConfig(
        N_fp=int(stellarator["N_fp"]),
        R0=_finite_positive("stellarator_parameters.R0", stellarator["R0"]),
        a=_finite_positive("stellarator_parameters.a", stellarator["a"]),
        B0=_finite_positive("stellarator_parameters.B0", stellarator["B0"]),
        iota_0=_finite_positive("stellarator_parameters.iota_0", stellarator["iota_0"]),
        iota_a=_finite_positive("stellarator_parameters.iota_a", stellarator["iota_a"]),
        mirror_ratio=_finite("stellarator_parameters.mirror_ratio", stellarator["mirror_ratio"]),
        helical_excursion=_finite(
            "stellarator_parameters.helical_excursion", stellarator["helical_excursion"]
        ),
        name=str(stellarator.get("name", "public_stellarator_config")),
    )
    magnetic_configuration = MagneticConfiguration(
        name=str(magnetic["name"]),
        device_class=str(magnetic["device_class"]),
        field_periods=int(magnetic["field_periods"]),
        coordinate_system=str(magnetic["coordinate_system"]),
        reference=str(magnetic["reference"]),
    )
    if magnetic_configuration.field_periods != config.N_fp:
        raise ValueError(
            "magnetic_configuration.field_periods must match stellarator_parameters.N_fp."
        )
    actuators = tuple(
        ActuatorChannel(
            name=str(entry["name"]),
            unit=str(entry["unit"]),
            min_value=_finite("actuators.min_value", entry["min_value"]),
            max_value=_finite("actuators.max_value", entry["max_value"]),
            slew_rate_per_s=_finite_positive("actuators.slew_rate_per_s", entry["slew_rate_per_s"]),
            latency_steps=int(entry.get("latency_steps", 0)),
            failure_mode=str(entry.get("failure_mode", "none")),
        )
        for entry in actuator_payloads
        if isinstance(entry, Mapping)
    )
    if len(actuators) != len(actuator_payloads):
        raise ValueError("actuators entries must be objects.")
    actuator_set = ActuatorSet(channels=actuators)
    primary_actuator = str(payload.get("primary_control_actuator", "")).strip()
    try:
        actuator_set.by_name(primary_actuator)
    except KeyError as exc:
        raise ValueError(f"unknown primary_control_actuator: {primary_actuator}") from exc
    objective = ControlObjective(
        target_metrics={
            key: _finite(f"control_objective.target_metrics.{key}", value)
            for key, value in _require_mapping(objective_payload, "target_metrics").items()
        },
        weights={
            key: _finite(f"control_objective.weights.{key}", value)
            for key, value in _require_mapping(objective_payload, "weights").items()
        },
        constraints={
            key: _finite(f"control_objective.constraints.{key}", value)
            for key, value in _require_mapping(objective_payload, "constraints").items()
        },
    )
    initial_frame = DiagnosticFrame(
        step=0,
        time_s=0.0,
        channels=(
            DiagnosticChannel(
                name="fieldline_spread",
                value=_open_loop_spread(config, 0.0, diagnostics),
                unit="rad",
                sigma=_finite(
                    "diagnostics.fieldline_spread_sigma", diagnostics["fieldline_spread_sigma"]
                ),
                provenance=str(_require_mapping(payload, "data_provenance")["diagnostics"]),
            ),
            DiagnosticChannel(
                name="effective_ripple",
                value=effective_ripple(
                    config, _finite("diagnostics.flux_surface_s", diagnostics["flux_surface_s"])
                ),
                unit="dimensionless",
                sigma=_finite(
                    "diagnostics.effective_ripple_sigma", diagnostics["effective_ripple_sigma"]
                ),
                provenance=str(_require_mapping(payload, "data_provenance")["diagnostics"]),
            ),
        ),
    )
    steps = int(payload["steps"])
    fault_schedule = _normalise_fault_schedule(
        _require_list(payload, "fault_schedule"),
        steps=steps,
        actuator_names={channel.name for channel in actuator_set.channels},
    )
    scenario = ReplayScenario(
        name=str(payload["name"]),
        seed=int(payload["seed"]),
        steps=steps,
        dt_s=_finite_positive("dt_s", payload["dt_s"]),
        magnetic_configuration=magnetic_configuration,
        actuator_set=actuator_set,
        objective=objective,
        initial_frame=initial_frame,
        fault_schedule=fault_schedule,
    )
    diagnostic_settings = {
        "flux_surface_s": _finite("diagnostics.flux_surface_s", diagnostics["flux_surface_s"]),
        "n_theta": int(diagnostics["n_theta"]),
        "n_phi": int(diagnostics["n_phi"]),
        "primary_control_actuator": primary_actuator,
    }
    return scenario, config, diagnostic_settings


def _open_loop_spread(
    config: StellaratorConfig,
    applied_current_A: float,
    diagnostics: Mapping[str, float | int | str],
) -> float:
    flux_surface_s = _finite("diagnostics.flux_surface_s", diagnostics["flux_surface_s"])
    _, _, field = stellarator_flux_surface(
        config,
        s=flux_surface_s,
        n_theta=int(diagnostics["n_theta"]),
        n_phi=int(diagnostics["n_phi"]),
    )
    normalised_field_rms = float(np.std(field / np.mean(field)))
    ripple = effective_ripple(config, flux_surface_s)
    control_reduction = 0.000041 * abs(float(applied_current_A))
    return float(max(0.008, normalised_field_rms + 0.45 * ripple - control_reduction))


def _latency_us(step: int, command_A: float) -> float:
    base = 142.0 + 3.0 * (step % 5)
    actuation_cost = 0.006 * abs(float(command_A))
    return float(base + actuation_cost)


def _run_replay(
    scenario: ReplayScenario,
    *,
    config: StellaratorConfig,
    diagnostics: Mapping[str, float | int | str],
) -> dict[str, Any]:
    actuator = scenario.actuator_set.by_name(str(diagnostics["primary_control_actuator"]))
    rng = np.random.default_rng(int(scenario.seed))
    requested_current = 0.0
    applied_current = 0.0
    previous_applied = 0.0
    delayed_queue = [0.0 for _ in range(int(actuator.latency_steps) + 1)]
    stuck_current: float | None = None
    target = float(scenario.objective.target_metrics["fieldline_spread"])
    trace: list[dict[str, float | int | str | bool]] = []

    for step in range(int(scenario.steps)):
        faults = scenario.fault_schedule.get(step, {})
        if faults.get(actuator.name) == "stuck":
            stuck_current = applied_current

        measured_spread = _open_loop_spread(config, applied_current, diagnostics)
        deterministic_probe = float(rng.normal(0.0, 0.00012))
        measured_spread = float(max(0.0, measured_spread + deterministic_probe))
        error = measured_spread - target
        requested_current = float(np.clip(5.4e4 * error, actuator.min_value, actuator.max_value))

        if stuck_current is not None:
            slewed_current = stuck_current
            fault_active = True
        else:
            slewed_current = actuator.apply_slew(previous_applied, requested_current, scenario.dt_s)
            fault_active = False
        previous_applied = slewed_current

        delayed_queue.append(slewed_current)
        applied_current = float(delayed_queue.pop(0))
        sigma = float(0.0018 + 0.08 * abs(measured_spread - target))
        trace.append(
            {
                "step": int(step),
                "time_s": round(float(step * scenario.dt_s), 9),
                "fieldline_spread": round(measured_spread, 9),
                "target_fieldline_spread": round(target, 9),
                "requested_current_A": round(requested_current, 6),
                "applied_current_A": round(applied_current, 6),
                "uncertainty_sigma": round(sigma, 9),
                "latency_us": round(_latency_us(step, applied_current), 6),
                "fault_active": fault_active,
            }
        )

    first_spread = float(trace[0]["fieldline_spread"])
    final_spread = float(trace[-1]["fieldline_spread"])
    improvement = (first_spread - final_spread) / max(first_spread, 1e-12)
    max_abs_current = max(abs(float(row["applied_current_A"])) for row in trace)
    p95_latency = float(np.percentile([float(row["latency_us"]) for row in trace], 95))
    sigma_values = np.asarray([float(row["uncertainty_sigma"]) for row in trace], dtype=float)
    final_sigma = float(sigma_values[-1])
    p95_high = float(final_spread + 1.96 * final_sigma)
    p95_low = float(max(0.0, final_spread - 1.96 * final_sigma))
    signature_payload = {"scenario": scenario.to_dict(), "trace": trace}
    return {
        "scenario": scenario.to_dict(),
        "trace": trace,
        "replay_signature": _signature(signature_payload),
        "summary": {
            "initial_fieldline_spread": round(first_spread, 9),
            "final_fieldline_spread": round(final_spread, 9),
            "improvement_fraction": round(float(improvement), 9),
            "max_abs_current_A": round(float(max_abs_current), 6),
            "p95_latency_us": round(p95_latency, 6),
        },
        "uncertainty": {
            "method": "deterministic seeded replay plus per-channel diagnostic sigma",
            "fieldline_spread_p95_low": round(p95_low, 9),
            "fieldline_spread_p95_high": round(p95_high, 9),
            "fieldline_spread_width": round(p95_high - p95_low, 9),
            "samples": int(len(trace)),
        },
    }


def load_config_schema() -> dict[str, Any]:
    schema_path = ROOT / "schemas" / "stellarator_control_replay_config.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_benchmark_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    resolved_path = config_path.resolve()
    try:
        payload["source_path"] = resolved_path.relative_to(ROOT).as_posix()
    except ValueError:
        payload["source_path"] = str(resolved_path)
    validate_benchmark_config(payload, load_config_schema())
    return payload


def validate_benchmark_config(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate public replay input without requiring an external schema package."""
    _validate_schema_contract(payload, schema, path="", label="config")
    allowed_keys = set(schema["properties"])
    for key in payload:
        if key not in allowed_keys:
            raise ValueError(f"unexpected config key: {key}")
    for key in schema["required"]:
        if key not in payload:
            raise ValueError(f"missing required config key: {key}")
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("unexpected config schema_version")
    steps = int(payload["steps"])
    if steps < 4:
        raise ValueError("steps must be >= 4.")
    _finite_positive("dt_s", payload["dt_s"])
    thresholds = _normalise_thresholds(_require_mapping(payload, "thresholds"))
    provenance = _require_mapping(payload, "data_provenance")
    if str(provenance.get("external_company_data", "")).strip().lower() != "none":
        raise ValueError("external_company_data must be 'none'.")
    payload_text = _stable_json(
        {key: value for key, value in payload.items() if key != "source_path"}
    ).lower()
    for marker in PRIVATE_DATA_MARKERS:
        if marker in payload_text:
            raise ValueError("config must not contain private or proprietary markers.")
    scenario, _, diagnostics = _scenario_from_config(payload)
    _normalise_fault_schedule(
        _require_list(payload, "fault_schedule"),
        steps=scenario.steps,
        actuator_names={channel.name for channel in scenario.actuator_set.channels},
    )
    if int(diagnostics["n_theta"]) < 8 or int(diagnostics["n_phi"]) < 8:
        raise ValueError("diagnostic grid dimensions must be >= 8.")
    for key, value in thresholds.items():
        if key != "min_improvement_fraction" and value <= 0.0:
            raise ValueError(f"thresholds.{key} must be > 0.")


def _passes(
    summary: Mapping[str, float], uncertainty: Mapping[str, float], thresholds: Mapping[str, float]
) -> bool:
    return bool(
        float(summary["final_fieldline_spread"]) <= float(thresholds["max_final_fieldline_spread"])
        and float(summary["improvement_fraction"]) >= float(thresholds["min_improvement_fraction"])
        and float(summary["max_abs_current_A"]) <= float(thresholds["max_abs_current_A"])
        and float(uncertainty["fieldline_spread_width"])
        <= float(thresholds["max_uncertainty_width"])
        and float(summary["p95_latency_us"]) <= float(thresholds["max_p95_latency_us"])
    )


def generate_report(
    *,
    steps: int | None = None,
    seed: int | None = None,
    config_path: str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Generate a deterministic report for the public stellarator benchmark."""
    if config_path is None:
        if steps is None:
            steps = 16
        if seed is None:
            seed = 271828
        config_payload = _default_config_payload(steps=int(steps), seed=int(seed))
    else:
        config_payload = load_benchmark_config(config_path)
        if steps is not None:
            config_payload["steps"] = int(steps)
        if seed is not None:
            config_payload["seed"] = int(seed)
        validate_benchmark_config(config_payload, load_config_schema())
    if int(config_payload["steps"]) < 4:
        raise ValueError("steps must be >= 4.")
    active_thresholds = _normalise_thresholds(
        thresholds if thresholds is not None else _require_mapping(config_payload, "thresholds")
    )
    scenario, physics_config, diagnostic_settings = _scenario_from_config(config_payload)
    replay_a = _run_replay(scenario, config=physics_config, diagnostics=diagnostic_settings)
    replay_b = _run_replay(scenario, config=physics_config, diagnostics=diagnostic_settings)
    deterministic = bool(replay_a["replay_signature"] == replay_b["replay_signature"])
    summary = replay_a["summary"]
    uncertainty = replay_a["uncertainty"]
    passes_thresholds = deterministic and _passes(summary, uncertainty, active_thresholds)
    flux_surface_s = float(diagnostic_settings["flux_surface_s"])
    report = {
        "stellarator_control_replay_benchmark": {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": "stellarator-control-replay-benchmark",
            "description": "Geometry-neutral reduced-order stellarator control replay benchmark.",
            "benchmark_config": {
                key: value for key, value in config_payload.items() if key != "thresholds"
            },
            "magnetic_configuration": scenario.magnetic_configuration.to_dict(),
            "control_objective": scenario.objective.to_dict(),
            "actuators": {
                **scenario.actuator_set.to_dict(),
                "primary_control_actuator": str(diagnostic_settings["primary_control_actuator"]),
                "max_abs_current_A": summary["max_abs_current_A"],
            },
            "replay": {
                "deterministic": deterministic,
                "signature": replay_a["replay_signature"],
                "steps": int(scenario.steps),
                "dt_s": float(scenario.dt_s),
                "trace": replay_a["trace"],
            },
            "metrics": summary,
            "uncertainty": uncertainty,
            "thresholds": {key: float(value) for key, value in active_thresholds.items()},
            "passes_thresholds": passes_thresholds,
            "data_provenance": {
                "geometry": "public_synthetic",
                "equilibrium": "reduced_order_vmec_like",
                "diagnostics": "deterministic_seeded_synthetic",
                "external_company_data": "none",
            },
            "physics_context": {
                "field_periods": int(physics_config.N_fp),
                "iss04_tau_E_s": round(iss04_scaling(physics_config, n_e=8.0, P_heat=12.0), 9),
                "effective_ripple": round(effective_ripple(physics_config, flux_surface_s), 9),
                "flux_surface_s": round(flux_surface_s, 9),
            },
            "limitations": [
                "This is not a production plant-control system.",
                "This is a reduced-order replay benchmark, not a VMEC/SIMSOPT replacement.",
                "No proprietary company geometry, coil, diagnostic, or operational data is used.",
                "The benchmark validates control-contract plumbing, replay determinism, actuator limits, and reporting discipline.",
            ],
        }
    }
    validate_report_against_schema(report, load_report_schema())
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    bench = report["stellarator_control_replay_benchmark"]
    metrics = bench["metrics"]
    uncertainty = bench["uncertainty"]
    thresholds = bench["thresholds"]
    lines = [
        "# Stellarator Control Replay Benchmark",
        "",
        f"- Schema: `{bench['schema_version']}`",
        f"- Deterministic replay: `{bench['replay']['deterministic']}`",
        f"- Replay signature: `{bench['replay']['signature']}`",
        f"- Threshold pass: `{'YES' if bench['passes_thresholds'] else 'NO'}`",
        "",
        "## Metrics",
        "",
        f"- Initial field-line spread: `{metrics['initial_fieldline_spread']:.6f}`",
        f"- Final field-line spread: `{metrics['final_fieldline_spread']:.6f}` (threshold `<= {thresholds['max_final_fieldline_spread']:.6f}`)",
        f"- Improvement fraction: `{metrics['improvement_fraction']:.6f}` (threshold `>= {thresholds['min_improvement_fraction']:.6f}`)",
        f"- Max absolute actuator current: `{metrics['max_abs_current_A']:.3f} A` (threshold `<= {thresholds['max_abs_current_A']:.3f} A`)",
        f"- P95 loop latency: `{metrics['p95_latency_us']:.3f} us` (threshold `<= {thresholds['max_p95_latency_us']:.3f} us`)",
        "",
        "## Uncertainty",
        "",
        f"- Method: `{uncertainty['method']}`",
        f"- Field-line spread p95 low/high: `{uncertainty['fieldline_spread_p95_low']:.6f}` / `{uncertainty['fieldline_spread_p95_high']:.6f}`",
        f"- Width: `{uncertainty['fieldline_spread_width']:.6f}` (threshold `<= {thresholds['max_uncertainty_width']:.6f}`)",
        "",
        "## Provenance",
        "",
        f"- Geometry: `{bench['data_provenance']['geometry']}`",
        f"- Equilibrium: `{bench['data_provenance']['equilibrium']}`",
        f"- External company data: `{bench['data_provenance']['external_company_data']}`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in bench["limitations"])
    lines.append("")
    return "\n".join(lines)


def load_report_schema() -> dict[str, Any]:
    schema_path = ROOT / "schemas" / "stellarator_control_replay_benchmark.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _require_path(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = payload
    for part in path:
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise ValueError("missing required report path: " + ".".join(path))
        cursor = cursor[part]
    return cursor


def validate_report_against_schema(report: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate the report against the repository JSON-schema contract.

    The project does not require the external ``jsonschema`` package at runtime,
    so this performs the strict contract checks used by the benchmark lane.
    """
    _require_path(report, ("stellarator_control_replay_benchmark",))
    root_properties = schema["properties"]
    for key in report:
        if key not in root_properties:
            raise ValueError(f"unexpected report key: {key}")
    bench = report["stellarator_control_replay_benchmark"]
    bench_schema = schema["properties"]["stellarator_control_replay_benchmark"]
    _validate_schema_contract(bench, bench_schema, path="", label="benchmark report")
    required = bench_schema["required"]
    allowed_keys = set(bench_schema["properties"])
    for key in bench:
        if key not in allowed_keys:
            raise ValueError(f"unexpected benchmark report key: {key}")
    for key in required:
        if key not in bench:
            raise ValueError(f"missing required benchmark key: {key}")
    if (
        bench["schema_version"]
        != schema["properties"]["stellarator_control_replay_benchmark"]["properties"][
            "schema_version"
        ]["const"]
    ):
        raise ValueError("unexpected schema_version")
    if not isinstance(bench["replay"]["trace"], list) or not bench["replay"]["trace"]:
        raise ValueError("replay.trace must be a non-empty list")
    for index, row in enumerate(bench["replay"]["trace"]):
        if not isinstance(row, Mapping):
            raise ValueError(f"replay.trace[{index}] must be an object")
        for key in TRACE_REQUIRED_KEYS:
            if key not in row:
                raise ValueError(f"missing required replay trace key: {key}")
        for key in row:
            if key not in TRACE_REQUIRED_KEYS:
                raise ValueError(f"unexpected replay trace key: {key}")
        if not isinstance(row["fault_active"], bool):
            raise ValueError("replay trace fault_active must be boolean")
        for key in TRACE_REQUIRED_KEYS:
            if key == "fault_active":
                continue
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f"replay trace value must be finite: {key}")
    for metric_key in (
        "initial_fieldline_spread",
        "final_fieldline_spread",
        "improvement_fraction",
        "max_abs_current_A",
        "p95_latency_us",
    ):
        value = float(bench["metrics"][metric_key])
        if not np.isfinite(value):
            raise ValueError(f"metric must be finite: {metric_key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "stellarator_control_replay_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "stellarator_control_replay_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(steps=args.steps, seed=args.seed, config_path=args.config)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    bench = report["stellarator_control_replay_benchmark"]
    metrics = bench["metrics"]
    print("Stellarator control replay benchmark complete.")
    print(
        "deterministic={deterministic}, final_fieldline_spread={final:.6f}, "
        "max_abs_current_A={current:.3f}, p95_latency_us={latency:.3f}, "
        "passes_thresholds={passes}".format(
            deterministic=bench["replay"]["deterministic"],
            final=metrics["final_fieldline_spread"],
            current=metrics["max_abs_current_A"],
            latency=metrics["p95_latency_us"],
            passes=bench["passes_thresholds"],
        )
    )
    if args.strict and not bench["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
