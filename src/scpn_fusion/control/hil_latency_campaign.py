# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Simulated HIL Sensor-to-Actuator Latency Campaign
"""Simulated sensor-to-actuator HIL latency campaign with fault injection.

Measures a host-side simulated ADC/DAC sensor-to-actuator loop across nominal
and degraded scenarios (sensor dropout, noisy sensor, actuator saturation,
non-finite controller output), reporting per-stage latency, fail-closed
fallback behaviour, and an explicit claim boundary:

- :func:`run_sensor_to_actuator_hil_latency_campaign` — top-level campaign.
- :func:`_run_sensor_to_actuator_hil_scenario` — one scenario measurement.
- :func:`_hil_measurement_for_step` — one sensor→actuator pipeline step.
- :func:`_latency_summary_us` — P50/P95/P99/max/mean summariser.

This is not a physical HIL rig, FPGA bitstream, plant CODAC, or actuator
hardware timing claim.
"""

from __future__ import annotations

import time
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .hil_sensors import SensorInterface

FloatArray: TypeAlias = NDArray[np.float64]

_SENSOR_TO_ACTUATOR_HIL_STAGES = (
    "adc_capture",
    "input_validation",
    "state_estimation",
    "controller_step",
    "actuator_guard",
    "dac_write",
    "output_serialization",
)
_SENSOR_TO_ACTUATOR_HIL_SCENARIOS = (
    "nominal",
    "sensor_dropout",
    "noisy_sensor",
    "actuator_saturation",
    "controller_nonfinite",
)


def _latency_summary_us(values: FloatArray) -> dict[str, float]:
    """Summarise a latency sample vector in microseconds.

    Parameters
    ----------
    values
        One-dimensional latency samples in microseconds.

    Returns
    -------
    dict[str, float]
        P50, P95, P99, maximum, and mean latency values in microseconds.
    """
    return {
        "p50_us": float(np.percentile(values, 50)),
        "p95_us": float(np.percentile(values, 95)),
        "p99_us": float(np.percentile(values, 99)),
        "max_us": float(np.max(values)),
        "mean_us": float(np.mean(values)),
    }


def _hil_measurement_for_step(
    *,
    sensor: SensorInterface,
    true_state: FloatArray,
    previous_state: FloatArray,
    estimator: FloatArray,
    controller: FloatArray,
    previous_actuator: FloatArray,
    scenario: str,
    step_index: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], FloatArray, FloatArray, bool, str, int, int]:
    stage_us: dict[str, float] = {}
    fallback = False
    fallback_reason = "nominal"
    safe_outputs = 0
    serialized_bytes = 0

    t0 = time.perf_counter_ns()
    measurement = np.asarray(
        [sensor.read_adc(float(value)) for value in true_state], dtype=np.float64
    )
    if scenario == "sensor_dropout" and step_index % 8 == 0:
        measurement[1] = np.nan
    elif scenario == "noisy_sensor":
        measurement += rng.normal(0.0, 0.012, size=measurement.shape)
    elif scenario == "actuator_saturation" and step_index % 9 == 0:
        measurement[3] = 1.45
    elif scenario == "controller_nonfinite" and step_index % 10 == 0:
        measurement[0] = np.inf
    stage_us["adc_capture"] = (time.perf_counter_ns() - t0) / 1e3

    t0 = time.perf_counter_ns()
    if not np.all(np.isfinite(measurement)):
        measurement = np.where(np.isfinite(measurement), measurement, previous_state)
        fallback = True
        fallback_reason = "invalid_sensor_sample_replaced"
    elif scenario == "noisy_sensor" and step_index % 6 == 0:
        fallback = True
        fallback_reason = "noisy_sensor_envelope_safe_mode"
    measurement = np.clip(measurement, -1.5, 1.5)
    stage_us["input_validation"] = (time.perf_counter_ns() - t0) / 1e3

    t0 = time.perf_counter_ns()
    estimated_state = (
        estimator @ previous_state + (np.eye(previous_state.size) - estimator) @ measurement
    )
    estimated_state = np.clip(estimated_state, -1.5, 1.5)
    stage_us["state_estimation"] = (time.perf_counter_ns() - t0) / 1e3

    t0 = time.perf_counter_ns()
    command = controller @ estimated_state
    if scenario == "actuator_saturation" and step_index % 9 == 0:
        command = command + np.sign(command + 1.0e-9) * 2.0
    if scenario == "controller_nonfinite" and step_index % 10 == 0:
        command[0] = np.nan
    stage_us["controller_step"] = (time.perf_counter_ns() - t0) / 1e3

    t0 = time.perf_counter_ns()
    if command.shape != previous_actuator.shape or not np.all(np.isfinite(command)):
        guarded_command = np.zeros_like(previous_actuator)
        fallback = True
        fallback_reason = "controller_nonfinite_fail_closed"
    else:
        guarded_command = np.clip(command, -1.0, 1.0)
        if bool(np.any(np.abs(command) > 1.0)):
            fallback = True
            fallback_reason = "actuator_saturation_clamped"
    stage_us["actuator_guard"] = (time.perf_counter_ns() - t0) / 1e3

    t0 = time.perf_counter_ns()
    max_change = sensor.dac.slew_rate_v_per_us
    delta = np.clip(guarded_command - previous_actuator, -max_change, max_change)
    vmin, vmax = sensor.dac.voltage_range
    slew_limited = np.clip(previous_actuator + delta, vmin, vmax)
    if np.all(np.isfinite(slew_limited)) and np.all(np.abs(slew_limited) <= vmax):
        safe_outputs = 1
    stage_us["dac_write"] = (time.perf_counter_ns() - t0) / 1e3

    t0 = time.perf_counter_ns()
    payload = {
        "step": int(step_index),
        "commands": [float(value) for value in slew_limited.tolist()],
        "fallback": fallback,
        "reason": fallback_reason,
    }
    serialized_bytes = len(str(payload).encode("utf-8"))
    stage_us["output_serialization"] = (time.perf_counter_ns() - t0) / 1e3

    return (
        stage_us,
        estimated_state,
        slew_limited,
        fallback,
        fallback_reason,
        safe_outputs,
        serialized_bytes,
    )


def _run_sensor_to_actuator_hil_scenario(
    *,
    scenario: str,
    n_steps: int,
    state_dim: int,
    actuator_count: int,
    rng_seed: int,
) -> dict[str, Any]:
    if scenario not in _SENSOR_TO_ACTUATOR_HIL_SCENARIOS:
        raise ValueError(f"Unknown HIL latency scenario: {scenario}")
    rng = np.random.default_rng(rng_seed)
    sensor = SensorInterface(rng_seed=rng_seed)
    estimator = np.eye(state_dim, dtype=np.float64) * 0.82
    controller = rng.standard_normal((actuator_count, state_dim)) * 0.18
    state = np.zeros(state_dim, dtype=np.float64)
    actuator = np.zeros(actuator_count, dtype=np.float64)
    stage_samples = {
        stage: np.zeros(n_steps, dtype=np.float64) for stage in _SENSOR_TO_ACTUATOR_HIL_STAGES
    }
    total_us = np.zeros(n_steps, dtype=np.float64)
    fallback_count = 0
    safe_output_count = 0
    fallback_reasons: dict[str, int] = {}
    byte_counts = np.zeros(n_steps, dtype=np.float64)

    for index in range(n_steps):
        phase = 0.021 * float(index)
        true_state = np.asarray(
            [
                0.45 * np.sin(phase),
                0.35 * np.cos(0.7 * phase),
                0.18 * np.sin(1.3 * phase),
                0.09 * np.cos(0.4 * phase),
            ],
            dtype=np.float64,
        )
        if state_dim > 4:
            pad = np.linspace(-0.05, 0.05, state_dim - 4, dtype=np.float64)
            true_state = np.concatenate((true_state, pad))
        step = _hil_measurement_for_step(
            sensor=sensor,
            true_state=true_state,
            previous_state=state,
            estimator=estimator,
            controller=controller,
            previous_actuator=actuator,
            scenario=scenario,
            step_index=index,
            rng=rng,
        )
        stage_us, state, actuator, fallback, fallback_reason, safe_outputs, serialized_bytes = step
        for stage, value in stage_us.items():
            stage_samples[stage][index] = value
        total_us[index] = float(sum(stage_us.values()))
        if fallback:
            fallback_count += 1
            fallback_reasons[fallback_reason] = fallback_reasons.get(fallback_reason, 0) + 1
        safe_output_count += safe_outputs
        byte_counts[index] = float(serialized_bytes)

    return {
        "scenario": scenario,
        "n_steps": int(n_steps),
        "state_dim": int(state_dim),
        "actuator_count": int(actuator_count),
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / max(n_steps, 1)),
        "fallback_reasons": fallback_reasons,
        "safe_output_rate": float(safe_output_count / max(n_steps, 1)),
        "mean_payload_bytes": float(np.mean(byte_counts)),
        "latency": _latency_summary_us(total_us),
        "stages": {stage: _latency_summary_us(values) for stage, values in stage_samples.items()},
        "passes_semantics": bool(
            safe_output_count == n_steps and (scenario == "nominal" or fallback_count > 0)
        ),
    }


def run_sensor_to_actuator_hil_latency_campaign(
    *,
    n_steps: int = 512,
    state_dim: int = 4,
    actuator_count: int = 256,
    rng_seed: int = 2026,
) -> dict[str, Any]:
    """Measure a simulated HIL sensor-to-actuator latency campaign.

    Parameters
    ----------
    n_steps
        Number of measured control-loop samples per scenario.
    state_dim
        Number of ADC-backed plant-state channels in the synthetic sensor vector.
    actuator_count
        Number of DAC-backed actuator command channels to emit. Use ``256`` to
        cover the `>200` actuator scaling target surface.
    rng_seed
        Seed for deterministic controller matrices and noisy-sensor scenarios.

    Returns
    -------
    dict[str, Any]
        Structured latency, stage-breakdown, fallback, and claim-boundary
        evidence for a host-side simulated ADC/DAC loop.

    Raises
    ------
    ValueError
        If ``n_steps``, ``state_dim``, or ``actuator_count`` is outside the
        accepted benchmark range.
    """
    steps = int(n_steps)
    width = int(state_dim)
    count = int(actuator_count)
    if steps < 32:
        raise ValueError("n_steps must be >= 32")
    if width < 4:
        raise ValueError("state_dim must be >= 4")
    if count < 1:
        raise ValueError("actuator_count must be >= 1")

    scenarios = {
        scenario: _run_sensor_to_actuator_hil_scenario(
            scenario=scenario,
            n_steps=steps,
            state_dim=width,
            actuator_count=count,
            rng_seed=int(rng_seed) + offset,
        )
        for offset, scenario in enumerate(_SENSOR_TO_ACTUATOR_HIL_SCENARIOS)
    }
    nominal = scenarios["nominal"]
    all_semantics_pass = all(bool(row["passes_semantics"]) for row in scenarios.values())
    return {
        "schema": "scpn-fusion-core.simulated_hil_sensor_to_actuator_latency.v1",
        "status": "measured_simulated_hil",
        "hardware_status": "simulated_host_adc_dac_loop",
        "n_steps": steps,
        "state_dim": width,
        "actuator_count": count,
        "rng_seed": int(rng_seed),
        "scenarios": scenarios,
        "nominal_latency": nominal["latency"],
        "passes_thresholds": bool(
            nominal["latency"]["p95_us"] <= 1000.0
            and nominal["latency"]["p99_us"] <= 2000.0
            and all_semantics_pass
        ),
        "thresholds": {
            "max_nominal_p95_us": 1000.0,
            "max_nominal_p99_us": 2000.0,
            "degraded_scenarios_require_safe_outputs": True,
            "required_actuator_count": 256,
        },
        "claim_boundary": (
            "Measured host-side simulated ADC/DAC sensor-to-actuator scaffold; "
            "not a physical HIL rig, FPGA bitstream, plant CODAC, or actuator hardware timing claim."
        ),
    }
