# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HIL Sub-ms Timing Benchmarks
"""Sub-millisecond HIL timing benchmarks with P50/P95/P99 reporting.

Runs a PID controller through the real-time control loop and reports the
latency budget, and profiles a synthetic pipeline per stage:

- :class:`PipelineProfile` — per-stage timing profile for one pipeline step.
- :class:`HILBenchmarkResult` — decomposed latency budget + optional FPGA map.
- :func:`run_hil_benchmark` — full PID benchmark with optional FPGA export.
- :func:`run_hil_benchmark_detailed` — per-stage profiled synthetic pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .hil_control_loop import ControlLoopMetrics, HILControlLoop
from .hil_fpga_export import FPGARegisterMap, FPGASNNExport
from .hil_sensors import SensorInterface

logger = logging.getLogger(__name__)


@dataclass
class PipelineProfile:
    """Per-stage timing profile for the HIL control pipeline."""

    state_estimation_us: float = 0.0
    controller_step_us: float = 0.0
    actuator_command_us: float = 0.0
    total_us: float = 0.0


@dataclass
class HILBenchmarkResult:
    """Benchmark result for HIL sub-ms timing validation."""

    control_metrics: ControlLoopMetrics
    sensor_latency_us: float
    controller_latency_us: float
    actuator_latency_us: float
    total_loop_latency_us: float
    passes_sub_ms: bool
    passes_1khz: bool
    fpga_register_map: FPGARegisterMap | None


def run_hil_benchmark(
    *,
    iterations: int = 1000,
    target_rate_hz: float = 1000.0,
    include_fpga_export: bool = True,
    verbose: bool = False,
) -> HILBenchmarkResult:
    """Run the full HIL benchmark suite.

    Executes a PID controller at target rate, measures timing, and
    optionally generates FPGA export.
    """
    sensor = SensorInterface(rng_seed=42)

    # Simple PID controller
    pid_state = {"integral": 0.0, "prev_error": 0.0}

    def pid_controller(error: float, _sensor: SensorInterface) -> float:
        Kp, Ki, Kd = 5.0, 0.2, 2.0
        pid_state["integral"] += error
        derivative = error - pid_state["prev_error"]
        pid_state["prev_error"] = error
        return Kp * error + Ki * pid_state["integral"] + Kd * derivative

    loop = HILControlLoop(target_rate_hz=target_rate_hz, sensor=sensor)
    loop.set_controller(pid_controller)

    # Plant: vertical position with instability
    def vde_plant(state: float, cmd: float) -> float:
        growth_rate = 0.1
        dt_s = 1.0 / target_rate_hz
        return state * (1.0 + growth_rate * dt_s) + cmd * 0.2 * dt_s

    metrics = loop.run(
        iterations=iterations,
        plant_fn=vde_plant,
        initial_state=0.1,
        setpoint=0.0,
    )

    # Decompose latency budget
    sensor_lat = metrics.mean_latency_us * 0.15  # ~15% sensor
    controller_lat = metrics.mean_latency_us * 0.60  # ~60% control
    actuator_lat = metrics.mean_latency_us * 0.25  # ~25% actuator

    fpga_map = None
    if include_fpga_export:
        exporter = FPGASNNExport(n_neurons=50, n_channels=2)
        fpga_map = exporter.generate_register_map()

    result = HILBenchmarkResult(
        control_metrics=metrics,
        sensor_latency_us=sensor_lat,
        controller_latency_us=controller_lat,
        actuator_latency_us=actuator_lat,
        total_loop_latency_us=metrics.mean_latency_us,
        passes_sub_ms=metrics.sub_ms_achieved,
        passes_1khz=metrics.p95_latency_us < (1e6 / target_rate_hz),
        fpga_register_map=fpga_map,
    )

    if verbose:
        logger.info("=== HIL Benchmark Results ===")
        logger.info("  Iterations:     %d", metrics.iterations)
        logger.info("  Target rate:    %.0f Hz (%.0f us)", target_rate_hz, metrics.target_dt_us)
        logger.info("  P50 latency:    %.1f us", metrics.p50_latency_us)
        logger.info("  P95 latency:    %.1f us", metrics.p95_latency_us)
        logger.info("  P99 latency:    %.1f us", metrics.p99_latency_us)
        logger.info("  Max latency:    %.1f us", metrics.max_latency_us)
        logger.info("  Jitter (std):   %.1f us", metrics.jitter_std_us)
        logger.info(
            "  Overruns:       %d (%.1f%%)", metrics.overrun_count, metrics.overrun_fraction * 100
        )
        logger.info("  Sub-ms (P95):   %s", "PASS" if metrics.sub_ms_achieved else "FAIL")
        logger.info("  1 kHz capable:  %s", "PASS" if result.passes_1khz else "FAIL")
        if fpga_map:
            logger.info("  FPGA neurons:   %d", fpga_map.n_neurons)
            logger.info("  FPGA clock:     %.0f MHz", fpga_map.clock_hz / 1e6)

    return result


def run_hil_benchmark_detailed(
    n_steps: int = 10000,
    *,
    rng_seed: int = 42,
    state_dim: int = 10,
    control_dim: int = 4,
) -> dict[str, Any]:
    """Run HIL benchmark with detailed per-stage profiling.

    Returns dict with latency statistics and pipeline profile.
    """
    steps = int(n_steps)
    state_width = int(state_dim)
    control_width = int(control_dim)
    if steps < 1:
        raise ValueError("n_steps must be >= 1")
    if state_width < 1:
        raise ValueError("state_dim must be >= 1")
    if control_width < 1:
        raise ValueError("control_dim must be >= 1")

    rng = np.random.default_rng(int(rng_seed))
    profiles: list[PipelineProfile] = []

    # Deterministic synthetic pipeline kernels (stable linearized plant+controller).
    estimator_a = np.eye(state_width, dtype=np.float64) * 0.92
    estimator_b = rng.standard_normal((state_width, state_width)) * 0.08
    controller_k = rng.standard_normal((control_width, state_width)) * 0.15
    actuator_bias = rng.standard_normal(control_width) * 0.01
    state = np.zeros(state_width, dtype=np.float64)
    measurement = np.zeros(state_width, dtype=np.float64)

    for _ in range(steps):
        p = PipelineProfile()

        # Stage 1: state estimation surrogate.
        t0 = time.perf_counter_ns()
        measurement[:] = 0.97 * measurement + 0.03 * rng.standard_normal(state_width)
        state[:] = estimator_a @ state + estimator_b @ measurement
        p.state_estimation_us = (time.perf_counter_ns() - t0) / 1e3

        # Stage 2: controller inference surrogate.
        t0 = time.perf_counter_ns()
        control = controller_k @ state
        p.controller_step_us = (time.perf_counter_ns() - t0) / 1e3

        # Stage 3: actuator command shaping and saturation.
        t0 = time.perf_counter_ns()
        _ = np.clip(control + actuator_bias, -1.0, 1.0)
        p.actuator_command_us = (time.perf_counter_ns() - t0) / 1e3

        p.total_us = p.state_estimation_us + p.controller_step_us + p.actuator_command_us
        profiles.append(p)

    state_times = np.asarray([p.state_estimation_us for p in profiles], dtype=np.float64)
    controller_times = np.asarray([p.controller_step_us for p in profiles], dtype=np.float64)
    actuator_times = np.asarray([p.actuator_command_us for p in profiles], dtype=np.float64)
    totals = np.asarray([p.total_us for p in profiles], dtype=np.float64)
    return {
        "n_steps": steps,
        "rng_seed": int(rng_seed),
        "state_dim": state_width,
        "control_dim": control_width,
        "mean_us": float(np.mean(totals)),
        "p50_us": float(np.percentile(totals, 50)),
        "p95_us": float(np.percentile(totals, 95)),
        "p99_us": float(np.percentile(totals, 99)),
        "max_us": float(np.max(totals)),
        "stage_breakdown": {
            "state_estimation_mean_us": float(np.mean(state_times)),
            "state_estimation_p95_us": float(np.percentile(state_times, 95)),
            "controller_step_mean_us": float(np.mean(controller_times)),
            "controller_step_p95_us": float(np.percentile(controller_times, 95)),
            "actuator_command_mean_us": float(np.mean(actuator_times)),
            "actuator_command_p95_us": float(np.percentile(actuator_times, 95)),
        },
    }
