# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HIL Real-Time Control Loop
"""Real-time 1 kHz control loop with measured timing for the HIL harness.

Executes a user-supplied control callback at a target rate (default 1 kHz)
against a plant model and measures actual loop timing using
``time.perf_counter_ns``:

- :class:`ControlLoopMetrics` — measured per-iteration timing statistics.
- :class:`HILControlLoop` — sensor-read → controller → actuator-write → plant
  loop with P50/P95/P99 latency, jitter, and overrun reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .hil_sensors import SensorInterface


@dataclass
class ControlLoopMetrics:
    """Measured timing metrics from a HIL control loop run."""

    iterations: int
    target_dt_us: float
    measured_dt_us: list[float]
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    max_latency_us: float
    min_latency_us: float
    mean_latency_us: float
    jitter_std_us: float
    overrun_count: int
    overrun_fraction: float
    sub_ms_achieved: bool


class HILControlLoop:
    """Real-time 1 kHz control loop with measured timing.

    Executes a user-supplied control callback at a target rate (default 1 kHz)
    and measures actual loop timing using ``time.perf_counter_ns``.

    Parameters
    ----------
    target_rate_hz : float
        Target control loop rate (Hz). Default: 1000 (1 kHz).
    sensor : SensorInterface or None
        Sensor/actuator interface. Created with defaults if None.
    """

    def __init__(
        self,
        target_rate_hz: float = 1000.0,
        sensor: SensorInterface | None = None,
    ) -> None:
        self.target_dt_us = 1e6 / float(target_rate_hz)
        self.sensor = sensor or SensorInterface()
        self._control_fn: Callable[[float, SensorInterface], float] | None = None

    def set_controller(self, fn: Callable[[float, SensorInterface], float]) -> None:
        """Register the control callback: fn(error, sensor) -> command."""
        self._control_fn = fn

    def run(
        self,
        iterations: int = 1000,
        plant_fn: Callable[[float, float], float] | None = None,
        initial_state: float = 0.0,
        setpoint: float = 0.0,
    ) -> ControlLoopMetrics:
        """Execute the control loop and measure timing.

        Parameters
        ----------
        iterations : int
            Number of control loop iterations.
        plant_fn : callable or None
            Plant model: state_new = plant_fn(state, command).
            Default: simple integrator (state += command * dt).
        initial_state : float
            Initial plant state.
        setpoint : float
            Control target.
        """
        if self._control_fn is None:
            raise RuntimeError("No controller registered. Call set_controller() first.")

        if plant_fn is None:
            dt_s = self.target_dt_us / 1e6

            def _default_plant(state: float, cmd: float) -> float:
                return state + cmd * dt_s

            plant_fn = _default_plant

        state = float(initial_state)
        dt_measurements: list[float] = []

        for _ in range(iterations):
            t_start = time.perf_counter_ns()

            # Sensor read
            measured = self.sensor.read_adc(state)
            error = setpoint - measured

            # Controller
            command = self._control_fn(error, self.sensor)

            # Actuator write
            self.sensor.write_dac(command)

            # Plant update
            state = plant_fn(state, command)

            t_end = time.perf_counter_ns()
            dt_us = (t_end - t_start) / 1e3
            dt_measurements.append(dt_us)

        dt_arr = np.array(dt_measurements)
        overrun_threshold = self.target_dt_us
        overruns = int(np.sum(dt_arr > overrun_threshold))

        return ControlLoopMetrics(
            iterations=iterations,
            target_dt_us=self.target_dt_us,
            measured_dt_us=dt_measurements,
            p50_latency_us=float(np.percentile(dt_arr, 50)),
            p95_latency_us=float(np.percentile(dt_arr, 95)),
            p99_latency_us=float(np.percentile(dt_arr, 99)),
            max_latency_us=float(np.max(dt_arr)),
            min_latency_us=float(np.min(dt_arr)),
            mean_latency_us=float(np.mean(dt_arr)),
            jitter_std_us=float(np.std(dt_arr)),
            overrun_count=overruns,
            overrun_fraction=overruns / max(iterations, 1),
            sub_ms_achieved=float(np.percentile(dt_arr, 95)) < 1000.0,
        )
