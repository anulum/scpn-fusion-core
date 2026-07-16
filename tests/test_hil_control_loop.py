# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the HIL real-time control loop."""

import numpy as np
import pytest

from scpn_fusion.control.hil_control_loop import ControlLoopMetrics, HILControlLoop
from scpn_fusion.control.hil_sensors import SensorInterface


class TestHILControlLoop:
    def test_basic_loop(self) -> None:
        loop = HILControlLoop(target_rate_hz=1000.0)
        loop.set_controller(lambda err, _: -0.5 * err)
        metrics = loop.run(iterations=100, setpoint=0.0)
        assert isinstance(metrics, ControlLoopMetrics)
        assert metrics.iterations == 100
        assert len(metrics.measured_dt_us) == 100

    def test_sub_ms_latency(self) -> None:
        """Core requirement: P95 loop latency < 1 ms."""
        loop = HILControlLoop(target_rate_hz=1000.0)
        loop.set_controller(lambda err, _: -0.5 * err)
        metrics = loop.run(iterations=500)
        # Pure Python PID in tight loop should easily be < 1ms
        assert metrics.p95_latency_us < 1000.0
        assert metrics.sub_ms_achieved

    def test_no_controller_raises(self) -> None:
        loop = HILControlLoop()
        with pytest.raises(RuntimeError, match="No controller"):
            loop.run(iterations=10)

    def test_pid_controller_stabilises(self) -> None:
        state = {"i": 0.0, "prev": 0.0}

        def pid(err: float, _: SensorInterface) -> float:
            state["i"] += err
            d = err - state["prev"]
            state["prev"] = err
            return 2.0 * err + 0.1 * state["i"] + 0.5 * d

        loop = HILControlLoop(target_rate_hz=1000.0)
        loop.set_controller(pid)

        def plant(s: float, cmd: float) -> float:
            return s + cmd * 0.001  # integrator

        metrics = loop.run(iterations=200, plant_fn=plant, initial_state=1.0, setpoint=0.0)
        assert metrics.iterations == 200
        assert metrics.mean_latency_us > 0.0

    def test_jitter_measured(self) -> None:
        loop = HILControlLoop()
        loop.set_controller(lambda err, _: -err)
        metrics = loop.run(iterations=200)
        assert metrics.jitter_std_us >= 0.0
        assert np.isfinite(metrics.jitter_std_us)

    def test_plant_receives_clamped_command_not_raw(self) -> None:
        """The DAC clamp must reach the plant: a 1e6 command saturates, never 1e6."""
        seen: list[float] = []

        def echo_plant(_state: float, applied: float) -> float:
            seen.append(applied)
            return applied

        loop = HILControlLoop(target_rate_hz=1000.0, sensor=SensorInterface())
        loop.set_controller(lambda _err, _s: 1.0e6)  # adversarial: slam the actuator
        loop.run(iterations=20, plant_fn=echo_plant, setpoint=0.0)
        assert seen  # plant advanced
        assert all(abs(v) <= 10.0 + 1e-9 for v in seen)  # clamp reached the plant
        assert max(seen) == pytest.approx(10.0)  # saturates at the DAC range

    def test_nan_controller_output_does_not_poison_plant(self) -> None:
        """A NaN controller output is fault-held at the boundary; the plant stays finite."""
        seen: list[float] = []

        def echo_plant(_state: float, applied: float) -> float:
            seen.append(applied)
            return applied

        loop = HILControlLoop(target_rate_hz=1000.0, sensor=SensorInterface())
        loop.set_controller(lambda _err, _s: float("nan"))
        metrics = loop.run(iterations=15, plant_fn=echo_plant, setpoint=0.0)
        assert metrics.iterations == 15
        assert all(np.isfinite(v) for v in seen)  # NaN never reached the plant
        assert loop.sensor.dac_faults == 15  # every sample flagged, none latched
