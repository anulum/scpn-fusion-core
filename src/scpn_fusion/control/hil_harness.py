# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Hardware-in-the-Loop Test Harness
r"""Hardware-in-the-Loop (HIL) test harness with sub-millisecond validation.

Public facade re-exporting the HIL harness surface from its responsibility
submodules:

1. **SensorInterface** — abstract ADC/DAC sensor/actuator layer
   (:mod:`~scpn_fusion.control.hil_sensors`).
2. **HILControlLoop** — real-time 1 kHz control loop with measured jitter
   (:mod:`~scpn_fusion.control.hil_control_loop`).
3. **FPGASNNExport** — FPGA-ready SNN register-level export
   (:mod:`~scpn_fusion.control.hil_fpga_export`).
4. **run_hil_benchmark** — sub-ms timing validation with P50/P95/P99 reporting
   (:mod:`~scpn_fusion.control.hil_benchmarks`).
5. **run_sensor_to_actuator_hil_latency_campaign** — simulated sensor-to-actuator
   latency campaign (:mod:`~scpn_fusion.control.hil_latency_campaign`).

This harness uses ``time.perf_counter_ns`` for nanosecond-resolution timing
to demonstrate sub-millisecond control loop latency on the host CPU. True
FPGA deployment requires synthesis via Vivado HLS or equivalent; the export
format provides the register map and port definitions.
"""

from __future__ import annotations

from .hil_benchmarks import (
    HILBenchmarkResult,
    PipelineProfile,
    run_hil_benchmark,
    run_hil_benchmark_detailed,
)
from .hil_control_loop import ControlLoopMetrics, HILControlLoop
from .hil_demo_runner import HILDemoRunner
from .hil_fpga_export import FPGARegisterMap, FPGASNNExport, SNNNeuronConfig
from .hil_latency_campaign import FloatArray, run_sensor_to_actuator_hil_latency_campaign
from .hil_sensors import ADCConfig, DACConfig, SensorInterface

__all__ = [
    "ADCConfig",
    "ControlLoopMetrics",
    "DACConfig",
    "FPGARegisterMap",
    "FPGASNNExport",
    "FloatArray",
    "HILBenchmarkResult",
    "HILControlLoop",
    "HILDemoRunner",
    "PipelineProfile",
    "SNNNeuronConfig",
    "SensorInterface",
    "run_hil_benchmark",
    "run_hil_benchmark_detailed",
    "run_sensor_to_actuator_hil_latency_campaign",
]
