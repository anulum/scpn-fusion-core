# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Facade contract tests for the HIL test harness re-export surface.

The behavioural tests live beside each responsibility submodule
(``test_hil_sensors``, ``test_hil_control_loop``, ``test_hil_fpga_export``,
``test_hil_benchmarks``, ``test_hil_latency_campaign``). These tests only pin
the public facade surface so downstream imports remain stable.
"""

from scpn_fusion.control import hil_harness
from scpn_fusion.control.hil_benchmarks import (
    HILBenchmarkResult,
    PipelineProfile,
    run_hil_benchmark,
    run_hil_benchmark_detailed,
)
from scpn_fusion.control.hil_control_loop import ControlLoopMetrics, HILControlLoop
from scpn_fusion.control.hil_demo_runner import HILDemoRunner
from scpn_fusion.control.hil_fpga_export import (
    FPGARegisterMap,
    FPGASNNExport,
    SNNNeuronConfig,
)
from scpn_fusion.control.hil_latency_campaign import (
    FloatArray,
    run_sensor_to_actuator_hil_latency_campaign,
)
from scpn_fusion.control.hil_sensors import ADCConfig, DACConfig, SensorInterface


def test_facade_exports_expected_surface() -> None:
    assert set(hil_harness.__all__) == {
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
    }


def test_facade_reexports_are_submodule_objects() -> None:
    # Every re-exported name must be the exact object defined in its submodule.
    assert hil_harness.ADCConfig is ADCConfig
    assert hil_harness.DACConfig is DACConfig
    assert hil_harness.SensorInterface is SensorInterface
    assert hil_harness.ControlLoopMetrics is ControlLoopMetrics
    assert hil_harness.HILControlLoop is HILControlLoop
    assert hil_harness.SNNNeuronConfig is SNNNeuronConfig
    assert hil_harness.FPGARegisterMap is FPGARegisterMap
    assert hil_harness.FPGASNNExport is FPGASNNExport
    assert hil_harness.PipelineProfile is PipelineProfile
    assert hil_harness.HILBenchmarkResult is HILBenchmarkResult
    assert hil_harness.run_hil_benchmark is run_hil_benchmark
    assert hil_harness.run_hil_benchmark_detailed is run_hil_benchmark_detailed
    assert hil_harness.FloatArray is FloatArray
    assert (
        hil_harness.run_sensor_to_actuator_hil_latency_campaign
        is run_sensor_to_actuator_hil_latency_campaign
    )
    assert hil_harness.HILDemoRunner is HILDemoRunner


def test_facade_names_all_importable() -> None:
    for name in hil_harness.__all__:
        assert hasattr(hil_harness, name), name
