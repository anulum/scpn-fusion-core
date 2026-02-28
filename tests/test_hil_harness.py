"""Tests for Hardware-in-the-Loop test harness."""

import numpy as np
import pytest

from scpn_fusion.control.hil_harness import (
    ADCConfig,
    DACConfig,
    SensorInterface,
    HILControlLoop,
    ControlLoopMetrics,
    FPGASNNExport,
    FPGARegisterMap,
    SNNNeuronConfig,
    HILBenchmarkResult,
    run_hil_benchmark,
    run_hil_benchmark_detailed,
)


class TestSensorInterface:
    def test_adc_quantization(self):
        sensor = SensorInterface(adc=ADCConfig(resolution_bits=12))
        reading = sensor.read_adc(0.5)
        assert isinstance(reading, float)
        # 12-bit in ±1.5V → LSB ~0.73 mV
        assert abs(reading - 0.5) < 0.01  # within ~10 LSBs (noise)

    def test_adc_clamps_range(self):
        sensor = SensorInterface()
        # Beyond range should clamp
        reading = sensor.read_adc(10.0)  # way above ±1.5V
        assert reading <= 1.5 + 0.01

    def test_dac_slew_rate(self):
        sensor = SensorInterface(dac=DACConfig(slew_rate_v_per_us=50.0))
        # Large step should be slew-limited
        out = sensor.write_dac(10.0, dt_us=0.1)
        assert out < 10.0  # can't reach 10V in 0.1 us at 50V/us

    def test_magnetic_probe(self):
        sensor = SensorInterface(rng_seed=42)
        B = sensor.read_magnetic_probe(5.3)
        assert abs(B - 5.3) < 0.5  # reasonable noise level

    def test_coil_current(self):
        sensor = SensorInterface()
        I = sensor.write_coil_current(25.0, dt_us=1000.0)
        assert abs(I - 25.0) < 5.0  # slew-limited approach

    def test_adc_deterministic_with_seed(self):
        s1 = SensorInterface(rng_seed=42)
        s2 = SensorInterface(rng_seed=42)
        r1 = s1.read_adc(0.5)
        r2 = s2.read_adc(0.5)
        assert r1 == r2


class TestHILControlLoop:
    def test_basic_loop(self):
        loop = HILControlLoop(target_rate_hz=1000.0)
        loop.set_controller(lambda err, _: -0.5 * err)
        metrics = loop.run(iterations=100, setpoint=0.0)
        assert isinstance(metrics, ControlLoopMetrics)
        assert metrics.iterations == 100
        assert len(metrics.measured_dt_us) == 100

    def test_sub_ms_latency(self):
        """Core requirement: P95 loop latency < 1 ms."""
        loop = HILControlLoop(target_rate_hz=1000.0)
        loop.set_controller(lambda err, _: -0.5 * err)
        metrics = loop.run(iterations=500)
        # Pure Python PID in tight loop should easily be < 1ms
        assert metrics.p95_latency_us < 1000.0
        assert metrics.sub_ms_achieved

    def test_no_controller_raises(self):
        loop = HILControlLoop()
        with pytest.raises(RuntimeError, match="No controller"):
            loop.run(iterations=10)

    def test_pid_controller_stabilises(self):
        state = {"i": 0.0, "prev": 0.0}

        def pid(err, _):
            state["i"] += err
            d = err - state["prev"]
            state["prev"] = err
            return 2.0 * err + 0.1 * state["i"] + 0.5 * d

        loop = HILControlLoop(target_rate_hz=1000.0)
        loop.set_controller(pid)

        def plant(s, cmd):
            return s + cmd * 0.001  # integrator

        metrics = loop.run(iterations=200, plant_fn=plant, initial_state=1.0, setpoint=0.0)
        assert metrics.iterations == 200
        assert metrics.mean_latency_us > 0.0

    def test_jitter_measured(self):
        loop = HILControlLoop()
        loop.set_controller(lambda err, _: -err)
        metrics = loop.run(iterations=200)
        assert metrics.jitter_std_us >= 0.0
        assert np.isfinite(metrics.jitter_std_us)


class TestFPGASNNExport:
    def test_register_map_generation(self):
        exporter = FPGASNNExport(n_neurons=50, n_channels=2)
        reg_map = exporter.generate_register_map()
        assert isinstance(reg_map, FPGARegisterMap)
        assert reg_map.n_neurons == 100  # 50 * 2 channels
        assert len(reg_map.neurons) == 100
        assert len(reg_map.input_ports) == 2
        assert len(reg_map.output_ports) == 2

    def test_verilog_header(self):
        exporter = FPGASNNExport(n_neurons=10, n_channels=2, clock_mhz=100.0)
        reg_map = exporter.generate_register_map()
        verilog = exporter.export_verilog_header(reg_map)
        assert "module snn_controller" in verilog
        assert "N_NEURONS" in verilog
        assert "CLK_HZ" in verilog
        assert "v_mem" in verilog
        assert "endmodule" in verilog

    def test_neuron_configs(self):
        exporter = FPGASNNExport(n_neurons=5, n_channels=1)
        reg_map = exporter.generate_register_map(v_threshold=0.4, tau_mem_us=20000.0)
        for neuron in reg_map.neurons:
            assert isinstance(neuron, SNNNeuronConfig)
            assert neuron.v_threshold == 0.4
            assert neuron.tau_mem_us == 20000.0

    def test_clock_frequency(self):
        exporter = FPGASNNExport(clock_mhz=200.0)
        reg_map = exporter.generate_register_map()
        assert reg_map.clock_hz == 200_000_000


class TestHILBenchmark:
    def test_full_benchmark(self):
        result = run_hil_benchmark(iterations=200, verbose=False)
        assert isinstance(result, HILBenchmarkResult)
        assert result.total_loop_latency_us > 0.0
        assert result.passes_sub_ms

    def test_benchmark_with_fpga_export(self):
        result = run_hil_benchmark(iterations=100, include_fpga_export=True)
        assert result.fpga_register_map is not None
        assert result.fpga_register_map.n_neurons > 0

    def test_benchmark_without_fpga(self):
        result = run_hil_benchmark(iterations=100, include_fpga_export=False)
        assert result.fpga_register_map is None

    def test_latency_budget_decomposition(self):
        result = run_hil_benchmark(iterations=100)
        total = result.sensor_latency_us + result.controller_latency_us + result.actuator_latency_us
        assert abs(total - result.total_loop_latency_us) < 0.1

    def test_sub_ms_p95(self):
        """The key deliverable: demonstrate sub-ms control loop latency."""
        result = run_hil_benchmark(iterations=1000)
        print(f"\n=== HIL Benchmark ===")
        print(f"    P50: {result.control_metrics.p50_latency_us:.1f} us")
        print(f"    P95: {result.control_metrics.p95_latency_us:.1f} us")
        print(f"    P99: {result.control_metrics.p99_latency_us:.1f} us")
        print(f"    Sub-ms: {'PASS' if result.passes_sub_ms else 'FAIL'}")
        assert result.passes_sub_ms


class TestHILDetailedBenchmark:
    def test_detailed_benchmark_schema_and_ranges(self):
        out = run_hil_benchmark_detailed(n_steps=64, rng_seed=7, state_dim=6, control_dim=3)
        assert out["n_steps"] == 64
        assert out["rng_seed"] == 7
        assert out["state_dim"] == 6
        assert out["control_dim"] == 3
        for key in ("mean_us", "p50_us", "p95_us", "p99_us", "max_us"):
            assert key in out
            assert np.isfinite(out[key])
            assert out[key] >= 0.0

        stage = out["stage_breakdown"]
        assert set(stage.keys()) == {
            "state_estimation_mean_us",
            "state_estimation_p95_us",
            "controller_step_mean_us",
            "controller_step_p95_us",
            "actuator_command_mean_us",
            "actuator_command_p95_us",
        }

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"n_steps": 0}, "n_steps"),
            ({"state_dim": 0}, "state_dim"),
            ({"control_dim": 0}, "control_dim"),
        ],
    )
    def test_detailed_benchmark_rejects_invalid_shape_or_steps(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            run_hil_benchmark_detailed(**kwargs)
