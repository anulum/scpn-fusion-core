# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the HIL sub-ms timing benchmarks."""

import logging
from typing import Any

import numpy as np
import pytest

from scpn_fusion.control.hil_benchmarks import (
    HILBenchmarkResult,
    run_hil_benchmark,
    run_hil_benchmark_detailed,
)


class TestHILBenchmark:
    def test_full_benchmark(self) -> None:
        result = run_hil_benchmark(iterations=200, verbose=False)
        assert isinstance(result, HILBenchmarkResult)
        assert result.total_loop_latency_us > 0.0
        assert result.passes_sub_ms

    def test_benchmark_with_fpga_export(self) -> None:
        result = run_hil_benchmark(iterations=100, include_fpga_export=True)
        assert result.fpga_register_map is not None
        assert result.fpga_register_map.n_neurons > 0

    def test_benchmark_without_fpga(self) -> None:
        result = run_hil_benchmark(iterations=100, include_fpga_export=False)
        assert result.fpga_register_map is None

    def test_latency_budget_decomposition(self) -> None:
        result = run_hil_benchmark(iterations=100)
        total = result.sensor_latency_us + result.controller_latency_us + result.actuator_latency_us
        assert abs(total - result.total_loop_latency_us) < 0.1

    def test_sub_ms_p95(self) -> None:
        """The key deliverable: demonstrate sub-ms control loop latency."""
        result = run_hil_benchmark(iterations=1000)
        print("\n=== HIL Benchmark ===")
        print(f"    P50: {result.control_metrics.p50_latency_us:.1f} us")
        print(f"    P95: {result.control_metrics.p95_latency_us:.1f} us")
        print(f"    P99: {result.control_metrics.p99_latency_us:.1f} us")
        print(f"    Sub-ms: {'PASS' if result.passes_sub_ms else 'FAIL'}")
        assert result.passes_sub_ms

    def test_verbose_logging_with_fpga(self, caplog: pytest.LogCaptureFixture) -> None:
        # Covers the verbose logging block and the truthy `if fpga_map` branch.
        with caplog.at_level(logging.INFO, logger="scpn_fusion.control.hil_benchmarks"):
            result = run_hil_benchmark(iterations=64, verbose=True, include_fpga_export=True)
        assert result.fpga_register_map is not None
        assert "HIL Benchmark Results" in caplog.text
        assert "FPGA neurons" in caplog.text

    def test_verbose_logging_without_fpga(self, caplog: pytest.LogCaptureFixture) -> None:
        # Covers the verbose logging block with the falsy `if fpga_map` branch.
        with caplog.at_level(logging.INFO, logger="scpn_fusion.control.hil_benchmarks"):
            result = run_hil_benchmark(iterations=64, verbose=True, include_fpga_export=False)
        assert result.fpga_register_map is None
        assert "HIL Benchmark Results" in caplog.text
        assert "FPGA neurons" not in caplog.text


class TestHILDetailedBenchmark:
    def test_detailed_benchmark_schema_and_ranges(self) -> None:
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

    def test_detailed_benchmark_rejects_invalid_shape_or_steps(self) -> None:
        cases: tuple[tuple[dict[str, Any], str], ...] = (
            ({"n_steps": 0}, "n_steps"),
            ({"state_dim": 0}, "state_dim"),
            ({"control_dim": 0}, "control_dim"),
        )
        for kwargs, match in cases:
            with pytest.raises(ValueError, match=match):
                run_hil_benchmark_detailed(**kwargs)
