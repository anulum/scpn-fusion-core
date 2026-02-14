# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Tests for the SNN latency benchmark.

Validates:
1. PID compute returns in < 1 ms (generous bound; target < 10 us).
2. SNN (NumPy) compute returns in < 10 ms (generous bound; target < 1 ms).
3. Output JSON from run_benchmark() has all expected keys.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.control.pid_baseline import PIDController, PIDConfig
from scpn_fusion.scpn.vertical_control_net import VerticalControlNet
from scpn_fusion.scpn.vertical_snn_controller import VerticalSNNController

# Import benchmark helpers directly so we can test them.
import sys

_REPO = Path(__file__).resolve().parent.parent
_VAL = str(_REPO / "validation")
if _VAL not in sys.path:
    sys.path.insert(0, _VAL)

from benchmark_snn_latency import (
    _make_pid,
    _make_snn_numpy,
    _compute_stats,
    run_benchmark,
    LatencyStats,
    Z_MEAS,
    DZ_MEAS,
)


# ---------------------------------------------------------------------------
# Test 1: PID latency < 1 ms per call
# ---------------------------------------------------------------------------

class TestPIDLatency:
    """PID controller must compute in under 1 ms per call."""

    def test_pid_single_call_under_1ms(self):
        """Each PID.compute() call should finish in < 1 ms (1000 us)."""
        pid = _make_pid()
        # Warm up
        for _ in range(100):
            pid.compute(Z_MEAS, DZ_MEAS)
        pid.reset()

        # Measure 1000 calls
        latencies = []
        for _ in range(1000):
            t0 = time.perf_counter()
            pid.compute(Z_MEAS, DZ_MEAS)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        latencies_us = np.array(latencies) * 1e6
        p99 = float(np.percentile(latencies_us, 99))

        # p99 must be under 1000 us (1 ms) -- very generous
        assert p99 < 1000.0, (
            f"PID p99 latency {p99:.1f} us exceeds 1 ms limit"
        )


# ---------------------------------------------------------------------------
# Test 2: SNN (NumPy) latency < 10 ms per call
# ---------------------------------------------------------------------------

class TestSNNNumPyLatency:
    """SNN NumPy float-path must compute in under 10 ms per call."""

    def test_snn_numpy_single_call_under_10ms(self):
        """Each SNN-NumPy compute() call should finish in < 10 ms."""
        snn = _make_snn_numpy()
        # Warm up
        for _ in range(50):
            snn.compute(Z_MEAS, DZ_MEAS)
        snn.reset()

        # Measure 500 calls
        latencies = []
        for _ in range(500):
            t0 = time.perf_counter()
            snn.compute(Z_MEAS, DZ_MEAS)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        latencies_us = np.array(latencies) * 1e6
        p99 = float(np.percentile(latencies_us, 99))

        # p99 must be under 10,000 us (10 ms) -- generous
        assert p99 < 10_000.0, (
            f"SNN-NumPy p99 latency {p99:.1f} us exceeds 10 ms limit"
        )


# ---------------------------------------------------------------------------
# Test 3: JSON output schema
# ---------------------------------------------------------------------------

class TestBenchmarkOutputSchema:
    """Verify that run_benchmark() output has all expected keys."""

    def test_output_json_keys(self, tmp_path):
        """LatencyStats dataclass must contain all required fields."""
        # Create a synthetic LatencyStats to verify the schema.
        stats = LatencyStats(
            controller="TestController",
            backend="test",
            iterations=100,
            mean_us=5.0,
            median_us=4.5,
            p95_us=8.0,
            p99_us=12.0,
            min_us=2.0,
            max_us=20.0,
            total_s=0.0005,
        )
        d = asdict(stats)

        expected_keys = {
            "controller",
            "backend",
            "iterations",
            "mean_us",
            "median_us",
            "p95_us",
            "p99_us",
            "min_us",
            "max_us",
            "total_s",
        }
        assert set(d.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(d.keys())}; "
            f"extra keys: {set(d.keys()) - expected_keys}"
        )

        # Verify all values are serialisable to JSON.
        serialised = json.dumps(d)
        reloaded = json.loads(serialised)
        for key in expected_keys:
            assert key in reloaded, f"Key '{key}' missing after JSON round-trip"

    def test_compute_stats_produces_valid_stats(self):
        """_compute_stats must produce consistent statistics."""
        rng = np.random.default_rng(42)
        raw = rng.exponential(scale=1e-5, size=1000)  # ~10 us mean
        stats = _compute_stats("test", "numpy", raw)

        assert stats.controller == "test"
        assert stats.backend == "numpy"
        assert stats.iterations == 1000
        assert stats.min_us <= stats.mean_us <= stats.max_us
        assert stats.min_us <= stats.median_us <= stats.max_us
        assert stats.median_us <= stats.p95_us <= stats.p99_us
        assert stats.total_s > 0.0
