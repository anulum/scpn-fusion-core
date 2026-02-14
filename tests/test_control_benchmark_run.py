# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Tests for the head-to-head PID vs SNN control benchmark runner."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure imports work regardless of how pytest is invoked.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scpn_fusion.control.pid_baseline import PIDController, PIDConfig
from scpn_fusion.control.vertical_stability import VerticalStabilityPlant, PlantConfig
from scpn_fusion.scpn.vertical_control_net import VerticalControlNet
from scpn_fusion.scpn.vertical_snn_controller import VerticalSNNController
from validation.control_benchmark import SCENARIOS, compute_metrics


# ------------------------------------------------------------------
# 1. Both controllers can be instantiated and produce finite outputs
# ------------------------------------------------------------------

class TestControllerInstantiation:
    """Verify both controllers instantiate and produce finite control signals."""

    def test_pid_instantiates_and_computes(self):
        """PID controller returns a finite float for a sample input."""
        pid = PIDController(PIDConfig())
        u = pid.compute(z_measured=0.005, dz_measured=0.0)
        assert isinstance(u, float)
        assert np.isfinite(u)

    def test_snn_instantiates_and_computes(self):
        """SNN controller returns a finite float for a sample input."""
        vcn = VerticalControlNet()
        vcn.create_net()
        snn = VerticalSNNController(vcn, force_numpy=True, seed=42)
        u = snn.compute(z_measured=0.005, dz_measured=0.0)
        assert isinstance(u, float)
        assert np.isfinite(u)

    def test_both_controllers_produce_opposing_sign(self):
        """Both controllers should oppose positive displacement (u < 0 for z > 0)."""
        pid = PIDController(PIDConfig())
        u_pid = pid.compute(z_measured=0.005, dz_measured=0.0)

        vcn = VerticalControlNet()
        vcn.create_net()
        snn = VerticalSNNController(vcn, force_numpy=True, seed=42)
        u_snn = snn.compute(z_measured=0.005, dz_measured=0.0)

        # PID has negative gains so u < 0 for positive z
        assert u_pid < 0.0, f"PID should oppose positive z, got u={u_pid}"
        # SNN also opposes displacement
        assert u_snn <= 0.0, f"SNN should oppose positive z, got u={u_snn}"


# ------------------------------------------------------------------
# 2. Full benchmark produces valid JSON-serializable results
# ------------------------------------------------------------------

class TestBenchmarkOutput:
    """Verify the benchmark runner produces valid structured output."""

    @pytest.fixture(scope="class")
    def benchmark_results(self):
        """Run the benchmark on the first scenario only for speed."""
        from validation.run_control_benchmark import _run_closed_loop, _make_pid, _make_snn

        scenario = SCENARIOS[0]
        pid = _make_pid(scenario)
        snn = _make_snn()

        pid_r = _run_closed_loop(scenario, pid, "PID")
        snn_r = _run_closed_loop(scenario, snn, "SNN")
        return pid_r, snn_r

    def test_results_are_json_serializable(self, benchmark_results):
        """BenchmarkResult.to_dict() must produce JSON-serializable output."""
        pid_r, snn_r = benchmark_results
        pid_d = pid_r.to_dict()
        snn_d = snn_r.to_dict()

        # This will raise if not serializable
        pid_json = json.dumps(pid_d)
        snn_json = json.dumps(snn_d)

        assert isinstance(pid_json, str)
        assert isinstance(snn_json, str)

    def test_results_have_all_metric_keys(self, benchmark_results):
        """Both results must contain all required metric keys."""
        required_keys = {
            "controller",
            "scenario",
            "settling_time_ms",
            "max_overshoot_mm",
            "steady_state_error_mm",
            "rms_control_effort",
            "peak_control_effort",
            "disrupted",
            "wall_time_us_per_step",
        }
        pid_r, snn_r = benchmark_results
        assert required_keys.issubset(set(pid_r.to_dict().keys()))
        assert required_keys.issubset(set(snn_r.to_dict().keys()))


# ------------------------------------------------------------------
# 3. Metrics dict has all required keys for a trivial trajectory
# ------------------------------------------------------------------

class TestMetricsKeys:
    """Verify compute_metrics returns the complete key set."""

    _REQUIRED_KEYS = {
        "settling_time_ms",
        "max_overshoot_mm",
        "steady_state_error_mm",
        "rms_control_effort",
        "peak_control_effort",
        "disrupted",
    }

    def test_compute_metrics_all_keys_present(self):
        """compute_metrics must return all required keys."""
        z = np.zeros(500)
        u = np.ones(500) * 0.1
        m = compute_metrics(z, u, dt=1e-4, z_max=0.1)
        assert self._REQUIRED_KEYS == set(m.keys())

    def test_compute_metrics_values_are_finite(self):
        """All numeric values returned by compute_metrics must be finite."""
        z = np.random.randn(500) * 0.001
        u = np.random.randn(500) * 0.5
        m = compute_metrics(z, u, dt=1e-4, z_max=0.1)
        for key, val in m.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), f"{key} is not finite: {val}"
