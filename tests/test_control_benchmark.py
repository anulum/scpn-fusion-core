# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Tests for validation.control_benchmark module."""
from __future__ import annotations

import sys
import os
import numpy as np
import pytest

# Ensure the repo root is on sys.path so the validation package is importable.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from validation.control_benchmark import (
    SCENARIOS,
    BenchmarkResult,
    PlantConfig,
    Scenario,
    compute_metrics,
)

# ------------------------------------------------------------------
# 1. SCENARIOS has exactly 6 entries
# ------------------------------------------------------------------

def test_scenarios_count():
    """SCENARIOS list must contain exactly 6 benchmark scenarios."""
    assert len(SCENARIOS) == 6


def test_scenario_names_unique():
    """Every scenario must have a unique name."""
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names))


def test_scenario_types():
    """Each entry in SCENARIOS must be a Scenario with a PlantConfig."""
    for s in SCENARIOS:
        assert isinstance(s, Scenario)
        assert isinstance(s.plant_config, PlantConfig)


# ------------------------------------------------------------------
# 2. compute_metrics returns the correct keys
# ------------------------------------------------------------------

_EXPECTED_KEYS = {
    "settling_time_ms",
    "max_overshoot_mm",
    "steady_state_error_mm",
    "rms_control_effort",
    "peak_control_effort",
    "disrupted",
}


def test_compute_metrics_keys():
    """compute_metrics must return all expected metric keys."""
    z = np.zeros(100)
    u = np.zeros(100)
    result = compute_metrics(z, u, dt=1e-4, z_max=0.1)
    assert set(result.keys()) == _EXPECTED_KEYS


# ------------------------------------------------------------------
# 3. BenchmarkResult.to_dict() works
# ------------------------------------------------------------------

def test_benchmark_result_to_dict():
    """to_dict() must return a flat dictionary with the right fields."""
    br = BenchmarkResult(
        controller_name="pid",
        scenario_name="step_5mm",
        settling_time_ms=1.5,
        max_overshoot_mm=5.0,
        steady_state_error_mm=0.01,
        rms_control_effort=2.3,
        peak_control_effort=8.0,
        disrupted=False,
        wall_time_us=12.0,
    )
    d = br.to_dict()
    assert isinstance(d, dict)
    assert d["controller"] == "pid"
    assert d["scenario"] == "step_5mm"
    assert d["settling_time_ms"] == 1.5
    assert d["max_overshoot_mm"] == 5.0
    assert d["steady_state_error_mm"] == 0.01
    assert d["rms_control_effort"] == 2.3
    assert d["peak_control_effort"] == 8.0
    assert d["disrupted"] is False
    assert d["wall_time_us_per_step"] == 12.0


def test_benchmark_result_to_dict_trajectories_excluded():
    """Raw trajectories must NOT appear in to_dict() output."""
    br = BenchmarkResult(
        controller_name="snn",
        scenario_name="step_noisy",
        settling_time_ms=2.0,
        max_overshoot_mm=6.0,
        steady_state_error_mm=0.1,
        rms_control_effort=3.0,
        peak_control_effort=9.0,
        disrupted=False,
        trajectory_z=[0.1, 0.2],
        trajectory_u=[1.0, 2.0],
    )
    d = br.to_dict()
    assert "trajectory_z" not in d
    assert "trajectory_u" not in d


# ------------------------------------------------------------------
# 4. compute_metrics with zero trajectory gives correct values
# ------------------------------------------------------------------

def test_zero_trajectory_settling_time():
    """A zero trajectory is already settled at t=0."""
    z = np.zeros(1000)
    u = np.zeros(1000)
    m = compute_metrics(z, u, dt=1e-4, z_max=0.1)
    assert m["settling_time_ms"] == 0.0


def test_zero_trajectory_overshoot():
    """Zero trajectory has zero overshoot."""
    z = np.zeros(1000)
    u = np.zeros(1000)
    m = compute_metrics(z, u, dt=1e-4, z_max=0.1)
    assert m["max_overshoot_mm"] == 0.0


def test_zero_trajectory_steady_state_error():
    """Zero trajectory has zero steady-state error."""
    z = np.zeros(1000)
    u = np.zeros(1000)
    m = compute_metrics(z, u, dt=1e-4, z_max=0.1)
    assert m["steady_state_error_mm"] == 0.0


def test_zero_trajectory_control_effort():
    """Zero control signal has zero RMS and peak effort."""
    z = np.zeros(1000)
    u = np.zeros(1000)
    m = compute_metrics(z, u, dt=1e-4, z_max=0.1)
    assert m["rms_control_effort"] == 0.0
    assert m["peak_control_effort"] == 0.0


def test_zero_trajectory_not_disrupted():
    """Zero trajectory must not be flagged as disrupted."""
    z = np.zeros(1000)
    u = np.zeros(1000)
    m = compute_metrics(z, u, dt=1e-4, z_max=0.1)
    assert m["disrupted"] is False


# ------------------------------------------------------------------
# 5. Additional edge-case / sanity checks
# ------------------------------------------------------------------

def test_disrupted_detection():
    """A trajectory exceeding z_max must be flagged as disrupted."""
    z = np.zeros(100)
    z[50] = 0.2  # exceeds z_max=0.1
    u = np.zeros(100)
    m = compute_metrics(z, u, dt=1e-4, z_max=0.1)
    assert m["disrupted"] is True


def test_settling_never_settles():
    """If |z| never drops below 1mm, settling_time equals full duration."""
    n = 1000
    dt = 1e-4
    z = np.full(n, 0.005)  # 5mm constant -- never settles
    u = np.zeros(n)
    m = compute_metrics(z, u, dt=dt, z_max=0.1)
    expected_ms = n * dt * 1000.0
    assert m["settling_time_ms"] == pytest.approx(expected_ms)


def test_known_rms_control_effort():
    """RMS of a constant signal equals its absolute value."""
    z = np.zeros(100)
    u = np.full(100, 3.0)
    m = compute_metrics(z, u, dt=1e-4, z_max=0.1)
    assert m["rms_control_effort"] == pytest.approx(3.0)
    assert m["peak_control_effort"] == pytest.approx(3.0)
