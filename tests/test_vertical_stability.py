# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Tests for the vertical stability plant model."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.vertical_stability import PlantConfig, VerticalStabilityPlant


# ── test_free_evolution_diverges ──────────────────────────────────────
def test_free_evolution_diverges():
    """With u=0, Z should grow due to the positive-feedback instability."""
    cfg = PlantConfig(noise_std=0.0, sensor_noise_std=0.0)
    plant = VerticalStabilityPlant(cfg)
    z0 = 0.001
    plant.reset(z0=z0, dz0=0.0)

    for _ in range(100):
        plant.step(u=0.0)

    assert abs(plant.z) > abs(z0), (
        f"Expected divergence: final |z|={abs(plant.z):.6e} should exceed "
        f"initial |z0|={abs(z0):.6e}"
    )


# ── test_growth_rate ──────────────────────────────────────────────────
def test_growth_rate():
    """With noise=0 the envelope should grow as exp(γ·t); fitted rate
    should match γ within 10%."""
    gamma = 500.0
    cfg = PlantConfig(gamma=gamma, noise_std=0.0, sensor_noise_std=0.0)
    plant = VerticalStabilityPlant(cfg)
    z0 = 1e-4
    plant.reset(z0=z0, dz0=0.0)

    n_steps = 50
    zs = [z0]
    for _ in range(n_steps):
        plant.step(u=0.0)
        zs.append(plant.z)

    zs_arr = np.array(zs)
    ts = np.arange(len(zs_arr)) * cfg.dt

    # z(t) = z0 * cosh(γt) ≈ (z0/2) * exp(γt) for large γt
    # Fit log(|z|) = log(z0/2) + γ_fit * t  over the latter half
    half = len(ts) // 2
    log_z = np.log(np.abs(zs_arr[half:]))
    coeffs = np.polyfit(ts[half:], log_z, 1)
    gamma_fit = coeffs[0]

    assert abs(gamma_fit - gamma) / gamma < 0.10, (
        f"Fitted growth rate {gamma_fit:.1f} deviates >10% from γ={gamma}"
    )


# ── test_actuator_saturation ─────────────────────────────────────────
def test_actuator_saturation():
    """Control input exceeding u_max should be clamped."""
    cfg = PlantConfig(gamma=0.0, gain=1.0, noise_std=0.0, sensor_noise_std=0.0,
                      u_max=10.0)
    plant = VerticalStabilityPlant(cfg)
    plant.reset(z0=0.0, dz0=0.0)

    # With gamma=0 and gain=1: ddz = gain * u_clamped = u_clamped
    # After 1 step: dz = u_clamped * dt, z = dz * dt
    plant.step(u=100.0)  # Should clamp to 10.0

    # Expected: ddz = 10.0, dz = 10.0 * 1e-4 = 1e-3, z = 1e-3 * 1e-4 = 1e-7
    expected_dz = cfg.u_max * cfg.dt
    assert abs(plant.dz - expected_dz) < 1e-12, (
        f"dz={plant.dz:.6e} should equal {expected_dz:.6e} (saturated at u_max)"
    )


# ── test_disruption_detection ────────────────────────────────────────
def test_disruption_detection():
    """Starting near z_max with u=0, the plant should disrupt."""
    cfg = PlantConfig(noise_std=0.0, sensor_noise_std=0.0, z_max=0.1)
    plant = VerticalStabilityPlant(cfg)
    plant.reset(z0=0.09, dz0=0.0)

    assert not plant.disrupted, "Should not be disrupted initially at z=0.09"

    for _ in range(200):
        plant.step(u=0.0)
        if plant.disrupted:
            break

    assert plant.disrupted, (
        f"Expected disruption (|z|>{cfg.z_max}), but |z|={abs(plant.z):.6e}"
    )


# ── test_sensor_noise_present ────────────────────────────────────────
def test_sensor_noise_present():
    """With sensor noise, measured Z should differ from true Z."""
    cfg = PlantConfig(noise_std=0.0, sensor_noise_std=1e-3)
    plant = VerticalStabilityPlant(cfg)
    plant.reset(z0=0.01, dz0=0.0)

    mismatches = 0
    n_steps = 50
    for _ in range(n_steps):
        z_meas, _ = plant.step(u=0.0)
        if abs(z_meas - plant.z) > 1e-12:
            mismatches += 1

    assert mismatches > 0, (
        "Sensor noise should cause measured Z to differ from true Z"
    )


# ── test_deterministic_with_seed ─────────────────────────────────────
def test_deterministic_with_seed():
    """Same config and seed should produce identical trajectories."""
    cfg = PlantConfig()

    def run_trajectory() -> list[float]:
        plant = VerticalStabilityPlant(cfg)
        plant.reset(z0=0.001, dz0=0.0)
        trajectory = []
        for _ in range(50):
            plant.step(u=0.0)
            trajectory.append(plant.z)
        return trajectory

    traj1 = run_trajectory()
    traj2 = run_trajectory()

    np.testing.assert_array_equal(
        traj1, traj2,
        err_msg="Trajectories with same seed should be identical"
    )


# ── test_zero_noise_analytical ───────────────────────────────────────
def test_zero_noise_analytical():
    """With zero noise, compare numerical solution to the analytical
    z(t) = z0 * cosh(γt) + (dz0/γ) * sinh(γt) within 1% for 10 ms."""
    gamma = 500.0
    z0 = 1e-4
    dz0 = 0.0
    cfg = PlantConfig(gamma=gamma, noise_std=0.0, sensor_noise_std=0.0,
                      dt=1e-5)  # finer dt for accuracy
    plant = VerticalStabilityPlant(cfg)
    plant.reset(z0=z0, dz0=dz0)

    t_end = 0.010  # 10 ms
    n_steps = int(t_end / cfg.dt)

    for i in range(n_steps):
        plant.step(u=0.0)
        t = (i + 1) * cfg.dt
        z_analytical = z0 * np.cosh(gamma * t) + (dz0 / gamma) * np.sinh(gamma * t)

        if abs(z_analytical) > 1e-12:
            rel_err = abs(plant.z - z_analytical) / abs(z_analytical)
            assert rel_err < 0.01, (
                f"At t={t*1000:.3f} ms: numerical z={plant.z:.6e}, "
                f"analytical z={z_analytical:.6e}, rel_err={rel_err:.4f} > 1%"
            )


# ── test_reset ───────────────────────────────────────────────────────
def test_reset():
    """After running, reset() should restore initial state."""
    plant = VerticalStabilityPlant()
    plant.reset(z0=0.01, dz0=0.5)

    for _ in range(100):
        plant.step(u=0.0)

    assert plant.step_count == 100
    assert plant.t > 0.0

    plant.reset(z0=0.0, dz0=0.0)

    assert plant.z == 0.0
    assert plant.dz == 0.0
    assert plant.t == 0.0
    assert plant.step_count == 0
    assert not plant.disrupted


# ── test_growth_time_ms ──────────────────────────────────────────────
def test_growth_time_ms():
    """growth_time_ms property should equal 1000/gamma."""
    cfg = PlantConfig(gamma=500.0)
    plant = VerticalStabilityPlant(cfg)
    assert plant.growth_time_ms == pytest.approx(2.0, rel=1e-9)

    cfg2 = PlantConfig(gamma=200.0)
    plant2 = VerticalStabilityPlant(cfg2)
    assert plant2.growth_time_ms == pytest.approx(5.0, rel=1e-9)
