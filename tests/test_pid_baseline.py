# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Tests for PID baseline controller against the vertical stability plant."""
from __future__ import annotations

import sys
import os
import numpy as np
import pytest

# Ensure the repo root is on sys.path so imports work without editable install.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from scpn_fusion.control.vertical_stability import VerticalStabilityPlant, PlantConfig
from scpn_fusion.control.pid_baseline import PIDController, PIDConfig, tune_ziegler_nichols


# ------------------------------------------------------------------
# Helper: run a closed-loop simulation for N steps
# ------------------------------------------------------------------

def _run_loop(
    plant: VerticalStabilityPlant,
    pid: PIDController,
    n_steps: int,
) -> tuple[list[float], list[float]]:
    """Run plant+PID closed loop, return (z_trajectory, u_trajectory)."""
    z_traj: list[float] = []
    u_traj: list[float] = []
    for _ in range(n_steps):
        z_meas, dz_meas = plant.step(pid.prev_output)
        u = pid.compute(z_meas, dz_meas)
        z_traj.append(plant.z)
        u_traj.append(u)
    return z_traj, u_traj


# ------------------------------------------------------------------
# 1. PID stabilizes a 5mm step within 1000 steps (100 ms)
# ------------------------------------------------------------------

def test_pid_stabilizes_step():
    """Start at z0=5mm, run PID+plant for 1000 steps, verify |z| < 1mm."""
    pcfg = PlantConfig(gamma=500, gain=1e6, dt=1e-4, noise_std=0, sensor_noise_std=0)
    plant = VerticalStabilityPlant(pcfg)
    plant.reset(z0=0.005)

    pid = PIDController(PIDConfig(dt=pcfg.dt, u_max=pcfg.u_max))

    z_traj, _ = _run_loop(plant, pid, 1000)
    assert abs(z_traj[-1]) < 1e-3, (
        f"PID failed to stabilize: final |z| = {abs(z_traj[-1])*1000:.3f} mm"
    )


# ------------------------------------------------------------------
# 2. Record maximum overshoot during recovery from 5mm step
# ------------------------------------------------------------------

def test_pid_overshoot():
    """Max |z| during recovery from 5mm should be bounded."""
    pcfg = PlantConfig(gamma=500, gain=1e6, dt=1e-4, noise_std=0, sensor_noise_std=0)
    plant = VerticalStabilityPlant(pcfg)
    plant.reset(z0=0.005)

    pid = PIDController(PIDConfig(dt=pcfg.dt, u_max=pcfg.u_max))

    z_traj, _ = _run_loop(plant, pid, 1000)
    max_overshoot_mm = max(abs(z) for z in z_traj) * 1000.0
    # Overshoot should not exceed initial displacement by more than 100%
    # (i.e., max |z| < 10mm = 2x initial 5mm)
    assert max_overshoot_mm < 10.0, (
        f"Overshoot too large: {max_overshoot_mm:.2f} mm"
    )


# ------------------------------------------------------------------
# 3. Steady-state error after 2000 steps should be < 0.1mm
# ------------------------------------------------------------------

def test_pid_steady_state_error():
    """After 2000 steps (200ms), verify |z| < 0.1mm."""
    pcfg = PlantConfig(gamma=500, gain=1e6, dt=1e-4, noise_std=0, sensor_noise_std=0)
    plant = VerticalStabilityPlant(pcfg)
    plant.reset(z0=0.005)

    pid = PIDController(PIDConfig(dt=pcfg.dt, u_max=pcfg.u_max))

    z_traj, _ = _run_loop(plant, pid, 2000)
    # Mean |z| over last 20% of trajectory
    tail = z_traj[int(0.8 * len(z_traj)):]
    ss_error_mm = np.mean([abs(z) for z in tail]) * 1000.0
    assert ss_error_mm < 0.1, (
        f"Steady-state error too large: {ss_error_mm:.4f} mm"
    )


# ------------------------------------------------------------------
# 4. Anti-windup: with very low saturation, integral should stay bounded
# ------------------------------------------------------------------

def test_anti_windup():
    """With u_max=1 (low), integral should not blow up due to anti-windup."""
    pcfg = PlantConfig(gamma=500, gain=1e6, dt=1e-4, noise_std=0, sensor_noise_std=0)
    plant = VerticalStabilityPlant(pcfg)
    plant.reset(z0=0.005)

    pid = PIDController(PIDConfig(dt=pcfg.dt, u_max=1.0, anti_windup=True))

    integrals: list[float] = []
    for _ in range(2000):
        z_meas, dz_meas = plant.step(pid.prev_output)
        pid.compute(z_meas, dz_meas)
        integrals.append(abs(pid.integral))

    max_integral = max(integrals)
    # With anti-windup the integral should remain bounded, not grow without
    # limit.  A reasonable bound: integral < 1.0 (i.e. no runaway).
    assert max_integral < 1.0, (
        f"Anti-windup failed: max |integral| = {max_integral:.4f}"
    )


# ------------------------------------------------------------------
# 5. Ziegler-Nichols auto-tuning stabilizes the plant
# ------------------------------------------------------------------

def test_ziegler_nichols_tuning():
    """Auto-tuned PID should stabilize z0=5mm -> |z_final| < 1mm."""
    gamma = 500.0
    gain_val = 1e6
    dt = 1e-4
    pcfg = PlantConfig(gamma=gamma, gain=gain_val, dt=dt, noise_std=0, sensor_noise_std=0)
    plant = VerticalStabilityPlant(pcfg)
    plant.reset(z0=0.005)

    auto_cfg = tune_ziegler_nichols(gamma, gain_val, dt)
    auto_cfg.u_max = pcfg.u_max
    pid = PIDController(auto_cfg)

    z_traj, _ = _run_loop(plant, pid, 1000)
    assert abs(z_traj[-1]) < 1e-3, (
        f"ZN-tuned PID failed: final |z| = {abs(z_traj[-1])*1000:.3f} mm"
    )


# ------------------------------------------------------------------
# 6. Disruption avoidance: z never exceeds z_max
# ------------------------------------------------------------------

def test_pid_disruption_avoidance():
    """5mm step should never cause |z| to exceed z_max (0.1m)."""
    pcfg = PlantConfig(gamma=500, gain=1e6, dt=1e-4, noise_std=0, sensor_noise_std=0)
    plant = VerticalStabilityPlant(pcfg)
    plant.reset(z0=0.005)

    pid = PIDController(PIDConfig(dt=pcfg.dt, u_max=pcfg.u_max))

    z_traj, _ = _run_loop(plant, pid, 1000)
    for i, z in enumerate(z_traj):
        assert abs(z) <= pcfg.z_max, (
            f"Disruption at step {i}: |z| = {abs(z)*1000:.2f} mm > "
            f"z_max = {pcfg.z_max*1000:.1f} mm"
        )


# ------------------------------------------------------------------
# 7. Reset clears integral and previous state
# ------------------------------------------------------------------

def test_reset():
    """After running, reset() should clear integral and prev state."""
    pid = PIDController(PIDConfig())
    # Drive some state into the controller
    pid.compute(0.005, 0.1)
    pid.compute(0.003, 0.05)
    assert pid.integral != 0.0 or pid.prev_error != 0.0

    pid.reset()
    assert pid.integral == 0.0
    assert pid.prev_error == 0.0
    assert pid.prev_output == 0.0
