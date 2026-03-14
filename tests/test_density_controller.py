# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Density Controller Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_fusion.control.density_controller import (
    DensityController,
    FuelingOptimizer,
    KalmanDensityEstimator,
    ParticleTransportModel,
)


def test_particle_transport_model_sources():
    model = ParticleTransportModel(n_rho=20, R0=6.2, a=2.0)

    gas = model.gas_puff_source(rate=1e21, penetration_depth=0.1)
    assert np.all(gas >= 0)
    assert gas[-1] > gas[0]  # Edge peaked

    pellet = model.pellet_source(speed_ms=500.0, radius_mm=2.0)
    assert np.all(pellet >= 0)
    assert pellet[10] > pellet[0]  # Deeply deposited

    nbi = model.nbi_source(beam_energy_keV=100.0, power_MW=10.0)
    assert np.all(nbi >= 0)
    assert nbi[5] > nbi[-1]  # Core peaked

    pump = model.cryopump_sink(pump_speed=10.0, ne_edge=1e19)
    assert pump[-1] > 0.0
    assert pump[0] == 0.0


def test_particle_transport_model_step():
    model = ParticleTransportModel(n_rho=10)
    ne = np.ones(10) * 1e19
    sources = np.zeros(10)
    sources[-1] = 1e20  # Strong edge puff

    # Use extremely small dt to avoid explicit diffusion instability (CFL limit)
    ne_new = model.step(ne, sources, dt=1e-6)

    # Check shape
    assert ne_new.shape == (10,)
    # Just verify no NaN or Inf (stability check)
    assert np.all(np.isfinite(ne_new))


def test_density_controller():
    model = ParticleTransportModel(n_rho=10)
    ctrl = DensityController(model, dt_control=0.01)

    # Target high density
    target = np.ones(10) * 5e19
    ctrl.set_target(target)

    # Measure low density
    meas_low = np.ones(10) * 1e19
    cmd = ctrl.step(meas_low)

    assert cmd.gas_puff_rate > 0.0
    assert cmd.cryo_pump_speed == 0.0

    # Measure high density -> pump
    meas_high = np.ones(10) * 6e19
    ctrl.integral_error = 0.0  # reset for clean test
    cmd_high = ctrl.step(meas_high)

    assert cmd_high.gas_puff_rate == 0.0
    assert cmd_high.cryo_pump_speed > 0.0


def test_greenwald_limit_override():
    model = ParticleTransportModel(n_rho=10)
    ctrl = DensityController(model, dt_control=0.01)

    target = np.ones(10) * 5e19
    ctrl.set_target(target)

    # Set n_GW such that meas is over Greenwald
    ctrl.set_constraints(n_GW=1e19, gas_max=1e22, pellet_freq_max=10.0, pump_max=10.0)

    meas = np.ones(10) * 2e19  # Higher than n_GW = 1e19 -> f_GW > 1.0
    cmd = ctrl.step(meas)

    # Should aggressively pump, regardless of target
    assert cmd.gas_puff_rate == 0.0
    assert cmd.cryo_pump_speed == ctrl.pump_max


def test_kalman_estimator():
    est = KalmanDensityEstimator(n_rho=20, n_chords=5)

    ne_pred = np.ones(20) * 1e19
    meas = np.ones(5) * 1e19 * 2.0  # chord integrated mock
    angles = np.zeros(5)

    ne_upd = est.update(ne_pred, meas, angles)
    assert ne_upd.shape == (20,)


def test_fueling_optimizer():
    opt = FuelingOptimizer()
    sched = opt.optimize_pellet_sequence(np.zeros(10), np.ones(10), n_pellets=3, time_horizon=1.0)

    assert len(sched.times) == 3
    assert len(sched.speeds) == 3
    assert len(sched.sizes) == 3
    assert sched.times[0] == 0.25  # 1.0 / 4
