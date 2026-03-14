# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Orbit Following Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math

import numpy as np

from scpn_fusion.core.orbit_following import (
    GuidingCenterOrbit,
    MonteCarloEnsemble,
    OrbitClassifier,
    SlowingDown,
    first_orbit_loss,
)


def mock_b_field(R, Z):
    # Pure toroidal field (no poloidal -> no drifts in passing)
    B0 = 5.0
    R0 = 6.0
    B_phi = B0 * R0 / R
    # Add small poloidal field for trapping
    B_R = -0.1 * Z
    B_Z = 0.1 * (R - R0)
    return B_R, B_Z, B_phi


def test_passing_orbit():
    # Pitch angle 0 -> purely parallel -> passing
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, 0.0, 6.2, 0.0)

    # Evolve a bit
    for _ in range(10):
        orbit.step(mock_b_field, 1e-6)

    assert orbit.v_par > 0.0  # Never reversed
    assert orbit.R > 0.0


def test_trapped_orbit():
    # Pitch angle pi/2 -> purely perpendicular -> trapped instantly
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, math.pi / 2 - 0.1, 6.2, 0.0)

    # We must have computed mu
    orbit.step(mock_b_field, 1e-6)
    assert orbit.mu > 0.0

    # In a full simulation it would reverse v_par


def test_orbit_classifier():
    R = np.ones(10) * 6.0
    Z = np.zeros(10)
    v_par_pass = np.ones(10)
    v_par_trap = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 1.0, 1.0, 1.0, 1.0])

    assert OrbitClassifier.classify(R, Z, v_par_pass, 10.0, 5.0) == "passing"
    assert OrbitClassifier.classify(R, Z, v_par_trap, 10.0, 5.0) == "trapped"

    # Lost
    R_lost = np.array([6.0, 7.0, 11.0, 12.0])
    v_lost = np.ones(4)
    assert OrbitClassifier.classify(R_lost, Z[:4], v_lost, 10.0, 5.0) == "lost"


def test_first_orbit_loss():
    # ITER: large Ip -> low loss
    iter_loss = first_orbit_loss(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0)
    assert iter_loss < 0.05

    # NSTX: small Ip, small a -> higher loss
    nstx_loss = first_orbit_loss(R0=0.9, a=0.6, B0=1.0, Ip_MA=1.0)
    assert nstx_loss > iter_loss
    assert nstx_loss > 0.5


def test_slowing_down():
    tau = SlowingDown.tau_sd(Te_keV=20.0, ne_20=1.0, Z_eff=1.5)
    # tau ~ 0.1 * 20^1.5 / 1 = 0.1 * 89 ~ 8.9 s (rough heuristic)
    assert tau > 1.0

    vc = SlowingDown.critical_velocity(20.0, 1.0)

    # 3.5 MeV alpha v ~ 1.3e7 m/s
    v_alpha = 1.3e7
    f_i, f_e = SlowingDown.heating_partition(v_alpha, vc)

    # Fast alpha heats electrons primarily
    assert f_e > f_i


def test_mc_ensemble():
    ens = MonteCarloEnsemble(10, 3500.0, 6.2, 2.0, 5.3)
    ens.initialize(np.ones(10), np.ones(10), np.linspace(0, 1, 10))

    assert len(ens.particles) == 10

    res = ens.follow(mock_b_field)
    assert res.n_passing + res.n_trapped + res.n_lost == 10
