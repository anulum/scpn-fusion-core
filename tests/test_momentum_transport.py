# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Momentum Transport Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.momentum_transport import (
    MomentumTransportSolver,
    RotationDiagnostics,
    exb_shearing_rate,
    nbi_torque,
    radial_electric_field,
    turbulence_suppression_factor,
)


def test_nbi_torque():
    P_nbi = np.ones(50) * 1e6
    torque_co = nbi_torque(P_nbi, R0=6.2, v_beam=1e6, theta_inj_deg=30.0)
    assert np.all(torque_co > 0.0)

    torque_counter = nbi_torque(P_nbi, R0=6.2, v_beam=1e6, theta_inj_deg=-30.0)
    assert np.all(torque_counter < 0.0)


def test_solver_viscous_damping():
    rho = np.linspace(0, 1, 50)
    solver = MomentumTransportSolver(rho, R0=6.2, a=2.0, B0=5.3)

    # Initialize with some rotation
    solver.omega_phi = np.ones(50) * 1e4

    chi_i = np.ones(50) * 1.0
    ne = np.ones(50) * 5.0
    Ti = np.ones(50) * 5.0
    T_zero = np.zeros(50)

    # Step forward
    for _ in range(10):
        solver.step(0.1, chi_i, ne, Ti, T_zero, T_zero)

    # Should damp towards zero
    assert solver.omega_phi[25] < 1e4


def test_solver_steady_state():
    rho = np.linspace(0, 1, 50)
    solver = MomentumTransportSolver(rho, R0=6.2, a=2.0, B0=5.3)

    chi_i = np.ones(50) * 1.0
    ne = np.ones(50) * 5.0
    Ti = np.ones(50) * 5.0

    T_nbi = np.ones(50) * 1.0
    T_zero = np.zeros(50)

    # Drive to steady state
    for _ in range(100):
        solver.step(0.1, chi_i, ne, Ti, T_nbi, T_zero)

    assert solver.omega_phi[0] > 0.0


def test_exb_shear_suppression():
    rho = np.linspace(0, 1, 50)
    omega = 1e5 * (1.0 - rho**2)  # Peaked rotation
    B_theta = 0.5 * rho

    rate = exb_shearing_rate(omega, B_theta, B0=5.3, R0=6.2, rho=rho, a=2.0)
    # Should be non-zero where shear is non-zero
    assert rate[25] > 0.0

    gamma_max = np.ones(50) * 1e4
    supp = turbulence_suppression_factor(rate, gamma_max)

    # Where rate > gamma, supp should be < 0.5
    if rate[25] > 1e4:
        assert supp[25] < 0.5


def test_radial_electric_field():
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 5.0
    Ti = 5.0 * (1.0 - rho**2)
    omega = 1e4 * np.ones(50)
    B_theta = 0.5 * rho

    Er = radial_electric_field(ne, Ti, omega, B_theta, B0=5.3, R0=6.2, rho=rho, a=2.0)

    # Expect Er to be composed of dp/dr (negative) and v*B (positive)
    assert len(Er) == 50


def test_radial_electric_field_includes_poloidal_flow_force_balance_term() -> None:
    """Poloidal flow contributes the v_theta B_phi term in radial force balance."""
    rho = np.linspace(0.0, 1.0, 5)
    ne = np.full(5, 5.0)
    ti = np.full(5, 4.0)
    omega = np.zeros(5)
    b_theta = np.full(5, 0.4)
    v_theta = np.full(5, 2_000.0)

    er = radial_electric_field(
        ne,
        ti,
        omega,
        b_theta,
        B0=5.0,
        R0=6.0,
        rho=rho,
        a=2.0,
        v_theta=v_theta,
    )

    np.testing.assert_allclose(er, -10_000.0)


def test_radial_electric_field_rejects_invalid_poloidal_flow_profile() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    profile = np.ones(5)

    with pytest.raises(ValueError, match="v_theta"):
        radial_electric_field(
            profile,
            profile,
            profile,
            profile,
            B0=5.0,
            R0=6.0,
            rho=rho,
            a=2.0,
            v_theta=np.ones(4),
        )


def test_diagnostics():
    omega = np.ones(50) * 1e4
    Ti = np.ones(50) * 10.0

    diag = RotationDiagnostics()
    mach = diag.mach_number(omega, Ti, R0=6.2)

    assert np.all(mach > 0.0)
    assert mach[0] < 1.0  # Usually subsonic

    stab = diag.rwm_stabilization_criterion(omega, tau_wall=0.01)
    assert stab  # 10000 * 0.01 = 100 > 0.01


def test_rwm_stabilization_requires_wall_time_order_unity_rotation():
    omega = np.ones(50) * 50.0

    assert not RotationDiagnostics.rwm_stabilization_criterion(omega, tau_wall=0.01)


def test_rwm_stabilization_rejects_invalid_domain():
    with pytest.raises(ValueError, match="omega_phi"):
        RotationDiagnostics.rwm_stabilization_criterion(np.array([np.nan]), tau_wall=0.01)
    with pytest.raises(ValueError, match="tau_wall"):
        RotationDiagnostics.rwm_stabilization_criterion(np.ones(10), tau_wall=0.0)


def test_momentum_solver_rejects_invalid_grid_and_step_domains():
    with pytest.raises(ValueError, match="rho"):
        MomentumTransportSolver(np.array([0.0, 1.0]), R0=6.2, a=2.0, B0=5.3)

    rho = np.linspace(0.0, 1.0, 50)
    solver = MomentumTransportSolver(rho, R0=6.2, a=2.0, B0=5.3)
    good = np.ones(50)

    with pytest.raises(ValueError, match="dt"):
        solver.step(0.0, good, good, good, good, good)
    with pytest.raises(ValueError, match="chi_i"):
        solver.step(0.1, -good, good, good, good, good)


def test_intrinsic_rotation_torque_tracks_ion_temperature_gradient_direction() -> None:
    """Residual-stress torque vanishes for flat Ti and reverses with Ti gradient."""
    from scpn_fusion.core.momentum_transport import intrinsic_rotation_torque

    grad_ne = np.zeros(5)
    flat = intrinsic_rotation_torque(np.zeros(5), grad_ne, R0=6.2, a=2.0)
    inward_hotter = intrinsic_rotation_torque(np.linspace(-4.0, -1.0, 5), grad_ne, R0=6.2, a=2.0)
    outward_hotter = intrinsic_rotation_torque(np.linspace(1.0, 4.0, 5), grad_ne, R0=6.2, a=2.0)

    np.testing.assert_allclose(flat, 0.0)
    assert np.all(inward_hotter > 0.0)
    assert np.all(outward_hotter < 0.0)
