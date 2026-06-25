# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real-Time EFIT Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.control.realtime_efit import (
    MagneticDiagnostics,
    RealtimeEFIT,
)


def create_mock_diagnostics() -> MagneticDiagnostics:
    flux_loops = [(2.0, 1.0), (3.0, 1.5), (4.0, 1.0)]
    b_probes = [(2.0, 1.0, "R"), (2.0, 1.0, "Z"), (4.0, 1.0, "R")]
    return MagneticDiagnostics(flux_loops, b_probes, rogowski_radius=3.0)


def test_simulate_measurements():
    diag = create_mock_diagnostics()
    R = np.linspace(2.0, 10.0, 30)
    Z = np.linspace(-6.0, 6.0, 30)

    efit = RealtimeEFIT(diag, R, Z)

    R2, Z2 = np.meshgrid(R, Z, indexing="ij")
    psi = (R2 - 6.0) ** 2 + Z2**2

    coils = np.zeros(5)
    meas = efit.response.simulate_measurements(psi, coils)

    assert len(meas["flux_loops"]) == len(diag.flux_loops)
    assert len(meas["b_probes"]) == len(diag.b_probes)
    assert "Ip" in meas


def test_reconstruction_solovev():
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 33)
    Z = np.linspace(-3.0, 3.0, 33)

    efit = RealtimeEFIT(diag, R, Z)

    meas = {
        "flux_loops": np.zeros(3),
        "b_probes": np.zeros(3),
        "Ip": 15.0e6,
        "coil_currents": np.zeros(5),
    }

    res = efit.reconstruct(meas)

    assert np.isclose(res.shape.R0, 6.2)
    assert np.isclose(res.shape.a, 2.0)
    assert res.shape.Ip_reconstructed == 15.0e6
    # The real-time property is the bounded, small iteration count (deterministic).
    # wall_time_ms is wall-clock and therefore load-sensitive — it can spike by
    # orders of magnitude under CI/host contention even though a single
    # reconstruction is well under a millisecond in isolation — so assert only a
    # generous ceiling that still catches a catastrophic algorithmic regression
    # (e.g. an accidental full equilibrium solve in the inner loop).
    assert res.n_iterations > 0
    assert res.n_iterations <= 64
    assert res.wall_time_ms < 5000.0


def test_lcfs_boundary_points_follow_flux_geometry():
    diag = create_mock_diagnostics()
    R = np.linspace(3.5, 8.5, 65)
    Z = np.linspace(-3.5, 3.5, 65)
    efit = RealtimeEFIT(diag, R, Z)

    R2, Z2 = np.meshgrid(R, Z, indexing="ij")
    psi = 1.0 - ((R2 - 6.1) / 1.4) ** 2 - (Z2 / 2.1) ** 2
    psi = np.clip(psi, 0.0, None)

    lcfs = efit.find_lcfs(psi)

    assert lcfs.shape[1] == 2
    assert len(lcfs) > 20
    assert np.ptp(lcfs[:, 0]) > 2.0
    assert np.ptp(lcfs[:, 1]) > 3.0


def test_shape_params_are_estimated_from_lcfs_not_constants():
    diag = create_mock_diagnostics()
    R = np.linspace(3.0, 9.0, 81)
    Z = np.linspace(-4.0, 4.0, 81)
    efit = RealtimeEFIT(diag, R, Z)

    R0 = 6.0
    minor_radius = 1.5
    elongation = 1.35
    R2, Z2 = np.meshgrid(R, Z, indexing="ij")
    vertical = Z2 / (elongation * minor_radius)
    shifted_R = R2 - (R0 + 0.18 * minor_radius * vertical)
    psi = 1.0 - (shifted_R / minor_radius) ** 2 - vertical**2
    psi = np.clip(psi, 0.0, None)

    shape = efit.compute_shape_params(psi)

    assert np.isclose(shape.R0, R0, atol=0.15)
    assert np.isclose(shape.a, minor_radius, atol=0.2)
    assert np.isclose(shape.kappa, elongation, atol=0.15)
    assert shape.kappa != 1.7
    assert abs(shape.delta_upper) > 0.05
    assert abs(shape.delta_lower) > 0.05


def test_reconstruct_rejects_mismatched_measurement_lengths():
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 33)
    Z = np.linspace(-3.0, 3.0, 33)
    efit = RealtimeEFIT(diag, R, Z)

    bad_meas = {
        "flux_loops": np.zeros(2),
        "b_probes": np.zeros(3),
        "Ip": 15.0e6,
        "coil_currents": np.zeros(5),
    }

    with np.testing.assert_raises(ValueError):
        efit.reconstruct(bad_meas)


def test_xpoint_detection():
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 33)
    Z = np.linspace(-3.0, 3.0, 33)

    efit = RealtimeEFIT(diag, R, Z)

    psi = np.zeros((33, 33))
    xp = efit.find_xpoint(psi)

    assert xp is not None
    assert xp[0] > 0.0
