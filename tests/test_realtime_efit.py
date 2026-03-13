# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Real-Time EFIT Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
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

    meas = {"flux_loops": np.zeros(3), "b_probes": np.zeros(3), "Ip": 15.0e6, "coil_currents": np.zeros(5)}

    res = efit.reconstruct(meas)

    assert np.isclose(res.shape.R0, 6.2)
    assert np.isclose(res.shape.a, 2.0)
    assert res.shape.Ip_reconstructed == 15.0e6
    assert res.wall_time_ms < 100.0
    assert res.n_iterations > 0


def test_xpoint_detection():
    diag = create_mock_diagnostics()
    R = np.linspace(4.2, 8.2, 33)
    Z = np.linspace(-3.0, 3.0, 33)

    efit = RealtimeEFIT(diag, R, Z)

    psi = np.zeros((33, 33))
    xp = efit.find_xpoint(psi)

    assert xp is not None
    assert xp[0] > 0.0
