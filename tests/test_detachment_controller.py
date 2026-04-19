# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Divertor Detachment Control Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.control.detachment_controller import (
    DetachmentBifurcation,
    DetachmentController,
    DetachmentState,
    MultiImpuritySeeding,
    RadiationFrontModel,
)
from scpn_fusion.core.sol_model import TwoPointSOL


def test_radiation_front_model():
    model = RadiationFrontModel("N2", R0=6.2, a=2.0, q95=3.0)

    assert model.radiation_temperature("N2") == 10.0
    assert model.radiation_temperature("Ne") == 30.0

    pos_low = model.front_position(P_SOL_MW=100.0, n_u_19=4.0, seeding_rate=1.0)
    pos_high = model.front_position(P_SOL_MW=100.0, n_u_19=4.0, seeding_rate=10.0)

    assert pos_high > pos_low
    assert 0.0 <= pos_low <= 1.0


def test_dod_calculation():
    model = RadiationFrontModel("N2", 6.2, 2.0, 3.0)

    # Attached DOD = 1.0
    assert model.degree_of_detachment(30.0, 1.0, 1.0) == 1.0

    # Detached DOD > 1.0
    assert model.degree_of_detachment(2.5, 1.0, 1.0) > 1.0


def test_detachment_controller_states():
    ctrl = DetachmentController()

    # Needs seeding
    cmd1 = ctrl.step(
        T_t_measured=20.0, n_t_measured=10.0, P_rad_measured=10.0, rho_front=0.1, dt=0.1
    )
    assert cmd1 > 0.0
    assert ctrl.state == DetachmentState.PARTIALLY_DETACHED

    # Deeply detached -> reduce seeding
    # Reset controller integral correctly so it only evaluates the new error
    ctrl.integral_e = 0.0
    cmd2 = ctrl.step(
        T_t_measured=1.0, n_t_measured=10.0, P_rad_measured=50.0, rho_front=0.1, dt=0.1
    )
    # cmd2 = Kp * (-2) + Ki * (-0.2) = -102 -> max(0, -102) = 0
    assert cmd2 < cmd1
    assert ctrl.state == DetachmentState.FULLY_DETACHED

    # MARFE risk -> hard drop
    cmd3 = ctrl.step(
        T_t_measured=1.0, n_t_measured=10.0, P_rad_measured=50.0, rho_front=0.9, dt=0.1
    )
    assert cmd3 <= cmd2
    assert ctrl.state == DetachmentState.XPOINT_MARFE


def test_detachment_bifurcation():
    sol = TwoPointSOL(R0=6.2, a=2.0, q95=3.0, B_pol=0.56)
    bif = DetachmentBifurcation(sol, "N2")

    sr_scan = np.linspace(0.0, 10.0, 50)
    pts = bif.scan_seeding(sr_scan, P_SOL_MW=100.0, n_u_19=4.0)

    assert len(pts) == 50
    assert pts[0].state == DetachmentState.ATTACHED
    assert pts[-1].state in [DetachmentState.FULLY_DETACHED, DetachmentState.XPOINT_MARFE]

    # Verify rollover exists by supplying low power so it detaches easily
    sr_rollover = bif.find_rollover_point(P_SOL_MW=10.0, n_u_19=4.0)
    assert sr_rollover >= 0.0


def test_multi_impurity_seeding():
    c_N2 = DetachmentController("N2")
    c_Ne = DetachmentController("Ne")

    multi = MultiImpuritySeeding(["N2", "Ne"], {"N2": c_N2, "Ne": c_Ne})

    diag = {"T_target_eV": 20.0, "rho_front": 0.1}
    rates = multi.step(diag, dt=0.1)

    assert "N2" in rates
    assert "Ne" in rates
    assert rates["N2"] > 0.0
