# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Flight Sim Controller Tests
"""Tests for flight_sim_controllers module (split from h_infinity_controller)."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.flight_sim_controllers import (
    LQRController,
    get_flight_sim_controller,
    get_flight_sim_controller_v2,
    get_flight_sim_lqr_controller,
)


class TestFlightSimController:
    def test_v1_synthesizes(self):
        ctrl = get_flight_sim_controller(response_gain=0.05, actuator_tau=0.06)
        assert ctrl.is_stable

    def test_v2_synthesizes(self):
        ctrl = get_flight_sim_controller_v2(position_sensitivity=0.567, sample_dt=0.05)
        assert ctrl.is_stable

    def test_v2_step_converges(self):
        ctrl = get_flight_sim_controller_v2(
            position_sensitivity=0.567,
            sample_dt=0.05,
            observer_q_scale=100.0,
        )
        dt = 0.05
        errors = []
        error = 0.01
        for _ in range(200):
            u = ctrl.step(error, dt)
            error = error * 0.99 - 0.01 * u
            errors.append(abs(error))
        assert errors[-1] < errors[0]

    def test_v1_rejects_bad_params(self):
        with pytest.raises(ValueError):
            get_flight_sim_controller(response_gain=-1.0)
        with pytest.raises(ValueError):
            get_flight_sim_controller(actuator_tau=0.0)

    def test_v2_rejects_bad_params(self):
        with pytest.raises(ValueError):
            get_flight_sim_controller_v2(position_sensitivity=-1.0)
        with pytest.raises(ValueError):
            get_flight_sim_controller_v2(sample_dt=0.0)


class TestLQRController:
    def test_lqr_synthesizes(self):
        ctrl = get_flight_sim_lqr_controller(position_sensitivity=0.567, sample_dt=0.05)
        assert isinstance(ctrl, LQRController)

    def test_lqr_step_returns_float(self):
        ctrl = get_flight_sim_lqr_controller()
        u = ctrl.step(0.01, 0.05)
        assert np.isfinite(u)

    def test_lqr_reset(self):
        ctrl = get_flight_sim_lqr_controller()
        ctrl.step(0.1, 0.05)
        ctrl.reset()
        assert np.allclose(ctrl.state, 0.0)

    def test_lqr_rejects_bad_params(self):
        with pytest.raises(ValueError):
            get_flight_sim_lqr_controller(position_sensitivity=0.0)
        with pytest.raises(ValueError):
            get_flight_sim_lqr_controller(actuator_tau=-1.0)
