# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fault-Tolerant Control Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.control.fault_tolerant_control import (
    FDIMonitor,
    FaultInjector,
    FaultType,
    ReconfigurableController,
)


def test_fdi_no_fault():
    fdi = FDIMonitor(n_sensors=3, n_actuators=2, threshold_sigma=3.0, n_alert=5)

    y_meas = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    for t in range(10):
        faults = fdi.update(y_meas, y_pred, float(t))
        assert len(faults) == 0


def test_fdi_sensor_dropout():
    fdi = FDIMonitor(n_sensors=3, n_actuators=2, threshold_sigma=3.0, n_alert=3)

    y_pred = np.array([1.0, 2.0, 3.0])

    fdi.update(np.array([1.0, 2.0, 3.0]), y_pred, 0.0)

    y_meas = np.array([1.0, 0.0, 3.0])
    fdi.S_diag = np.array([1.0, 0.1, 1.0])

    all_faults = []
    for t in range(1, 5):
        faults = fdi.update(y_meas, y_pred, float(t))
        all_faults.extend(faults)

    assert len(all_faults) == 1
    assert all_faults[0].component_index == 1
    assert all_faults[0].fault_type == FaultType.SENSOR_DROPOUT


def test_fdi_sensor_drift():
    fdi = FDIMonitor(n_sensors=3, n_actuators=2, threshold_sigma=3.0, n_alert=3)

    y_pred = np.array([1.0, 2.0, 3.0])
    y_meas = np.array([1.0, 2.0, 3.0])

    injector = FaultInjector(
        fault_time=5.0, component_index=2, fault_type=FaultType.SENSOR_DRIFT, severity=2.0
    )

    for t in range(10):
        corrupted = injector.inject(float(t), y_meas)
        faults = fdi.update(corrupted, y_pred, float(t))
        if len(faults) > 0:
            assert faults[0].fault_type == FaultType.SENSOR_DRIFT
            assert faults[0].component_index == 2
            break

    assert len(fdi.detected_faults) > 0


def test_reconfigurable_controller_actuator_loss():
    J = np.eye(3)
    ctrl = ReconfigurableController(None, J, 3, 3)

    error = np.array([1.0, 1.0, 1.0])
    u_nom = ctrl.step(error, 0.1)

    assert np.allclose(u_nom, [1.0, 1.0, 1.0], atol=1e-3)

    ctrl.handle_actuator_fault(1, FaultType.OPEN_CIRCUIT_ACTUATOR)

    u_fault = ctrl.step(error, 0.1)

    assert np.isclose(u_fault[1], 0.0)
    assert np.isclose(u_fault[0], 1.0, atol=1e-3)
    assert np.isclose(u_fault[2], 1.0, atol=1e-3)


def test_controllability_check():
    J = np.eye(5)
    ctrl = ReconfigurableController(None, J, 5, 5)

    assert ctrl.controllability_check()

    ctrl.handle_actuator_fault(0, FaultType.OPEN_CIRCUIT_ACTUATOR)
    ctrl.handle_actuator_fault(1, FaultType.OPEN_CIRCUIT_ACTUATOR)
    ctrl.handle_actuator_fault(2, FaultType.OPEN_CIRCUIT_ACTUATOR)

    assert not ctrl.controllability_check()

    sd = ctrl.graceful_shutdown()
    assert np.all(sd == 0.0)
