# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fault-Tolerant Control Tests
from __future__ import annotations

import numpy as np
import pytest

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


def test_sensor_fault_dropout_zeroes_faulted_measurement_weight():
    J = np.eye(3)
    ctrl = ReconfigurableController(None, J, 3, 3)
    ctrl.handle_sensor_fault(1, FaultType.SENSOR_DROPOUT)

    assert np.isclose(ctrl.W[1, 1], 0.0)
    assert 1 in ctrl.faulted_sensors


def test_sensor_fault_noise_and_drift_apply_distinct_weights():
    J = np.eye(4)
    ctrl = ReconfigurableController(None, J, 4, 4)

    ctrl.handle_sensor_fault(2, FaultType.SENSOR_NOISE_INCREASE)
    ctrl.handle_sensor_fault(3, FaultType.SENSOR_DRIFT)

    assert np.isclose(ctrl.W[2, 2], 0.2)
    assert np.isclose(ctrl.W[3, 3], 0.5)


def test_sensor_fault_out_of_range_rejected():
    J = np.eye(2)
    ctrl = ReconfigurableController(None, J, 2, 2)

    with pytest.raises(IndexError, match="out of range"):
        ctrl.handle_sensor_fault(5, FaultType.SENSOR_DROPOUT)


def _ctrl() -> ReconfigurableController:
    return ReconfigurableController(None, np.eye(3), n_coils=3, n_sensors=3)


def test_handle_actuator_fault_is_idempotent_per_coil() -> None:
    ctrl = _ctrl()
    ctrl.handle_actuator_fault(1, FaultType.OPEN_CIRCUIT_ACTUATOR)
    ctrl.handle_actuator_fault(1, FaultType.OPEN_CIRCUIT_ACTUATOR)  # early return
    assert ctrl.faulted_coils == {1}


def test_handle_actuator_fault_records_stuck_value() -> None:
    ctrl = _ctrl()
    ctrl.handle_actuator_fault(0, FaultType.STUCK_ACTUATOR, stuck_val=5.0)
    assert ctrl.stuck_values[0] == 5.0


def test_compute_gain_falls_back_to_zero_on_singular_inverse(monkeypatch) -> None:
    ctrl = _ctrl()

    def _raise(_: object) -> object:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "inv", _raise)
    ctrl.handle_actuator_fault(0, FaultType.OPEN_CIRCUIT_ACTUATOR)
    assert np.allclose(ctrl.K, 0.0)


def test_handle_sensor_fault_rejects_out_of_range_index() -> None:
    ctrl = _ctrl()
    with pytest.raises(IndexError, match="out of range"):
        ctrl.handle_sensor_fault(3, FaultType.SENSOR_DROPOUT)


def test_handle_sensor_fault_is_idempotent_per_sensor() -> None:
    ctrl = _ctrl()
    ctrl.handle_sensor_fault(0, FaultType.SENSOR_DROPOUT)
    ctrl.handle_sensor_fault(0, FaultType.SENSOR_DROPOUT)  # early return
    assert ctrl.faulted_sensors == {0}


def test_handle_sensor_fault_guards_unmappable_weight_row() -> None:
    ctrl = _ctrl()
    ctrl.W = np.eye(2)  # weight space smaller than the sensor index space
    with pytest.raises(IndexError, match="cannot be mapped"):
        ctrl.handle_sensor_fault(2, FaultType.SENSOR_DROPOUT)


def test_handle_sensor_fault_downweights_unknown_sensor_mode() -> None:
    ctrl = _ctrl()
    ctrl.handle_sensor_fault(1, FaultType.OPEN_CIRCUIT_ACTUATOR)  # not a sensor class
    assert ctrl.W[1, 1] == 0.5


def test_fault_injector_zeroes_signal_on_dropout() -> None:
    injector = FaultInjector(
        fault_time=1.0, component_index=2, fault_type=FaultType.SENSOR_DROPOUT, severity=1.0
    )
    corrupted = injector.inject(2.0, np.ones(4))
    assert corrupted[2] == 0.0


def test_step_compensates_for_stuck_actuator_offset() -> None:
    ctrl = _ctrl()
    ctrl.handle_actuator_fault(0, FaultType.STUCK_ACTUATOR, stuck_val=2.0)
    delta_u = ctrl.step(np.array([0.1, 0.2, 0.3]), dt=0.01)
    assert delta_u.shape == (3,)
    assert np.all(np.isfinite(delta_u))
