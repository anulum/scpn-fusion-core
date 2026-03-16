# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fault-Tolerant Reconfigurable Control
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np


class FaultType(Enum):
    """Actuator and sensor fault modes for FDI classification."""

    STUCK_ACTUATOR = auto()
    OPEN_CIRCUIT_ACTUATOR = auto()
    SENSOR_DROPOUT = auto()
    SENSOR_DRIFT = auto()
    SENSOR_NOISE_INCREASE = auto()


@dataclass
class FaultReport:
    """Detected fault: which component, fault mode, confidence, and detection time."""

    component_index: int
    is_sensor: bool
    fault_type: FaultType
    confidence: float
    time_detected: float


class FDIMonitor:
    """Fault Detection and Isolation based on innovation monitoring."""

    def __init__(
        self, n_sensors: int, n_actuators: int, threshold_sigma: float = 3.0, n_alert: int = 5
    ):
        self.n_sensors = n_sensors
        self.n_actuators = n_actuators
        self.threshold_sigma = threshold_sigma
        self.n_alert = n_alert

        self.innovation_history = np.zeros((n_alert, n_sensors))
        self.innovation_idx = 0

        self.S_diag = np.ones(n_sensors)

        self.detected_faults: list[FaultReport] = []
        self.faulted_sensors: set[int] = set()

    def update(
        self, y_measured: np.ndarray, y_predicted: np.ndarray, t: float
    ) -> list[FaultReport]:
        """Compare measurements to predictions; flag sensors exceeding threshold_sigma for n_alert consecutive steps."""
        nu = y_measured - y_predicted

        self.innovation_history[self.innovation_idx] = nu
        self.innovation_idx = (self.innovation_idx + 1) % self.n_alert

        new_faults = []

        for i in range(self.n_sensors):
            if i in self.faulted_sensors:
                continue

            hist = self.innovation_history[:, i]
            sigma = np.sqrt(self.S_diag[i])

            if np.all(np.abs(hist) > self.threshold_sigma * sigma):
                if np.isnan(y_measured[i]) or abs(y_measured[i]) < 1e-6:
                    ftype = FaultType.SENSOR_DROPOUT
                else:
                    ftype = FaultType.SENSOR_DRIFT

                report = FaultReport(
                    component_index=i,
                    is_sensor=True,
                    fault_type=ftype,
                    confidence=1.0,
                    time_detected=t,
                )
                new_faults.append(report)
                self.detected_faults.append(report)
                self.faulted_sensors.add(i)

        return new_faults


class ReconfigurableController:
    """Adjusts control allocation based on detected faults."""

    def __init__(self, base_controller: Any, jacobian: np.ndarray, n_coils: int, n_sensors: int):
        self.base_controller = base_controller
        self.nominal_jacobian = jacobian.copy()
        self.current_jacobian = jacobian.copy()
        self.n_coils = n_coils
        self.n_sensors = n_sensors

        self.faulted_coils: set[int] = set()
        self.stuck_values: dict[int, float] = {}

        self.W = np.eye(jacobian.shape[0])
        self.lambda_reg = 1e-6

        self.K = self._compute_gain()

    def _compute_gain(self) -> np.ndarray:
        """Tikhonov pseudoinverse with faulted-coil rows zeroed."""
        J = self.current_jacobian
        J_T_W = J.T @ self.W
        H = J_T_W @ J + self.lambda_reg * np.eye(self.n_coils)

        try:
            H_inv = np.linalg.inv(H)
            K = H_inv @ J_T_W
        except np.linalg.LinAlgError:
            K = np.zeros_like(J.T)

        for i in self.faulted_coils:
            K[i, :] = 0.0

        return np.asarray(K)

    def handle_actuator_fault(
        self, coil_index: int, fault_type: FaultType, stuck_val: float = 0.0
    ) -> None:
        """Zero out the faulted coil column in J and recompute gain."""
        if coil_index in self.faulted_coils:
            return

        self.faulted_coils.add(coil_index)

        if fault_type == FaultType.STUCK_ACTUATOR:
            self.stuck_values[coil_index] = stuck_val

        self.current_jacobian[:, coil_index] = 0.0
        self.K = self._compute_gain()

    def handle_sensor_fault(self, sensor_index: int, fault_type: FaultType) -> None:
        """Placeholder for sensor fault accommodation (e.g. observer reconfiguration)."""
        pass

    def step(self, error: np.ndarray, dt: float) -> np.ndarray:
        """Compute coil current corrections, compensating stuck-actuator offsets."""
        adjusted_error = error.copy()
        for c_idx, val in self.stuck_values.items():
            adjusted_error -= self.nominal_jacobian[:, c_idx] * val

        delta_u = self.K @ adjusted_error

        for c_idx in self.faulted_coils:
            delta_u[c_idx] = 0.0

        return np.asarray(delta_u)

    def controllability_check(self) -> bool:
        """True if enough healthy coils remain for minimum-rank controllability."""
        if len(self.faulted_coils) > self.n_coils // 2:
            return False

        J = self.current_jacobian
        rank = np.linalg.matrix_rank(J)

        min_required_rank = 2
        return bool(rank >= min_required_rank)

    def graceful_shutdown(self) -> np.ndarray:
        """Return zero-current command vector for safe ramp-down."""
        return np.zeros(self.n_coils)


class FaultInjector:
    """Injects sensor faults (dropout or drift) at a specified time for testing."""

    def __init__(
        self, fault_time: float, component_index: int, fault_type: FaultType, severity: float = 1.0
    ):
        self.fault_time = fault_time
        self.component_index = component_index
        self.fault_type = fault_type
        self.severity = severity

    def inject(self, t: float, signals: np.ndarray) -> np.ndarray:
        if t < self.fault_time:
            return signals

        corrupted = signals.copy()

        if self.fault_type == FaultType.SENSOR_DROPOUT:
            corrupted[self.component_index] = 0.0
        elif self.fault_type == FaultType.SENSOR_DRIFT:
            drift = self.severity * (t - self.fault_time)
            corrupted[self.component_index] += drift

        return corrupted
