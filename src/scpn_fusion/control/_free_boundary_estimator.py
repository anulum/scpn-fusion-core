# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core - Free-Boundary State Estimation
"""Bias-aware state estimator for supervisory free-boundary control."""

from __future__ import annotations

import numpy as np

from scpn_fusion.control.neural_surrogate_mpc import NeuralSurrogate
from scpn_fusion.control._free_boundary_supervisory_types import (
    FloatArray,
    FreeBoundaryEstimate,
    _require_positive_finite,
)


class FreeBoundaryStateEstimator:
    """Observer with persistent-bias tracking for free-boundary state."""

    def __init__(
        self,
        surrogate: NeuralSurrogate,
        *,
        measurement_gain: float = 0.55,
        bias_gain: float = 0.08,
        bias_decay: float = 0.985,
        actuator_bias_gain: float = 0.06,
        actuator_bias_decay: float = 0.992,
        max_actuator_bias: float = 0.35,
    ) -> None:
        self.surrogate = surrogate
        self.measurement_gain = _require_positive_finite("measurement_gain", measurement_gain)
        self.bias_gain = _require_positive_finite("bias_gain", bias_gain)
        self.bias_decay = float(bias_decay)
        if not np.isfinite(self.bias_decay) or not (0.0 < self.bias_decay <= 1.0):
            raise ValueError("bias_decay must be finite and in (0, 1].")
        self.actuator_bias_gain = _require_positive_finite("actuator_bias_gain", actuator_bias_gain)
        self.actuator_bias_decay = float(actuator_bias_decay)
        if not np.isfinite(self.actuator_bias_decay) or not (0.0 < self.actuator_bias_decay <= 1.0):
            raise ValueError("actuator_bias_decay must be finite and in (0, 1].")
        self.max_actuator_bias = _require_positive_finite("max_actuator_bias", max_actuator_bias)
        self.allocation = np.linalg.pinv(
            np.asarray(self.surrogate.B, dtype=np.float64),
            rcond=1e-6,
        )
        self.state_hat = np.zeros(4, dtype=np.float64)
        self.bias_hat = np.zeros(4, dtype=np.float64)
        self.actuator_bias_hat = np.zeros(self.surrogate.B.shape[1], dtype=np.float64)
        self.innovation = np.zeros(4, dtype=np.float64)
        self.corrected_state = np.zeros(4, dtype=np.float64)
        self._initialized = False

    def reset(self, initial_state: FloatArray | list[float] | tuple[float, ...]) -> None:
        """Initialise the observer from a finite four-component geometry state."""
        state = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        if state.size != 4 or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must be a finite 4-vector.")
        self.state_hat = state.copy()
        self.bias_hat.fill(0.0)
        self.actuator_bias_hat.fill(0.0)
        self.innovation.fill(0.0)
        self.corrected_state = state.copy()
        self._initialized = True

    def step(
        self,
        measurement: FloatArray,
        applied_action: FloatArray,
        measured_actuator_action: FloatArray | None = None,
    ) -> FreeBoundaryEstimate:
        """Assimilate one measurement and return corrected geometry plus bias estimates."""
        meas = np.asarray(measurement, dtype=np.float64).reshape(-1)
        action = np.asarray(applied_action, dtype=np.float64).reshape(-1)
        measured_action = (
            np.asarray(measured_actuator_action, dtype=np.float64).reshape(-1)
            if measured_actuator_action is not None
            else None
        )
        if meas.size != 4 or not np.all(np.isfinite(meas)):
            raise ValueError("measurement must be a finite 4-vector.")
        if action.size != self.surrogate.B.shape[1] or not np.all(np.isfinite(action)):
            raise ValueError("applied_action must match the surrogate control width.")
        if measured_action is not None and (
            measured_action.size != action.size or not np.all(np.isfinite(measured_action))
        ):
            raise ValueError("measured_actuator_action must match the surrogate control width.")

        if not self._initialized:
            self.reset(meas)

        effective_action = action + self.actuator_bias_hat
        predicted = self.surrogate.predict(self.corrected_state, effective_action)
        innovation = meas - predicted
        self.state_hat = predicted + self.measurement_gain * innovation
        self.bias_hat = self.bias_decay * self.bias_hat + self.bias_gain * innovation
        if measured_action is not None:
            actuator_residual = measured_action - action
        else:
            actuator_residual = self.allocation @ (innovation - self.bias_hat)
        actuator_prior = self.actuator_bias_decay * self.actuator_bias_hat
        self.actuator_bias_hat = np.clip(
            actuator_prior + self.actuator_bias_gain * (actuator_residual - actuator_prior),
            -self.max_actuator_bias,
            self.max_actuator_bias,
        )
        self.innovation = innovation
        self.corrected_state = self.state_hat + self.bias_hat
        return FreeBoundaryEstimate(
            state_hat=self.state_hat.copy(),
            bias_hat=self.bias_hat.copy(),
            actuator_bias_hat=self.actuator_bias_hat.copy(),
            innovation=self.innovation.copy(),
            corrected_state=self.corrected_state.copy(),
            uncertainty_norm=float(
                np.linalg.norm(innovation) + 0.25 * np.linalg.norm(self.actuator_bias_hat)
            ),
        )
