# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for the bias-aware free-boundary state estimator.

Covers the constructor decay-range guards, the reset and step input-shape /
finiteness validation, the implicit first-step initialisation, and a nominal
assimilation step that returns finite geometry and bias estimates.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control._free_boundary_estimator import FreeBoundaryStateEstimator
from scpn_fusion.control.neural_surrogate_mpc import NeuralSurrogate


def _surrogate() -> NeuralSurrogate:
    return NeuralSurrogate(n_coils=4, n_state=4, verbose=False)


def _estimator() -> FreeBoundaryStateEstimator:
    return FreeBoundaryStateEstimator(_surrogate())


class TestConstructorGuards:
    """Decay coefficients must be finite and within ``(0, 1]``."""

    @pytest.mark.parametrize("bias_decay", [1.5, 0.0, -0.1, float("nan")])
    def test_invalid_bias_decay_raises(self, bias_decay: float) -> None:
        """A bias decay outside ``(0, 1]`` is rejected."""
        with pytest.raises(ValueError, match="bias_decay must be finite"):
            FreeBoundaryStateEstimator(_surrogate(), bias_decay=bias_decay)

    @pytest.mark.parametrize("actuator_bias_decay", [1.5, 0.0, float("inf")])
    def test_invalid_actuator_bias_decay_raises(self, actuator_bias_decay: float) -> None:
        """An actuator bias decay outside ``(0, 1]`` is rejected."""
        with pytest.raises(ValueError, match="actuator_bias_decay must be finite"):
            FreeBoundaryStateEstimator(_surrogate(), actuator_bias_decay=actuator_bias_decay)


class TestResetValidation:
    """The observer reset requires a finite four-component state."""

    @pytest.mark.parametrize("state", [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0, float("nan")]])
    def test_invalid_initial_state_raises(self, state: list[float]) -> None:
        """A non-4-vector or non-finite initial state is rejected."""
        with pytest.raises(ValueError, match="initial_state must be a finite 4-vector"):
            _estimator().reset(state)

    def test_reset_marks_initialized(self) -> None:
        """A valid reset seeds the corrected state and flags initialisation."""
        est = _estimator()
        est.reset([0.1, 0.2, 0.3, 0.4])
        assert est._initialized is True
        assert np.allclose(est.corrected_state, [0.1, 0.2, 0.3, 0.4])


class TestStepValidation:
    """Assimilation validates measurement and action shapes and finiteness."""

    def test_non_finite_measurement_raises(self) -> None:
        """A measurement that is not a finite 4-vector is rejected."""
        with pytest.raises(ValueError, match="measurement must be a finite 4-vector"):
            _estimator().step(np.array([1.0, 2.0, 3.0]), np.zeros(4))

    def test_wrong_width_action_raises(self) -> None:
        """An applied action not matching the surrogate control width is rejected."""
        with pytest.raises(ValueError, match="applied_action must match"):
            _estimator().step(np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(3))

    def test_wrong_width_measured_action_raises(self) -> None:
        """A measured actuator action of the wrong width is rejected."""
        with pytest.raises(ValueError, match="measured_actuator_action must match"):
            _estimator().step(np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4), np.zeros(3))


class TestAssimilation:
    """Nominal stepping behaviour and implicit initialisation."""

    def test_first_step_auto_initialises(self) -> None:
        """Stepping before an explicit reset initialises from the measurement."""
        est = _estimator()
        assert est._initialized is False
        estimate = est.step(np.array([0.1, 0.2, 0.3, 0.4]), np.zeros(4))
        assert est._initialized is True
        assert np.all(np.isfinite(estimate.corrected_state))
        assert np.isfinite(estimate.uncertainty_norm)

    def test_step_with_measured_actuator_action_is_finite(self) -> None:
        """Supplying a measured actuator action yields finite bias estimates."""
        est = _estimator()
        est.reset([0.0, 0.0, 0.0, 0.0])
        estimate = est.step(
            np.array([0.05, -0.02, 0.01, 0.03]),
            np.array([0.1, 0.0, -0.1, 0.0]),
            measured_actuator_action=np.array([0.12, 0.0, -0.08, 0.0]),
        )
        assert np.all(np.isfinite(estimate.actuator_bias_hat))
        assert np.all(np.isfinite(estimate.state_hat))
        assert np.all(np.abs(estimate.actuator_bias_hat) <= est.max_actuator_bias + 1e-9)
