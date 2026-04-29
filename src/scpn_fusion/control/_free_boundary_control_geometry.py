# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core - Free-Boundary Control Geometry
from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.control.fusion_sota_mpc import NeuralSurrogate
from scpn_fusion.control._free_boundary_supervisory_types import (
    FloatArray,
    FreeBoundaryEstimate,
    FreeBoundaryTarget,
    _normalize_vector,
    _require_positive_finite,
)


class FreeBoundarySupervisoryController:
    """Target-tracking controller for axis and X-point free-boundary geometry."""

    def __init__(
        self,
        surrogate: NeuralSurrogate,
        target: FreeBoundaryTarget,
        *,
        state_gains: np.ndarray | tuple[float, ...] = (1.8, 2.1, 1.4, 1.7),
        bias_rejection_gains: np.ndarray | tuple[float, ...] = (0.7, 0.8, 0.55, 0.65),
        innovation_damping: float = 0.18,
        actuator_compensation_gain: float = 0.85,
    ) -> None:
        self.surrogate = surrogate
        self.target = target
        self.state_gains = _normalize_vector(state_gains, length=4, name="state_gains")
        self.bias_rejection_gains = _normalize_vector(
            bias_rejection_gains, length=4, name="bias_rejection_gains"
        )
        self.innovation_damping = _require_positive_finite("innovation_damping", innovation_damping)
        self.actuator_compensation_gain = _require_positive_finite(
            "actuator_compensation_gain",
            actuator_compensation_gain,
        )
        self.allocation = np.linalg.pinv(
            np.asarray(self.surrogate.B, dtype=np.float64),
            rcond=1e-6,
        )

    def propose_action(self, estimate: FreeBoundaryEstimate) -> FloatArray:
        error = self.target.as_vector() - estimate.corrected_state
        desired_delta = (
            self.state_gains * error
            - self.bias_rejection_gains * estimate.bias_hat
            - self.innovation_damping * estimate.innovation
        )
        action = (
            self.allocation @ desired_delta
            - self.actuator_compensation_gain * estimate.actuator_bias_hat
        )
        return np.asarray(action, dtype=np.float64).reshape(-1)


def extract_free_boundary_state(kernel: Any) -> FloatArray:
    psi = np.asarray(kernel.Psi, dtype=np.float64)
    idx_max = int(np.argmax(psi))
    iz, ir = np.unravel_index(idx_max, psi.shape)
    r_axis = float(kernel.R[ir])
    z_axis = float(kernel.Z[iz])

    if hasattr(kernel, "dR") and 1 <= ir < len(kernel.R) - 1:
        a = float(psi[iz, ir - 1])
        b = float(psi[iz, ir])
        c = float(psi[iz, ir + 1])
        denom = 2.0 * (a - 2.0 * b + c)
        if abs(denom) > 1e-30:
            r_axis += float(np.clip(-(c - a) / denom, -0.5, 0.5)) * float(kernel.dR)
    if hasattr(kernel, "dZ") and 1 <= iz < len(kernel.Z) - 1:
        a = float(psi[iz - 1, ir])
        b = float(psi[iz, ir])
        c = float(psi[iz + 1, ir])
        denom = 2.0 * (a - 2.0 * b + c)
        if abs(denom) > 1e-30:
            z_axis += float(np.clip(-(c - a) / denom, -0.5, 0.5)) * float(kernel.dZ)

    x_point, _ = kernel.find_x_point(psi)
    return np.array([r_axis, z_axis, float(x_point[0]), float(x_point[1])], dtype=np.float64)
