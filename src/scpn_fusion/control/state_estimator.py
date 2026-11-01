# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — State Estimation
"""
Extended Kalman Filter (EKF) for plasma state estimation.

Provides robust filtering of noisy diagnostic signals for position,
current, and temperature tracking.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.spd_linalg import joseph_update, require_psd, spd_solve, symmetrize

FloatArray = NDArray[np.float64]


class ExtendedKalmanFilter:
    """Extended Kalman Filter for 6D plasma state estimation.

    State vector x: [R, Z, vR, vZ, Ip, Te_core]
    Measurement vector z: [R, Z, Ip, Te_core]

    Parameters
    ----------
    x0 : np.ndarray
        Initial state estimate (6D).
    P0 : np.ndarray
        Initial covariance estimate (6x6).
    Q : np.ndarray
        Process noise covariance (6x6).
    R_cov : np.ndarray
        Measurement noise covariance (4x4); must be symmetric positive
        semi-definite. ``R_cov = 0`` (noise-free measurements) is allowed: the
        innovation covariance ``H P H^T + R`` is checked for positive
        definiteness at every update instead.

    Raises
    ------
    NonSPDMatrixError
        If ``P0`` is not symmetric or ``R_cov`` is not positive semi-definite.
    """

    def __init__(
        self,
        x0: FloatArray,
        P0: FloatArray,
        Q: FloatArray,
        R_cov: FloatArray,
    ) -> None:
        self.x = x0.astype(float)
        self.P = symmetrize(P0, name="P0")
        self.Q = Q.astype(float)
        self.R = require_psd(R_cov, name="R_cov")

        # Linear measurement matrix H (4x6)
        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1.0  # R
        self.H[1, 1] = 1.0  # Z
        self.H[2, 4] = 1.0  # Ip
        self.H[3, 5] = 1.0  # Te_core

    def predict(self, dt: float, u: FloatArray | None = None) -> FloatArray:
        """Advance the state estimate and covariance using the process model.

        Parameters
        ----------
        dt : float
            Time step [s].
        u : np.ndarray, optional
            Control input (not used in constant-velocity model).

        Returns
        -------
        np.ndarray — Predicted state.
        """
        # 1. State transition (Constant velocity model)
        # R_new = R + vR * dt
        # Z_new = Z + vZ * dt
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt

        self.x = F @ self.x

        # 2. Covariance propagation
        # Q is usually scaled by dt or dt^2
        self.P = F @ self.P @ F.T + self.Q * dt

        return self.x

    def update(self, z: FloatArray) -> FloatArray:
        """Correct the state estimate using a new measurement.

        Parameters
        ----------
        z : np.ndarray
            Measured values [R, Z, Ip, Te_core] (4D).

        Returns
        -------
        np.ndarray — Updated state.
        """
        # 1. Innovation
        y = z - self.H @ self.x

        # 2. Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # 3. Optimal Kalman gain K = P H^T S^-1, solved as K^T = S^-1 (H P)
        #    with a Cholesky factor of the SPD innovation covariance.
        K = spd_solve(S, self.H @ self.P, name="innovation covariance").T

        # 4. Update state; Joseph-form covariance stays symmetric PSD.
        self.x = self.x + K @ y
        self.P = joseph_update(self.P, K, self.H, self.R)

        return self.x

    def estimate(self) -> FloatArray:
        """Return the current state estimate."""
        return self.x.copy()
