# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
        Measurement noise covariance (4x4).
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R_cov: np.ndarray,
    ) -> None:
        self.x = x0.astype(float)
        self.P = P0.astype(float)
        self.Q = Q.astype(float)
        self.R = R_cov.astype(float)

        # Linear measurement matrix H (4x6)
        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1.0  # R
        self.H[1, 1] = 1.0  # Z
        self.H[2, 4] = 1.0  # Ip
        self.H[3, 5] = 1.0  # Te_core

    def predict(self, dt: float, u: np.ndarray | None = None) -> np.ndarray:
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

    def update(self, z: np.ndarray) -> np.ndarray:
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

        # 3. Optimal Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 4. Update state and covariance
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        return self.x

    def estimate(self) -> np.ndarray:
        """Return the current state estimate."""
        return self.x.copy()
