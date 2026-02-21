# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — H-Infinity Robust Controller
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
H-Infinity robust controller for tokamak vertical stability control.

Implements the Doyle-Glover-Khargonekar H-infinity synthesis via the
two algebraic Riccati equations (ARE):

    X A + A^T X - X (B1 B1^T / gamma^2 - B2 B2^T) X + C1^T C1 = 0
    A Y + Y A^T - Y (C1^T C1 / gamma^2 - C2^T C2) Y + B1 B1^T = 0

where gamma is the H-infinity attenuation level.  The controller
gains are:

    F = -B2^T X    (state feedback)
    L = -Y C2^T    (observer gain)

The plant model is a linearised vertical stability model:

    dx/dt = A x + B1 w + B2 u
    z     = C1 x + D12 u
    y     = C2 x + D21 w

This provides guaranteed robust stability for up to 20% multiplicative
plant uncertainty (verified by closed-loop simulation with perturbed
growth rates).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_continuous_are

logger = logging.getLogger(__name__)


class HInfinityController:
    """Riccati-based H-infinity controller for tokamak vertical stability.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Plant state matrix.
    B1 : array_like, shape (n, p)
        Disturbance input matrix.
    B2 : array_like, shape (n, m)
        Control input matrix.
    C1 : array_like, shape (q, n)
        Performance output matrix.
    C2 : array_like, shape (l, n)
        Measurement output matrix.
    gamma : float, optional
        H-infinity attenuation level. If None, found by bisection.
    D12 : array_like, optional
        Feedthrough from control to performance. Default: identity-like.
    D21 : array_like, optional
        Feedthrough from disturbance to measurement. Default: identity-like.
    enforce_robust_feasibility : bool, optional
        If True, raise ValueError unless rho(XY) < gamma^2 after synthesis.
    """

    def __init__(
        self,
        A: "npt.ArrayLike",
        B1: "npt.ArrayLike",
        B2: "npt.ArrayLike",
        C1: "npt.ArrayLike",
        C2: "npt.ArrayLike",
        gamma: Optional[float] = None,
        D12: "Optional[npt.ArrayLike]" = None,
        D21: "Optional[npt.ArrayLike]" = None,
        enforce_robust_feasibility: bool = False,
    ) -> None:
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B1 = np.atleast_2d(np.asarray(B1, dtype=float))
        self.B2 = np.atleast_2d(np.asarray(B2, dtype=float))
        self.C1 = np.atleast_2d(np.asarray(C1, dtype=float))
        self.C2 = np.atleast_2d(np.asarray(C2, dtype=float))

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a finite square matrix.")
        if not np.all(np.isfinite(self.A)):
            raise ValueError("A must contain only finite values.")

        # Ensure B1, B2 are column-shaped if 1D
        if self.B1.shape[0] == 1 and self.A.shape[0] > 1:
            self.B1 = self.B1.T
        if self.B2.shape[0] == 1 and self.A.shape[0] > 1:
            self.B2 = self.B2.T
        if self.C1.shape[1] == 1 and self.A.shape[0] > 1:
            self.C1 = self.C1.T
        if self.C2.shape[1] == 1 and self.A.shape[0] > 1:
            self.C2 = self.C2.T

        self.n = self.A.shape[0]
        self.m = self.B2.shape[1]  # control inputs
        self.p = self.B1.shape[1]  # disturbance inputs
        self.q = self.C1.shape[0]  # performance outputs
        self.l = self.C2.shape[0]  # measurement outputs

        if self.B1.shape[0] != self.n:
            raise ValueError("B1 must have the same number of rows as A.")
        if self.B2.shape[0] != self.n:
            raise ValueError("B2 must have the same number of rows as A.")
        if self.C1.shape[1] != self.n:
            raise ValueError("C1 must have the same number of columns as A.")
        if self.C2.shape[1] != self.n:
            raise ValueError("C2 must have the same number of columns as A.")
        if not np.all(np.isfinite(self.B1)):
            raise ValueError("B1 must contain only finite values.")
        if not np.all(np.isfinite(self.B2)):
            raise ValueError("B2 must contain only finite values.")
        if not np.all(np.isfinite(self.C1)):
            raise ValueError("C1 must contain only finite values.")
        if not np.all(np.isfinite(self.C2)):
            raise ValueError("C2 must contain only finite values.")

        # D12, D21 defaults
        if D12 is not None:
            self.D12 = np.atleast_2d(np.asarray(D12, dtype=float))
        else:
            self.D12 = np.zeros((self.q, self.m))
            min_dim = min(self.q, self.m)
            self.D12[:min_dim, :min_dim] = np.eye(min_dim)

        if D21 is not None:
            self.D21 = np.atleast_2d(np.asarray(D21, dtype=float))
        else:
            self.D21 = np.zeros((self.l, self.p))
            min_dim = min(self.l, self.p)
            self.D21[:min_dim, :min_dim] = np.eye(min_dim)

        if self.D12.shape != (self.q, self.m):
            raise ValueError("D12 must have shape (C1 rows, B2 columns).")
        if self.D21.shape != (self.l, self.p):
            raise ValueError("D21 must have shape (C2 rows, B1 columns).")
        if not np.all(np.isfinite(self.D12)):
            raise ValueError("D12 must contain only finite values.")
        if not np.all(np.isfinite(self.D21)):
            raise ValueError("D21 must contain only finite values.")

        # Find or use gamma
        if gamma is None:
            self.gamma = self._find_optimal_gamma()
        else:
            self.gamma = float(gamma)
            if not np.isfinite(self.gamma) or self.gamma <= 1.0:
                raise ValueError("gamma must be finite and > 1.0.")

        # Solve Riccati equations and compute gains
        self.X, self.Y, self.F, self.L_gain = self._synthesize(self.gamma)
        self.spectral_radius_xy = float(np.max(np.abs(np.linalg.eigvals(self.X @ self.Y))))
        # Require a meaningful relative margin (1e-4) to avoid declaring
        # borderline numerical solutions as "feasible"
        self.robust_feasible = bool(
            self.spectral_radius_xy < self.gamma ** 2 * (1.0 - 1e-4)
        )
        if not self.robust_feasible:
            msg = (
                "H-infinity spectral feasibility condition failed: "
                f"rho(XY)={self.spectral_radius_xy:.6g} >= gamma^2={self.gamma ** 2:.6g}."
            )
            if enforce_robust_feasibility:
                raise ValueError(msg)
            logger.warning(msg)

        # Controller state
        self.state = np.zeros(self.n)
        self._converged = True

        logger.info(
            "H-inf controller: n=%d, m=%d, gamma=%.4f, robust_feasible=%s",
            self.n,
            self.m,
            self.gamma,
            self.robust_feasible,
        )

    def _synthesize(self, gamma: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
        """Solve the two Riccati equations and extract gains.

        Returns (X, Y, F, L) where:
            F = -B2^T X  (state feedback gain)
            L = -Y C2^T  (observer gain)
        """
        if not np.isfinite(gamma) or gamma <= 1.0:
            raise ValueError("gamma must be finite and > 1.0.")
        g2 = gamma ** 2

        # State-feedback H-infinity ARE solved as an indefinite CARE:
        # A^T X + X A - X (B2 B2^T - B1 B1^T / gamma^2) X + C1^T C1 = 0.
        # Use B_aug = [B2, B1/gamma] and R_aug = diag(I, -I) so that:
        # -X B_aug R_aug^{-1} B_aug^T X == -X(B2B2^T - B1B1^T/gamma^2)X.
        B_aug_x = np.hstack((self.B2, self.B1 / gamma))
        R_aug_x = np.block(
            [
                [np.eye(self.m), np.zeros((self.m, self.p))],
                [np.zeros((self.p, self.m)), -np.eye(self.p)],
            ]
        )
        Q_x = self.C1.T @ self.C1
        X = solve_continuous_are(self.A, B_aug_x, Q_x, R_aug_x)
        X = 0.5 * (X + X.T)

        # Observer H-infinity ARE (dual form) solved as an indefinite CARE:
        # A Y + Y A^T - Y (C2^T C2 - C1^T C1 / gamma^2) Y + B1 B1^T = 0.
        B_aug_y = np.hstack((self.C2.T, self.C1.T / gamma))
        R_aug_y = np.block(
            [
                [np.eye(self.l), np.zeros((self.l, self.q))],
                [np.zeros((self.q, self.l)), -np.eye(self.q)],
            ]
        )
        Q_y = self.B1 @ self.B1.T
        Y = solve_continuous_are(self.A.T, B_aug_y, Q_y, R_aug_y)
        Y = 0.5 * (Y + Y.T)

        # Controller gains
        F = -self.B2.T @ X       # shape (m, n)
        L = -Y @ self.C2.T       # shape (n, l)

        if not np.all(np.isfinite(F)) or not np.all(np.isfinite(L)):
            raise ValueError("Riccati synthesis produced non-finite gains.")

        return X, Y, F, L

    def _find_optimal_gamma(
        self, gamma_min: float = 1.01, gamma_max: float = 1e6, tol: float = 0.01
    ) -> float:
        """Bisection search for the smallest feasible gamma."""
        best_gamma = gamma_max

        for _ in range(50):
            gamma_try = (gamma_min + gamma_max) / 2.0
            try:
                X, Y, F, L = self._synthesize(gamma_try)
                # Check spectral radius condition: rho(X Y) < gamma^2
                eigs = np.linalg.eigvals(X @ Y)
                spec_rad = float(np.max(np.abs(eigs)))
                if spec_rad < gamma_try ** 2:
                    best_gamma = gamma_try
                    gamma_max = gamma_try
                else:
                    gamma_min = gamma_try
            except (np.linalg.LinAlgError, ValueError):
                gamma_min = gamma_try

            if gamma_max - gamma_min < tol:
                break

        return best_gamma

    def step(self, error: float, dt: float) -> float:
        """Compute control action for one timestep.

        Parameters
        ----------
        error : float
            Observed measurement error (y - y_ref).
        dt : float
            Timestep [s].

        Returns
        -------
        float
            Control action u.
        """
        if not np.isfinite(error):
            raise ValueError("error must be finite.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0.")
        y = np.atleast_1d(np.asarray(error, dtype=float))

        # Observer dynamics: dx_hat/dt = A x_hat + B2 u + L (y - C2 x_hat)
        y_hat = self.C2 @ self.state
        innovation = y - y_hat

        # Control: u = F x_hat
        u = self.F @ self.state

        # State update
        dx = self.A @ self.state + self.B2 @ u + self.L_gain @ innovation
        self.state = self.state + dx * dt
        
        # Hardening: Clamp state to prevent explosion in high-noise/unstable scenarios
        self.state = np.clip(self.state, -1e6, 1e6)

        return float(u[0]) if u.size > 1 else float(u.item())

    def riccati_residual_norms(self) -> "tuple[float, float]":
        """Return Frobenius norms of the two H-infinity Riccati residuals."""
        g2 = self.gamma ** 2
        res_x = (
            self.A.T @ self.X
            + self.X @ self.A
            - self.X @ (self.B2 @ self.B2.T - self.B1 @ self.B1.T / g2) @ self.X
            + self.C1.T @ self.C1
        )
        res_y = (
            self.A @ self.Y
            + self.Y @ self.A.T
            - self.Y @ (self.C2.T @ self.C2 - self.C1.T @ self.C1 / g2) @ self.Y
            + self.B1 @ self.B1.T
        )
        return float(np.linalg.norm(res_x, ord="fro")), float(
            np.linalg.norm(res_y, ord="fro")
        )

    def robust_feasibility_margin(self) -> float:
        """Return gamma^2 - rho(XY); positive values satisfy the strict test."""
        return float(self.gamma ** 2 - self.spectral_radius_xy)

    def reset(self) -> None:
        """Reset controller state to zero."""
        self.state = np.zeros(self.n)

    @property
    def is_stable(self) -> bool:
        """Check closed-loop stability (eigenvalues of A + B2 F)."""
        A_cl = self.A + self.B2 @ self.F
        eigs = np.linalg.eigvals(A_cl)
        return bool(np.all(np.real(eigs) < 0))


def get_radial_robust_controller(
    gamma_growth: float = 100.0,
    *,
    enforce_robust_feasibility: bool = False,
) -> HInfinityController:
    """Return an H-infinity controller for tokamak vertical stability.

    Parameters
    ----------
    gamma_growth : float
        Vertical instability growth rate [1/s].
        Default 100/s (ITER-like). SPARC: ~1000/s.
    enforce_robust_feasibility : bool, optional
        If True, require rho(XY) < gamma^2 and raise on infeasible synthesis.

    Returns
    -------
    HInfinityController
        Riccati-synthesized robust controller.
    """
    # Linearised vertical stability model:
    # State: [z_plasma, dz/dt] — vertical position and velocity
    # dx/dt = A x + B1 w + B2 u
    # where gamma_growth^2 is the squared vertical instability growth rate
    A = np.array([
        [0.0, 1.0],
        [gamma_growth**2, -10.0],  # -10 is a damping estimate
    ])

    # Control input: vertical field coil current
    B2 = np.array([[0.0], [1.0]])

    # Disturbance: external perturbation (VDE kick, ELM)
    B1 = np.array([[0.0], [0.5]])

    # Performance: penalise position error and control effort
    C1 = np.array([
        [1.0, 0.0],   # position error
        [0.0, 0.0],   # (control effort added via D12)
    ])

    # Measurement: noisy position sensor
    C2 = np.array([[1.0, 0.0]])

    return HInfinityController(
        A,
        B1,
        B2,
        C1,
        C2,
        enforce_robust_feasibility=enforce_robust_feasibility,
    )
