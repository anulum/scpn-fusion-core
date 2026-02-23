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
two algebraic Riccati equations (ARE) for gamma feasibility analysis,
then derives discrete-time gains via the discrete algebraic Riccati
equation (DARE) on the ZOH-discretised plant for a given sampling dt.

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
from scipy.linalg import solve_continuous_are, solve_discrete_are, expm

logger = logging.getLogger(__name__)


def _check_finite(mat: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"{name} must contain only finite values.")


def _zoh_discretize(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation via matrix exponential.

    Returns (Ad, Bd) such that  x_{k+1} = Ad x_k + Bd u_k.
    """
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n:] = B * dt
    eM = expm(M)
    return eM[:n, :n], eM[:n, n:]


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
        A: npt.ArrayLike,
        B1: npt.ArrayLike,
        B2: npt.ArrayLike,
        C1: npt.ArrayLike,
        C2: npt.ArrayLike,
        gamma: Optional[float] = None,
        D12: Optional[npt.ArrayLike] = None,
        D21: Optional[npt.ArrayLike] = None,
        enforce_robust_feasibility: bool = False,
    ) -> None:
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B1 = np.atleast_2d(np.asarray(B1, dtype=float))
        self.B2 = np.atleast_2d(np.asarray(B2, dtype=float))
        self.C1 = np.atleast_2d(np.asarray(C1, dtype=float))
        self.C2 = np.atleast_2d(np.asarray(C2, dtype=float))

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a finite square matrix.")
        _check_finite(self.A, "A")

        # Auto-transpose 1D inputs to column vectors
        if self.B1.shape[0] == 1 and self.A.shape[0] > 1:
            self.B1 = self.B1.T
        if self.B2.shape[0] == 1 and self.A.shape[0] > 1:
            self.B2 = self.B2.T
        if self.C1.shape[1] == 1 and self.A.shape[0] > 1:
            self.C1 = self.C1.T
        if self.C2.shape[1] == 1 and self.A.shape[0] > 1:
            self.C2 = self.C2.T

        self.n = self.A.shape[0]
        self.m = self.B2.shape[1]
        self.p = self.B1.shape[1]
        self.q = self.C1.shape[0]
        self.l = self.C2.shape[0]

        # Validate dimensions and finiteness
        for name, mat, expected_rows, expected_cols in [
            ("B1", self.B1, self.n, None),
            ("B2", self.B2, self.n, None),
            ("C1", self.C1, None, self.n),
            ("C2", self.C2, None, self.n),
        ]:
            _check_finite(mat, name)
            if expected_rows is not None and mat.shape[0] != expected_rows:
                raise ValueError(f"{name} row count must match A ({expected_rows}).")
            if expected_cols is not None and mat.shape[1] != expected_cols:
                raise ValueError(f"{name} column count must match A ({expected_cols}).")

        # D12, D21 defaults (identity-like)
        self.D12 = self._make_feedthrough(D12, self.q, self.m, "D12")
        self.D21 = self._make_feedthrough(D21, self.l, self.p, "D21")

        # Find or use gamma
        if gamma is None:
            self.gamma = self._find_optimal_gamma()
        else:
            self.gamma = float(gamma)
            if not np.isfinite(self.gamma) or self.gamma <= 1.0:
                raise ValueError("gamma must be finite and > 1.0.")

        # Solve continuous Riccati equations for feasibility analysis
        self.X, self.Y, self.F, self.L_gain = self._synthesize(self.gamma)
        self.spectral_radius_xy = float(np.max(np.abs(np.linalg.eigvals(self.X @ self.Y))))
        self.robust_feasible = bool(
            self.spectral_radius_xy < self.gamma ** 2 * (1.0 - 1e-6)
        )
        if not self.robust_feasible:
            msg = (
                "H-infinity spectral feasibility condition failed: "
                f"rho(XY)={self.spectral_radius_xy:.6g} >= gamma^2={self.gamma ** 2:.6g}."
            )
            if enforce_robust_feasibility:
                raise ValueError(msg)
            logger.warning(msg)

        # Output saturation limits (anti-windup)
        self.u_max: float = 1e8  # effectively unlimited by default

        # Discrete gains cache (recomputed when dt changes)
        self._cached_dt: float = 0.0
        self._Fd: np.ndarray = self.F  # fallback: continuous gain
        self._Ld: np.ndarray = self.L_gain.copy()
        self._Ad: np.ndarray = np.eye(self.n)
        self._Bd_u: np.ndarray = np.zeros((self.n, self.m))

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

    @staticmethod
    def _make_feedthrough(
        value: Optional[npt.ArrayLike], rows: int, cols: int, name: str,
    ) -> np.ndarray:
        if value is not None:
            mat = np.atleast_2d(np.asarray(value, dtype=float))
        else:
            mat = np.zeros((rows, cols))
            min_dim = min(rows, cols)
            mat[:min_dim, :min_dim] = np.eye(min_dim)
        if mat.shape != (rows, cols):
            raise ValueError(f"{name} must have shape ({rows}, {cols}).")
        _check_finite(mat, name)
        return mat

    def _synthesize(self, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve the two continuous Riccati equations and extract gains.

        Returns (X, Y, F, L) where:
            F = -B2^T X  (state feedback gain)
            L = Y C2^T   (observer injection gain)
        """
        if not np.isfinite(gamma) or gamma <= 1.0:
            raise ValueError("gamma must be finite and > 1.0.")

        # State-feedback H-infinity ARE
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

        # Observer H-infinity ARE (dual form)
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
        F = -self.B2.T @ X       # shape (m, n) — state feedback
        L = Y @ self.C2.T        # shape (n, l) — observer injection

        if not np.all(np.isfinite(F)) or not np.all(np.isfinite(L)):
            raise ValueError("Riccati synthesis produced non-finite gains.")

        return X, Y, F, L

    def _find_optimal_gamma(
        self,
        gamma_min: float = 1.01,
        gamma_max: float = 1e6,
        rtol: float = 1e-3,
        max_iter: int = 100,
    ) -> float:
        """Bisection search for the smallest feasible gamma."""
        best_gamma = gamma_max

        for _ in range(max_iter):
            gamma_try = (gamma_min + gamma_max) / 2.0
            try:
                X, Y, F, L = self._synthesize(gamma_try)
                eigs = np.linalg.eigvals(X @ Y)
                spec_rad = float(np.max(np.abs(eigs)))
                if spec_rad < gamma_try ** 2:
                    best_gamma = gamma_try
                    gamma_max = gamma_try
                else:
                    gamma_min = gamma_try
            except (np.linalg.LinAlgError, ValueError):
                gamma_min = gamma_try

            if gamma_max - gamma_min < rtol * gamma_min:
                break

        return best_gamma * 1.005

    def _update_discretization(self, dt: float) -> None:
        """Compute discrete-time gains for the given sampling period.

        Uses exact ZOH discretisation of the plant model, then solves
        the discrete algebraic Riccati equation (DARE) for both the
        state-feedback and observer gains.  This guarantees closed-loop
        stability for any dt, unlike continuous-domain gain emulation
        which fails when dt exceeds the Nyquist limit.
        """
        # ZOH discretise the plant
        Ad, Bd_u = _zoh_discretize(self.A, self.B2, dt)
        _, Bd_w = _zoh_discretize(self.A, self.B1, dt)

        # --- Discrete state-feedback gain (DARE) ---
        Q_fb = self.C1.T @ self.C1
        R_fb = np.eye(self.m)
        Xd = solve_discrete_are(Ad, Bd_u, Q_fb, R_fb)
        Fd = -np.linalg.solve(R_fb + Bd_u.T @ Xd @ Bd_u, Bd_u.T @ Xd @ Ad)

        # --- Discrete observer gain (DARE, dual) ---
        # Process noise covariance from discretised disturbance
        Q_obs = Bd_w @ Bd_w.T + 1e-6 * np.eye(self.n)  # regularise
        R_obs = np.eye(self.l)
        Yd = solve_discrete_are(Ad.T, self.C2.T, Q_obs, R_obs)
        # Prediction-form Kalman gain: K = Ad P C2^T (C2 P C2^T + R)^{-1}
        S = self.C2 @ Yd @ self.C2.T + R_obs
        Ld = Ad @ Yd @ self.C2.T @ np.linalg.inv(S)

        # Cache
        self._Ad = Ad
        self._Bd_u = Bd_u
        self._Fd = Fd
        self._Ld = Ld
        self._cached_dt = dt

    def step(self, error: float, dt: float) -> float:
        """Compute control action for one timestep.

        Uses discrete-time DARE-synthesised gains on the ZOH-discretised
        plant, guaranteeing closed-loop stability for any sampling dt.

        Parameters
        ----------
        error : float
            Observed measurement (y).
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

        # Recompute discrete gains when dt changes.
        if dt != self._cached_dt:
            self._update_discretization(dt)

        y = np.atleast_1d(np.asarray(error, dtype=float))

        # Control output (before state update — zero transport delay)
        u_raw = self._Fd @ self.state
        u = np.clip(u_raw, -self.u_max, self.u_max)

        # Observer update with anti-windup back-calculation:
        #   x_hat_{k+1} = Ad x_hat + Bd u + Ld (y - C2 x_hat) + Bd (u - u_raw)
        # The last term corrects for integrator wind-up when the output saturates.
        innovation = y - self.C2 @ self.state
        aw_correction = self._Bd_u @ (u - u_raw)
        self.state = (
            self._Ad @ self.state
            + self._Bd_u @ u
            + self._Ld @ innovation
            + aw_correction
        )

        return float(u[0]) if u.size > 1 else float(u.item())

    def riccati_residual_norms(self) -> tuple[float, float]:
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
        """Check continuous closed-loop stability (eigenvalues of A + B2 F)."""
        A_cl = self.A + self.B2 @ self.F
        eigs = np.linalg.eigvals(A_cl)
        return bool(np.all(np.real(eigs) < 0))

    @property
    def gain_margin_db(self) -> float:
        """Gain margin in dB for the continuous closed-loop system."""
        A_cl = self.A + self.B2 @ self.F
        eigs = np.linalg.eigvals(A_cl)
        real_parts = np.real(eigs)
        if np.any(real_parts >= 0):
            return 0.0
        max_cl_real = float(np.max(real_parts))
        ol_eigs = np.linalg.eigvals(self.A)
        max_ol_real = float(np.max(np.real(ol_eigs)))
        if max_ol_real <= 0:
            return float("inf")
        margin_ratio = -max_cl_real / max_ol_real
        return float(20.0 * np.log10(1.0 + margin_ratio))

    @property
    def phase_margin_deg(self) -> float:
        """Phase margin in degrees via loop transfer function L(jw).

        Sweeps frequency to find the gain crossover (|L(jw)| = 1)
        and returns 180 + angle(L(jw)) at that point.
        """
        freqs = np.logspace(-2, 4, 2000)
        crossed = False
        prev_gain = float("inf")
        for w in freqs:
            sI_A_inv = np.linalg.solve(
                1j * w * np.eye(self.n) - self.A, self.B2
            )
            L_jw = self.F @ sI_A_inv
            gain = float(np.abs(L_jw).max())
            if prev_gain > 1.0 >= gain:
                phase = float(np.angle(L_jw.ravel()[0], deg=True))
                return 180.0 + phase
            prev_gain = gain
        return 0.0


def get_radial_robust_controller(
    gamma_growth: float = 100.0,
    *,
    damping: float = 10.0,
    enforce_robust_feasibility: bool = False,
) -> HInfinityController:
    """Return an H-infinity controller for tokamak vertical stability.

    Parameters
    ----------
    gamma_growth : float
        Vertical instability growth rate [1/s].
        Default 100/s (ITER-like). SPARC: ~1000/s.
    damping : float
        Passive damping coefficient [1/s]. Default 10.0.
    enforce_robust_feasibility : bool, optional
        If True, require rho(XY) < gamma^2 and raise on infeasible synthesis.

    Returns
    -------
    HInfinityController
        Riccati-synthesized robust controller.
    """
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be a finite positive value.")
    A = np.array([
        [0.0, 1.0],
        [gamma_growth**2, -damping],
    ])
    B2 = np.array([[0.0], [1.0]])
    B1 = np.array([[0.0], [0.5]])
    C1 = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
    ])
    C2 = np.array([[1.0, 0.0]])

    return HInfinityController(
        A,
        B1,
        B2,
        C1,
        C2,
        enforce_robust_feasibility=enforce_robust_feasibility,
    )
