# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Flight Sim Controllers (LQR + H-inf plant models)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Flight-simulation-matched controllers for tokamak position control.

Provides LQR and H-infinity controllers pre-configured for the
IsoFluxController flight-sim dynamics (quasi-static equilibrium
response through a first-order actuator).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_discrete_are

from .h_infinity_controller import HInfinityController, _zoh_discretize


class LQRController:
    """Discrete LQR controller via DARE with Kalman observer.

    Computes optimal state-feedback K from the discrete algebraic Riccati
    equation and a Kalman observer gain L from the dual DARE.

    Parameters
    ----------
    A, B2, C2 : array_like
        Continuous-time plant matrices (discretized internally via ZOH).
    Q_diag : array_like or float
        State penalty diagonal (scalar broadcast to all states).
    R_diag : array_like or float
        Control penalty diagonal.
    """

    def __init__(
        self,
        A: npt.ArrayLike,
        B2: npt.ArrayLike,
        C2: npt.ArrayLike,
        Q_diag: float = 1.0,
        R_diag: float = 0.01,
    ) -> None:
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B2 = np.atleast_2d(np.asarray(B2, dtype=float))
        self.C2 = np.atleast_2d(np.asarray(C2, dtype=float))
        self.n = self.A.shape[0]
        self.m = self.B2.shape[1]
        self.l = self.C2.shape[0]
        self.Q = float(Q_diag) * np.eye(self.n)
        self.R = float(R_diag) * np.eye(self.m)

        self._cached_dt: float = 0.0
        self._K = np.zeros((self.m, self.n))
        self._L = np.zeros((self.n, self.l))
        self._Ad = np.eye(self.n)
        self._Bd = np.zeros_like(self.B2)
        self.state = np.zeros(self.n)

    def _update_discretization(self, dt: float) -> None:
        Ad, Bd = _zoh_discretize(self.A, self.B2, dt)

        P = solve_discrete_are(Ad, Bd, self.Q, self.R)
        self._K = -np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

        Q_obs = 0.1 * np.eye(self.n)
        R_obs = np.eye(self.l)
        Pk = solve_discrete_are(Ad.T, self.C2.T, Q_obs, R_obs)
        S = self.C2 @ Pk @ self.C2.T + R_obs
        self._L = Ad @ Pk @ self.C2.T @ np.linalg.inv(S)

        self._Ad = Ad
        self._Bd = Bd
        self._cached_dt = dt

    def step(self, error: float, dt: float) -> float:
        if dt != self._cached_dt:
            self._update_discretization(dt)

        y = np.atleast_1d(np.asarray(error, dtype=float))
        u = self._K @ self.state
        innovation = y - self.C2 @ self.state
        self.state = self._Ad @ self.state + self._Bd @ u + self._L @ innovation

        return float(u[0]) if u.size > 1 else float(u.item())

    def reset(self) -> None:
        self.state = np.zeros(self.n)


def get_flight_sim_controller(
    response_gain: float = 0.05,
    actuator_tau: float = 0.06,
    enforce_robust_feasibility: bool = False,
) -> HInfinityController:
    """H-inf controller matched to IsoFluxController flight-sim dynamics.

    Plant model: quasi-static equilibrium response through first-order actuator.

        dx1/dt = -g * x2          error driven by actuator output
        dx2/dt = (u - x2) / tau   first-order actuator lag
        y = x1                    error measurement

    Parameters
    ----------
    response_gain : float
        Sensitivity of position error to accumulated coil current [1/s].
        Radial channel: ~0.05, vertical: ~0.02.
    actuator_tau : float
        First-order actuator time constant [s]. Default 0.06.
    """
    if not np.isfinite(response_gain) or response_gain <= 0.0:
        raise ValueError("response_gain must be finite and > 0.")
    if not np.isfinite(actuator_tau) or actuator_tau <= 0.0:
        raise ValueError("actuator_tau must be finite and > 0.")
    inv_tau = 1.0 / actuator_tau
    A = np.array([[1.0, -response_gain], [0.0, -inv_tau]])
    B2 = np.array([[0.0], [inv_tau]])
    B1 = np.array([[1.0], [0.0]])
    C1 = np.array([[1.0, 0.0], [0.0, 0.01]])
    C2 = np.array([[1.0, 0.0]])
    return HInfinityController(
        A, B1, B2, C1, C2,
        enforce_robust_feasibility=enforce_robust_feasibility,
    )


def get_flight_sim_controller_v2(
    position_sensitivity: float = 0.567,
    actuator_tau: float = 0.06,
    sample_dt: float = 0.05,
    drift_rate: float = 0.1,
    observer_q_scale: float = 100.0,
    enforce_robust_feasibility: bool = False,
) -> HInfinityController:
    """H-inf controller with corrected integrator plant model.

    The flight sim accumulates actuator output into coil current each step
    (discrete integrator: I[k+1] = I[k] + u_applied[k]).  This version
    captures the integrator via an effective continuous-time gain of
    position_sensitivity / sample_dt.

    Plant model:
        dx1/dt = drift * x1 - (g/dt) * x2    error with Shafranov drift
        dx2/dt = (u - x2) / tau               first-order actuator lag
        y = x1                                 error measurement

    Parameters
    ----------
    position_sensitivity : float
        dR/dI sensitivity [m/MA].  ITER radial PF3: ~0.567.
    actuator_tau : float
        Actuator time constant [s].
    sample_dt : float
        Flight sim timestep [s].
    drift_rate : float
        Natural error growth rate [1/s] from Shafranov shift during Ip ramp.
    observer_q_scale : float
        Scales process noise covariance in the observer DARE.  Default 100.0.
    """
    if position_sensitivity <= 0:
        raise ValueError("position_sensitivity must be > 0.")
    if actuator_tau <= 0:
        raise ValueError("actuator_tau must be > 0.")
    if sample_dt <= 0:
        raise ValueError("sample_dt must be > 0.")

    inv_tau = 1.0 / actuator_tau
    g_eff = position_sensitivity / sample_dt

    A = np.array([[drift_rate, -g_eff], [0.0, -inv_tau]])
    B2 = np.array([[0.0], [inv_tau]])
    B1 = np.array([[1.0], [0.0]])
    C1 = np.array([[1.0, 0.0], [0.0, 0.01]])
    C2 = np.array([[1.0, 0.0]])

    ctrl = HInfinityController(
        A, B1, B2, C1, C2,
        enforce_robust_feasibility=enforce_robust_feasibility,
    )
    ctrl.observer_q_scale = float(observer_q_scale)
    return ctrl


def get_flight_sim_lqr_controller(
    position_sensitivity: float = 0.567,
    actuator_tau: float = 0.06,
    sample_dt: float = 0.05,
    drift_rate: float = 0.1,
    Q_diag: float = 10.0,
    R_diag: float = 0.01,
) -> LQRController:
    """LQR controller matched to the flight sim's integrator plant.

    Uses the same corrected plant model as get_flight_sim_controller_v2.
    """
    if position_sensitivity <= 0:
        raise ValueError("position_sensitivity must be > 0.")
    if actuator_tau <= 0:
        raise ValueError("actuator_tau must be > 0.")
    if sample_dt <= 0:
        raise ValueError("sample_dt must be > 0.")

    inv_tau = 1.0 / actuator_tau
    g_eff = position_sensitivity / sample_dt

    A = np.array([[drift_rate, -g_eff], [0.0, -inv_tau]])
    B2 = np.array([[0.0], [inv_tau]])
    C2 = np.array([[1.0, 0.0]])

    return LQRController(A, B2, C2, Q_diag=Q_diag, R_diag=R_diag)
