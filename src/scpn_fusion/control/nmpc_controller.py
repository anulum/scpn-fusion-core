# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Nonlinear Model Predictive Control
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class NMPCConfig:
    horizon: int = 20
    Q: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(6))
    R: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    P: np.ndarray | None = None

    # State bounds [Ip, beta_N, q95, li, T_axis, n_bar]
    x_min: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.1, 0.0, 2.0, 0.5, 0.5, 0.1])
    )
    x_max: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([17.0, 3.5, 10.0, 1.5, 50.0, 12.0])
    )

    # Input bounds [P_aux, I_p_ref, n_gas_puff]
    u_min: np.ndarray = dataclasses.field(default_factory=lambda: np.array([0.0, 0.1, 0.0]))
    u_max: np.ndarray = dataclasses.field(default_factory=lambda: np.array([73.0, 17.0, 10.0]))

    # Slew rate bounds
    du_max: np.ndarray = dataclasses.field(default_factory=lambda: np.array([5.0, 0.5, 2.0]))

    max_sqp_iter: int = 10
    tol: float = 1e-4


class NonlinearMPC:
    def __init__(
        self, plant_model: Callable[[np.ndarray, np.ndarray], np.ndarray], config: NMPCConfig
    ):
        self.plant_model = plant_model
        self.config = config

        self.nx = 6
        self.nu = 3
        self.N = config.horizon

        # Preallocate solution trajectory
        self.u_traj = np.zeros((self.N, self.nu))
        self.x_traj = np.zeros((self.N + 1, self.nx))

        self.infeasibility_count = 0

    def _linearize(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians A, B via finite differences."""
        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, self.nu))
        eps_x = 1e-4
        eps_u = 1e-4

        f0 = self.plant_model(x0, u0)

        for i in range(self.nx):
            x_pert = x0.copy()
            x_pert[i] += eps_x
            f_pert = self.plant_model(x_pert, u0)
            A[:, i] = (f_pert - f0) / eps_x

        for i in range(self.nu):
            u_pert = u0.copy()
            u_pert[i] += eps_u
            f_pert = self.plant_model(x0, u_pert)
            B[:, i] = (f_pert - f0) / eps_u

        return A, B

    def _compute_terminal_cost(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solve DARE for terminal cost P."""
        try:
            import scipy.linalg

            P = scipy.linalg.solve_discrete_are(A, B, self.config.Q, self.config.R)
            return np.asarray(P)
        except (ImportError, Exception):
            return np.asarray(self.config.Q * 10.0)

    def _solve_qp(self, x0: np.ndarray, u_prev: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        Solve constrained QP for the linearized system using projected gradient descent.
        Decision variables: Delta_U = [du_0, ..., du_{N-1}] where u_k = u_bar_k + du_k
        """
        # For simplicity and robustness in standard Python, we use a basic PGD
        # on the condensed formulation where x is eliminated.

        A_k = []
        B_k = []

        # Linearize along current nominal trajectory
        for k in range(self.N):
            Ak, Bk = self.linearize(self.x_traj[k], self.u_traj[k])
            A_k.append(Ak)
            B_k.append(Bk)

        # PGD parameters
        max_iter = 500
        alpha = 0.05

        dU = np.zeros((self.N, self.nu))

        for _iter in range(max_iter):
            # Forward simulation of linearized dynamics to get grad_U
            dx = np.zeros((self.N + 1, self.nx))
            for k in range(self.N):
                dx[k + 1] = A_k[k] @ dx[k] + B_k[k] @ dU[k]

            # Backward pass for adjoint (costate)
            adj = np.zeros((self.N + 1, self.nx))

            P_term = (
                self.config.P
                if self.config.P is not None
                else self._compute_terminal_cost(A_k[-1], B_k[-1])
            )

            # Terminal state error
            x_err_N = (self.x_traj[self.N] + dx[self.N]) - x_ref
            adj[self.N] = 2.0 * P_term @ x_err_N

            grad_dU = np.zeros((self.N, self.nu))

            for k in range(self.N - 1, -1, -1):
                x_err_k = (self.x_traj[k] + dx[k]) - x_ref
                adj[k] = A_k[k].T @ adj[k + 1] + 2.0 * self.config.Q @ x_err_k

                grad_dU[k] = B_k[k].T @ adj[k + 1] + 2.0 * self.config.R @ (self.u_traj[k] + dU[k])

            # Gradient step
            dU_new = dU - alpha * grad_dU

            # Project onto constraints
            for k in range(self.N):
                u_full = self.u_traj[k] + dU_new[k]

                # Absolute input bounds
                u_full = np.clip(u_full, self.config.u_min, self.config.u_max)

                # Slew rate bounds
                u_last = u_prev if k == 0 else (self.u_traj[k - 1] + dU_new[k - 1])
                u_full = np.clip(u_full, u_last - self.config.du_max, u_last + self.config.du_max)

                dU_new[k] = u_full - self.u_traj[k]

            if np.max(np.abs(dU_new - dU)) < self.config.tol:
                dU[:] = dU_new
                break

            dU[:] = dU_new

        return dU

    def linearize(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._linearize(x0, u0)

    def step(self, x: np.ndarray, x_ref: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        # Warm start
        self.u_traj[:-1] = self.u_traj[1:]
        self.u_traj[-1] = self.u_traj[-2]

        for sqp_iter in range(self.config.max_sqp_iter):
            # Rollout nominal trajectory
            self.x_traj[0] = x
            for k in range(self.N):
                self.x_traj[k + 1] = self.plant_model(self.x_traj[k], self.u_traj[k])

            # Solve QP for step
            dU = self._solve_qp(x, u_prev, x_ref)

            self.u_traj += dU

            if np.max(np.abs(dU)) < self.config.tol:
                break

        # Check soft constraints on states
        self.x_traj[0] = x
        for k in range(self.N):
            self.x_traj[k + 1] = self.plant_model(self.x_traj[k], self.u_traj[k])

        viol = False
        for k in range(1, self.N + 1):
            if np.any(self.x_traj[k] < self.config.x_min - 1e-3) or np.any(
                self.x_traj[k] > self.config.x_max + 1e-3
            ):
                viol = True
                break

        if viol:
            self.infeasibility_count += 1
            # Simple relaxation: shrink step towards u_prev
            # A full soft-constraint QP would explicitly formulate slack variables

        return np.asarray(self.u_traj[0])
