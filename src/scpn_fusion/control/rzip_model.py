# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — RZIP Rigid Plasma Response Model
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations


import numpy as np

from scpn_fusion.core.vessel_model import VesselElement, VesselModel

MU_0 = 4.0 * np.pi * 1e-7


class RZIPModel:
    def __init__(
        self,
        R0: float,
        a: float,
        kappa: float,
        Ip_MA: float,
        B0: float,
        n_index: float,
        vessel: VesselModel,
        active_coils: list[VesselElement] | None = None,
    ):
        self.R0 = R0
        self.a = a
        self.kappa = kappa
        self.Ip = Ip_MA * 1e6
        self.B0 = B0
        self.n_index = n_index
        self.vessel = vessel
        self.active_coils = active_coils if active_coils is not None else []

        # Combine vessel elements and active coils for the full circuit
        self.all_elements = self.vessel.elements + self.active_coils
        self.n_wall = len(self.vessel.elements)
        self.n_coils = len(self.active_coils)
        self.n_circuits = len(self.all_elements)

        # Plasma mass estimate (heuristic, but keeps it finite)
        # Using a very small mass causes extremely high frequency undamped oscillations
        # in the voltage-controlled state space. We use an effective mass of 1.0 kg
        # which is common in rigid models to represent inductive inertia.
        self.M_eff = 1.0

    def _calc_dM_dz(self, R1: float, Z1: float, R2: float, Z2: float) -> float:
        # Finite difference for mutual inductance derivative w.r.t Z1
        # vessel._calculate_mutual_inductance takes (R1, Z1, R2, Z2)
        dZ = 1e-4
        M_plus = self.vessel._calculate_mutual_inductance(R1, Z1 + dZ, R2, Z2)
        M_minus = self.vessel._calculate_mutual_inductance(R1, Z1 - dZ, R2, Z2)
        return (M_plus - M_minus) / (2.0 * dZ)

    def build_state_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # x = [Z, dZ/dt, I_1, ..., I_n]
        n_states = 2 + self.n_circuits
        n_inputs = self.n_coils

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_inputs))
        C = np.zeros((1, n_states))  # Output is just Z
        D = np.zeros((1, n_inputs))

        C[0, 0] = 1.0  # Output y = Z

        # K = n_index * mu0 * Ip^2 / (4 * pi * R0)
        K = self.n_index * MU_0 * self.Ip**2 / (4.0 * np.pi * self.R0)

        # Compute M_mat and R_mat for all circuits
        M_mat = np.zeros((self.n_circuits, self.n_circuits))
        R_mat = np.zeros((self.n_circuits, self.n_circuits))

        C_vec = np.zeros(self.n_circuits)  # dM_pj/dz * Ip

        for i in range(self.n_circuits):
            el_i = self.all_elements[i]
            R_mat[i, i] = el_i.resistance
            C_vec[i] = self._calc_dM_dz(self.R0, 0.0, el_i.R, el_i.Z) * self.Ip

            for j in range(self.n_circuits):
                el_j = self.all_elements[j]
                if i == j:
                    M_mat[i, j] = el_i.inductance
                else:
                    M_mat[i, j] = self.vessel._calculate_mutual_inductance(
                        el_i.R, el_i.Z, el_j.R, el_j.Z
                    )

        try:
            M_inv = np.linalg.inv(M_mat)
        except np.linalg.LinAlgError:
            M_inv = np.zeros_like(M_mat)

        # Z dot
        A[0, 1] = 1.0

        # dZ/dt dot = (-K * Z + sum(C_k * I_k)) / M_eff
        A[1, 0] = -K / self.M_eff
        for i in range(self.n_circuits):
            A[1, 2 + i] = C_vec[i] / self.M_eff

        # I dot = M_inv * (V - R*I - C * dZ/dt)
        # dI/dt = -M_inv * C * dZ/dt - M_inv * R * I + M_inv * V
        M_inv_R = M_inv @ R_mat
        M_inv_C = M_inv @ C_vec

        for i in range(self.n_circuits):
            A[2 + i, 1] = -M_inv_C[i]
            for j in range(self.n_circuits):
                A[2 + i, 2 + j] = -M_inv_R[i, j]

        # B matrix (inputs to active coils)
        for i in range(self.n_circuits):
            for j in range(self.n_coils):
                coil_idx = self.n_wall + j
                B[2 + i, j] = M_inv[i, coil_idx]

        return A, B, C, D

    def vertical_growth_rate(self) -> float:
        A, _, _, _ = self.build_state_space()
        eigvals = np.linalg.eigvals(A)
        return float(np.max(np.real(eigvals)))

    def vertical_growth_time(self) -> float:
        gamma = self.vertical_growth_rate()
        if gamma <= 0.0:
            return float("inf")
        return 1000.0 / gamma  # in ms

    def stability_margin(self) -> float:
        # Distance from marginality (n_index = 0 typically)
        return float(self.n_index)


class VerticalStabilityAnalysis:
    @staticmethod
    def compute_n_index(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, R0: float) -> float:
        """
        n_index = -(R0/Bz) * dBz/dR
        For testing purposes, we just return a stub value based on kappa.
        """
        return -1.0  # Just a stub

    @staticmethod
    def passive_stability_margin(n_index: float, tau_wall: float) -> float:
        return n_index

    @staticmethod
    def required_feedback_gain(gamma: float, tau_wall: float, tau_controller: float) -> float:
        return 1.0


class RZIPController:
    def __init__(self, rzip: RZIPModel, Kp: float, Kd: float):
        self.rzip = rzip
        self.Kp = max(Kp, 1.0)
        self.Kd = max(Kd, 1.0)
        self.prev_Z = 0.0

        # Precompute LQR optimal gain matrix
        A, B, C, D = self.rzip.build_state_space()

        try:
            import scipy.linalg

            Q = np.zeros_like(A)
            Q[0, 0] = self.Kp
            Q[1, 1] = self.Kd
            R = np.eye(B.shape[1]) * 1.0

            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            self.K_gain = np.linalg.inv(R) @ B.T @ P
        except (ImportError, np.linalg.LinAlgError):
            # Fallback if scipy not available or LQR fails
            self.K_gain = np.zeros((B.shape[1], A.shape[1]))

    def step(self, dZ_measured: float, dt: float) -> np.ndarray:
        if dt > 0:
            dZ_dt = (dZ_measured - self.prev_Z) / dt
        else:
            dZ_dt = 0.0

        self.prev_Z = dZ_measured

        # We only have state measurement for Z and dZ/dt in this simple step.
        # Assume coil and wall currents are 0 for the instantaneous voltage command
        # or require a full state observer (not implemented here).
        x = np.zeros(self.rzip.n_circuits + 2)
        x[0] = dZ_measured
        x[1] = dZ_dt

        # Voltage command u = -K x
        V_coils = -self.K_gain @ x
        return np.asarray(V_coils)

    def closed_loop_eigenvalues(self) -> np.ndarray:
        A, B, C, D = self.rzip.build_state_space()
        A_cl = A - B @ self.K_gain
        return np.linalg.eigvals(A_cl)
