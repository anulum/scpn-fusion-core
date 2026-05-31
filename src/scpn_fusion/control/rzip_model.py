# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RZIP Rigid Plasma Response Model
"""Rigid-plasma vertical response model with passive vessel coupling.

The implementation is intentionally compact and deterministic for contract tests
that exercise state-space build, stability metrics, and controller gain flows.
"""

from __future__ import annotations


import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.vessel_model import VesselElement, VesselModel

MU_0 = 4.0 * np.pi * 1e-7
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


class RZIPModel:
    """Rigid-plasma response model coupled to vessel and active-coil circuits."""

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
        """Initialize rigid-plasma, vessel, and active-coil circuit parameters."""
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

    def build_state_space(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Build continuous-time state-space matrices for vertical motion.

        Returns:
            A, B, C, D matrices with order
            ``x = [Z, dZ/dt, I_1, ..., I_n]`` and output ``y = Z``.

        Raises:
            numpy.linalg.LinAlgError: Implicitly handled to fall back to a stable
                zero inverse matrix when circuit coupling is ill-conditioned.
        """
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
        """Return the maximum real eigenvalue of the open-loop model.

        Returns:
            float: Open-loop vertical growth rate (s^-1).
        """
        A, _, _, _ = self.build_state_space()
        eigvals = np.linalg.eigvals(A)
        return float(np.max(np.real(eigvals)))

    def vertical_growth_time(self) -> float:
        """Return the open-loop vertical growth time in milliseconds.

        Returns:
            float: ``inf`` for non-growing or damped cases.
        """
        gamma = self.vertical_growth_rate()
        if gamma <= 0.0:
            return float("inf")
        return 1000.0 / gamma  # in ms

    def stability_margin(self) -> float:
        """Return the signed vertical stability margin proxy.

        The proxy is currently aligned to the model ``n_index`` value to keep a
        transparent interpretation in lightweight sanity checks.

        Returns:
            float: Stability margin surrogate.
        """
        # Distance from marginality (n_index = 0 typically)
        return float(self.n_index)


class VerticalStabilityAnalysis:
    """Utility formulas for vertical stability index and feedback sizing."""

    @staticmethod
    def compute_n_index(psi: FloatArray, R: FloatArray, Z: FloatArray, R0: float) -> float:
        """Compute the local vertical stability index on the mid-plane.

        Args:
            psi: Flux function grid sampled over ``(len(Z), len(R))``.
            R: Strictly increasing major-radius coordinates.
            Z: Vertical coordinates.
            R0: Magnetic-axis major radius.

        Returns:
            Signed vertical stability index based on on-axis ``dBz/dR``.
        """
        psi_arr = np.asarray(psi, dtype=float)
        r_arr = np.asarray(R, dtype=float)
        z_arr = np.asarray(Z, dtype=float)

        if psi_arr.ndim != 2:
            raise ValueError("psi must be a 2D array with shape (len(Z), len(R)).")
        if r_arr.ndim != 1 or z_arr.ndim != 1:
            raise ValueError("R and Z must be 1D coordinate arrays.")
        if psi_arr.shape != (z_arr.size, r_arr.size):
            raise ValueError("psi shape must equal (len(Z), len(R)).")
        if r_arr.size < 3:
            raise ValueError("R must contain at least 3 points for gradients.")
        if z_arr.size < 1:
            raise ValueError("Z must not be empty.")
        if (
            not np.all(np.isfinite(psi_arr))
            or not np.all(np.isfinite(r_arr))
            or not np.all(np.isfinite(z_arr))
        ):
            raise ValueError("psi, R, and Z must be finite.")
        if np.any(r_arr <= 0.0):
            raise ValueError("R coordinates must be strictly positive.")
        if not np.all(np.diff(r_arr) > 0.0):
            raise ValueError("R must be strictly increasing.")

        z0_idx = int(np.argmin(np.abs(z_arr)))
        psi_mid = psi_arr[z0_idx, :]
        dpsi_dR = np.gradient(psi_mid, r_arr, edge_order=2)
        bz_profile = dpsi_dR / r_arr

        dbz_dR = np.gradient(bz_profile, r_arr, edge_order=2)
        r0_idx = int(np.argmin(np.abs(r_arr - float(R0))))
        bz_local = float(bz_profile[r0_idx])
        dbz_local = float(dbz_dR[r0_idx])

        if not np.isfinite(bz_local) or not np.isfinite(dbz_local):
            raise ValueError("Computed Bz profile is non-finite near R0.")
        eps = 1e-12
        if abs(bz_local) < eps:
            raise ValueError("Bz near R0 is too small for a stable n-index estimate.")

        return float(-(float(R0) / bz_local) * dbz_local)

    @staticmethod
    def passive_stability_margin(n_index: float, tau_wall: float) -> float:
        """Return the passive stability margin for a wall time constant.

        Args:
            n_index: Vertical stability index.
            tau_wall: Resistive wall time constant.

        Returns:
            The same ``n_index`` in the current lightweight contract.
        """
        return n_index

    @staticmethod
    def required_feedback_gain(gamma: float, tau_wall: float, tau_controller: float) -> float:
        """Return a minimum loop-gain proxy for vertical stabilization.

        Uses an additive lag rule:
        ``g_min = gamma * (tau_wall + tau_controller)``.

        Args:
            gamma: Open-loop growth rate in s^-1.
            tau_wall: Resistive wall time constant in seconds.
            tau_controller: Controller lag in seconds.

        Returns:
            Minimum dimensionless gain proxy required by the deterministic
            control-contract model.

        Raises:
            ValueError: If any argument is non-finite or non-positive.
        """
        gamma_f = float(gamma)
        tau_wall_f = float(tau_wall)
        tau_ctrl_f = float(tau_controller)

        if not np.isfinite(gamma_f) or not np.isfinite(tau_wall_f) or not np.isfinite(tau_ctrl_f):
            raise ValueError("gamma, tau_wall, and tau_controller must be finite.")
        if gamma_f <= 0.0:
            raise ValueError("gamma must be strictly positive for unstable-mode compensation.")
        if tau_wall_f <= 0.0:
            raise ValueError("tau_wall must be strictly positive.")
        if tau_ctrl_f <= 0.0:
            raise ValueError("tau_controller must be strictly positive.")

        return float(gamma_f * (tau_wall_f + tau_ctrl_f))


class RZIPController:
    """LQR-style voltage controller for RZIP vertical-position states."""

    def __init__(self, rzip: RZIPModel, Kp: float, Kd: float):
        """Initialize feedback weights and precompute the controller gain matrix."""
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

    def step(self, dZ_measured: float, dt: float) -> FloatArray:
        """Compute active-coil voltage commands from a vertical displacement sample.

        Args:
            dZ_measured: Latest measured vertical displacement.
            dt: Sample period in seconds.

        Returns:
            Estimated coil voltages that damp observed displacement and derivative.
        """
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

    def closed_loop_eigenvalues(self) -> ComplexArray:
        """Return eigenvalues of the controller-closed RZIP state matrix.

        Returns:
            Complex eigenvalues used by stability smoke tests.
        """
        A, B, C, D = self.rzip.build_state_space()
        A_cl = A - B @ self.K_gain
        return np.linalg.eigvals(A_cl)
