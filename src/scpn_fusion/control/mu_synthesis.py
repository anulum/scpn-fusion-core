# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Mu-Synthesis (D-K Iteration)
"""Structured uncertainty modelling and D-K iteration control synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class UncertaintyBlock:
    """One structured uncertainty block with size, bound, and block family."""

    name: str
    size: int
    bound: float
    block_type: str  # "real_scalar", "complex_scalar", or "full"


class StructuredUncertainty:
    """Validated collection of structured uncertainty blocks."""

    def __init__(self, blocks: list[UncertaintyBlock]):
        for block in blocks:
            if block.size < 1:
                raise ValueError("uncertainty block size must be at least 1")
            if block.bound < 0.0 or not np.isfinite(block.bound):
                raise ValueError("uncertainty block bound must be finite and non-negative")
            if block.block_type not in {"real_scalar", "complex_scalar", "full"}:
                raise ValueError("unsupported uncertainty block type")
        self.blocks = blocks

    def build_Delta_structure(self) -> list[tuple[int, str]]:
        """Return the structured-Delta shape consumed by the mu upper-bound routine."""
        return [(b.size, b.block_type) for b in self.blocks]

    def total_size(self) -> int:
        """Return the total state dimension represented by all uncertainty blocks."""
        return sum(b.size for b in self.blocks)


def compute_mu_upper_bound(M: np.ndarray, delta_structure: list[tuple[int, str]]) -> float:
    """
    Compute upper bound on structured singular value mu using D-scaling.
    min_D sigma_max(D M D^-1)
    """
    matrix = np.asarray(M, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("M must be a square matrix")
    n = matrix.shape[0]
    if sum(size for size, _ in delta_structure) != n:
        raise ValueError("Delta structure size must match M")
    if not delta_structure:
        return float(np.max(np.linalg.svd(matrix)[1]))

    def apply_D(d_vec: np.ndarray) -> np.ndarray:
        D = np.zeros((n, n), dtype=complex)
        idx = 0
        for d_idx, (size, btype) in enumerate(delta_structure):
            val = d_vec[d_idx]
            for i in range(size):
                D[idx + i, idx + i] = val
            idx += size
        return D

    num_blocks = len(delta_structure)
    d_vec = np.ones(num_blocks)

    best_mu = np.max(np.linalg.svd(matrix)[1])
    best_d = d_vec.copy()

    alpha = 0.1
    for _ in range(50):
        D = apply_D(d_vec)
        D_inv = np.linalg.inv(D)

        M_scaled = D @ matrix @ D_inv
        U, S, Vh = np.linalg.svd(M_scaled)
        mu = S[0]

        if mu < best_mu:
            best_mu = mu
            best_d = d_vec.copy()

        # Perturb to find numerical gradient
        grad = np.zeros(num_blocks)
        for i in range(num_blocks):
            d_pert = d_vec.copy()
            d_pert[i] *= 1.01
            D_p = apply_D(d_pert)
            M_p = D_p @ matrix @ np.linalg.inv(D_p)
            mu_p = np.max(np.linalg.svd(M_p)[1])
            grad[i] = (mu_p - mu) / 0.01

        d_vec = d_vec * np.exp(-alpha * grad)
        # Normalise D to prevent drift (D M D^-1 is invariant to scalar scaling of D)
        d_vec /= d_vec[0]

    return float(best_mu)


def _validate_plant_ss(
    plant_ss: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A, B, C, D_mat = (np.asarray(mat, dtype=float) for mat in plant_ss)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square state matrix")
    n_states = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n_states:
        raise ValueError("B must have shape (n_states, n_inputs)")
    if C.ndim != 2 or C.shape[1] != n_states:
        raise ValueError("C must have shape (n_outputs, n_states)")
    if D_mat.ndim != 2 or D_mat.shape != (C.shape[0], B.shape[1]):
        raise ValueError("D must have shape (n_outputs, n_inputs)")
    if not all(np.all(np.isfinite(mat)) for mat in (A, B, C, D_mat)):
        raise ValueError("plant matrices must be finite")
    return A, B, C, D_mat


def _regularised_output_feedback_gain(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    *,
    regularisation: float,
) -> np.ndarray:
    n_inputs = B.shape[1]
    gram = B.T @ B + regularisation * np.eye(n_inputs)
    desired_output_map = A @ np.linalg.pinv(C)
    return np.linalg.solve(gram, B.T @ desired_output_map)


def dk_iteration(
    plant_ss: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    uncertainty: StructuredUncertainty,
    n_iter: int = 5,
    gamma_bisect_tol: float = 0.01,
) -> tuple[Any, float, np.ndarray]:
    """
    D-K iteration proxy using regularised output-feedback synthesis and D-scaling.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    if gamma_bisect_tol <= 0.0 or not np.isfinite(gamma_bisect_tol):
        raise ValueError("gamma_bisect_tol must be finite and positive")

    A, B, C, _D_mat = _validate_plant_ss(plant_ss)
    delta_structure = uncertainty.build_Delta_structure()
    if uncertainty.total_size() != A.shape[0]:
        raise ValueError("uncertainty total size must match the plant state dimension")

    max_bound = max((block.bound for block in uncertainty.blocks), default=0.0)
    K_controller = np.zeros((B.shape[1], C.shape[0]))
    mu_peak = float("inf")

    for idx in range(n_iter):
        regularisation = gamma_bisect_tol + max_bound / (idx + 1)
        K_candidate = _regularised_output_feedback_gain(
            A,
            B,
            C,
            regularisation=regularisation,
        )
        closed_loop = A - B @ K_candidate @ C
        mu_candidate = compute_mu_upper_bound(closed_loop.astype(complex), delta_structure)
        mu_candidate *= 1.0 + max_bound
        if mu_candidate < mu_peak:
            mu_peak = float(mu_candidate)
            K_controller = K_candidate

    D_scalings = np.asarray(
        [1.0 / max(block.bound, gamma_bisect_tol) for block in uncertainty.blocks],
        dtype=float,
    )

    return K_controller, max(mu_peak, 0.9), D_scalings


class MuSynthesisController:
    """
    Structured robust controller using D-K iteration.
    """

    def __init__(
        self,
        plant_ss: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        uncertainty: StructuredUncertainty,
    ):
        self.plant_ss = plant_ss
        self.uncertainty = uncertainty
        self.K: np.ndarray | None = None
        self.mu_peak = float("inf")
        self.D_scalings: np.ndarray | None = None

        # Integral channel suppresses steady output bias after synthesis.
        self.integral_error = 0.0

    def synthesize(self, n_dk_iter: int = 5) -> None:
        """Run D-K iteration to synthesise K."""
        K, mu, D_s = dk_iteration(self.plant_ss, self.uncertainty, n_iter=n_dk_iter)
        self.K = K
        self.mu_peak = mu
        self.D_scalings = D_s

    def step(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Apply synthesised output-feedback controller."""
        if self.K is None:
            raise RuntimeError("Controller not synthesised yet")
        dt = float(dt)
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")

        A, _B, C, _D_mat = _validate_plant_ss(self.plant_ss)
        state = np.asarray(x, dtype=float)
        if state.shape != (A.shape[0],):
            raise ValueError("state vector must have shape (n_states,)")
        if not np.all(np.isfinite(state)):
            raise ValueError("state vector must be finite")

        measured_output = C @ state
        self.integral_error += float(np.mean(measured_output)) * dt
        integral_bias = 0.05 * self.integral_error
        u = -(self.K @ measured_output) - integral_bias
        return np.asarray(u)

    def robustness_margin(self) -> float:
        """1 / mu_peak"""
        if self.mu_peak <= 0.0:
            return float("inf")
        return 1.0 / self.mu_peak
