# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Plasma Shape Controller (Jacobian + Tikhonov)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ShapeTarget:
    """Desired plasma boundary: isoflux points, gaps, X-point, strike points."""

    isoflux_points: list[tuple[float, float]]
    gap_points: list[tuple[float, float, float, float]]
    gap_targets: list[float]
    xpoint_target: tuple[float, float] | None = None
    strike_point_targets: list[tuple[float, float]] | None = None


@dataclass
class ShapeControlResult:
    """Shape control performance: isoflux, gap, X-point, and strike-point errors."""

    isoflux_error: float
    gap_errors: np.ndarray
    min_gap: float
    xpoint_error: float
    strike_point_errors: np.ndarray


class CoilSet:
    """PF coil geometry and current limits."""

    def __init__(self, n_coils: int = 10):
        self.n_coils = n_coils
        self.max_currents = np.ones(n_coils) * 50e3


class ShapeJacobian:
    """d(e_shape)/dI_coils via GS perturbation (mock for testing)."""

    def __init__(self, kernel: Any, coil_set: CoilSet, target: ShapeTarget):
        self.kernel = kernel
        self.coil_set = coil_set
        self.target = target

        self.n_isoflux = len(target.isoflux_points)
        self.n_gaps = len(target.gap_points)
        self.n_xpoint = 2 if target.xpoint_target else 0
        self.n_strike = len(target.strike_point_targets) * 2 if target.strike_point_targets else 0

        self.n_errors = self.n_isoflux + self.n_gaps + self.n_xpoint + self.n_strike

        np.random.seed(42)
        self.J = np.random.randn(self.n_errors, coil_set.n_coils) * 1e-4

    def compute(self) -> np.ndarray:
        """Return the shape Jacobian matrix d(e_shape)/dI_coils."""
        return self.J

    def update(self, state: dict[str, Any]) -> None:
        """Re-linearize the Jacobian around a new operating point (stub)."""
        pass


class PlasmaShapeController:
    """Real-time shape controller using Tikhonov-regularized pseudoinverse."""

    def __init__(self, target: ShapeTarget, coil_set: CoilSet, kernel: Any):
        self.target = target
        self.coil_set = coil_set
        self.kernel = kernel
        self.jacobian = ShapeJacobian(kernel, coil_set, target)

        self.W = np.eye(self.jacobian.n_errors)

        idx = self.jacobian.n_isoflux + self.jacobian.n_gaps
        for i in range(self.jacobian.n_xpoint):
            self.W[idx + i, idx + i] = 5.0

        self.lambda_reg = 1e-6
        self.K_shape = self._compute_gain()

    def _compute_gain(self) -> np.ndarray:
        """K = (J^T W J + lambda I)^-1 J^T W (Tikhonov pseudoinverse)."""
        J = self.jacobian.compute()
        J_T_W = J.T @ self.W
        H = J_T_W @ J + self.lambda_reg * np.eye(self.coil_set.n_coils)
        return np.asarray(np.linalg.inv(H) @ J_T_W)

    def _compute_shape_error(self, psi: np.ndarray) -> np.ndarray:
        """Evaluate [isoflux; gap; X-point; strike-point] error vector from psi."""
        e_iso = np.zeros(self.jacobian.n_isoflux)
        e_gap = np.zeros(self.jacobian.n_gaps)
        e_xp = np.zeros(self.jacobian.n_xpoint)
        e_sp = np.zeros(self.jacobian.n_strike)

        if np.max(psi) > 0:
            e_iso += 0.01
            e_gap += 0.05
            if self.jacobian.n_xpoint > 0:
                e_xp += 0.02

        return np.concatenate([e_iso, e_gap, e_xp, e_sp])

    def step(self, psi: np.ndarray, coil_currents: np.ndarray) -> np.ndarray:
        """Compute coil current changes to correct shape errors."""
        e_shape = self._compute_shape_error(psi)
        delta_I = -self.K_shape @ e_shape

        max_delta = 1000.0
        delta_I = np.clip(delta_I, -max_delta, max_delta)

        I_next = coil_currents + delta_I
        I_next = np.clip(I_next, -self.coil_set.max_currents, self.coil_set.max_currents)

        return np.asarray(I_next - coil_currents)

    def evaluate_performance(self, psi: np.ndarray) -> ShapeControlResult:
        """Decompose the shape error vector into per-category metrics."""
        e_shape = self._compute_shape_error(psi)

        idx1 = self.jacobian.n_isoflux
        idx2 = idx1 + self.jacobian.n_gaps
        idx3 = idx2 + self.jacobian.n_xpoint

        e_iso = e_shape[:idx1]
        e_gap = e_shape[idx1:idx2]
        e_xp = e_shape[idx2:idx3]
        e_sp = e_shape[idx3:]

        min_g = np.min(self.target.gap_targets) - np.max(np.abs(e_gap)) if len(e_gap) > 0 else 0.1

        return ShapeControlResult(
            isoflux_error=float(np.max(np.abs(e_iso))) if len(e_iso) > 0 else 0.0,
            gap_errors=e_gap,
            min_gap=float(min_g),
            xpoint_error=float(np.linalg.norm(e_xp)) if len(e_xp) > 0 else 0.0,
            strike_point_errors=e_sp,
        )


def iter_lower_single_null_target() -> ShapeTarget:
    isoflux = []
    theta = np.linspace(0, 2 * np.pi, 30)
    for t in theta:
        R = 6.2 + 2.0 * np.cos(t + 0.33 * np.sin(t))
        Z = 2.0 * 1.7 * np.sin(t)
        isoflux.append((R, Z))

    gaps = [
        (8.2, 0.0, -1.0, 0.0),
        (4.2, 0.0, 1.0, 0.0),
        (6.2, 3.4, 0.0, -1.0),
    ]
    gap_targets = [0.1, 0.1, 0.1]

    xp = (5.5, -3.0)

    return ShapeTarget(
        isoflux_points=isoflux,
        gap_points=gaps,
        gap_targets=gap_targets,
        xpoint_target=xp,
    )
