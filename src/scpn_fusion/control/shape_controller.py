# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Plasma Shape Controller (Jacobian + Tikhonov)
"""Plasma shape-control targets, Jacobians, and Tikhonov feedback laws.

This module hosts deterministic abstractions for feed-forward shape targets,
numerical Jacobian construction, and regularized controller synthesis used by
shape-control smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


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
    gap_errors: FloatArray
    min_gap: float
    xpoint_error: float
    strike_point_errors: FloatArray


class CoilSet:
    """PF coil geometry and current limits."""

    def __init__(self, n_coils: int = 10):
        """Initialize an idealized PF coil set with symmetric current limits."""
        self.n_coils = n_coils
        self.max_currents = np.ones(n_coils) * 50e3


class ShapeJacobian:
    """d(e_shape)/dI_coils sensitivity matrix with deterministic geometry basis."""

    def __init__(self, kernel: Any, coil_set: CoilSet, target: ShapeTarget):
        """Initialize the shape-error sensitivity model for a target and coil set."""
        self.kernel = kernel
        self.coil_set = coil_set
        self.target = target

        self.n_isoflux = len(target.isoflux_points)
        self.n_gaps = len(target.gap_points)
        self.n_xpoint = 2 if target.xpoint_target else 0
        self.n_strike = len(target.strike_point_targets) * 2 if target.strike_point_targets else 0

        self.n_errors = self.n_isoflux + self.n_gaps + self.n_xpoint + self.n_strike

        self.J = self._build_reference_jacobian()
        self._reference_J = np.array(self.J, copy=True)

    def _build_reference_jacobian(self) -> FloatArray:
        """
        Build deterministic, full-rank baseline sensitivities.

        The matrix uses smooth Fourier-like coil basis functions modulated by
        feature groups (isoflux/gap/x-point/strike) so it is reproducible and
        structured for re-linearisation updates.
        """
        n_coils = self.coil_set.n_coils
        n_err = self.n_errors
        if n_coils < 1 or n_err < 1:
            raise ValueError("ShapeJacobian requires at least one coil and one error channel.")

        # Coil phase basis over a notional toroidal/poloidal ordering.
        coil_phase = np.linspace(0.0, 2.0 * np.pi, n_coils, endpoint=False)
        mat = np.zeros((n_err, n_coils), dtype=float)

        row = 0
        for i in range(self.n_isoflux):
            mode = 1 + (i % 4)
            mat[row, :] = (1.0 + 0.08 * i) * np.cos(mode * coil_phase) + 0.35 * np.sin(
                (mode + 1) * coil_phase
            )
            row += 1
        for i in range(self.n_gaps):
            mode = 2 + (i % 3)
            mat[row, :] = (0.9 + 0.06 * i) * np.sin(mode * coil_phase) + 0.25 * np.cos(
                (mode + 1) * coil_phase
            )
            row += 1
        for i in range(self.n_xpoint):
            mode = 3 + i
            mat[row, :] = 1.4 * np.cos(mode * coil_phase) - 0.5 * np.sin((mode + 1) * coil_phase)
            row += 1
        for i in range(self.n_strike):
            mode = 1 + (i % 5)
            mat[row, :] = 0.7 * np.cos(mode * coil_phase) + 0.45 * np.sin((mode + 2) * coil_phase)
            row += 1

        # Ensure deterministic non-degenerate column scaling.
        col_scale = 1.0 + 0.03 * np.arange(n_coils, dtype=float)
        mat *= col_scale[np.newaxis, :]

        # Guarantee full column rank by embedding a direct control basis in the
        # first min(n_err, n_coils) rows.
        for c in range(min(n_err, n_coils)):
            mat[c, c] += 2.0

        # Controller uses small-signal derivatives [shape_error / A].
        return mat * 1e-4

    def compute(self) -> FloatArray:
        """Return the shape Jacobian matrix d(e_shape)/dI_coils."""
        return np.array(self.J, copy=True)

    def update(self, state: dict[str, Any]) -> None:
        """
        Re-linearize Jacobian around a new operating point.

        Supported state paths:

        * ``jacobian``: explicit updated matrix with shape ``(n_errors, n_coils)``.
        * Baseline multiplicative scaling via ``plasma_current_ma``, ``beta_p``,
          ``coil_coupling``, and ``error_coupling``.
        """
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary.")

        if "jacobian" in state:
            j_new = np.asarray(state["jacobian"], dtype=float)
            expected = (self.n_errors, self.coil_set.n_coils)
            if j_new.shape != expected:
                raise ValueError(f"state['jacobian'] must have shape {expected}.")
            if not np.all(np.isfinite(j_new)):
                raise ValueError("state['jacobian'] must be finite.")
            self.J = np.array(j_new, copy=True)
            return

        scale = 1.0
        used_scaling = False

        if "plasma_current_ma" in state:
            ip_ma = float(state["plasma_current_ma"])
            if not np.isfinite(ip_ma) or ip_ma <= 0.0:
                raise ValueError("plasma_current_ma must be finite and > 0.")
            scale *= ip_ma / 15.0
            used_scaling = True

        if "beta_p" in state:
            beta_p = float(state["beta_p"])
            if not np.isfinite(beta_p) or beta_p <= 0.0:
                raise ValueError("beta_p must be finite and > 0.")
            scale *= 1.0 + 0.25 * (beta_p - 1.0)
            used_scaling = True

        j_new = np.array(self._reference_J * scale, copy=True)

        if "coil_coupling" in state:
            coil_coupling = np.asarray(state["coil_coupling"], dtype=float)
            if coil_coupling.shape != (self.coil_set.n_coils,):
                raise ValueError("coil_coupling must have shape (n_coils,).")
            if np.any(~np.isfinite(coil_coupling)) or np.any(coil_coupling <= 0.0):
                raise ValueError("coil_coupling entries must be finite and > 0.")
            j_new *= coil_coupling[np.newaxis, :]
            used_scaling = True

        if "error_coupling" in state:
            error_coupling = np.asarray(state["error_coupling"], dtype=float)
            if error_coupling.shape != (self.n_errors,):
                raise ValueError("error_coupling must have shape (n_errors,).")
            if np.any(~np.isfinite(error_coupling)) or np.any(error_coupling <= 0.0):
                raise ValueError("error_coupling entries must be finite and > 0.")
            j_new *= error_coupling[:, np.newaxis]
            used_scaling = True

        if not used_scaling:
            raise ValueError(
                "state must provide either 'jacobian' or re-linearisation scalars/vectors."
            )

        if np.any(~np.isfinite(j_new)):
            raise ValueError("Re-linearised Jacobian contains non-finite values.")
        self.J = j_new


class PlasmaShapeController:
    """Real-time shape controller using Tikhonov-regularized pseudoinverse."""

    def __init__(self, target: ShapeTarget, coil_set: CoilSet, kernel: Any):
        """Initialize target weighting, Jacobian state, and regularized shape gain."""
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

    def _compute_gain(self) -> FloatArray:
        """K = (J^T W J + lambda I)^-1 J^T W (Tikhonov pseudoinverse)."""
        J = self.jacobian.compute()
        J_T_W = J.T @ self.W
        H = J_T_W @ J + self.lambda_reg * np.eye(self.coil_set.n_coils)
        return np.asarray(np.linalg.inv(H) @ J_T_W)

    def _compute_shape_error(self, psi: FloatArray) -> FloatArray:
        """Evaluate [isoflux; gap; X-point; strike-point] error vector from psi."""
        field = np.asarray(psi, dtype=float)
        if field.ndim != 2:
            raise ValueError("psi must be a 2D array.")
        if not np.all(np.isfinite(field)):
            raise ValueError("psi must contain finite values.")

        nr, nz = field.shape
        if nr < 3 or nz < 3:
            raise ValueError("psi grid must be at least 3x3.")

        # Normalise by peak magnitude to keep error scales stable.
        scale = max(float(np.max(np.abs(field))), 1e-12)
        f = field / scale
        grad_r, grad_z = np.gradient(f)
        h_rr = np.gradient(grad_r, axis=0)
        h_zz = np.gradient(grad_z, axis=1)
        h_rz = np.gradient(grad_r, axis=1)

        e_iso = np.zeros(self.jacobian.n_isoflux, dtype=float)
        e_gap = np.zeros(self.jacobian.n_gaps, dtype=float)
        e_xp = np.zeros(self.jacobian.n_xpoint, dtype=float)
        e_sp = np.zeros(self.jacobian.n_strike, dtype=float)

        # Isoflux target in normalised flux space is zero (LCFS-like level set).
        for i, (r_t, z_t) in enumerate(self.target.isoflux_points):
            ir, iz = self._map_point_to_index(r_t, z_t, nr, nz)
            e_iso[i] = f[ir, iz]

        for i, (r_t, z_t, n_r, n_z) in enumerate(self.target.gap_points):
            ir, iz = self._map_point_to_index(r_t, z_t, nr, nz)
            n_vec = np.array([float(n_r), float(n_z)], dtype=float)
            n_norm = float(np.linalg.norm(n_vec))
            if n_norm <= 1e-12:
                raise ValueError("gap normal vector must be non-zero.")
            n_hat = n_vec / n_norm
            local_grad = np.array([grad_r[ir, iz], grad_z[ir, iz]], dtype=float)
            signed_gap_proxy = float(np.dot(local_grad, n_hat))
            target_gap = (
                float(self.target.gap_targets[i]) if i < len(self.target.gap_targets) else 0.0
            )
            e_gap[i] = signed_gap_proxy - target_gap

        if self.target.xpoint_target and self.jacobian.n_xpoint > 0:
            ir, iz = self._map_point_to_index(
                self.target.xpoint_target[0], self.target.xpoint_target[1], nr, nz
            )
            # Saddle-point proxy: gradient should vanish.
            e_xp[0] = grad_r[ir, iz]
            if self.jacobian.n_xpoint > 1:
                e_xp[1] = grad_z[ir, iz]

            # If gradients are near zero but local curvature is not saddle-like,
            # inject curvature residual to avoid silent false-zero x-point error.
            if float(np.linalg.norm(e_xp)) < 1e-6:
                det_h = h_rr[ir, iz] * h_zz[ir, iz] - h_rz[ir, iz] ** 2
                e_xp[0] = det_h + 1e-3
                if self.jacobian.n_xpoint > 1:
                    e_xp[1] = h_rr[ir, iz] + h_zz[ir, iz]

        if self.target.strike_point_targets and self.jacobian.n_strike > 0:
            for i, (r_t, z_t) in enumerate(self.target.strike_point_targets):
                ir, iz = self._map_point_to_index(r_t, z_t, nr, nz)
                e_sp[2 * i] = grad_r[ir, iz]
                e_sp[2 * i + 1] = grad_z[ir, iz]

        return np.concatenate([e_iso, e_gap, e_xp, e_sp])

    def _map_point_to_index(self, r_t: float, z_t: float, nr: int, nz: int) -> tuple[int, int]:
        """Map physical target coordinates into psi-grid index space deterministically."""
        all_r = [p[0] for p in self.target.isoflux_points] + [p[0] for p in self.target.gap_points]
        all_z = [p[1] for p in self.target.isoflux_points] + [p[1] for p in self.target.gap_points]
        if self.target.xpoint_target:
            all_r.append(self.target.xpoint_target[0])
            all_z.append(self.target.xpoint_target[1])
        if self.target.strike_point_targets:
            all_r.extend(p[0] for p in self.target.strike_point_targets)
            all_z.extend(p[1] for p in self.target.strike_point_targets)

        if not all_r or not all_z:
            return nr // 2, nz // 2

        r_min, r_max = float(min(all_r)), float(max(all_r))
        z_min, z_max = float(min(all_z)), float(max(all_z))
        r_span = max(r_max - r_min, 1e-9)
        z_span = max(z_max - z_min, 1e-9)

        rr = float(np.clip((float(r_t) - r_min) / r_span, 0.0, 1.0))
        zz = float(np.clip((float(z_t) - z_min) / z_span, 0.0, 1.0))
        ir = int(np.clip(round(rr * (nr - 1)), 0, nr - 1))
        iz = int(np.clip(round(zz * (nz - 1)), 0, nz - 1))
        return ir, iz

    def step(self, psi: FloatArray, coil_currents: FloatArray) -> FloatArray:
        """Compute coil current changes to correct shape errors."""
        e_shape = self._compute_shape_error(psi)
        delta_I = -self.K_shape @ e_shape

        max_delta = 1000.0
        delta_I = np.clip(delta_I, -max_delta, max_delta)

        I_next = coil_currents + delta_I
        I_next = np.clip(I_next, -self.coil_set.max_currents, self.coil_set.max_currents)

        return np.asarray(I_next - coil_currents)

    def evaluate_performance(self, psi: FloatArray) -> ShapeControlResult:
        """Decompose the shape error vector into per-category metrics."""
        e_shape = self._compute_shape_error(psi)

        idx1 = self.jacobian.n_isoflux
        idx2 = idx1 + self.jacobian.n_gaps
        idx3 = idx2 + self.jacobian.n_xpoint

        e_iso = e_shape[:idx1]
        e_gap = e_shape[idx1:idx2]
        e_xp = e_shape[idx2:idx3]
        e_sp = e_shape[idx3:]

        min_g: float
        if len(e_gap) > 0:
            # Only inward-closure residuals consume safety margin.
            inward_closure = np.maximum(e_gap, 0.0)
            min_g = np.min(self.target.gap_targets) - np.max(inward_closure)
        else:
            min_g = 0.1

        return ShapeControlResult(
            isoflux_error=float(np.max(np.abs(e_iso))) if len(e_iso) > 0 else 0.0,
            gap_errors=e_gap,
            min_gap=float(min_g),
            xpoint_error=float(np.linalg.norm(e_xp)) if len(e_xp) > 0 else 0.0,
            strike_point_errors=e_sp,
        )


def iter_lower_single_null_target() -> ShapeTarget:
    """Return an ITER-like lower-single-null plasma boundary target."""
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
