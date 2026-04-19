# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Shared runtime helpers for FusionKernel solver mixins."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from scpn_fusion.core.fusion_kernel_numerics import (
    FloatArray,
    stable_rms as _stable_rms,
)

logger = logging.getLogger(__name__)


def compute_gs_residual(kernel: Any, Source: FloatArray) -> FloatArray:
    """Compute the GS residual r = L*[psi] - Source on interior points."""
    Psi = kernel.Psi
    dR2 = kernel.dR**2
    dZ2 = kernel.dZ**2

    residual = np.zeros_like(Psi)
    R_int = kernel.RR[1:-1, 1:-1]
    R_safe = np.maximum(R_int, 1e-10)

    # 5-point toroidal stencil
    d2R = (Psi[1:-1, 2:] - 2.0 * Psi[1:-1, 1:-1] + Psi[1:-1, 0:-2]) / dR2
    d1R = (Psi[1:-1, 2:] - Psi[1:-1, 0:-2]) / (2.0 * kernel.dR)
    d2Z = (Psi[2:, 1:-1] - 2.0 * Psi[1:-1, 1:-1] + Psi[0:-2, 1:-1]) / dZ2

    Lpsi = d2R - d1R / R_safe + d2Z
    residual[1:-1, 1:-1] = Lpsi - Source[1:-1, 1:-1]
    return residual


def compute_gs_residual_rms(kernel: Any, Source: FloatArray) -> float:
    """Return RMS GS residual over interior points."""
    residual = compute_gs_residual(kernel, Source)
    interior = residual[1:-1, 1:-1]
    if interior.size == 0:
        return 0.0
    return _stable_rms(interior)


def apply_gs_operator(kernel: Any, v: FloatArray) -> FloatArray:
    """Apply the discrete GS* operator to array *v*."""
    dR2 = kernel.dR**2
    dZ2 = kernel.dZ**2

    result = np.zeros_like(v)
    R_int = kernel.RR[1:-1, 1:-1]
    R_safe = np.maximum(R_int, 1e-10)

    d2R = (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, 0:-2]) / dR2
    d1R = (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2.0 * kernel.dR)
    d2Z = (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[0:-2, 1:-1]) / dZ2

    result[1:-1, 1:-1] = d2R - d1R / R_safe + d2Z
    return result


def compute_profile_jacobian(
    kernel: Any, Psi_axis: float, Psi_boundary: float, mu0: float
) -> FloatArray:
    """Compute dJ_phi/dpsi as a 2D diagonal scaling field."""
    psi_axis = float(np.nan_to_num(Psi_axis, nan=0.0, posinf=0.0, neginf=0.0))
    psi_boundary = float(
        np.nan_to_num(
            Psi_boundary, nan=psi_axis + 1e-9, posinf=psi_axis + 1e-9, neginf=psi_axis - 1e-9
        )
    )
    denom = psi_boundary - psi_axis
    if not np.isfinite(denom):
        denom = 1e-9
    if abs(denom) < 1e-9:
        denom = 1e-9 if denom >= 0.0 else -1e-9

    psi_raw = np.asarray(kernel.Psi, dtype=np.float64)
    psi_safe = np.nan_to_num(psi_raw, nan=psi_axis, posinf=psi_axis + 1e9, neginf=psi_axis - 1e9)
    rr_raw = np.asarray(kernel.RR, dtype=np.float64)
    rr_safe = np.nan_to_num(rr_raw, nan=0.0, posinf=1e6, neginf=0.0)
    rr_safe = np.clip(rr_safe, 0.0, 1e6)

    Psi_norm = (psi_safe - psi_axis) / denom
    mask_plasma = (Psi_norm >= 0) & (Psi_norm < 1.0)

    # External profile mode injects J_phi from transport/runtime coupling.
    # Use a bounded secant-like diagonal approximation instead of the
    # fixed L-mode linear profile derivative.
    if bool(getattr(kernel, "external_profile_mode", False)):
        dJ_dpsi = np.zeros_like(psi_safe)
        j_abs = np.abs(
            np.nan_to_num(np.asarray(kernel.J_phi, dtype=float), nan=0.0, posinf=1e12, neginf=1e12)
        )
        scale = max(abs(float(denom)), 1e-9)
        dJ_dpsi[mask_plasma] = -j_abs[mask_plasma] / scale
        return np.nan_to_num(dJ_dpsi, nan=0.0, posinf=0.0, neginf=-1e12)

    # For the linear L-mode profile: Source = -mu0 * R * J_phi
    # J_phi = c * (1 - psi_norm) * R  =>  dJ_phi/dpsi_norm = -c * R
    # dJ_phi/dpsi = dJ_phi/dpsi_norm * dpsi_norm/dpsi = -c * R / denom
    I_target = float(
        np.nan_to_num(
            kernel.cfg["physics"]["plasma_current_target"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )
    s = float(np.sum(np.where(mask_plasma, (1 - Psi_norm) * rr_safe, 0.0))) * kernel.dR * kernel.dZ
    if not np.isfinite(s):
        s = 0.0
    c = I_target / max(abs(s), 1e-9)

    dJ_dpsi = np.zeros_like(psi_safe)
    dJ_dpsi[mask_plasma] = -c * rr_safe[mask_plasma] / denom

    return np.nan_to_num(dJ_dpsi, nan=0.0, posinf=1e12, neginf=-1e12)


def solve_via_rust_multigrid(
    kernel: Any,
    preserve_initial_state: bool = False,
    boundary_flux: FloatArray | None = None,
) -> dict[str, Any]:
    """Delegate the full equilibrium solve to the Rust multigrid backend."""
    from scpn_fusion.core._rust_compat import _rust_available, RustAcceleratedKernel

    if preserve_initial_state or boundary_flux is not None:
        logger.warning(
            "Boundary-constrained solve requested with rust_multigrid; falling back to Python SOR."
        )
        prior_method = kernel.cfg["solver"].get("solver_method", "rust_multigrid")
        kernel.cfg["solver"]["solver_method"] = "sor"
        try:
            return kernel.solve_equilibrium(
                preserve_initial_state=preserve_initial_state,
                boundary_flux=boundary_flux,
            )
        finally:
            kernel.cfg["solver"]["solver_method"] = prior_method

    if not _rust_available():
        logger.warning("Rust unavailable; falling back to Python SOR.")
        prior_method = kernel.cfg["solver"].get("solver_method", "rust_multigrid")
        kernel.cfg["solver"]["solver_method"] = "sor"
        try:
            return kernel.solve_equilibrium()
        finally:
            kernel.cfg["solver"]["solver_method"] = prior_method

    t0 = time.time()
    rk = RustAcceleratedKernel(kernel._config_path)
    rk.set_solver_method("multigrid")
    rust_result = rk.solve_equilibrium()

    # Sync state back
    kernel.Psi = rk.Psi
    kernel.J_phi = rk.J_phi
    kernel.B_R = rk.B_R
    kernel.B_Z = rk.B_Z

    mu0: float = kernel.cfg["physics"]["vacuum_permeability"]
    source = -mu0 * kernel.RR * kernel.J_phi
    gs_residual = compute_gs_residual_rms(kernel, source)
    elapsed = time.time() - t0
    solver_tol = float(kernel.cfg.get("solver", {}).get("convergence_threshold", 1e-4))
    practical_tol = max(solver_tol, 2e-3)
    converged = bool(rust_result.converged or rust_result.residual <= practical_tol)
    return {
        "psi": kernel.Psi,
        "converged": converged,
        "iterations": rust_result.iterations,
        "residual": rust_result.residual,
        "residual_history": [],
        "gs_residual": gs_residual,
        "gs_residual_best": gs_residual,
        "gs_residual_history": [],
        "wall_time_s": elapsed,
        "solver_method": "rust_multigrid",
    }


__all__ = [
    "apply_gs_operator",
    "compute_gs_residual",
    "compute_gs_residual_rms",
    "compute_profile_jacobian",
    "solve_newton_linear_system",
    "solve_via_rust_multigrid",
]


def solve_newton_linear_system(
    *,
    kernel: Any,
    J_op: Any,
    rhs: np.ndarray,
    diag_term: FloatArray,
    n_interior: int,
    nz: int,
    nr: int,
    gmres_preconditioner_mode: str,
    ilu_drop_tol: float,
    ilu_fill_factor: float,
    iter_idx: int,
) -> tuple[np.ndarray, int]:
    """Solve ``J * delta = rhs`` with optional GMRES preconditioning."""
    from scipy.sparse.linalg import LinearOperator, gmres

    def _build_diagonal_preconditioner() -> LinearOperator:
        diag_laplacian = -2.0 / (kernel.dR**2) - 2.0 / (kernel.dZ**2)
        diag_jac = diag_laplacian - diag_term[1:-1, 1:-1]
        safe_diag = np.where(
            np.abs(diag_jac) > 1e-12,
            diag_jac,
            np.where(diag_jac >= 0.0, 1e-12, -1e-12),
        )
        inv_diag = (1.0 / safe_diag).ravel()
        return LinearOperator(
            shape=(n_interior, n_interior),
            matvec=lambda x: inv_diag * x,
            dtype=np.float64,
        )

    M_op: Any | None = None
    if gmres_preconditioner_mode == "diagonal":
        M_op = _build_diagonal_preconditioner()
    elif gmres_preconditioner_mode == "ilu":
        try:
            from scipy.sparse import diags, eye, kron
            from scipy.sparse.linalg import spilu

            nz_int = nz - 2
            nr_int = nr - 2
            main_r = np.full(nr_int, -2.0 / (kernel.dR**2))
            off_r = np.full(max(nr_int - 1, 0), 1.0 / (kernel.dR**2))
            lap_r = diags(
                [off_r, main_r, off_r],
                offsets=[-1, 0, 1],
                shape=(nr_int, nr_int),
                format="csc",
            )
            main_z = np.full(nz_int, -2.0 / (kernel.dZ**2))
            off_z = np.full(max(nz_int - 1, 0), 1.0 / (kernel.dZ**2))
            lap_z = diags(
                [off_z, main_z, off_z],
                offsets=[-1, 0, 1],
                shape=(nz_int, nz_int),
                format="csc",
            )
            laplace_approx = kron(eye(nz_int, format="csc"), lap_r, format="csc") + kron(
                lap_z, eye(nr_int, format="csc"), format="csc"
            )
            jac_diag = diags(
                -diag_term[1:-1, 1:-1].ravel(),
                offsets=0,
                format="csc",
            )
            jacobian_approx = laplace_approx + jac_diag
            # Shift to avoid singular preconditioner builds near nullspace.
            jacobian_approx = jacobian_approx + diags(
                np.full(n_interior, 1e-9), offsets=0, format="csc"
            )
            ilu = spilu(
                jacobian_approx,
                drop_tol=max(1e-12, ilu_drop_tol),
                fill_factor=max(1.0, ilu_fill_factor),
            )
            M_op = LinearOperator(
                shape=(n_interior, n_interior),
                matvec=lambda x: ilu.solve(x),
                dtype=np.float64,
            )
        except Exception as exc:
            logger.warning(
                "ILU preconditioner unavailable at Newton iter %d (%s); falling back to diagonal.",
                iter_idx,
                exc,
            )
            M_op = _build_diagonal_preconditioner()

    if M_op is not None:
        delta_flat, info = gmres(
            J_op,
            rhs,
            maxiter=100,
            restart=50,
            atol=1e-8,
            rtol=1e-6,
            M=M_op,
        )
    else:
        delta_flat, info = gmres(
            J_op,
            rhs,
            maxiter=100,
            restart=50,
            atol=1e-8,
            rtol=1e-6,
        )

    return delta_flat, int(info)
