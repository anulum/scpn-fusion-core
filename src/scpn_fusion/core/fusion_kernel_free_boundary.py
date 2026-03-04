"""Free-boundary and coil-optimisation helpers for ``FusionKernel``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from scpn_fusion.core.fusion_kernel_numerics import FloatArray

if TYPE_CHECKING:
    from scpn_fusion.core.fusion_kernel import CoilSet

logger = logging.getLogger(__name__)


def green_function(R_src: float, Z_src: float, R_obs: float, Z_obs: float) -> float:
    """Toroidal Green's function using elliptic integrals."""
    mu0 = 4e-7 * np.pi
    denom = (R_obs + R_src) ** 2 + (Z_obs - Z_src) ** 2
    if denom < 1e-30:
        return 0.0
    k2 = 4.0 * R_obs * R_src / denom
    k2 = np.clip(k2, 1e-12, 1.0 - 1e-12)
    k = np.sqrt(k2)
    from scipy.special import ellipk, ellipe

    K_val = ellipk(k2)
    E_val = ellipe(k2)
    prefactor = mu0 / (2.0 * np.pi) * np.sqrt(R_obs * R_src)
    psi = prefactor * ((2.0 - k2) * K_val - 2.0 * E_val) / k
    return float(psi)


def compute_external_flux(kernel: Any, coils: "CoilSet") -> FloatArray:
    """Sum Green's function contributions on boundary from ``CoilSet``."""
    NR, NZ = len(kernel.R), len(kernel.Z)
    psi_ext = np.zeros((NZ, NR))
    for idx, (pos, current) in enumerate(zip(coils.positions, coils.currents)):
        R_c, Z_c = pos
        turns = coils.turns[idx] if idx < len(coils.turns) else 1
        I_eff = current * turns
        for iz in range(NZ):
            for ir in range(NR):
                psi_ext[iz, ir] += I_eff * green_function(R_c, Z_c, kernel.R[ir], kernel.Z[iz])
    return psi_ext


def build_mutual_inductance_matrix(
    kernel: Any,
    coils: "CoilSet",
    obs_points: FloatArray,
) -> FloatArray:
    """Build mutual-inductance matrix M[k, p] for coil optimisation."""
    n_coils = len(coils.positions)
    n_pts = obs_points.shape[0]
    M = np.zeros((n_coils, n_pts))

    for k, (Rc, Zc) in enumerate(coils.positions):
        turns = coils.turns[k] if k < len(coils.turns) else 1
        for p in range(n_pts):
            R_obs, Z_obs = obs_points[p]
            M[k, p] = turns * green_function(Rc, Zc, R_obs, Z_obs)

    return M


def optimize_coil_currents(
    kernel: Any,
    coils: "CoilSet",
    target_flux: FloatArray,
    tikhonov_alpha: float = 1e-4,
) -> FloatArray:
    """Find coil currents that best reproduce target flux at control points."""
    from scipy.optimize import lsq_linear

    if coils.target_flux_points is None:
        raise ValueError("CoilSet.target_flux_points must be set for optimisation.")
    if len(coils.positions) == 0:
        raise ValueError("CoilSet.positions must contain at least one coil.")
    if len(coils.currents) != len(coils.positions):
        raise ValueError(
            "CoilSet.currents length must match number of coil positions."
        )
    if not np.isfinite(tikhonov_alpha) or tikhonov_alpha < 0.0:
        raise ValueError("tikhonov_alpha must be finite and non-negative.")

    obs = np.asarray(coils.target_flux_points, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != 2 or obs.shape[0] == 0:
        raise ValueError("target_flux_points must have shape (n_points, 2) with n_points > 0.")
    if not np.all(np.isfinite(obs)):
        raise ValueError("target_flux_points must contain finite values only.")

    target = np.asarray(target_flux, dtype=np.float64).reshape(-1)
    if target.shape[0] != obs.shape[0]:
        raise ValueError(
            "target_flux must have the same length as target_flux_points."
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("target_flux must contain finite values only.")

    M = build_mutual_inductance_matrix(kernel, coils, obs)  # (n_coils, n_pts)
    if not np.all(np.isfinite(M)):
        raise ValueError("Mutual inductance matrix contains non-finite entries.")

    # Build augmented system: [M^T; sqrt(alpha)*I] I = [target; 0]
    n_coils = M.shape[0]
    A = np.vstack([M.T, np.sqrt(tikhonov_alpha) * np.eye(n_coils)])
    b = np.concatenate([target, np.zeros(n_coils)])

    # Bounds
    if coils.current_limits is not None:
        limits = np.asarray(coils.current_limits, dtype=np.float64).reshape(-1)
        if limits.shape[0] != n_coils:
            raise ValueError(
                "current_limits must have one entry per coil."
            )
        if not np.all(np.isfinite(limits)):
            raise ValueError("current_limits must contain finite values only.")
        lb = -np.abs(limits)
        ub = np.abs(limits)
    else:
        lb = -np.inf * np.ones(n_coils)
        ub = np.inf * np.ones(n_coils)

    result = lsq_linear(A, b, bounds=(lb, ub), method="trf")
    if not bool(getattr(result, "success", False)) or not np.all(np.isfinite(result.x)):
        logger.warning(
            "Coil optimisation failed (status=%s): %s. Falling back to prior currents.",
            getattr(result, "status", "unknown"),
            getattr(result, "message", "no message"),
        )
        prior_currents = np.asarray(coils.currents, dtype=np.float64).copy()
        return np.clip(prior_currents, lb, ub).astype(np.float64)

    logger.info(
        "Coil optimisation: cost=%.4e, status=%d (%s)",
        result.cost,
        result.status,
        result.message,
    )
    return np.asarray(result.x, dtype=np.float64)


def interp_psi(kernel: Any, R_pt: float, Z_pt: float) -> float:
    """Bilinear interpolation of Psi at an arbitrary (R, Z) point."""
    ir = np.searchsorted(kernel.R, R_pt) - 1
    iz = np.searchsorted(kernel.Z, Z_pt) - 1
    ir = max(0, min(ir, kernel.NR - 2))
    iz = max(0, min(iz, kernel.NZ - 2))

    t_r = (R_pt - kernel.R[ir]) / kernel.dR
    t_z = (Z_pt - kernel.Z[iz]) / kernel.dZ
    t_r = max(0.0, min(1.0, t_r))
    t_z = max(0.0, min(1.0, t_z))

    psi = (
        (1 - t_r) * (1 - t_z) * kernel.Psi[iz, ir]
        + t_r * (1 - t_z) * kernel.Psi[iz, ir + 1]
        + (1 - t_r) * t_z * kernel.Psi[iz + 1, ir]
        + t_r * t_z * kernel.Psi[iz + 1, ir + 1]
    )
    return float(psi)


def resolve_shape_target_flux(kernel: Any, coils: "CoilSet") -> FloatArray:
    """Resolve target flux vector for shape optimisation control points."""
    if coils.target_flux_points is None:
        raise ValueError("CoilSet.target_flux_points must be set for shape optimisation.")

    obs = np.asarray(coils.target_flux_points, dtype=np.float64)
    n_pts = int(obs.shape[0])
    if obs.ndim != 2 or obs.shape[1] != 2 or n_pts == 0:
        raise ValueError("target_flux_points must have shape (n_points, 2) with n_points > 0.")

    if coils.target_flux_values is not None:
        target = np.asarray(coils.target_flux_values, dtype=np.float64).reshape(-1)
        if target.shape[0] != n_pts:
            raise ValueError(
                "target_flux_values must have the same length as target_flux_points."
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("target_flux_values must contain finite values only.")
        return target

    psi_samples = np.array(
        [float(interp_psi(kernel, R_t, Z_t)) for R_t, Z_t in obs], dtype=np.float64
    )
    iso_level = float(np.mean(psi_samples))
    return np.full(n_pts, iso_level, dtype=np.float64)


def solve_free_boundary(
    kernel: Any,
    coils: "CoilSet",
    max_outer_iter: int = 20,
    tol: float = 1e-4,
    optimize_shape: bool = False,
    tikhonov_alpha: float = 1e-4,
) -> dict[str, Any]:
    """Free-boundary GS solve with external coil currents."""
    if max_outer_iter < 1:
        raise ValueError("max_outer_iter must be >= 1.")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and >= 0.")

    psi_ext = compute_external_flux(kernel, coils)
    diff = float("inf")

    for outer in range(max_outer_iter):
        # Apply external flux as boundary condition
        kernel.Psi[0, :] = psi_ext[0, :]
        kernel.Psi[-1, :] = psi_ext[-1, :]
        kernel.Psi[:, 0] = psi_ext[:, 0]
        kernel.Psi[:, -1] = psi_ext[:, -1]

        # Inner GS solve (use existing Picard iteration)
        psi_old = kernel.Psi.copy()
        kernel.solve_equilibrium(
            preserve_initial_state=True,
            boundary_flux=psi_ext,
        )

        # Optional: optimise coil currents to match target shape
        if optimize_shape and coils.target_flux_points is not None:
            target_psi = resolve_shape_target_flux(kernel, coils)
            new_currents = optimize_coil_currents(
                kernel, coils, target_psi, tikhonov_alpha=tikhonov_alpha,
            )
            coils.currents = new_currents
            psi_ext = compute_external_flux(kernel, coils)

        # Check convergence
        diff = float(np.max(np.abs(kernel.Psi - psi_old)))
        if diff < tol:
            logger.info("Free-boundary converged at outer iter %d (diff=%.2e)", outer, diff)
            break

    return {
        "outer_iterations": outer + 1,
        "final_diff": diff,
        "coil_currents": coils.currents.copy(),
    }


__all__ = [
    "build_mutual_inductance_matrix",
    "compute_external_flux",
    "green_function",
    "interp_psi",
    "optimize_coil_currents",
    "resolve_shape_target_flux",
    "solve_free_boundary",
]
