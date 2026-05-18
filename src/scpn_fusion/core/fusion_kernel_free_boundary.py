# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Free-boundary and coil-optimisation helpers for ``FusionKernel``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import ellipe, ellipk

from scpn_fusion.core.fusion_kernel_numerics import FloatArray

if TYPE_CHECKING:
    from scpn_fusion.core.fusion_kernel import CoilSet

logger = logging.getLogger(__name__)

_MU0 = 4e-7 * np.pi


def green_function(R_src: float, Z_src: float, R_obs: float, Z_obs: float) -> float:
    """Toroidal Green's function using elliptic integrals (scalar entry point)."""
    denom = (R_obs + R_src) ** 2 + (Z_obs - Z_src) ** 2
    if denom < 1e-30:
        return 0.0
    k2 = 4.0 * R_obs * R_src / denom
    k2 = np.clip(k2, 1e-12, 1.0 - 1e-12)
    k = np.sqrt(k2)
    K_val = ellipk(k2)
    E_val = ellipe(k2)
    prefactor = _MU0 / (2.0 * np.pi) * np.sqrt(R_obs * R_src)
    psi = prefactor * ((2.0 - k2) * K_val - 2.0 * E_val) / k
    return float(psi)


def _green_function_vectorised(
    R_src: float,
    Z_src: float,
    R_obs: FloatArray,
    Z_obs: FloatArray,
) -> FloatArray:
    """Vectorised toroidal Green's function over observation grid arrays."""
    denom = (R_obs + R_src) ** 2 + (Z_obs - Z_src) ** 2
    k2 = np.where(denom > 1e-30, 4.0 * R_obs * R_src / np.maximum(denom, 1e-30), 0.0)
    k2 = np.clip(k2, 1e-12, 1.0 - 1e-12)
    k = np.sqrt(k2)
    K_val = ellipk(k2)
    E_val = ellipe(k2)
    prefactor = _MU0 / (2.0 * np.pi) * np.sqrt(R_obs * R_src)
    return prefactor * ((2.0 - k2) * K_val - 2.0 * E_val) / k


def compute_external_flux(kernel: Any, coils: "CoilSet") -> FloatArray:
    """Sum Green's function contributions on boundary from ``CoilSet``."""
    NR, NZ = len(kernel.R), len(kernel.Z)
    psi_ext = np.zeros((NZ, NR))
    RR, ZZ = np.meshgrid(kernel.R, kernel.Z)
    for idx, (pos, current) in enumerate(zip(coils.positions, coils.currents)):
        R_c, Z_c = pos
        turns = coils.turns[idx] if idx < len(coils.turns) else 1
        I_eff = current * turns
        psi_ext += I_eff * _green_function_vectorised(R_c, Z_c, RR, ZZ)
    return psi_ext


def _as_finite_vector(value: Any, *, name: str, length: int | None = None) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if length is not None and arr.shape != (length,):
        raise ValueError(f"{name} must have length {length}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values only.")
    return arr


def _as_finite_points(value: Any, *, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n_points, 2) with n_points > 0.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values only.")
    return arr


def build_coilset_from_config(kernel: Any) -> "CoilSet":
    """Build a validated :class:`CoilSet` from a FusionKernel configuration."""
    from scpn_fusion.core.fusion_kernel import CoilSet

    coil_cfg = kernel.cfg.get("coils", [])
    if not isinstance(coil_cfg, list) or len(coil_cfg) < 1:
        raise ValueError("coils must contain at least one external coil.")

    positions: list[tuple[float, float]] = []
    currents: list[float] = []
    turns: list[int] = []
    for idx, coil in enumerate(coil_cfg):
        if not isinstance(coil, dict):
            raise ValueError(f"coils[{idx}] must be an object.")
        try:
            r = float(coil["r"])
            z = float(coil["z"])
            current = float(coil.get("current", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"coils[{idx}] must define finite r, z, and current fields.") from exc
        if not np.isfinite(r) or r <= 0.0 or not np.isfinite(z) or not np.isfinite(current):
            raise ValueError(f"coils[{idx}] must define finite r > 0, finite z, and current.")
        raw_turns = coil.get("turns", 1)
        if isinstance(raw_turns, bool):
            raise ValueError(f"coils[{idx}].turns must be a positive integer.")
        try:
            n_turns = int(raw_turns)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"coils[{idx}].turns must be a positive integer.") from exc
        if n_turns < 1 or float(n_turns) != float(raw_turns):
            raise ValueError(f"coils[{idx}].turns must be a positive integer.")
        positions.append((r, z))
        currents.append(current)
        turns.append(n_turns)

    n_coils = len(positions)
    fb_cfg = kernel.cfg.get("free_boundary", {})
    if fb_cfg is None:
        fb_cfg = {}
    if not isinstance(fb_cfg, dict):
        raise ValueError("free_boundary must be an object when provided.")

    current_limits = None
    if "current_limits" in fb_cfg and fb_cfg["current_limits"] is not None:
        current_limits = _as_finite_vector(
            fb_cfg["current_limits"], name="current_limits", length=n_coils
        )
        if np.any(current_limits <= 0.0):
            raise ValueError("current_limits must contain finite positive values only.")

    target_flux_points = None
    if "target_flux_points" in fb_cfg and fb_cfg["target_flux_points"] is not None:
        target_flux_points = _as_finite_points(
            fb_cfg["target_flux_points"], name="target_flux_points"
        )

    target_flux_values = None
    if "target_flux_values" in fb_cfg and fb_cfg["target_flux_values"] is not None:
        if target_flux_points is None:
            raise ValueError("target_flux_points must be set when target_flux_values is provided.")
        target_flux_values = _as_finite_vector(
            fb_cfg["target_flux_values"],
            name="target_flux_values",
            length=int(target_flux_points.shape[0]),
        )

    x_point_target = None
    if "x_point_target" in fb_cfg and fb_cfg["x_point_target"] is not None:
        x_point_target = _as_finite_vector(
            fb_cfg["x_point_target"], name="x_point_target", length=2
        )

    x_point_flux_target = None
    if "x_point_flux_target" in fb_cfg and fb_cfg["x_point_flux_target"] is not None:
        x_point_flux_target = float(fb_cfg["x_point_flux_target"])
        if not np.isfinite(x_point_flux_target):
            raise ValueError("x_point_flux_target must be finite.")

    divertor_strike_points = None
    if "divertor_strike_points" in fb_cfg and fb_cfg["divertor_strike_points"] is not None:
        divertor_strike_points = _as_finite_points(
            fb_cfg["divertor_strike_points"], name="divertor_strike_points"
        )

    divertor_flux_values = None
    if "divertor_flux_values" in fb_cfg and fb_cfg["divertor_flux_values"] is not None:
        if divertor_strike_points is None:
            raise ValueError(
                "divertor_strike_points must be set when divertor_flux_values is provided."
            )
        divertor_flux_values = _as_finite_vector(
            fb_cfg["divertor_flux_values"],
            name="divertor_flux_values",
            length=int(divertor_strike_points.shape[0]),
        )

    return CoilSet(
        positions=positions,
        currents=np.asarray(currents, dtype=np.float64),
        turns=turns,
        current_limits=current_limits,
        target_flux_points=target_flux_points,
        target_flux_values=target_flux_values,
        x_point_target=x_point_target,
        x_point_flux_target=x_point_flux_target,
        divertor_strike_points=divertor_strike_points,
        divertor_flux_values=divertor_flux_values,
    )


def build_mutual_inductance_matrix(
    kernel: Any,
    coils: "CoilSet",
    obs_points: FloatArray,
) -> FloatArray:
    """Build mutual-inductance matrix M[k, p] for coil optimisation."""
    n_coils = len(coils.positions)
    n_pts = obs_points.shape[0]
    R_obs = obs_points[:, 0]
    Z_obs = obs_points[:, 1]
    M = np.zeros((n_coils, n_pts))

    for k, (Rc, Zc) in enumerate(coils.positions):
        turns = coils.turns[k] if k < len(coils.turns) else 1
        M[k, :] = turns * _green_function_vectorised(Rc, Zc, R_obs, Z_obs)

    return M


def sample_flux_at_points(kernel: Any, points: FloatArray) -> FloatArray:
    """Sample kernel flux at validated ``(R, Z)`` points."""
    obs = _as_finite_points(points, name="points")
    return np.asarray([interp_psi(kernel, float(r), float(z)) for r, z in obs], dtype=np.float64)


def _validate_probe_directions(
    directions: list[str] | tuple[str, ...], *, length: int
) -> list[str]:
    if len(directions) != length:
        raise ValueError("b_probe_directions must have one entry per b_probe_point.")
    out = [str(direction).upper() for direction in directions]
    invalid = [direction for direction in out if direction not in {"R", "Z"}]
    if invalid:
        raise ValueError("b_probe_directions entries must be 'R' or 'Z'.")
    return out


def _unit_coil_flux_response(
    r_coil: float,
    z_coil: float,
    turns: int,
    r_obs: float,
    z_obs: float,
) -> float:
    return float(turns) * green_function(r_coil, z_coil, r_obs, z_obs)


def _unit_coil_b_probe_response(
    r_coil: float,
    z_coil: float,
    turns: int,
    r_obs: float,
    z_obs: float,
    direction: str,
) -> float:
    eps_r = max(1.0e-5, 1.0e-5 * abs(r_obs))
    eps_z = max(1.0e-5, 1.0e-5 * (1.0 + abs(z_obs)))
    r_safe = max(float(r_obs), eps_r)
    if direction == "R":
        plus = _unit_coil_flux_response(r_coil, z_coil, turns, r_safe, z_obs + eps_z)
        minus = _unit_coil_flux_response(r_coil, z_coil, turns, r_safe, z_obs - eps_z)
        return float(-(plus - minus) / (2.0 * eps_z * r_safe))
    plus = _unit_coil_flux_response(r_coil, z_coil, turns, r_safe + eps_r, z_obs)
    minus = _unit_coil_flux_response(r_coil, z_coil, turns, r_safe - eps_r, z_obs)
    return float((plus - minus) / (2.0 * eps_r * r_safe))


def build_magnetic_probe_response_matrix(
    kernel: Any,
    coils: "CoilSet",
    *,
    flux_points: FloatArray | None = None,
    b_probe_points: FloatArray | None = None,
    b_probe_directions: list[str] | tuple[str, ...] | None = None,
) -> FloatArray:
    """Build linear coil-current response rows for flux loops and B probes."""
    del kernel
    if len(coils.positions) < 1:
        raise ValueError("CoilSet.positions must contain at least one coil.")
    if len(coils.currents) != len(coils.positions):
        raise ValueError("CoilSet.currents length must match number of coil positions.")

    flux_obs = None if flux_points is None else _as_finite_points(flux_points, name="flux_points")
    b_obs = (
        None if b_probe_points is None else _as_finite_points(b_probe_points, name="b_probe_points")
    )
    if flux_obs is None and b_obs is None:
        raise ValueError("At least one flux point or B probe point must be provided.")
    directions: list[str] = []
    if b_obs is not None:
        if b_probe_directions is None:
            raise ValueError("b_probe_directions must be provided with b_probe_points.")
        directions = _validate_probe_directions(b_probe_directions, length=int(b_obs.shape[0]))

    n_flux = 0 if flux_obs is None else int(flux_obs.shape[0])
    n_b = 0 if b_obs is None else int(b_obs.shape[0])
    n_coils = len(coils.positions)
    response = np.zeros((n_flux + n_b, n_coils), dtype=np.float64)

    for coil_idx, (r_coil, z_coil) in enumerate(coils.positions):
        turns = coils.turns[coil_idx] if coil_idx < len(coils.turns) else 1
        if flux_obs is not None:
            for row_idx, (r_obs, z_obs) in enumerate(flux_obs):
                response[row_idx, coil_idx] = _unit_coil_flux_response(
                    float(r_coil), float(z_coil), int(turns), float(r_obs), float(z_obs)
                )
        if b_obs is not None:
            for probe_idx, (r_obs, z_obs) in enumerate(b_obs):
                response[n_flux + probe_idx, coil_idx] = _unit_coil_b_probe_response(
                    float(r_coil),
                    float(z_coil),
                    int(turns),
                    float(r_obs),
                    float(z_obs),
                    directions[probe_idx],
                )

    if not np.all(np.isfinite(response)):
        raise ValueError("Magnetic probe response matrix contains non-finite entries.")
    return response


def reconstruct_coil_currents_from_magnetic_probes(
    kernel: Any,
    coils: "CoilSet",
    *,
    flux_points: FloatArray | None = None,
    flux_measurements: FloatArray | None = None,
    b_probe_points: FloatArray | None = None,
    b_probe_directions: list[str] | tuple[str, ...] | None = None,
    b_probe_measurements: FloatArray | None = None,
    measurement_sigma: FloatArray | None = None,
    tikhonov_alpha: float = 1.0e-6,
) -> dict[str, Any]:
    """Fit coil currents from magnetic flux-loop and B-probe measurements."""
    response = build_magnetic_probe_response_matrix(
        kernel,
        coils,
        flux_points=flux_points,
        b_probe_points=b_probe_points,
        b_probe_directions=b_probe_directions,
    )

    targets: list[np.ndarray] = []
    if flux_points is not None:
        if flux_measurements is None:
            raise ValueError("flux_measurements must be provided with flux_points.")
        n_flux = _as_finite_points(flux_points, name="flux_points").shape[0]
        targets.append(
            _as_finite_vector(flux_measurements, name="flux_measurements", length=n_flux)
        )
    elif flux_measurements is not None:
        raise ValueError("flux_points must be provided with flux_measurements.")

    if b_probe_points is not None:
        if b_probe_measurements is None:
            raise ValueError("b_probe_measurements must be provided with b_probe_points.")
        n_probe = _as_finite_points(b_probe_points, name="b_probe_points").shape[0]
        targets.append(
            _as_finite_vector(b_probe_measurements, name="b_probe_measurements", length=n_probe)
        )
    elif b_probe_measurements is not None:
        raise ValueError("b_probe_points must be provided with b_probe_measurements.")

    if not targets:
        raise ValueError("At least one measurement vector must be provided.")
    target = np.concatenate(targets).astype(np.float64, copy=False)
    if target.shape != (response.shape[0],):
        raise ValueError("measurement vector length must match response rows.")

    weights = np.ones(response.shape[0], dtype=np.float64)
    if measurement_sigma is not None:
        sigma = _as_finite_vector(
            measurement_sigma, name="measurement_sigma", length=response.shape[0]
        )
        if np.any(sigma <= 0.0):
            raise ValueError("measurement_sigma must contain finite positive values only.")
        weights = 1.0 / sigma

    alpha = float(tikhonov_alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("tikhonov_alpha must be finite and non-negative.")
    n_coils = len(coils.positions)
    weighted_response = response * weights[:, None]
    weighted_target = target * weights
    prior = np.asarray(coils.currents, dtype=np.float64).reshape(-1)
    if prior.shape != (n_coils,) or not np.all(np.isfinite(prior)):
        raise ValueError("CoilSet.currents must be finite with one entry per coil.")

    if alpha > 0.0:
        reg = np.sqrt(alpha) * np.eye(n_coils, dtype=np.float64)
        A = np.vstack([weighted_response, reg])
        b = np.concatenate([weighted_target, np.sqrt(alpha) * prior])
    else:
        A = weighted_response
        b = weighted_target

    if coils.current_limits is not None:
        limits = _as_finite_vector(coils.current_limits, name="current_limits", length=n_coils)
        if np.any(limits <= 0.0):
            raise ValueError("current_limits must contain finite positive values only.")
        lb = -np.abs(limits)
        ub = np.abs(limits)
    else:
        lb = -np.inf * np.ones(n_coils, dtype=np.float64)
        ub = np.inf * np.ones(n_coils, dtype=np.float64)

    from scipy.optimize import lsq_linear

    result = lsq_linear(A, b, bounds=(lb, ub), method="trf")
    if not bool(getattr(result, "success", False)) or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"Magnetic probe inverse reconstruction failed: {getattr(result, 'message', '')}"
        )
    currents = np.asarray(result.x, dtype=np.float64)
    residual = response @ currents - target
    weighted_residual = residual * weights
    return {
        "coil_currents": currents,
        "residual": residual,
        "weighted_residual": weighted_residual,
        "residual_rms": float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0,
        "weighted_residual_rms": (
            float(np.sqrt(np.mean(weighted_residual**2))) if weighted_residual.size else 0.0
        ),
        "response_rank": int(np.linalg.matrix_rank(response)),
        "response_condition": float(np.linalg.cond(response)) if response.size else float("inf"),
        "active_bounds": int(np.count_nonzero(np.isclose(currents, lb) | np.isclose(currents, ub))),
    }


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
        raise ValueError("CoilSet.currents length must match number of coil positions.")
    if not np.isfinite(tikhonov_alpha) or tikhonov_alpha < 0.0:
        raise ValueError("tikhonov_alpha must be finite and non-negative.")

    obs = np.asarray(coils.target_flux_points, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != 2 or obs.shape[0] == 0:
        raise ValueError("target_flux_points must have shape (n_points, 2) with n_points > 0.")
    if not np.all(np.isfinite(obs)):
        raise ValueError("target_flux_points must contain finite values only.")

    target = np.asarray(target_flux, dtype=np.float64).reshape(-1)
    if target.shape[0] != obs.shape[0]:
        raise ValueError("target_flux must have the same length as target_flux_points.")
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
            raise ValueError("current_limits must have one entry per coil.")
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
            raise ValueError("target_flux_values must have the same length as target_flux_points.")
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
            # Route through the kernel method to preserve monkeypatch/test hooks.
            new_currents = kernel.optimize_coil_currents(
                coils,
                target_psi,
                tikhonov_alpha=tikhonov_alpha,
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
