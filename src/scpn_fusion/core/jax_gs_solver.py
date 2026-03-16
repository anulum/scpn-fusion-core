# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX-Differentiable Grad-Shafranov Solver
"""JAX-differentiable fixed-boundary Grad-Shafranov equilibrium solver.

Picard iteration with damped Jacobi inner sweeps, all in pure JAX for
JIT compilation and automatic differentiation. Enables ``jax.grad``
through the full equilibrium solve — closing the autodiff depth gap
with TORAX (JAX) and FUSE (Julia AD).

The GS* operator in cylindrical (R, Z) coordinates:

    d²ψ/dR² - (1/R) dψ/dR + d²ψ/dZ² = -μ₀ R J_φ

where J_φ = R p'(ψ_norm) + FF'(ψ_norm) / (μ₀ R).

Key function:
    jax_gs_solve — Fixed-boundary GS equilibrium, differentiable via jax.grad

NumPy fallback provided when JAX is unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    lax = None  # type: ignore[assignment]
    _HAS_JAX = False


def has_jax() -> bool:
    return _HAS_JAX


# ── NumPy fallback ────────────────────────────────────────────────


def _jacobi_gs_step_np(
    psi: NDArray,
    source: NDArray,
    R_interior: NDArray,
    dR: float,
    dZ: float,
    omega_j: float,
) -> NDArray:
    """Single damped Jacobi step for GS* with toroidal 1/R stencil (NumPy)."""
    dR2 = dR * dR
    dZ2 = dZ * dZ
    R_safe = np.maximum(R_interior, 1e-10)
    # GS* east/west: a_E = 1/dR² − 1/(2R·dR), a_W = 1/dR² + 1/(2R·dR)
    a_E = 1.0 / dR2 - 1.0 / (2.0 * R_safe * dR)
    a_W = 1.0 / dR2 + 1.0 / (2.0 * R_safe * dR)
    a_NS = 1.0 / dZ2
    a_C = 2.0 / dR2 + 2.0 / dZ2

    update = (
        a_E * psi[1:-1, 2:]
        + a_W * psi[1:-1, :-2]
        + a_NS * (psi[:-2, 1:-1] + psi[2:, 1:-1])
        - source[1:-1, 1:-1]
    ) / a_C

    psi_new = psi.copy()
    psi_new[1:-1, 1:-1] = (1.0 - omega_j) * psi[1:-1, 1:-1] + omega_j * update
    return psi_new


def _compute_source_np(
    psi: NDArray,
    R_grid: NDArray,
    mu0: float,
    Ip_target: float,
    beta_mix: float,
    dR: float,
    dZ: float,
) -> NDArray:
    """Compute GS RHS = -μ₀ R J_φ from L-mode profiles (NumPy)."""
    psi_axis = np.max(psi[1:-1, 1:-1])
    psi_bdry = 0.0  # Dirichlet ψ=0 on boundary
    denom = psi_bdry - psi_axis
    if abs(denom) < 1e-9:
        denom = np.sign(denom) * 1e-9 if denom != 0 else 1e-9

    psi_norm = (psi - psi_axis) / denom
    psi_norm = np.clip(psi_norm, 0.0, 1.0)
    in_plasma = (psi_norm >= 0.0) & (psi_norm < 1.0)

    profile = np.where(in_plasma, 1.0 - psi_norm, 0.0)

    R_safe = np.maximum(R_grid, 1e-10)
    J_p = R_grid * profile
    J_f = profile / (mu0 * R_safe)
    J_raw = beta_mix * J_p + (1.0 - beta_mix) * J_f

    I_current = np.sum(J_raw) * dR * dZ
    scale = Ip_target / max(abs(I_current), 1e-9)
    J_phi = J_raw * scale

    result: NDArray = -mu0 * R_grid * J_phi
    return result


def gs_solve_np(
    R_min: float,
    R_max: float,
    Z_min: float,
    Z_max: float,
    NR: int,
    NZ: int,
    Ip_target: float,
    mu0: float = 4e-7 * np.pi,
    n_picard: int = 80,
    n_jacobi: int = 200,
    alpha: float = 0.1,
    omega_j: float = 0.667,
    beta_mix: float = 0.5,
) -> NDArray:
    """Fixed-boundary GS solve via Picard iteration (NumPy fallback).

    Returns psi on the (NZ, NR) grid with zero Dirichlet boundary.
    """
    R = np.linspace(R_min, R_max, NR)
    Z = np.linspace(Z_min, Z_max, NZ)
    RR, _ = np.meshgrid(R, Z)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]

    R_interior = RR[1:-1, 1:-1]

    # Gaussian seed
    R_center = 0.5 * (R_min + R_max)
    dist_sq = (RR - R_center) ** 2
    psi = np.exp(-dist_sq / 0.5) * 0.01
    psi[0, :] = 0.0
    psi[-1, :] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0

    for _ in range(n_picard):
        source = _compute_source_np(psi, RR, mu0, Ip_target, beta_mix, dR, dZ)
        psi_elliptic = psi.copy()
        for _ in range(n_jacobi):
            psi_elliptic = _jacobi_gs_step_np(psi_elliptic, source, R_interior, dR, dZ, omega_j)
        psi = (1.0 - alpha) * psi + alpha * psi_elliptic

    result: NDArray = psi
    return result


# ── JAX implementation ────────────────────────────────────────────

if _HAS_JAX:

    @jax.jit
    def _precompute_stencil(
        R_grid: jnp.ndarray,
        dR: float,
        dZ: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, float, float]:
        """Compute GS* stencil coefficients from the R grid.

        Returns (a_E, a_W, a_NS, a_C) where a_E, a_W have shape (NZ-2, NR-2).
        """
        R_int = R_grid[1:-1, 1:-1]
        R_safe = jnp.maximum(R_int, 1e-10)
        dR2 = dR * dR
        dZ2 = dZ * dZ
        a_E = 1.0 / dR2 - 1.0 / (2.0 * R_safe * dR)
        a_W = 1.0 / dR2 + 1.0 / (2.0 * R_safe * dR)
        a_NS = 1.0 / dZ2
        a_C = 2.0 / dR2 + 2.0 / dZ2
        return a_E, a_W, a_NS, a_C

    def _jacobi_gs_step_jax(
        psi: jnp.ndarray,
        source: jnp.ndarray,
        a_E: jnp.ndarray,
        a_W: jnp.ndarray,
        a_NS: float,
        a_C: float,
        omega_j: float,
    ) -> jnp.ndarray:
        """Single damped Jacobi step for GS* (JAX, JIT-compatible)."""
        update = (
            a_E * psi[1:-1, 2:]
            + a_W * psi[1:-1, :-2]
            + a_NS * (psi[:-2, 1:-1] + psi[2:, 1:-1])
            - source[1:-1, 1:-1]
        ) / a_C
        interior = (1.0 - omega_j) * psi[1:-1, 1:-1] + omega_j * update
        return psi.at[1:-1, 1:-1].set(interior)

    def _compute_source_jax(
        psi: jnp.ndarray,
        R_grid: jnp.ndarray,
        mu0: float,
        Ip_target: float,
        beta_mix: float,
        dR: float,
        dZ: float,
    ) -> jnp.ndarray:
        """Compute GS RHS = -μ₀ R J_φ from L-mode linear profiles (JAX)."""
        psi_axis = jnp.max(psi[1:-1, 1:-1])
        psi_bdry = 0.0  # Dirichlet ψ=0 on boundary
        denom = psi_bdry - psi_axis
        denom = jnp.where(jnp.abs(denom) < 1e-9, 1e-9, denom)

        psi_norm = (psi - psi_axis) / denom
        psi_norm = jnp.clip(psi_norm, 0.0, 1.0)
        in_plasma = (psi_norm >= 0.0) & (psi_norm < 1.0)

        profile = jnp.where(in_plasma, 1.0 - psi_norm, 0.0)

        R_safe = jnp.maximum(R_grid, 1e-10)
        J_p = R_grid * profile
        J_f = profile / (mu0 * R_safe)
        J_raw = beta_mix * J_p + (1.0 - beta_mix) * J_f

        I_current = jnp.sum(J_raw) * dR * dZ
        scale = Ip_target / jnp.maximum(jnp.abs(I_current), 1e-9)

        return -mu0 * R_grid * J_raw * scale

    def _jax_gs_solve_impl(
        R_grid: jnp.ndarray,
        dR: float,
        dZ: float,
        psi_init: jnp.ndarray,
        Ip_target: float,
        mu0: float,
        n_picard: int,
        n_jacobi: int,
        alpha: float,
        omega_j: float,
        beta_mix: float,
    ) -> jnp.ndarray:
        """Fixed-boundary GS solve via Picard + Jacobi (JAX, differentiable).

        Uses lax.fori_loop for both inner (Jacobi) and outer (Picard)
        iterations, enabling jax.grad through the full solve.
        """
        a_E, a_W, a_NS, a_C = _precompute_stencil(R_grid, dR, dZ)

        def picard_body(_: int, psi: jnp.ndarray) -> jnp.ndarray:
            source = _compute_source_jax(psi, R_grid, mu0, Ip_target, beta_mix, dR, dZ)

            def inner_body(__: int, psi_j: jnp.ndarray) -> jnp.ndarray:
                return _jacobi_gs_step_jax(psi_j, source, a_E, a_W, a_NS, a_C, omega_j)

            psi_elliptic: jnp.ndarray = lax.fori_loop(0, n_jacobi, inner_body, psi)
            return (1.0 - alpha) * psi + alpha * psi_elliptic

        result: jnp.ndarray = lax.fori_loop(0, n_picard, picard_body, psi_init)
        return result


# ── Public API ────────────────────────────────────────────────────


def jax_gs_solve(
    R_min: float = 0.1,
    R_max: float = 2.0,
    Z_min: float = -1.5,
    Z_max: float = 1.5,
    NR: int = 33,
    NZ: int = 33,
    Ip_target: float = 1e6,
    mu0: float = 4e-7 * np.pi,
    n_picard: int = 80,
    n_jacobi: int = 200,
    alpha: float = 0.1,
    omega_j: float = 0.667,
    beta_mix: float = 0.5,
    *,
    use_jax: bool = True,
) -> NDArray:
    """Fixed-boundary Grad-Shafranov equilibrium solve.

    Picard iteration with damped Jacobi inner sweeps. When JAX is
    available, the solve is JIT-compiled and differentiable via
    ``jax.grad``. Falls back to NumPy otherwise.

    Parameters
    ----------
    R_min, R_max : radial domain [m]
    Z_min, Z_max : vertical domain [m]
    NR, NZ : grid resolution
    Ip_target : target plasma current [A]
    mu0 : vacuum permeability [H/m]
    n_picard : outer Picard iterations (fixed count for differentiability)
    n_jacobi : inner Jacobi sweeps per Picard step
    alpha : Picard under-relaxation factor
    omega_j : Jacobi damping (2/3 standard, Hackbusch 1985 §4.3)
    beta_mix : pressure vs current profile weight
    use_jax : attempt JAX backend

    Returns
    -------
    psi : (NZ, NR) poloidal flux, zero on boundary
    """
    if use_jax and _HAS_JAX:
        R = jnp.linspace(R_min, R_max, NR)
        Z = jnp.linspace(Z_min, Z_max, NZ)
        RR, _ = jnp.meshgrid(R, Z)
        dR = float(R[1] - R[0])
        dZ = float(Z[1] - Z[0])

        R_center = 0.5 * (R_min + R_max)
        dist_sq = (RR - R_center) ** 2
        psi_init = jnp.exp(-dist_sq / 0.5) * 0.01
        psi_init = psi_init.at[0, :].set(0.0)
        psi_init = psi_init.at[-1, :].set(0.0)
        psi_init = psi_init.at[:, 0].set(0.0)
        psi_init = psi_init.at[:, -1].set(0.0)

        psi_jax = _jax_gs_solve_impl(
            RR,
            dR,
            dZ,
            psi_init,
            Ip_target,
            mu0,
            n_picard,
            n_jacobi,
            alpha,
            omega_j,
            beta_mix,
        )
        return np.asarray(psi_jax)

    return gs_solve_np(
        R_min,
        R_max,
        Z_min,
        Z_max,
        NR,
        NZ,
        Ip_target,
        mu0,
        n_picard,
        n_jacobi,
        alpha,
        omega_j,
        beta_mix,
    )


def jax_gs_solve_from_grid(
    R_grid: NDArray,
    psi_init: NDArray,
    dR: float,
    dZ: float,
    Ip_target: float = 1e6,
    mu0: float = 4e-7 * np.pi,
    n_picard: int = 80,
    n_jacobi: int = 200,
    alpha: float = 0.1,
    omega_j: float = 0.667,
    beta_mix: float = 0.5,
) -> NDArray:
    """GS solve on a pre-existing grid. JAX-only, differentiable.

    Parameters
    ----------
    R_grid : (NZ, NR) meshgrid of major radius values
    psi_init : (NZ, NR) initial flux with boundary conditions applied
    dR, dZ : grid spacing
    Ip_target : target plasma current [A]
    """
    if not _HAS_JAX:
        raise RuntimeError("JAX required for jax_gs_solve_from_grid")

    psi_jax = _jax_gs_solve_impl(
        jnp.asarray(R_grid, dtype=jnp.float64),
        float(dR),
        float(dZ),
        jnp.asarray(psi_init, dtype=jnp.float64),
        float(Ip_target),
        float(mu0),
        n_picard,
        n_jacobi,
        float(alpha),
        float(omega_j),
        float(beta_mix),
    )
    return np.asarray(psi_jax)


def jax_gs_grad_Ip(
    Ip_target: float,
    R_min: float = 0.1,
    R_max: float = 2.0,
    Z_min: float = -1.5,
    Z_max: float = 1.5,
    NR: int = 33,
    NZ: int = 33,
    mu0: float = 4e-7 * np.pi,
    n_picard: int = 40,
    n_jacobi: int = 100,
    alpha: float = 0.1,
    omega_j: float = 0.667,
    beta_mix: float = 0.5,
) -> float:
    """Compute d(sum(psi))/d(Ip_target) via JAX autodiff.

    Demonstrates full-stack differentiability through the GS solve.
    """
    if not _HAS_JAX:
        raise RuntimeError("JAX required for gradient computation")

    R = jnp.linspace(R_min, R_max, NR)
    Z = jnp.linspace(Z_min, Z_max, NZ)
    RR, _ = jnp.meshgrid(R, Z)
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])

    R_center = 0.5 * (R_min + R_max)
    dist_sq = (RR - R_center) ** 2
    psi_init = jnp.exp(-dist_sq / 0.5) * 0.01
    psi_init = psi_init.at[0, :].set(0.0)
    psi_init = psi_init.at[-1, :].set(0.0)
    psi_init = psi_init.at[:, 0].set(0.0)
    psi_init = psi_init.at[:, -1].set(0.0)

    def objective(Ip: Any) -> Any:
        psi = _jax_gs_solve_impl(
            RR,
            dR,
            dZ,
            psi_init,
            Ip,
            mu0,
            n_picard,
            n_jacobi,
            alpha,
            omega_j,
            beta_mix,
        )
        return jnp.sum(psi)

    return float(jax.grad(objective)(jnp.float64(Ip_target)))
