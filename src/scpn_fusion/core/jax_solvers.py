# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX-Accelerated Transport Primitives
"""JAX-JIT transport solver primitives with autodiff support.

Provides JAX-traced equivalents of the Thomas tridiagonal solver and
Crank-Nicolson diffusion operator from integrated_transport_solver.py.
When JAX is available, these run on CPU/GPU with automatic differentiation.
Without JAX, NumPy fallbacks are used.

Key functions:
    thomas_solve_jax     — O(n) tridiagonal solver, differentiable via custom_vjp
    diffusion_rhs_jax    — Cylindrical diffusion L_h(T) = (1/r) d/dr(r chi dT/dr)
    crank_nicolson_step  — Single implicit diffusion step (build tridiag + solve)
    batched_transport_step — vmap'd multi-channel transport for ensemble runs

All functions accept and return JAX arrays when JAX is present, or NumPy
arrays otherwise. GPU execution is automatic when jaxlib has CUDA/ROCm.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    lax = None  # type: ignore[assignment]
    _HAS_JAX = False


def has_jax() -> bool:
    """Return whether the optional JAX dependency is available."""
    return _HAS_JAX


def has_jax_gpu() -> bool:
    """Return whether any JAX-visible accelerator device is a GPU."""
    if not _HAS_JAX:
        return False
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


# ── NumPy fallbacks ───────────────────────────────────────────────


def _thomas_solve_np(
    a: FloatArray,
    b: FloatArray,
    c: FloatArray,
    d: FloatArray,
) -> FloatArray:
    """Thomas algorithm (NumPy). Same semantics as TransportSolver._thomas_solve."""
    n = len(d)
    cp = np.empty(n - 1)
    dp = np.empty(n)

    m = b[0]
    if abs(m) < 1e-30:
        m = 1e-30
    cp[0] = c[0] / m
    dp[0] = d[0] / m

    for i in range(1, n):
        m = b[i] - a[i - 1] * cp[i - 1] if i > 0 else b[i]
        if abs(m) < 1e-30:  # pragma: no cover — near-singular pivot guard
            m = 1e-30
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m
        if i < n - 1:
            cp[i] = c[i] / m

    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _diffusion_rhs_np(
    T: FloatArray,
    chi: FloatArray,
    rho: FloatArray,
    drho: float,
) -> FloatArray:
    """Evaluate cylindrical diffusion with a conservative axis control volume."""
    n = len(T)
    Lh = np.zeros(n)
    if rho[0] == 0.0:
        chi_axis_face = 0.5 * (chi[0] + chi[1])
        Lh[0] = 4.0 * chi_axis_face * (T[1] - T[0]) / (drho * drho)
    for i in range(1, n - 1):
        r = rho[i]
        chi_ip = 0.5 * (chi[i] + chi[i + 1])
        chi_im = 0.5 * (chi[i] + chi[i - 1])
        r_ip = r + 0.5 * drho
        r_im = r - 0.5 * drho
        flux_ip = chi_ip * r_ip * (T[i + 1] - T[i]) / drho
        flux_im = chi_im * r_im * (T[i] - T[i - 1]) / drho
        Lh[i] = (flux_ip - flux_im) / (r * drho)
    return Lh


# ── JAX implementations ──────────────────────────────────────────

if _HAS_JAX:

    @jax.jit
    def _thomas_solve_jax_impl(
        a: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
        d: jnp.ndarray,
    ) -> jnp.ndarray:
        """Thomas algorithm via lax.scan (JIT-compiled, GPU-compatible).

        Forward elimination: scan i=0..n-1 building (cp, dp).
        Back substitution: reversed scan building x.
        """
        n = d.shape[0]

        # Forward sweep
        def fwd_step(
            carry: tuple[Any, Any], i: jnp.ndarray
        ) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
            """Forward Thomas sweep step building running cp/dp coefficients."""
            cp_prev, dp_prev = carry
            # Use where to handle i==0 (no previous cp/dp)
            ai = jnp.where(i > 0, a[i - 1], 0.0)
            m = b[i] - ai * cp_prev
            m = jnp.where(jnp.abs(m) < 1e-30, 1e-30, m)
            dp_i = (d[i] - ai * dp_prev) / m
            cp_i = jnp.where(i < n - 1, c[i] / m, 0.0)
            return (cp_i, dp_i), (cp_i, dp_i)

        init = (jnp.float64(0.0), jnp.float64(0.0))
        scan_result: Any = lax.scan(fwd_step, init, jnp.arange(n))
        _, stacked = scan_result
        cp_all: jnp.ndarray = stacked[0]
        dp_all: jnp.ndarray = stacked[1]

        # Back substitution
        def bwd_step(x_next: jnp.ndarray, i: jnp.ndarray) -> tuple[Any, Any]:
            """Backward Thomas sweep step solving x[i] from previously solved x[i+1]."""
            x_i = dp_all[i] - cp_all[i] * x_next
            return x_i, x_i

        bwd_result: Any = lax.scan(bwd_step, dp_all[-1], jnp.arange(n - 2, -1, -1))
        _, x_rev = bwd_result
        x = jnp.concatenate([jnp.flip(x_rev), dp_all[-1:]])
        return x

    @jax.jit
    def _diffusion_rhs_jax_impl(
        T: jnp.ndarray,
        chi: jnp.ndarray,
        rho: jnp.ndarray,
        drho: float,
    ) -> jnp.ndarray:
        """Evaluate cylindrical diffusion with a conservative axis control volume."""
        n = T.shape[0]
        chi_right = 0.5 * (chi[1:-1] + chi[2:])
        chi_left = 0.5 * (chi[1:-1] + chi[:-2])
        r_right = rho[1:-1] + 0.5 * drho
        r_left = rho[1:-1] - 0.5 * drho
        r_center = rho[1:-1]

        flux_r = chi_right * r_right * (T[2:] - T[1:-1]) / drho
        flux_l = chi_left * r_left * (T[1:-1] - T[:-2]) / drho

        Lh_interior = (flux_r - flux_l) / (r_center * drho)
        Lh = jnp.zeros(n)
        chi_axis_face = 0.5 * (chi[0] + chi[1])
        axis_value = 4.0 * chi_axis_face * (T[1] - T[0]) / (drho * drho)
        Lh = Lh.at[0].set(jnp.where(rho[0] == 0.0, axis_value, 0.0))
        Lh = Lh.at[1:-1].set(Lh_interior)
        return Lh

    @jax.jit
    def _cn_step_jax(
        T: jnp.ndarray,
        chi: jnp.ndarray,
        source: jnp.ndarray,
        rho: jnp.ndarray,
        drho: float,
        dt: float,
        T_edge: float,
    ) -> jnp.ndarray:
        """Single Crank-Nicolson implicit diffusion step.

        Solves (I - 0.5*dt*L_h) T^{n+1} = (I + 0.5*dt*L_h) T^n + dt*source
        """
        n = T.shape[0]
        Lh = _diffusion_rhs_jax_impl(T, chi, rho, drho)

        # Build tridiagonal coefficients for interior
        chi_right = 0.5 * (chi[1:-1] + chi[2:])
        chi_left = 0.5 * (chi[1:-1] + chi[:-2])
        r_right = rho[1:-1] + 0.5 * drho
        r_left = rho[1:-1] - 0.5 * drho
        r_center = rho[1:-1]

        coeff_ip = chi_right * r_right / (r_center * drho * drho)
        coeff_im = chi_left * r_left / (r_center * drho * drho)

        # Full diagonals
        b_diag = jnp.ones(n)
        b_diag = b_diag.at[1:-1].set(1.0 + 0.5 * dt * (coeff_ip + coeff_im))
        a_sub = jnp.zeros(n - 1)
        a_sub = a_sub.at[:-1].set(-0.5 * dt * coeff_im)
        c_sup = jnp.zeros(n - 1)
        c_sup = c_sup.at[1:].set(-0.5 * dt * coeff_ip)

        has_axis = rho[0] == 0.0
        axis_coeff = 4.0 * 0.5 * (chi[0] + chi[1]) / (drho * drho)
        b_diag = b_diag.at[0].set(jnp.where(has_axis, 1.0 + 0.5 * dt * axis_coeff, 1.0))
        c_sup = c_sup.at[0].set(jnp.where(has_axis, -0.5 * dt * axis_coeff, 0.0))

        rhs = T + 0.5 * dt * Lh + dt * source
        T_new = _thomas_solve_jax_impl(a_sub, b_diag, c_sup, rhs)

        T_new = T_new.at[0].set(jnp.where(has_axis, T_new[0], T_new[1]))
        T_new = T_new.at[-1].set(T_edge)
        result: jnp.ndarray = T_new
        return result


# ── Public API ────────────────────────────────────────────────────


def crank_nicolson_step_jax(
    T: Any,
    chi: Any,
    source: Any,
    rho: Any,
    drho: Any,
    dt: Any,
    T_edge: Any = 0.1,
) -> Any:
    """Advance one profile through the differentiable JAX CN kernel.

    This is the JAX-native counterpart of :func:`crank_nicolson_step`. It
    retains JAX arrays on the selected device and does not convert the result
    through NumPy, so gradients propagate through temperature, diffusivity,
    source, time step, and edge value.

    Parameters
    ----------
    T : array-like
        Temperature profile of length ``n`` in keV.
    chi : array-like
        Diffusivity profile of length ``n`` in m²/s.
    source : array-like
        Net temperature source profile of length ``n`` in keV/s.
    rho : array-like
        Strictly increasing uniform radial grid.
    drho : scalar
        Uniform radial spacing.
    dt : scalar
        Positive time step in seconds.
    T_edge : scalar, default=0.1
        Prescribed outer-edge temperature in keV.

    Returns
    -------
    jax.Array
        Updated float64 temperature profile on the active JAX device.

    Raises
    ------
    RuntimeError
        If the optional JAX backend is unavailable.

    Notes
    -----
    Callers needing input validation should use
    :func:`scpn_fusion.core.jax_transport_solver.transport_step_checked`.
    """
    if not _HAS_JAX or jnp is None:
        raise RuntimeError("JAX transport requested but JAX is unavailable.")
    return _cn_step_jax(
        jnp.asarray(T, dtype=jnp.float64),
        jnp.asarray(chi, dtype=jnp.float64),
        jnp.asarray(source, dtype=jnp.float64),
        jnp.asarray(rho, dtype=jnp.float64),
        drho,
        dt,
        T_edge,
    )


def thomas_solve(
    a: FloatArray,
    b: FloatArray,
    c: FloatArray,
    d: FloatArray,
    *,
    use_jax: bool = True,
) -> FloatArray:
    """Tridiagonal solve with automatic JAX/GPU dispatch.

    Parameters
    ----------
    a : sub-diagonal, length n-1
    b : main diagonal, length n
    c : super-diagonal, length n-1
    d : right-hand side, length n
    use_jax : attempt JAX backend (falls back to NumPy if unavailable)
    """
    if use_jax and _HAS_JAX:
        return np.asarray(
            _thomas_solve_jax_impl(
                jnp.asarray(a, dtype=jnp.float64),
                jnp.asarray(b, dtype=jnp.float64),
                jnp.asarray(c, dtype=jnp.float64),
                jnp.asarray(d, dtype=jnp.float64),
            )
        )
    return _thomas_solve_np(a, b, c, d)


def diffusion_rhs(
    T: FloatArray,
    chi: FloatArray,
    rho: FloatArray,
    drho: float,
    *,
    use_jax: bool = True,
) -> FloatArray:
    """Evaluate cylindrical diffusion with JAX/GPU dispatch.

    A grid beginning at ``rho=0`` uses the conservative cylindrical
    half-control-volume axis row. A grid beginning inside the domain retains a
    zero explicit inner-boundary row for the caller's boundary condition.
    """
    if use_jax and _HAS_JAX:
        return np.asarray(
            _diffusion_rhs_jax_impl(
                jnp.asarray(T, dtype=jnp.float64),
                jnp.asarray(chi, dtype=jnp.float64),
                jnp.asarray(rho, dtype=jnp.float64),
                float(drho),
            )
        )
    return _diffusion_rhs_np(T, chi, rho, drho)


def crank_nicolson_step(
    T: FloatArray,
    chi: FloatArray,
    source: FloatArray,
    rho: FloatArray,
    drho: float,
    dt: float,
    T_edge: float = 0.1,
    *,
    use_jax: bool = True,
) -> FloatArray:
    """Advance one cylindrical profile by a Crank-Nicolson step.

    Parameters
    ----------
    T : FloatArray
        Temperature profile of length ``n``.
    chi : FloatArray
        Diffusivity profile of length ``n``.
    source : FloatArray
        Net temperature source profile of length ``n``.
    rho : FloatArray
        Strictly increasing radial grid. A first value of zero activates the
        conservative cylindrical axis row.
    drho : float
        Uniform radial spacing.
    dt : float
        Time step.
    T_edge : float, default=0.1
        Prescribed outer-edge temperature in keV.
    use_jax : bool, default=True
        Dispatch through the JAX implementation when available.

    Returns
    -------
    FloatArray
        Updated temperature profile with the outer Dirichlet value applied.
    """
    if use_jax and _HAS_JAX:
        return np.asarray(
            _cn_step_jax(
                jnp.asarray(T, dtype=jnp.float64),
                jnp.asarray(chi, dtype=jnp.float64),
                jnp.asarray(source, dtype=jnp.float64),
                jnp.asarray(rho, dtype=jnp.float64),
                float(drho),
                float(dt),
                float(T_edge),
            )
        )
    # NumPy fallback: explicit diffusion + thomas solve
    Lh = _diffusion_rhs_np(T, chi, rho, drho)
    n = len(T)
    dr = drho
    a_sub = np.zeros(n - 1)
    b_diag = np.ones(n)
    c_sup = np.zeros(n - 1)
    if rho[0] == 0.0:
        axis_coeff = 4.0 * 0.5 * (chi[0] + chi[1]) / (dr * dr)
        b_diag[0] = 1.0 + 0.5 * dt * axis_coeff
        c_sup[0] = -0.5 * dt * axis_coeff
    for i in range(1, n - 1):
        r = rho[i]
        chi_ip = 0.5 * (chi[i] + chi[i + 1])
        chi_im = 0.5 * (chi[i] + chi[i - 1])
        r_ip = r + 0.5 * dr
        r_im = r - 0.5 * dr
        coeff_ip = chi_ip * r_ip / (r * dr * dr)
        coeff_im = chi_im * r_im / (r * dr * dr)
        b_diag[i] = 1.0 + 0.5 * dt * (coeff_ip + coeff_im)
        if i < n - 1:
            c_sup[i] = -0.5 * dt * coeff_ip
        a_sub[i - 1] = -0.5 * dt * coeff_im

    rhs = T + 0.5 * dt * Lh + dt * source
    T_new = _thomas_solve_np(a_sub, b_diag, c_sup, rhs)
    if rho[0] != 0.0:
        T_new[0] = T_new[1]
    T_new[-1] = T_edge
    return T_new


def batched_crank_nicolson(
    T_batch: FloatArray,
    chi: FloatArray,
    source: FloatArray,
    rho: FloatArray,
    drho: float,
    dt: float,
    T_edge: float = 0.1,
) -> FloatArray:
    """Batched transport step via jax.vmap for ensemble/sensitivity runs.

    Parameters
    ----------
    T_batch : (batch, n) initial temperature profiles
    chi, source, rho, drho, dt, T_edge : shared across batch

    Returns
    -------
    T_new : (batch, n) updated profiles
    """
    if not _HAS_JAX:
        return np.stack(
            [
                crank_nicolson_step(T_batch[i], chi, source, rho, drho, dt, T_edge, use_jax=False)
                for i in range(T_batch.shape[0])
            ]
        )

    chi_j = jnp.asarray(chi, dtype=jnp.float64)
    source_j = jnp.asarray(source, dtype=jnp.float64)
    rho_j = jnp.asarray(rho, dtype=jnp.float64)

    @jax.vmap
    def step(T_single: jnp.ndarray) -> jnp.ndarray:
        """Single batched profile step forwarding through JAX Crank-Nicolson kernel."""
        result: jnp.ndarray = _cn_step_jax(
            T_single, chi_j, source_j, rho_j, float(drho), float(dt), float(T_edge)
        )
        return result

    return np.asarray(step(jnp.asarray(T_batch, dtype=jnp.float64)))
