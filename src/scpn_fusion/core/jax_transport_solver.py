# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX Differentiable Transport
"""Differentiable and dispatched cylindrical transport surfaces.

The JAX-native functions and the NumPy dispatcher tier share the conservative
cylindrical Crank–Nicolson operator implemented in :mod:`jax_solvers`. The
direct JAX functions retain device arrays for differentiation. The checked
dispatcher surface validates host inputs and selects the fastest registered,
numerically reconciled backend.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.jax_solvers import crank_nicolson_step_jax

FloatArray = NDArray[np.float64]


def _validate_transport_inputs(
    te: FloatArray,
    ti: FloatArray,
    chi_e: FloatArray,
    chi_i: FloatArray,
    s_heat_e: FloatArray,
    s_heat_i: FloatArray,
    rho: FloatArray,
    dt: float,
    t_edge_e: float,
    t_edge_i: float,
) -> None:
    arrays = {
        "te": te,
        "ti": ti,
        "chi_e": chi_e,
        "chi_i": chi_i,
        "s_heat_e": s_heat_e,
        "s_heat_i": s_heat_i,
        "rho": rho,
    }
    if te.ndim != 1 or te.size < 3:
        raise ValueError("te must be one-dimensional with length >= 3.")
    nodes = te.size
    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if array.size != nodes:
            raise ValueError(f"{name} must have length {nodes}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
    if np.any(te <= 0.0) or np.any(ti <= 0.0):
        raise ValueError("temperature profiles must be strictly positive.")
    if np.any(chi_e < 0.0) or np.any(chi_i < 0.0):
        raise ValueError("diffusivity profiles must be non-negative.")
    spacing = np.diff(rho)
    if np.any(spacing <= 0.0):
        raise ValueError("rho must be strictly increasing.")
    if not np.allclose(spacing, spacing[0], rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("rho must be uniformly spaced.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0.")
    if not np.isfinite(t_edge_e) or t_edge_e <= 0.0:
        raise ValueError("t_edge_e must be finite and > 0.")
    if not np.isfinite(t_edge_i) or t_edge_i <= 0.0:
        raise ValueError("t_edge_i must be finite and > 0.")


def transport_step_jax(
    te: jnp.ndarray,
    ti: jnp.ndarray,
    chi_e: jnp.ndarray,
    chi_i: jnp.ndarray,
    s_heat_e: jnp.ndarray,
    s_heat_i: jnp.ndarray,
    rho: jnp.ndarray,
    dt: float,
    *,
    t_edge_e: float = 0.1,
    t_edge_i: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Advance electron and ion temperature with canonical JAX CN solves.

    Parameters
    ----------
    te, ti : jax.Array
        Positive electron and ion temperature profiles in keV.
    chi_e, chi_i : jax.Array
        Electron and ion diffusivity profiles in m²/s.
    s_heat_e, s_heat_i : jax.Array
        Net temperature source profiles in keV/s. Density normalisation belongs
        to source construction, matching the production transport runtime.
    rho : jax.Array
        Uniform, strictly increasing normalised radial grid.
    dt : float
        Positive time step in seconds.
    t_edge_e, t_edge_i : float, default=0.1
        Prescribed electron and ion outer-edge temperatures in keV.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Electron and ion profiles retained on the active JAX device.

    Notes
    -----
    This kernel is traceable and performs no host-side validation. Use
    :func:`transport_step_checked` at untrusted input boundaries.
    """
    spacing = rho[1] - rho[0]
    new_te = crank_nicolson_step_jax(te, chi_e, s_heat_e, rho, spacing, dt, t_edge_e)
    new_ti = crank_nicolson_step_jax(ti, chi_i, s_heat_i, rho, spacing, dt, t_edge_i)
    return cast(jnp.ndarray, new_te), cast(jnp.ndarray, new_ti)


@jax.jit
def simulate_scenario_jax(
    initial_te: jnp.ndarray,
    initial_ti: jnp.ndarray,
    chi_e: jnp.ndarray,
    chi_i: jnp.ndarray,
    source_e_history: jnp.ndarray,
    source_i_history: jnp.ndarray,
    rho: jnp.ndarray,
    dt: float,
    *,
    t_edge_e: float = 0.1,
    t_edge_i: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Roll out explicit source histories through canonical JAX CN steps.

    Parameters
    ----------
    initial_te, initial_ti : jax.Array
        Initial temperature profiles in keV.
    chi_e, chi_i : jax.Array
        Fixed diffusivity profiles in m²/s for this linear rollout.
    source_e_history, source_i_history : jax.Array
        Explicit, already density-normalised sources with shape
        ``(steps, radial_nodes)`` in keV/s.
    rho : jax.Array
        Uniform normalised radial grid.
    dt : float
        Positive time step in seconds.
    t_edge_e, t_edge_i : float, default=0.1
        Prescribed outer-edge temperatures in keV.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Electron and ion histories with shape ``(steps, radial_nodes)``.
    """

    def body_fn(
        state: tuple[jnp.ndarray, jnp.ndarray],
        sources: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
        te, ti = state
        source_e, source_i = sources
        next_state = transport_step_jax(
            te,
            ti,
            chi_e,
            chi_i,
            source_e,
            source_i,
            rho,
            dt,
            t_edge_e=t_edge_e,
            t_edge_i=t_edge_i,
        )
        return next_state, next_state

    _, history = jax.lax.scan(
        body_fn,
        (initial_te, initial_ti),
        (source_e_history, source_i_history),
    )
    return history


def transport_step_checked(
    te: FloatArray,
    ti: FloatArray,
    chi_e: FloatArray,
    chi_i: FloatArray,
    s_heat_e: FloatArray,
    s_heat_i: FloatArray,
    rho: FloatArray,
    dt: float,
    *,
    t_edge_e: float = 0.1,
    t_edge_i: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Validate host inputs, then execute the differentiable JAX kernel.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Electron and ion profiles on the active JAX device.

    Raises
    ------
    ValueError
        If shapes, finiteness, positivity, spacing, or scalar controls violate
        the transport contract.
    """
    _validate_transport_inputs(
        te,
        ti,
        chi_e,
        chi_i,
        s_heat_e,
        s_heat_i,
        rho,
        dt,
        t_edge_e,
        t_edge_i,
    )
    return transport_step_jax(
        jnp.asarray(te, dtype=jnp.float64),
        jnp.asarray(ti, dtype=jnp.float64),
        jnp.asarray(chi_e, dtype=jnp.float64),
        jnp.asarray(chi_i, dtype=jnp.float64),
        jnp.asarray(s_heat_e, dtype=jnp.float64),
        jnp.asarray(s_heat_i, dtype=jnp.float64),
        jnp.asarray(rho, dtype=jnp.float64),
        dt,
        t_edge_e=t_edge_e,
        t_edge_i=t_edge_i,
    )


def simulate_transport_scenario(
    initial_te: FloatArray,
    initial_ti: FloatArray,
    chi_e: FloatArray,
    chi_i: FloatArray,
    source_e_history: FloatArray,
    source_i_history: FloatArray,
    rho: FloatArray,
    dt: float,
    *,
    t_edge_e: float = 0.1,
    t_edge_i: float = 0.1,
    backend: str = "auto",
) -> tuple[FloatArray, FloatArray]:
    """Run a checked source history through the registered runtime tiers.

    ``auto`` uses the evidence-backed per-kernel order. On the retained local
    small-grid cohort this selects NumPy because JAX device transfers dominate.
    ``jax`` is an explicit, fail-closed request for the reconciled JAX/XLA
    provider; it never silently falls back.

    Parameters
    ----------
    initial_te, initial_ti : FloatArray
        Positive initial temperature profiles in keV.
    chi_e, chi_i : FloatArray
        Fixed diffusivity profiles in m²/s for this linear rollout.
    source_e_history, source_i_history : FloatArray
        Explicit, density-normalised source arrays with shape
        ``(steps, radial_nodes)`` in keV/s.
    rho : FloatArray
        Uniform, strictly increasing normalised radial grid.
    dt : float
        Positive time step in seconds.
    t_edge_e, t_edge_i : float, default=0.1
        Prescribed outer-edge temperatures in keV.
    backend : {"auto", "numpy", "jax"}, default="auto"
        Runtime provider selection. Explicit requests do not fall back.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Electron and ion histories with shape ``(steps, radial_nodes)``.

    Raises
    ------
    ValueError
        If any input violates the transport contract.
    RuntimeError
        If an explicitly requested registered backend is unavailable.
    """
    if source_e_history.ndim != 2 or source_e_history.shape[0] == 0:
        raise ValueError("source_e_history must have shape (steps, radial_nodes).")
    if source_i_history.shape != source_e_history.shape:
        raise ValueError("source_i_history must match source_e_history shape.")
    if source_e_history.shape[1] != initial_te.size:
        raise ValueError("source histories must match the radial profile length.")
    if not np.all(np.isfinite(source_e_history)) or not np.all(np.isfinite(source_i_history)):
        raise ValueError("source histories must contain only finite values.")
    _validate_transport_inputs(
        initial_te,
        initial_ti,
        chi_e,
        chi_i,
        source_e_history[0],
        source_i_history[0],
        rho,
        dt,
        t_edge_e,
        t_edge_i,
    )
    from scpn_fusion.core._multi_compat import dispatch, dispatch_for_tier

    backend_key = backend.strip().lower()
    if backend_key == "auto":
        implementation = dispatch("transport_cn_rollout")
    elif backend_key in {"numpy", "jax"}:
        implementation = dispatch_for_tier("transport_cn_rollout", backend_key)
    else:
        raise ValueError("backend must be one of: auto, numpy, jax.")
    result = implementation(
        initial_te,
        initial_ti,
        chi_e,
        chi_i,
        source_e_history,
        source_i_history,
        rho,
        dt,
        t_edge_e=t_edge_e,
        t_edge_i=t_edge_i,
    )
    return cast("tuple[FloatArray, FloatArray]", result)
