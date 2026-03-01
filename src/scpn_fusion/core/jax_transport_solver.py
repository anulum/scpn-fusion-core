# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — JAX Differentiable Transport
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
JAX-traceable 1.5D transport solver.
Enables jax.grad through the transport evolution step.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap

# ── Validation wrappers ──────────────────────────────────────────────

def _validate_transport_inputs(
    te: jnp.ndarray,
    ti: jnp.ndarray,
    ne: jnp.ndarray,
    chi_e: jnp.ndarray,
    chi_i: jnp.ndarray,
    s_heat_e: jnp.ndarray,
    s_heat_i: jnp.ndarray,
    rho: jnp.ndarray,
    dt: float,
) -> None:
    arrays = {
        "te": te,
        "ti": ti,
        "ne": ne,
        "chi_e": chi_e,
        "chi_i": chi_i,
        "s_heat_e": s_heat_e,
        "s_heat_i": s_heat_i,
        "rho": rho,
    }
    n = int(te.shape[0])
    if n < 3:
        raise ValueError("te must have length >= 3.")
    for name, arr in arrays.items():
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D.")
        if arr.shape[0] != n:
            raise ValueError(f"{name} must have length {n}.")
    rho_np = jnp.asarray(rho)
    if not bool(jnp.all(jnp.isfinite(rho_np))):
        raise ValueError("rho must be finite.")
    if not bool(jnp.all(rho_np[1:] > rho_np[:-1])):
        raise ValueError("rho must be strictly increasing.")
    dt_val = float(dt)
    if not jnp.isfinite(dt_val) or dt_val <= 0.0:
        raise ValueError("dt must be finite and > 0.")

# ── JAX Kernels ──────────────────────────────────────────────────────

@jit
def transport_step_jax(
    te: jnp.ndarray,
    ti: jnp.ndarray,
    ne: jnp.ndarray,
    chi_e: jnp.ndarray,
    chi_i: jnp.ndarray,
    s_heat_e: jnp.ndarray,
    s_heat_i: jnp.ndarray,
    rho: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single differentiable transport step (Explicit finite difference).
    
    dT/dt = 1/n * div(n * chi * grad(T)) + S
    """
    drho = rho[1] - rho[0]
    
    def evolve(T, chi, S):
        # Cylindrical-like divergence in rho: (1/rho) * d/drho (rho * n * chi * dT/drho)
        # Reduced-order 1.5D closure: d/drho (D * dT/drho)
        grad_T = jnp.gradient(T, drho)
        flux = -ne * chi * grad_T
        div_flux = jnp.gradient(flux, drho) / jnp.maximum(ne, 1e-6)
        
        # Numerical Hardening: Prevent NaN propagation
        new_T = T + dt * (-div_flux + S)
        return jnp.where(jnp.isfinite(new_T), new_T, T)

    new_te = evolve(te, chi_e, s_heat_e)
    new_ti = evolve(ti, chi_i, s_heat_i)
    
    # Boundary conditions (Axis symmetry, Edge fixed)
    new_te = new_te.at[0].set(new_te[1])
    new_te = new_te.at[-1].set(0.1)
    new_ti = new_ti.at[0].set(new_ti[1])
    new_ti = new_ti.at[-1].set(0.1)
    
    return jnp.maximum(new_te, 0.01), jnp.maximum(new_ti, 0.01)

@jit
def simulate_scenario_jax(
    initial_te: jnp.ndarray,
    initial_ti: jnp.ndarray,
    ne: jnp.ndarray,
    chi_e: jnp.ndarray,
    chi_i: jnp.ndarray,
    p_aux_mw: jnp.ndarray, # Time-series of heating
    rho: jnp.ndarray,
    dt: float,
):
    """Rollout a transport simulation in JAX."""
    
    def body_fn(carry, p_now):
        te, ti = carry
        # Simple heating model: uniform distribution for now
        s_heat = p_now * 1e6 / (jnp.sum(ne) * 1.6e-16) # Mock scaling
        new_te, new_ti = transport_step_jax(te, ti, ne, chi_e, chi_i, s_heat, s_heat, rho, dt)
        return (new_te, new_ti), (new_te, new_ti)

    (_, _), history = jax.lax.scan(body_fn, (initial_te, initial_ti), p_aux_mw)
    return history


def transport_step_checked(
    te: jnp.ndarray,
    ti: jnp.ndarray,
    ne: jnp.ndarray,
    chi_e: jnp.ndarray,
    chi_i: jnp.ndarray,
    s_heat_e: jnp.ndarray,
    s_heat_i: jnp.ndarray,
    rho: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Validated wrapper around ``transport_step_jax``."""
    _validate_transport_inputs(te, ti, ne, chi_e, chi_i, s_heat_e, s_heat_i, rho, dt)
    return transport_step_jax(te, ti, ne, chi_e, chi_i, s_heat_e, s_heat_i, rho, dt)

if __name__ == "__main__":
    # Simple verification
    nr = 65
    rho = jnp.linspace(0, 1, nr)
    te = jnp.ones(nr) * 5.0
    ti = jnp.ones(nr) * 5.0
    ne = jnp.ones(nr) * 5.0
    chi = jnp.ones(nr) * 1.0
    s = jnp.zeros(nr)
    
    new_te, new_ti = transport_step_jax(te, ti, ne, chi, chi, s, s, rho, 0.01)
    print("JAX Transport Step OK. Core Te:", new_te[0])
    
    # Gradient test
    def cost(p):
        hist_te, _ = simulate_scenario_jax(te, ti, ne, chi, chi, p, rho, 0.01)
        return jnp.mean(hist_te[-1]) # Final average Te

    p_series = jnp.ones(10) * 10.0
    g = jax.grad(cost)(p_series)
    print("Gradient of Final Te wrt Heating calculated. First element:", g[0])
