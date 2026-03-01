# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — JAX Transport Solver Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scpn_fusion.core.jax_transport_solver import (
    transport_step_checked,
    transport_step_jax,
    simulate_scenario_jax,
)

def test_jax_transport_step_returns_finite():
    """Verify that a single JAX transport step returns valid finite profiles."""
    nr = 32
    rho = jnp.linspace(0, 1, nr)
    te = jnp.ones(nr) * 5.0
    ti = jnp.ones(nr) * 5.0
    ne = jnp.ones(nr) * 5.0
    chi = jnp.ones(nr) * 1.0
    s = jnp.zeros(nr)
    
    new_te, new_ti = transport_step_jax(te, ti, ne, chi, chi, s, s, rho, 0.01)
    
    assert new_te.shape == (nr,)
    assert new_ti.shape == (nr,)
    assert jnp.all(jnp.isfinite(new_te))
    assert jnp.all(jnp.isfinite(new_ti))
    assert jnp.all(new_te > 0)

def test_jax_transport_differentiability():
    """Verify that we can calculate gradients through the transport step."""
    nr = 32
    rho = jnp.linspace(0, 1, nr)
    te0 = jnp.ones(nr) * 5.0
    ti0 = jnp.ones(nr) * 5.0
    ne = jnp.ones(nr) * 5.0
    chi = jnp.ones(nr) * 1.0
    
    def cost_fn(heating):
        # Scalar heating source
        s = jnp.ones(nr) * heating
        new_te, _ = transport_step_jax(te0, ti0, ne, chi, chi, s, s, rho, 0.01)
        return jnp.mean(new_te)
        
    grad_val = jax.grad(cost_fn)(10.0)
    assert jnp.isfinite(grad_val)
    assert grad_val > 0 # More heating should increase mean temperature

def test_jax_transport_vmap_batching():
    """Verify that the transport step can be batched using vmap."""
    nr = 32
    rho = jnp.linspace(0, 1, nr)
    te = jnp.ones(nr) * 5.0
    ti = jnp.ones(nr) * 5.0
    ne = jnp.ones(nr) * 5.0
    chi = jnp.ones(nr) * 1.0
    
    # Batch of 4 different heating sources
    s_batch = jnp.linspace(0, 10, 4).reshape(4, 1) * jnp.ones((4, nr))
    
    # Vmap over the source S
    vmapped_step = jax.vmap(
        lambda s_val: transport_step_jax(te, ti, ne, chi, chi, s_val, s_val, rho, 0.01)
    )
    
    new_te_batch, new_ti_batch = vmapped_step(s_batch)
    
    assert new_te_batch.shape == (4, nr)
    assert jnp.all(jnp.isfinite(new_te_batch))
    # Higher heating should result in higher temperature
    assert new_te_batch[3, 0] > new_te_batch[0, 0]


def test_transport_step_checked_rejects_non_monotonic_rho():
    nr = 16
    rho = jnp.concatenate([jnp.linspace(0.0, 0.5, nr // 2), jnp.linspace(0.4, 1.0, nr // 2)])
    te = jnp.ones(nr) * 5.0
    ti = jnp.ones(nr) * 5.0
    ne = jnp.ones(nr) * 5.0
    chi = jnp.ones(nr)
    src = jnp.zeros(nr)
    with pytest.raises(ValueError, match="rho"):
        transport_step_checked(te, ti, ne, chi, chi, src, src, rho, 0.01)
