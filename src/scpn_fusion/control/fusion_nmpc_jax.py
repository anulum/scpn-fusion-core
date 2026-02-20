# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Nonlinear MPC (JAX/NumPy Hybrid)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Nonlinear Model Predictive Control (NMPC) using JAX for gradient acceleration.
Falls back to SciPy/NumPy if JAX is unavailable.

Implements a Neural ODE surrogate for plasma dynamics:
    dx/dt = f_theta(x, u)
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, Tuple, Optional, Callable

import numpy as np
from numpy.typing import NDArray

# ── JAX Import Shim ──────────────────────────────────────────────────
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    # Fallback types
    jax = None
    jnp = None
    grad = None
    jit = None

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# ── Neural Dynamics Model ────────────────────────────────────────────

class NeuralODEDynamics:
    """
    Simple MLP representing continuous-time plasma dynamics.
    dx/dt = MLP(x, u)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32, seed: int = 42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights (Xavier-ish)
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(state_dim + action_dim)
        
        # Layer 1: [state, action] -> hidden
        self.W1 = rng.standard_normal((hidden_dim, state_dim + action_dim)) * scale
        self.b1 = np.zeros(hidden_dim)
        
        # Layer 2: hidden -> d_state/dt
        self.W2 = rng.standard_normal((state_dim, hidden_dim)) * scale
        self.b2 = np.zeros(state_dim)
        
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward_numpy(self, x: FloatArray, u: FloatArray) -> FloatArray:
        """NumPy forward pass."""
        xu = np.concatenate([x, u])
        h = np.tanh(self.W1 @ xu + self.b1)
        dxdt = self.W2 @ h + self.b2
        return dxdt

    def forward_jax(self, params, x, u):
        """JAX-traceable forward pass."""
        W1, b1, W2, b2 = params
        xu = jnp.concatenate([x, u])
        h = jnp.tanh(jnp.dot(W1, xu) + b1)
        dxdt = jnp.dot(W2, h) + b2
        return dxdt

# ── NMPC Controller ──────────────────────────────────────────────────

class NonlinearMPC:
    def __init__(
        self,
        dynamics: NeuralODEDynamics,
        horizon: int = 10,
        dt: float = 0.1,
        learning_rate: float = 0.01,
        iterations: int = 50,
        l2_reg: float = 0.01
    ):
        self.dynamics = dynamics
        self.horizon = horizon
        self.dt = dt
        self.lr = learning_rate
        self.iterations = iterations
        self.l2_reg = l2_reg
        
        self._jax_grad_fn = None
        self._compile_jax()

    def _compile_jax(self):
        """Compile JAX gradient function if available."""
        if not _HAS_JAX:
            return

        def loss_fn(U_flat, x0, target, params):
            """
            U_flat: (horizon * action_dim) flattened actions
            x0: initial state
            target: target state
            """
            U = U_flat.reshape((self.horizon, self.dynamics.action_dim))
            
            def body_fun(carry, u):
                x, cost = carry
                dxdt = self.dynamics.forward_jax(params, x, u)
                x_next = x + dxdt * self.dt
                # Stage cost: ||x - target||^2 + reg * ||u||^2
                err = x_next - target
                step_cost = jnp.sum(err**2) + self.l2_reg * jnp.sum(u**2)
                return (x_next, cost + step_cost), None

            (_, total_cost), _ = jax.lax.scan(body_fun, (x0, 0.0), U)
            return total_cost

        self._jax_loss = loss_fn
        self._jax_grad = jax.jit(jax.grad(loss_fn))
        logger.info("JAX NMPC: Compiled gradient function.")

    def plan_trajectory(self, x0: FloatArray, target: FloatArray, u_guess: Optional[FloatArray] = None) -> FloatArray:
        """
        Optimize control sequence U = [u_0, ..., u_H-1] to minimize cost.
        Returns: u_0 (optimal action for current step)
        """
        action_dim = self.dynamics.action_dim
        if u_guess is None:
            # Warm start with zeros
            U = np.zeros((self.horizon, action_dim))
        else:
            U = u_guess.copy()

        if _HAS_JAX:
            return self._plan_jax(x0, target, U)
        else:
            return self._plan_numpy(x0, target, U)

    def _plan_jax(self, x0: FloatArray, target: FloatArray, U: FloatArray) -> FloatArray:
        """Gradient descent optimization using JAX."""
        params = self.dynamics.params
        # Convert to JAX arrays
        U_curr = jnp.asarray(U.ravel())
        x0_jax = jnp.asarray(x0)
        target_jax = jnp.asarray(target)
        # Assuming params are already numpy arrays, JAX handles them fine or convert
        params_jax = [jnp.asarray(p) for p in params]

        for _ in range(self.iterations):
            grads = self._jax_grad(U_curr, x0_jax, target_jax, params_jax)
            # Simple GD update
            U_curr = U_curr - self.lr * grads
        
        # Return first action
        best_U = np.array(U_curr).reshape(self.horizon, self.dynamics.action_dim)
        return best_U[0]

    def _plan_numpy(self, x0: FloatArray, target: FloatArray, U: FloatArray) -> FloatArray:
        """Fallback optimization using SciPy L-BFGS-B or finite diff."""
        from scipy.optimize import minimize

        def np_loss(U_flat):
            U_mat = U_flat.reshape(self.horizon, self.dynamics.action_dim)
            x = x0.copy()
            cost = 0.0
            for t in range(self.horizon):
                u = U_mat[t]
                dxdt = self.dynamics.forward_numpy(x, u)
                x = x + dxdt * self.dt
                err = x - target
                cost += np.sum(err**2) + self.l2_reg * np.sum(u**2)
            return cost

        # Use L-BFGS-B for faster convergence than simple GD
        res = minimize(
            np_loss, 
            U.ravel(), 
            method='L-BFGS-B', 
            options={'maxiter': self.iterations, 'disp': False}
        )
        
        best_U = res.x.reshape(self.horizon, self.dynamics.action_dim)
        return best_U[0]

# ── Factory ──────────────────────────────────────────────────────────

def get_nmpc_controller(
    state_dim: int = 4,
    action_dim: int = 4,
    horizon: int = 10
) -> NonlinearMPC:
    """Factory to create a default NMPC instance."""
    dyn = NeuralODEDynamics(state_dim, action_dim)
    return NonlinearMPC(dyn, horizon=horizon)
