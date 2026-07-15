# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Linear-Gaussian Policy
"""Diagonal linear-Gaussian policy with exact policy-gradient terms.

The policy maps an observation ``s`` to a Gaussian action distribution

    a ~ N(mu = W s + b, diag(sigma^2)),   sigma = exp(log_std),

with a state-independent (fixed) exploration standard deviation. Keeping
``log_std`` fixed makes the score function closed-form and deterministic, which
is what the constrained proximal-policy update in
:mod:`scpn_fusion.control.safe_rl_controller` needs to stay auditable and
reproducible under a seeded RNG. The class carries no environment coupling: it
is pure parameters plus the Gaussian score terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class PolicyGradient:
    """Score-function gradient of ``log pi(a|s)`` w.r.t. the mean parameters."""

    grad_w: FloatArray
    grad_b: FloatArray


class LinearGaussianPolicy:
    """Diagonal Gaussian policy that is linear in the observation.

    :param obs_dim: Dimension of the observation vector ``s``.
    :param act_dim: Dimension of the action vector ``a``.
    :param log_std: Initial log standard deviation applied to every action
        component (fixed during optimisation).
    """

    def __init__(self, obs_dim: int, act_dim: int, *, log_std: float = -0.5) -> None:
        if obs_dim < 1 or act_dim < 1:
            raise ValueError("obs_dim and act_dim must both be >= 1")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.w: FloatArray = np.zeros((self.act_dim, self.obs_dim), dtype=np.float64)
        self.b: FloatArray = np.zeros(self.act_dim, dtype=np.float64)
        self.log_std: FloatArray = np.full(self.act_dim, float(log_std), dtype=np.float64)

    @property
    def std(self) -> FloatArray:
        """Return the per-component exploration standard deviation."""
        return np.asarray(np.exp(self.log_std), dtype=np.float64)

    def mean(self, obs: FloatArray) -> FloatArray:
        """Return the deterministic policy mean ``mu = W s + b``."""
        s = np.asarray(obs, dtype=np.float64).ravel()
        if s.shape != (self.obs_dim,):
            raise ValueError(f"obs must have shape ({self.obs_dim},), got {s.shape}")
        return self.w @ s + self.b

    def sample(self, obs: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Draw a stochastic action ``a ~ N(mu, diag(sigma^2))``."""
        mu = self.mean(obs)
        noise = rng.standard_normal(self.act_dim)
        return mu + self.std * noise

    def log_prob(self, obs: FloatArray, action: FloatArray) -> float:
        """Return ``log pi(a|s)`` for a diagonal Gaussian."""
        mu = self.mean(obs)
        a = np.asarray(action, dtype=np.float64).ravel()
        if a.shape != (self.act_dim,):
            raise ValueError(f"action must have shape ({self.act_dim},), got {a.shape}")
        var = self.std**2
        quad = np.sum((a - mu) ** 2 / var)
        norm = np.sum(self.log_std) + 0.5 * self.act_dim * np.log(2.0 * np.pi)
        return float(-0.5 * quad - norm)

    def grad_log_prob(self, obs: FloatArray, action: FloatArray) -> PolicyGradient:
        """Return the score ``d log pi(a|s) / d(W, b)`` at the mean parameters.

        For a diagonal Gaussian with fixed ``sigma`` the mean-parameter score is
        ``d log pi / d mu_j = (a_j - mu_j) / sigma_j^2`` and, since
        ``mu = W s + b``, ``d log pi / d W_jk = ((a_j - mu_j)/sigma_j^2) s_k`` and
        ``d log pi / d b_j = (a_j - mu_j)/sigma_j^2``.
        """
        s = np.asarray(obs, dtype=np.float64).ravel()
        mu = self.mean(s)
        a = np.asarray(action, dtype=np.float64).ravel()
        if a.shape != (self.act_dim,):
            raise ValueError(f"action must have shape ({self.act_dim},), got {a.shape}")
        d_mu = (a - mu) / (self.std**2)
        grad_w = np.asarray(np.outer(d_mu, s), dtype=np.float64)
        grad_b = np.asarray(d_mu, dtype=np.float64)
        return PolicyGradient(grad_w=grad_w, grad_b=grad_b)

    def apply_gradient(self, gradient: PolicyGradient, learning_rate: float) -> None:
        """Ascend the mean parameters by ``learning_rate`` along ``gradient``."""
        self.w = self.w + learning_rate * gradient.grad_w
        self.b = self.b + learning_rate * gradient.grad_b
