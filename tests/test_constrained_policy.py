# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Linear-Gaussian Policy Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.constrained_policy import LinearGaussianPolicy, PolicyGradient


def test_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="obs_dim and act_dim"):
        LinearGaussianPolicy(0, 2)
    with pytest.raises(ValueError, match="obs_dim and act_dim"):
        LinearGaussianPolicy(2, 0)


def test_mean_is_affine_in_observation():
    policy = LinearGaussianPolicy(3, 2, log_std=0.0)
    policy.w = np.array([[1.0, 0.0, 2.0], [0.0, -1.0, 0.5]])
    policy.b = np.array([0.5, -0.5])
    obs = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(policy.mean(obs), np.array([1.0 + 6.0 + 0.5, -2.0 + 1.5 - 0.5]))


def test_std_matches_exp_log_std():
    policy = LinearGaussianPolicy(2, 2, log_std=-0.5)
    np.testing.assert_allclose(policy.std, np.exp(np.full(2, -0.5)))


def test_mean_rejects_wrong_shape():
    policy = LinearGaussianPolicy(3, 1)
    with pytest.raises(ValueError, match="obs must have shape"):
        policy.mean(np.array([1.0, 2.0]))


def test_log_prob_rejects_wrong_action_shape():
    policy = LinearGaussianPolicy(2, 2)
    with pytest.raises(ValueError, match="action must have shape"):
        policy.log_prob(np.array([1.0, 2.0]), np.array([0.0]))


def test_grad_log_prob_rejects_wrong_action_shape():
    policy = LinearGaussianPolicy(2, 2)
    with pytest.raises(ValueError, match="action must have shape"):
        policy.grad_log_prob(np.array([1.0, 2.0]), np.array([0.0, 0.0, 0.0]))


def test_sample_is_deterministic_under_seeded_rng():
    policy = LinearGaussianPolicy(2, 2, log_std=-0.3)
    policy.b = np.array([0.2, -0.1])
    a1 = policy.sample(np.array([1.0, 1.0]), np.random.default_rng(7))
    a2 = policy.sample(np.array([1.0, 1.0]), np.random.default_rng(7))
    np.testing.assert_array_equal(a1, a2)
    # A different seed gives a different draw.
    a3 = policy.sample(np.array([1.0, 1.0]), np.random.default_rng(8))
    assert not np.array_equal(a1, a3)


def test_log_prob_matches_scipy_free_gaussian_formula():
    policy = LinearGaussianPolicy(1, 1, log_std=0.0)
    policy.b = np.array([0.0])
    # N(0,1) log-density at a=0 is -0.5*log(2*pi).
    lp = policy.log_prob(np.array([0.0]), np.array([0.0]))
    assert lp == pytest.approx(-0.5 * np.log(2.0 * np.pi))


def test_grad_log_prob_matches_finite_differences():
    rng = np.random.default_rng(11)
    policy = LinearGaussianPolicy(3, 2, log_std=-0.2)
    policy.w = rng.standard_normal((2, 3))
    policy.b = rng.standard_normal(2)
    obs = rng.standard_normal(3)
    action = policy.sample(obs, rng)

    analytic = policy.grad_log_prob(obs, action)

    eps = 1e-6
    numeric_w = np.zeros_like(policy.w)
    for j in range(2):
        for k in range(3):
            base = policy.w[j, k]
            policy.w[j, k] = base + eps
            plus = policy.log_prob(obs, action)
            policy.w[j, k] = base - eps
            minus = policy.log_prob(obs, action)
            policy.w[j, k] = base
            numeric_w[j, k] = (plus - minus) / (2 * eps)
    numeric_b = np.zeros_like(policy.b)
    for j in range(2):
        base = policy.b[j]
        policy.b[j] = base + eps
        plus = policy.log_prob(obs, action)
        policy.b[j] = base - eps
        minus = policy.log_prob(obs, action)
        policy.b[j] = base
        numeric_b[j] = (plus - minus) / (2 * eps)

    np.testing.assert_allclose(analytic.grad_w, numeric_w, atol=1e-6)
    np.testing.assert_allclose(analytic.grad_b, numeric_b, atol=1e-6)


def test_apply_gradient_ascends_parameters():
    policy = LinearGaussianPolicy(2, 1, log_std=0.0)
    grad = PolicyGradient(grad_w=np.ones((1, 2)), grad_b=np.array([2.0]))
    policy.apply_gradient(grad, learning_rate=0.5)
    np.testing.assert_allclose(policy.w, np.full((1, 2), 0.5))
    np.testing.assert_allclose(policy.b, np.array([1.0]))
