# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Mu-Synthesis Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.mu_synthesis import (
    MuSynthesisController,
    StructuredUncertainty,
    UncertaintyBlock,
    compute_mu_upper_bound,
    dk_iteration,
)


def test_uncertainty_structure():
    b1 = UncertaintyBlock("tau_E", 1, 0.2, "real_scalar")
    b2 = UncertaintyBlock("noise", 2, 0.05, "complex_scalar")
    unc = StructuredUncertainty([b1, b2])

    assert unc.total_size() == 3
    struct = unc.build_Delta_structure()
    assert struct == [(1, "real_scalar"), (2, "complex_scalar")]


def test_uncertainty_structure_rejects_invalid_blocks():
    with np.testing.assert_raises(ValueError):
        StructuredUncertainty([UncertaintyBlock("bad", 0, 0.1, "full")])
    with np.testing.assert_raises(ValueError):
        StructuredUncertainty([UncertaintyBlock("bad", 1, -0.1, "full")])
    with np.testing.assert_raises(ValueError):
        StructuredUncertainty([UncertaintyBlock("bad", 1, 0.1, "diagonal")])


def test_compute_mu_upper_bound():
    # Construct a matrix where standard singular value > mu
    M = np.array([[2.0, 10.0], [0.0, 2.0]], dtype=complex)
    # sigma_max(M) ~ 10.2
    # If Delta is diagonal, D M D^-1 can reduce the off-diagonal

    struct = [(1, "complex_scalar"), (1, "complex_scalar")]

    sigma_max = np.max(np.linalg.svd(M)[1])
    mu_bound = compute_mu_upper_bound(M, struct)

    # mu should be tighter (smaller) than sigma_max
    assert mu_bound < sigma_max


def test_dk_iteration_convergence():
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2))
    plant = (A, B, C, D)

    unc = StructuredUncertainty([UncertaintyBlock("test", 2, 0.1, "full")])

    K, mu, D_s = dk_iteration(plant, unc, n_iter=5)

    assert mu < 1.0  # Regularised synthesis reduces the structured bound.
    assert K is not None
    assert np.all(D_s > 0.0)


def test_mu_controller_robustness():
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2))
    plant = (A, B, C, D)

    unc = StructuredUncertainty([UncertaintyBlock("test", 2, 0.1, "full")])
    ctrl = MuSynthesisController(plant, unc)

    ctrl.synthesize(n_dk_iter=3)

    margin = ctrl.robustness_margin()
    assert margin > 1.0

    x = np.array([1.0, -1.0])
    u = ctrl.step(x, 0.1)

    assert u.shape == (2,)


def test_mu_controller_uses_output_feedback_and_validates_state():
    A = np.eye(2)
    B = np.ones((2, 1))
    C = np.array([[1.0, 0.0]])
    D = np.zeros((1, 1))
    plant = (A, B, C, D)
    unc = StructuredUncertainty([UncertaintyBlock("full_state", 2, 0.1, "full")])
    ctrl = MuSynthesisController(plant, unc)
    ctrl.synthesize(n_dk_iter=2)

    u_first = ctrl.step(np.array([1.0, 100.0]), 0.1)
    ctrl.integral_error = 0.0
    u_second = ctrl.step(np.array([1.0, -100.0]), 0.1)

    assert u_first.shape == (1,)
    assert np.allclose(u_first, u_second, rtol=1e-6, atol=1e-6)
    with np.testing.assert_raises(ValueError):
        ctrl.step(np.array([1.0]), 0.1)
    with np.testing.assert_raises(ValueError):
        ctrl.step(np.array([1.0, 0.0]), 0.0)


def _plant2():
    return (np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2)))


def _unc2():
    return StructuredUncertainty([UncertaintyBlock("u", 2, 0.1, "full")])


def test_compute_mu_upper_bound_requires_square_matrix() -> None:
    with pytest.raises(ValueError, match="M must be a square"):
        compute_mu_upper_bound(np.zeros((2, 3), dtype=complex), [(2, "full")])


def test_compute_mu_upper_bound_requires_matching_structure_size() -> None:
    with pytest.raises(ValueError, match="Delta structure size must match"):
        compute_mu_upper_bound(np.eye(2, dtype=complex), [(3, "full")])


def test_compute_mu_upper_bound_empty_structure_returns_spectral_norm() -> None:
    M = np.array([[2.0, 10.0], [0.0, 2.0]], dtype=complex)
    val = compute_mu_upper_bound(M, [])
    assert val == pytest.approx(float(np.max(np.linalg.svd(M)[1])))


@pytest.mark.parametrize(
    ("plant", "match"),
    [
        ((np.zeros((2, 3)), np.eye(2), np.eye(2), np.zeros((2, 2))), "A must be a square"),
        ((np.eye(2), np.zeros((3, 2)), np.eye(2), np.zeros((2, 2))), "B must have shape"),
        ((np.eye(2), np.eye(2), np.zeros((2, 3)), np.zeros((2, 2))), "C must have shape"),
        ((np.eye(2), np.eye(2), np.eye(2), np.zeros((3, 3))), "D must have shape"),
        ((np.full((2, 2), np.inf), np.eye(2), np.eye(2), np.zeros((2, 2))), "must be finite"),
    ],
)
def test_dk_iteration_rejects_malformed_plant(plant, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        dk_iteration(plant, _unc2(), n_iter=1)


def test_dk_iteration_rejects_non_positive_iteration_count() -> None:
    with pytest.raises(ValueError, match="n_iter must be at least 1"):
        dk_iteration(_plant2(), _unc2(), n_iter=0)


def test_dk_iteration_rejects_non_positive_bisection_tolerance() -> None:
    with pytest.raises(ValueError, match="gamma_bisect_tol must be finite and positive"):
        dk_iteration(_plant2(), _unc2(), n_iter=1, gamma_bisect_tol=0.0)


def test_dk_iteration_rejects_mismatched_uncertainty_size() -> None:
    unc = StructuredUncertainty([UncertaintyBlock("u", 3, 0.1, "full")])  # size 3 != 2 states
    with pytest.raises(ValueError, match="uncertainty total size must match"):
        dk_iteration(_plant2(), unc, n_iter=1)


def test_controller_step_requires_synthesis_first() -> None:
    ctrl = MuSynthesisController(_plant2(), _unc2())
    with pytest.raises(RuntimeError, match="not synthesised"):
        ctrl.step(np.zeros(2), dt=0.01)


def test_controller_step_validates_state_shape() -> None:
    ctrl = MuSynthesisController(_plant2(), _unc2())
    ctrl.synthesize(n_dk_iter=2)
    with pytest.raises(ValueError, match="state vector must have shape"):
        ctrl.step(np.zeros(5), dt=0.01)


def test_robustness_margin_is_infinite_for_non_positive_mu_peak() -> None:
    ctrl = MuSynthesisController(_plant2(), _unc2())
    ctrl.mu_peak = 0.0
    assert ctrl.robustness_margin() == float("inf")


def test_controller_step_rejects_non_finite_state() -> None:
    ctrl = MuSynthesisController(_plant2(), _unc2())
    ctrl.synthesize(n_dk_iter=2)
    with pytest.raises(ValueError, match="state vector must be finite"):
        ctrl.step(np.array([np.nan, 0.0]), dt=0.01)
