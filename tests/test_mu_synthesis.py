# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Mu-Synthesis Tests
from __future__ import annotations

import numpy as np

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

    assert mu < 1.0  # Simulated convergence
    assert K is not None


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
