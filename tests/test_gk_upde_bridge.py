# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GK → UPDE Bridge Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.gk_interface import GKOutput
from scpn_fusion.phase.gk_upde_bridge import adaptive_knm, gk_natural_frequencies


@pytest.fixture
def k_base():
    return 0.3 * np.ones((8, 8))


@pytest.fixture
def gk_output_unstable():
    return GKOutput(
        chi_i=2.5,
        chi_e=1.8,
        D_e=0.4,
        gamma=np.array([0.1, 0.25, 0.15]),
        omega_r=np.array([-0.3, -0.5, 0.2]),
        k_y=np.array([0.1, 0.3, 0.5]),
        dominant_mode="ITG",
    )


@pytest.fixture
def gk_output_stable():
    return GKOutput(
        chi_i=0.01,
        chi_e=0.01,
        D_e=0.001,
        gamma=np.array([0.0, 0.0]),
        omega_r=np.array([0.0, 0.0]),
        k_y=np.array([0.1, 0.3]),
        dominant_mode="stable",
    )


def test_adaptive_knm_shape(k_base, gk_output_unstable):
    K = adaptive_knm(k_base, gk_output_unstable)
    assert K.shape == (8, 8)


def test_adaptive_knm_symmetric_coupling(k_base, gk_output_unstable):
    K = adaptive_knm(k_base, gk_output_unstable)
    assert K[0, 1] == K[1, 0]
    assert K[1, 4] == K[4, 1]


def test_adaptive_knm_increases_with_gamma(k_base, gk_output_unstable, gk_output_stable):
    K_unstable = adaptive_knm(k_base, gk_output_unstable)
    K_stable = adaptive_knm(k_base, gk_output_stable)
    # P0↔P1 coupling should be stronger with instability
    assert K_unstable[0, 1] >= K_stable[0, 1]


def test_adaptive_knm_pedestal_ratio(k_base, gk_output_unstable):
    chi_i_profile = np.ones(50)
    chi_i_profile[-10:] = 0.1  # low pedestal transport
    K = adaptive_knm(k_base, gk_output_unstable, chi_i_profile=chi_i_profile)
    # P3↔P4 should be modified
    assert K[3, 4] != k_base[3, 4]


def test_adaptive_knm_no_modify_small_matrix():
    K_small = np.eye(4) * 0.3
    gk = GKOutput(
        chi_i=1.0,
        chi_e=0.8,
        D_e=0.1,
        gamma=np.array([0.2]),
        omega_r=np.array([-0.3]),
        k_y=np.array([0.3]),
    )
    K_out = adaptive_knm(K_small, gk)
    np.testing.assert_array_equal(K_out, K_small)


def test_gk_natural_frequencies(gk_output_unstable):
    omega_base = np.ones(8)
    omega = gk_natural_frequencies(omega_base, gk_output_unstable)
    assert omega[0] > omega_base[0]  # layer 0 frequency increased
    np.testing.assert_array_equal(omega[1:], omega_base[1:])  # others unchanged


def test_gk_natural_frequencies_stable(gk_output_stable):
    omega_base = np.ones(8)
    omega = gk_natural_frequencies(omega_base, gk_output_stable)
    assert omega[0] == omega_base[0]  # no growth → no change
