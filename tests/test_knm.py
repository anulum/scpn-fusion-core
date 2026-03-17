# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Knm Coupling Matrix Tests
"""
Tests for KnmSpec dataclass and build_knm_paper27 constructor.
test_phase_kuramoto.py covers basic shape/anchors/zeta. This file
targets validation logic, edge cases, and mathematical properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.phase.knm import KnmSpec, OMEGA_N_16, build_knm_paper27


# ── OMEGA_N_16 ───────────────────────────────────────────────────────


class TestOmegaN16:
    def test_length(self):
        assert OMEGA_N_16.shape == (16,)

    def test_all_positive(self):
        assert np.all(OMEGA_N_16 > 0)

    def test_finite(self):
        assert np.all(np.isfinite(OMEGA_N_16))

    def test_dtype(self):
        assert OMEGA_N_16.dtype == np.float64


# ── KnmSpec validation ──────────────────────────────────────────────


class TestKnmSpecValidation:
    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            KnmSpec(K=np.zeros((3, 4)))

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="square"):
            KnmSpec(K=np.zeros(5))

    def test_alpha_wrong_shape(self):
        with pytest.raises(ValueError, match="alpha"):
            KnmSpec(K=np.eye(3), alpha=np.zeros((2, 2)))

    def test_zeta_wrong_shape(self):
        with pytest.raises(ValueError, match="zeta"):
            KnmSpec(K=np.eye(3), zeta=np.zeros(4))

    def test_layer_names_wrong_length(self):
        with pytest.raises(ValueError, match="layer_names"):
            KnmSpec(K=np.eye(3), layer_names=["a", "b"])

    def test_valid_construction(self):
        K = np.eye(3)
        alpha = np.zeros((3, 3))
        zeta = np.ones(3)
        names = ["L1", "L2", "L3"]
        spec = KnmSpec(K=K, alpha=alpha, zeta=zeta, layer_names=names)
        assert spec.L == 3
        assert spec.layer_names == names

    def test_L_property(self):
        spec = KnmSpec(K=np.eye(5))
        assert spec.L == 5

    def test_accepts_list_K(self):
        K = [[1.0, 0.5], [0.5, 1.0]]
        spec = KnmSpec(K=np.array(K))
        assert spec.L == 2

    def test_none_optionals(self):
        spec = KnmSpec(K=np.eye(4))
        assert spec.alpha is None
        assert spec.zeta is None
        assert spec.layer_names is None


# ── build_knm_paper27 ───────────────────────────────────────────────


class TestBuildKnmPaper27:
    def test_default_16_layers(self):
        spec = build_knm_paper27()
        assert spec.K.shape == (16, 16)
        assert spec.L == 16

    def test_custom_L(self):
        spec = build_knm_paper27(L=8)
        assert spec.K.shape == (8, 8)

    def test_L_1(self):
        spec = build_knm_paper27(L=1)
        assert spec.K.shape == (1, 1)
        assert spec.K[0, 0] == pytest.approx(0.45)  # K_base * exp(0)

    def test_diagonal_positive(self):
        spec = build_knm_paper27()
        assert np.all(np.diag(spec.K) > 0)

    def test_off_diagonal_non_negative(self):
        spec = build_knm_paper27()
        assert np.all(spec.K >= 0)

    def test_exponential_decay(self):
        spec = build_knm_paper27(L=16, K_base=0.45, K_alpha=0.3)
        K = spec.K
        # Non-anchor off-diag: K[5,6] should be K_base*exp(-0.3*1)
        # (unless overridden by an anchor or boost)
        expected_base = 0.45 * np.exp(-0.3)
        # K[5,6] is NOT an anchor, check it follows decay
        assert K[5, 6] == pytest.approx(expected_base, rel=0.01)

    def test_calibration_anchors_symmetric(self):
        spec = build_knm_paper27()
        anchors = [(0, 1, 0.302), (1, 2, 0.201), (2, 3, 0.252), (3, 4, 0.154)]
        for i, j, val in anchors:
            assert spec.K[i, j] == pytest.approx(val)
            assert spec.K[j, i] == pytest.approx(val)

    def test_cross_hierarchy_L1_L16(self):
        spec = build_knm_paper27()
        assert spec.K[0, 15] >= 0.05
        assert spec.K[15, 0] >= 0.05

    def test_cross_hierarchy_L5_L7(self):
        spec = build_knm_paper27()
        assert spec.K[4, 6] >= 0.15
        assert spec.K[6, 4] >= 0.15

    def test_no_cross_hierarchy_small_L(self):
        spec = build_knm_paper27(L=4)
        # L<7 and L<16: no cross-hierarchy boosts applied
        assert spec.K.shape == (4, 4)

    def test_zeta_none_by_default(self):
        spec = build_knm_paper27()
        assert spec.zeta is None

    def test_zeta_uniform(self):
        spec = build_knm_paper27(zeta_uniform=0.5)
        assert spec.zeta is not None
        assert spec.zeta.shape == (16,)
        np.testing.assert_allclose(spec.zeta, 0.5)

    def test_zeta_zero_gives_none(self):
        spec = build_knm_paper27(zeta_uniform=0.0)
        assert spec.zeta is None

    def test_custom_K_base(self):
        spec = build_knm_paper27(L=2, K_base=1.0, K_alpha=0.0)
        # With alpha=0 and no anchors overriding K[0,1] for L=2:
        # Actually anchors (0,1,0.302) will override K[0,1]
        assert spec.K[0, 0] == pytest.approx(1.0)
        assert spec.K[0, 1] == pytest.approx(0.302)  # anchor overrides

    def test_custom_K_alpha(self):
        spec = build_knm_paper27(L=16, K_base=0.5, K_alpha=1.0)
        # Steeper decay: distant layers should be weaker
        assert spec.K[0, 10] < spec.K[0, 1]

    def test_matrix_all_finite(self):
        spec = build_knm_paper27()
        assert np.all(np.isfinite(spec.K))
