# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Extended MHD Stability Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3
# ──────────────────────────────────────────────────────────────────────
"""Tests for stability_mhd_extended (Troyon, NTM, RWM, peeling-ballooning)."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.stability_mhd import QProfile
from scpn_fusion.core.stability_mhd_extended import (
    ntm_stability,
    peeling_ballooning_stability,
    rwm_stability,
    troyon_beta_limit,
)


@pytest.fixture
def iter_qprofile():
    """ITER-like Q-profile for testing."""
    rho = np.linspace(0, 1, 100)
    q = 1.0 + 2.0 * rho**2
    shear = 4.0 * rho**2 / q
    shear[0] = 0.0
    alpha_mhd = 0.1 * np.exp(-((rho - 0.5) ** 2) / 0.05)
    return QProfile(
        rho=rho,
        q=q,
        shear=shear,
        alpha_mhd=alpha_mhd,
        q_min=1.0,
        q_min_rho=0.0,
        q_edge=3.0,
    )


class TestTroyon:
    def test_below_nowall_limit(self):
        r = troyon_beta_limit(beta_t=0.02, Ip_MA=15.0, a=2.0, B0=5.3)
        assert r.stable_nowall

    def test_above_nowall_limit(self):
        r = troyon_beta_limit(beta_t=0.10, Ip_MA=5.0, a=2.0, B0=5.3)
        assert not r.stable_nowall

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            troyon_beta_limit(beta_t=-0.01, Ip_MA=15.0, a=2.0, B0=5.3)
        with pytest.raises(ValueError):
            troyon_beta_limit(beta_t=0.02, Ip_MA=0.0, a=2.0, B0=5.3)


class TestNTM:
    def test_classical_stable(self, iter_qprofile):
        rho = iter_qprofile.rho
        j_bs = 0.1e6 * np.exp(-((rho - 0.5) ** 2) / 0.02)
        j_total = 1.0e6 * (1 - rho**2)
        r = ntm_stability(iter_qprofile, j_bs, j_total, a=2.0)
        assert r.most_unstable_rho is not None  # bootstrap drives NTM

    def test_shape_mismatch_raises(self, iter_qprofile):
        with pytest.raises(ValueError):
            ntm_stability(iter_qprofile, np.zeros(5), np.ones(5), a=2.0)


class TestRWM:
    def test_below_nowall(self):
        r = rwm_stability(beta_N=2.0)
        assert r.stable
        assert r.mode_growth_rate == 0.0

    def test_above_nowall(self):
        r = rwm_stability(beta_N=3.0)
        assert not r.stable
        assert r.mode_growth_rate > 0.0


class TestPeelingBallooning:
    def test_stable_pedestal(self, iter_qprofile):
        r = peeling_ballooning_stability(
            iter_qprofile,
            j_edge=1e3,
            p_ped_Pa=5e3,
            R0=6.2,
            a=2.0,
            B0=5.3,
        )
        assert r.stable

    def test_unstable_high_pressure(self, iter_qprofile):
        r = peeling_ballooning_stability(
            iter_qprofile,
            j_edge=1e6,
            p_ped_Pa=1e6,
            R0=6.2,
            a=2.0,
            B0=5.3,
        )
        assert not r.stable
        assert r.elm_type in ("type_I", "type_III")
