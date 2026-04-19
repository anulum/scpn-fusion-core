# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — ELM Model Full Coverage Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.elm_model import (
    ELMCrashModel,
    ELMCrashResult,
    PeelingBallooningBoundary,
    RMPSuppression,
    elm_power_balance_frequency,
)


class TestPeelingBallooningBoundary:
    def test_peeling_limit_positive(self):
        pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.33, a=2.0, R0=6.2)
        j_crit = pb.peeling_limit(j_edge=1e5)
        assert j_crit > 0

    def test_ballooning_limit_positive(self):
        pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.33, a=2.0, R0=6.2)
        a_crit = pb.ballooning_limit(s_edge=2.0)
        assert a_crit > 0

    def test_is_unstable_low_gradients_stable(self):
        pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.33, a=2.0, R0=6.2)
        assert not pb.is_unstable(alpha_edge=0.01, j_edge=100.0, s_edge=2.0)

    def test_is_unstable_high_gradients(self):
        pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.33, a=2.0, R0=6.2)
        assert pb.is_unstable(alpha_edge=10.0, j_edge=1e7, s_edge=2.0)

    def test_stability_margin_positive_when_stable(self):
        pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.33, a=2.0, R0=6.2)
        margin = pb.stability_margin(alpha_edge=0.01, j_edge=100.0, s_edge=2.0)
        assert margin > 0

    def test_stability_margin_negative_when_unstable(self):
        pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.33, a=2.0, R0=6.2)
        margin = pb.stability_margin(alpha_edge=10.0, j_edge=1e7, s_edge=2.0)
        assert margin < 0


class TestELMCrashModel:
    def test_crash_returns_result(self):
        model = ELMCrashModel(f_elm_fraction=0.08)
        result = model.crash(T_ped=5.0, n_ped=8.0, W_ped=1.0, A_wet=1.0)
        assert isinstance(result, ELMCrashResult)
        assert result.delta_W_MJ > 0
        assert result.T_ped_post < 5.0
        assert result.n_ped_post < 8.0

    def test_crash_energy_fraction(self):
        model = ELMCrashModel(f_elm_fraction=0.1)
        result = model.crash(T_ped=5.0, n_ped=8.0, W_ped=10.0)
        assert result.delta_W_MJ == pytest.approx(1.0)

    def test_apply_to_profiles(self):
        model = ELMCrashModel(f_elm_fraction=0.08)
        rho = np.linspace(0, 1, 50)
        Te = 10.0 * (1 - rho**2)
        ne = 10.0 * (1 - 0.5 * rho**2)
        Te_new, ne_new = model.apply_to_profiles(rho, Te, ne, rho_ped=0.9)
        # Pedestal region should be reduced (check at rho=0.95, not 1.0 where Te=0)
        idx_95 = np.searchsorted(rho, 0.95)
        assert Te_new[idx_95] < Te[idx_95]
        assert ne_new[idx_95] < ne[idx_95]
        # Core should be unchanged
        assert Te_new[0] == Te[0]


class TestRMPSuppression:
    def test_chirikov_zero_for_zero_perturbation(self):
        rmp = RMPSuppression(n_coils=3, I_rmp_kA=90.0, n_toroidal=3)
        rho = np.linspace(0, 1, 30)
        q = 1.0 + 2.0 * rho**2
        sigma = rmp.chirikov_parameter(q, rho, delta_B_r=0.0, B0=5.3, R0=6.2)
        assert sigma == 0.0

    def test_chirikov_positive_for_perturbation(self):
        rmp = RMPSuppression(n_coils=3, I_rmp_kA=90.0, n_toroidal=3)
        rho = np.linspace(0, 1, 30)
        q = 1.0 + 2.0 * rho**2
        sigma = rmp.chirikov_parameter(q, rho, delta_B_r=1e-3, B0=5.3, R0=6.2)
        assert sigma > 0

    def test_suppressed_above_threshold(self):
        rmp = RMPSuppression()
        assert rmp.suppressed(1.5) is True
        assert rmp.suppressed(0.5) is False

    def test_pedestal_transport_enhancement(self):
        rmp = RMPSuppression()
        assert rmp.pedestal_transport_enhancement(0.5) == 1.0  # no enhancement below 1
        assert rmp.pedestal_transport_enhancement(2.0) > 1.0

    def test_density_pump_out(self):
        rmp = RMPSuppression()
        assert rmp.density_pump_out(0.5) == 0.0
        assert rmp.density_pump_out(1.5) == pytest.approx(0.2)


class TestELMPowerBalanceFrequency:
    def test_positive_frequency(self):
        f = elm_power_balance_frequency(P_SOL_MW=50.0, W_ped_MJ=1.0, f_elm_fraction=0.08)
        assert f > 0

    def test_zero_for_zero_power(self):
        f = elm_power_balance_frequency(P_SOL_MW=50.0, W_ped_MJ=0.0, f_elm_fraction=0.08)
        assert f == 0.0

    def test_zero_for_zero_fraction(self):
        f = elm_power_balance_frequency(P_SOL_MW=50.0, W_ped_MJ=1.0, f_elm_fraction=0.0)
        assert f == 0.0
