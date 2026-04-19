# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — Coverage Batch 3 (API-verified tests)
from __future__ import annotations

import numpy as np
import pytest


class TestELMCrashModel:
    def test_crash_returns_result(self):
        from scpn_fusion.core.elm_model import ELMCrashModel

        model = ELMCrashModel(f_elm_fraction=0.08)
        result = model.crash(T_ped=5.0, n_ped=8.0, W_ped=1.0, A_wet=1.0)
        assert result.delta_W_MJ == pytest.approx(0.08)
        assert result.T_ped_post < 5.0
        assert result.n_ped_post < 8.0

    def test_apply_to_profiles(self):
        from scpn_fusion.core.elm_model import ELMCrashModel

        model = ELMCrashModel(f_elm_fraction=0.08)
        rho = np.linspace(0, 1, 50)
        Te = 10.0 * (1 - rho**2)
        ne = 10.0 * (1 - 0.5 * rho**2)
        Te_new, ne_new = model.apply_to_profiles(rho, Te, ne, rho_ped=0.9)
        assert Te_new[0] == Te[0]  # core unchanged
        idx_95 = np.searchsorted(rho, 0.95)
        assert Te_new[idx_95] < Te[idx_95]  # pedestal reduced


class TestRMPSuppression:
    def test_chirikov_positive(self):
        from scpn_fusion.core.elm_model import RMPSuppression

        rmp = RMPSuppression(n_coils=3, I_rmp_kA=90.0, n_toroidal=3)
        rho = np.linspace(0, 1, 30)
        q = 1.0 + 2.0 * rho**2
        sigma = rmp.chirikov_parameter(q, rho, delta_B_r=1e-3, B0=5.3, R0=6.2)
        assert sigma > 0

    def test_suppressed(self):
        from scpn_fusion.core.elm_model import RMPSuppression

        rmp = RMPSuppression()
        assert rmp.suppressed(1.5) is True
        assert rmp.suppressed(0.5) is False

    def test_transport_enhancement(self):
        from scpn_fusion.core.elm_model import RMPSuppression

        rmp = RMPSuppression()
        assert rmp.pedestal_transport_enhancement(2.0) > 1.0

    def test_density_pump_out(self):
        from scpn_fusion.core.elm_model import RMPSuppression

        rmp = RMPSuppression()
        assert rmp.density_pump_out(1.5) == pytest.approx(0.2)


class TestELMFrequency:
    def test_positive(self):
        from scpn_fusion.core.elm_model import elm_power_balance_frequency

        f = elm_power_balance_frequency(P_SOL_MW=50.0, W_ped_MJ=1.0, f_elm_fraction=0.08)
        assert f > 0


class TestModeLocking:
    def test_em_torque_positive(self):
        from scpn_fusion.core.locked_mode import ModeLocking

        ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)
        torque = ml.em_torque(B_res=1e-3, r_s=0.5, m=2, n=1)
        assert torque > 0

    def test_evolve_rotation(self):
        from scpn_fusion.core.locked_mode import ModeLocking

        ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)
        result = ml.evolve_rotation(B_res=1e-3, r_s=0.5, tau_visc=0.01, dt=1e-4, n_steps=100)
        assert len(result.omega_trace) == 100
        assert isinstance(result.locked, bool)


class TestRadiationCondensation:
    def test_growth_rate(self):
        from scpn_fusion.core.marfe import RadiationCondensation

        rc = RadiationCondensation(impurity="C", ne_20=1.0, f_imp=0.03)
        gamma = rc.growth_rate(Te_eV=50.0, k_par=1.0, kappa_par=1e20)
        assert np.isfinite(gamma)

    def test_is_unstable(self):
        from scpn_fusion.core.marfe import RadiationCondensation

        rc = RadiationCondensation(impurity="C", ne_20=1.0, f_imp=0.03)
        result = rc.is_unstable(Te_eV=50.0, k_par=1.0, kappa_par=1e20)
        assert isinstance(result, bool)


class TestDensityLimitPredictor:
    def test_greenwald_limit(self):
        from scpn_fusion.core.marfe import DensityLimitPredictor

        n_gw = DensityLimitPredictor.greenwald_limit(Ip_MA=15.0, a=2.0)
        assert n_gw > 0
        assert np.isfinite(n_gw)

    def test_greenwald_zero_a(self):
        from scpn_fusion.core.marfe import DensityLimitPredictor

        n_gw = DensityLimitPredictor.greenwald_limit(Ip_MA=15.0, a=0.0)
        assert n_gw == float("inf")


class TestMomentumTransport:
    def test_nbi_torque(self):
        from scpn_fusion.core.momentum_transport import nbi_torque

        P_profile = np.ones(30) * 1e6
        T = nbi_torque(P_profile, R0=6.2, v_beam=1e7, theta_inj_deg=30.0)
        assert len(T) == 30
        assert np.all(T > 0)

    def test_nbi_torque_zero_beam(self):
        from scpn_fusion.core.momentum_transport import nbi_torque

        P_profile = np.ones(30) * 1e6
        T = nbi_torque(P_profile, R0=6.2, v_beam=0.0, theta_inj_deg=30.0)
        assert np.all(T == 0.0)

    def test_exb_shearing(self):
        from scpn_fusion.core.momentum_transport import exb_shearing_rate

        rho = np.linspace(0.01, 0.99, 30)
        omega = np.sin(np.pi * rho) * 1e4
        B_theta = 0.5 * np.ones(30)
        rate = exb_shearing_rate(omega, B_theta, B0=5.3, R0=6.2, rho=rho, a=2.0)
        assert len(rate) == 30
        assert np.all(np.isfinite(rate))

    def test_intrinsic_rotation(self):
        from scpn_fusion.core.momentum_transport import intrinsic_rotation_torque

        grad_Ti = np.linspace(-1e3, 0, 30)
        grad_ne = np.linspace(-1e19, 0, 30)
        T = intrinsic_rotation_torque(grad_Ti, grad_ne, R0=6.2, a=2.0)
        assert len(T) == 30


class TestDisruptionSequence:
    def test_thermal_quench_init(self):
        from scpn_fusion.core.disruption_sequence import ThermalQuench

        tq = ThermalQuench(W_th_MJ=300.0, a=2.0, R0=6.2, q=2.0, B0=5.3)
        assert tq is not None

    def test_current_quench_init(self):
        from scpn_fusion.core.disruption_sequence import CurrentQuench

        cq = CurrentQuench(Ip_MA=15.0, L_plasma_uH=10.0, R0=6.2, a=2.0, kappa=1.7)
        assert cq is not None


class TestPelletInjection:
    def test_ngs_ablation_physical(self):
        from scpn_fusion.core.pellet_injection import ngs_ablation_rate

        # ngs_ablation_rate(r_p, ne, Te_eV, M_p) all positional
        rate = ngs_ablation_rate(0.003, 1e20, 5000.0, 2.0)
        assert rate > 0

    def test_pellet_params(self):
        from scpn_fusion.core.pellet_injection import PelletParams

        pp = PelletParams(r_p_mm=3.0, v_p_m_s=500.0)
        assert pp.r_p_mm == 3.0
        assert pp.v_p_m_s == 500.0
