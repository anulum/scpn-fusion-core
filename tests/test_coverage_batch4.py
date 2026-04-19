# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — Coverage Batch 4 (API-verified)
from __future__ import annotations

import numpy as np


class TestThermalQuench:
    def test_rechester_rosenbluth(self):
        from scpn_fusion.core.disruption_sequence import ThermalQuench

        tq = ThermalQuench(W_th_MJ=300.0, a=2.0, R0=6.2, q=2.0, B0=5.3)
        chi = tq.rechester_rosenbluth_chi(dBr_over_B=1e-3, v_e=1e7)
        assert chi > 0

    def test_quench_timescale(self):
        from scpn_fusion.core.disruption_sequence import ThermalQuench

        tq = ThermalQuench(W_th_MJ=300.0, a=2.0, R0=6.2, q=2.0, B0=5.3)
        tau = tq.quench_timescale(dBr_over_B=1e-3, Te_pre_keV=10.0)
        assert tau > 0

    def test_post_tq_temperature(self):
        from scpn_fusion.core.disruption_sequence import ThermalQuench

        tq = ThermalQuench(W_th_MJ=300.0, a=2.0, R0=6.2, q=2.0, B0=5.3)
        T_post = tq.post_tq_temperature(Te_pre_keV=10.0, tau_tq_ms=1.0)
        assert T_post > 0


class TestCurrentQuench:
    def test_resistivity(self):
        from scpn_fusion.core.disruption_sequence import CurrentQuench

        cq = CurrentQuench(Ip_MA=15.0, L_plasma_uH=10.0, R0=6.2, a=2.0, kappa=1.7)
        eta = cq.resistivity_post_tq(Te_eV=10.0, Z_eff=2.0)
        assert eta > 0

    def test_cq_timescale(self):
        from scpn_fusion.core.disruption_sequence import CurrentQuench

        cq = CurrentQuench(Ip_MA=15.0, L_plasma_uH=10.0, R0=6.2, a=2.0, kappa=1.7)
        tau = cq.cq_timescale(Te_eV=10.0, Z_eff=2.0)
        assert tau > 0


class TestPaschenBreakdown:
    def test_breakdown_voltage(self):
        from scpn_fusion.core.plasma_startup import PaschenBreakdown

        pb = PaschenBreakdown(gas="D2", R0=6.2, a=2.0)
        V = pb.breakdown_voltage(p_Pa=0.1, connection_length_m=100.0)
        assert V > 0

    def test_is_breakdown(self):
        from scpn_fusion.core.plasma_startup import PaschenBreakdown

        pb = PaschenBreakdown(gas="D2", R0=6.2, a=2.0)
        result = pb.is_breakdown(V_loop=100.0, p_Pa=0.1)
        assert isinstance(result, bool)

    def test_paschen_curve(self):
        from scpn_fusion.core.plasma_startup import PaschenBreakdown

        pb = PaschenBreakdown(gas="D2")
        p = np.logspace(-2, 1, 20)
        V = pb.paschen_curve(p)
        assert len(V) == 20
        assert np.all(V > 0)


class TestTownsendAvalanche:
    def test_ionization_rate(self):
        from scpn_fusion.core.plasma_startup import TownsendAvalanche

        ta = TownsendAvalanche(V_loop=10.0, p_Pa=0.1, R0=6.2, a=2.0)
        rate = ta.ionization_rate(Te_eV=20.0)
        assert rate > 0

    def test_evolve(self):
        from scpn_fusion.core.plasma_startup import TownsendAvalanche

        ta = TownsendAvalanche(V_loop=10.0, p_Pa=0.1, R0=6.2, a=2.0)
        result = ta.evolve(dt=1e-6, n_steps=100)
        assert result.ne_trace.shape == (100,)
        assert result.Te_trace.shape == (100,)


class TestBlobDynamics:
    def test_critical_size(self):
        from scpn_fusion.core.blob_transport import BlobDynamics

        bd = BlobDynamics(R0=6.2, B0=5.3, Te_eV=100.0, Ti_eV=100.0)
        delta_c = bd.critical_size(L_parallel=20.0)
        assert delta_c > 0

    def test_blob_velocity(self):
        from scpn_fusion.core.blob_transport import BlobDynamics

        bd = BlobDynamics(R0=6.2, B0=5.3, Te_eV=100.0, Ti_eV=100.0)
        v, regime = bd.blob_velocity(delta_b=0.02, n_e=1e19, L_parallel=20.0)
        assert v >= 0
        assert regime in ("sheath", "inertial")

    def test_max_velocity(self):
        from scpn_fusion.core.blob_transport import BlobDynamics

        bd = BlobDynamics(R0=6.2, B0=5.3, Te_eV=100.0, Ti_eV=100.0)
        v = bd.max_velocity(L_parallel=20.0)
        assert v > 0


class TestAlfvenContinuum:
    def test_alfven_speed(self):
        from scpn_fusion.core.alfven_eigenmodes import AlfvenContinuum

        rho = np.linspace(0, 1, 30)
        q = 1.0 + 2.0 * rho**2
        ne = 1e20 * (1 - 0.5 * rho**2)
        ac = AlfvenContinuum(rho=rho, q=q, ne=ne, R0=6.2, B0=5.3)
        v_A = ac.alfven_speed(rho_eval=0.5)
        assert v_A > 0

    def test_continuum_frequencies(self):
        from scpn_fusion.core.alfven_eigenmodes import AlfvenContinuum

        rho = np.linspace(0, 1, 30)
        q = 1.0 + 2.0 * rho**2
        ne = 1e20 * (1 - 0.5 * rho**2)
        ac = AlfvenContinuum(rho=rho, q=q, ne=ne, R0=6.2, B0=5.3)
        freqs = ac.continuum(m=1, n=1)
        assert len(freqs) == 30

    def test_find_gaps(self):
        from scpn_fusion.core.alfven_eigenmodes import AlfvenContinuum

        rho = np.linspace(0, 1, 30)
        q = 1.0 + 2.0 * rho**2
        ne = 1e20 * (1 - 0.5 * rho**2)
        ac = AlfvenContinuum(rho=rho, q=q, ne=ne, R0=6.2, B0=5.3)
        gaps = ac.find_gaps(n=1)
        assert isinstance(gaps, list)


class TestTAEMode:
    def test_frequency(self):
        from scpn_fusion.core.alfven_eigenmodes import TAEMode

        tae = TAEMode(n=1, q_rational=1.5, v_A=1e7, R0=6.2)
        f = tae.frequency()
        assert f > 0

    def test_frequency_kHz(self):
        from scpn_fusion.core.alfven_eigenmodes import TAEMode

        tae = TAEMode(n=1, q_rational=1.5, v_A=1e7, R0=6.2)
        f_kHz = tae.frequency_kHz()
        assert f_kHz > 0


class TestImpurityTransport:
    def test_cooling_curve(self):
        from scpn_fusion.core.impurity_transport import CoolingCurve

        cc = CoolingCurve(element="C")
        Te = np.array([10.0, 50.0, 100.0, 500.0])
        L = cc.L_z(Te)
        assert len(L) == 4
        assert np.all(L > 0)

    def test_neoclassical_pinch(self):
        from scpn_fusion.core.impurity_transport import neoclassical_impurity_pinch

        # neoclassical_impurity_pinch(Z, ne, Te_eV, Ti_eV, q, rho, R0, a)
        rho = np.linspace(0.1, 0.9, 20)
        ne = 10.0 * (1 - rho**2) * 1e19
        Te = (5.0 * (1 - rho**2) + 0.5) * 1e3  # eV
        Ti = Te.copy()
        q = 1.0 + 2.0 * rho**2
        epsilon = rho * 2.0 / 6.2  # rho * a / R0
        result = neoclassical_impurity_pinch(74, ne, Te, Ti, q, rho, R0=6.2, a=2.0, epsilon=epsilon)
        assert len(result) == 20

    def test_tungsten_diagnostic(self):
        from scpn_fusion.core.impurity_transport import tungsten_accumulation_diagnostic

        n_W = np.array([1e16, 2e16, 3e16])
        ne = np.array([1e20, 1e20, 1e20])
        result = tungsten_accumulation_diagnostic(n_W, ne)
        assert isinstance(result, dict)
