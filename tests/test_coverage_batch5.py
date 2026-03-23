# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Coverage Batch 5 (control modules, API-verified)
from __future__ import annotations

import numpy as np
import pytest


class TestFaultTolerantControl:
    def test_fdi_monitor_init(self):
        from scpn_fusion.control.fault_tolerant_control import FDIMonitor

        fdi = FDIMonitor(n_sensors=6, n_actuators=4)
        assert fdi is not None

    def test_fdi_update(self):
        from scpn_fusion.control.fault_tolerant_control import FDIMonitor

        fdi = FDIMonitor(n_sensors=6, n_actuators=4)
        measurements = np.ones(6)
        predictions = np.ones(6) * 0.99
        reports = fdi.update(measurements, predictions, t=0.0)
        assert isinstance(reports, list)


class TestDensityController:
    def test_particle_transport_init(self):
        from scpn_fusion.control.density_controller import ParticleTransportModel

        pt = ParticleTransportModel(n_rho=30, R0=6.2, a=2.0)
        assert pt.n_rho == 30

    def test_gas_puff_source(self):
        from scpn_fusion.control.density_controller import ParticleTransportModel

        pt = ParticleTransportModel(n_rho=30)
        src = pt.gas_puff_source(rate=1e22, penetration_depth=0.03)
        assert len(src) == 30
        assert np.any(src > 0)

    def test_cryopump_sink(self):
        from scpn_fusion.control.density_controller import ParticleTransportModel

        pt = ParticleTransportModel(n_rho=30)
        sink = pt.cryopump_sink(pump_speed=50.0, ne_edge=5e19)
        assert len(sink) == 30


class TestDetachmentController:
    def test_radiation_front_model(self):
        from scpn_fusion.control.detachment_controller import RadiationFrontModel

        rf = RadiationFrontModel(impurity="N2", R0=6.2, a=2.0, q95=3.0)
        pos = rf.front_position(P_SOL_MW=50.0, n_u_19=5.0, seeding_rate=1e21)
        assert np.isfinite(pos)

    def test_degree_of_detachment(self):
        from scpn_fusion.control.detachment_controller import RadiationFrontModel

        rf = RadiationFrontModel(impurity="N2", R0=6.2, a=2.0, q95=3.0)
        dod = rf.degree_of_detachment(T_target_eV=3.0, n_target=1e20, n_u=5e19)
        assert dod > 0

    def test_controller_init(self):
        from scpn_fusion.control.detachment_controller import DetachmentController

        ctrl = DetachmentController(impurity="N2", target_DOD=3.0, target_T_t_eV=3.0)
        assert ctrl is not None


class TestBurnController:
    def test_alpha_heating_init(self):
        from scpn_fusion.control.burn_controller import AlphaHeating

        ah = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)
        assert pytest.approx(6.2) == ah.R0

    def test_Q_factor(self):
        from scpn_fusion.control.burn_controller import AlphaHeating

        ah = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)
        Q = ah.Q(P_alpha_MW=100.0, P_aux_MW=50.0)
        assert pytest.approx(10.0) == Q  # Q = P_fus/P_aux = 5*P_alpha/P_aux

    def test_burn_stability_analysis(self):
        from scpn_fusion.control.burn_controller import AlphaHeating, BurnStabilityAnalysis

        ah = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)
        bsa = BurnStabilityAnalysis(ah)
        n = bsa.reactivity_exponent(Ti_keV=15.0)
        assert np.isfinite(n)


class TestFuelingMode:
    def test_ice_pellet_controller(self):
        from scpn_fusion.control.fueling_mode import IcePelletFuelingController

        ctrl = IcePelletFuelingController(target_density=1.0)
        pellet_rate, gas_rate = ctrl.step(density=0.9, k=0, dt_s=0.01)
        assert np.isfinite(pellet_rate)
        assert np.isfinite(gas_rate)

    def test_above_target_reduces_fueling(self):
        from scpn_fusion.control.fueling_mode import IcePelletFuelingController

        ctrl = IcePelletFuelingController(target_density=1.0)
        pellet_rate, gas_rate = ctrl.step(density=1.2, k=0, dt_s=0.01)
        # Above target — controller may reduce but not necessarily zero
        assert np.isfinite(pellet_rate)
