# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Coverage Batch 6 (API-verified, corrected)
from __future__ import annotations

import numpy as np
import pytest


class TestVesselModel:
    def test_vessel_element(self):
        from scpn_fusion.core.vessel_model import VesselElement

        ve = VesselElement(R=6.2, Z=0.0, resistance=1e-6, cross_section=0.04, inductance=1e-6)
        assert pytest.approx(6.2) == ve.R

    def test_vessel_model_init(self):
        from scpn_fusion.core.vessel_model import VesselElement, VesselModel

        elements = [
            VesselElement(R=4.0, Z=2.0, resistance=1e-6, cross_section=0.04, inductance=1e-6),
            VesselElement(R=8.0, Z=-2.0, resistance=1e-6, cross_section=0.04, inductance=1e-6),
        ]
        vm = VesselModel(elements=elements)
        assert len(vm.elements) == 2


class TestSawtooth:
    def test_kadomtsev_crash(self):
        from scpn_fusion.core.sawtooth import kadomtsev_crash

        rho = np.linspace(0, 1, 50)
        q = 0.8 + 2.0 * rho**2
        Te = 10.0 * (1 - rho**2) + 0.5
        ne = 10.0 * (1 - 0.5 * rho**2)
        result = kadomtsev_crash(rho, Te, ne, q, R0=6.2, a=2.0)
        assert len(result) >= 3

    def test_sawtooth_cycler_init(self):
        from scpn_fusion.core.sawtooth import SawtoothCycler

        rho = np.linspace(0, 1, 50)
        cycler = SawtoothCycler(rho, R0=6.2, a=2.0)
        assert cycler is not None


class TestCurrentDiffusion:
    def test_neoclassical_resistivity(self):
        from scpn_fusion.core.current_diffusion import neoclassical_resistivity

        eta = neoclassical_resistivity(Te_keV=10.0, ne_19=10.0, Z_eff=1.5, epsilon=0.3)
        assert eta > 0

    def test_q_from_psi(self):
        from scpn_fusion.core.current_diffusion import q_from_psi

        rho = np.linspace(0.01, 1.0, 50)
        psi = 1.0 - rho**2
        q = q_from_psi(rho, psi, R0=6.2, a=2.0, B0=5.3)
        assert len(q) == 50
        assert np.all(np.isfinite(q))

    def test_resistive_diffusion_time(self):
        from scpn_fusion.core.current_diffusion import resistive_diffusion_time

        tau = resistive_diffusion_time(a=2.0, eta=1e-8)
        assert tau > 0

    def test_solver_init(self):
        from scpn_fusion.core.current_diffusion import CurrentDiffusionSolver

        rho = np.linspace(0, 1, 30)
        solver = CurrentDiffusionSolver(rho, R0=6.2, a=2.0, B0=5.3)
        assert pytest.approx(6.2) == solver.R0


class TestScalingLaws:
    def test_require_positive_finite(self):
        from scpn_fusion.core.scaling_laws import _require_positive_finite

        assert _require_positive_finite("x", 3.14) == pytest.approx(3.14)
        with pytest.raises(ValueError):
            _require_positive_finite("x", -1.0)

    def test_require_finite_number(self):
        from scpn_fusion.core.scaling_laws import _require_finite_number

        assert _require_finite_number("x", 42) == pytest.approx(42.0)
        with pytest.raises(ValueError):
            _require_finite_number("x", float("nan"))

    def test_assess_domain(self):
        from scpn_fusion.core.scaling_laws import assess_ipb98y2_domain

        result = assess_ipb98y2_domain(
            Ip=15.0,
            BT=5.3,
            ne19=101.0,
            Ploss=50.0,
            R=6.2,
            kappa=1.7,
            epsilon=0.32,
            M=2.5,
        )
        assert isinstance(result, dict)
