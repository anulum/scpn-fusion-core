# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — Integrated Scenario Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.integrated_scenario import (
    IntegratedScenarioSimulator,
    ScenarioConfig,
    ScenarioState,
    iter_baseline_scenario,
    iter_hybrid_scenario,
    nstx_u_scenario,
)


class TestScenarioConfig:
    def test_iter_baseline_geometry(self):
        cfg = iter_baseline_scenario()
        assert pytest.approx(6.2) == cfg.R0
        assert cfg.a == pytest.approx(2.0)
        assert cfg.Ip_MA == pytest.approx(15.0)

    def test_iter_hybrid_lower_current(self):
        cfg = iter_hybrid_scenario()
        assert cfg.Ip_MA == pytest.approx(12.0)

    def test_nstx_u_compact(self):
        cfg = nstx_u_scenario()
        assert cfg.R0 < 1.0
        assert cfg.a < 1.0

    def test_iter_baseline_has_cd(self):
        cfg = iter_baseline_scenario()
        assert cfg.P_eccd_MW > 0
        assert cfg.P_nbi_MW > 0

    def test_defaults(self):
        cfg = ScenarioConfig(R0=3.0, a=1.0, B0=5.0, kappa=1.5, delta=0.3, Ip_MA=5.0, P_aux_MW=20.0)
        assert cfg.t_start == 0.0
        assert cfg.dt == 0.1
        assert cfg.transport_model == "gyro_bohm"


class TestSimulatorInit:
    def test_nstx_u_initializes(self):
        cfg = nstx_u_scenario()
        sim = IntegratedScenarioSimulator(cfg)
        assert sim.nr == 50
        assert len(sim.rho) == 50
        assert sim.time == cfg.t_start

    def test_iter_baseline_initializes(self):
        cfg = iter_baseline_scenario()
        sim = IntegratedScenarioSimulator(cfg)
        assert sim.eccd is not None
        assert sim.n_crashes == 0

    def test_no_eccd_when_zero_power(self):
        cfg = ScenarioConfig(
            R0=3.0,
            a=1.0,
            B0=5.0,
            kappa=1.5,
            delta=0.3,
            Ip_MA=5.0,
            P_aux_MW=20.0,
            P_eccd_MW=0.0,
        )
        sim = IntegratedScenarioSimulator(cfg)
        assert sim.eccd is None

    def test_ntm_widths_empty_at_start(self):
        cfg = nstx_u_scenario()
        sim = IntegratedScenarioSimulator(cfg)
        assert sim.ntm_widths == {}


class TestScenarioState:
    def test_dataclass_fields(self):
        state = ScenarioState(
            time=0.0,
            rho=np.linspace(0, 1, 10),
            Te=np.ones(10),
            Ti=np.ones(10),
            ne=np.ones(10),
            q=np.ones(10),
            psi=np.zeros(10),
            j_total=np.ones(10),
            j_bs=np.zeros(10),
            j_cd=np.zeros(10),
            Ip_MA=15.0,
            beta_N=1.8,
            tau_E=3.0,
            P_loss=50.0,
            W_thermal=300.0,
            li=0.8,
            ballooning_stable=True,
            troyon_stable=True,
            ntm_island_widths={},
            T_target=500.0,
            q_peak=5.0,
            detached=False,
            last_crash_time=0.0,
            n_crashes=0,
        )
        assert state.Ip_MA == 15.0
        assert state.ballooning_stable is True
        assert len(state.rho) == 10
