# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Scenario Simulator Tests
"""Requires TransportSolver(nr=...) API not yet available in fusion-core."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="TransportSolver API differs from scpn-control")

from scpn_fusion.core.integrated_scenario import (
    IntegratedScenarioSimulator,
    iter_baseline_scenario,
    nstx_u_scenario,
)


def test_pure_ohmic_iter_scenario():
    # Scenario without aux power or CD
    config = iter_baseline_scenario()
    config.P_aux_MW = 0.0
    config.P_eccd_MW = 0.0
    config.P_nbi_MW = 0.0
    config.t_end = 5.0
    config.dt = 1.0
    config.include_sawteeth = False
    config.include_ntm = False

    sim = IntegratedScenarioSimulator(config)
    state0 = sim.initialize()

    states = sim.run()
    assert len(states) == 5

    # Check that current profile evolves
    assert not (states[-1].q == state0.q).all()


def test_sawtooth_cycling():
    config = iter_baseline_scenario()
    config.t_end = 2.0
    config.dt = 1e-4  # Very small dt so q-profile doesn't relax before triggering
    config.include_sawteeth = True

    sim = IntegratedScenarioSimulator(config)

    # Force a q-profile that triggers a sawtooth
    import numpy as np

    rho = np.linspace(0, 1, 50)
    q_unstable = 0.8 + 2.0 * rho**2

    state0 = sim.initialize({"q": q_unstable})

    # Actually we need to override the psi profile so q is unstable
    a = config.a
    B0 = config.B0
    R0 = config.R0
    dpsi = -rho * a**2 * B0 / (R0 * q_unstable)
    psi = np.zeros(50)
    for i in range(1, 50):
        psi[i] = psi[i - 1] + dpsi[i] * (rho[1] - rho[0])
    psi -= psi[-1]

    sim.initialize({"psi": psi})

    # Step should trigger sawtooth
    sim.step()

    assert sim.n_crashes > 0
    assert sim.last_crash_time > 0.0


def test_energy_and_current_conservation_contracts():
    config = nstx_u_scenario()
    config.t_end = 0.05
    config.dt = 0.01

    sim = IntegratedScenarioSimulator(config)
    sim.initialize()

    state = sim.step()
    # Basic sanity checks
    assert state.W_thermal > 0
    assert state.Ip_MA == config.Ip_MA
    # check no negative temperature
    assert (state.Te > 0).all()
    assert (state.Ti > 0).all()
