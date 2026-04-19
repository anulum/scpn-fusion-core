# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — L-H Transition Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.lh_transition import (
    IPhaseDetector,
    LHTransitionController,
    LHTrigger,
    MartinThreshold,
    PredatorPreyModel,
)


def test_predator_prey_l_mode():
    model = PredatorPreyModel()
    res = model.evolve(Q_heating=1.0, t_span=(0.0, 1.0), dt=0.001)

    assert res.regime == "L_MODE"
    assert res.epsilon_trace[-1] > 1e4
    assert res.V_ZF_trace[-1] < 10.0


def test_predator_prey_h_mode():
    model = PredatorPreyModel(gamma_damp=0.01, alpha3=1e-3)
    # High enough Q to push pressure and thus turbulence into the saturation regime where ZF suppress it
    res = model.evolve(Q_heating=1000.0, t_span=(0.0, 10.0), dt=0.001)

    # We should have triggered H-mode -> high V_ZF, low epsilon
    assert res.regime == "H_MODE"
    assert res.V_ZF_trace[-1] > 100.0


def test_lh_trigger_threshold():
    model = PredatorPreyModel()
    trigger = LHTrigger(model)

    Q_range = np.linspace(1.0, 500.0, 10)
    Q_th = trigger.find_threshold(Q_range)

    assert 1.0 < Q_th <= 500.0


def test_martin_scaling():
    # ITER typical parameters
    # a=2.0, R=6.2, kappa=1.7 -> S ~ 4 pi^2 a R kappa ~ 40 * 2 * 6.2 * 1.7 ~ 840 m^2
    S_iter = 840.0
    P_th = MartinThreshold.power_threshold_MW(ne_19=5.0, B_T=5.3, S_m2=S_iter)

    # Should be around 333 MW for high density and large area
    assert 100.0 < P_th < 500.0


def test_i_phase_detector():
    det = IPhaseDetector(window_size=100)

    # Stable trace -> no I-phase
    trace_stable = np.ones(200) * 1e4
    assert not det.detect(trace_stable)

    # Oscillating trace -> I-phase
    t = np.linspace(0, 10, 200)
    trace_osc = 1e4 * (1.0 + 0.5 * np.sin(t))
    assert det.detect(trace_osc)


def test_transition_controller():
    model = PredatorPreyModel()
    ctrl = LHTransitionController(model, Q_target=50.0)

    # Still L-mode -> ramp
    Q1 = ctrl.step(epsilon_measured=1e5, Q_current=10.0, dt=0.1)
    assert Q1 > 10.0

    # Hit H-mode -> jump to target
    Q2 = ctrl.step(epsilon_measured=1e4, Q_current=20.0, dt=0.1)
    assert Q2 == 50.0
