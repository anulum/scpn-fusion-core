# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — L-H Transition Tests
from __future__ import annotations

import numpy as np
import pytest

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


def test_predator_prey_rejects_invalid_step_domain():
    model = PredatorPreyModel()

    with pytest.raises(ValueError, match="state"):
        model.step(np.array([1.0, np.nan, 1.0]), dt=0.001, Q_heating=1.0)
    with pytest.raises(ValueError, match="dt"):
        model.step(np.array([1.0, 1.0, 1.0]), dt=0.0, Q_heating=1.0)
    with pytest.raises(ValueError, match="Q_heating"):
        model.step(np.array([1.0, 1.0, 1.0]), dt=0.001, Q_heating=-1.0)


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


def test_transition_controller_clamps_ramp_to_target():
    model = PredatorPreyModel()
    ctrl = LHTransitionController(model, Q_target=50.0)

    Q_next = ctrl.step(epsilon_measured=1e5, Q_current=49.9, dt=1.0)

    assert Q_next == 50.0


def test_transition_controller_holds_during_i_phase():
    model = PredatorPreyModel()
    ctrl = LHTransitionController(model, Q_target=50.0)
    oscillation = 1e5 * (1.0 + 0.4 * np.sin(np.linspace(0.0, 8.0 * np.pi, 120)))

    Q = 20.0
    for eps in oscillation:
        Q = ctrl.step(epsilon_measured=float(eps), Q_current=Q, dt=0.01)

    Q_hold = ctrl.step(epsilon_measured=1e4, Q_current=Q, dt=0.01)

    assert Q_hold < 50.0


def test_predator_prey_rejects_invalid_drive_parameters() -> None:
    with pytest.raises(ValueError, match="p0"):
        PredatorPreyModel(p0=0.0)
    with pytest.raises(ValueError, match="drive_gain"):
        PredatorPreyModel(drive_gain=-1.0)


def test_turbulence_drive_increases_with_heating_and_pressure() -> None:
    model = PredatorPreyModel(p0=10.0, drive_gain=100.0)
    s_low = model.turbulence_drive(p_edge=2.0, Q_heating=1.0)
    s_hi_q = model.turbulence_drive(p_edge=2.0, Q_heating=20.0)
    s_hi_p = model.turbulence_drive(p_edge=20.0, Q_heating=20.0)
    assert s_hi_q > s_low
    assert s_hi_p > s_hi_q


def test_step_is_sensitive_to_drive_gain() -> None:
    state = np.array([1e4, 1.0, 2.0], dtype=float)
    weak = PredatorPreyModel(drive_gain=10.0)
    strong = PredatorPreyModel(drive_gain=400.0)
    next_weak = weak.step(state, dt=1e-4, Q_heating=10.0)
    next_strong = strong.step(state, dt=1e-4, Q_heating=10.0)
    assert next_strong[0] > next_weak[0]


def test_i_phase_detector_rejects_invalid_configuration_and_trace() -> None:
    with pytest.raises(ValueError, match="window_size"):
        IPhaseDetector(window_size=1)
    with pytest.raises(ValueError, match="relative_std_threshold"):
        IPhaseDetector(relative_std_threshold=0.0)
    with pytest.raises(ValueError, match="min_absolute_std"):
        IPhaseDetector(min_absolute_std=-1.0)

    det = IPhaseDetector(window_size=10)
    bad_trace = np.ones(20)
    bad_trace[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        det.detect(bad_trace)


def test_i_phase_detector_threshold_parameter_controls_detection() -> None:
    t = np.linspace(0, 6.0 * np.pi, 200)
    trace = 1e4 * (1.0 + 0.18 * np.sin(t))
    strict = IPhaseDetector(window_size=100, relative_std_threshold=0.15)
    permissive = IPhaseDetector(window_size=100, relative_std_threshold=0.05)
    assert not strict.detect(trace)
    assert permissive.detect(trace)


def test_transition_controller_rejects_invalid_hmode_threshold_and_ramp() -> None:
    model = PredatorPreyModel()
    with pytest.raises(ValueError, match="epsilon_hmode_threshold"):
        LHTransitionController(model, Q_target=50.0, epsilon_hmode_threshold=0.0)
    with pytest.raises(ValueError, match="q_ramp_rate"):
        LHTransitionController(model, Q_target=50.0, q_ramp_rate=0.0)


def test_transition_controller_hmode_threshold_and_ramp_are_configurable() -> None:
    model = PredatorPreyModel()
    slow = LHTransitionController(
        model, Q_target=50.0, epsilon_hmode_threshold=2.0e4, q_ramp_rate=5.0
    )
    fast = LHTransitionController(
        model, Q_target=50.0, epsilon_hmode_threshold=8.0e4, q_ramp_rate=20.0
    )
    # same measured epsilon above low threshold but below high threshold:
    # fast controller should classify as H-mode and jump to target.
    q_slow = slow.step(epsilon_measured=5.0e4, Q_current=10.0, dt=0.1)
    q_fast = fast.step(epsilon_measured=5.0e4, Q_current=10.0, dt=0.1)
    assert q_slow > 10.0 and q_slow < 50.0
    assert q_fast == 50.0

    # both in ramp regime, faster ramp should increase more per step.
    q_slow_ramp = slow.step(epsilon_measured=9.0e4, Q_current=10.0, dt=0.1)
    q_fast_ramp = fast.step(epsilon_measured=9.0e4, Q_current=10.0, dt=0.1)
    assert q_fast_ramp > q_slow_ramp


def test_predator_prey_turbulence_drive_validation() -> None:
    model = PredatorPreyModel()
    with pytest.raises(ValueError, match="p_edge"):
        model.turbulence_drive(-1.0, 10.0)
    with pytest.raises(ValueError, match="Q_heating"):
        model.turbulence_drive(1.0, -1.0)


def test_martin_threshold_zero_for_nonpositive_inputs() -> None:
    from scpn_fusion.core.lh_transition import MartinThreshold

    assert MartinThreshold.power_threshold_MW(0.0, 5.0, 100.0) == 0.0
    assert MartinThreshold.power_threshold_MW(5.0, 5.0, 100.0) > 0.0


def test_lh_transition_controller_validation() -> None:
    from scpn_fusion.core.lh_transition import LHTransitionController, PredatorPreyModel

    with pytest.raises(ValueError, match="Q_target"):
        LHTransitionController(PredatorPreyModel(), Q_target=-1.0)
    with pytest.raises(ValueError, match="epsilon_hmode_threshold"):
        LHTransitionController(PredatorPreyModel(), Q_target=10.0, epsilon_hmode_threshold=0.0)
    with pytest.raises(ValueError, match="q_ramp_rate"):
        LHTransitionController(PredatorPreyModel(), Q_target=10.0, q_ramp_rate=0.0)
    ctrl = LHTransitionController(PredatorPreyModel(), Q_target=10.0)
    with pytest.raises(ValueError, match="epsilon_measured"):
        ctrl.step(-1.0, 1.0, 0.01)
    with pytest.raises(ValueError, match="Q_current"):
        ctrl.step(0.1, -1.0, 0.01)


def test_lh_trigger_find_threshold_returns_power() -> None:
    from scpn_fusion.core.lh_transition import LHTrigger, PredatorPreyModel

    trig = LHTrigger(PredatorPreyModel())
    q = trig.find_threshold(np.linspace(0.1, 50.0, 40))
    assert q > 0.0


def test_lh_controller_step_rejects_bad_dt() -> None:
    from scpn_fusion.core.lh_transition import LHTransitionController, PredatorPreyModel

    ctrl = LHTransitionController(PredatorPreyModel(), Q_target=10.0)
    with pytest.raises(ValueError, match="dt must be finite"):
        ctrl.step(0.1, 1.0, 0.0)
