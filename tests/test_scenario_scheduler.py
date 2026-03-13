# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Scenario Scheduler Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_fusion.control.scenario_scheduler import (
    FeedforwardController,
    ScenarioOptimizer,
    ScenarioSchedule,
    ScenarioWaveform,
    iter_15ma_baseline,
    nstx_u_1ma_standard,
)


def test_scenario_waveform_interpolation():
    times = np.array([0.0, 10.0, 20.0])
    values = np.array([0.0, 100.0, 50.0])
    wf = ScenarioWaveform("test", times, values)

    assert np.isclose(wf(0.0), 0.0)
    assert np.isclose(wf(10.0), 100.0)
    assert np.isclose(wf(20.0), 50.0)

    assert np.isclose(wf(5.0), 50.0)
    assert np.isclose(wf(15.0), 75.0)

    assert np.isclose(wf(-5.0), 0.0)
    assert np.isclose(wf(25.0), 50.0)


def test_scenario_validation():
    times = np.array([0.0, 10.0, 5.0])
    values = np.array([0.0, 10.0, 20.0])
    wf = ScenarioWaveform("Ip", times, values)
    sched = ScenarioSchedule({"Ip": wf})

    errors = sched.validate()
    assert len(errors) > 0
    assert "monotonic" in errors[0]

    times2 = np.array([0.0, 10.0, 20.0])
    values2 = np.array([0.0, -10.0, 20.0])
    wf2 = ScenarioWaveform("Ip", times2, values2)
    sched2 = ScenarioSchedule({"Ip": wf2})

    errors2 = sched2.validate()
    assert len(errors2) > 0
    assert "negative" in errors2[0]


def test_feedforward_controller():
    sched = iter_15ma_baseline()

    def dummy_feedback(x, x_ref, t, dt):
        return np.array([1.0, 2.0, 3.0])

    ctrl = FeedforwardController(sched, dummy_feedback)

    x = np.zeros(1)
    u = ctrl.step(x, 60.0, 0.1)

    assert np.isclose(u[0], 51.0)
    assert np.isclose(u[1], 17.0)


def test_scenario_optimizer():
    def plant(x, u, dt):
        return x + u * dt

    target = np.array([10.0, 15.0])

    opt = ScenarioOptimizer(plant, target, T_total=10.0, dt=1.0)

    sched = opt.optimize(n_iter=50)
    assert sched is not None
    assert sched.duration() == 10.0


def test_factory_scenarios():
    iter_sched = iter_15ma_baseline()
    assert len(iter_sched.validate()) == 0
    assert iter_sched.duration() == 480.0

    nstx_sched = nstx_u_1ma_standard()
    assert len(nstx_sched.validate()) == 0
    assert nstx_sched.duration() == 2.0
