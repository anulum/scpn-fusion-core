.. -----------------------------------------------------------------------
   SCPN Fusion Core -- Scenario Design Tutorial
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

=============================================
Scenario Design & Gain-Scheduled Control
=============================================

A tokamak discharge follows a scripted sequence of phases: current ramp-up,
heating ramp, L-H transition, flat-top burn, ramp-down, and termination.
Each phase has different control requirements.  This tutorial covers:

1. Defining scenario waveforms for actuator trajectories
2. Using the ITER 15 MA baseline and NSTX-U factory scenarios
3. Designing a feedforward + feedback controller
4. Gain scheduling across operating regimes with bumpless transfer

Prerequisites: :doc:`realtime_reconstruction`,
:doc:`../learning/fusion_engineering_101` (section 7).


Part I: Scenario Waveforms
=============================

A scenario waveform defines how a plant parameter (plasma current, NBI
power, gas fuelling rate, etc.) evolves through the discharge.

.. code-block:: python

   import numpy as np
   from scpn_fusion.control.scenario_scheduler import (
       ScenarioWaveform,
       ScenarioSchedule,
       FeedforwardController,
       iter_15ma_baseline,
       nstx_u_1ma_standard,
   )

   # Custom waveform: plasma current ramp
   times = np.array([0, 20, 60, 400, 450, 480])
   values = np.array([0, 5, 15, 15, 5, 0])  # MA
   Ip_waveform = ScenarioWaveform("Ip", times, values)

   # Evaluate at any time
   print(f"Ip(30 s) = {Ip_waveform(30.0):.1f} MA")
   print(f"Ip(200 s) = {Ip_waveform(200.0):.1f} MA")
   print(f"Ip(460 s) = {Ip_waveform(460.0):.1f} MA")


ITER 15 MA Baseline Scenario
-------------------------------

.. code-block:: python

   sched = iter_15ma_baseline()

   # Validate: no negative values, monotonic times
   errors = sched.validate()
   if errors:
       for e in errors:
           print(f"  ERROR: {e}")
   else:
       print("Scenario valid")

   print(f"Total duration: {sched.duration():.0f} s")

   # Evaluate at flat-top
   vals = sched.evaluate(200.0)
   print(f"At t = 200 s:")
   for name, val in vals.items():
       print(f"  {name} = {val:.1f}")


NSTX-U 1 MA Scenario
-----------------------

.. code-block:: python

   nstx = nstx_u_1ma_standard()
   print(f"NSTX-U duration: {nstx.duration():.1f} s")

   vals = nstx.evaluate(0.8)
   print(f"At t = 0.8 s:")
   for name, val in vals.items():
       print(f"  {name} = {val:.2f}")


Part II: Feedforward + Feedback
==================================

A scenario controller combines a feedforward trajectory (the scheduled
waveforms) with a feedback correction that compensates for disturbances:

.. math::

   u(t) = u_\text{ff}(t) + u_\text{fb}(x, x_\text{ref}, t)

.. code-block:: python

   sched = iter_15ma_baseline()

   # Simple PID feedback
   def pid_feedback(x, x_ref, t, dt):
       Kp = np.array([2.0, 1.0, 0.5])
       error = x_ref - x
       return Kp * error

   ctrl = FeedforwardController(sched, pid_feedback)

   # Simulate: state tracks the reference with disturbances
   x = np.zeros(1)
   trajectory = []

   for t_sec in np.arange(0, 480, 1.0):
       u = ctrl.step(x, t_sec, dt=1.0)
       trajectory.append((t_sec, u.copy()))

   print(f"Control commands at t=0: {trajectory[0][1]}")
   print(f"Control commands at t=200: {trajectory[200][1]}")


Part III: Gain-Scheduled Control
===================================

A gain-scheduled controller maintains separate PID gains for each
operating regime and switches between them as the plasma transitions.

Operating Regimes
-------------------

.. code-block:: python

   from scpn_fusion.control.gain_scheduled_controller import (
       GainScheduledController,
       OperatingRegime,
       RegimeController,
       RegimeDetector,
       iter_baseline_schedule,
   )

   # The regime detector uses confinement time, disruption probability,
   # and current ramp rate to classify the operating regime
   detector = RegimeDetector()

   state = np.zeros(2)

   # During ramp-up (large dI/dt)
   dstate = np.array([0.5, 0.0])
   for _ in range(10):
       regime = detector.detect(state, dstate, tau_E=0.5, p_disrupt=0.0)
   print(f"Ramp-up regime: {regime.name}")

   # During L-mode flat-top (small dI/dt, low tau_E)
   dstate = np.array([0.01, 0.0])
   for _ in range(10):
       regime = detector.detect(state, dstate, tau_E=1.0, p_disrupt=0.0)
   print(f"L-mode regime: {regime.name}")

   # During H-mode (high tau_E)
   for _ in range(10):
       regime = detector.detect(state, dstate, tau_E=2.5, p_disrupt=0.0)
   print(f"H-mode regime: {regime.name}")

   # During disruption approach (high p_disrupt)
   for _ in range(10):
       regime = detector.detect(state, dstate, tau_E=2.5, p_disrupt=0.85)
   print(f"Disruption regime: {regime.name}")


Bumpless Transfer
-------------------

When switching regimes, the gain-scheduled controller performs
**bumpless transfer** — a linear interpolation over a configurable
interval that prevents actuator jumps:

.. code-block:: python

   controllers = {
       OperatingRegime.RAMP_UP: RegimeController(
           OperatingRegime.RAMP_UP,
           Kp=np.array([3.0]), Ki=np.array([0.5]), Kd=np.zeros(1),
           x_ref=np.array([15.0]),  # target: 15 MA
           constraints={"u_max": 100.0},
       ),
       OperatingRegime.L_MODE_FLAT: RegimeController(
           OperatingRegime.L_MODE_FLAT,
           Kp=np.array([1.0]), Ki=np.array([0.2]), Kd=np.zeros(1),
           x_ref=np.array([15.0]),
           constraints={"u_max": 50.0},
       ),
       OperatingRegime.H_MODE_FLAT: RegimeController(
           OperatingRegime.H_MODE_FLAT,
           Kp=np.array([0.5]), Ki=np.array([0.1]), Kd=np.array([0.05]),
           x_ref=np.array([15.0]),
           constraints={"u_max": 30.0},
       ),
   }

   gsc = GainScheduledController(controllers, transfer_time=0.5)

   # Simulate regime transition
   x = np.array([10.0])  # current state: 10 MA

   u_ramp = gsc.step(x, t=5.0, dt=0.1, regime=OperatingRegime.RAMP_UP)
   print(f"Ramp-up command: {u_ramp[0]:.1f}")

   # Transition to L-mode
   u_trans = gsc.step(x, t=60.0, dt=0.1, regime=OperatingRegime.L_MODE_FLAT)
   print(f"During transfer: {u_trans[0]:.1f}")

   # Fully in L-mode
   for _ in range(10):
       u_lmode = gsc.step(x, t=70.0, dt=0.1, regime=OperatingRegime.L_MODE_FLAT)
   print(f"L-mode command: {u_lmode[0]:.1f}")


Full Scenario Simulation
---------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   sched = iter_baseline_schedule()

   # Simulate 480 s ITER discharge
   dt = 1.0
   t_range = np.arange(0, 480, dt)
   x = np.array([0.0])  # Ip = 0 at start

   Ip_history = []
   regime_history = []

   for t in t_range:
       vals = sched.evaluate(t)
       x_ref = np.array([vals["Ip"]])

       # Detect regime
       dstate = np.gradient(x) if len(Ip_history) > 1 else np.array([0.5])
       tau_E = 1.0 + (vals.get("P_NBI", 0) / 33.0) * 1.5
       regime = detector.detect(x, dstate, tau_E=tau_E, p_disrupt=0.0)

       u = gsc.step(x, t, dt, regime=regime)

       # Simple plant model
       x = x + 0.1 * u * dt
       x = np.clip(x, 0, 20)

       Ip_history.append(x[0])
       regime_history.append(regime.value)

   fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

   # Plot scheduled vs. achieved Ip
   scheduled_Ip = [sched.evaluate(t)["Ip"] for t in t_range]
   axes[0].plot(t_range, scheduled_Ip, "b--", label="Scheduled")
   axes[0].plot(t_range, Ip_history, "r-", label="Achieved")
   axes[0].set_ylabel("Ip [MA]")
   axes[0].legend()
   axes[0].set_title("ITER 15 MA Baseline — Gain-Scheduled Control")

   axes[1].plot(t_range, regime_history)
   axes[1].set_ylabel("Regime")
   axes[1].set_xlabel("Time [s]")
   axes[1].set_yticks([0, 1, 2, 3, 4])
   axes[1].set_yticklabels(["Ramp-Up", "L-Flat", "H-Flat", "Ramp-Down", "Disruption"])

   plt.tight_layout()
   plt.show()


Design Tips
------------

**Waveform design:**

- NBI power ramp should lead the current ramp by 2--5 s to ensure
  adequate heating during the Ohmic-to-auxiliary transition.
- Gas fuelling should track the Greenwald fraction
  :math:`f_G = \bar{n} / n_G < 0.85`.
- ECCD power should be staged: start with q-profile shaping during
  ramp-up, switch to NTM suppression duty during flat-top.

**Regime transitions:**

- The L-H transition is the most sensitive point: a too-aggressive
  power ramp can trigger ELMs before the pedestal is established.
- Transfer time of 0.3--0.5 s gives smooth actuator transitions
  without losing tracking performance.
- The H-L back-transition during ramp-down requires reducing NBI
  power gradually to avoid a hard mode lock.


Related Modules
-----------------

- :mod:`scpn_fusion.control.scenario_scheduler` -- waveforms, schedules
- :mod:`scpn_fusion.control.gain_scheduled_controller` -- gain scheduling
- :mod:`scpn_fusion.control.fusion_sota_mpc` -- MPC for fine control
- :doc:`realtime_reconstruction` -- EFIT provides state feedback
- :doc:`fault_tolerant_operations` -- when actuators fail mid-scenario
