.. -----------------------------------------------------------------------
   SCPN Fusion Core -- Fault-Tolerant Operations Tutorial
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

=============================================
Fault-Tolerant Control & Safe Reinforcement Learning
=============================================

A tokamak control system must continue operating safely even when
actuators degrade or fail.  This tutorial covers:

1. Fault Detection and Isolation (FDI) using innovation-based monitoring
2. Reconfigurable control that adapts to actuator failures
3. Constrained safe RL that respects hard safety limits

Prerequisites: :doc:`realtime_reconstruction`.


Part I: Fault Detection and Isolation
=======================================

The FDI monitor compares actual sensor readings against model predictions.
When the innovation (prediction error) exceeds a statistical threshold,
a fault is declared and the affected actuator is identified.

.. code-block:: python

   import numpy as np
   from scpn_fusion.control.fault_tolerant_control import (
       FDIMonitor,
       ReconfigurableController,
       FaultInjector,
   )

   # 4 actuators: 2 NBI sources, 1 ECCD, 1 gas valve
   n_actuators = 4
   n_sensors = 6

   fdi = FDIMonitor(
       n_actuators=n_actuators,
       n_sensors=n_sensors,
       innovation_threshold=3.0,   # 3-sigma detection
       window_size=20,             # sliding window for statistics
   )

   # Nominal operation: all innovations below threshold
   for step in range(50):
       y_meas = np.random.randn(n_sensors) * 0.1  # small noise
       y_pred = np.zeros(n_sensors)                 # perfect model
       fault = fdi.update(y_meas, y_pred)
       assert not fault.detected

   print("50 nominal steps: no fault detected")


Injecting a Fault
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   injector = FaultInjector()

   # Simulate NBI-1 degradation (actuator 0): output drops to 60%
   for step in range(30):
       y_meas = np.random.randn(n_sensors) * 0.1
       y_meas[0] += 2.0  # large innovation on sensor 0 (linked to NBI-1)
       y_pred = np.zeros(n_sensors)
       fault = fdi.update(y_meas, y_pred)

   print(f"Fault detected: {fault.detected}")
   print(f"Isolated actuator: {fault.actuator_index}")
   print(f"Confidence: {fault.confidence:.1%}")


Part II: Reconfigurable Control
==================================

When a fault is detected, the controller reconfigures: it removes the
faulty actuator from the control allocation and redistributes its
authority among the remaining healthy actuators.

.. code-block:: python

   # Nominal controller: PID with 4 actuator channels
   controller = ReconfigurableController(
       n_states=3,
       n_actuators=4,
       Kp=np.diag([1.0, 0.8, 0.5, 0.3]),
       Ki=np.diag([0.1, 0.1, 0.05, 0.02]),
   )

   # Normal step
   x = np.array([1.0, 0.5, 0.2])
   x_ref = np.array([1.0, 0.5, 0.0])
   u = controller.step(x, x_ref, dt=0.01)
   print(f"Nominal u: {u}")

   # Fault on actuator 0: reconfigure
   controller.isolate_actuator(0)
   u_reconfig = controller.step(x, x_ref, dt=0.01)
   print(f"Reconfigured u: {u_reconfig}")
   print(f"Actuator 0 output: {u_reconfig[0]:.6f} (should be ~0)")

   # Restore actuator 0
   controller.restore_actuator(0)


Full Fault-Tolerant Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt

   n_steps = 300
   dt = 0.01
   x_history = np.zeros((n_steps, 3))
   u_history = np.zeros((n_steps, 4))
   fault_history = np.zeros(n_steps)

   fdi.reset()
   controller.reset()

   x = np.array([2.0, 1.0, 0.5])
   x_ref = np.array([1.0, 0.5, 0.0])

   for step in range(n_steps):
       # Inject fault at t = 1 s (step 100)
       if step == 100:
           controller.isolate_actuator(0)
           fault_history[step:] = 1

       u = controller.step(x, x_ref, dt)

       # Simple plant: x_{k+1} = x_k + B @ u * dt
       B = np.array([[0.5, 0.3, 0.0, 0.0],
                      [0.0, 0.4, 0.3, 0.0],
                      [0.0, 0.0, 0.2, 0.5]])
       x = x + B @ u * dt + np.random.randn(3) * 0.01

       x_history[step] = x
       u_history[step] = u

   fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

   t = np.arange(n_steps) * dt

   for i in range(3):
       axes[0].plot(t, x_history[:, i], label=f"x[{i}]")
   axes[0].axvline(1.0, color="red", ls="--", alpha=0.5, label="Fault")
   axes[0].set_ylabel("State")
   axes[0].legend()

   for i in range(4):
       axes[1].plot(t, u_history[:, i], label=f"u[{i}]")
   axes[1].axvline(1.0, color="red", ls="--", alpha=0.5)
   axes[1].set_ylabel("Actuator")
   axes[1].legend()

   axes[2].fill_between(t, fault_history, alpha=0.3, color="red")
   axes[2].set_ylabel("Fault Active")
   axes[2].set_xlabel("Time [s]")

   plt.suptitle("Fault-Tolerant Control: NBI-1 Failure at t = 1 s")
   plt.tight_layout()
   plt.show()


Part III: Safe Reinforcement Learning
========================================

For advanced control scenarios where the plant model is uncertain,
reinforcement learning can discover optimal policies.  However, RL
must respect hard safety constraints (thermal limits, density limits,
vertical stability margins).

The **Lagrangian PPO** algorithm augments the standard PPO objective
with constraint penalties:

.. math::

   \max_\theta \min_\lambda \;\; \mathbb{E}\left[\sum_t r_t\right]
   - \sum_i \lambda_i \left(\mathbb{E}\left[\sum_t c_{i,t}\right] - d_i\right)

where :math:`c_{i,t}` are constraint costs and :math:`d_i` are
tolerance budgets.

.. code-block:: python

   from scpn_fusion.control.safe_rl_controller import (
       LagrangianPPO,
       ConstrainedGymTokamakEnv,
   )

   # Constrained environment with safety limits
   env = ConstrainedGymTokamakEnv(
       max_beta=0.05,          # beta limit
       max_q_edge_violation=0.1,  # q-edge must stay > 2
       max_displacement=0.1,    # vertical displacement [m]
   )

   agent = LagrangianPPO(
       obs_dim=env.observation_space.shape[0],
       act_dim=env.action_space.shape[0],
       constraint_dims=3,
       lr_policy=3e-4,
       lr_lambda=1e-3,
   )

   # Training loop (abbreviated)
   for episode in range(10):
       obs, _ = env.reset()
       total_reward = 0
       total_cost = np.zeros(3)

       for step in range(200):
           action = agent.act(obs)
           obs, reward, terminated, truncated, info = env.step(action)
           total_reward += reward
           total_cost += info.get("costs", np.zeros(3))

           if terminated or truncated:
               break

       print(f"Episode {episode}: reward={total_reward:.1f}, "
             f"constraint violations={total_cost.sum():.2f}")

.. note::

   Full RL training requires ``gymnasium`` (optional dependency).
   Install with ``pip install scpn-fusion[rl]``.


Design Philosophy
-------------------

**Defence in depth:**

1. **Layer 1 — Physics model:** The nominal controller uses the plant
   model (equilibrium + transport + stability) to compute optimal
   actuator commands.

2. **Layer 2 — FDI:** Continuous monitoring detects actuator degradation
   within 20--50 ms and triggers reconfiguration.

3. **Layer 3 — Reconfigurable control:** The surviving actuators are
   re-allocated to maintain stability (possibly with degraded performance).

4. **Layer 4 — Safe RL:** For scenarios outside the model's validity,
   RL policies with hard constraint enforcement provide a safety net.

5. **Layer 5 — Disruption mitigation:** If all else fails, SPI (shattered
   pellet injection) is triggered to safely terminate the discharge.


Related Modules
-----------------

- :mod:`scpn_fusion.control.fault_tolerant_control` -- FDI + reconfiguration
- :mod:`scpn_fusion.control.safe_rl_controller` -- Lagrangian PPO
- :mod:`scpn_fusion.control.spi_mitigation` -- disruption mitigation
- :mod:`scpn_fusion.control.disruption_predictor` -- disruption detection
- :doc:`realtime_reconstruction` -- EFIT + shape + vertical control
