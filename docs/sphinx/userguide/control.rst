==============================
Control Systems
==============================

SCPN-Fusion-Core provides a comprehensive suite of plasma control
algorithms, from classical PID to state-of-the-art model-predictive
control (MPC) and spiking neural network (SNN) controllers.  The
control-first architecture is the central innovation of the framework.

Design Philosophy
------------------

Unlike conventional fusion codes that bolt controllers onto physics
simulators, SCPN-Fusion-Core inverts the hierarchy: **control logic is
the primary artifact**.  Policies are expressed in a formally verifiable
Petri net formalism, compiled to spiking neural networks, and executed
against reduced-order plant models at 1 kHz+ control loop rates.

The control stack is layered:

1. **Classical controllers** (PID, analytic) for baseline comparison
2. **Model-predictive control** (MPC) for constrained optimisation
3. **Neuro-cybernetic controllers** (SNN) for sub-millisecond latency
4. **Digital twin + RL** for online policy adaptation
5. **Disruption prediction + mitigation** for fault management

Tokamak Flight Simulator
--------------------------

The ``tokamak_flight_sim`` module (``tokamak_flight_sim.py``) provides a
real-time flight simulator with:

- Actuator lag dynamics (first-order transfer functions)
- Plasma position, current, and shape control
- IsoFlux controller for shape maintenance
- Deterministic replay mode for reproducible testing

Usage::

    scpn-fusion flight

The flight simulator runs the plant model at configurable time steps
(default :math:`\Delta t = 1\,\text{ms}`) with real-time feedback from
the PID or MPC controller.

Model-Predictive Control
--------------------------

Two MPC implementations are provided:

**Gradient-Descent MPC** (``fusion_optimal_control.py``)
   Trajectory optimisation using gradient descent on a quadratic cost
   function with linear constraints.  Suitable for smooth reference
   tracking with bounded actuator inputs.

**State-of-the-Art MPC** (``fusion_sota_mpc.py``)
   A ``ModelPredictiveController`` class with neural surrogate plant
   models (``NeuralSurrogate``), supporting:

   - Configurable prediction horizon
   - Hard and soft constraints on actuator outputs
   - Real-time feasibility checks
   - Warm-starting from previous solutions

The MPC control law solves at each time step:

.. math::

   \min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1}
   \left[\|\mathbf{x}_k - \mathbf{x}_\text{ref}\|_Q^2 +
         \|\mathbf{u}_k\|_R^2\right]
   + \|\mathbf{x}_N - \mathbf{x}_\text{ref}\|_P^2

subject to :math:`\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)`,
actuator bounds, and safety constraints.

Disruption Prediction
-----------------------

The ``disruption_predictor`` module provides ML-based early warning for
plasma disruptions:

- **Deterministic scoring** using engineered features (locked-mode
  amplitude, plasma current derivative, radiated power fraction)
- **Optional Transformer model** for sequence-level disruption
  classification
- **Anomaly campaigns** for stress-testing the predictor against
  out-of-distribution scenarios
- **Checkpoint fallback** ensuring graceful degradation if the model
  file is corrupted or missing

The predictor outputs a disruption probability :math:`p_\text{disrupt}(t)`
and a time-to-disruption estimate :math:`\Delta t_\text{disrupt}` used
to trigger mitigation actions.

Shattered Pellet Injection (SPI)
----------------------------------

When the disruption predictor triggers an alarm, the ``spi_mitigation``
module simulates shattered pellet injection (Commaux et al., Nuclear
Fusion 56, 2016):

- Pellet shattering geometry and fragment distribution
- Impurity (:math:`Z_\text{eff}`) dilution computation
- Current quench (CQ) time constant estimation
- Radiative energy dissipation to prevent localised wall damage

Digital Twin
--------------

The ``tokamak_digital_twin`` module provides a real-time digital twin
with:

- Live telemetry ingestion (``digital_twin_ingest.py``)
- RL-trained MLP policy for adaptive control
- Chaos monkey fault injection for resilience testing
- Bit-flip resilience verification
- Deterministic replay for offline analysis

The digital twin maintains a shadow copy of the plasma state and
continuously compares its predictions against incoming measurements,
enabling:

- **Anomaly detection** -- divergence between twin and reality triggers
  alerts
- **Predictive control** -- the twin runs ahead of real-time to
  anticipate instabilities
- **What-if analysis** -- exploring alternative control strategies
  without risking the physical device

Integrated Control Room
-------------------------

The ``fusion_control_room`` module unifies all control systems into a
single simulation environment:

- Analytic or kernel-backed equilibrium computation
- Simultaneous PID, MPC, and SNN controller execution
- Real-time status dashboard (via Streamlit UI)
- CI-safe non-plot mode for automated testing

Neuro-Cybernetic Controller
------------------------------

The ``neuro_cybernetic_controller`` module implements a spiking neural
network (SNN) controller using leaky integrate-and-fire (LIF) neurons.
This is the execution backend for the SCPN neuro-symbolic compiler
(see :doc:`scpn_compiler`).

The LIF neuron dynamics follow:

.. math::

   \tau_m \frac{dV}{dt} = -(V - V_\text{rest}) + R_m I_\text{syn}

where :math:`\tau_m` is the membrane time constant, :math:`V` is the
membrane potential, :math:`V_\text{rest}` is the resting potential,
:math:`R_m` is the membrane resistance, and :math:`I_\text{syn}` is
the total synaptic input current.

Safety Interlocks (v3.5.0)
-----------------------------

The neuro-cybernetic lane now integrates a canonical inhibitor-arc
safety net (``scpn_fusion.scpn.safety_interlocks``).  Five safety places
map to five control transitions:

- ``thermal_limit``   -> inhibits ``heat_ramp``
- ``density_limit``   -> inhibits ``density_ramp``
- ``beta_limit``      -> inhibits ``power_ramp``
- ``current_limit``   -> inhibits ``current_ramp``
- ``vertical_limit``  -> inhibits ``position_move``

At runtime, ``SafetyInterlockRuntime`` derives binary safety tokens from
the state vector and computes deterministic transition enablement under
inhibitor semantics.  The controller summary now includes:

- ``safety_position_allow_rate``
- ``safety_interlock_trips``
- ``safety_contract_violations``

These metrics make control-lane safety behavior auditable in validation
campaigns and notebook demos.

Self-Organised Criticality Learning
--------------------------------------

The ``advanced_soc_fusion_learning`` module combines SOC sandpile
dynamics with Q-learning reinforcement learning for adaptive plasma
control.  The RL agent learns to operate the plasma near the criticality
boundary, maximising confinement while avoiding disruptions.

Analytic Solver
-----------------

The ``analytic_solver`` module provides closed-form solutions for
simplified equilibrium and transport problems, used primarily for:

- Controller unit testing (known analytical solutions)
- Rapid prototyping of control strategies
- Benchmarking numerical solvers against exact results

TORAX Hybrid Loop
-------------------

The ``torax_hybrid_loop`` module provides a coupling interface to the
TORAX integrated modelling code (JAX-based), enabling hybrid simulation
campaigns where SCPN-Fusion-Core controllers drive TORAX plant models.

Related Modules
-----------------

- :mod:`scpn_fusion.control.tokamak_flight_sim` -- flight simulator
- :mod:`scpn_fusion.control.fusion_optimal_control` -- gradient MPC
- :mod:`scpn_fusion.control.fusion_sota_mpc` -- neural surrogate MPC
- :mod:`scpn_fusion.control.disruption_predictor` -- ML disruption warning
- :mod:`scpn_fusion.control.spi_mitigation` -- shattered pellet injection
- :mod:`scpn_fusion.control.tokamak_digital_twin` -- digital twin
- :mod:`scpn_fusion.control.fusion_control_room` -- integrated control room
- :mod:`scpn_fusion.control.neuro_cybernetic_controller` -- SNN controller
- :mod:`scpn_fusion.scpn.safety_interlocks` -- inhibitor-arc safety runtime
- :mod:`scpn_fusion.control.advanced_soc_fusion_learning` -- SOC + RL
- :mod:`scpn_fusion.control.analytic_solver` -- closed-form solutions
- :mod:`scpn_fusion.control.director_interface` -- external director API
