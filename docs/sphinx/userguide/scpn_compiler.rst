=======================================
Neuro-Symbolic Compiler (SCPN)
=======================================

The SCPN neuro-symbolic compiler is the core innovation that
distinguishes SCPN-Fusion-Core from conventional fusion simulation
codes.  It compiles plasma control policies -- expressed as stochastic
Petri nets -- into spiking neural network controllers that execute at
sub-millisecond latency with formal verification guarantees.

Compilation Pipeline
---------------------

The pipeline transforms a high-level control specification into a
deployable neural controller in five stages:

.. code-block:: text

   Petri Net (places + transitions + contracts)
       |
       v  compiler.py -- structure-preserving mapping
   Stochastic LIF Network (neurons + synapses + thresholds)
       |
       v  controller.py -- closed-loop execution
   Real-Time Plasma Control (sub-ms latency, deterministic replay)
       |
       v  artifact.py -- versioned, signed compilation artifact
   Deployment Package (JSON + schema version + git SHA)

Stage 1: Petri Net Definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plasma control logic is expressed as a place/transition Petri net using
the ``StochasticPetriNet`` data structure (``structure.py``).

- **Places** represent system states (e.g. "plasma_heating",
  "current_ramp", "disruption_imminent")
- **Transitions** represent control actions (e.g. "increase_heating_power",
  "inject_pellet", "trigger_SPI")
- **Tokens** represent activation levels (continuous or discrete)
- **Stochastic weights** encode firing probabilities

The Petri net formalism provides:

- **Compositional semantics** -- control policies can be built from
  smaller verified sub-policies
- **Formal analysis** -- boundedness, liveness, and reachability are
  decidable for bounded Petri nets
- **Visual clarity** -- the net graph directly represents the control
  flow

Stage 2: Formal Contracts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``contracts`` module (``contracts.py``) defines formal verification
contracts that compiled artifacts must satisfy:

- **Boundedness** -- no place can accumulate unbounded tokens
- **Liveness** -- every transition can eventually fire
- **Reachability** -- target markings are reachable from the initial
  marking
- **Invariant preservation** -- place/transition invariants from the
  source Petri net are preserved in the compiled SNN

These contracts are checked both at compile time and at runtime during
controller execution.

Stage 3: Compilation
^^^^^^^^^^^^^^^^^^^^^^

The ``FusionCompiler`` class (``compiler.py``) performs a
structure-preserving mapping from the Petri net to a spiking neural
network:

- Each **transition** maps to one ``StochasticLIFNeuron`` (a pure
  threshold comparator)
- **Arc weights** map to synaptic weights
- **Token counts** map to pre-synaptic input currents
- **Firing thresholds** are derived from the Petri net marking conditions

When `SC-NeuroCore <https://github.com/anulum/sc-neurocore>`_ is
installed, the compiler uses hardware-accurate stochastic LIF neurons
and Bernoulli bitstream encoding (``uint64`` packed weight bitstreams
for AND+popcount forward pass).  Without it, the compiler falls back
to NumPy float computation with identical API.

The leaky integrate-and-fire neuron model:

.. math::

   \tau_m \frac{dV_j}{dt} = -(V_j - V_\text{rest})
   + R_m \sum_i w_{ij} \, s_i(t)

where :math:`s_i(t)` is the spike train from pre-synaptic neuron
:math:`i`, :math:`w_{ij}` is the synaptic weight, and the neuron fires
when :math:`V_j > V_\text{thresh}`, resetting to :math:`V_\text{rest}`.

Stage 4: Execution
^^^^^^^^^^^^^^^^^^^^^

The ``SCPNController`` class (``controller.py``) executes the compiled
SNN in closed loop against the physics plant model:

- Reads sensor inputs (plasma state)
- Propagates activity through the SNN
- Produces actuator commands (heating power, coil currents, gas puff)
- Records execution trace for deterministic replay

The controller supports:

- **Sub-millisecond latency** -- the SNN forward pass is O(N) in the
  number of neurons, typically < 100 microseconds
- **Deterministic replay** -- given the same input sequence, the
  controller produces bit-identical outputs (37 dedicated hardening
  tasks in the H5 wave)
- **Fault injection** -- configurable fault modes for resilience testing

Stage 5: Artifact Export
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``artifact`` module (``artifact.py``) produces versioned compilation
artifacts containing:

- Compiled SNN weights and topology
- Source Petri net specification
- Formal verification results (contracts satisfied/violated)
- Package version, schema version, and git SHA stamps
- Timestamp and build environment metadata

Artifacts are serialised as JSON and can be deployed to SC-NeuroCore
hardware targets, NumPy simulation, or future neuromorphic silicon.

Hardware Targets
-----------------

The same Petri net compiles to multiple execution backends:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Backend
     - Description
     - Latency
   * - NumPy
     - Float-path simulation
     - ~100 microseconds
   * - SC-NeuroCore
     - FPGA-accurate stochastic neurons
     - ~10 microseconds
   * - Neuromorphic silicon
     - Future hardware target
     - Sub-microsecond (projected)

Why Neuro-Symbolic Control?
-----------------------------

Most fusion control systems bolt a PID or MPC controller onto a physics
code.  This approach has three fundamental limitations:

1. **No formal verification** -- PID/MPC parameters are tuned
   empirically; there is no proof that the controller satisfies safety
   invariants under all reachable states.

2. **High latency** -- MPC requires solving an optimisation problem at
   each time step, typically taking milliseconds to seconds. SNN
   controllers execute a single forward pass in microseconds.

3. **No hardware path** -- classical controllers cannot be directly
   compiled to neuromorphic hardware for ultra-low-latency deployment.

The SCPN approach addresses all three: Petri net invariants provide
formal safety guarantees, SNN execution provides sub-millisecond
latency, and the compilation pipeline targets multiple hardware
backends.

Related Modules
-----------------

- :mod:`scpn_fusion.scpn.structure` -- Petri net data structures
- :mod:`scpn_fusion.scpn.contracts` -- formal verification contracts
- :mod:`scpn_fusion.scpn.compiler` -- Petri net to SNN compiler
- :mod:`scpn_fusion.scpn.controller` -- SNN-driven plasma controller
- :mod:`scpn_fusion.scpn.artifact` -- compilation artifact storage
