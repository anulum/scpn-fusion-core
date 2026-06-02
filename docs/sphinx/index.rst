============================
SCPN-Fusion-Core
============================

*Neuro-symbolic control framework for tokamak fusion reactors*

.. rubric:: Version |release|

SCPN-Fusion-Core is a dual-language (Python + Rust) open-source framework
that treats AI and digital twins as first-class architectural primitives
for tokamak plasma control.  It compiles plasma control logic -- expressed
as stochastic Petri nets -- into spiking neural network controllers that
run at sub-millisecond latency, backed by a Grad-Shafranov equilibrium
solver, 1.5D radial transport, and AI surrogates for turbulence, disruption
prediction, and real-time digital twins.

**What makes it different:** Most fusion codes are physics-first (solve
equations, then bolt on control).  SCPN-Fusion-Core is **control-first**
-- it provides a contract-checked neuro-symbolic compilation pipeline
where plasma control policies are expressed as Petri nets, compiled to
stochastic LIF neurons, and executed against physics-informed plant models.

.. note::

   **Honest scope:** This is not a replacement for TRANSP, JINTRAC, GENE,
   CGYRO, GS2, DREAM, Aurora, STRAHL, or EFIT.  The native solver stack now
   exposes nonlinear 5D gyrokinetic state contracts, local electromagnetic
   diagnostics, decomposition contracts, and fail-closed full-fidelity
   benchmark gates, but production parity still requires same-case external
   reference outputs and quantitative thresholds.  Treat the current public
   evidence as a control-algorithm, native-kernel, and validation-framework
   release, not as a completed production turbulence or reconstruction code.

Reader Orientation
------------------

SCPN-Fusion-Core is best read as three connected systems: a control compiler, a
native physics-kernel laboratory, and a reproducible validation/reporting
surface.  The public reports distinguish validated local contracts from blocked
full-fidelity parity rows.  Use the project overview, onboarding guide, and
benchmark taxonomy before treating any number as a production claim.

Key Features
------------

- **Neuro-symbolic compiler** -- Petri net to SNN compilation with formal
  verification (37 hardening tasks)
- **Safety interlocks** -- inhibitor-arc hard-stop channels with
  contract-proof checks for thermal/density/beta/current/vertical limits
- **Grad-Shafranov equilibrium** -- Picard + Red-Black SOR or multigrid
  V-cycle, validated on 8 SPARC GEQDSK files
- **1.5D radial transport** -- coupled energy/particle transport with
  IPB98(y,2) confinement scaling
- **AI surrogates** -- FNO turbulence, neural equilibrium, neural transport
  MLP, ML disruption predictor
- **Digital twin** -- real-time twin with RL-trained MLP policy and chaos
  monkey fault injection
- **Rust acceleration** -- 11-crate Rust workspace providing 10--50x
  speedups with pure-Python fallback
- **Real data validation** -- SPARC GEQDSK, ITER 15 MA baseline, ITPA
  H-mode confinement database
- **Fail-closed full-fidelity campaign** -- explicit GENE/CGYRO/GS2,
  DREAM, Aurora/STRAHL, FreeGS, electromagnetic, and decomposition blockers
  tracked as reports rather than promoted to passes
- **Graceful degradation** -- every module works without Rust, without
  SC-NeuroCore, without GPU

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

Public Documentation Hubs
-------------------------

- `Project overview <../PROJECT_OVERVIEW.md>`_
- `Onboarding guide <../ONBOARDING.md>`_
- `API overview <../API_OVERVIEW.md>`_
- `Applications and market context <../APPLICATIONS_AND_MARKET.md>`_
- `Benchmark taxonomy <../BENCHMARKS.md>`_

.. toctree::
   :maxdepth: 2
   :caption: Learning Path

   learning/plasma_physics_primer
   learning/fusion_engineering_101
   learning/first_simulation
   learning/tokamak_physics_textbook

.. toctree::
   :maxdepth: 2
   :caption: Advanced Tutorials

   tutorials/current_profile_evolution
   tutorials/mhd_instabilities
   tutorials/edge_sol_physics
   tutorials/realtime_reconstruction
   tutorials/fault_tolerant_operations
   tutorials/scenario_design

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   userguide/equilibrium
   userguide/transport
   userguide/control
   userguide/hil
   userguide/nuclear
   userguide/diagnostics
   userguide/scpn_compiler
   userguide/hpc
   userguide/validation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/control
   api/nuclear
   api/diagnostics
   api/engineering
   api/scpn
   api/hpc
   api/io

.. toctree::
   :maxdepth: 2
   :caption: Example Notebooks

   notebooks

.. toctree::
   :maxdepth: 1
   :caption: Reference

   workflows

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   changelog
   license

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
