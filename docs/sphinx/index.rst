.. -----------------------------------------------------------------------
   SCPN Fusion Core -- Documentation Root
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

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
-- it provides a formally verified neuro-symbolic compilation pipeline
where plasma control policies are expressed as Petri nets, compiled to
stochastic LIF neurons, and executed against physics-informed plant models.

.. note::

   **Honest scope:** This is not a replacement for TRANSP, JINTRAC, or
   GENE.  It does not solve 5D gyrokinetics or full 3D MHD.  It is a
   **control-algorithm development and surrogate-modelling framework**
   with enough physics fidelity to validate reactor control strategies
   against real equilibrium data (8 SPARC GEQDSK files, ITPA H-mode
   database).

Key Features
------------

- **Neuro-symbolic compiler** -- Petri net to SNN compilation with formal
  verification (37 hardening tasks)
- **Grad-Shafranov equilibrium** -- Picard + Red-Black SOR or multigrid
  V-cycle, validated on 8 SPARC GEQDSK files
- **1.5D radial transport** -- coupled energy/particle transport with
  IPB98(y,2) confinement scaling
- **AI surrogates** -- FNO turbulence, neural equilibrium, neural transport
  MLP, ML disruption predictor
- **Digital twin** -- real-time twin with RL-trained MLP policy and chaos
  monkey fault injection
- **Rust acceleration** -- 10-crate Rust workspace providing 10--50x
  speedups with pure-Python fallback
- **Real data validation** -- SPARC GEQDSK, ITER 15 MA baseline, ITPA
  H-mode confinement database
- **Graceful degradation** -- every module works without Rust, without
  SC-NeuroCore, without GPU

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

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
   :maxdepth: 1
   :caption: Reference

   workflows
   gpu_roadmap

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
