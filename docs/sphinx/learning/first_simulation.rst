.. -----------------------------------------------------------------------
   SCPN Fusion Core -- First Simulation
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

=====================
Your First Simulation
=====================

This tutorial uses the maintained command-line and public API paths. By the end
you will have run a compact equilibrium/controller example, inspected a
redistributable GEQDSK reference, evaluated a confinement scaling law, and read
the repository's fail-closed evidence status.

Prerequisite: :doc:`fusion_engineering_101` for physics context.

Installation
------------

Install the Python package and its independently distributed native extension::

   python -m venv .venv
   . .venv/bin/activate
   python -m pip install scpn-fusion scpn-fusion-rs

For a source checkout, use the pinned review path in :doc:`../installation`.
If a compatible ``scpn-fusion-rs`` wheel is unavailable for a target platform,
the Rust extension remains optional and can be built from ``scpn-fusion-rs/``
with Maturin.

Verify what actually loaded::

   python -c "import scpn_fusion; print(scpn_fusion.__version__)"
   python -c "import scpn_fusion_rs; print(scpn_fusion_rs.__version__)"
   python -c "from scpn_fusion.core import RUST_BACKEND; print(RUST_BACKEND)"

The backend flag reports availability, not a speedup. Performance claims need
an equivalent workload and a committed hardware-specific artifact.

Step 1: Run the Maintained Minimal Example
-------------------------------------------

From a source checkout::

   python examples/minimal.py --grid 17 --equilibrium-iters 4

The command prints one compact equilibrium status, one deterministic
neuro-symbolic controller action, and a JSON summary. A short four-iteration
smoke run may report ``converged=false``; the tutorial checks wiring and data
flow, not production convergence or parity.

To isolate either lane::

   python examples/minimal.py --skip-controller --grid 17 --equilibrium-iters 8
   python examples/minimal.py --skip-equilibrium --seed 42

Step 2: Run the Equilibrium CLI Smoke Path
-------------------------------------------

The installed CLI exposes the maintained equilibrium demo::

   scpn-fusion kernel

Use this as an installation smoke test. Do not quote its output as a reference
solver comparison; measured and parity claims live in ``RESULTS.md``,
``docs/BENCHMARKS.md``, and ``validation/reports/``.

Step 3: Inspect a Licensed Reference Equilibrium
-------------------------------------------------

The public core facade exports the GEQDSK reader::

   from scpn_fusion.core import read_geqdsk

   eq = read_geqdsk("validation/reference_data/sparc/lmode_vv.geqdsk")
   print(eq.psirz.shape)
   print(f"R_axis = {eq.rmaxis:.3f} m")
   print(f"Z_axis = {eq.zmaxis:.3f} m")
   print(f"B_T = {eq.bcentr:.2f} T")
   print(f"I_p = {eq.current / 1e6:.2f} MA")

The bundled SPARC files are redistributed under MIT terms recorded in
``validation/reference_data/sparc/LICENSE`` and ``REUSE.toml``. Loading a file
proves parser/runtime behavior only; see the linked validation reports before
claiming equilibrium accuracy.

Step 4: Evaluate a Scaling Law With Explicit Units
---------------------------------------------------

The IPB98(y,2) helper accepts plasma current in MA, field in T, density in
``10^19 m^-3``, loss power in MW, and geometry in metres::

   from scpn_fusion.core import ipb98y2_tau_e

   tau_e_s = ipb98y2_tau_e(
       Ip=15.0,
       BT=5.3,
       ne19=10.0,
       Ploss=50.0,
       R=6.2,
       kappa=1.7,
       epsilon=2.0 / 6.2,
       M=2.5,
       warn_if_extrapolated=True,
   )
   print(f"IPB98(y,2) tau_E = {tau_e_s:.3f} s")

This is an empirical scaling evaluation, not a time-dependent transport solve.
Use :doc:`../userguide/transport` for transport models and their evidence
boundaries.

Step 5: Refresh the Evidence Ledger
------------------------------------

From a source checkout::

   scpn-fusion repro --full

The command refreshes the checksummed public evidence wrapper. The expected
top-level state can remain ``not_full_fidelity`` even when every local contract
behaves correctly; external same-case, licensing, hardware, or threshold gaps
keep individual rows blocked by design.

Step 6: Choose the Next Tutorial
---------------------------------

- **Neuro-symbolic control:**
  ``examples/neuro_symbolic_control_demo_v2.ipynb``
- **Equilibrium:** ``examples/03_grad_shafranov_equilibrium.ipynb`` and
  :doc:`../userguide/equilibrium`
- **Transport:** ``examples/07_multi_ion_transport.ipynb`` and
  :doc:`../userguide/transport`
- **MHD:** ``examples/08_mhd_stability.ipynb`` and
  :doc:`../tutorials/mhd_instabilities`
- **Validation:** ``examples/05_validation_against_experiments.ipynb`` and
  :doc:`../userguide/validation`
- **Control systems:** :doc:`../tutorials/realtime_reconstruction`,
  :doc:`../tutorials/fault_tolerant_operations`, and
  :doc:`../tutorials/scenario_design`

Notebook output is educational unless it links to a tracked report whose gate
is explicitly accepted. See ``docs/notebooks/README.md`` for the full catalogue
and claim boundaries.
