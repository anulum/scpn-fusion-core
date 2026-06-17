===========
Quick Start
===========

This guide walks through the essential operations: solving an equilibrium,
running a validation suite, using the Rust accelerated kernel, and launching
the neuro-symbolic compiler.  It is intended to give a new reader a useful
first run, not to establish production parity by itself.

Choose the Right First Run
--------------------------

Use ``scpn-fusion kernel`` when you want a fast equilibrium smoke test. Use
``python validation/full_fidelity_end_to_end_campaign.py`` when you want the
current evidence-bound maturity report. Use the notebooks when you want a
learning path, not production parity evidence.

The shortest sensible path is:

1. run a smoke command,
2. inspect the generated or tracked report,
3. follow the linked evidence surface before quoting a result.


Solve a Grad-Shafranov Equilibrium
-----------------------------------

The primary entry point is ``scpn-fusion``, which dispatches to
the appropriate simulation mode::

    scpn-fusion kernel

This invokes the Picard-iteration Grad-Shafranov solver on a default
ITER-class configuration.  The solver iterates the nonlinear elliptic
equation

.. math::

   \Delta^{*}\psi = -\mu_0 R^2 p'(\psi) - F(\psi) F'(\psi)

where :math:`\Delta^{*}` is the toroidal elliptic operator, :math:`p(\psi)`
is the plasma pressure, and :math:`F(\psi) = R B_\phi` is the poloidal
current function.

Expected output includes the converged magnetic axis position
:math:`(R_\text{axis}, Z_\text{axis})`, the safety factor on axis
:math:`q_0`, and the total plasma current :math:`I_p`.

Read a SPARC GEQDSK File
-------------------------

The ``eqdsk`` module reads standard tokamak equilibrium files::

    from scpn_fusion.core.eqdsk import read_geqdsk

    eq = read_geqdsk("validation/reference_data/sparc/lmode_vv.geqdsk")
    print(f"B_T = {eq.bcentr:.1f} T")
    print(f"I_p = {eq.current / 1e6:.1f} MA")
    print(f"R_axis = {eq.rmaxis:.3f} m")
    print(f"Z_axis = {eq.zmaxis:.3f} m")

Eight SPARC GEQDSK files are included in ``validation/reference_data/sparc/``
from the `CFS SPARCPublic repository <https://github.com/cfs-energy/SPARCPublic>`_
(MIT license).

Run the Validation Suite
------------------------

For the current full-fidelity production-parity snapshot, run the fail-closed
campaign first::

    python validation/full_fidelity_end_to_end_campaign.py

This report is intentionally conservative.  Local contracts can pass while the
overall campaign remains ``not_full_fidelity`` until same-case public outputs
from external reference solvers are available.

Generate an RMSE dashboard report comparing computed confinement times
against the ITPA H-mode database and SPARC equilibrium topology::

    python validation/rmse_dashboard.py

This writes a JSON and Markdown report to ``validation/reports/``.  The
validation checks include:

- IPB98(y,2) confinement time prediction vs. measured values from JET,
  DIII-D, ASDEX Upgrade, and Alcator C-Mod
- Equilibrium topology checks on 8 SPARC GEQDSK files (axis position,
  safety factor monotonicity, GS operator sign)
- Beta normalised surrogate accuracy

Run a Compact Reactor Search
-----------------------------

The ``optimizer`` mode performs a multi-objective design-space exploration
to find the smallest tokamak that achieves :math:`Q \geq 10` ignition::

    scpn-fusion optimizer

The optimizer sweeps major/minor radius, elongation, triangularity,
magnetic field strength, plasma current, and heating power allocation
subject to the Greenwald density limit, beta limit, and Lawson criterion.
The result is the MVR-0.96 (Minimum Viable Reactor) design point with
:math:`R = 0.965\,\text{m}`, :math:`A = 2.0`, :math:`P_\text{fus} = 5.3\,\text{MW}`.

Launch the Tokamak Flight Simulator
------------------------------------

The flight simulator provides a real-time control loop with actuator
lag dynamics::

    scpn-fusion flight

This mode demonstrates PID and MPC controllers maintaining plasma
position, current, and shape against perturbations.

Compile a Petri Net to an SNN Controller
-----------------------------------------

The neuro-symbolic compiler is the core innovation of SCPN-Fusion-Core.
A simple example::

    scpn-fusion neuro-control

This compiles a plasma control policy (expressed as a stochastic Petri
net) into a population of leaky integrate-and-fire (LIF) neurons and
executes it in closed loop against the physics plant model.

For a detailed walkthrough, see the tutorial notebook
``examples/02_neuro_symbolic_compiler.ipynb``.

Generate a 3D Flux-Surface Mesh
---------------------------------

Generate an OBJ mesh of the last closed flux surface from a validated
ITER configuration::

    python examples/run_3d_flux_quickstart.py --toroidal 24 --poloidal 24

To include a PNG preview::

    python examples/run_3d_flux_quickstart.py \
        --toroidal 24 --poloidal 24 \
        --preview-png artifacts/SCPN_Plasma_3D_quickstart.png

Output files are written to ``artifacts/``.

Use the Rust Accelerated Kernel
---------------------------------

If the Rust extension is installed (see :doc:`installation`), the solver
automatically dispatches to the compiled backend::

    from scpn_fusion.core import FusionKernel, RUST_BACKEND

    print(f"Using Rust backend: {RUST_BACKEND}")

    kernel = FusionKernel(config)
    result = kernel.solve()

The Rust kernel supports solver method selection::

    kernel.set_solver_method("multigrid")   # or "sor" (default)

All API signatures are identical between the Python and Rust paths.

Available Simulation Modes
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Mode
     - Description
     - Maturity
   * - ``kernel``
     - Grad-Shafranov equilibrium + 1.5D transport
     - Validated local contract
   * - ``neuro-control``
     - SNN-based controller
     - Validated local contract
   * - ``optimal``
     - Model-predictive controller
     - Validated local contract
   * - ``flight``
     - Real-time tokamak flight simulator
     - Validated local contract
   * - ``digital-twin``
     - Live digital twin with RL policy
     - Research contract
   * - ``safety``
     - ML disruption predictor
     - Validated local contract
   * - ``control-room``
     - Integrated control room simulation
     - Research contract
   * - ``optimizer``
     - Compact reactor design search (MVR-0.96)
     - Validated
   * - ``breeding``
     - Tritium breeding blanket neutronics
     - Validated
   * - ``diagnostics``
     - Synthetic sensors + tomography
     - Validated
   * - ``heating``
     - RF heating (ICRH / ECRH / LHCD)
     - Validated
   * - ``neural``
     - Neural equilibrium solver (surrogate)
     - Reduced-order
   * - ``geometry``
     - 3D flux-surface geometry
     - Reduced-order

Tutorial Notebooks
------------------

Fifteen Jupyter notebooks are provided in ``examples/``:

1. ``neuro_symbolic_control_demo_v2`` -- **Golden Base v2**: formal proofs, closed-loop control, shot replay
2. ``01_compact_reactor_search`` -- MVR-0.96 compact reactor optimizer
3. ``02_neuro_symbolic_compiler`` -- Petri net to SNN compilation pipeline
4. ``03_grad_shafranov_equilibrium`` -- Free-boundary equilibrium solver
5. ``04_divertor_and_neutronics`` -- Divertor heat flux and TBR
6. ``05_validation_against_experiments`` -- Cross-validation vs SPARC and ITPA
7. ``06_inverse_and_transport_benchmarks`` -- Inverse solver and neural transport
8. ``07_multi_ion_transport`` -- D/T/He-ash multi-species transport evolution
9. ``08_mhd_stability`` -- Mercier, ballooning, KS, Troyon, and NTM criteria
10. ``09_coil_optimization`` -- Free-boundary coil current optimization (Tikhonov)
11. ``10_uncertainty_quantification`` -- Full-chain Monte Carlo UQ
12. ``Q10_closed_loop_demo`` -- ITER-like Q=10 closed-loop with PID/H-inf switching
13. ``platinum_standard_demo_v1`` -- Vertical integration demo (NMPC, Rutherford MHD, SOC)
