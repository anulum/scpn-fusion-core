==================
Example Notebooks
==================

Fifteen Jupyter notebooks in ``examples/`` demonstrate the full SCPN-Fusion-Core
stack, from equilibrium solves to closed-loop control with uncertainty
quantification.  Pre-built HTML exports are published to GitHub Pages on
every push to ``main``.

Golden Base
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Notebook
     - Description
   * - ``neuro_symbolic_control_demo_v2``
     - **Golden Base v2** -- formal proofs, closed-loop SNN control, DIII-D shot
       replay, and deterministic reproducibility.  The single most comprehensive
       demonstration of the neuro-symbolic pipeline.
   * - ``neuro_symbolic_control_demo``
     - Legacy v1 frozen base (preserved for reproducibility of published results).

Core Tutorials (01--10)
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Notebook
     - Description
   * - ``01_compact_reactor_search``
     - MVR-0.96 compact reactor optimizer: multi-objective sweep over
       :math:`R`, :math:`A`, :math:`\kappa`, :math:`B_t`, :math:`I_p` subject
       to Greenwald, Troyon, and Lawson constraints.
   * - ``02_neuro_symbolic_compiler``
     - Petri net to SNN compilation pipeline: define places, transitions, and
       contracts; compile to LIF neurons; execute in closed loop.
   * - ``03_grad_shafranov_equilibrium``
     - Free-boundary Grad-Shafranov solver tutorial with Picard iteration,
       coil geometry, and SPARC GEQDSK validation.
   * - ``04_divertor_and_neutronics``
     - Divertor heat flux calculation, TBR estimation, and blanket neutronics
       with port-coverage and streaming correction factors.
   * - ``05_validation_against_experiments``
     - Cross-validation against 8 SPARC GEQDSK equilibria and the ITPA H-mode
       confinement database (53 shots, 24 machines).
   * - ``06_inverse_and_transport_benchmarks``
     - Inverse equilibrium reconstruction and QLKNN-10D neural transport
       surrogate benchmarks.
   * - ``07_multi_ion_transport``
     - D/T/He-ash multi-species transport evolution: fuel depletion, He-ash
       accumulation and pumping, independent Te/Ti, coronal tungsten radiation.
   * - ``08_mhd_stability``
     - Five-criterion MHD stability suite: Mercier interchange, ballooning
       (:math:`s`-:math:`\alpha` diagram), Kruskal-Shafranov, Troyon beta limit,
       and NTM seeding (modified Rutherford equation).
   * - ``09_coil_optimization``
     - Free-boundary coil current optimization via Tikhonov-regularised
       least-squares with self-consistent GS-coil outer iteration.
   * - ``10_uncertainty_quantification``
     - Full-chain Monte Carlo UQ: IPB98(y,2) scaling-law exponents, transport
       model coefficients, and equilibrium boundary perturbation propagated
       through to :math:`Q`, :math:`P_{\mathrm{fus}}`, and :math:`\beta_N`
       confidence envelopes.  ITER vs SPARC comparison.

Supplementary Demos
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Notebook
     - Description
   * - ``Q10_closed_loop_demo``
     - ITER-like Q=10 closed-loop simulation with PID ramp-up and H-infinity
       flat-top control, including a tearing-mode disruption scenario with
       controller recovery.
   * - ``platinum_standard_demo_v1``
     - Vertical integration demo: NMPC-JAX, Rutherford MHD island evolution,
       quantitative SOC (calibrated MJ), Bosch-Hale D-T reactivity.
   * - ``02_neuro_symbolic_compiler_secondary``
     - Extended compiler walkthrough with alternative Petri net topologies.

Running Notebooks
-----------------

All notebooks can be run locally after installing the package::

    pip install -e ".[dev,ml]"
    jupyter lab examples/

Or launch directly in the cloud:

- `Open in Colab <https://colab.research.google.com/github/anulum/scpn-fusion-core/blob/main/examples/neuro_symbolic_control_demo_v2.ipynb>`_
- `Launch on Binder <https://mybinder.org/v2/gh/anulum/scpn-fusion-core/main?labpath=examples%2Fneuro_symbolic_control_demo_v2.ipynb>`_
