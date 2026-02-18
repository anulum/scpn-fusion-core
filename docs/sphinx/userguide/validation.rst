==============================
Validation Framework
==============================

SCPN-Fusion-Core is validated against published experimental data from
real tokamaks.  The validation framework provides regression-grade
checks that detect numerical regressions and ensure physically
reasonable behaviour across releases.

Validation Datasets
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Dataset
     - Source
     - Contents
   * - SPARC GEQDSK
     - `CFS SPARCPublic <https://github.com/cfs-energy/SPARCPublic>`_ (MIT)
     - 8 equilibrium files (:math:`B = 12.2\,\text{T}`,
       :math:`I_p` up to :math:`8.7\,\text{MA}`)
   * - ITPA H-mode
     - Verdoolaege et al., NF 61 (2021)
     - 20-row confinement dataset from 10 tokamaks
       (JET, DIII-D, C-Mod, ASDEX-U, ...)
   * - ITER baseline
     - ITER Physics Basis
     - 15 MA :math:`Q = 10` scenario parameters
   * - SPARC V2C
     - Creely et al., JPP 2020
     - Compact high-field scenario parameters
   * - DIII-D
     - Luxon, NF 42 (2002)
     - Medium-size tokamak reference parameters
   * - JET
     - Pamela et al. (2007)
     - Largest tokamak, DT fusion reference

IPB98(y,2) Confinement Validation
-----------------------------------

The regression suite validates the confinement time prediction against
the IPB98(y,2) scaling law (ITER Physics Basis, Nuclear Fusion 39, 1999):

.. math::

   \tau_E = 0.0562 \; I_p^{0.93} \; B_T^{0.15} \; \bar{n}_{e,19}^{0.41}
            \; P_\text{loss}^{-0.69} \; R^{1.97} \; \kappa^{0.78}
            \; \varepsilon^{0.58} \; M^{0.19}

Validated accuracy against the ITPA H-mode database (20-shot subset):

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Metric
     - Value
     - Source
   * - :math:`\tau_E` RMSE
     - ``0.1287 s``
     - ``validation/validate_transport_itpa.py``
   * - :math:`\tau_E` relative RMSE
     - ``28.6%``
     - ``validation/validate_transport_itpa.py``
   * - :math:`\tau_E` mean abs. relative error
      - ``32.5%``
      - ``validation/reports/rmse_dashboard.json``
   * - Aux MW->keV/s source max rel. error
      - ``2.4e-16``
      - ``validation/benchmark_transport_power_balance.py``
   * - 2-sigma coverage
      - ``95%`` (19/20 shots)
      - ``validation/validate_transport_itpa.py``

.. note::

   The ``13.5%`` value reported for the neural transport MLP is a
   surrogate-fit metric and is **not** the same as full physics-transport
   validation on the 20-shot ITPA lane.

SPARC Equilibrium Topology Validation
---------------------------------------

The 8 SPARC GEQDSK files are validated for equilibrium topology:

- **Magnetic axis position**: :math:`R_\text{axis}` and :math:`Z_\text{axis}`
  within expected ranges for the SPARC geometry
- **Safety factor monotonicity**: :math:`q(\rho)` is monotonically
  increasing from axis to edge (no reversed shear in these scenarios)
- **GS operator sign**: the discrete Grad-Shafranov operator applied to
  the stored :math:`\psi` field has the correct sign pattern (current
  density is positive inside the plasma)

Point-Wise Psi RMSE
^^^^^^^^^^^^^^^^^^^^^

The ``psi_pointwise_rmse`` module performs point-wise :math:`\psi(R,Z)`
reconstruction error analysis on SPARC equilibria:

- Finite-difference GS operator :math:`\Delta^*\psi`
- Relative :math:`L^2` and max-abs GS residuals
- Manufactured-solution Red-Black SOR verification
- Normalised :math:`\psi` RMSE on the plasma region

ITER Reference Scenarios
--------------------------

Three reference scenarios are used for regression testing:

**ITER 15 MA baseline** (:math:`Q = 10` target):

- :math:`I_p = 15.0\,\text{MA}`, :math:`B_T = 5.3\,\text{T}`
- :math:`R = 6.2\,\text{m}`, :math:`a = 2.0\,\text{m}`,
  :math:`\kappa = 1.7`
- :math:`\tau_E = 3.7\,\text{s}`, :math:`P_\text{fus} = 500\,\text{MW}`

**SPARC V2C** (compact high-field):

- :math:`I_p = 8.7\,\text{MA}`, :math:`B_T = 12.2\,\text{T}`
- :math:`R = 1.85\,\text{m}`, :math:`a = 0.57\,\text{m}`,
  :math:`\kappa = 1.97`
- :math:`\tau_E = 0.77\,\text{s}`, :math:`P_\text{fus} = 140\,\text{MW}`

**DIII-D** (L-mode sanity check):

- :math:`I_p = 1.0\,\text{MA}`, :math:`B_T = 2.1\,\text{T}`
- :math:`R = 1.67\,\text{m}`, :math:`a = 0.67\,\text{m}`

Running the Validation Suite
-----------------------------

Generate an RMSE dashboard report::

    python validation/rmse_dashboard.py

This writes JSON and Markdown reports to ``validation/reports/``
containing confinement time accuracy, beta normalised surrogate RMSE,
SPARC axis position errors, and point-wise :math:`\psi` RMSE.

Benchmark transport-source power-balance reconstruction::

    python validation/benchmark_transport_power_balance.py

Validate against SPARC GEQDSK files::

    python validation/validate_against_sparc.py

Run the full regression test suite::

    pytest tests/test_validation_regression.py -v

ITER configuration validation::

    python validation/validate_iter.py

Property-Based Testing
------------------------

The test suite includes property-based tests using
`Hypothesis <https://hypothesis.readthedocs.io/>`_ (Python) and
`proptest <https://crates.io/crates/proptest>`_ (Rust), covering:

- **Numerical invariants** -- symmetry of solution operators, positive
  definiteness of energy functionals
- **Topology preservation** -- equilibrium topology is preserved under
  perturbations
- **Solver convergence** -- residuals monotonically decrease for
  well-conditioned problems
- **Physical bounds** -- temperatures, densities, and pressures remain
  non-negative

Code Health
-----------

The codebase has undergone 8 systematic hardening waves (248 tasks)
that replaced silent clamping, ``unwrap()`` calls, and implicit
coercion with explicit ``FusionResult<T>`` error propagation throughout
the Rust workspace.

.. list-table::
   :header-rows: 1
   :widths: 10 30 10 50

   * - Wave
     - Scope
     - Tasks
     - Highlights
   * - S2
     - Scaffold integrity
     - 8
     - Module wiring, import consistency
   * - S3
     - CI pipeline
     - 6
     - ``cargo fmt --check``, ``clippy``, test gates
   * - S4
     - Baseline coverage
     - 4
     - Property-based tests (Hypothesis + proptest)
   * - H5
     - SCPN compiler/controller
     - 37
     - Deterministic replay, fault injection, contracts
   * - H6
     - Digital twin + RL
     - 9
     - Chaos monkey, bit-flip resilience
   * - H7
     - Control + diagnostics
     - 90
     - Scoped RNG isolation, sensor guards, MPC validation
   * - H8
     - All 10 Rust crates
     - 94
     - Every ``unwrap()`` -> ``FusionResult``, input guards
