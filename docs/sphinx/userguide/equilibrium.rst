.. SPDX-License-Identifier: AGPL-3.0-or-later
.. Commercial license available
.. © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
.. © Code 2020–2026 Miroslav Šotek. All rights reserved.
.. ORCID: 0009-0009-3560-0851
.. Contact: www.anulum.li | protoscience@anulum.li
.. SCPN Fusion Core — Equilibrium user guide

==============================
Equilibrium Solver
==============================

The equilibrium solver is the numerical core of SCPN-Fusion-Core.  It
solves the Grad-Shafranov equation for axisymmetric magnetostatic
equilibria, the foundation upon which all transport, stability, and
control computations are built.

The Grad-Shafranov Equation
----------------------------

In an axisymmetric toroidal plasma, force balance between the plasma
pressure gradient, the magnetic pressure, and the magnetic tension
reduces to a single nonlinear elliptic PDE known as the Grad-Shafranov
(GS) equation (Grad & Rubin 1958, Shafranov 1966):

.. math::
   :label: gs-equation

   \Delta^{*}\psi \;=\; -\mu_0 R^2\,p'(\psi) \;-\; F(\psi)\,F'(\psi)

where:

- :math:`\psi(R,Z)` is the poloidal magnetic flux function
- :math:`\Delta^{*} = R \frac{\partial}{\partial R}\!\left(\frac{1}{R}\frac{\partial}{\partial R}\right) + \frac{\partial^2}{\partial Z^2}` is the toroidal elliptic (Grad-Shafranov) operator
- :math:`p(\psi)` is the plasma pressure profile
- :math:`F(\psi) = R B_\phi` is the poloidal current function (diamagnetic function)
- :math:`\mu_0` is the permeability of free space

The magnetic field components are recovered from :math:`\psi` as:

.. math::

   B_R = -\frac{1}{R}\frac{\partial\psi}{\partial Z}, \qquad
   B_Z = \frac{1}{R}\frac{\partial\psi}{\partial R}, \qquad
   B_\phi = \frac{F(\psi)}{R}

Numerical Method
-----------------

SCPN-Fusion-Core discretises equation :eq:`gs-equation` on a uniform
:math:`(R, Z)` grid using a five-point finite-difference stencil with
the correct :math:`1/R` toroidal geometry factor.

Picard Iteration
^^^^^^^^^^^^^^^^

The nonlinear system is solved by Picard (fixed-point) iteration with
under-relaxation:

.. math::

   \psi^{(k+1)} = (1 - \alpha)\,\psi^{(k)} + \alpha\,\psi_{\text{new}}^{(k)}

where :math:`\alpha` is the relaxation factor (default 0.1) and
:math:`\psi_{\text{new}}` is obtained by solving the linearised GS
equation at the current iterate.

The inner linear system is solved by one of two methods:

**Red-Black SOR** (default)
   Successive Over-Relaxation with red-black ordering for cache-friendly
   memory access.  This is the production default and is well-suited for
   grids up to 128x128.

**Multigrid V-Cycle** (selectable)
   A geometric multigrid solver with standard restriction, prolongation,
   and Gauss-Seidel smoothing.  Activated via::

       kernel.set_solver_method("multigrid")

   The multigrid path is asymptotically faster for large grids but
   carries slightly more implementation complexity.

Field-Reversed Configuration Analytical Limit
---------------------------------------------

The FRC workstream begins with the Steinhauer rigid-rotor no-rotation
analytical limit.  The public API is intentionally narrow until the rotating
BVP lands, but the accepted no-rotation contract is now implemented in both
Python and Rust, with optional PyO3 exposure when the native extension is
available.

.. code-block:: python

   import numpy as np
   from scpn_fusion.core import RigidRotorFRCInputs, solve_frc_equilibrium

   inputs = RigidRotorFRCInputs(
       n0=4.138e21,
       T_i_eV=10_000.0,
       T_e_eV=5_000.0,
       theta_dot=0.0,
       R_s=0.20,
       B_ext=5.0,
       delta=0.02,
   )
   state = solve_frc_equilibrium(inputs, np.linspace(0.0, 0.4, 129))

The implemented axial field is:

.. math::

   B_z(r) = -B_{\rm ext}\tanh\left(\frac{r^2 - R_s^2}{2 R_s \delta}\right)

Rotating rigid-rotor cases fail closed with ``NotImplementedError`` until the
dedicated FRC BVP task is implemented and validated.

The no-rotation pressure profile follows local magnetic-pressure balance,
``p = (B_ext^2 - B_z^2) / (2 mu_0)``. The scalar input thermal pressure
``n0 * (T_i + T_e) * e`` is reported as a consistency ratio against the
magnetic-pressure-balance peak; it is not substituted for the local solved
profile. The solver also derives ``n(r) = p(r) / ((T_i + T_e) * e)`` and gates
the configured central density against the solved peak density. The validation
report includes the density-consistency row, local ``beta = p / (B_ext^2 /
(2 mu_0))``, separatrix-averaged beta, particle line density
``integral_0^R_s n(r) 2 pi r dr``, separatrix pressure-energy inventory,
magnetic-deficit inventory, energy-closure relative error, the pressure-balance
residual ``p + B_z^2/(2 mu_0) - B_ext^2/(2 mu_0)``, the analytical flux-primitive closure
residual ``dpsi/dr - r B_z``, the Ampere closure residual ``mu_0 J_theta +
dB_z/dr``, and the radial ideal-MHD force-balance residual ``dp/dr - (J x
B)_r``. It also checks the separatrix current-sheet identities
``dB_z/dr|_R_s = -B_ext / delta`` and
``J_theta(R_s) = B_ext / (mu_0 delta)``. Density, beta limit, separatrix energy
inventory, current-sheet closure, pressure, flux, and Ampere closure are active
finite-grid gates by default. Force balance is reported as a diagnostic by
default; pass an explicit ``force_balance_tolerance`` to make that residual part
of the acceptance gate for a specific run.

The Rust surface is exposed as ``fusion_physics::frc`` and benchmarked by
``cargo bench -p fusion-physics --bench frc_rigid_rotor_bench``.  The tracked
cross-surface report is created with ``python benchmarks/bench_frc_rigid_rotor.py``
and stored at ``validation/reports/frc_rigid_rotor_benchmark.json``.  Go, Julia, and Lean
rows remain explicit ``not_applicable_no_frc_surface`` rows until those
languages expose equivalent solver logic; wrappers are not treated as parity.

Profile Models
^^^^^^^^^^^^^^

Two pressure/current profile parametrisations are supported:

**L-mode** (default)
   Linear profiles: :math:`p(\hat\psi) = p_0(1 - \hat\psi)` and
   :math:`FF'(\hat\psi) = c(1 - \hat\psi)` where :math:`\hat\psi` is
   the normalised flux.

**H-mode**
   Modified hyperbolic tangent (mtanh) pedestal profiles following the
   Hirshman-Whitson parametrisation:

   .. math::

      T(\rho) = T_{\text{ped}} \cdot \frac{1}{2}\left[1 + \tanh\!\left(\frac{\rho_{\text{ped}} - \rho}{\Delta}\right)\right] + T_{\text{core}} \cdot (1 - \rho^2)

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

The solver supports both fixed-boundary (Dirichlet) and free-boundary
modes.  In free-boundary mode, the vacuum field from external coils is
computed from the circular-filament Green's function and matched at the
computational boundary.  The free-boundary adapter validates configured
coil positions, turns, current limits, flux-control points, X-point targets,
and divertor strike-point targets before building the coil set.

Shape-control operation can optionally solve a bounded least-squares problem
for coil currents at target flux points.  Magnetic-diagnostic reconstruction
uses the same Green's-function model to fit coil currents from flux loops and
finite-difference magnetic-probe responses with optional measurement
uncertainties and Tikhonov regularisation.

Convergence Criteria
^^^^^^^^^^^^^^^^^^^^

Convergence is declared when the relative :math:`L^2` residual of the
GS operator falls below a threshold (default :math:`10^{-6}`)::

   ||Delta* psi + mu_0 R^2 p' + F F'||_2 / ||psi||_2 < tol

Solver Tuning
--------------

The relaxation factor, grid resolution, and convergence threshold are
the primary tuning parameters.  Detailed guidance is provided in
``docs/SOLVER_TUNING_GUIDE.md``.

.. list-table:: Relaxation factor recommendations
   :header-rows: 1
   :widths: 20 40 40

   * - Range
     - Behaviour
     - When to use
   * - 0.02--0.05
     - Very conservative, slow but stable
     - High-beta plasmas, strongly shaped equilibria
   * - 0.08--0.12
     - Default range (0.1 shipped)
     - Most ITER-class and medium-beta scenarios
   * - 0.15--0.25
     - Aggressive, faster convergence
     - Low-beta L-mode, small grids (33x33)
   * - > 0.3
     - Likely to diverge
     - Not recommended

For compact high-field designs (SPARC-class, :math:`B > 10\,\text{T}`),
a relaxation factor of 0.08 is usually more stable than 0.1.

Configuration
--------------

The solver reads its parameters from a JSON configuration file:

.. code-block:: json

   {
       "solver": {
           "max_iterations": 500,
           "convergence_threshold": 1e-6,
           "relaxation_factor": 0.1,
           "grid_size": 65,
           "method": "sor"
       }
   }

GEQDSK I/O
-----------

The ``eqdsk`` module reads and validates the standard G-EQDSK equilibrium
file format used throughout the fusion community (Lao et al. 1985).
Eight SPARC GEQDSK files from CFS are included for validation.

.. code-block:: python

   from scpn_fusion.core.eqdsk import read_geqdsk

   eq = read_geqdsk("validation/reference_data/sparc/lmode_vv.geqdsk")

The reader extracts the full set of equilibrium data: poloidal flux grid
:math:`\psi(R,Z)`, pressure and current profiles :math:`p(\psi)` and
:math:`FF'(\psi)`, boundary/limiter geometry, and scalar parameters
(:math:`B_\text{T}`, :math:`I_p`, axis position, separatrix flux values).

Neural Equilibrium Solver
--------------------------

For real-time control applications where millisecond-scale equilibrium
updates are required, the neural equilibrium solver (``neural_equilibrium.py``)
provides a PCA + MLP surrogate that predicts flux-surface geometry from
an 8-dimensional input vector:

.. math::

   (I_p,\; B_T,\; R_\text{axis},\; Z_\text{axis},\;
    p'_\text{scale},\; FF'_\text{scale},\; \psi_\text{axis},\; \psi_\text{bry})
   \;\longrightarrow\;
   \hat{\psi}(R, Z)

The surrogate is trained on SPARC GEQDSK data with profile perturbations.
Inference time is approximately 5 microseconds per point (Rust backend,
synthetic weights).

.. warning::

   No pre-trained neural weights are shipped with the package.  Users
   must train the surrogate on their own simulation data using the
   provided ``train_on_sparc()`` convenience function.

Force Balance and 3D Geometry
------------------------------

The ``force_balance`` module provides iterative force-balance checking
against the GS equation.  The ``geometry_3d`` module generates 3D
flux-surface meshes (OBJ format) from Fourier boundary parametrisations
for visualisation and CAD integration.

Related Modules
----------------

- :mod:`scpn_fusion.core.fusion_kernel` -- main equilibrium solver
- :mod:`scpn_fusion.core.eqdsk` -- GEQDSK reader/writer
- :mod:`scpn_fusion.core.neural_equilibrium` -- PCA+MLP surrogate
- :mod:`scpn_fusion.core.force_balance` -- force-balance verification
- :mod:`scpn_fusion.core.geometry_3d` -- 3D mesh generation
- :mod:`scpn_fusion.core.equilibrium_3d` -- Fourier-mode 3D equilibrium
