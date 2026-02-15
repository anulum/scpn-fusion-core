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
computed and matched at the computational boundary.

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
