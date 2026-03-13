.. -----------------------------------------------------------------------
   SCPN Fusion Core -- First Simulation
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

==============================
Your First Simulation
==============================

This hands-on tutorial walks you through your first SCPN-Fusion-Core
simulation, from installation to plotting results.  By the end you will
have solved a Grad-Shafranov equilibrium, computed transport profiles,
and run a basic stability check.

Prerequisite: :doc:`fusion_engineering_101` (for physics context).


Installation
------------

.. code-block:: bash

   pip install scpn-fusion

For the Rust-accelerated backend (10--50x faster):

.. code-block:: bash

   pip install scpn-fusion-rs

Verify the installation:

.. code-block:: python

   import scpn_fusion
   print(scpn_fusion.__version__)  # should print 3.9.3 or later


Step 1: Solve an Equilibrium
------------------------------

Create an ITER-like equilibrium with 65x65 grid resolution:

.. code-block:: python

   from scpn_fusion.core import FusionKernel

   kernel = FusionKernel(
       R0=6.2,       # major radius [m]
       a=2.0,        # minor radius [m]
       B0=5.3,       # toroidal field [T]
       Ip=15e6,      # plasma current [A]
       kappa=1.7,    # elongation
       delta=0.33,   # triangularity
       grid_size=65,
   )

   result = kernel.solve()
   print(f"Converged in {result.iterations} iterations")
   print(f"Beta_p = {result.beta_p:.3f}")
   print(f"li = {result.li:.3f}")

The solver computes the poloidal flux :math:`\psi(R, Z)` and derives
all equilibrium quantities: flux surfaces, q-profile, pressure, and
current density.


Step 2: Visualise Flux Surfaces
---------------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(5, 8))
   ax.contour(kernel.R_grid, kernel.Z_grid, result.psi, levels=20)
   ax.set_xlabel("R [m]")
   ax.set_ylabel("Z [m]")
   ax.set_aspect("equal")
   ax.set_title("Poloidal Flux Surfaces")
   plt.tight_layout()
   plt.show()

You should see nested elliptical contours centred on the magnetic axis,
with a separatrix at the outermost closed surface.


Step 3: Inspect the q-Profile
-------------------------------

.. code-block:: python

   fig, ax = plt.subplots()
   ax.plot(result.rho, result.q_profile)
   ax.set_xlabel(r"$\rho$ (normalised radius)")
   ax.set_ylabel("q")
   ax.set_title("Safety Factor Profile")
   ax.axhline(1.0, color="red", linestyle="--", label="q = 1 (sawtooth)")
   ax.axhline(1.5, color="orange", linestyle="--", label="q = 3/2 (NTM)")
   ax.axhline(2.0, color="green", linestyle="--", label="q = 2 (NTM)")
   ax.legend()
   plt.show()

The q-profile typically starts near 1 on axis and rises to 3--5 at
the edge.  Rational surfaces where :math:`q = m/n` are marked: these
are the locations of MHD instabilities.


Step 4: Run an MHD Stability Check
-------------------------------------

.. code-block:: python

   from scpn_fusion.core import run_full_stability_check

   stability = run_full_stability_check(
       q_profile=result.q_profile,
       rho=result.rho,
       pressure=result.pressure_profile,
       R0=6.2, a=2.0, B0=5.3, Ip=15e6, kappa=1.7,
   )

   print(f"Mercier stable: {stability.mercier.stable}")
   print(f"Kruskal-Shafranov: q_edge = {stability.kruskal_shafranov.q_edge:.2f}")
   print(f"Troyon beta_N limit: {stability.troyon.beta_N_limit:.2f}")
   print(f"Overall stable: {stability.stable}")


Step 5: Compute Transport
---------------------------

Run the 1.5D radial transport solver for a single timestep:

.. code-block:: python

   from scpn_fusion.core.integrated_transport_solver import IntegratedTransportSolver
   import numpy as np

   nr = 50
   rho = np.linspace(0, 1, nr)

   solver = IntegratedTransportSolver(
       rho=rho,
       R0=6.2,
       a=2.0,
       B0=5.3,
   )

   # Parabolic initial profiles
   Te = 10.0 * (1 - rho**2)   # keV
   ne = 10.0 * (1 - rho**2)   # 10^19 m^-3

   # Heating source: 50 MW NBI, Gaussian deposition
   P_heat = 50e6 * np.exp(-((rho - 0.3) / 0.15)**2)

   Te_new = solver.step_energy(Te, ne, P_heat, dt=0.01)

   fig, ax = plt.subplots()
   ax.plot(rho, Te, label="Before")
   ax.plot(rho, Te_new, label="After (10 ms)")
   ax.set_xlabel(r"$\rho$")
   ax.set_ylabel("Te [keV]")
   ax.legend()
   plt.show()


Step 6: Compute Confinement Scaling
--------------------------------------

Check how your scenario compares to the ITER H-mode database:

.. code-block:: python

   from scpn_fusion.core import ipb98y2_tau_e, compute_h_factor

   tau_scaling = ipb98y2_tau_e(
       Ip_MA=15.0, B_T=5.3, n_19=10.0, P_MW=50.0,
       R_m=6.2, epsilon=2.0/6.2, kappa=1.7, M=2.5,
   )
   print(f"IPB98(y,2) tau_E = {tau_scaling:.2f} s")

   tau_actual = 3.7  # measured or simulated
   H = compute_h_factor(tau_actual, tau_scaling)
   print(f"H-factor = {H:.2f}")

An H-factor of 1.0 means the plasma confines exactly as the scaling
predicts; ITER targets :math:`H \geq 1.0`.


Step 7: Read Real Experimental Data
--------------------------------------

Load a SPARC GEQDSK file and compare:

.. code-block:: python

   from scpn_fusion.core import read_geqdsk

   eq = read_geqdsk("validation/reference_data/sparc/lmode_vv.geqdsk")
   print(f"R0 = {eq.R0:.3f} m, a = {eq.a:.3f} m")
   print(f"B_T = {eq.B_T:.2f} T, Ip = {eq.Ip/1e6:.1f} MA")


What's Next?
-------------

You've completed the basics.  Depending on your interests:

- **Physics deep dives:** :doc:`../tutorials/current_profile_evolution`,
  :doc:`../tutorials/mhd_instabilities`, :doc:`../tutorials/edge_sol_physics`
- **Control systems:** :doc:`../tutorials/realtime_reconstruction`,
  :doc:`../tutorials/fault_tolerant_operations`,
  :doc:`../tutorials/scenario_design`
- **Full reference:** :doc:`../userguide/equilibrium`,
  :doc:`../userguide/transport`, :doc:`../userguide/control`
- **API details:** :doc:`../api/core`, :doc:`../api/control`
