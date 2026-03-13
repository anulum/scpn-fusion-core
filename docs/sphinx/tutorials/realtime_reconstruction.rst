.. -----------------------------------------------------------------------
   SCPN Fusion Core -- Real-Time Reconstruction & Shape Control
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

====================================================
Real-Time Equilibrium Reconstruction & Shape Control
====================================================

This tutorial covers the real-time plasma control chain: magnetic
equilibrium reconstruction (EFIT), plasma shape control via poloidal
field coils, and vertical stabilisation of elongated plasmas.

You will learn to:

1. Run a streaming EFIT reconstruction from magnetic measurements
2. Compute the Jacobian of boundary shape vs. coil currents
3. Design a shape controller with Tikhonov regularisation
4. Stabilise vertical displacement events (VDEs) with a super-twisting
   sliding mode controller

Prerequisite: :doc:`../learning/fusion_engineering_101` (section 7).


Part I: Real-Time EFIT
========================

EFIT (Equilibrium FITting) is the standard equilibrium reconstruction
code used at every major tokamak.  It solves an inverse problem: given
magnetic measurements (pickup coils, flux loops, Rogowski coils), find
:math:`\psi(R,Z)` and the source profiles :math:`p'(\psi)`, :math:`FF'(\psi)`
that best fit the data.

SCPN-Fusion-Core provides a lightweight streaming EFIT (``RealtimeEFIT``)
suitable for real-time control (target: < 1 ms per update).

Setting Up Magnetic Diagnostics
---------------------------------

.. code-block:: python

   from scpn_fusion.control.realtime_efit import RealtimeEFIT, MagneticDiagnostics

   # Define diagnostic positions (simplified: 12 flux loops, 8 B_p probes)
   diag = MagneticDiagnostics.iter_standard()
   print(f"Flux loops: {diag.n_flux_loops}")
   print(f"B_p probes: {diag.n_bp_probes}")
   print(f"Total measurements: {diag.n_measurements}")


Running a Reconstruction
--------------------------

.. code-block:: python

   import numpy as np

   efit = RealtimeEFIT(
       R_range=(4.0, 8.5),
       Z_range=(-5.0, 5.0),
       grid_size=33,         # coarse grid for speed
       diagnostics=diag,
   )

   # Synthetic magnetic measurements (from a reference equilibrium)
   measurements = diag.synthesize(
       Ip=15e6, R0=6.2, a=2.0, kappa=1.7, delta=0.33, B0=5.3,
   )

   # Reconstruct
   result = efit.reconstruct(measurements)

   print(f"Reconstructed R_axis = {result.R_axis:.3f} m")
   print(f"Reconstructed Z_axis = {result.Z_axis:.3f} m")
   print(f"Reconstructed Ip = {result.Ip/1e6:.2f} MA")
   print(f"Reconstruction time: {result.elapsed_ms:.2f} ms")
   print(f"Residual: {result.residual:.2e}")


Streaming Mode
^^^^^^^^^^^^^^^^

In a real control loop, EFIT runs continuously at the plasma control
system cycle rate:

.. code-block:: python

   # Simulate 100 ms of streaming reconstruction
   dt = 1e-3  # 1 ms cycle
   for step in range(100):
       # In practice: read live diagnostic data
       meas = diag.synthesize(
           Ip=15e6, R0=6.2, a=2.0,
           kappa=1.7 + 0.01 * np.sin(2 * np.pi * step * dt / 0.05),
           delta=0.33, B0=5.3,
       )
       result = efit.reconstruct(meas)

   print(f"Final kappa estimate: {result.kappa:.3f}")


Part II: Plasma Shape Control
===============================

Shape control adjusts poloidal field coil currents to maintain the
desired plasma boundary (elongation, triangularity, X-point position,
gap distances).

Coil Set and Jacobian
-----------------------

.. code-block:: python

   from scpn_fusion.control.shape_controller import PlasmaShapeController, CoilSet

   # ITER-like PF coil set (6 coils)
   coils = CoilSet.iter_standard()
   print(f"Number of PF coils: {coils.n_coils}")

   # Shape controller
   shape_ctrl = PlasmaShapeController(
       coils=coils,
       R0=6.2, a=2.0, B0=5.3,
       n_boundary_points=32,
       regularisation=1e-3,  # Tikhonov lambda
   )

   # Compute response Jacobian at operating point
   J = shape_ctrl.compute_jacobian(Ip=15e6, kappa=1.7, delta=0.33)
   print(f"Jacobian shape: {J.shape}")  # (n_boundary, n_coils)


Computing Coil Current Corrections
-------------------------------------

.. code-block:: python

   # Target: increase elongation by 0.05
   boundary_error = shape_ctrl.compute_boundary_error(
       target_kappa=1.75, target_delta=0.33,
       current_kappa=1.70, current_delta=0.33,
   )

   dI = shape_ctrl.solve(boundary_error)
   print("Coil current corrections [kA]:")
   for i, name in enumerate(coils.names):
       print(f"  {name}: {dI[i]/1e3:+.1f} kA")


X-Point Tracking
^^^^^^^^^^^^^^^^^^

The X-point position is critical: if it moves inside the first wall,
the plasma disrupts.  The shape controller tracks the X-point as part
of its boundary optimisation:

.. code-block:: python

   xp = shape_ctrl.x_point_position(result.psi)
   print(f"X-point: R = {xp[0]:.3f} m, Z = {xp[1]:.3f} m")

   # Gap distance to first wall
   gap = shape_ctrl.min_gap_to_wall(result.psi, wall=coils.first_wall)
   print(f"Minimum gap: {gap*100:.1f} cm")


Part III: Vertical Stability
==============================

Elongated plasmas (:math:`\kappa > 1`) are inherently unstable to
vertical displacements.  The growth rate scales as:

.. math::

   \gamma_\text{VDE} \sim \sqrt{\frac{(\kappa^2 - 1) \mu_0 I_p^2}{4\pi R_0 m_p b^2}}

For ITER-class plasmas, :math:`\gamma \sim 10^3` s\ :sup:`-1` (growth
time :math:`\sim 1` ms).  This demands a fast, robust controller.

Super-Twisting Sliding Mode Controller
-----------------------------------------

.. code-block:: python

   from scpn_fusion.control.sliding_mode_vertical import (
       SuperTwistingSMC,
       VerticalStabilizer,
   )

   smc = SuperTwistingSMC(
       alpha1=50.0,    # sliding gain
       alpha2=100.0,   # integral gain
       z_max=0.5,      # max allowed displacement [m]
   )

   stabilizer = VerticalStabilizer(
       controller=smc,
       m_plasma=1e-3,     # effective plasma mass [normalised]
       stability_index=-0.5,  # n_idx < 0 → unstable
       R0=6.2,
       Ip=15e6,
   )

   # Simulate a VDE: initial displacement of 5 cm
   z = 0.05
   dz = 0.0
   dt = 1e-4   # 100 microsecond control cycle

   z_history = []
   u_history = []

   for step in range(1000):  # 100 ms simulation
       u = stabilizer.step(z, dz, dt)
       z, dz = stabilizer.plant_step(z, dz, u, dt)
       z_history.append(z * 100)  # cm
       u_history.append(u)

   print(f"Final displacement: {z*100:.3f} cm")
   print(f"Stabilised: {abs(z) < 0.001}")


Lyapunov Certificate
^^^^^^^^^^^^^^^^^^^^^^

The super-twisting SMC provides a **Lyapunov stability certificate**:
a scalar function :math:`V(s, v)` that is guaranteed to decrease along
trajectories:

.. code-block:: python

   V = smc.lyapunov_value(z, dz)
   print(f"Lyapunov function V = {V:.6f}")
   print(f"V > 0 and decreasing → stable")


Plotting the Stabilisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

   t = np.arange(len(z_history)) * dt * 1e3  # ms

   axes[0].plot(t, z_history)
   axes[0].set_ylabel("z [cm]")
   axes[0].set_title("Vertical Displacement Event — SMC Stabilisation")
   axes[0].axhline(0, color="gray", ls="--")

   axes[1].plot(t, u_history)
   axes[1].set_ylabel("Control force [a.u.]")
   axes[1].set_xlabel("Time [ms]")

   plt.tight_layout()
   plt.show()


Related Modules
-----------------

- :mod:`scpn_fusion.control.realtime_efit` -- streaming EFIT
- :mod:`scpn_fusion.control.shape_controller` -- Jacobian + Tikhonov
- :mod:`scpn_fusion.control.sliding_mode_vertical` -- super-twisting SMC
- :doc:`fault_tolerant_operations` -- what happens when actuators fail
- :doc:`scenario_design` -- full scenario with shape + current control
