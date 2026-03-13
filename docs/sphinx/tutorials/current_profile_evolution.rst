.. -----------------------------------------------------------------------
   SCPN Fusion Core -- Current Profile Evolution Tutorial
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

=============================================
Current Profile Evolution & Steady State
=============================================

This tutorial covers the physics of current profile evolution and
demonstrates how to design a steady-state current drive scenario using
SCPN-Fusion-Core.  We will:

1. Initialise the Crank-Nicolson flux diffusion solver
2. Set up ECCD, NBI, and LHCD current drive sources
3. Evolve the current profile over a resistive timescale
4. Analyse the resulting q-profile for MHD stability

Prerequisites: :doc:`../learning/fusion_engineering_101` (sections 3, 5).


Physics Background
--------------------

The current profile :math:`j(\rho)` evolves on the **resistive timescale**
:math:`\tau_R = \mu_0 a^2 / \eta \sim 100`--:math:`500` s.  The
evolution is governed by the poloidal flux diffusion equation
(see :eq:`flux-diffusion` in the textbook chapter):

.. math::

   \frac{\partial\psi}{\partial t} = D(\rho)\,\mathcal{L}[\psi]
   + R_0 \eta(\rho)\,(j_\text{bs} + j_\text{CD})

The **resistivity** :math:`\eta(\rho)` determines the diffusion
coefficient :math:`D = \eta / (\mu_0 a^2)`.  In a tokamak, neoclassical
effects (trapped particles) enhance the resistivity by a factor
:math:`C_R / (1 - f_t)` relative to the Spitzer value.


Step 1: Initialise the Solver
-------------------------------

.. code-block:: python

   import numpy as np
   from scpn_fusion.core.current_diffusion import (
       CurrentDiffusionSolver,
       neoclassical_resistivity,
       q_from_psi,
       resistive_diffusion_time,
   )

   # ITER-like parameters
   nr = 100
   rho = np.linspace(0, 1, nr)
   R0 = 6.2    # major radius [m]
   a = 2.0     # minor radius [m]
   B0 = 5.3    # toroidal field [T]

   solver = CurrentDiffusionSolver(rho, R0, a, B0)

   # Check initial q-profile (parabolic approximation)
   q0 = q_from_psi(rho, solver.psi, R0, a, B0)
   print(f"q(0) = {q0[0]:.2f}, q(1) = {q0[-1]:.2f}")

   # Resistive timescale
   eta_core = neoclassical_resistivity(Te_keV=10.0, ne_19=10.0, Z_eff=1.5, epsilon=0.01)
   print(f"tau_R = {resistive_diffusion_time(a, eta_core):.0f} s")


Step 2: Set Up Current Drive Sources
---------------------------------------

A steady-state scenario combines multiple current drive sources to
shape the q-profile:

.. code-block:: python

   from scpn_fusion.core.current_drive import (
       ECCDSource,
       NBISource,
       LHCDSource,
       CurrentDriveMix,
   )

   eccd = ECCDSource(
       rho_dep=0.4,        # deposition at mid-radius
       width=0.08,          # narrow deposition
       power_mw=20.0,       # 20 MW ECCD
       efficiency=0.25,     # 0.25 A/W
       R0=R0, a=a,
   )

   nbi = NBISource(
       rho_dep=0.25,        # core deposition
       width=0.15,          # broad deposition
       power_mw=33.0,       # 33 MW NBI
       efficiency=0.35,     # 0.35 A/W
       R0=R0, a=a,
   )

   lhcd = LHCDSource(
       rho_dep=0.7,         # off-axis deposition
       width=0.10,
       power_mw=10.0,
       efficiency=0.20,
       R0=R0, a=a,
   )

   mix = CurrentDriveMix([eccd, nbi, lhcd])

   # Evaluate total current drive profile
   j_cd = mix.evaluate(rho)

   print(f"Total non-inductive current: {mix.total_current(rho):.2f} MA")


Step 3: Evolve the Current Profile
-------------------------------------

Run the solver for 200 seconds (roughly one resistive timescale):

.. code-block:: python

   Te = 10.0 * (1 - 0.8 * rho**2)   # electron temperature [keV]
   ne = 10.0 * (1 - 0.5 * rho**2)   # electron density [10^19 m^-3]
   Z_eff = 1.5

   # Bootstrap current (simplified model)
   dTe_drho = np.gradient(Te, rho[1] - rho[0])
   dne_drho = np.gradient(ne, rho[1] - rho[0])
   epsilon = rho * a / R0
   j_bs = -2.4 * np.sqrt(epsilon) * (ne * dTe_drho + Te * dne_drho) * 1e3

   dt = 0.1  # timestep [s]
   n_steps = 2000  # 200 seconds total

   psi_history = [solver.psi.copy()]
   q_history = [q_from_psi(rho, solver.psi, R0, a, B0)]

   for step in range(n_steps):
       solver.step(dt, Te, ne, Z_eff, j_bs, j_cd)
       if step % 200 == 0:
           psi_history.append(solver.psi.copy())
           q_history.append(q_from_psi(rho, solver.psi, R0, a, B0))

   print(f"Final q(0) = {q_history[-1][0]:.2f}")
   print(f"Final q(edge) = {q_history[-1][-1]:.2f}")


Step 4: Plot the Evolution
----------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # q-profile evolution
   for i, q in enumerate(q_history):
       t = i * 20  # each snapshot is 20 s apart
       axes[0].plot(rho, q, label=f"t = {t} s")
   axes[0].set_xlabel(r"$\rho$")
   axes[0].set_ylabel("q")
   axes[0].set_title("Safety Factor Evolution")
   axes[0].axhline(1.0, color="red", ls="--", alpha=0.5)
   axes[0].axhline(1.5, color="orange", ls="--", alpha=0.5)
   axes[0].legend(fontsize=8)

   # Current drive profile
   axes[1].plot(rho, eccd.profile(rho) * 1e-6, label="ECCD")
   axes[1].plot(rho, nbi.profile(rho) * 1e-6, label="NBI")
   axes[1].plot(rho, lhcd.profile(rho) * 1e-6, label="LHCD")
   axes[1].plot(rho, j_bs * 1e-6, label="Bootstrap", ls="--")
   axes[1].set_xlabel(r"$\rho$")
   axes[1].set_ylabel("j [MA/m²]")
   axes[1].set_title("Current Density Components")
   axes[1].legend()

   plt.tight_layout()
   plt.show()


Step 5: Check MHD Stability of the Final State
-------------------------------------------------

.. code-block:: python

   from scpn_fusion.core import run_full_stability_check
   from scpn_fusion.core.ntm_dynamics import find_rational_surfaces

   q_final = q_history[-1]

   # Find rational surfaces
   surfaces = find_rational_surfaces(rho, q_final, [1.0, 1.5, 2.0])
   for q_val, rho_loc in surfaces.items():
       if rho_loc is not None:
           print(f"q = {q_val} surface at rho = {rho_loc:.3f}")
       else:
           print(f"q = {q_val} surface: absent (good for NTM avoidance)")


Design Principles
-------------------

**Reversed shear:** Targeting :math:`q_\text{min} > 1` (no sawtooth
surface) with :math:`q_0 > 2` (reversed shear in the core) provides
access to internal transport barriers and improved confinement.

**NTM avoidance:** Keeping :math:`q_\text{min} > 1.5` eliminates the
:math:`q = 3/2` NTM rational surface, removing the dominant cause of
confinement degradation.

**Steady-state current alignment:** When
:math:`j_\text{bs} + j_\text{CD} \approx j_\text{total}`, the ohmic
drive vanishes and the plasma can operate indefinitely (limited by
tritium fuel supply, not transformer flux).


Related Modules
----------------

- :mod:`scpn_fusion.core.current_diffusion` -- Crank-Nicolson solver
- :mod:`scpn_fusion.core.current_drive` -- ECCD, NBI, LHCD sources
- :mod:`scpn_fusion.core.sawtooth` -- Porcelli trigger, Kadomtsev crash
- :mod:`scpn_fusion.core.ntm_dynamics` -- Modified Rutherford Equation
- :doc:`mhd_instabilities` -- companion tutorial on sawteeth and NTMs
