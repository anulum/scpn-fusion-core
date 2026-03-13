.. -----------------------------------------------------------------------
   SCPN Fusion Core -- Edge/SOL Physics Tutorial
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

=============================================
Edge Physics: SOL and Divertor Heat Flux
=============================================

This tutorial covers the Scrape-Off Layer (SOL) and divertor target
heat flux, the most critical engineering constraint for steady-state
fusion reactors.  You will learn to:

1. Compute the midplane heat flux width from the Eich scaling
2. Estimate divertor target conditions using the two-point model
3. Assess whether a reactor design stays within material limits

Prerequisite: :doc:`../learning/fusion_engineering_101` (section 6).


The Power Exhaust Problem
--------------------------

In a tokamak, all heating power that is not radiated must flow through
the SOL to the divertor targets.  The total power crossing the
separatrix:

.. math::

   P_\text{SOL} = P_\text{heat} - P_\text{rad,core}

For ITER: :math:`P_\text{SOL} \approx 100` MW.  This power is
concentrated in a narrow annulus of width :math:`\lambda_q` (a few mm)
mapped along field lines to the divertor.  The resulting heat flux
density can exceed 100 MW/m\ :sup:`2` in the parallel direction, far
above material limits.


Step 1: Compute the Heat Flux Width
--------------------------------------

.. code-block:: python

   from scpn_fusion.core.sol_model import eich_heat_flux_width

   # ITER parameters
   B_p = 1.2     # poloidal field at outboard midplane [T]
   lambda_q = eich_heat_flux_width(B_p)
   print(f"Eich lambda_q = {lambda_q:.2f} mm")

The Eich scaling (Eich 2013):

.. math::

   \lambda_q\;[\text{mm}] = 0.63 \; B_{p,\text{omp}}^{-1.19}

predicts :math:`\lambda_q \sim 1` mm for ITER — a severe engineering
challenge.


Step 2: Two-Point Model
--------------------------

.. code-block:: python

   from scpn_fusion.core.sol_model import TwoPointSOL

   sol = TwoPointSOL(
       R0=6.2,
       a=2.0,
       B0=5.3,
       Ip_MA=15.0,
       P_sol_mw=100.0,
       kappa=1.7,
   )

   result = sol.solve()

   print(f"Upstream Te: {result.T_u_eV:.0f} eV")
   print(f"Target Te: {result.T_t_eV:.1f} eV")
   print(f"Target density: {result.n_t:.2e} m^-3")
   print(f"Heat flux width: {result.lambda_q_mm:.2f} mm")
   print(f"Peak parallel heat flux: {result.q_parallel_peak:.1f} MW/m^2")
   print(f"Peak surface heat flux: {result.q_surface_peak:.1f} MW/m^2")


Step 3: Compare Reactor Designs
----------------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   machines = {
       "ITER": dict(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, P_sol_mw=100, kappa=1.7),
       "SPARC": dict(R0=1.85, a=0.57, B0=12.2, Ip_MA=8.7, P_sol_mw=30, kappa=1.97),
       "ARC": dict(R0=3.3, a=1.13, B0=9.2, Ip_MA=7.8, P_sol_mw=80, kappa=1.8),
       "EU-DEMO": dict(R0=9.1, a=2.93, B0=5.7, Ip_MA=18.0, P_sol_mw=150, kappa=1.59),
   }

   results = {}
   for name, params in machines.items():
       sol = TwoPointSOL(**params)
       results[name] = sol.solve()
       print(f"{name:10s}: q_surf = {results[name].q_surface_peak:6.1f} MW/m^2, "
             f"lambda_q = {results[name].lambda_q_mm:.2f} mm")

   # Material limit
   print(f"\nTungsten steady-state limit: ~10 MW/m^2")


Step 4: Sensitivity Study
----------------------------

How does :math:`P_\text{SOL}` affect target heat flux?

.. code-block:: python

   P_range = np.linspace(20, 150, 30)
   q_surf = []

   for P in P_range:
       sol = TwoPointSOL(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, P_sol_mw=P, kappa=1.7)
       r = sol.solve()
       q_surf.append(r.q_surface_peak)

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(P_range, q_surf, "b-", linewidth=2)
   ax.axhline(10.0, color="red", ls="--", label="Tungsten limit (10 MW/m²)")
   ax.set_xlabel("P_SOL [MW]")
   ax.set_ylabel("Peak surface heat flux [MW/m²]")
   ax.set_title("Divertor Heat Flux vs. SOL Power (ITER)")
   ax.legend()
   plt.tight_layout()
   plt.show()


Mitigation Strategies
-----------------------

When :math:`q_\text{surf}` exceeds material limits, the options are:

1. **Impurity seeding** (N, Ne, Ar): Radiate 50--80% of :math:`P_\text{SOL}`
   before it reaches the target.  Reduces :math:`T_t` to a few eV
   (detached divertor regime).

2. **Flux expansion**: Increase the target-to-midplane flux tube area
   ratio by moving the X-point or adding a secondary X-point (snowflake
   divertor).

3. **Long-legged divertor**: Increase connection length :math:`L_\parallel`
   to enhance radiation and recombination.

4. **Liquid metal targets**: Replace solid tungsten with flowing lithium
   or tin, which can absorb higher heat fluxes.

5. **Strike-point sweeping**: Oscillate the separatrix strike point
   across the target to spread the heat load temporally.

These strategies are outside the scope of the two-point model but inform
the design targets passed to SCPN-Fusion-Core's control modules.


Related Modules
-----------------

- :mod:`scpn_fusion.core.sol_model` -- two-point model, Eich scaling
- :mod:`scpn_fusion.core.divertor_thermal_sim` -- detailed divertor simulation
- :mod:`scpn_fusion.nuclear.blanket_neutronics` -- first-wall loading
