.. -----------------------------------------------------------------------
   SCPN Fusion Core -- MHD Instabilities Tutorial
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

==================================================
MHD Instabilities: Sawteeth and NTMs
==================================================

This tutorial covers the two resistive MHD instabilities that most
directly affect tokamak operation: **sawteeth** (core relaxation at the
:math:`q = 1` surface) and **neoclassical tearing modes** (NTMs, magnetic
islands at :math:`q = 3/2` and :math:`q = 2`).  You will learn how to:

1. Detect sawtooth crashes using the Porcelli trigger model
2. Apply the Kadomtsev reconnection model
3. Evolve NTM island width via the Modified Rutherford Equation
4. Stabilise NTMs with targeted ECCD

Prerequisites: :doc:`current_profile_evolution`.


Part I: Sawteeth
==================

Physics
--------

When the safety factor drops below 1 in the plasma core, the internal
kink mode (m=1, n=1) becomes unstable.  The resulting **sawtooth cycle**
consists of:

1. **Ramp phase** (slow, :math:`\sim 10`--:math:`100` ms): Temperature
   and density rise in the core as heating exceeds transport.
2. **Crash** (fast, :math:`\sim 100` :math:`\mu`\s): Magnetic reconnection
   at :math:`q = 1` expels core energy, flattening profiles out to the
   mixing radius :math:`r_\text{mix}`.

Sawteeth are generally benign (they flush helium ash from the core) but
large crashes can seed NTMs by generating magnetic perturbations at the
:math:`q = 3/2` and :math:`q = 2` surfaces.


Sawtooth Simulation
---------------------

.. code-block:: python

   import numpy as np
   from scpn_fusion.core.sawtooth import (
       SawtoothCycler,
       SawtoothMonitor,
       kadomtsev_crash,
   )

   nr = 100
   rho = np.linspace(0, 1, nr)

   # Initial profiles
   Te = 15.0 * (1 - 0.9 * rho**2)  # core-peaked temperature [keV]
   q = 0.85 + 1.5 * rho**2          # q(0) = 0.85, so q=1 surface exists

   # Sawtooth monitor: checks Porcelli trigger each call
   monitor = SawtoothMonitor(s_crit=0.3, rho_star_crit=0.05)

   # Check trigger
   triggered, info = monitor.check_trigger(rho, q, Te, ne_keV=10.0, B0=5.3)
   print(f"Sawtooth triggered: {triggered}")
   if triggered:
       print(f"  q=1 location: rho = {info['rho_q1']:.3f}")
       print(f"  Mixing radius: rho = {info['rho_mix']:.3f}")

       # Apply Kadomtsev crash
       Te_post = kadomtsev_crash(rho, Te, info['rho_q1'], info['rho_mix'])
       print(f"  Te(0): {Te[0]:.1f} -> {Te_post[0]:.1f} keV")


Full Sawtooth Cycling
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   cycler = SawtoothCycler(
       rho=rho, R0=6.2, a=2.0, B0=5.3,
       s_crit=0.3, rho_star_crit=0.05,
   )

   # Simulate 50 ms of plasma evolution with sawteeth
   dt = 1e-3  # 1 ms timestep
   Te_current = Te.copy()
   q_current = q.copy()
   crash_times = []

   for step in range(50):
       t = step * dt

       # Artificial core heating (ramp phase)
       Te_current[:nr//3] += 0.1  # 0.1 keV per ms in the core

       crashed, Te_current, info = cycler.step(
           Te_current, q_current, ne_keV=10.0, dt=dt,
       )
       if crashed:
           crash_times.append(t)
           print(f"  Crash at t = {t*1e3:.0f} ms")

   print(f"\n{len(crash_times)} sawtooth crashes in 50 ms")
   if len(crash_times) > 1:
       period = np.diff(crash_times).mean() * 1e3
       print(f"Average period: {period:.1f} ms")


Part II: Neoclassical Tearing Modes
======================================

Physics
--------

NTMs are magnetic islands at rational surfaces (:math:`q = m/n`)
driven by the loss of bootstrap current inside the island.  The
island creates a flat pressure region (no gradient, no bootstrap
current), reducing the total current and further destabilising the
island through a positive feedback loop.

The island half-width :math:`w` evolves according to the **Modified
Rutherford Equation** (see :eq:`mre` in the textbook):

.. math::

   \tau_R \frac{dw}{dt} = r_s \left[
   \Delta' + \alpha_\text{bs}\frac{j_\text{bs}}{j_\phi}\frac{r_s}{w}
   - \alpha_\text{GGJ}\frac{w_d^2}{w^3}
   - \alpha_\text{pol}\frac{w_\text{pol}^4}{w^3(w^2 + w_d^2)}
   + \alpha_\text{ECCD}\frac{j_\text{ECCD}}{j_\phi}\frac{r_s}{w}
   \right]

Key terms:

- :math:`\Delta' < 0` (classically stable) — tearing is subcritical
- :math:`\alpha_\text{bs}` (bootstrap drive) — makes the mode grow
- :math:`\alpha_\text{GGJ}` (curvature) — stabilises small islands
- :math:`\alpha_\text{pol}` (polarisation) — provides a seed threshold
- :math:`\alpha_\text{ECCD}` — targeted ECCD adds stabilising current


NTM Simulation
----------------

.. code-block:: python

   from scpn_fusion.core.ntm_dynamics import (
       NTMIslandDynamics,
       NTMController,
       find_rational_surfaces,
   )

   # Find rational surfaces in the q-profile
   q_profile = 0.85 + 1.5 * rho**2 + 0.5 * rho**4
   surfaces = find_rational_surfaces(rho, q_profile, [1.0, 1.5, 2.0])

   for q_val, rho_loc in surfaces.items():
       if rho_loc is not None:
           print(f"q = {q_val} at rho = {rho_loc:.3f}")

   # Initialise NTM at q = 2 surface
   ntm = NTMIslandDynamics(
       m=2, n=1,
       r_s=surfaces[2.0] * 2.0,  # dimensional radius [m]
       a=2.0,
       R0=6.2,
       B0=5.3,
       tau_R=300.0,  # resistive time [s]
   )

   # Seed the island (e.g. from a sawtooth crash)
   ntm.w = 0.02  # 2 cm seed island

   # Set plasma parameters at the rational surface
   ntm.j_bs = 0.5e6   # bootstrap current density [A/m^2]
   ntm.j_phi = 1.0e6   # total current density [A/m^2]

   # Evolve without ECCD (island grows)
   dt = 0.1
   for step in range(200):
       ntm.step(dt, j_eccd=0.0)

   print(f"Island width after 20 s (no ECCD): {ntm.w*100:.1f} cm")
   print(f"Saturated: {ntm.is_saturated()}")


ECCD Stabilisation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Reset and apply ECCD
   ntm.w = 0.02

   controller = NTMController(
       target_modes=[(2, 1)],
       eccd_efficiency=0.25,
       eccd_power_max_mw=5.0,
   )

   w_history = []
   for step in range(500):
       # Controller decides ECCD power based on island size
       j_eccd = controller.compute_eccd(ntm)
       ntm.step(dt, j_eccd=j_eccd)
       w_history.append(ntm.w * 100)  # cm

   print(f"Island width after 50 s (with ECCD): {ntm.w*100:.2f} cm")


Plotting the Result
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(np.arange(len(w_history)) * dt, w_history)
   ax.set_xlabel("Time [s]")
   ax.set_ylabel("Island Width [cm]")
   ax.set_title("NTM (2,1) Stabilisation with ECCD")
   ax.axhline(2.0, color="red", ls="--", label="Seed width")
   ax.legend()
   plt.tight_layout()
   plt.show()


Design Guidelines
-------------------

**Sawtooth control:**

- Longer sawtooth periods → larger crashes → bigger NTM seeds.  ECCD at
  the :math:`q = 1` surface (pacing) can shorten the period.
- Ion cyclotron resonance heating (ICRH) with minority ions stabilises
  sawteeth by enhancing fast-ion pressure — but delays the crash,
  potentially creating a monster sawtooth.

**NTM prevention:**

- Keep :math:`q_\text{min} > 1.5` (no :math:`q = 3/2` surface).
- Maintain ECCD standby at the :math:`q = 2` surface.
- Monitor island width via locked-mode detection coils; trigger ECCD
  above a threshold island size.

**ECCD alignment:**

- The ECCD deposition must be localised within :math:`\pm 2` cm of the
  rational surface.  Real-time EFIT (see :doc:`realtime_reconstruction`)
  provides the equilibrium update needed to steer the ECCD mirrors.


Related Modules
-----------------

- :mod:`scpn_fusion.core.sawtooth` -- Porcelli trigger, Kadomtsev crash
- :mod:`scpn_fusion.core.ntm_dynamics` -- Modified Rutherford Equation
- :mod:`scpn_fusion.core.stability_mhd` -- full stability suite
- :doc:`current_profile_evolution` -- current drive for q-profile control
- :doc:`realtime_reconstruction` -- EFIT for ECCD targeting
