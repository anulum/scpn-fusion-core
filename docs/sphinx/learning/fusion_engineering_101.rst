.. -----------------------------------------------------------------------
   SCPN Fusion Core -- Fusion Engineering 101
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

=================================
Fusion Engineering 101
=================================

This chapter surveys the physics disciplines that govern tokamak
operation.  Each section corresponds to a module family in
SCPN-Fusion-Core.  The goal is to build physical intuition before
touching the code.

Prerequisite: :doc:`plasma_physics_primer`.


1. Magnetohydrostatic Equilibrium
----------------------------------

A tokamak plasma must be in force balance: the magnetic force exactly
counteracts the pressure gradient.  In axisymmetry this reduces to
the Grad-Shafranov (GS) equation:

.. math::

   \Delta^{*}\psi = -\mu_0 R^2 p'(\psi) - F(\psi) F'(\psi)

where :math:`\psi(R,Z)` is the poloidal flux, :math:`p(\psi)` is the
pressure, and :math:`F(\psi) = R B_\phi` is the diamagnetic function.

**Physical picture:** The plasma is a current-carrying fluid in a
magnetic bottle.  The :math:`J \times B` force (from the toroidal
current crossing the poloidal field) pushes inward, balancing the
outward pressure gradient.  The GS equation encodes this balance at
every point in the :math:`(R, Z)` poloidal plane.

**In the code:** :mod:`scpn_fusion.core.fusion_kernel` solves the GS
equation numerically (Picard iteration + Red-Black SOR or multigrid).
For real-time applications, :mod:`scpn_fusion.core.neural_equilibrium`
provides a PCA+MLP surrogate with microsecond inference.

See :doc:`../userguide/equilibrium` for the solver tutorial.


2. Transport
-------------

Even with perfect confinement by flux surfaces, particles and energy
leak across the magnetic field.  Two mechanisms dominate:

**Neoclassical transport** (collisional + geometry):
   Particles trapped in magnetic mirrors (banana orbits) at the
   outboard side undergo random-walk steps of order the banana width
   :math:`\Delta_b \sim q \rho_L / \sqrt{\varepsilon}`, much larger
   than the Larmor radius.  The resulting diffusion coefficient is
   :math:`D_\text{neo} \sim q^2 \varepsilon^{-3/2}` times the
   classical value.

**Turbulent transport** (anomalous):
   Microinstabilities (ion temperature gradient modes, trapped electron
   modes, electron temperature gradient modes) drive turbulent eddies
   that transport particles and energy across flux surfaces.  Turbulent
   transport typically exceeds neoclassical by a factor of 5--100 and
   is the dominant loss channel.

The competition between heating power and transport losses determines
the **energy confinement time** :math:`\tau_E`.  The empirical
IPB98(y,2) scaling law:

.. math::

   \tau_E = 0.0562 \; I_p^{0.93} B_T^{0.15} n_{19}^{0.41}
   P_\text{loss}^{-0.69} R^{1.97} \varepsilon^{0.58}
   \kappa^{0.78} M^{0.19}

is the workhorse for design-point scoping.

**In the code:** :mod:`scpn_fusion.core.integrated_transport_solver`
implements 1.5D radial transport (energy + particle); the gyro-Bohm
coefficient and :math:`\chi_e` profile are computed at each radial
point.  :mod:`scpn_fusion.core.scaling_laws` provides the IPB98(y,2)
evaluation.

See :doc:`../userguide/transport` for the transport tutorial.


3. Current Profile and Resistive Diffusion
-------------------------------------------

The plasma current :math:`I_p` is distributed across the minor radius
as :math:`j(r)`.  This current profile determines :math:`q(r)` and
hence MHD stability.  The current evolves on the **resistive diffusion
timescale** :math:`\tau_R = \mu_0 a^2 / \eta` (hundreds of seconds
for ITER).

The poloidal flux diffusion equation governs this evolution:

.. math::

   \frac{\partial\psi}{\partial t} = \frac{\eta}{\mu_0 a^2}
   \frac{1}{r}\frac{\partial}{\partial r}
   \left(r \frac{\partial\psi}{\partial r}\right)
   + R_0 \eta \, j_\text{source}

where :math:`\eta(r)` is the neoclassical resistivity (Sauter 2002)
and :math:`j_\text{source}` includes bootstrap, ECCD, NBI, and LHCD
contributions.

**In the code:** :mod:`scpn_fusion.core.current_diffusion` implements
the Crank-Nicolson implicit solver for this equation.
:mod:`scpn_fusion.core.current_drive` provides the source models.

See the advanced tutorial :doc:`../tutorials/current_profile_evolution`.


4. MHD Stability
-----------------

Magnetohydrodynamic (MHD) instabilities set hard limits on plasma
parameters and can cause catastrophic disruptions.  The key modes:

**Kruskal-Shafranov limit:**
   The edge safety factor must satisfy :math:`q_\text{edge} > 1`;
   violating this triggers a global kink instability that destroys the
   plasma.

**Sawteeth** (:math:`q = 1` surface):
   Periodic relaxation oscillations in the plasma core.  A slow ramp
   phase (temperature rises) is terminated by a rapid crash (Kadomtsev
   reconnection) that flattens the core profiles.  Frequency: 1--50 Hz.
   Trigger: the Porcelli model based on internal kink, diamagnetic
   stabilisation, and trapped-particle effects.

**Neoclassical Tearing Modes** (NTMs, :math:`q = 3/2, 2`):
   Magnetic islands that grow due to the loss of bootstrap current in
   the island O-point.  Governed by the Modified Rutherford Equation.
   Stabilised by localised ECCD.

**Ballooning modes:**
   Pressure-driven modes localised on the outboard (unfavourable
   curvature) side.  Limit the achievable pressure gradient.

**Troyon limit:**
   The maximum normalised beta :math:`\beta_N = \beta \cdot a B_T / I_p`
   is limited to :math:`\sim 2.8`--:math:`4.0` depending on profile
   shape.

**In the code:** :mod:`scpn_fusion.core.stability_mhd` provides
Mercier, ballooning, kink, Troyon, and NTM checks.
:mod:`scpn_fusion.core.sawtooth` implements the Porcelli trigger
and Kadomtsev crash.  :mod:`scpn_fusion.core.ntm_dynamics` solves
the Modified Rutherford Equation.

See :doc:`../tutorials/mhd_instabilities`.


5. Heating and Current Drive
-------------------------------

Tokamak plasmas require external heating to reach fusion temperatures
and non-inductive current drive for steady-state operation.

**Neutral Beam Injection (NBI):**
   High-energy neutral atoms (80--1000 keV) are injected tangentially.
   They ionise in the plasma and transfer momentum and energy through
   collisions.  NBI also drives toroidal current (NBCD).

**Electron Cyclotron Resonance Heating (ECRH) / Current Drive (ECCD):**
   Millimetre-wave beams (typically 110--170 GHz) resonantly heat
   electrons at the electron cyclotron frequency.  By tilting the
   launch angle, current can be driven at a specific radial location
   -- essential for NTM stabilisation.

**Lower Hybrid Current Drive (LHCD):**
   GHz-range waves launched from waveguide arrays drive current in the
   outer half of the plasma.  Efficient for driving current at moderate
   density.

**Ion Cyclotron Resonance Heating (ICRH):**
   Waves at 40--80 MHz heat ions (minority heating, 2nd harmonic).

**In the code:** :mod:`scpn_fusion.core.current_drive` implements
Gaussian-profile models for ECCD, NBI, and LHCD sources with the
``CurrentDriveMix`` combiner for multi-source scenarios.

See :doc:`../tutorials/current_profile_evolution`.


6. Edge Physics: SOL and Divertor
----------------------------------

Beyond the separatrix, field lines are open and connect to material
surfaces.  This region is the **Scrape-Off Layer** (SOL).  Heat and
particles flow along field lines to the **divertor** targets, where
power exhaust is a critical engineering challenge.

The **two-point model** connects upstream (midplane) conditions to
the divertor target:

.. math::

   T_t = T_u \left(\frac{2}{7} \frac{q_\parallel L_\parallel}
   {\kappa_0 T_u^{7/2}} + 1 \right)^{-2/7}

The **Eich scaling** (Eich 2013, multi-machine database) gives the
heat flux decay length at the midplane:

.. math::

   \lambda_q\;[\text{mm}] = 0.63 \; B_{p,\text{omp}}^{-1.19}

This sets the peak target heat flux, which must remain below the
material limit (:math:`\sim 10` MW/m\ :sup:`2` for tungsten).

**In the code:** :mod:`scpn_fusion.core.sol_model` implements the
two-point model and Eich scaling.

See :doc:`../tutorials/edge_sol_physics`.


7. Plasma Control
------------------

A tokamak plasma is inherently unstable and requires active control
of multiple quantities simultaneously:

**Shape control:**
   The plasma cross-section (elongation, triangularity, X-point
   position) is controlled by adjusting poloidal field coil currents.
   This is a MIMO control problem requiring a Jacobian-based approach.

**Vertical stability:**
   Elongated plasmas (:math:`\kappa > 1`) are vertically unstable
   (the plasma drifts up or down on a fast timescale).  A fast
   feedback loop (typically < 1 ms) using internal coils or power
   supplies is essential.

**Current profile control:**
   The safety factor profile :math:`q(r)` determines stability.
   Controlling it requires coordinating NBI, ECCD, and LHCD sources
   with the resistive evolution timescale.

**Density control:**
   Gas puffing and pellet injection maintain the density profile.
   Overfuelling risks Greenwald density limit disruptions.

**Disruption mitigation:**
   When a disruption is detected (loss of control, locked modes),
   massive material injection (shattered pellet injection) dilutes
   the plasma energy and converts the thermal quench into manageable
   radiation.

**In the code:** See :mod:`scpn_fusion.control` for the full control
suite: MPC, SNN controllers, digital twin, EFIT, shape control,
vertical stability, fault-tolerant operations, gain scheduling,
and scenario planning.

See :doc:`../userguide/control` and the advanced tutorials
:doc:`../tutorials/realtime_reconstruction`,
:doc:`../tutorials/fault_tolerant_operations`,
:doc:`../tutorials/scenario_design`.


8. Burning Plasma Physics
---------------------------

A **burning plasma** is one where alpha heating dominates over
external heating.  The Q-factor:

.. math::

   Q = \frac{P_\text{fusion}}{P_\text{aux}}

measures the energy multiplication.  :math:`Q = 1` is scientific
breakeven; :math:`Q = 10` is the ITER design target; :math:`Q = \infty`
is ignition (self-sustaining burn).

Burning plasmas exhibit new physics:

- **Alpha particle pressure** modifies equilibrium and stability
- **Energetic particle modes** (TAE, EPMs) can eject fast ions
- **Helium ash** dilutes the fuel, requiring active pumping
- **Burn control** is inherently nonlinear: more heating raises
  temperature, which increases fusion power, which raises temperature
  further (thermal instability)

**In the code:** :mod:`scpn_fusion.core.fusion_ignition_sim` provides
a 0-D dynamic burn model with alpha heating, Bremsstrahlung, helium
ash accumulation, and energy balance.

.. admonition:: What's Next?

   You now have a map of the physics landscape.  To get hands-on with
   the code, proceed to :doc:`first_simulation` for your first
   simulation walkthrough.
