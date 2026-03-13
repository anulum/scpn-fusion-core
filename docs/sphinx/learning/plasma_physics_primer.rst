.. -----------------------------------------------------------------------
   SCPN Fusion Core -- Plasma Physics Primer
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

==============================
Plasma Physics Primer
==============================

This chapter introduces the physics of plasmas and magnetic confinement
fusion from first principles.  No prior knowledge of plasma physics is
assumed; familiarity with classical electromagnetism at an undergraduate
level is sufficient.

What Is a Plasma?
------------------

A plasma is a quasi-neutral gas of charged and neutral particles that
exhibits collective behaviour.  When a gas is heated to temperatures
above roughly :math:`10^4` K, atoms ionise: electrons separate from
nuclei, producing a soup of free electrons and ions.  This is the
**fourth state of matter** -- distinct from solids, liquids, and gases
because the long-range Coulomb force between charged particles gives
rise to collective phenomena (waves, instabilities, shielding) that
have no counterpart in neutral gases.

Over 99% of visible matter in the universe is plasma: stars, the solar
wind, interstellar medium, and lightning.  On Earth, plasmas must be
created and sustained artificially.

Key plasma parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Parameter
     - Definition
     - Fusion-relevant value
   * - Temperature :math:`T`
     - Kinetic energy per particle, usually in keV (1 keV = 11.6 MK)
     - 10--20 keV (100--200 million K)
   * - Density :math:`n`
     - Number of particles per unit volume
     - :math:`10^{20}` m\ :sup:`-3`
   * - Debye length :math:`\lambda_D`
     - :math:`\sqrt{\varepsilon_0 T / (n e^2)}`; the distance over which charge is screened
     - :math:`\sim 10^{-4}` m
   * - Plasma frequency :math:`\omega_p`
     - :math:`\sqrt{n e^2 / (\varepsilon_0 m_e)}`; fastest electrostatic response
     - :math:`\sim 10^{11}` rad/s

A plasma behaves collectively when the number of particles in a Debye
sphere :math:`N_D = \frac{4}{3}\pi n \lambda_D^3 \gg 1`, so that the
mean-field approximation is valid.  For fusion plasmas, :math:`N_D \sim 10^8`.


Why Fusion?
-----------

Nuclear fusion combines light nuclei into heavier products, releasing
energy from the mass deficit (:math:`E = \Delta m \, c^2`).  The most
accessible reaction is deuterium-tritium (D-T):

.. math::

   \text{D} + \text{T} \;\longrightarrow\; {}^4\text{He}\,(3.5\;\text{MeV})
   + n\,(14.1\;\text{MeV})

This reaction has the largest cross-section :math:`\langle\sigma v\rangle`
at accessible temperatures (:math:`\sim 10`--:math:`20` keV) and releases
17.6 MeV per event.  The alpha particle (3.5 MeV) deposits its energy in
the plasma, sustaining the burn; the neutron escapes and is captured in a
lithium blanket to breed fresh tritium and extract heat.

Fusion fuel is abundant: deuterium from seawater (1 in 6700 hydrogen atoms),
tritium bred from lithium.  A 1 GW fusion plant would consume roughly
250 kg of fuel per year.  There is no long-lived radioactive waste and
no chain-reaction risk.


The Lawson Criterion
---------------------

A self-sustaining fusion plasma requires that the alpha heating exceeds
all power losses.  Lawson (1957) showed that this imposes a minimum on
the **triple product**:

.. math::
   :label: lawson

   n \, T \, \tau_E \;\geq\; 3 \times 10^{21} \;\text{m}^{-3}\,\text{keV}\,\text{s}

where:

- :math:`n` is the plasma density
- :math:`T` is the temperature
- :math:`\tau_E` is the energy confinement time (the e-folding time for
  stored energy loss)

The triple product is the single most important figure of merit for any
confinement scheme.  Achieving it requires simultaneously:

1. **High temperature** (:math:`T \sim 10`--:math:`20` keV) to maximise
   :math:`\langle\sigma v\rangle`
2. **High density** (:math:`n \sim 10^{20}` m\ :sup:`-3`)
3. **Good confinement** (:math:`\tau_E \sim 1`--:math:`10` s)


Magnetic Confinement
---------------------

Charged particles gyrate around magnetic field lines with a radius
:math:`\rho_L = m v_\perp / (q B)` (the Larmor radius).  For a
10 keV deuterium ion in a 5 T field, :math:`\rho_L \approx 4` mm --
much smaller than the plasma size (:math:`\sim 1` m).  This means
magnetic fields can confine charged particles perpendicular to the
field.

The problem is **parallel transport**: particles stream freely along
field lines.  To prevent end losses, the field lines must close on
themselves.  This is achieved by bending the magnetic field into a
torus.

The Tokamak
^^^^^^^^^^^^

A tokamak (from the Russian acronym for "toroidal chamber with
magnetic coils") is the most successful magnetic confinement device.
It uses two magnetic field components:

1. **Toroidal field** :math:`B_\phi` -- produced by external coils
   wrapped around the torus.  Typically 2--13 T.

2. **Poloidal field** :math:`B_\theta` -- produced by the plasma
   current :math:`I_p` flowing in the toroidal direction.  This
   current is driven inductively by a central solenoid (transformer
   action) or non-inductively by neutral beam injection (NBI) or
   radiofrequency waves (ECCD, LHCD).

The combination :math:`\mathbf{B} = B_\phi \hat\phi + B_\theta \hat\theta`
produces helical field lines that wrap around nested **flux surfaces**
(topological tori).  Particles confined to flux surfaces undergo only
slow cross-field transport (diffusion), while parallel losses are
eliminated by the closed topology.


Tokamak Geometry
-----------------

.. math::

   R &= R_0 + r \cos\theta \\
   Z &= r \sin\theta

where:

- :math:`R_0` is the major radius (distance from the machine axis to
  the plasma centre)
- :math:`a` is the minor radius (half-width of the plasma cross-section)
- :math:`r` is the radial coordinate from the magnetic axis
- :math:`\theta` is the poloidal angle
- :math:`\phi` is the toroidal angle (around the torus)

Key dimensionless parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 35 25 20

   * - Symbol
     - Name
     - Definition
     - Typical value
   * - :math:`A = R_0/a`
     - Aspect ratio
     - Major / minor radius
     - 2.5--4.0
   * - :math:`\varepsilon = a/R_0`
     - Inverse aspect ratio
     - Minor / major radius
     - 0.25--0.4
   * - :math:`\kappa`
     - Elongation
     - Vertical / horizontal half-widths
     - 1.6--2.0
   * - :math:`\delta`
     - Triangularity
     - Horizontal shift of extremal points
     - 0.2--0.5
   * - :math:`q`
     - Safety factor
     - :math:`\frac{r B_\phi}{R_0 B_\theta}`
     - 1 (axis) to 3--5 (edge)

The safety factor :math:`q` measures the pitch of the helical field
lines: a field line completes :math:`q` toroidal transits for every
poloidal transit.  Rational values :math:`q = m/n` (where field lines
close on themselves) are sites of MHD instabilities.  The :math:`q = 1`
surface is where sawteeth occur; :math:`q = 2` and :math:`q = 3/2`
surfaces host neoclassical tearing modes (NTMs).

The Magnetic Axis and Separatrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **magnetic axis** is the closed field line at the centre of the
nested flux surfaces (the O-point of the poloidal flux function
:math:`\psi`).  The **separatrix** (or last closed flux surface, LCFS)
is the outermost closed flux surface; beyond it, field lines intersect
material surfaces (the divertor).  The X-point is a saddle point of
:math:`\psi` where the separatrix crosses itself.

.. admonition:: What's Next?

   Now that you understand what a plasma is, why we want fusion, and
   how a tokamak confines it, proceed to :doc:`fusion_engineering_101`
   to learn about the physics models (equilibrium, transport, stability,
   control) that SCPN-Fusion-Core implements.
