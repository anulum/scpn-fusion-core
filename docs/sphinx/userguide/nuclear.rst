==============================
Nuclear Engineering
==============================

The nuclear engineering subpackage provides models for tritium breeding
blanket neutronics, plasma-wall interaction, erosion physics, and
thermoelectric MHD effects in liquid metal divertors.

Blanket Neutronics
-------------------

The ``blanket_neutronics`` module (``blanket_neutronics.py``) computes
the tritium breeding ratio (TBR) using a 1D slab transport model with
realistic albedo and neutron multiplication.

The tritium breeding ratio is defined as:

.. math::
   :label: tbr

   \text{TBR} = \frac{\text{tritium atoms produced per unit time}}
                     {\text{tritium atoms consumed per unit time}}

For a self-sustaining fusion reactor, :math:`\text{TBR} > 1.0` is
required (with margin :math:`\text{TBR} \geq 1.05` to account for
losses in the tritium fuel cycle).

The neutron source from D-T fusion at 14.1 MeV is:

.. math::

   S_n = \frac{1}{4} n_D n_T \langle\sigma v\rangle_{\text{DT}}

The module evaluates breeding performance for different blanket
concepts:

- **Lithium-lead** (:math:`\text{Pb-Li}` eutectic) blankets
- **Ceramic breeder** (:math:`\text{Li}_4\text{SiO}_4`,
  :math:`\text{Li}_2\text{TiO}_3`) blankets
- **Lithium** (pure liquid Li) blankets

The ``BreedingBlanket`` class computes the ``VolumetricBlanketReport``
containing TBR, neutron multiplication factor, and energy deposition
profiles.

Plasma-Wall Interaction
-------------------------

The ``nuclear_wall_interaction`` module (``nuclear_wall_interaction.py``)
provides the ``NuclearEngineeringLab`` class for simulating first-wall
damage from the fusion neutron spectrum:

- **Displacement damage** (dpa) from 14.1 MeV neutrons
- **Helium production** (appm He) via :math:`(n,\alpha)` transmutation
- **Hydrogen production** via :math:`(n,p)` reactions
- **Activation** products and decay heat

The neutron wall loading is:

.. math::

   \Gamma_n = \frac{P_\text{fus} \times 0.8}{4\pi R \times 2\pi a \kappa}

where the factor 0.8 reflects the 14.1 MeV neutron fraction of the
17.6 MeV total D-T fusion energy.

PWI Erosion Model
^^^^^^^^^^^^^^^^^^

The ``pwi_erosion`` module (``pwi_erosion.py``) implements the
``SputteringPhysics`` class for plasma-facing component erosion:

- **Physical sputtering** yield :math:`Y(E, \theta)` as a function of
  ion energy :math:`E` and incidence angle :math:`\theta`
- **Chemical sputtering** for carbon-based materials
- **Self-sputtering** cascade effects
- **Erosion rate** computation for tungsten, carbon, and beryllium PFCs
- Angle-energy invariant testing for physical consistency

The sputtering yield follows the Yamamura-Tawara parametrisation:

.. math::

   Y(E) = Q \cdot s_n(E) \cdot
   \left[1 - \left(\frac{E_\text{th}}{E}\right)^{2/3}\right]
   \cdot \left(1 - \frac{E_\text{th}}{E}\right)^2

where :math:`Q` is a fitting parameter, :math:`s_n(E)` is the nuclear
stopping cross-section, and :math:`E_\text{th}` is the sputtering
threshold energy.

Divertor Thermal Simulation
-----------------------------

The ``divertor_thermal_sim`` module models the heat flux profile on the
divertor target plates using the Eich model (Eich et al., Nuclear
Fusion 53, 2013):

.. math::

   q(s) = \frac{q_0}{2} \exp\!\left(\frac{S^2}{4\lambda_q^2 f_x^2}\right)
   \cdot \text{erfc}\!\left(\frac{S}{2\lambda_q f_x} - \frac{s - s_0}{\lambda_q}\right)

where :math:`\lambda_q` is the SOL power width, :math:`f_x` is the
flux expansion factor, :math:`S` is the divertor broadening parameter,
and :math:`s` is the coordinate along the divertor target.

TEMHD Peltier Effects
-----------------------

The ``temhd_peltier`` module (``temhd_peltier.py``) implements the
``TEMHD_Stabilizer`` for thermoelectric magnetohydrodynamic effects in
liquid metal divertors.

In a liquid metal flowing perpendicular to a strong magnetic field,
thermoelectric currents driven by temperature gradients generate
:math:`\mathbf{J} \times \mathbf{B}` forces that can either stabilise
or destabilise the flow.  The TEMHD effect is characterised by the
thermoelectric figure of merit:

.. math::

   ZT = \frac{S^2 \sigma T}{\kappa}

where :math:`S` is the Seebeck coefficient, :math:`\sigma` is the
electrical conductivity, :math:`T` is the temperature, and
:math:`\kappa` is the thermal conductivity.

For the MVR-0.96 compact reactor design, the TEMHD liquid metal
divertor is essential for handling heat loads exceeding 90 MW/m^2.

Related Modules
-----------------

- :mod:`scpn_fusion.nuclear.blanket_neutronics` -- TBR computation
- :mod:`scpn_fusion.nuclear.nuclear_wall_interaction` -- first-wall damage
- :mod:`scpn_fusion.nuclear.pwi_erosion` -- sputtering physics
- :mod:`scpn_fusion.nuclear.temhd_peltier` -- TEMHD stabilisation
- :mod:`scpn_fusion.core.divertor_thermal_sim` -- divertor heat flux
