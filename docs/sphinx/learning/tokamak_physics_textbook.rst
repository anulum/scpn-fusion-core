.. -----------------------------------------------------------------------
   SCPN Fusion Core -- Tokamak Physics Reference
   Copyright 1998-2026 Miroslav Sotek. All rights reserved.
   License: GNU AGPL v3 | Commercial licensing available
   -----------------------------------------------------------------------

===============================================
Tokamak Physics: A Computational Reference
===============================================

This reference chapter covers the mathematical and physical foundations
of tokamak plasma physics as implemented in SCPN-Fusion-Core.  It is
intended as a self-contained reference for graduate students and
researchers entering the field.  Equations are given in SI units
throughout.

.. contents:: Chapter Contents
   :depth: 2
   :local:


I. Magnetohydrodynamic Equilibrium
====================================

Ideal MHD Force Balance
------------------------

A magnetically confined plasma in static equilibrium satisfies:

.. math::

   \nabla p = \mathbf{J} \times \mathbf{B}

Combined with Maxwell's equations :math:`\nabla \times \mathbf{B} = \mu_0 \mathbf{J}`
and :math:`\nabla \cdot \mathbf{B} = 0`, this forms a closed system.

In axisymmetric geometry (:math:`\partial/\partial\phi = 0`), the magnetic
field can be written:

.. math::

   \mathbf{B} = \frac{1}{R}\nabla\psi \times \hat\phi + \frac{F(\psi)}{R}\hat\phi

where :math:`\psi(R,Z)` is the poloidal flux per radian and :math:`F(\psi) = R B_\phi`.
Substituting into the force balance yields the **Grad-Shafranov equation**:

.. math::
   :label: gs-full

   R\frac{\partial}{\partial R}\left(\frac{1}{R}\frac{\partial\psi}{\partial R}\right)
   + \frac{\partial^2\psi}{\partial Z^2}
   = -\mu_0 R^2 p'(\psi) - F(\psi) F'(\psi)

This is a nonlinear elliptic PDE.  The free functions :math:`p(\psi)` and
:math:`F(\psi)` are constrained by experimental data (pressure profile and
current profile measurements) or parametric models.


Solov'ev Analytic Solution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For :math:`p'(\psi) = \text{const}` and :math:`FF'(\psi) = \text{const}`,
the GS equation becomes linear and admits exact solutions (Solov'ev 1968).
The Solov'ev equilibrium is:

.. math::

   \psi(R,Z) = \frac{\psi_0}{R_0^4}\left[\frac{R^4}{8} + A\left(\frac{R^2 Z^2}{2}
   - \frac{R^4}{8}\right)\right] + c_1 + c_2 R^2 + c_3 \left(R^4 - 4R^2 Z^2\right)

where :math:`A = -\mu_0 R_0^2 p'/\psi_0` and the :math:`c_i` satisfy
boundary conditions.  This solution provides a useful benchmark for
numerical solvers.


Numerical Solution: Picard Iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GS equation is discretised on a uniform :math:`(R,Z)` grid.  At each
Picard step, the right-hand side is evaluated at the current
:math:`\psi^{(k)}`, and the resulting linear elliptic PDE is solved by
Red-Black SOR or multigrid.  Under-relaxation
:math:`\psi^{(k+1)} = (1-\alpha)\psi^{(k)} + \alpha\psi_\text{new}`
with :math:`\alpha \sim 0.1` ensures convergence.


Flux-Surface Geometry
^^^^^^^^^^^^^^^^^^^^^^

From the converged :math:`\psi(R,Z)`, the code extracts:

- Magnetic axis :math:`(R_\text{ax}, Z_\text{ax})` — the O-point
- Separatrix geometry — the :math:`\psi = \psi_\text{bry}` contour
- X-point(s) — saddle points of :math:`\psi`
- Normalised flux coordinate :math:`\hat\psi = (\psi - \psi_\text{ax})/(\psi_\text{bry} - \psi_\text{ax})`
- Safety factor :math:`q(\hat\psi) = \frac{1}{2\pi}\oint\frac{B_\phi}{R B_p}\,dl`


II. Neoclassical Transport Theory
====================================

Classical vs. Neoclassical
---------------------------

In a uniform magnetic field, Coulomb collisions produce **classical**
cross-field diffusion with step size :math:`\rho_L` (Larmor radius) and
step rate :math:`\nu_{ei}` (electron-ion collision frequency):

.. math::

   D_\perp^\text{classical} = \rho_L^2 \, \nu_{ei}

In toroidal geometry, particles with small parallel velocity are
**trapped** by the :math:`1/R` variation of :math:`B_\phi`.  The trapping
condition is :math:`v_\parallel / v < \sqrt{2\varepsilon}`, where
:math:`\varepsilon = r/R_0`.  These particles execute **banana orbits**
with width:

.. math::

   \Delta_b \sim \frac{q \rho_L}{\sqrt{\varepsilon}}

Since :math:`\Delta_b \gg \rho_L`, the neoclassical diffusion coefficient
is enhanced:

.. math::

   D_\perp^\text{neo} \sim q^2 \varepsilon^{-3/2} \, D_\perp^\text{classical}


Collisionality Regimes
^^^^^^^^^^^^^^^^^^^^^^^^

The dimensionless electron collisionality:

.. math::

   \nu_e^* = \frac{\nu_{ei} q R_0}{\varepsilon^{3/2} v_{te}}

determines which transport regime applies:

.. list-table::
   :header-rows: 1

   * - Regime
     - Condition
     - Transport scaling
   * - Banana
     - :math:`\nu_e^* < 1`
     - :math:`D \propto \nu_{ei} q^2 \rho_L^2 \varepsilon^{-3/2}`
   * - Plateau
     - :math:`1 < \nu_e^* < \varepsilon^{-3/2}`
     - :math:`D \propto q \rho_L^2 v_{te} / (R_0)`
   * - Pfirsch-Schlüter
     - :math:`\nu_e^* > \varepsilon^{-3/2}`
     - :math:`D \propto 2 q^2 D_\perp^\text{classical}`


Neoclassical Resistivity
^^^^^^^^^^^^^^^^^^^^^^^^^^

The parallel resistivity in a tokamak is modified by trapped-particle
effects.  The Sauter formulation (Sauter, Angioni & Lin-Liu 1999, 2002)
gives:

.. math::

   \eta_\parallel = \frac{\eta_\text{Spitzer}}{1 - f_t} \, C_R(Z_\text{eff}, f_t)

where :math:`f_t = 1 - (1-\varepsilon)^2 / [\sqrt{1-\varepsilon^2}(1 + 1.46\sqrt{\varepsilon})]`
is the trapped fraction and :math:`C_R` is a correction factor.

The Spitzer resistivity is:

.. math::

   \eta_\text{Spitzer} = \frac{1.65 \times 10^{-9} \, Z_\text{eff} \, \ln\Lambda}{T_e^{3/2}\;[\text{keV}]}
   \quad [\Omega\,\text{m}]


Bootstrap Current
^^^^^^^^^^^^^^^^^^

Trapped particles in a density or temperature gradient produce a
collisionless current (the bootstrap current).  For a tokamak with
:math:`n'` and :math:`T'` gradients:

.. math::

   j_\text{bs} \approx -\frac{\sqrt{\varepsilon}}{B_\theta}
   \left(p_e' L_{31} + p_e' L_{32} + p_i' L_{34}\right)

where the :math:`L_{3k}` are the Sauter bootstrap coefficients.  At
high :math:`\beta_p`, bootstrap current can provide 50--80% of the total
plasma current, enabling steady-state operation.


III. Resistive MHD Instabilities
==================================

Sawteeth
---------

The sawtooth instability is a periodic relaxation oscillation in the
plasma core.  It occurs when the :math:`q = 1` surface exists inside
the plasma.

**Trigger (Porcelli 1996):**

The internal kink mode becomes unstable when the potential energy
:math:`\delta W` crosses a threshold.  The Porcelli criterion includes
three stabilising effects:

1. **Ideal MHD drive** :math:`\delta W_\text{MHD}` (destabilising
   when :math:`s_1 \hat\beta_p > s_\text{crit}`)
2. **Diamagnetic stabilisation** :math:`\omega_{*i}` (stabilising at
   low :math:`\hat\beta_p`)
3. **Fast-ion stabilisation** :math:`\delta W_h` (from NBI or ICRH
   fast ions)

A crash is triggered when the growth rate of the reconnecting mode
exceeds all stabilising frequencies:

.. math::

   \gamma_\rho > c_\rho \omega_{*i}

**Crash (Kadomtsev 1975):**

During the crash, magnetic reconnection at the :math:`q = 1` surface
flattens the core profiles.  The mixing radius :math:`r_\text{mix}`
is found by flux conservation:

.. math::

   \int_0^{r_\text{mix}} [q(r) - 1] \, r \, dr = 0


NTM: Modified Rutherford Equation
-----------------------------------

Neoclassical tearing modes grow magnetic islands at rational surfaces
(:math:`q = m/n`).  The island half-width :math:`w` evolves according
to the Modified Rutherford Equation:

.. math::
   :label: mre

   \tau_R \frac{dw}{dt} = r_s \left[
   \Delta'(w) + \alpha_\text{bs}\,\frac{j_\text{bs}}{j_\phi}\,\frac{r_s}{w}
   - \alpha_\text{GGJ}\,\frac{w_d^2}{w^3}
   - \alpha_\text{pol}\,\frac{w_\text{pol}^4}{w^3 (w^2 + w_d^2)}
   + \alpha_\text{ECCD}\,\frac{j_\text{ECCD}}{j_\phi}\,\frac{r_s}{w}
   \right]

where:

- :math:`\Delta'` is the tearing stability index
- :math:`\alpha_\text{bs}` is the bootstrap drive (destabilising)
- :math:`\alpha_\text{GGJ}` is the Glasser-Greene-Johnson curvature
  stabilisation
- :math:`\alpha_\text{pol}` is the polarisation current threshold
- :math:`\alpha_\text{ECCD}` is the ECCD stabilisation term

A seed island (from a sawtooth crash or ELM) grows if the bootstrap
drive exceeds the threshold set by curvature and polarisation terms.
ECCD at the rational surface adds a stabilising term that can suppress
the island.


IV. Current Diffusion and Profile Control
===========================================

Flux Diffusion Equation
--------------------------

The poloidal flux evolves according to:

.. math::
   :label: flux-diffusion

   \frac{\partial\psi}{\partial t} = \frac{\eta(\rho)}{\mu_0 a^2}
   \mathcal{L}[\psi] + R_0 \eta(\rho) \, j_\text{source}(\rho)

where :math:`\mathcal{L}` is the cylindrical diffusion operator:

.. math::

   \mathcal{L}[\psi] = \frac{1}{\rho}\frac{\partial}{\partial\rho}
   \left(\rho\frac{\partial\psi}{\partial\rho}\right)

and :math:`j_\text{source} = j_\text{bs} + j_\text{ECCD} + j_\text{NBI} + j_\text{LHCD}`.

**Boundary conditions:**

- Axis (:math:`\rho = 0`): :math:`\partial\psi/\partial\rho = 0` (symmetry)
- Edge (:math:`\rho = 1`): :math:`\psi = \text{const}` (Dirichlet, fixed total current)

**Numerical scheme:** Crank-Nicolson (second-order in time, unconditionally
stable):

.. math::

   \frac{\psi^{n+1} - \psi^n}{\Delta t} = \frac{1}{2}\left(\mathcal{L}[\psi^{n+1}]
   + \mathcal{L}[\psi^n]\right) + j_\text{source}

This leads to a tridiagonal system solved by ``scipy.linalg.solve_banded``.


Current Drive Sources
-----------------------

**ECCD:** Gaussian deposition peaked at the resonance location:

.. math::

   j_\text{ECCD}(\rho) = \frac{\eta_\text{CD} P_\text{ECCD}}{(2\pi)^{3/2} R_0 a^2 \sigma}
   \exp\left(-\frac{(\rho - \rho_\text{dep})^2}{2\sigma^2}\right)

where :math:`\eta_\text{CD} \sim 0.2`--:math:`0.3` A/W is the current
drive efficiency.

**NBI:** Broader deposition with tangential injection geometry:

.. math::

   j_\text{NBI}(\rho) = j_0 \exp\left(-\frac{(\rho - \rho_0)^2}{2\sigma^2}\right)

**LHCD:** Off-axis deposition for current profile shaping:

.. math::

   j_\text{LHCD}(\rho) = j_0 \exp\left(-\frac{(\rho - \rho_0)^2}{2\sigma^2}\right)


V. Edge Physics
==================

Two-Point Model
-----------------

The SOL parallel heat transport along open field lines connects the
upstream (midplane) temperature :math:`T_u` to the target (divertor)
temperature :math:`T_t`:

.. math::
   :label: two-point

   T_t^{7/2} = T_u^{7/2} - \frac{7}{2}\frac{q_\parallel L_\parallel}{\kappa_0}

where :math:`q_\parallel` is the parallel heat flux, :math:`L_\parallel`
is the connection length, and :math:`\kappa_0 = 2000` W/(m·eV\ :sup:`7/2`)
is the Spitzer-Härm parallel conductivity.

The upstream electron temperature is estimated from pressure balance:

.. math::

   n_u T_u = n_t T_t + \frac{1}{2} m_i n_t c_{s,t}^2


Eich Scaling
--------------

The heat flux width at the outboard midplane:

.. math::

   \lambda_q = 0.63 \, B_{p,\text{omp}}^{-1.19} \quad [\text{mm}]

where :math:`B_{p,\text{omp}}` is the poloidal field at the outboard
midplane in Tesla.  This multi-machine regression (Eich 2013) is based
on data from ASDEX-U, JET, DIII-D, MAST, NSTX, and C-Mod.

The peak parallel heat flux to the divertor:

.. math::

   q_{\parallel,\text{peak}} = \frac{P_\text{SOL}}{2\pi R_t \lambda_q f_x}

where :math:`R_t` is the strike-point major radius and :math:`f_x` is
the flux expansion factor.  The surface heat flux
:math:`q_\text{surf} = q_\parallel \sin\alpha` (where :math:`\alpha`
is the field-line incidence angle) must remain below :math:`\sim 10`
MW/m\ :sup:`2` for steady-state tungsten operation.


VI. Plasma Control Theory
============================

Shape Control
--------------

The plasma boundary position :math:`\mathbf{r}_b` depends on the coil
currents :math:`\mathbf{I}_c` through a response matrix (Jacobian):

.. math::

   \delta\mathbf{r}_b = \mathbf{J} \, \delta\mathbf{I}_c

The control problem is to find :math:`\delta\mathbf{I}_c` that minimises
boundary error with bounded coil currents.  The Tikhonov-regularised
pseudoinverse:

.. math::

   \delta\mathbf{I}_c = (\mathbf{J}^T \mathbf{J} + \lambda \mathbf{I})^{-1}
   \mathbf{J}^T \delta\mathbf{r}_b

balances tracking accuracy against actuator effort.


Vertical Stability
--------------------

An elongated plasma (:math:`\kappa > 1`) is unstable to vertical
displacement events (VDEs) on the ideal MHD timescale.  The linearised
dynamics:

.. math::

   m_p \ddot{z} = -n_\text{idx} \frac{\mu_0 I_p^2}{4\pi R_0}\frac{z}{b^2}
   + F_\text{coil}(t)

where :math:`n_\text{idx}` is the stability index (negative for
instability).  The growth rate :math:`\gamma \sim 10^3`--:math:`10^4`
s\ :sup:`-1` demands sub-millisecond feedback.

The **super-twisting sliding mode controller** provides finite-time
convergence to zero tracking error with robustness to model uncertainty:

.. math::

   u = -\alpha_1 |s|^{1/2} \text{sign}(s) + v, \quad
   \dot{v} = -\alpha_2 \text{sign}(s)

where :math:`s = z - z_\text{ref}` is the sliding variable.


Gain Scheduling
-----------------

A tokamak discharge traverses multiple operating regimes (ramp-up,
L-mode flat-top, L-H transition, H-mode flat-top, ramp-down).  Each
regime has different plant dynamics and requires different controller
gains.

A gain-scheduled controller maintains a bank of regime-specific
controllers and switches between them based on a regime detector
(confinement time :math:`\tau_E`, disruption probability, current
ramp rate).  **Bumpless transfer** ensures smooth actuator output
during transitions:

.. math::

   u(t) = (1 - \beta(t)) \, u_\text{old}(t) + \beta(t) \, u_\text{new}(t)

where :math:`\beta(t)` ramps linearly from 0 to 1 over the transfer
interval.


VII. Burning Plasma and Ignition
==================================

The power balance of a D-T fusion plasma:

.. math::

   \frac{dW}{dt} = P_\alpha + P_\text{aux} - P_\text{loss} - P_\text{rad}

where:

- :math:`P_\alpha = \frac{1}{4} n_D n_T \langle\sigma v\rangle E_\alpha V`
  is the alpha heating power
- :math:`P_\text{aux}` is the external heating (NBI, ECRH, ICRH)
- :math:`P_\text{loss} = W / \tau_E` is the transport loss
- :math:`P_\text{rad}` includes Bremsstrahlung and line radiation

The D-T fusion reactivity peaks near :math:`T = 14` keV with
:math:`\langle\sigma v\rangle \approx 4 \times 10^{-22}` m\ :sup:`3`/s
(Bosch-Hale parametrisation).

Ignition (:math:`Q \to \infty`) requires :math:`P_\alpha > P_\text{loss} + P_\text{rad}`
with :math:`P_\text{aux} = 0`.  This sets a minimum triple product:

.. math::

   n T \tau_E > 3 \times 10^{21} \;\text{m}^{-3}\,\text{keV}\,\text{s}


VIII. Confinement Scaling Laws
================================

The empirical IPB98(y,2) scaling (ITER Physics Basis 1999) for
H-mode energy confinement time:

.. math::

   \tau_E^\text{IPB98(y,2)} = 0.0562 \, I_p^{0.93} \, B_T^{0.15} \,
   \bar{n}_{19}^{0.41} \, P_\text{loss}^{-0.69} \, R^{1.97} \,
   \varepsilon^{0.58} \, \kappa^{0.78} \, M^{0.19}

where :math:`I_p` is in MA, :math:`B_T` in T, :math:`\bar{n}_{19}`
in :math:`10^{19}` m\ :sup:`-3`, :math:`P_\text{loss}` in MW,
:math:`R` in m, and :math:`M` is the isotope mass number.

The H-factor :math:`H = \tau_E / \tau_E^\text{IPB98(y,2)}` measures
performance relative to the database.  ITER requires :math:`H \geq 1.0`
for :math:`Q = 10`.

**Domain of validity:** The scaling was derived from conventional tokamaks
(:math:`A > 2.5`, :math:`\kappa < 2.0`).  Spherical tokamaks (NSTX, MAST)
and stellarators fall outside this domain and typically show :math:`H < 1`.


IX. Nuclear Engineering
=========================

Tritium Breeding
------------------

The :math:`n + {}^6\text{Li} \to T + {}^4\text{He} + 4.8` MeV reaction
breeds fresh tritium.  The tritium breeding ratio (TBR) must exceed 1.0
for a self-sufficient reactor.  A lithium blanket with neutron multiplier
(Be or Pb) achieves TBR = 1.05--1.15.

Neutron Wall Loading
^^^^^^^^^^^^^^^^^^^^^

The first-wall neutron flux:

.. math::

   \Gamma_n = \frac{P_\text{fusion} \times 0.8}{4\pi R_0 a \kappa \times 14.1\;\text{MeV}}

For a 2 GW fusion reactor, :math:`\Gamma_n \sim 2`--:math:`3` MW/m\ :sup:`2`.
Structural materials (RAFM steels, SiC/SiC composites) must withstand
this for :math:`\sim 10` full-power years (50--100 dpa).


X. Glossary
=============

.. glossary::

   banana orbit
      The trajectory of a trapped particle in a tokamak, shaped like a
      banana in the poloidal cross-section due to the :math:`\nabla B`
      drift.

   bootstrap current
      Self-generated toroidal current arising from density and temperature
      gradients in the presence of trapped particles.

   divertor
      The region where open field lines are directed to target plates
      for controlled power exhaust.

   H-mode
      High-confinement mode, characterised by a steep pressure gradient
      (pedestal) at the plasma edge.

   L-mode
      Low-confinement mode, the default confinement state without a
      transport barrier.

   NTM
      Neoclassical tearing mode; a magnetic island driven by the loss
      of bootstrap current in the island region.

   Pfirsch-Schlüter
      The high-collisionality regime of neoclassical transport.

   safety factor
      :math:`q = rB_\phi / (R_0 B_\theta)`; the number of toroidal
      transits per poloidal transit of a field line.

   sawtooth
      Periodic core relaxation oscillation at the :math:`q = 1` surface.

   separatrix
      The last closed flux surface; the boundary between confined plasma
      and the SOL.

   SOL
      Scrape-Off Layer; the region of open field lines outside the
      separatrix.

   Troyon limit
      The maximum normalised beta :math:`\beta_N` before ideal
      ballooning/kink instability.

   VDE
      Vertical displacement event; rapid vertical motion of an elongated
      plasma.
