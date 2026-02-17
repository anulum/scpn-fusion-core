==============================
Transport and Stability
==============================

SCPN-Fusion-Core couples the equilibrium solver with 1.5D radial
transport, multiple stability analysis tools, and AI-based turbulence
surrogates.

1.5D Radial Transport
----------------------

The integrated transport solver (``integrated_transport_solver.py``)
solves the coupled energy and particle diffusion equations on the
normalised radial coordinate :math:`\rho`:

.. math::
   :label: transport-particle

   \frac{\partial n}{\partial t} = \frac{1}{V'}\frac{\partial}{\partial \rho}
   \!\left[V'\!\left(D\frac{\partial n}{\partial \rho}\right)\right] + S_n

.. math::
   :label: transport-energy

   \frac{3}{2}\frac{\partial (n T)}{\partial t} = \frac{1}{V'}\frac{\partial}{\partial \rho}
   \!\left[V'\!\left(\chi\frac{\partial T}{\partial \rho}\right)\right]
   + P_\text{heat} - P_\text{rad} - P_\text{loss}

where :math:`D` is the particle diffusion coefficient, :math:`\chi` is
the thermal diffusivity, :math:`V'(\rho)` is the flux-surface volume
derivative, and :math:`S_n` is the particle source.

The anomalous transport coefficients :math:`D` and :math:`\chi` are
derived from the turbulence oracle (see below) or from prescribed
scaling laws.

IPB98(y,2) Confinement Scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The global energy confinement time is cross-checked against the
IPB98(y,2) scaling law (ITER Physics Basis, Nuclear Fusion 39, 1999):

.. math::
   :label: ipb98

   \tau_E = 0.0562 \; I_p^{0.93} \; B_T^{0.15} \; \bar{n}_{e,19}^{0.41}
            \; P_\text{loss}^{-0.69} \; R^{1.97} \; \kappa^{0.78}
            \; \varepsilon^{0.58} \; M^{0.19}

where :math:`I_p` is in MA, :math:`B_T` in T,
:math:`\bar{n}_{e,19}` in :math:`10^{19}\,\text{m}^{-3}`,
:math:`P_\text{loss}` in MW, :math:`R` in m, :math:`\kappa` is the
elongation, :math:`\varepsilon = a/R` is the inverse aspect ratio,
and :math:`M` is the effective ion mass in amu.

The uncertainty quantification follows the Bayesian regression framework
of Verdoolaege et al. (Nuclear Fusion 61, 2021) using the 20-shot ITPA
H-mode confinement database.

RF Heating Models
------------------

The ``rf_heating`` module simulates auxiliary heating deposition profiles
using simplified ray-tracing:

**ICRH** -- Ion Cyclotron Resonance Heating
   Power is deposited at the ion cyclotron resonance layer
   :math:`\omega = n \omega_{ci}` where :math:`\omega_{ci} = eB / m_i`.

**ECRH** -- Electron Cyclotron Resonance Heating
   Power deposited at the electron cyclotron resonance
   :math:`\omega = n \omega_{ce}` where :math:`\omega_{ce} = eB / m_e`.

**LHCD** -- Lower Hybrid Current Drive
   Non-inductive current drive via lower-hybrid wave absorption.

Turbulence Models
------------------

FNO Turbulence Suppressor
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Fourier Neural Operator (FNO) turbulence model
(``fno_turbulence_suppressor.py``) provides real-time spectral
turbulence prediction using 12 Fourier modes.  The FNO architecture
(Li et al. 2021) learns the solution operator of the turbulence PDE
directly in Fourier space:

.. math::

   u_{l+1}(x) = \sigma\!\left(W_l\,u_l(x) + \mathcal{F}^{-1}
   \!\left[R_l \cdot \mathcal{F}(u_l)\right](x)\right)

where :math:`\mathcal{F}` is the Fourier transform, :math:`R_l` is
a learnable spectral filter, and :math:`\sigma` is a nonlinear
activation.

Multi-regime training data is generated from a modified
Hasegawa-Wakatani model with regime-dependent parameters (ITG, TEM,
ETG).

Turbulence Oracle
^^^^^^^^^^^^^^^^^^

The turbulence oracle (``turbulence_oracle.py``) predicts the dominant
instability regime (ITG, TEM, or ETG) from local plasma parameters and
provides anomalous transport coefficient estimates for the transport
solver.

MHD Stability
--------------

Sawtooth Oscillations
^^^^^^^^^^^^^^^^^^^^^^

The ``mhd_sawtooth`` module models sawtooth crash dynamics using a
modified Kadomtsev reconnection model (Kadomtsev 1975).  The sawtooth
period is estimated from the resistive evolution of the :math:`q = 1`
surface, and the crash amplitude is computed from the flux reconnection
geometry.

Hall-MHD Effects
^^^^^^^^^^^^^^^^^

The ``hall_mhd_discovery`` module incorporates two-fluid Hall-MHD
effects (Huba, NRL Plasma Formulary 2019) that become significant
in compact, high-field tokamaks where the ion skin depth
:math:`d_i = c/\omega_{pi}` is comparable to the equilibrium scale
length.  Hall-MHD modifications include:

- Whistler-wave dispersion at frequencies above :math:`\omega_{ci}`
- Hall-term correction to the magnetic induction equation:
  :math:`\partial \mathbf{B}/\partial t = \nabla \times [(\mathbf{v} - d_i \mathbf{J}/ne) \times \mathbf{B}]`
- Modified reconnection rates relevant to sawtooth crash timing

Stability Analysis
^^^^^^^^^^^^^^^^^^^

The ``stability_analyzer`` module provides:

- **Nyquist stability** analysis for closed-loop feedback systems
- **Lyapunov stability** margins for nonlinear dynamics
- **Vertical stability** index (decay index :math:`n = -R/B_z \cdot \partial B_z/\partial R`) for positional control
- **Beta limits** (:math:`\beta_N` Troyon limit, :math:`\beta_p` critical)

Self-Organised Criticality (Legacy Research Lane)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``sandpile_fusion_reactor`` module is retained as a legacy SOC
research lane for reproducibility and exploratory avalanche studies.
It is not part of the release-gated transport validation path (which is
based on Gyro-Bohm + Chang-Hinton + EPED-like boundary scaling).

Fusion Ignition and Burn Physics
----------------------------------

The ``fusion_ignition_sim`` module computes the Lawson criterion and
ignition margins using the Bosch-Hale D-T fusion reactivity
parametrisation:

.. math::

   \langle\sigma v\rangle_{\text{DT}} = C_1\,\theta\,
   \sqrt{\frac{\xi}{m_r c^2 T^3}} \exp(-3\xi)

where :math:`\theta`, :math:`\xi`, and :math:`C_1` are fit parameters
(Bosch & Hale, Nuclear Fusion 32, 1992).

The ignition condition :math:`Q \to \infty` requires that alpha-particle
heating alone sustains the plasma temperature:

.. math::

   P_\alpha = \frac{1}{4} n_D n_T \langle\sigma v\rangle E_\alpha
   \;\geq\; P_\text{loss}

where :math:`E_\alpha = 3.52\,\text{MeV}` per fusion event.

Warm Dense Matter EOS
^^^^^^^^^^^^^^^^^^^^^^

The ``wdm_engine`` module provides a reduced equation-of-state model
for warm dense matter conditions relevant to inertial confinement
scenarios and pellet ablation physics.

Related Modules
----------------

- :mod:`scpn_fusion.core.integrated_transport_solver` -- coupled transport
- :mod:`scpn_fusion.core.rf_heating` -- ICRH/ECRH/LHCD heating
- :mod:`scpn_fusion.core.fno_turbulence_suppressor` -- FNO model
- :mod:`scpn_fusion.core.turbulence_oracle` -- turbulence regime predictor
- :mod:`scpn_fusion.core.mhd_sawtooth` -- sawtooth crash model
- :mod:`scpn_fusion.core.hall_mhd_discovery` -- Hall-MHD effects
- :mod:`scpn_fusion.core.stability_analyzer` -- stability margins
- :mod:`scpn_fusion.core.sandpile_fusion_reactor` -- SOC criticality (legacy lane)
- :mod:`scpn_fusion.core.fusion_ignition_sim` -- ignition/burn physics
- :mod:`scpn_fusion.core.wdm_engine` -- warm dense matter EOS
