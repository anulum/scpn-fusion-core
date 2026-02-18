==============================
Diagnostics
==============================

The diagnostics subpackage provides synthetic diagnostic instruments
for virtual tokamak experiments, enabling end-to-end validation of the
measurement-reconstruction-control pipeline without access to a
physical device.

Synthetic Sensors
------------------

The ``synthetic_sensors`` module (``synthetic_sensors.py``) implements
the ``SensorSuite`` class, which generates synthetic measurement signals
from the plasma state:

**Magnetic diagnostics:**

- Rogowski coil (plasma current :math:`I_p`)
- Diamagnetic loop (stored energy :math:`W_\text{dia}`)
- Magnetic flux loops (poloidal flux :math:`\psi` at probe locations)
- Mirnov coils (magnetic fluctuation :math:`\delta B_\theta`)

**Kinetic diagnostics:**

- Thomson scattering (electron temperature :math:`T_e(\rho)` and
  density :math:`n_e(\rho)` profiles)
- Interferometer (line-integrated electron density
  :math:`\int n_e \, dl`)
- Bolometer (radiated power :math:`P_\text{rad}`)

**Neutron diagnostics:**

- Neutron flux monitor (fusion reaction rate)
- Activation foils (neutron spectrum unfolding)

Each sensor model includes configurable noise characteristics (Gaussian
noise, systematic offsets, calibration drift) for realistic synthetic
data generation.

Forward Diagnostic Models
--------------------------

The ``forward`` module (``forward.py``) provides physics-based forward
models that compute what a real diagnostic would measure given a known
plasma state:

**Interferometer phase shift:**

.. math::

   \Delta\phi = \frac{e^2}{2\varepsilon_0 m_e c \omega}
   \int_{\text{LOS}} n_e(R, Z)\, dl

where the integral is along the line of sight (LOS) through the plasma.

**Neutron count rate:**

.. math::

   \dot{N} = \frac{1}{4} \int n_D n_T \langle\sigma v\rangle_{\text{DT}}
   \, \Omega(\mathbf{r}) \, dV

where :math:`\Omega(\mathbf{r})` is the solid angle subtended by the
detector from position :math:`\mathbf{r}`.

The ``ForwardDiagnosticChannels`` class bundles multiple diagnostic
channels into a single forward-model evaluation, suitable for feeding
into inverse reconstruction algorithms.

Soft X-Ray Tomography
-----------------------

The ``tomography`` module (``tomography.py``) implements the
``PlasmaTomography`` class for soft X-ray (SXR) tomographic inversion:

Given a set of line-integrated SXR brightness measurements:

.. math::

   g_i = \int_{\text{LOS}_i} \varepsilon(R, Z) \, dl + \eta_i

where :math:`\varepsilon(R, Z)` is the local SXR emissivity and
:math:`\eta_i` is measurement noise, the tomographic inversion recovers
the 2D emissivity field by solving:

.. math::

   \min_{\varepsilon} \; \|\mathbf{T}\varepsilon - \mathbf{g}\|_2^2
   + \lambda \|\mathbf{L}\varepsilon\|_2^2

where :math:`\mathbf{T}` is the geometry matrix (line-of-sight integrals
through the grid), :math:`\lambda` is the Tikhonov regularisation
parameter, and :math:`\mathbf{L}` is a smoothness operator.

The implementation uses SciPy's sparse linear algebra (``scipy.sparse.linalg``)
with automatic fallback for environments where SciPy is not optimised.

Diagnostic Runner
-------------------

The ``run_diagnostics`` module (``run_diagnostics.py``) provides a
convenience function for running a full diagnostic suite against a
plasma state.  This is used in the ``diagnostics`` simulation mode::

    scpn-fusion diagnostics

The runner generates synthetic measurements, applies noise, performs
tomographic inversion, and reports reconstruction quality metrics.

Related Modules
-----------------

- :mod:`scpn_fusion.diagnostics.synthetic_sensors` -- virtual instruments
- :mod:`scpn_fusion.diagnostics.forward` -- forward diagnostic models
- :mod:`scpn_fusion.diagnostics.tomography` -- SXR tomographic inversion
- :mod:`scpn_fusion.diagnostics.run_diagnostics` -- diagnostic runner
