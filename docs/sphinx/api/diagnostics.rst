====================================================
:mod:`scpn_fusion.diagnostics` -- Diagnostics
====================================================

The diagnostics subpackage provides synthetic diagnostic instruments,
forward models, and tomographic inversion for virtual tokamak
experiments.

Magnetic measurements require native float64 matrices with at least two nodes
on each axis. Valid strided/readonly inputs are preserved; deterministic
outputs own independent C-contiguous storage. See
`Numerical contracts <../../NUMERICAL_CONTRACTS.md>`_ for scalar admission,
explicit conversion and typed failures.

Synthetic Sensors
-------------------

.. automodule:: scpn_fusion.diagnostics.synthetic_sensors
   :members:
   :undoc-members:
   :show-inheritance:

Forward Diagnostic Models
---------------------------

.. automodule:: scpn_fusion.diagnostics.forward
   :members:
   :undoc-members:
   :show-inheritance:

Tomographic Inversion
-----------------------

.. automodule:: scpn_fusion.diagnostics.tomography
   :members:
   :undoc-members:
   :show-inheritance:

Diagnostic Runner
-------------------

.. automodule:: scpn_fusion.diagnostics.run_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:
