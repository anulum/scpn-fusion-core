External Solver Integrations
============================

TORAX runtime
-------------

The TORAX caller surface is process-isolated. Importing these contracts does
not import TORAX or JAX. Runtime outcomes preserve the complete backend
DataTree, while semantic consumers use the deterministic review-only envelope.

.. automodule:: scpn_fusion.integrations.torax
   :members:
   :undoc-members:
   :show-inheritance:

Runtime client
~~~~~~~~~~~~~~

.. automodule:: scpn_fusion.integrations.torax.client
   :members:
   :show-inheritance:

Contracts
~~~~~~~~~

.. automodule:: scpn_fusion.integrations.torax.contracts
   :members:
   :show-inheritance:

Deterministic review envelope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: scpn_fusion.integrations.torax.review
   :members:
   :show-inheritance:

Uniform-DT operating maps
-------------------------

.. automodule:: validation.cfspopcon_operating_map
   :members:

PROCESS power diagnostics
-------------------------

.. automodule:: validation.process_power_report
   :members:

Source-bound PROCESS execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: validation.process_reference_run
   :members:

RustBCA particle-surface diagnostics
------------------------------------

These optional APIs retain local build and run observations. They do not
establish independent source attestation or physical/material validation.

.. automodule:: validation.rustbca_request
   :members:

.. automodule:: validation.rustbca_reference
   :members:

.. automodule:: validation.rustbca_build_receipt
   :members:

.. automodule:: tools.build_rustbca_reference
   :members:
