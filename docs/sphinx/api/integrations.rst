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
