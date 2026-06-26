==============================================
:mod:`scpn_fusion.studio` -- Studio Federation
==============================================

The Studio package emits the schema-A federation manifest, architecture-map
extension, evidence-schema vocabulary, and exactness-class reproduction
comparator.

Package Re-exports
------------------

.. automodule:: scpn_fusion.studio
   :members:
   :undoc-members:
   :show-inheritance:

Manifest
--------

.. automodule:: scpn_fusion.studio.manifest
   :members:
   :undoc-members:
   :show-inheritance:

Federation Document
-------------------

.. automodule:: scpn_fusion.studio.federation
   :members:
   :undoc-members:
   :show-inheritance:

Exactness Comparator
--------------------

.. automodule:: scpn_fusion.studio.exactness
   :members:
   :undoc-members:
   :show-inheritance:

Verb and Evidence Schema Vocabulary
-----------------------------------

The advertised verb names are ``reconstruct``, ``simulate``, ``analyse``,
``validate``, ``benchmark``, ``replay``, ``control``, and ``predict``. The
schema-A SDK stores them as ``Verb`` values in
``scpn_fusion.studio.verbs.FUSION_VERBS``.

.. autofunction:: scpn_fusion.studio.verbs.evidence_schemas
