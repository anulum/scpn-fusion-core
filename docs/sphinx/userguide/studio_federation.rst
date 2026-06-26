=========================
Studio Federation Surface
=========================

The Studio federation surface publishes a machine-readable description of the
FUSION package for Hub ingestion. It is optional at install time and enabled by
the ``studio`` extra:

.. code-block:: bash

   pip install 'scpn-fusion[studio]'

The public package is ``scpn_fusion.studio``. It exposes three contracts:

``manifest``
   Builds the schema-A capability manifest: studio id, package version,
   platform SDK range, protocol version, verbs, evidence schemas, transport
   profile, and content digest.

``federation``
   Emits the complete JSON document to
   ``docs/_generated/studio_manifest.json``. The document contains the schema-A
   block plus the additive architecture-map extension.

``exactness``
   Compares reproduced claim values using the declared exactness class:
   bit-exact digest equality, tolerance-aware float comparison, or a
   caller-reduced stochastic comparison. The exactness-class and verdict wire
   vocabulary is re-exported from the Studio Platform SDK so Hub, Studio, and
   FUSION share one wire contract; FUSION only owns the NumPy tolerance
   comparator behind that shared axis.

Command-line workflow
=====================

Check the committed generated document:

.. code-block:: bash

   scpn-emit-studio-manifest --check

Regenerate it after changing verbs, evidence schemas, architecture-map fields,
or version metadata:

.. code-block:: bash

   scpn-emit-studio-manifest

The generated file is committed because downstream Studio/Hub tooling consumes
it without importing the package. CI runs the drift check so stale federation
documents do not ship.

Evidence boundary
=================

The Studio document is an index of capabilities and evidence schemas. It does
not certify plant-control readiness, live machine execution, or accepted
full-fidelity solver parity. Those states still require the validation reports,
same-case external outputs, checksums, thresholds, and pass/fail rows described
in the validation guide.

Related API pages:

- :mod:`scpn_fusion.studio.manifest`
- :mod:`scpn_fusion.studio.federation`
- :mod:`scpn_fusion.studio.exactness`
- :mod:`scpn_fusion.studio.verbs`
