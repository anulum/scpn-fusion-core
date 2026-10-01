=========================
Studio Federation Surface
=========================

The Studio federation surface publishes a machine-readable description of the
FUSION package for Hub ingestion. It is optional at install time and enabled by
the ``studio`` extra. This extra requires Python 3.11 or later and
``scpn-studio-platform>=0.11.3.dev0,<0.12``; the core package retains Python
3.10 support. Installing the extra on Python 3.10 fails dependency resolution.
The published SDK 0.11.2 does not implement the v2 contract; install the exact
canonical source first:

.. code-block:: bash

   # Set this to your authorised canonical Platform checkout.
   PLATFORM_SDK_SOURCE=/path/to/SCPN-STUDIO-PLATFORM
   test "$(git -C "$PLATFORM_SDK_SOURCE" rev-parse HEAD)" = 6488f45b49f74e6e1d1580db8563698787250913
   python -m pip install --require-hashes -r requirements/studio.txt
   python -m pip install --no-deps --no-build-isolation "$PLATFORM_SDK_SOURCE"
   python -m pip install -e '.[studio]'
   python -m pip check

For a hash-locked review install on Python 3.11 or 3.12, install
``requirements/studio.txt`` after the corresponding core/CI requirements.
The Studio lock pins the SDK's runtime and build dependencies. The SDK source
is pinned separately to the exact commit above; the lock does not substitute
the older PyPI wheel. A standalone PyPI-only extra install remains gated until
a compatible SDK release is authorised and available.

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

Consumer admission
==================

New manifests explicitly declare contract era ``v2``. Evidence schema names
retain their ``studio.*.v1`` suffix: schema version and federation era are
separate fields. Consumers must negotiate the implemented manifest era:

.. code-block:: python

   from scpn_fusion.studio import build_federation_document
   from scpn_studio_platform.manifest import validate_studio_manifest

   document = build_federation_document()
   verdict = validate_studio_manifest(document["schema_a"], supported_eras={"v2"})
   assert verdict.admitted

The published SDK 0.11.2 supports this explicit manifest gate. Its keeper
aggregation API still implements a v1-only manifest policy and refuses the
emitted v2 document. Its evidence gate also predates the freshness-required
v2 contract. Upgrading to 0.11.2 alone therefore cannot establish v2 consumer
conformance. A keeper must provide both the v2 evidence contract and the
explicit ``supported_eras`` API before using the following aggregation recipe:

.. code-block:: python

   from scpn_studio_platform.manifest.aggregate import aggregate_federation

   snapshot = aggregate_federation([document], supported_eras={"v2"})

That API is implemented in Platform's canonical source commit
``6488f45b49f74e6e1d1580db8563698787250913`` (0.11.3.dev0); it is absent from
the published 0.11.2 wheel. Use a keeper containing that change and select its
supported era explicitly. A v1-only keeper must continue to refuse the v2
manifest. The manifest declaration grants no authority to alter the evidence
gate, create seals, or reclassify historical results.

Every new evidence admission passes the SDK's
``validate_studio_bundle(wire, era="v2")`` gate. An admitted
``reference-validated`` claim must carry ``verified-at-source`` freshness
backed by an actual source check. Missing, ``traceable-unchecked``, or
``untraceable`` freshness on that pair is refused. Bounded records retain
their boundary. The v1 transition mode is reserved for replaying previously
admitted snapshots; retain their original bytes and do not use it to admit new
evidence. Rechecking a source creates new evidence with its own provenance.

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
