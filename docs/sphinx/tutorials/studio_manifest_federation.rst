=================================
Emit a Studio Federation Manifest
=================================

This tutorial shows the release-safe path for producing the Studio federation
document.

Prerequisite: install the optional Studio SDK dependency.

.. code-block:: bash

   pip install 'scpn-fusion[studio]'

1. Inspect the manifest in Python
=================================

.. code-block:: python

   from scpn_fusion.studio import build_federation_document

   doc = build_federation_document()
   print(doc["schema_a"]["studio"])
   print(doc["schema_a"]["studio_version"])
   print(len(doc["schema_a"]["verbs"]))
   print(doc["architecture_map"]["version"])

2. Regenerate the committed document
====================================

.. code-block:: bash

   scpn-emit-studio-manifest

The command writes ``docs/_generated/studio_manifest.json``.

3. Check release drift
======================

.. code-block:: bash

   scpn-emit-studio-manifest --check

The check fails when the committed JSON no longer byte-matches the generator.
Run it after editing:

- ``src/scpn_fusion/studio/verbs.py``
- ``src/scpn_fusion/studio/manifest.py``
- ``src/scpn_fusion/studio/federation.py``
- package version metadata

4. Compare a reproduced value
=============================

.. code-block:: python

   import numpy as np
   from scpn_fusion.studio import ExactnessClass, reproduce

   verdict = reproduce(
       ExactnessClass.TOLERANCE,
       recomputed_value=np.array([1.0, 2.0, 3.0]),
       reference_value=np.array([1.0, 2.0, 3.0 + 1e-12]),
       rtol=1e-9,
       atol=0.0,
   )
   assert verdict.reproduced

Use ``ExactnessClass.BIT_EXACT`` only for integer, fixed-point, or otherwise
toolchain-independent values. Floating-point kernels should declare a tolerance
or a caller-reduced stochastic comparison.
