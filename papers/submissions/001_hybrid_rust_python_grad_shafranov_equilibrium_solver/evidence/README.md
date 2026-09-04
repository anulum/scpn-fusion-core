<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — equilibrium-paper evidence custody. -->

# Evidence

`real_diiid_145419_validation.json` preserves the repository's
provenance-bound same-case DIII-D validation payload used by the manuscript.
The public copy adds only required structured Tier-0 metadata.  The manifest
binds it to evidence revision `476908debdd886d3a35bf0ae85216e684727adce`
through the tracked source path, Git blob identity, original SHA-256,
packaged-file SHA-256 and generator SHA-256.

`sparc_geqdsk_rmse_benchmark.json` preserves the pointwise benchmark payload
generated at `2026-08-26T16:44:27.724185+00:00`, with the same structured
metadata addition.  Its source artifact was Git-ignored and has no historical
Git object; the paper-local JSON is the authoritative public custody copy.  The
manifest records the original and packaged SHA-256 values, the immediately
preceding repository-revision context and the generator SHA-256.

The gated and diagnostic rows must be interpreted separately.  All 16 gated
SPARC rows use the neural backend and pass the declared pointwise threshold;
only 8 of 16 also pass the adapted source-convention contract.  Passing those
rows does not supply missing free-boundary coil currents or establish discharge
prediction, experimental accuracy or production admission.
